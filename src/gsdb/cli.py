from __future__ import annotations

import json
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .catalog import build_catalog
from .doctor import run_doctor
from .manifests import load_capture_manifest, save_yaml
from .mask_finalize import finalize_mask_dataset
from .masking import validate_mask_filter
from .mask_review import create_mask_review_server
from .run_lock import run_cli_locked
from .models import (
    IMAGE_SUFFIXES,
    Camera,
    CaptureManifest,
    CaptureSource,
    LocationManifest,
    NormalizationSettings,
    PreparedInputConfig,
    Region,
    Rights,
    RunConfig,
    SceneManifest,
    SourceFile,
    SourceProbe,
    TimeSelection,
    utc_now,
)
from .paths import find_data_root, host_path, location_dir, scene_dir
from .pipeline import (
    ingest_capture,
    mask_run,
    preprocess_run,
    reconstruct_run,
    review_run,
    write_qa_report,
)
from .runs import create_run, load_run
from .sharp_select import run_select_sharp
from .sources import prepare_capture_input, probe_capture_source
from .media import sha256_file


app = typer.Typer(
    no_args_is_help=True,
    help=(
        "360 and perspective video to Gaussian Splatting workflow and catalog. "
        "Windows path: doctor → location/scene/capture → media/ingest → preprocess → "
        "mask → reconstruct → qa → train --segment → cleanup."
    ),
)
location_app = typer.Typer(no_args_is_help=True, help="Manage location manifests")
scene_app = typer.Typer(no_args_is_help=True, help="Manage scene manifests")
catalog_app = typer.Typer(no_args_is_help=True, help="Build and inspect the SQLite catalog")
qa_app = typer.Typer(no_args_is_help=True, help="Generate and review QA reports")
capture_app = typer.Typer(no_args_is_help=True, help="Manage capture manifests")
media_app = typer.Typer(no_args_is_help=True, help="Probe source media")
app.add_typer(location_app, name="location")
app.add_typer(scene_app, name="scene")
app.add_typer(catalog_app, name="catalog")
app.add_typer(qa_app, name="qa")
app.add_typer(capture_app, name="capture")
app.add_typer(media_app, name="media")
console = Console()

# --smoke ceilings: upper bounds only, so an explicitly smaller flag still wins and
# the resulting config hash stays reproducible.
SMOKE_DURATION_SECONDS = 5.0
SMOKE_PROJECTION_SIZE = 1024
SMOKE_MASK_QA_SAMPLE_COUNT = 8


def _data_root() -> Path:
    return find_data_root()


def _scene(location_id: str, scene_id: str) -> Path:
    path = scene_dir(_data_root(), location_id, scene_id)
    if not (path / "scene.yaml").is_file():
        raise typer.BadParameter(f"Scene does not exist: {location_id}/{scene_id}")
    return path


def _fatal(error: Exception) -> None:
    console.print(f"[red]Error:[/red] {error}")
    raise typer.Exit(1)


@app.command()
def doctor(
    minimum_free_gib: Annotated[float, typer.Option(help="Required free-space reserve")] = 20.0,
    backend: Annotated[str, typer.Option(help="Training backend to check: postshot, gsplat, or all")] = "postshot",
    require: Annotated[
        list[str] | None,
        typer.Option("--require", help="Require additional component: mediasdk, postshot, or colmap"),
    ] = None,
) -> None:
    """Check Windows reconstruction deps and the training backend."""
    try:
        checks = run_doctor(
            _data_root(), minimum_free_gib=minimum_free_gib, require=set(require or []), backend=backend
        )
    except Exception as error:
        _fatal(error)
    table = Table("Check", "Result", "Detail")
    for name, result in checks.items():
        if name == "ok":
            continue
        ok, detail = result
        optional = name in {"mediasdk", "postshot"} and name not in set(require or [])
        if name == "postshot" and backend in {"postshot", "all"}:
            optional = False
        label = "PASS" if ok else ("UNAVAILABLE" if optional else "FAIL")
        table.add_row(
            name,
            label,
            str(detail),
            style=None if ok or optional else "red",
        )
    console.print(table)
    if not checks["ok"]:
        raise typer.Exit(1)


def _source_files(source_type: str, sources: list[Path]) -> list[Path]:
    resolved = [host_path(item).absolute() for item in sources]
    if source_type == "equirect_sequence" and len(resolved) == 1 and resolved[0].is_dir():
        resolved = sorted(
            item
            for item in resolved[0].iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    if not resolved or any(not item.is_file() for item in resolved):
        raise FileNotFoundError("Every --source must resolve to an existing file")
    return resolved


def _stored_windows_path(path: Path) -> str:
    return str(path.absolute())


@capture_app.command("init")
def capture_init(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
    source_type: Annotated[
        str,
        typer.Option(
            "--source-type",
            help="equirect_video, equirect_sequence, insta360_insv, or perspective_video",
        ),
    ],
    source: Annotated[list[Path], typer.Option("--source", help="Explicit source path")],
    camera_model: Annotated[str, typer.Option("--camera-model")] = "unknown",
    camera_make: Annotated[str, typer.Option("--camera-make")] = "Insta360",
    fps: Annotated[float | None, typer.Option(help="Image-sequence FPS")] = None,
    start_seconds: Annotated[float, typer.Option()] = 0.0,
    end_seconds: Annotated[float | None, typer.Option()] = None,
    output_width: Annotated[int | None, typer.Option()] = None,
    output_height: Annotated[int | None, typer.Option()] = None,
    horizontal_fov: Annotated[
        float,
        typer.Option(help="Horizontal FOV in degrees; perspective_video focal prior"),
    ] = 84.0,
) -> None:
    """Create and fingerprint a capture (panorama or landscape perspective video)."""
    try:
        if source_type not in {
            "equirect_video",
            "equirect_sequence",
            "insta360_insv",
            "perspective_video",
        }:
            raise ValueError(f"Unsupported source type: {source_type}")
        path = _scene(location_id, scene_id)
        manifest_path = path / "captures" / f"{capture_id}.yaml"
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        paths = _source_files(source_type, source)
        records = [
            SourceFile(
                windows_path=_stored_windows_path(item),
                sha256=sha256_file(item),
                byte_size=item.stat().st_size,
            )
            for item in paths
        ]
        perspective = source_type == "perspective_video"
        provisional = CaptureManifest(
            id=capture_id,
            location_id=location_id,
            scene_id=scene_id,
            source=CaptureSource(
                kind=source_type,
                files=records,
                projection=(
                    "dual_fisheye"
                    if source_type == "insta360_insv"
                    else "perspective"
                    if perspective
                    else "equirectangular"
                ),
                probe=SourceProbe(fps=fps),
            ),
            camera=Camera(make=camera_make, model=camera_model),
            selection=TimeSelection(start_seconds=0.0, end_seconds=1.0),
            normalization=NormalizationSettings(
                width=output_width, height=output_height
            ),
            horizontal_fov_degrees=horizontal_fov if perspective else None,
        )
        probe = probe_capture_source(provisional, path / "logs" / "probe-protocol")
        allowed_probe_fields = {
            name: value
            for name, value in probe.items()
            if name in SourceProbe.model_fields
        }
        provisional.source.probe = SourceProbe(**allowed_probe_fields)
        stop = float(
            probe["duration_seconds"] if end_seconds is None else end_seconds
        )
        provisional.selection = TimeSelection(
            start_seconds=start_seconds, end_seconds=stop
        )
        if stop > float(probe["duration_seconds"]) + 0.5:
            raise ValueError("Capture selection extends beyond the source duration")
        if not perspective and provisional.normalization.width is None:
            provisional.normalization.width = int(probe["width"])
            provisional.normalization.height = int(probe["height"])
        # The probe-derived values above are assigned after the provisional model
        # is constructed. Revalidate the complete document so a malformed source
        # (for example a non-2:1 panorama) can never be persisted.
        provisional = CaptureManifest.model_validate(provisional)
        save_yaml(manifest_path, provisional)
        console.print(
            f"Created [green]{manifest_path.relative_to(_data_root())}[/green]; "
            f"source={source_type}; files={len(records)}"
        )
    except Exception as error:
        _fatal(error)


@media_app.command("probe")
def media_probe(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
) -> None:
    """Probe a capture without preparing frames or changing its manifest."""
    try:
        path = _scene(location_id, scene_id)
        capture = load_capture_manifest(path / "captures" / f"{capture_id}.yaml")
        if isinstance(capture, CaptureManifest):
            from .media import probe_video

            result = probe_video(path / capture.stitched_video.relative_path)
        else:
            result = probe_capture_source(
                capture, path / "prepared" / ".protocol"
            )
        console.print_json(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        _fatal(error)


@location_app.command("init")
def location_init(
    location_id: Annotated[str, typer.Argument(help="Stable ASCII slug")],
    display_name: Annotated[str, typer.Option("--name", help="Human-readable name")],
    capture_date: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    province: Annotated[str | None, typer.Option()] = None,
    city: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Create a location manifest without any media files."""
    try:
        root = _data_root()
        path = location_dir(root, location_id)
        manifest_path = path / "location.yaml"
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        manifest = LocationManifest(
            id=location_id,
            display_name=display_name,
            capture_date=date.fromisoformat(capture_date) if capture_date else None,
            region=Region(province=province, city=city),
            rights=Rights(),
        )
        save_yaml(manifest_path, manifest)
        console.print(f"Created [green]{manifest_path.relative_to(root)}[/green]")
    except Exception as error:
        _fatal(error)


@scene_app.command("init")
def scene_init(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument(help="Stable ASCII slug")],
    display_name: Annotated[str, typer.Option("--name")],
    description: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Create an independently reconstructable scene under a location."""
    try:
        root = _data_root()
        location_path = location_dir(root, location_id)
        load_model(location_path / "location.yaml", LocationManifest)
        path = scene_dir(root, location_id, scene_id)
        manifest_path = path / "scene.yaml"
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        manifest = SceneManifest(
            id=scene_id,
            location_id=location_id,
            display_name=display_name,
            description=description,
        )
        save_yaml(manifest_path, manifest)
        for relative in (
            "captures",
            "stitched",
            "exports",
            "qa",
        ):
            (path / relative).mkdir(parents=True, exist_ok=True)
        console.print(f"Created [green]{manifest_path.relative_to(root)}[/green]")
    except Exception as error:
        _fatal(error)


@app.command()
def ingest(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
) -> None:
    """Re-probe the registered source set and refresh its recorded media facts."""
    try:
        path = _scene(location_id, scene_id)
        result = ingest_capture(path, capture_id)
        probe = result.source.probe
        console.print(
            f"Ingested [green]{probe.width}x{probe.height}[/green] "
            f"{probe.duration_seconds:.2f}s; files={len(result.source.files)}"
        )
    except Exception as error:
        _fatal(error)


@app.command()
def preprocess(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str | None, typer.Option(help="Existing run ID when resuming")] = None,
    resume: Annotated[bool, typer.Option(help="Reuse completed work in the same run")] = False,
    candidate_fps: Annotated[float, typer.Option(help="Candidate frame sampling rate")] = 1.0,
    selected_per_second: Annotated[
        int, typer.Option(help="Best candidates retained in each one-second bucket")
    ] = 1,
    primary_per_second: Annotated[
        int, typer.Option(help="Temporal ranks used by the primary attempt")
    ] = 1,
    primary_fov: Annotated[
        float, typer.Option(help="Horizontal FOV of each projected view in degrees")
    ] = 110.0,
    primary_projection_size: Annotated[int, typer.Option(help="Square projected view size")] = 1746,
    keep_intermediates: Annotated[bool, typer.Option(help="Keep intermediate files for debugging")] = False,
    mask_device: Annotated[str, typer.Option(help="Person segmenter device: cuda or cpu")] = "cuda",
    mask_score_threshold: Annotated[
        float, typer.Option(help="Mask R-CNN person confidence threshold")
    ] = 0.25,
    mask_probability_threshold: Annotated[
        float, typer.Option(help="Per-pixel probability threshold inside a person instance")
    ] = 0.50,
    mask_gamma: Annotated[
        float, typer.Option(help="Gamma used only for night-time segmentation inference")
    ] = 0.75,
    mask_dilation_pixels: Annotated[
        int, typer.Option(help="Pixel radius added around detected people")
    ] = 24,
    mask_discard_threshold: Annotated[
        float, typer.Option(help="Delete a perspective view when its ignored fraction exceeds this value")
    ] = 0.005,
    mask_review_gate: Annotated[
        bool,
        typer.Option("--mask-review-gate/--no-mask-review-gate", help="Require human mask finalization"),
    ] = False,
    mask_qa_sample_count: Annotated[
        int, typer.Option(help="Perspective views sampled for person-mask QA")
    ] = 16,
    mask_classes: Annotated[
        str,
        typer.Option(help="Comma-separated COCO dynamic classes; default person"),
    ] = "person",
    loop_closure: Annotated[
        bool, typer.Option("--loop-closure/--no-loop-closure")
    ] = False,
    vocabulary_tree: Annotated[
        Path | None,
        typer.Option(help="Explicit local COLMAP vocabulary tree; never downloaded"),
    ] = None,
    loop_period: Annotated[int, typer.Option()] = 20,
    loop_num_images: Annotated[int, typer.Option()] = 5,
    vision_qa: Annotated[
        bool,
        typer.Option(
            "--vision-qa/--no-vision-qa",
            help="Gate masks through DeepSeek V4 Flash Vision",
        ),
    ] = False,
    smoke: Annotated[
        bool,
        typer.Option(
            "--smoke",
            help=(
                "Process only the first 5 seconds at the same 1/1 temporal density, "
                "with smaller projections; does not start training"
            ),
        ),
    ] = False,
) -> None:
    """Create a run, sample candidates by time density, and rank each second."""
    try:
        path = _scene(location_id, scene_id)
        if run_id:
            run = load_run(path, run_id)
        else:
            capture = load_capture_manifest(path / "captures" / f"{capture_id}.yaml")
            if smoke:
                if (
                    abs(candidate_fps - 1.0) > 1e-9
                    or selected_per_second != 1
                    or primary_per_second != 1
                ):
                    raise ValueError(
                        "--smoke fixes temporal density at 1 candidate fps, 1 selected "
                        "per second, primary rank 1"
                    )
                primary_projection_size = min(primary_projection_size, SMOKE_PROJECTION_SIZE)
                mask_qa_sample_count = min(mask_qa_sample_count, SMOKE_MASK_QA_SAMPLE_COUNT)
            effective_end = (
                min(
                    capture.selection.end_seconds,
                    capture.selection.start_seconds + SMOKE_DURATION_SECONDS,
                )
                if smoke
                else capture.selection.end_seconds
            )
            import uuid

            # The run owns its candidates: prepare straight into the run directory
            # under the preparation hash the config pins.
            run_id = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
            work = path / run_id
            work.mkdir(parents=True, exist_ok=True)
            try:
                candidate_set = prepare_capture_input(
                    path,
                    capture,
                    work / "inputs" / "primary",
                    candidate_fps,
                    selection_end_seconds=effective_end,
                    resume=resume,
                )
            except Exception:
                shutil.rmtree(work, ignore_errors=True)
                raise
            classes = [item.strip() for item in mask_classes.split(",") if item.strip()]
            loop_path: str | None = None
            loop_sha256: str | None = None
            if loop_closure:
                if vocabulary_tree is None:
                    raise ValueError("--loop-closure requires --vocabulary-tree")
                resolved_tree = host_path(vocabulary_tree).absolute()
                if not resolved_tree.is_file():
                    raise FileNotFoundError(resolved_tree)
                loop_path = str(resolved_tree)
                loop_sha256 = sha256_file(resolved_tree)
            perspective = capture.source.is_perspective
            config = RunConfig(
                capture_id=capture_id,
                input_dataset_sha256=candidate_set.dataset_sha256,
                input_relative_path=f"inputs/primary/{candidate_set.path.name}",
                retention={"mode": "keep" if keep_intermediates else "minimal"},
                input=PreparedInputConfig(
                    source_kind=capture.source.kind,
                    source_sha256=[item.sha256 for item in capture.source.files],
                    source_probe=candidate_set.source_probe,
                    normalization=capture.normalization,
                    selection=TimeSelection(
                        start_seconds=capture.selection.start_seconds,
                        end_seconds=effective_end,
                    ),
                    candidate_frame_indices=list(candidate_set.frame_indices),
                    candidate_fps=candidate_fps,
                    helper_version=candidate_set.helper_version,
                    sdk_version=candidate_set.sdk_version,
                ),
                preprocess={
                    "candidate_fps": candidate_fps,
                    "selected_per_second": selected_per_second,
                },
                masking={
                    "device": mask_device,
                    "score_threshold": mask_score_threshold,
                    "probability_threshold": mask_probability_threshold,
                    "inference_gamma": mask_gamma,
                    "dilation_pixels": mask_dilation_pixels,
                    "mask_discard_threshold": mask_discard_threshold,
                    "qa_sample_count": mask_qa_sample_count,
                    "classes": classes,
                    "mask_review_required": mask_review_gate,
                },
                vision_qa={"enabled": vision_qa},
                reconstruction={
                    "primary": {
                        "temporal_rank_limit": primary_per_second,
                        # A perspective source is already one frame per view; a
                        # panorama is cut into 14 overlapping views of 110 deg.
                        "projection_fov_degrees": (
                            float(capture.horizontal_fov_degrees) if perspective else primary_fov
                        ),
                        "panorama": (
                            None
                            if perspective
                            else {
                                "views": 14,
                                "size": primary_projection_size,
                                "crop_bottom": 0.15,
                            }
                        ),
                    },
                    "loop_closure": {
                        "enabled": loop_closure,
                        "period": loop_period,
                        "num_images": loop_num_images,
                        "vocabulary_tree_path": loop_path,
                        "vocabulary_tree_sha256": loop_sha256,
                    },
                },
            )
            run = create_run(path, location_id, scene_id, config, run_id=run_id)
            protocol = work / "inputs" / "primary" / ".protocol"
            if protocol.exists():
                (work / "logs").mkdir(exist_ok=True)
                protocol.rename(work / "logs" / "prepare-protocol")
            if smoke:
                console.print(
                    "[yellow]Smoke profile[/yellow]: first 5 seconds at 1 candidate fps "
                    "and 1 selected per second, smaller projections; training is separate. "
                    "Use it to validate the pipeline, never to judge quality."
                )
        run = preprocess_run(path, run, resume=resume)
        console.print(f"Run [green]{run.id}[/green]: preprocess={run.stages['preprocess'].status.value}")
        console.print(f"RUN_ID={run.id}", markup=False)
    except Exception as error:
        _fatal(error)


@app.command("mask")
def mask_command(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Project panoramas, segment people, and create the alignment masks."""
    try:
        path = _scene(location_id, scene_id)
        run = mask_run(path, load_run(path, run_id), resume=resume)
        metrics = run.metrics.get("masking", {}).get("primary", {})
        console.print(
            f"Run [green]{run.id}[/green]: mask={run.stages['mask'].status.value}; "
            f"views={metrics.get('planar_images', 0)}"
        )
    except Exception as error:
        _fatal(error)


@app.command()
def reconstruct(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Align masked images with RealityScan and apply the single bounded repair when required."""
    try:
        path = _scene(location_id, scene_id)
        run = reconstruct_run(path, load_run(path, run_id), resume=resume)
        console.print(f"Run [green]{run.id}[/green]: selected={run.selected_dataset}")
    except Exception as error:
        _fatal(error)


def _mask_attempt_path(path: Path, run_id: str, attempt: str) -> Path:
    if attempt not in {"primary", "repair"}:
        raise ValueError("Attempt must be primary or repair")
    dataset = path / run_id / f"reconstruction-{attempt}"
    if not (dataset / "images").is_dir() or not (dataset / "masks").is_dir():
        raise FileNotFoundError(f"Masked {attempt} dataset is missing: {dataset}")
    return dataset


@app.command("mask-review")
@run_cli_locked
def mask_review(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    attempt: Annotated[str, typer.Option(help="primary or repair")] = "primary",
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=0, max=65535)] = 8765,
) -> None:
    """Review reconstruction masks; labels are bound to each current mask hash."""
    server = None
    try:
        path = _scene(location_id, scene_id)
        dataset = _mask_attempt_path(path, run_id, attempt)
        server = create_mask_review_server(dataset, host=host, port=port)
        actual_host, actual_port = server.server_address[:2]
        browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
        console.print(
            f"Mask review: [green]http://{browser_host}:{actual_port}[/green]\n"
            f"Dataset: {dataset.absolute()}\nPress Ctrl+C to stop."
        )
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            console.print("\nMask review stopped.")
    except Exception as error:
        _fatal(error)
    finally:
        if server is not None:
            server.server_close()


@app.command("mask-finalize")
@run_cli_locked
def mask_finalize(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    attempt: Annotated[str, typer.Option(help="primary or repair")] = "primary",
) -> None:
    """Freeze the reviewed mask set and exclusion list for reconstruction."""
    try:
        path = _scene(location_id, scene_id)
        run = load_run(path, run_id)
        dataset = _mask_attempt_path(path, run_id, attempt)
        filtered = validate_mask_filter(dataset, verify_hashes=False)
        result = finalize_mask_dataset(
            dataset,
            {str(item["image"]) for item in filtered["accepted"]},
            maximum_included_masked_fraction=run.config.masking.mask_discard_threshold,
        )
        from .retention import auto_cleanup
        auto_cleanup(path, run)
        console.print(
            f"Mask final [green]passed[/green]: masks={result['counts']['masks']}; "
            f"excluded={result['counts']['excluded']}; "
            f"sha256={result['finalization_sha256']}"
        )
    except Exception as error:
        _fatal(error)


@app.command("postshot-review")
def postshot_review(
    dataset: Annotated[
        Path,
        typer.Option(help="Prepared Postshot input directory, e.g. .../postshot-input"),
    ],
    host: Annotated[
        str, typer.Option(help="Review server bind address; keep loopback for local use")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=0, max=65535)] = 8765,
) -> None:
    """Review Postshot images and occluder masks in a local web interface."""
    server = None
    try:
        target = host_path(dataset)
        server = create_mask_review_server(target, host=host, port=port)
        actual_host, actual_port = server.server_address[:2]
        browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
        console.print(
            f"Mask review: [green]http://{browser_host}:{actual_port}[/green]\n"
            f"Dataset: {target.absolute()}\nPress Ctrl+C to stop."
        )
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            console.print("\nMask review stopped.")
    except Exception as error:
        _fatal(error)
    finally:
        if server is not None:
            server.server_close()


@qa_app.command("report")
def qa_report(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Generate the review checklist and move the run to needs_review."""
    try:
        path = _scene(location_id, scene_id)
        report = write_qa_report(path, load_run(path, run_id), resume=resume)
        console.print(f"QA report: [green]{report}[/green]")
    except Exception as error:
        _fatal(error)


@qa_app.command("approve")
def qa_approve(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    notes: Annotated[str, typer.Option(help="Required human review note")],
) -> None:
    """Mark a human-reviewed artifact accepted."""
    try:
        path = _scene(location_id, scene_id)
        run = review_run(path, load_run(path, run_id), accepted=True, notes=notes)
        console.print(f"Run [green]{run.id}[/green] accepted")
    except Exception as error:
        _fatal(error)


@qa_app.command("reject")
def qa_reject(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    notes: Annotated[str, typer.Option(help="Required rejection reason")],
) -> None:
    """Reject a human-reviewed artifact without deleting it."""
    try:
        path = _scene(location_id, scene_id)
        run = review_run(path, load_run(path, run_id), accepted=False, notes=notes)
        console.print(f"Run [yellow]{run.id}[/yellow] rejected")
    except Exception as error:
        _fatal(error)


@catalog_app.command("build")
def catalog_build() -> None:
    """Atomically rebuild catalog/catalog.sqlite from YAML manifests."""
    try:
        root = _data_root()
        counts = build_catalog(root)
        console.print("Catalog rebuilt: " + json.dumps(counts, ensure_ascii=False))
    except Exception as error:
        _fatal(error)


@qa_app.command("segments")
def qa_segments(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    attempt: Annotated[str | None, typer.Option(help="primary or repair; defaults to the selected attempt")] = None,
    output: Annotated[Path | None, typer.Option(help="Report path; defaults to the dataset")] = None,
) -> None:
    """Recompute auditable segment QA without changing reconstruction stage status."""
    from .segments import write_segments
    from .masking import validate_mask_filter
    from .mask_finalize import validate_mask_finalization
    try:
        scene=_scene(location_id,scene_id)
        run=load_run(scene,run_id)
        label=attempt or run.metrics.get('selected_attempt','primary')
        if label not in ('primary','repair'):
            raise ValueError('Unknown reconstruction attempt')
        work=scene/run_id
        dataset=work/f'reconstruction-{label}'
        records=[json.loads(line) for line in (work/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
        filtered=validate_mask_filter(dataset,verify_hashes=True)
        included={r['image'] for r in filtered['accepted']}
        final=validate_mask_finalization(dataset,included)
        included-=set(final.get('excluded_images',[]))
        report=write_segments(dataset,records,included,run.config.reconstruction.primary,run.config.segment_qa,output)
        console.print(json.dumps({k:report[k] for k in ('integrity','training_status','coverage_status')},ensure_ascii=False))
    except Exception as error:
        _fatal(error)


@app.command("train")
@run_cli_locked
def train_segment(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    segment: Annotated[str, typer.Option(help="Passed segment ID, e.g. segment-001")],
    output: Annotated[Path, typer.Option(help="New training experiment directory")],
    backend: Annotated[str, typer.Option(help="gsplat or postshot (ADC)")] = "postshot",
    steps: Annotated[int | None, typer.Option(min=1)] = None,
    photo_comp: Annotated[bool, typer.Option()] = True,
    use_bilateral_grid: Annotated[bool, typer.Option(help="gsplat: Bilateral Grid photometric compensation")] = True,
    use_sparse_depth: Annotated[bool, typer.Option(help="gsplat: COLMAP sparse-depth anchors")] = True,
    resume: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option(help="Prepare validated input and command without training")] = False,
    duration_seconds: Annotated[float | None, typer.Option(min=0.001,help="Earliest fully validated window within the segment")] = None,
) -> None:
    """Train a QA-approved segment with the gsplat or Postshot backend."""
    from .training_data import prepare_segment
    from .training import train_package
    from .runs import save_run
    try:
        scene = _scene(location_id,scene_id)
        run = load_run(scene,run_id)
        if run.stages['reconstruct'].status.value != 'succeeded' or not run.selected_dataset:
            raise RuntimeError('train --segment requires a successful reconstruction')
        label = run.metrics['selected_attempt']
        records = [json.loads(line) for line in (scene/run_id/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
        dataset = scene/run.selected_dataset
        package_name=segment if duration_seconds is None else f'{segment}-first-{duration_seconds:g}s'
        package = scene/run_id/'training-data'/package_name
        if segment not in {s['id'] for s in json.loads((dataset/'segments.json').read_text(encoding='utf-8'))['segments']}:
            raise ValueError('Unknown segment ID')
        prepare_segment(dataset,records,run.config.reconstruction.primary,run.config.segment_qa,segment,package,duration_seconds=duration_seconds,reuse_files=True)
        experiment = dict(segment=segment,backend=backend,output=str(output.resolve()),status='preparing')
        run.metrics.setdefault('training_experiments',[]).append(experiment)
        save_run(scene,run)
        try:
            result = train_package(package,output,backend,steps,photo_comp,resume,dry_run,
                use_bilateral_grid=use_bilateral_grid,use_sparse_depth=use_sparse_depth)
            experiment.update(result)
        except Exception as error:
            experiment.update(status='failed',error=str(error))
            raise
        finally:
            save_run(scene,run)
        console.print(json.dumps(result,ensure_ascii=False,indent=2))
    except Exception as error:
        _fatal(error)


@app.command("cleanup")
def cleanup_command(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    apply: Annotated[bool, typer.Option(help="Execute verified cleanup; default is preview")] = False,
) -> None:
    """Preview or apply verified retention cleanup for a Run."""
    from .retention import cleanup_run
    try:
        scene = _scene(location_id, scene_id)
        console.print_json(json.dumps(cleanup_run(scene, load_run(scene, run_id), apply=apply)))
    except Exception as error:
        _fatal(error)


@app.command("select-sharp")
def select_sharp(
    source: Annotated[list[Path], typer.Option("--source", help="Source media file; repeatable")],
    out: Annotated[Path, typer.Option(help="Output directory for the filtered frames")],
    kind: Annotated[
        str,
        typer.Option(help="equirect_video, equirect_sequence, insta360_insv, or perspective_video"),
    ] = "perspective_video",
    stage: Annotated[
        str,
        typer.Option(help="probe, extract, select, project, mask, filter, or all"),
    ] = "all",
    window: Annotated[
        str, typer.Option(help="Time window as start..end seconds; default is the whole source")
    ] = "",
    candidate_fps: Annotated[float, typer.Option(help="Candidate frame sampling rate")] = 5.0,
    selected_per_second: Annotated[int, typer.Option(help="Frames kept in each one-second bucket")] = 1,
    mask_threshold: Annotated[
        float, typer.Option(help="Reject a view when its ignored fraction exceeds this value")
    ] = 0.01,
    views: Annotated[int, typer.Option(help="Projected views per panorama: 8 or 14")] = 14,
    view_fov: Annotated[float, typer.Option(help="Horizontal FOV of each projected view")] = 110.0,
    view_size: Annotated[int, typer.Option(help="Square projected view size")] = 1746,
    crop_bottom: Annotated[float, typer.Option(help="Fraction of the panorama cropped from below")] = 0.15,
    target: Annotated[
        str,
        typer.Option(help="Image set for mask/filter: auto, flat, planar, or all"),
    ] = "auto",
    keep_candidates: Annotated[bool, typer.Option(help="Keep the extracted candidates")] = False,
    jpeg_quality: Annotated[int, typer.Option(min=1, max=100)] = 95,
    device: Annotated[str, typer.Option(help="Person segmenter device: cuda or cpu")] = "cuda",
) -> None:
    """Extract the sharpest frames from explicit sources and drop person-heavy views.

    A workbench tool outside the Run pipeline: no capture manifest, no run directory,
    no training. Interrupted work resumes by repeating the same command. Not a
    quality acceptance path.
    """
    try:
        run_select_sharp(
            sources=[host_path(item) for item in source],
            out=host_path(out),
            kind=kind,
            stage=stage,
            window=window,
            candidate_fps=candidate_fps,
            selected_per_second=selected_per_second,
            mask_threshold=mask_threshold,
            views=views,
            view_fov=view_fov,
            view_size=view_size,
            crop_bottom=crop_bottom,
            target=target,
            keep_candidates=keep_candidates,
            jpeg_quality=jpeg_quality,
            device=device,
        )
    except Exception as error:
        _fatal(error)


if __name__ == "__main__":
    app()
