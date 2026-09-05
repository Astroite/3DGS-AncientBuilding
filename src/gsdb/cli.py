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
from .manifests import load_capture_manifest, load_model, save_yaml
from .mask_finalize import expected_reconstruction_images, finalize_mask_dataset
from .mask_review import create_mask_review_server
from .models import (
    CaptureManifest,
    CaptureManifestV2,
    Camera,
    LocationManifest,
    NormalizationSettings,
    PanoramaSource,
    PreparedInputConfig,
    Region,
    Rights,
    RunConfig,
    RunConfigV3,
    SceneManifest,
    SourceFile,
    SourceProbe,
    TimeSelection,
)
from .paths import find_project_root, host_path, location_dir, scene_dir, wsl_to_windows
from .postshot import default_postshot_dataset
from .pipeline import (
    CullSettings,
    export_run,
    ingest_capture,
    mask_run,
    postshot_prepare_run,
    postshot_train_run,
    preprocess_run,
    reconstruct_run,
    review_run,
    write_qa_report,
)
from .runs import create_run, load_run
from .sources import (
    IMAGE_SUFFIXES,
    prepare_capture_input,
    probe_capture_source,
)
from .media import sha256_file


app = typer.Typer(no_args_is_help=True, help="360 video to Gaussian Splatting workflow and catalog")
location_app = typer.Typer(no_args_is_help=True, help="Manage location manifests")
scene_app = typer.Typer(no_args_is_help=True, help="Manage scene manifests")
catalog_app = typer.Typer(no_args_is_help=True, help="Build and inspect the SQLite catalog")
qa_app = typer.Typer(no_args_is_help=True, help="Generate and review QA reports")
capture_app = typer.Typer(no_args_is_help=True, help="Manage capture manifests")
media_app = typer.Typer(no_args_is_help=True, help="Probe panorama media")
app.add_typer(location_app, name="location")
app.add_typer(scene_app, name="scene")
app.add_typer(catalog_app, name="catalog")
app.add_typer(qa_app, name="qa")
app.add_typer(capture_app, name="capture")
app.add_typer(media_app, name="media")
console = Console()

# Ceilings applied by --smoke. Every value is an upper bound, so an explicitly
# smaller flag still wins and the resulting config hash stays reproducible.
SMOKE_PROFILE: dict[str, int] = {
    "target_frames": 80,
    "primary_frames": 40,
    "fallback_frames": 48,
    "projection_size": 1024,
    "mask_qa_sample_count": 8,
    "train_iterations": 5000,
}


SMOKE_CEILINGS: dict[str, int] = {
    "target_frames": SMOKE_PROFILE["target_frames"],
    "primary_frames": SMOKE_PROFILE["primary_frames"],
    "fallback_frames": SMOKE_PROFILE["fallback_frames"],
    "primary_projection_size": SMOKE_PROFILE["projection_size"],
    "fallback_projection_size": SMOKE_PROFILE["projection_size"],
    "mask_qa_sample_count": SMOKE_PROFILE["mask_qa_sample_count"],
    "train_iterations": SMOKE_PROFILE["train_iterations"],
}


def apply_smoke_profile(settings: dict[str, int | None]) -> dict[str, int | None]:
    """Clamp preprocess settings to the smoke ceilings without ever raising them."""
    unknown = set(settings) - set(SMOKE_CEILINGS)
    if unknown:
        raise KeyError(f"No smoke ceiling defined for: {sorted(unknown)}")
    return {
        key: SMOKE_CEILINGS[key] if value is None else min(int(value), SMOKE_CEILINGS[key])
        for key, value in settings.items()
    }


def _root() -> Path:
    return find_project_root()


def _scene(location_id: str, scene_id: str) -> Path:
    path = scene_dir(_root(), location_id, scene_id)
    if not (path / "scene.yaml").is_file():
        raise typer.BadParameter(f"Scene does not exist: {location_id}/{scene_id}")
    return path


def _fatal(error: Exception) -> None:
    console.print(f"[red]Error:[/red] {error}")
    raise typer.Exit(1)


@app.command()
def doctor(
    minimum_free_gib: Annotated[float, typer.Option(help="Required free-space reserve")] = 20.0,
    require: Annotated[
        list[str] | None,
        typer.Option("--require", help="Require optional component: mediasdk or postshot"),
    ] = None,
) -> None:
    """Verify project write access, command-line tools, CUDA, and gsplat."""
    checks = run_doctor(
        _root(), minimum_free_gib=minimum_free_gib, require=set(require or [])
    )
    table = Table("Check", "Result", "Detail")
    for name, result in checks.items():
        if name == "ok":
            continue
        ok, detail = result
        optional = name in {"mediasdk", "postshot"} and name not in set(require or [])
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
    value = wsl_to_windows(str(path.absolute()))
    return value if value != str(path.absolute()) or os.name == "nt" else str(path.absolute())


@capture_app.command("init")
def capture_init(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
    source_type: Annotated[
        str,
        typer.Option(
            "--source-type",
            help="equirect_video, equirect_sequence, or insta360_insv",
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
) -> None:
    """Create and fingerprint a schema 2 panorama capture."""
    try:
        if source_type not in {
            "equirect_video",
            "equirect_sequence",
            "insta360_insv",
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
        provisional = CaptureManifestV2(
            id=capture_id,
            location_id=location_id,
            scene_id=scene_id,
            source=PanoramaSource(
                kind=source_type,
                files=records,
                projection=(
                    "dual_fisheye"
                    if source_type == "insta360_insv"
                    else "equirectangular"
                ),
                probe=SourceProbe(fps=fps),
            ),
            camera=Camera(make=camera_make, model=camera_model),
            selection=TimeSelection(start_seconds=0.0, end_seconds=1.0),
            normalization=NormalizationSettings(
                width=output_width, height=output_height
            ),
        )
        probe = probe_capture_source(
            provisional, path / "inputs" / "prepared" / ".protocol"
        )
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
        if provisional.normalization.width is None:
            provisional.normalization.width = int(probe["width"])
            provisional.normalization.height = int(probe["height"])
        # The probe-derived values above are assigned after the provisional model
        # is constructed. Revalidate the complete document so a malformed source
        # (for example a non-2:1 panorama) can never be persisted.
        provisional = CaptureManifestV2.model_validate(
            provisional.model_dump(mode="python")
        )
        save_yaml(manifest_path, provisional)
        console.print(
            f"Created [green]{manifest_path.relative_to(_root())}[/green]; "
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
                capture, path / "inputs" / "prepared" / ".protocol"
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
        root = _root()
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
        (path / "scenes").mkdir(parents=True, exist_ok=True)
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
        root = _root()
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
            "inputs/stitched",
            "work",
            "runs",
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
    stitched: Annotated[
        Path | None,
        typer.Option(help="2:1 equirectangular MP4; defaults to the capture manifest path"),
    ] = None,
    resume: Annotated[bool, typer.Option(help="Revalidate an already ingested capture")] = False,
) -> None:
    """Validate and fingerprint a manually stitched 360 MP4."""
    try:
        root = _root()
        path = _scene(location_id, scene_id)
        capture_manifest = load_capture_manifest(
            path / "captures" / f"{capture_id}.yaml"
        )
        if isinstance(capture_manifest, CaptureManifest):
            input_path = (
                host_path(stitched)
                if stitched is not None
                else path / capture_manifest.stitched_video.relative_path
            )
        else:
            input_path = host_path(stitched) if stitched is not None else None
        if input_path is not None and not input_path.is_absolute():
            input_path = root / input_path
        result = ingest_capture(root, path, capture_id, input_path, resume=resume)
        if isinstance(result, CaptureManifest):
            console.print(
                f"Ingested [green]{result.stitched_video.width}x{result.stitched_video.height}[/green] "
                f"{result.stitched_video.duration_seconds:.2f}s; sha256={result.stitched_video.sha256}"
            )
        else:
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
    target_frames: Annotated[int, typer.Option()] = 270,
    primary_frames: Annotated[
        int, typer.Option(help="Blur-aware frames selected for the primary reconstruction")
    ] = 135,
    primary_fov: Annotated[
        float, typer.Option(help="Primary perspective horizontal FOV in degrees")
    ] = 120.0,
    primary_projection_size: Annotated[
        int, typer.Option(help="Primary square perspective image size")
    ] = 2048,
    fallback_frames: Annotated[int, typer.Option()] = 180,
    fallback_fov: Annotated[float, typer.Option()] = 110.0,
    fallback_projection_size: Annotated[int, typer.Option()] = 1746,
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
    max_masked_fraction: Annotated[
        float, typer.Option(help="Reject a view when its ignored fraction exceeds this value")
    ] = 0.45,
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
                "Reduced end-to-end profile for plumbing checks: fewer frames, smaller "
                "projections and a short training run. Never a quality baseline."
            ),
        ),
    ] = False,
    train_iterations: Annotated[
        int | None, typer.Option(help="Override the Splatfacto iteration count")
    ] = None,
) -> None:
    """Create a run, extract uniform panorama frames, and measure input quality."""
    try:
        path = _scene(location_id, scene_id)
        if run_id:
            run = load_run(path, run_id)
        else:
            if smoke:
                # A distinct config hash keeps smoke output in its own run directory,
                # so it can never be mistaken for or overwrite a full-quality run.
                reduced = apply_smoke_profile(
                    {
                        "target_frames": target_frames,
                        "primary_frames": primary_frames,
                        "fallback_frames": fallback_frames,
                        "primary_projection_size": primary_projection_size,
                        "fallback_projection_size": fallback_projection_size,
                        "mask_qa_sample_count": mask_qa_sample_count,
                        "train_iterations": train_iterations,
                    }
                )
                target_frames = int(reduced["target_frames"])
                primary_frames = int(reduced["primary_frames"])
                fallback_frames = int(reduced["fallback_frames"])
                primary_projection_size = int(reduced["primary_projection_size"])
                fallback_projection_size = int(reduced["fallback_projection_size"])
                mask_qa_sample_count = int(reduced["mask_qa_sample_count"])
                train_iterations = int(reduced["train_iterations"])
            capture = load_capture_manifest(
                path / "captures" / f"{capture_id}.yaml"
            )
            masking = {
                "device": mask_device,
                "score_threshold": mask_score_threshold,
                "probability_threshold": mask_probability_threshold,
                "inference_gamma": mask_gamma,
                "dilation_pixels": mask_dilation_pixels,
                "max_masked_fraction": max_masked_fraction,
                "qa_sample_count": mask_qa_sample_count,
            }
            reconstruction = {
                "primary": {
                    "frame_count": primary_frames,
                    "images_per_equirect": 8,
                    "projection_fov_degrees": primary_fov,
                    "projection_size": primary_projection_size,
                    "crop_bottom": 0.20,
                    "use_rig": True,
                },
                "fallback": {
                    "frame_count": fallback_frames,
                    "images_per_equirect": 14,
                    "projection_fov_degrees": fallback_fov,
                    "projection_size": fallback_projection_size,
                    "crop_bottom": 0.15,
                    "use_rig": True,
                },
            }
            if isinstance(capture, CaptureManifest):
                if not capture.stitched_video.sha256:
                    raise RuntimeError(
                        "Capture has not been ingested; run gsdb ingest first"
                    )
                if loop_closure:
                    raise ValueError(
                        "Optional loop closure is available only for schema 2 captures/new v3 runs"
                    )
                config = RunConfig(
                    capture_id=capture_id,
                    input_sha256=capture.stitched_video.sha256,
                    preprocess={"target_frames": target_frames},
                    masking=masking,
                    vision_qa={"enabled": vision_qa},
                    reconstruction=reconstruction,
                )
            else:
                candidate_set = prepare_capture_input(
                    path, capture, target_frames=target_frames, resume=resume
                )
                capture.prepared_relative_path = candidate_set.path.relative_to(
                    path
                ).as_posix()
                save_yaml(path / "captures" / f"{capture_id}.yaml", capture)
                classes = [
                    item.strip() for item in mask_classes.split(",") if item.strip()
                ]
                loop_path: str | None = None
                loop_sha256: str | None = None
                if loop_closure:
                    if vocabulary_tree is None:
                        raise ValueError(
                            "--loop-closure requires --vocabulary-tree"
                        )
                    resolved_tree = host_path(vocabulary_tree).absolute()
                    if not resolved_tree.is_file():
                        raise FileNotFoundError(resolved_tree)
                    loop_path = str(resolved_tree)
                    loop_sha256 = sha256_file(resolved_tree)
                reconstruction["loop_closure"] = {
                    "enabled": loop_closure,
                    "period": loop_period,
                    "num_images": loop_num_images,
                    "vocabulary_tree_path": loop_path,
                    "vocabulary_tree_sha256": loop_sha256,
                }
                config = RunConfigV3(
                    capture_id=capture_id,
                    input_dataset_sha256=candidate_set.dataset_sha256,
                    prepared_relative_path=capture.prepared_relative_path,
                    input=PreparedInputConfig(
                        source_kind=capture.source.kind,
                        source_sha256=[
                            item.sha256 for item in capture.source.files
                        ],
                        source_probe=SourceProbe.model_validate(
                            candidate_set.source_probe
                        ),
                        normalization=capture.normalization,
                        selection=capture.selection,
                        candidate_frame_indices=list(candidate_set.frame_indices),
                        helper_version=candidate_set.helper_version,
                        sdk_version=candidate_set.sdk_version,
                    ),
                    preprocess={"target_frames": target_frames},
                    masking={**masking, "classes": classes},
                    vision_qa={"enabled": vision_qa},
                    reconstruction=reconstruction,
                )
            if train_iterations is not None:
                config.train.max_iterations = train_iterations
            run = create_run(path, location_id, scene_id, config)
            if smoke:
                console.print(
                    "[yellow]Smoke profile[/yellow]: reduced frames, projection size and "
                    "training length. Use it to validate the pipeline, never to judge quality."
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
    """Project panoramas, segment people, and create COLMAP/Nerfstudio masks."""
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
    """Align masked images with RealityScan and apply the single bounded fallback when required."""
    try:
        path = _scene(location_id, scene_id)
        run = reconstruct_run(path, load_run(path, run_id), resume=resume)
        console.print(f"Run [green]{run.id}[/green]: selected={run.selected_dataset}")
    except Exception as error:
        _fatal(error)


def _mask_attempt_path(path: Path, run_id: str, attempt: str) -> Path:
    if attempt not in {"primary", "fallback"}:
        raise ValueError("Attempt must be primary or fallback")
    dataset = path / "work" / run_id / f"reconstruction-{attempt}"
    if not (dataset / "images").is_dir() or not (dataset / "masks").is_dir():
        raise FileNotFoundError(f"Masked {attempt} dataset is missing: {dataset}")
    return dataset


@app.command("mask-review")
def mask_review(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    attempt: Annotated[str, typer.Option(help="primary or fallback")] = "primary",
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=0, max=65535)] = 8765,
) -> None:
    """Review reconstruction masks; labels are bound to each current mask hash."""
    server = None
    try:
        path = _scene(location_id, scene_id)
        run = load_run(path, run_id)
        if getattr(run.config, "schema_version", 1) != 3:
            raise RuntimeError("mask-review is available only for RunConfigV3 runs")
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
def mask_finalize(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    attempt: Annotated[str, typer.Option(help="primary or fallback")] = "primary",
) -> None:
    """Freeze the reviewed mask set and exclusion list for reconstruction."""
    try:
        path = _scene(location_id, scene_id)
        run = load_run(path, run_id)
        if getattr(run.config, "schema_version", 1) != 3:
            raise RuntimeError("mask-finalize is available only for RunConfigV3 runs")
        dataset = _mask_attempt_path(path, run_id, attempt)
        attempt_config = getattr(run.config.reconstruction, attempt)
        result = finalize_mask_dataset(
            dataset,
            expected_reconstruction_images(
                attempt_config.frame_count, attempt_config.images_per_equirect
            ),
        )
        console.print(
            f"Mask final [green]passed[/green]: masks={result['counts']['masks']}; "
            f"excluded={result['counts']['excluded']}; "
            f"sha256={result['finalization_sha256']}"
        )
    except Exception as error:
        _fatal(error)


@app.command("postshot-prepare")
def postshot_prepare(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    resume: Annotated[
        bool, typer.Option(help="Continue a partial .building dataset")
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(help="Optional output directory; v3 defaults to inputs/postshot/<run-id>"),
    ] = None,
) -> None:
    """Prepare registered images, poses, points and occluder masks for Postshot."""
    try:
        root = _root()
        path = _scene(location_id, scene_id)
        target = host_path(output) if output is not None else None
        if target is not None and not target.is_absolute():
            target = root / target
        run = postshot_prepare_run(
            path,
            load_run(path, run_id),
            resume=resume,
            output=target,
        )
        console.print(
            f"Run [green]{run.id}[/green]: "
            f"postshot_prepare={run.stages['postshot_prepare'].status.value}"
        )
    except Exception as error:
        _fatal(error)


@app.command("postshot-review")
def postshot_review(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    dataset: Annotated[
        Path | None,
        typer.Option(help="Prepared Postshot directory; defaults to the run's native staging path"),
    ] = None,
    host: Annotated[
        str, typer.Option(help="Review server bind address; keep loopback for local use")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=0, max=65535)] = 8765,
) -> None:
    """Review Postshot images and occluder masks in a local web interface."""
    server = None
    try:
        root = _root()
        path = _scene(location_id, scene_id)
        run = load_run(path, run_id)
        target = (
            host_path(dataset)
            if dataset is not None
            else default_postshot_dataset(path, run)
        )
        if not target.is_absolute():
            target = root / target
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


@app.command("postshot-train")
def postshot_train(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    resume: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    allow_low_vram: Annotated[bool, typer.Option("--allow-low-vram")] = False,
    profile: Annotated[str, typer.Option()] = "Splat3",
    ksteps: Annotated[int | None, typer.Option(help="Postshot training kSteps")] = None,
    max_splats: Annotated[
        int | None, typer.Option(help="Postshot maximum splats in kSplats")
    ] = None,
    gpu: Annotated[int, typer.Option()] = 0,
    dataset: Annotated[
        Path | None, typer.Option(help="Prepared Postshot dataset override")
    ] = None,
    output: Annotated[Path | None, typer.Option()] = None,
    store_training_context: Annotated[
        bool,
        typer.Option("--store-training-context/--no-store-training-context"),
    ] = True,
    export_ply: Annotated[Path | None, typer.Option()] = None,
    export_spz: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Validate and launch Postshot with imported COLMAP poses and occluder masks."""
    try:
        root = _root()
        path = _scene(location_id, scene_id)
        def resolved(value: Path | None) -> Path | None:
            if value is None:
                return None
            target = host_path(value)
            return target if target.is_absolute() else root / target

        run, result = postshot_train_run(
            path,
            load_run(path, run_id),
            resume=resume,
            dry_run=dry_run,
            allow_low_vram=allow_low_vram,
            profile=profile,
            ksteps=ksteps,
            max_splats=max_splats,
            gpu=gpu,
            dataset=resolved(dataset),
            output=resolved(output),
            store_training_context=store_training_context,
            export_ply=resolved(export_ply),
            export_spz=resolved(export_spz),
        )
        if dry_run:
            console.print_json(json.dumps(result, ensure_ascii=False))
        else:
            console.print(
                f"Run [green]{run.id}[/green]: train={run.stages['train'].status.value}; "
                f"output={result['output']}; sha256={result['output_sha256']}"
            )
    except Exception as error:
        _fatal(error)


@app.command("export")
def export_command(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    version: Annotated[str, typer.Option()] = "v001",
    resume: Annotated[bool, typer.Option()] = False,
    cull: Annotated[
        bool,
        typer.Option(
            "--cull/--no-cull",
            help="Drop Gaussians far from the capture path or larger than the scene",
        ),
    ] = True,
    cull_distance_factor: Annotated[
        float,
        typer.Option(help="Cull beyond this multiple of the capture path radius"),
    ] = 3.0,
    cull_scale_factor: Annotated[
        float,
        typer.Option(help="Cull Gaussians wider than this multiple of the path radius"),
    ] = 1.0,
    cull_max_removed_fraction: Annotated[
        float,
        typer.Option(help="Refuse to publish if culling would remove more than this"),
    ] = 0.05,
    allow_unsafe_publish_frame: Annotated[
        bool,
        typer.Option(
            "--allow-unsafe-publish-frame",
            help=(
                "Explicitly bypass publish-frame gravity/height gates and mark the "
                "artifact as requiring manual review"
            ),
        ),
    ] = False,
) -> None:
    """Export PLY, preview, thumbnail, transforms, and artifact manifest.

    Culling is a publishing decision, not a training one, so these options are not
    part of the run's config hash: the same run can publish several versions.
    """
    try:
        path = _scene(location_id, scene_id)
        run = export_run(
            path,
            load_run(path, run_id),
            version=version,
            resume=resume,
            cull=CullSettings(
                enabled=cull,
                distance_factor=cull_distance_factor,
                scale_factor=cull_scale_factor,
                max_removed_fraction=cull_max_removed_fraction,
            ),
            allow_unsafe_publish_frame=allow_unsafe_publish_frame,
        )
        published = run.metrics.get("export", {}).get("cull", {})
        console.print(f"Run [green]{run.id}[/green]: artifacts={len(run.artifacts)}")
        if published.get("removed_total") is not None:
            console.print(
                f"Culled [yellow]{published['removed_total']:,}[/yellow] of "
                f"{published['input_gaussians']:,} Gaussians "
                f"({published['removed_fraction']:.2%}); published "
                f"{published['published_gaussians']:,}"
            )
        publish_frame = run.metrics.get("export", {}).get("publish_frame", {})
        if publish_frame.get("manual_review_required"):
            console.print(
                "[bold red]UNSAFE PUBLISH FRAME: manual viewer review is required; "
                "see UNSAFE-PUBLISH-FRAME.txt[/bold red]"
            )
    except Exception as error:
        _fatal(error)


@qa_app.command("report")
def qa_report(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    baseline_run_id: Annotated[
        str | None, typer.Option(help="Previous run ID used for a v1/v2 comparison")
    ] = None,
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Generate the review checklist and move the run to needs_review."""
    try:
        path = _scene(location_id, scene_id)
        baseline = load_run(path, baseline_run_id) if baseline_run_id else None
        report = write_qa_report(
            path, load_run(path, run_id), baseline=baseline, resume=resume
        )
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
        root = _root()
        counts = build_catalog(root)
        console.print("Catalog rebuilt: " + json.dumps(counts, ensure_ascii=False))
    except Exception as error:
        _fatal(error)


@app.command()
def clean(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
) -> None:
    """Preview generated work data that could be cleaned; never deletes files in v1."""
    path = _scene(location_id, scene_id) / "work"
    total = 0
    for run_path in sorted(item for item in path.iterdir() if item.is_dir()):
        size = sum(item.stat().st_size for item in run_path.rglob("*") if item.is_file())
        total += size
        console.print(f"{run_path.name}: {size / 1024**3:.2f} GiB")
    console.print(f"Total preview only: {total / 1024**3:.2f} GiB; nothing was deleted")


if __name__ == "__main__":
    app()
