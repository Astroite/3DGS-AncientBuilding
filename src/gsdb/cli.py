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
from .masking import validate_mask_filter
from .mask_review import create_mask_review_server
from .run_lock import run_cli_locked
from .models import (
    CaptureManifest,
    CaptureManifestV2,
    Camera,
    LocationManifest,
    NormalizationSettings,
    PanoramaSource,
    PreparedInputConfig,
    PreparedInputConfigV2,
    Region,
    Rights,
    RunConfig,
    RunConfigV3,
    RunConfigV4,
    RunConfigV5,
    RunConfigV6,
    SceneManifest,
    SourceFile,
    SourceProbe,
    TimeSelection,
)
from .paths import find_data_root, host_path, location_dir, scene_dir, wsl_to_windows
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


app = typer.Typer(
    no_args_is_help=True,
    help=(
        "360 video to Gaussian Splatting workflow and catalog. "
        "Current Windows path (schema 6): doctor → location/scene/capture → media/ingest → "
        "preprocess → mask → reconstruct → qa → train --segment → cleanup. "
        "Commands marked (legacy) remain for historical schema 1–4 Runs only."
    ),
)
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

SMOKE_DURATION_SECONDS = 5.0


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
        typer.Option("--require", help="Require additional component: mediasdk, postshot, or legacy colmap"),
    ] = None,
) -> None:
    """Current workflow: check Windows reconstruction deps and the training backend."""
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
            provisional, path / "prepared" / ".protocol"
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
    stitched: Annotated[
        Path | None,
        typer.Option(help="2:1 equirectangular MP4; defaults to the capture manifest path"),
    ] = None,
    resume: Annotated[bool, typer.Option(help="Revalidate an already ingested capture")] = False,
) -> None:
    """Validate and fingerprint a manually stitched 360 MP4."""
    try:
        root = _data_root()
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
    target_frames: Annotated[
        int | None,
        typer.Option(help="Deprecated fixed candidate count; legacy runs only"),
    ] = None,
    primary_frames: Annotated[
        int | None, typer.Option(help="Deprecated; legacy runs only")
    ] = None,
    primary_fov: Annotated[
        float, typer.Option(help="Primary perspective horizontal FOV in degrees")
    ] = 110.0,
    primary_projection_size: Annotated[
        int, typer.Option(help="Primary square perspective image size")
    ] = 1746,
    fallback_frames: Annotated[int | None, typer.Option(help="Deprecated; legacy runs only")] = None,
    candidate_fps: Annotated[float, typer.Option(help="Candidate panorama sampling rate")] = 1.0,
    selected_per_second: Annotated[
        int, typer.Option(help="Best panorama candidates retained in each one-second bucket")
    ] = 1,
    primary_per_second: Annotated[
        int, typer.Option(help="Temporal ranks used by the primary attempt")
    ] = 1,
    fallback_per_second: Annotated[
        int, typer.Option(help="Temporal ranks used by the fallback attempt")
    ] = 1,
    keep_intermediates: Annotated[bool, typer.Option(help="Keep schema 6 intermediate files for debugging")] = False,
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
        float | None,
        typer.Option(help="Deprecated mask ceiling; legacy runs only"),
    ] = None,
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
    train_iterations: Annotated[
        int | None, typer.Option(help="Legacy Splatfacto setting; segment training uses train --steps")
    ] = None,
) -> None:
    """Create a run, sample candidates by time density, and rank each second."""
    try:
        path = _scene(location_id, scene_id)
        if run_id:
            run = load_run(path, run_id)
        else:
            capture = load_capture_manifest(
                path / "captures" / f"{capture_id}.yaml"
            )
            if isinstance(capture, CaptureManifest):
                legacy_target = target_frames if target_frames is not None else 270
                legacy_primary = primary_frames if primary_frames is not None else 135
                legacy_fallback = fallback_frames if fallback_frames is not None else 180
                if smoke:
                    reduced = apply_smoke_profile(
                        {
                            "target_frames": legacy_target,
                            "primary_frames": legacy_primary,
                            "fallback_frames": legacy_fallback,
                            "primary_projection_size": primary_projection_size,
                            "fallback_projection_size": fallback_projection_size,
                            "mask_qa_sample_count": mask_qa_sample_count,
                            "train_iterations": train_iterations,
                        }
                    )
                    legacy_target = int(reduced["target_frames"])
                    legacy_primary = int(reduced["primary_frames"])
                    legacy_fallback = int(reduced["fallback_frames"])
                    primary_projection_size = int(reduced["primary_projection_size"])
                    fallback_projection_size = int(reduced["fallback_projection_size"])
                    mask_qa_sample_count = int(reduced["mask_qa_sample_count"])
                    train_iterations = int(reduced["train_iterations"])
                if not capture.stitched_video.sha256:
                    raise RuntimeError(
                        "Capture has not been ingested; run gsdb ingest first"
                    )
                if loop_closure:
                    raise ValueError(
                        "Optional loop closure is available only for schema 2 captures/new v3 runs"
                    )
                legacy_masking = {
                    "device": mask_device,
                    "score_threshold": mask_score_threshold,
                    "probability_threshold": mask_probability_threshold,
                    "inference_gamma": mask_gamma,
                    "dilation_pixels": mask_dilation_pixels,
                    "max_masked_fraction": (
                        max_masked_fraction
                        if max_masked_fraction is not None
                        else 0.45
                    ),
                    "qa_sample_count": mask_qa_sample_count,
                }
                legacy_reconstruction = {
                    "primary": {
                        "frame_count": legacy_primary,
                        "images_per_equirect": 8,
                        "projection_fov_degrees": primary_fov,
                        "projection_size": primary_projection_size,
                        "crop_bottom": 0.20,
                        "use_rig": True,
                    },
                    "fallback": {
                        "frame_count": legacy_fallback,
                        "images_per_equirect": 14,
                        "projection_fov_degrees": fallback_fov,
                        "projection_size": fallback_projection_size,
                        "crop_bottom": 0.15,
                        "use_rig": True,
                    },
                }
                config = RunConfig(
                    capture_id=capture_id,
                    input_sha256=capture.stitched_video.sha256,
                    preprocess={"target_frames": legacy_target},
                    masking=legacy_masking,
                    vision_qa={"enabled": vision_qa},
                    reconstruction=legacy_reconstruction,
                )
            else:
                if any(
                    value is not None
                    for value in (target_frames, primary_frames, fallback_frames, max_masked_fraction)
                ):
                    raise ValueError(
                        "Fixed frame-count and max-mask flags are retired for new runs; "
                        "use --candidate-fps, --selected-per-second, "
                        "--primary-per-second, --fallback-per-second and "
                        "--mask-discard-threshold"
                    )
                if smoke and (
                    abs(candidate_fps - 1.0) > 1e-9
                    or selected_per_second != 1
                    or primary_per_second != 1
                    or fallback_per_second != 1
                ):
                    raise ValueError(
                        "--smoke fixes temporal density at 1 candidate fps, 1 selected "
                        "per second, Primary and repair baseline rank 1"
                    )
                if smoke:
                    primary_projection_size = min(primary_projection_size, 1024)
                    fallback_projection_size = min(fallback_projection_size, 1024)
                    mask_qa_sample_count = min(mask_qa_sample_count, 8)
                    train_iterations = min(train_iterations or 5000, 5000)
                effective_end = (
                    min(
                        capture.selection.end_seconds,
                        capture.selection.start_seconds + SMOKE_DURATION_SECONDS,
                    )
                    if smoke
                    else capture.selection.end_seconds
                )
                import uuid
                staging = path / '.preparing' / uuid.uuid4().hex
                candidate_set = prepare_capture_input(
                    path,
                    capture,
                    candidate_fps=candidate_fps,
                    selection_end_seconds=effective_end,
                    resume=resume,
                    cache_root=staging,
                    anchored=True,
                )
                owned_input = f'inputs/primary/{candidate_set.path.name}'
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
                reconstruction = {
                    "primary": {
                        "temporal_rank_limit": primary_per_second,
                        "images_per_equirect": 14,
                        "projection_fov_degrees": primary_fov,
                        "projection_size": primary_projection_size,
                        "crop_bottom": 0.15,
                        "use_rig": False,
                    },
                    "fallback": {
                        "temporal_rank_limit": fallback_per_second,
                        "images_per_equirect": 14,
                        "projection_fov_degrees": fallback_fov,
                        "projection_size": fallback_projection_size,
                        "crop_bottom": 0.15,
                        "use_rig": False,
                    },
                    "loop_closure": {
                        "enabled": loop_closure,
                        "period": loop_period,
                        "num_images": loop_num_images,
                        "vocabulary_tree_path": loop_path,
                        "vocabulary_tree_sha256": loop_sha256,
                    },
                }
                config = RunConfigV6(
                    capture_id=capture_id,
                    input_dataset_sha256=candidate_set.dataset_sha256,
                    prepared_relative_path=owned_input,
                    retention={"mode": "keep" if keep_intermediates else "minimal"},
                    input=PreparedInputConfigV2(
                        source_kind=capture.source.kind,
                        source_sha256=[
                            item.sha256 for item in capture.source.files
                        ],
                        source_probe=SourceProbe.model_validate(
                            candidate_set.source_probe
                        ),
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
                    reconstruction=reconstruction,
                )
            if train_iterations is not None:
                config.train.max_iterations = train_iterations
            run = create_run(path, location_id, scene_id, config)
            if config.schema_version == 6:
                from .training_data import json_write
                json_write(path / run.id / 'input-adoption.json', dict(
                    source=candidate_set.path.relative_to(path).as_posix(),
                    target=config.prepared_relative_path, input_sha256=config.input_dataset_sha256))
                destination = path / run.id / config.prepared_relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                candidate_set.path.rename(destination)
                protocol = staging / '.protocol'
                if protocol.exists():
                    (path / run.id / 'logs').mkdir(exist_ok=True)
                    protocol.rename(path / run.id / 'logs' / 'prepare-protocol')
                staging.rmdir()
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
    if attempt not in {"primary", "fallback", "repair"}:
        raise ValueError("Attempt must be primary, fallback or repair")
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
    attempt: Annotated[str, typer.Option(help="primary, repair, or legacy fallback")] = "primary",
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=0, max=65535)] = 8765,
) -> None:
    """Review reconstruction masks; labels are bound to each current mask hash."""
    server = None
    try:
        path = _scene(location_id, scene_id)
        run = load_run(path, run_id)
        if getattr(run.config, "schema_version", 1) not in (3, 4, 5, 6):
            raise RuntimeError("mask-review is available only for schema 3-6 runs")
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
    attempt: Annotated[str, typer.Option(help="primary, repair, or legacy fallback")] = "primary",
) -> None:
    """Freeze the reviewed mask set and exclusion list for reconstruction."""
    try:
        path = _scene(location_id, scene_id)
        run = load_run(path, run_id)
        schema_version = int(getattr(run.config, "schema_version", 1))
        if schema_version not in (3, 4, 5, 6):
            raise RuntimeError("mask-finalize is available only for schema 3-6 runs")
        dataset = _mask_attempt_path(path, run_id, attempt)
        attempt_config = getattr(run.config.reconstruction, "primary" if attempt == "repair" else attempt)
        if schema_version >= 4:
            filtered = validate_mask_filter(dataset, verify_hashes=False)
            expected_images = {str(item["image"]) for item in filtered["accepted"]}
            maximum_fraction = run.config.masking.mask_discard_threshold
        else:
            expected_images = expected_reconstruction_images(
                attempt_config.frame_count, attempt_config.images_per_equirect
            )
            maximum_fraction = None
        result = finalize_mask_dataset(
            dataset,
            expected_images,
            maximum_included_masked_fraction=maximum_fraction,
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
        typer.Option(help="Optional legacy dataset output directory; schema 5/6 uses train --segment"),
    ] = None,
) -> None:
    """(legacy) Historical Run preparation. Schema 5/6: train --backend postshot --segment --dry-run."""
    try:
        root = _data_root()
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
        root = _data_root()
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
    """(legacy) Historical dataset training. Schema 5/6: train --backend postshot --segment."""
    try:
        root = _data_root()
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
    """(legacy) Nerfstudio asset publishing; not schema 5/6 segment PLY export.

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
        root = _data_root()
        counts = build_catalog(root)
        console.print("Catalog rebuilt: " + json.dumps(counts, ensure_ascii=False))
    except Exception as error:
        _fatal(error)


@app.command()
def clean(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
) -> None:
    """(legacy) Preview Run directory sizes only. Schema 6 cleanup: use `cleanup`."""
    path = _scene(location_id, scene_id)
    total = 0
    for run_path in sorted(
        item for item in path.iterdir() if item.is_dir() and (item / "manifest.yaml").is_file()
    ):
        size = sum(item.stat().st_size for item in run_path.rglob("*") if item.is_file())
        total += size
        console.print(f"{run_path.name}: {size / 1024**3:.2f} GiB")
    console.print(f"Total preview only: {total / 1024**3:.2f} GiB; nothing was deleted")


@qa_app.command("segments")
def qa_segments(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    attempt: Annotated[str | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option(help="Legacy reports must be outside their Run directory")] = None,
) -> None:
    """Recompute auditable segment QA without changing reconstruction stage status."""
    from .segments import write_segments
    from .masking import validate_mask_filter
    from .mask_finalize import validate_mask_finalization
    from .models import SegmentQAConfig
    try:
        scene=_scene(location_id,scene_id)
        run=load_run(scene,run_id)
        if run.config.schema_version not in (4,5,6):
            raise ValueError('Temporal segment QA requires schema 4, 5 or 6')
        label=attempt or run.metrics.get('selected_attempt','primary')
        if label not in ('primary','fallback','repair') or (label=='repair' and run.config.schema_version not in (5,6)):
            raise ValueError('Unknown reconstruction attempt')
        work=scene/run_id
        if run.config.schema_version==4 and (output is None or output.resolve()==work.resolve() or work.resolve() in output.resolve().parents):
            raise ValueError('Legacy replay needs --output outside the historical Run')
        dataset=work/f'reconstruction-{label}'
        records=[json.loads(line) for line in (work/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
        filtered=validate_mask_filter(dataset,verify_hashes=True)
        included={r['image'] for r in filtered['accepted']}
        final=validate_mask_finalization(dataset,included)
        included-=set(final.get('excluded_images',[]))
        config=getattr(run.config.reconstruction,'primary' if label=='repair' else label)
        report=write_segments(dataset,records,included,config,getattr(run.config,'segment_qa',SegmentQAConfig()),output)
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
    resume: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option(help="Prepare validated input and command without training")] = False,
    duration_seconds: Annotated[float | None, typer.Option(min=0.001,help="Earliest fully validated window within the segment")] = None,
) -> None:
    """Current workflow: train a QA-approved schema 5/6 segment (gsplat or postshot)."""
    from .training_data import prepare_segment
    from .training import train_package
    from .runs import save_run
    try:
        scene = _scene(location_id,scene_id)
        run = load_run(scene,run_id)
        if run.config.schema_version not in (5, 6) or run.stages['reconstruct'].status.value != 'succeeded' or not run.selected_dataset:
            raise RuntimeError('train --segment requires a successful schema 5/6 reconstruction')
        label = run.metrics['selected_attempt']
        records = [json.loads(line) for line in (scene/run_id/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
        dataset = scene/run.selected_dataset
        package_name=segment if duration_seconds is None else f'{segment}-first-{duration_seconds:g}s'
        package = scene/run_id/'training-data'/package_name
        if segment not in {s['id'] for s in json.loads((dataset/'segments.json').read_text(encoding='utf-8'))['segments']}:
            raise ValueError('Unknown segment ID')
        prepare_segment(dataset,records,run.config.reconstruction.primary,run.config.segment_qa,segment,package,duration_seconds=duration_seconds, **({"reuse_files":True} if run.config.schema_version == 6 else {}))
        experiment = dict(segment=segment,backend=backend,output=str(output.resolve()),status='preparing')
        run.metrics.setdefault('training_experiments',[]).append(experiment)
        save_run(scene,run)
        try:
            result = train_package(package,output,backend,steps,photo_comp,resume,dry_run)
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
    apply: Annotated[bool, typer.Option(help="Execute verified schema 6 cleanup; default is preview")] = False,
) -> None:
    """Current workflow: preview/apply schema 6 retention cleanup for a Run."""
    from .retention import cleanup_run
    try:
        scene = _scene(location_id, scene_id)
        console.print_json(json.dumps(cleanup_run(scene, load_run(scene, run_id), apply=apply)))
    except Exception as error:
        _fatal(error)


if __name__ == "__main__":
    app()
