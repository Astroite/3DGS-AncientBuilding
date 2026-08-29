from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .catalog import build_catalog
from .doctor import run_doctor
from .manifests import load_model, save_yaml
from .models import (
    CaptureManifest,
    LocationManifest,
    Region,
    Rights,
    RunConfig,
    SceneManifest,
)
from .paths import find_project_root, host_path, location_dir, scene_dir
from .pipeline import (
    CullSettings,
    export_run,
    ingest_capture,
    mask_run,
    preprocess_run,
    reconstruct_run,
    review_run,
    train_run,
    write_qa_report,
)
from .runs import create_run, load_run


app = typer.Typer(no_args_is_help=True, help="360 video to Gaussian Splatting workflow and catalog")
location_app = typer.Typer(no_args_is_help=True, help="Manage location manifests")
scene_app = typer.Typer(no_args_is_help=True, help="Manage scene manifests")
catalog_app = typer.Typer(no_args_is_help=True, help="Build and inspect the SQLite catalog")
qa_app = typer.Typer(no_args_is_help=True, help="Generate and review QA reports")
app.add_typer(location_app, name="location")
app.add_typer(scene_app, name="scene")
app.add_typer(catalog_app, name="catalog")
app.add_typer(qa_app, name="qa")
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
) -> None:
    """Verify project write access, command-line tools, CUDA, and gsplat."""
    checks = run_doctor(_root(), minimum_free_gib=minimum_free_gib)
    table = Table("Check", "Result", "Detail")
    for name, result in checks.items():
        if name == "ok":
            continue
        ok, detail = result
        table.add_row(name, "PASS" if ok else "FAIL", str(detail), style=None if ok else "red")
    console.print(table)
    if not checks["ok"]:
        raise typer.Exit(1)


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
        capture_manifest = load_model(path / "captures" / f"{capture_id}.yaml", CaptureManifest)
        input_path = (
            host_path(stitched)
            if stitched is not None
            else path / capture_manifest.stitched_video.relative_path
        )
        if not input_path.is_absolute():
            input_path = root / input_path
        result = ingest_capture(root, path, capture_id, input_path, resume=resume)
        console.print(
            f"Ingested [green]{result.stitched_video.width}x{result.stitched_video.height}[/green] "
            f"{result.stitched_video.duration_seconds:.2f}s; sha256={result.stitched_video.sha256}"
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
            capture = load_model(path / "captures" / f"{capture_id}.yaml", CaptureManifest)
            if not capture.stitched_video.sha256:
                raise RuntimeError("Capture has not been ingested; run gsdb ingest first")
            config = RunConfig(
                capture_id=capture_id,
                input_sha256=capture.stitched_video.sha256,
                preprocess={"target_frames": target_frames},
                masking={
                    "device": mask_device,
                    "score_threshold": mask_score_threshold,
                    "probability_threshold": mask_probability_threshold,
                    "inference_gamma": mask_gamma,
                    "dilation_pixels": mask_dilation_pixels,
                    "max_masked_fraction": max_masked_fraction,
                    "qa_sample_count": mask_qa_sample_count,
                },
                vision_qa={"enabled": vision_qa},
                reconstruction={
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
                },
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
    """Run masked COLMAP and apply the single bounded fallback when required."""
    try:
        path = _scene(location_id, scene_id)
        run = reconstruct_run(path, load_run(path, run_id), resume=resume)
        console.print(f"Run [green]{run.id}[/green]: selected={run.selected_dataset}")
    except Exception as error:
        _fatal(error)


@app.command()
def train(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train Splatfacto with one OOM downscale retry."""
    try:
        path = _scene(location_id, scene_id)
        run = train_run(path, load_run(path, run_id), resume=resume)
        console.print(f"Run [green]{run.id}[/green]: train={run.stages['train'].status.value}")
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
