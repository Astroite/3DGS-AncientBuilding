from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .doctor import collect_tool_versions
from .manifests import load_capture_manifest, load_model, save_yaml
from .masking import (
    create_mask_contact_sheets,
    generate_person_masks,
    image_files,
    summarize_detection_records,
    validate_mask_set,
)
from .mask_finalize import (
    MaskFinalizationMissingError,
    expected_reconstruction_images,
    validate_mask_finalization,
)
from .media import (
    analyze_frames,
    check_disk_budget,
    create_blur_aware_subset,
    extract_uniform_frames,
    probe_video,
    sha256_file,
    select_blur_aware_records,
    summarize_frame_metrics,
    validate_equirectangular,
)
from .models import (
    ArtifactManifest,
    ArtifactRecord,
    CaptureManifest,
    CaptureManifestV2,
    ReconstructionAttempt,
    RunManifest,
    RunStatus,
    SceneManifest,
    utc_now,
)
from .frames import (
    applied_transform,
    camera_positions,
    check_pose_normalisation,
    load_dataparser_transform,
    rig_gravity,
    to_model_frame,
    trajectory_frame,
    upright_rotation,
    write_published_transforms,
)
from .paths import ensure_within, ensure_work_dir, host_path
from .ply import CullSpec, gaussian_count, publish_gaussian_ply, rotate_gaussian_ply_y_up
from .postshot import prepare_postshot_dataset, train_postshot
from .processes import CommandError, run_logged
from .reconstruction import (
    _selected_model_dir,
    build_image_pyramid,
    center_spread_metrics,
    projection_fov_degrees,
    project_equirectangular_frames,
)
from .reconstruction_realityscan import run_realityscan_alignment
from .runs import begin_stage, complete_stage, fail_stage, save_run
from .sources import copy_candidate_frames, validate_source_fingerprints
from .vision_qa import load_local_mask_qa_review, run_deepseek_mask_qa
from .trajectory import (
    timestamps_from_metrics,
    validate_trajectory_qa,
    write_trajectory_qa,
)


VERSION_PATTERN = re.compile(r"^v\d{3}$")

# A walking capture holds a near-constant camera height, so a large spread along
# the published vertical means the axis is wrong rather than the path steep.
PUBLISHED_HEIGHT_SPAN_LIMIT = 0.10
UNSAFE_PUBLISH_MARKER = "UNSAFE-PUBLISH-FRAME.txt"


# Splatfacto's densification schedule is written in steps, but what actually
# matters is how many times each training view is sampled before splitting stops.
# Nerfstudio's defaults were tuned on datasets around this size, and the validated
# 90-second run happens to sit exactly here (135 frames x 8 views), so the defaults
# are reproduced unchanged at this count and scaled from it.
DENSIFICATION_BASELINE_IMAGES = 1080
DENSIFICATION_BASELINE_STEPS = {
    "warmup_length": 500,
    "stop_screen_size_at": 4000,
    "stop_split_at": 15000,
}
# Densification must always leave a refinement tail, however large the dataset.
DENSIFICATION_MAX_FRACTION = 0.75

# gsplat's DefaultStrategy (wired up in nerfstudio's SplatfactoModel) pauses ALL
# refinement -- both growth and opacity-based pruning -- for
# ``num_train_data + refine_every`` steps after every opacity reset, and resets fire
# every ``reset_alpha_every * refine_every`` steps. On the Yunxiu run (5,128 images)
# that pause (~5,228 steps) outlasted the reset interval at nerfstudio's default
# reset_alpha_every=30 (3,000 steps), so each reset re-armed the pause before the
# last one expired and refinement locked up for good around step 3,000: the model
# trained to completion with 1,363,440 gaussians frozen since shortly after warmup,
# 98.6% of them decayed to near-zero opacity and never pruned because pruning was
# paused too, and the exporter's own opacity filter -- not gsdb's culling -- is what
# actually dropped them, down to 19,705. Splatfacto's default refine_every (100) is
# pinned explicitly here because reset_alpha_every is derived assuming this exact
# value; if nerfstudio ever changes its own default, this schedule must not silently
# drift out of sync with it.
SPLATFACTO_REFINE_EVERY = 100
SPLATFACTO_DEFAULT_RESET_ALPHA_EVERY = 30


def densification_schedule(image_count: int, max_iterations: int) -> dict[str, int]:
    """Step bounds that keep densification coverage constant as a scene grows.

    A 427-second capture is 5,128 perspective views against the baseline's 1,080.
    Left at the default 15,000 steps, splitting would stop after each view had been
    sampled 2.8 times instead of 13.4, so most of the scene would never accumulate
    enough positional gradient to densify at all -- the geometry would not be
    under-refined, it would be missing. Scaling the schedule with the view count
    keeps the per-view budget that the defaults assume.
    """
    if image_count < 1:
        raise ValueError(f"Image count must be positive, got {image_count}")
    ratio = image_count / DENSIFICATION_BASELINE_IMAGES
    schedule = {
        name: max(1, round(steps * ratio))
        for name, steps in DENSIFICATION_BASELINE_STEPS.items()
    }
    schedule["stop_split_at"] = min(
        schedule["stop_split_at"], max(1, int(max_iterations * DENSIFICATION_MAX_FRACTION))
    )
    schedule["stop_screen_size_at"] = min(
        schedule["stop_screen_size_at"], max(1, schedule["stop_split_at"] - 1)
    )
    schedule["warmup_length"] = min(
        schedule["warmup_length"], max(1, schedule["stop_screen_size_at"] - 1)
    )
    # image_count over-estimates nerfstudio's actual num_train_data (it applies its
    # own train/eval split on top of this), which only makes the margin below safer.
    minimum_reset_alpha_every = (
        math.ceil((image_count + SPLATFACTO_REFINE_EVERY) / SPLATFACTO_REFINE_EVERY) + 1
    )
    schedule["reset_alpha_every"] = max(
        SPLATFACTO_DEFAULT_RESET_ALPHA_EVERY, minimum_reset_alpha_every
    )
    schedule["refine_every"] = SPLATFACTO_REFINE_EVERY
    return schedule


def training_image_count(run: RunManifest) -> int:
    """Views the training set will hold, from configuration alone.

    Deliberately not the registered-image count: that depends on how
    reconstruction went, and the schedule must stay a pure function of the hashed
    configuration so one config hash still means one model.
    """
    attempt = getattr(
        run.config.reconstruction, "fallback" if run.fallback_attempted else "primary"
    )
    return int(attempt.frame_count) * int(attempt.images_per_equirect)


@dataclass(frozen=True)
class TrainSettings:
    """Instrumentation around training that does not change what is learned.

    Checkpoint frequency and logging backend affect neither the loss nor the
    Gaussians, so like culling they stay out of the run's config hash — otherwise
    turning on a metric would invalidate every manifest already on disk.
    """

    # Mirrors nerfstudio 1.1.5's TrainerConfig.vis literals, so a typo fails here
    # rather than four hours into a run.
    VIS_CHOICES = ("viewer", "wandb", "tensorboard", "comet", "viewer+tensorboard", "viewer+wandb")

    steps_per_save: int = 10_000
    vis: str = "tensorboard"

    def __post_init__(self) -> None:
        if self.vis not in self.VIS_CHOICES:
            raise ValueError(f"vis must be one of {self.VIS_CHOICES}, got {self.vis!r}")
        if self.steps_per_save < 1:
            raise ValueError(f"steps_per_save must be positive, got {self.steps_per_save}")

    def command_arguments(self) -> list[str]:
        return [
            # Densification stops at splatfacto's stop_split_at (15k), so the rest of
            # a 100k run only refines a fixed Gaussian set. Keeping every checkpoint
            # lets one run answer "where does quality stop improving?" by exporting
            # from several steps, instead of retraining once per candidate.
            "--save-only-latest-checkpoint",
            "False",
            "--steps-per-save",
            str(self.steps_per_save),
            # An unattended batch run has nobody watching a live viewer, and
            # tensorboard leaves the eval curve on disk for that decision.
            "--vis",
            self.vis,
        ]

    def as_metrics(self) -> dict[str, Any]:
        return {"steps_per_save": self.steps_per_save, "vis": self.vis}


@dataclass(frozen=True)
class CullSettings:
    """Publish-time culling knobs.

    Deliberately not part of RunConfig: culling changes what is published, not
    what was trained, so adjusting it must not invalidate a run's config hash and
    force a four-hour retrain.
    """

    enabled: bool = True
    distance_factor: float = 3.0
    scale_factor: float = 1.0
    max_removed_fraction: float = 0.05

    def as_metrics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "distance_factor": self.distance_factor,
            "scale_factor": self.scale_factor,
            "max_removed_fraction": self.max_removed_fraction,
        }


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _record_resources(
    run: RunManifest,
    work: Path,
    command_metrics: dict[str, float] | None = None,
    extra_path: Path | None = None,
) -> None:
    resources = run.metrics.setdefault("resources", {})
    observed = _tree_size(work) + (_tree_size(extra_path) if extra_path else 0)
    resources["disk_observed_peak_bytes"] = max(
        int(resources.get("disk_observed_peak_bytes", 0)), observed
    )
    if command_metrics and "gpu_memory_peak_mib" in command_metrics:
        resources["gpu_memory_observed_peak_mib"] = max(
            float(resources.get("gpu_memory_observed_peak_mib", 0)),
            command_metrics["gpu_memory_peak_mib"],
        )


def capture_path(scene_path: Path, capture_id: str) -> Path:
    return scene_path / "captures" / f"{capture_id}.yaml"


def load_capture(
    scene_path: Path, capture_id: str
) -> CaptureManifest | CaptureManifestV2:
    return load_capture_manifest(capture_path(scene_path, capture_id))


def ingest_capture(
    project_root: Path,
    scene_path: Path,
    capture_id: str,
    stitched_path: Path | None,
    resume: bool = False,
) -> CaptureManifest | CaptureManifestV2:
    capture = load_capture(scene_path, capture_id)
    if isinstance(capture, CaptureManifestV2):
        from .sources import probe_capture_source

        if stitched_path is not None:
            raise ValueError(
                "Schema 2 sources are immutable and already explicit; omit --stitched"
            )
        probe = probe_capture_source(
            capture, scene_path / "inputs" / "prepared" / ".protocol"
        )
        capture.source.probe = type(capture.source.probe)(
            **{
                name: value
                for name, value in probe.items()
                if name in type(capture.source.probe).model_fields
            }
        )
        save_yaml(capture_path(scene_path, capture_id), capture)
        return capture
    if stitched_path is None:
        raise ValueError("A stitched video path is required for a schema 1 capture")
    stitched = ensure_within(stitched_path, project_root)
    if not stitched.is_file():
        raise FileNotFoundError(
            f"Stitched video is missing: {stitched}. "
            "Export a standard 2:1 equirectangular MP4 from Insta360 Studio first."
        )
    raw = host_path(capture.raw_source.windows_path)
    if not raw.is_file():
        raise FileNotFoundError(f"External raw source is missing: {raw}")
    raw_sha256 = sha256_file(raw)
    if capture.raw_source.sha256 and capture.raw_source.sha256 != raw_sha256:
        raise RuntimeError("External raw source hash changed; refusing to ingest a different capture")
    capture.raw_source.sha256 = raw_sha256
    capture.raw_source.byte_size = raw.stat().st_size
    probe = probe_video(stitched)
    validate_equirectangular(probe)
    stitched_sha256 = sha256_file(stitched)
    if capture.stitched_video.sha256:
        if capture.stitched_video.sha256 != stitched_sha256:
            raise RuntimeError(
                "Stitched video hash changed after ingest; create a new capture manifest "
                "instead of silently replacing lineage"
            )
        if not resume:
            raise RuntimeError("Capture was already ingested; use --resume to revalidate it")
    if capture.selection.end_seconds > probe["duration_seconds"] + 0.5:
        raise ValueError(
            f"Selected range ends at {capture.selection.end_seconds:.2f}s but the stitched video "
            f"is only {probe['duration_seconds']:.2f}s"
        )
    relative_to_scene = stitched.relative_to(scene_path).as_posix()
    capture.stitched_video.relative_path = relative_to_scene
    capture.stitched_video.sha256 = stitched_sha256
    capture.stitched_video.byte_size = probe["byte_size"]
    capture.stitched_video.width = probe["width"]
    capture.stitched_video.height = probe["height"]
    capture.stitched_video.fps = probe["fps"]
    capture.stitched_video.duration_seconds = probe["duration_seconds"]
    capture.stitched_video.codec = probe["codec"]
    save_yaml(capture_path(scene_path, capture_id), capture)
    return capture


def preprocess_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "preprocess", resume=resume):
        return run
    save_run(scene_path, run)
    work = ensure_work_dir(scene_path, run.id)
    log_path = work / "logs" / "preprocess.log"
    try:
        if not run.tool_versions:
            run.tool_versions = collect_tool_versions()
        capture = load_capture(scene_path, run.config.capture_id)
        schema_version = int(getattr(run.config, "schema_version", 1))
        modern = schema_version >= 2
        candidates_dir = work / (
            "equirect-candidates" if modern else "equirect-primary"
        )
        explicit_timestamps: tuple[float, ...] | None = None
        if schema_version == 3:
            if not isinstance(capture, CaptureManifestV2):
                raise RuntimeError("RunConfigV3 requires a schema 2 capture")
            from .sources import _dataset_from_manifest

            prepared = ensure_within(
                scene_path / run.config.prepared_relative_path, scene_path
            )
            candidate_set = _dataset_from_manifest(prepared)
            if candidate_set.dataset_sha256 != run.config.input_dataset_sha256:
                raise RuntimeError("Prepared input hash changed after the run was created")
            frames = copy_candidate_frames(candidate_set, candidates_dir, resume=resume)
            explicit_timestamps = candidate_set.timestamps_seconds
            probe = capture.source.probe.model_dump(mode="json")
            selection_duration = (
                capture.selection.end_seconds - capture.selection.start_seconds
            )
            budget = check_disk_budget(
                work,
                sum(item.byte_size for item in capture.source.files),
                run.config.preprocess.minimum_free_gib,
            )
        else:
            if not isinstance(capture, CaptureManifest):
                raise RuntimeError("Legacy RunConfig requires a schema 1 capture")
            if capture.stitched_video.sha256 != run.config.input_sha256:
                raise RuntimeError("Capture input hash changed after the run was created")
            video = ensure_within(
                scene_path / capture.stitched_video.relative_path, scene_path
            )
            if not video.is_file():
                raise FileNotFoundError(
                    f"Stitched video is missing: {video}. Export a 2:1 MP4 from the raw 360 media first."
                )
            probe = probe_video(video)
            validate_equirectangular(probe)
            selection_duration = (
                capture.selection.end_seconds - capture.selection.start_seconds
            )
            budget = check_disk_budget(
                work,
                video.stat().st_size,
                run.config.preprocess.minimum_free_gib,
            )
            frames = extract_uniform_frames(
                video,
                candidates_dir,
                run.config.preprocess.target_frames,
                selection_duration,
                run.config.preprocess.jpeg_quality,
                log_path,
                start_seconds=capture.selection.start_seconds,
            )
        metrics_path = work / "frame-metrics.jsonl"
        records = analyze_frames(
            frames,
            selection_duration,
            metrics_path,
            start_seconds=capture.selection.start_seconds,
            timestamps_seconds=explicit_timestamps,
            selection_algorithm=(
                "composite_v1" if schema_version == 3 else "legacy_laplacian"
            ),
        )
        run.metrics["input"] = probe
        run.metrics["selection"] = capture.selection.model_dump(mode="json")
        run.metrics["disk_budget"] = budget
        candidate_summary = summarize_frame_metrics(records)
        if modern:
            selection_metric = (
                "selection_score"
                if schema_version == 3
                else "blur_laplacian_variance"
            )
            primary_count = run.config.reconstruction.primary.frame_count
            primary_dir = work / "equirect-primary"
            selected_frames = create_blur_aware_subset(
                candidates_dir,
                records,
                primary_dir,
                primary_count,
                selection_metric=selection_metric,
            )
            selected_records = select_blur_aware_records(
                records, primary_count, selection_metric=selection_metric
            )
            (work / "selected-primary-metrics.jsonl").write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n"
                    for item in selected_records
                ),
                encoding="utf-8",
            )
            selected_summary = summarize_frame_metrics(selected_records)
            candidate_blur = float(candidate_summary["blur_median"])
            selected_blur = float(selected_summary["blur_median"])
            run.metrics["preprocess"] = {
                "candidate": candidate_summary,
                "selected_primary": selected_summary,
                "candidate_frame_count": len(frames),
                "selected_frame_count": len(selected_frames),
                "blur_median_improvement": selected_blur - candidate_blur,
                "blur_median_improvement_ratio": (
                    selected_blur / candidate_blur if candidate_blur else 0.0
                ),
            }
        else:
            run.metrics["preprocess"] = candidate_summary
        _record_resources(run, work)
        complete_stage(
            run,
            "preprocess",
            message=(
                f"Extracted and analyzed {len(frames)} candidate equirectangular frames"
                + (
                    f"; selected {run.config.reconstruction.primary.frame_count} blur-aware frames"
                    if modern
                    else ""
                )
            ),
            log_path=log_path.relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        fail_stage(run, "preprocess", str(error), log_path.relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def _cached_masking_result(
    scene_path: Path,
    dataset: Path,
    attempt: ReconstructionAttempt,
    run: RunManifest,
    label: str,
) -> dict[str, Any] | None:
    cached = run.metrics.get("masking", {}).get(label)
    if not isinstance(cached, dict):
        return None
    expected_count = attempt.frame_count * attempt.images_per_equirect
    if int(cached.get("planar_images", 0)) != expected_count:
        return None
    images_dir = dataset / "images"
    masks_dir = dataset / "masks"
    images = image_files(images_dir)
    image_names = {path.relative_to(images_dir).as_posix() for path in images}
    if len(image_names) != expected_count:
        return None
    expected_masks = {
        (Path(name).parent / f"{Path(name).name}.png").as_posix()
        for name in image_names
    }
    actual_masks = {
        path.relative_to(masks_dir).as_posix() for path in image_files(masks_dir)
    }
    if actual_masks != expected_masks:
        return None
    for level in range(1, attempt.num_downscales + 1):
        factor = 2**level
        image_pyramid = dataset / f"images_{factor}"
        mask_pyramid = dataset / f"masks_{factor}"
        if {
            path.relative_to(image_pyramid).as_posix()
            for path in image_files(image_pyramid)
        } != image_names:
            return None
        if {
            path.relative_to(mask_pyramid).as_posix()
            for path in image_files(mask_pyramid)
        } != expected_masks:
            return None
    sheets = [scene_path / str(path) for path in cached.get("contact_sheets", [])]
    if not sheets or not all(path.is_file() for path in sheets):
        return None
    if run.config.vision_qa.enabled:
        local_review_path = dataset / "mask-qa" / "codex-local-review.json"
        if not local_review_path.is_file():
            return None
        _, verdict = load_local_mask_qa_review(
            local_review_path, sheets, run.config.vision_qa
        )
        if verdict.decision != "pass":
            return None
    return dict(cached)


def _prepare_masked_dataset(
    scene_path: Path,
    work: Path,
    source: Path,
    dataset: Path,
    attempt: ReconstructionAttempt,
    run: RunManifest,
    label: str,
) -> dict[str, Any]:
    cached = _cached_masking_result(scene_path, dataset, attempt, run, label)
    if cached is not None:
        print(f"Masked dataset: {label} (validated cached result)", flush=True)
        return cached
    images = project_equirectangular_frames(source, dataset, attempt)
    build_image_pyramid(
        dataset / "images", dataset, "images", attempt.num_downscales, is_mask=False
    )
    records = generate_person_masks(
        dataset / "images",
        dataset / "masks",
        run.config.masking,
        work / f"mask-metrics-{label}.jsonl",
    )
    deterministic = validate_mask_set(
        dataset / "images",
        dataset / "masks",
        run.config.masking.max_masked_fraction,
    )
    build_image_pyramid(
        dataset / "masks", dataset, "masks", attempt.num_downscales, is_mask=True
    )
    sheets = create_mask_contact_sheets(
        dataset / "images",
        dataset / "masks",
        records,
        dataset / "mask-qa",
        run.config.masking.qa_sample_count,
        run.config.vision_qa.max_contact_sheets,
    )
    vision: dict[str, Any]
    if run.config.vision_qa.enabled:
        local_review_path = dataset / "mask-qa" / "codex-local-review.json"
        if local_review_path.is_file():
            local_review, verdict = load_local_mask_qa_review(
                local_review_path, sheets, run.config.vision_qa
            )
            vision = {
                "provider": local_review.reviewer,
                "review_file": local_review_path.relative_to(scene_path).as_posix(),
                "contact_sheet_sha256": local_review.contact_sheet_sha256,
                **verdict.model_dump(mode="json"),
            }
        else:
            verdict = run_deepseek_mask_qa(sheets, run.config.vision_qa)
            vision = {"provider": "deepseek", **verdict.model_dump(mode="json")}
        if verdict.decision != "pass":
            raise RuntimeError(
                f"LLM mask QA rejected {label}: {verdict.rationale}; "
                f"false negatives={verdict.false_negative_views}"
            )
    else:
        vision = {
            "status": "disabled",
            "reason": (
                "Set vision_qa.enabled=true in a new run only after confirming an authorized "
                "DeepSeek multimodal endpoint"
            ),
        }
    return {
        "planar_images": len(images),
        "model": run.config.masking.model if run.config.masking.enabled else "disabled-all-white",
        "detection": summarize_detection_records(records),
        "validation": deterministic,
        "contact_sheets": [path.relative_to(scene_path).as_posix() for path in sheets],
        "vision_qa": vision,
    }


def mask_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "mask", resume=resume):
        return run
    save_run(scene_path, run)
    work = ensure_work_dir(scene_path, run.id)
    log_path = work / "logs" / "mask.log"
    try:
        primary = _prepare_masked_dataset(
            scene_path,
            work,
            work / "equirect-primary",
            work / "reconstruction-primary",
            run.config.reconstruction.primary,
            run,
            "primary",
        )
        run.metrics.setdefault("masking", {})["primary"] = primary
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(primary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _record_resources(run, work)
        complete_stage(
            run,
            "mask",
            message=(
                f"Projected and masked {primary['planar_images']} primary perspective views; "
                f"deterministic QA passed"
            ),
            log_path=log_path.relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        fail_stage(run, "mask", str(error), log_path.relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def reconstruct_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "reconstruct", resume=resume):
        return run
    save_run(scene_path, run)
    work = ensure_work_dir(scene_path, run.id)
    metrics_records = [
        json.loads(line)
        for line in (work / "frame-metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    attempts: dict[str, Any] = {}
    try:
        primary_output = work / "reconstruction-primary"
        if getattr(run.config, "schema_version", 1) == 3:
            primary_attempt = run.config.reconstruction.primary
            validate_mask_finalization(
                primary_output,
                expected_reconstruction_images(
                    primary_attempt.frame_count,
                    primary_attempt.images_per_equirect,
                ),
            )
        primary_metrics = run_realityscan_alignment(
            primary_output,
            run.config.reconstruction.primary,
            work / "logs" / "reconstruct-primary",
            run.config.reconstruction,
        )
        attempts["primary"] = primary_metrics
        threshold = run.config.reconstruction.registration_threshold
        selected = primary_output

        if min(
            primary_metrics["registration_ratio"],
            primary_metrics["largest_component_coverage"],
        ) < threshold:
            run.fallback_attempted = True
            fallback_frames = work / "equirect-fallback"
            create_blur_aware_subset(
                work
                / (
                    "equirect-candidates"
                    if getattr(run.config, "schema_version", 1) >= 2
                    else "equirect-primary"
                ),
                metrics_records,
                fallback_frames,
                run.config.reconstruction.fallback.frame_count,
                selection_metric=(
                    "selection_score"
                    if getattr(run.config, "schema_version", 1) == 3
                    else "blur_laplacian_variance"
                ),
            )
            fallback_records = select_blur_aware_records(
                metrics_records,
                run.config.reconstruction.fallback.frame_count,
                selection_metric=(
                    "selection_score"
                    if getattr(run.config, "schema_version", 1) == 3
                    else "blur_laplacian_variance"
                ),
            )
            (work / "selected-fallback-metrics.jsonl").write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n"
                    for item in fallback_records
                ),
                encoding="utf-8",
            )
            fallback_output = work / "reconstruction-fallback"
            fallback_masking = _prepare_masked_dataset(
                scene_path,
                work,
                fallback_frames,
                fallback_output,
                run.config.reconstruction.fallback,
                run,
                "fallback",
            )
            run.metrics.setdefault("masking", {})["fallback"] = fallback_masking
            if getattr(run.config, "schema_version", 1) == 3:
                try:
                    fallback_attempt = run.config.reconstruction.fallback
                    validate_mask_finalization(
                        fallback_output,
                        expected_reconstruction_images(
                            fallback_attempt.frame_count,
                            fallback_attempt.images_per_equirect,
                        ),
                    )
                except MaskFinalizationMissingError:
                    run.metrics["reconstruction"] = attempts
                    run.status = RunStatus.WAITING_REVIEW
                    run.active_stage = "reconstruct"
                    run.stages["reconstruct"].message = (
                        "Fallback masks are ready; run mask-review and mask-finalize "
                        "for --attempt fallback, then reconstruct --resume"
                    )
                    _record_resources(run, work)
                    save_run(scene_path, run)
                    return run
            fallback_metrics = run_realityscan_alignment(
                fallback_output,
                run.config.reconstruction.fallback,
                work / "logs" / "reconstruct-fallback",
                run.config.reconstruction,
            )
            attempts["fallback"] = fallback_metrics
            selected = fallback_output
            if min(
                fallback_metrics["registration_ratio"],
                fallback_metrics["largest_component_coverage"],
            ) < threshold:
                raise RuntimeError(
                    "Both reconstruction attempts were below the 70% registration/coverage threshold; "
                    "stop and prepare a recapture report"
                )

        if getattr(run.config, "schema_version", 1) == 3:
            selected_label = (
                "fallback" if selected.name.endswith("fallback") else "primary"
            )
            selected_attempt = getattr(
                run.config.reconstruction, selected_label
            )
            selected_metrics_path = work / f"selected-{selected_label}-metrics.jsonl"
            timestamps_by_frame = timestamps_from_metrics(selected_metrics_path)
            expected_frames = range(1, selected_attempt.frame_count + 1)
            trajectory = write_trajectory_qa(
                selected,
                timestamps_by_frame=timestamps_by_frame,
                expected_frames=expected_frames,
            )
            run.metrics["trajectory_qa"] = {
                **trajectory,
                "report": (selected / "trajectory-qa.json")
                .relative_to(scene_path)
                .as_posix(),
                "top_view": (selected / "trajectory-top.svg")
                .relative_to(scene_path)
                .as_posix(),
            }
            if trajectory["blocking"]:
                raise RuntimeError(
                    f"Trajectory QA found {len(trajectory['blocking'])} blocking issue(s)"
                )
        run.metrics["reconstruction"] = attempts
        run.selected_dataset = selected.relative_to(scene_path).as_posix()
        _record_resources(run, work)
        complete_stage(
            run,
            "reconstruct",
            message=f"Selected dataset: {run.selected_dataset}",
            log_path=(work / "logs").relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        run.metrics["reconstruction"] = attempts
        fail_stage(run, "reconstruct", str(error), (work / "logs").relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def _training_checkpoint_steps(config: Path) -> list[int]:
    return sorted(
        int(match.group(1))
        for path in (config.parent / "nerfstudio_models").glob("step-*.ckpt")
        if (match := re.search(r"step-(\d+)\.ckpt$", path.name))
    )


def _find_training_config(output: Path) -> Path | None:
    configs = sorted(output.rglob("config.yml"), key=lambda item: item.stat().st_mtime)
    return configs[-1] if configs else None


def _find_completed_training_config(output: Path, max_iterations: int) -> Path | None:
    """Newest config whose final checkpoint proves training actually completed.

    Nerfstudio writes ``config.yml`` before it loads the first image. Treating that
    file alone as completion made ``--resume`` accept a run that had failed while
    filling the CPU image cache. The trainer's final checkpoint is written at
    ``max_iterations - 1`` (for example step 29,999 on a 30,000-step run), so it is
    the durable completion marker we need here.
    """
    final_step = max_iterations - 1
    configs = sorted(
        output.rglob("config.yml"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    return next(
        (config for config in configs if final_step in _training_checkpoint_steps(config)),
        None,
    )


def _log_reports_memory_exhaustion(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    content = log_path.read_text(encoding="utf-8", errors="replace").lower()
    return any(
        marker in content
        for marker in (
            "out of memory",
            "cannot allocate memory",
            "errno 12",
            "std::bad_alloc",
        )
    )


def _log_reports_nested_downscale_path_failure(log_path: Path) -> bool:
    """Nerfstudio 1.1.5 discarded view subdirectories for downscaled images."""
    if not log_path.is_file():
        return False
    content = log_path.read_text(encoding="utf-8", errors="replace").lower()
    return "filenotfounderror" in content and re.search(
        r"images_\d+[/\\]frame_\d+\.(?:jpg|jpeg|png)", content
    ) is not None


def _train_command(
    run: RunManifest,
    dataset: Path,
    output: Path,
    downscale: int | None = None,
    settings: TrainSettings | None = None,
) -> list[str]:
    settings = settings or TrainSettings()
    if getattr(run.config, "schema_version", 1) >= 2:
        effective_downscale = downscale or run.config.train.downscale_factor
        return [
            "ns-train",
            run.config.train.method,
            "--output-dir",
            str(output),
            "--max-num-iterations",
            str(run.config.train.max_iterations),
            "--viewer.quit-on-train-completion",
            "True",
            *settings.command_arguments(),
            *(
                argument
                for name, value in densification_schedule(
                    training_image_count(run), run.config.train.max_iterations
                ).items()
                for argument in (f"--pipeline.model.{name.replace('_', '-')}", str(value))
            ),
            "--pipeline.datamanager.cache-images",
            run.config.train.cache_images,
            "--pipeline.datamanager.cache-images-type",
            run.config.train.cache_images_type,
            "--pipeline.model.use-scale-regularization",
            str(run.config.train.use_scale_regularization),
            "--pipeline.model.rasterize-mode",
            run.config.train.rasterize_mode,
            "--pipeline.model.use-bilateral-grid",
            str(run.config.train.use_bilateral_grid),
            "--pipeline.model.camera-optimizer.mode",
            run.config.train.camera_optimizer_mode,
            "nerfstudio-data",
            "--data",
            str(dataset),
            "--downscale-factor",
            str(effective_downscale),
        ]
    command = [
        "ns-train",
        run.config.train.method,
        "--output-dir",
        str(output),
        "--max-num-iterations",
        str(run.config.train.max_iterations),
        "--viewer.quit-on-train-completion",
        "True",
        "nerfstudio-data",
        "--data",
        str(dataset),
    ]
    if downscale is not None:
        command.extend(["--downscale-factor", str(downscale)])
    return command


def postshot_prepare_run(
    scene_path: Path,
    run: RunManifest,
    resume: bool = False,
    output: Path | None = None,
) -> RunManifest:
    if not begin_stage(run, "postshot_prepare", resume=resume):
        return run
    save_run(scene_path, run)
    work = ensure_work_dir(scene_path, run.id)
    log_path = work / "logs" / "postshot-prepare.log"
    try:
        result = prepare_postshot_dataset(scene_path, run, output=output, resume=resume)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "output_path": result["output_path"],
                    "counts": result["counts"],
                    "reused": result["reused"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        complete_stage(
            run,
            "postshot_prepare",
            message=(
                f"Postshot dataset ready at {result['output_path']}; "
                f"images={result['counts']['images']}; cameras={result['counts']['cameras']}; "
                f"reused={result['reused']}"
            ),
            log_path=log_path.relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        fail_stage(
            run, "postshot_prepare", str(error), log_path.relative_to(scene_path).as_posix()
        )
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def postshot_train_run(
    scene_path: Path,
    run: RunManifest,
    *,
    resume: bool = False,
    dry_run: bool = False,
    allow_low_vram: bool = False,
    profile: str = "Splat3",
    ksteps: int | None = None,
    max_splats: int | None = None,
    gpu: int = 0,
    dataset: Path | None = None,
    output: Path | None = None,
    store_training_context: bool = True,
    export_ply: Path | None = None,
    export_spz: Path | None = None,
) -> tuple[RunManifest, dict[str, Any]]:
    """Postshot-backed equivalent of the retired Nerfstudio train_run().

    A dry run previews the training command without training anything, so it
    deliberately does not touch RunManifest.stages -- only a real run marks
    the "train" stage started/succeeded/failed.
    """
    if dry_run:
        result = train_postshot(
            scene_path,
            run,
            dry_run=True,
            allow_low_vram=allow_low_vram,
            profile=profile,
            ksteps=ksteps,
            max_splats=max_splats,
            gpu=gpu,
            dataset=dataset,
            output=output,
            store_training_context=store_training_context,
            export_ply=export_ply,
            export_spz=export_spz,
        )
        return run, result

    if not begin_stage(run, "train", resume=resume):
        return run, {"skipped": True}
    save_run(scene_path, run)
    work = ensure_work_dir(scene_path, run.id)
    log_path = work / "logs" / "postshot-train.log"
    try:
        result = train_postshot(
            scene_path,
            run,
            dry_run=False,
            allow_low_vram=allow_low_vram,
            profile=profile,
            ksteps=ksteps,
            max_splats=max_splats,
            gpu=gpu,
            dataset=dataset,
            output=output,
            store_training_context=store_training_context,
            export_ply=export_ply,
            export_spz=export_spz,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        complete_stage(
            run,
            "train",
            message=f"Postshot project: {result['output']}; sha256={result['output_sha256']}",
            log_path=log_path.relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        fail_stage(run, "train", str(error), log_path.relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run, result


def train_run(
    scene_path: Path,
    run: RunManifest,
    resume: bool = False,
    settings: TrainSettings | None = None,
) -> RunManifest:
    settings = settings or TrainSettings()
    if not begin_stage(run, "train", resume=resume):
        return run
    save_run(scene_path, run)
    assert run.selected_dataset is not None
    dataset = ensure_within(scene_path / run.selected_dataset, scene_path)
    work = ensure_work_dir(scene_path, run.id)
    training_root = work / "training"
    attempt_one = training_root / "attempt-1"
    log_one = work / "logs" / "train-attempt-1.log"
    attempt_two = training_root / "attempt-2-downscaled"
    log_two = work / "logs" / "train-attempt-2.log"
    path_fix_attempt = training_root / "attempt-2-downscaled-path-fix"
    path_fix_log = work / "logs" / "train-attempt-2-path-fix.log"
    max_iterations = run.config.train.max_iterations
    selected_config = _find_completed_training_config(path_fix_attempt, max_iterations)
    selected_config = selected_config or _find_completed_training_config(
        attempt_two, max_iterations
    )
    selected_config = selected_config or _find_completed_training_config(
        attempt_one, max_iterations
    )
    try:
        if getattr(run.config, "schema_version", 1) == 3:
            selected_label = "fallback" if run.fallback_attempted else "primary"
            selected_attempt = getattr(run.config.reconstruction, selected_label)
            validate_mask_finalization(
                dataset,
                expected_reconstruction_images(
                    selected_attempt.frame_count,
                    selected_attempt.images_per_equirect,
                ),
            )
            timestamps = timestamps_from_metrics(
                work / f"selected-{selected_label}-metrics.jsonl"
            )
            validate_trajectory_qa(
                dataset,
                timestamps,
                range(1, selected_attempt.frame_count + 1),
            )
        if selected_config is None:
            retry_for_memory = _log_reports_memory_exhaustion(log_one)
            incomplete_attempt_one = _find_training_config(attempt_one)
            incomplete_attempt_two = _find_training_config(attempt_two)
            incomplete_path_fix = _find_training_config(path_fix_attempt)
            retry_output = attempt_two
            retry_log = log_two
            retry_name = "attempt-2-downscaled"
            if incomplete_path_fix is not None:
                raise RuntimeError(
                    "The nested-path-fixed downscaled retry has an incomplete checkpoint "
                    "set; automatic resume is not safe"
                )
            if incomplete_attempt_two is not None:
                if _log_reports_nested_downscale_path_failure(log_two):
                    # Preserve the failed retry and its log. The environment patch now
                    # keeps view_XX in images_2/view_XX/frame.jpg, so a fresh timestamp
                    # in a distinct output root is deterministic and safe.
                    retry_output = path_fix_attempt
                    retry_log = path_fix_log
                    retry_name = "attempt-2-downscaled-path-fix"
                else:
                    raise RuntimeError(
                        "The downscaled training retry has an incomplete checkpoint set; "
                        "automatic resume is not safe"
                    )
            if incomplete_attempt_one is not None and not retry_for_memory:
                raise RuntimeError(
                    "Training has an incomplete checkpoint set and did not fail from "
                    "memory exhaustion; automatic resume is not safe"
                )
            if not retry_for_memory:
                try:
                    train_metrics = run_logged(
                        _train_command(run, dataset, attempt_one, settings=settings),
                        log_one,
                        monitor_gpu=True,
                    )
                    run.metrics.setdefault("train", {}).setdefault("attempts", []).append(
                        {"name": "attempt-1", **train_metrics}
                    )
                except CommandError as command_error:
                    run.metrics.setdefault("train", {}).setdefault("attempts", []).append(
                        {"name": "attempt-1", **command_error.metrics}
                    )
                    if not _log_reports_memory_exhaustion(log_one):
                        raise
                    retry_for_memory = True
                else:
                    selected_config = _find_completed_training_config(
                        attempt_one, max_iterations
                    )
            if retry_for_memory:
                retry_metrics = run_logged(
                    _train_command(
                        run,
                        dataset,
                        retry_output,
                        downscale=run.config.train.oom_retry_downscale,
                        settings=settings,
                    ),
                    retry_log,
                    monitor_gpu=True,
                )
                run.metrics.setdefault("train", {}).setdefault("attempts", []).append(
                    {"name": retry_name, **retry_metrics}
                )
                selected_config = _find_completed_training_config(
                    retry_output, max_iterations
                )
                run.metrics.setdefault("train", {})["oom_retry"] = True
        if selected_config is None:
            raise RuntimeError("Training finished without producing its final checkpoint")
        run.metrics.setdefault("train", {})["config_path"] = selected_config.relative_to(scene_path).as_posix()
        run.metrics["train"]["instrumentation"] = settings.as_metrics()
        run.metrics["train"]["densification"] = {
            "training_images": training_image_count(run),
            **densification_schedule(
                training_image_count(run), run.config.train.max_iterations
            ),
        }
        run.metrics["train"]["eval_curve"] = read_eval_curve(selected_config.parent)
        run.metrics["train"]["checkpoints"] = _training_checkpoint_steps(selected_config)
        attempts = run.metrics.get("train", {}).get("attempts", [])
        for item in attempts:
            _record_resources(run, work, item)
        complete_stage(
            run,
            "train",
            message=f"Training config: {selected_config.relative_to(scene_path).as_posix()}",
            log_path=(work / "logs").relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        fail_stage(run, "train", str(error), (work / "logs").relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def metrics_for_version(
    run: RunManifest, version: str, existing: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run metrics with the export block describing this version, not the newest one.

    One run can publish several versions, so stamping the run's current export
    metrics onto every artifact manifest would make an older version claim the
    Gaussian count and cull settings of a later one.
    """
    metrics = {key: value for key, value in run.metrics.items() if key != "export"}
    recorded = run.metrics.get("export", {}).get("versions", {}).get(version)
    if recorded is not None:
        metrics["export"] = recorded
    elif existing is not None and "export" in existing:
        # Published before per-version metrics existed: keep what it already says
        # rather than overwriting it with another version's numbers.
        metrics["export"] = existing["export"]
    return metrics


EVAL_SCALARS = ("psnr", "ssim", "lpips")


def read_eval_curve(training_dir: Path) -> list[dict[str, float]]:
    """Eval metrics against step, read back from the TensorBoard event files.

    Recorded so the iteration count can be argued from a quality plateau instead of
    guessed. Missing or unreadable events are not fatal — this is instrumentation,
    and a run must not fail because a curve could not be plotted.
    """
    events = sorted(training_dir.rglob("events.out.tfevents.*"))
    if not events:
        return []
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return []
    by_step: dict[int, dict[str, float]] = {}
    for path in events:
        try:
            accumulator = event_accumulator.EventAccumulator(
                str(path), size_guidance={event_accumulator.SCALARS: 0}
            )
            accumulator.Reload()
        except Exception:
            continue
        for tag in accumulator.Tags().get("scalars", []):
            name = tag.rsplit("/", 1)[-1].lower()
            if name not in EVAL_SCALARS or not tag.lower().startswith("eval"):
                continue
            for scalar in accumulator.Scalars(tag):
                by_step.setdefault(int(scalar.step), {})[name] = float(scalar.value)
    return [{"step": step, **values} for step, values in sorted(by_step.items())]


def _artifact_record(path: Path, scene_path: Path, kind: str, version: str) -> ArtifactRecord:
    return ArtifactRecord(
        kind=kind,
        relative_path=path.relative_to(scene_path).as_posix(),
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        version=version,
    )


def _preview_render_command(config: Path, preview: Path) -> list[str]:
    """Build a short preview command compatible with splatfacto's full-image datamanager."""
    return [
        "ns-render",
        "interpolate",
        "--load-config",
        str(config),
        "--output-path",
        str(preview),
        "--pose-source",
        "eval",
        "--interpolation-steps",
        "1",
        "--frame-rate",
        "72",
        "--downscale-factor",
        "4",
        "--rendered-output-names",
        "rgb",
    ]


def resolve_publish_frame(
    scene_path: Path,
    run: RunManifest,
    training_config: Path,
    allow_unsafe: bool = False,
) -> dict[str, Any]:
    """Work out the published orientation and the camera path, in the model's frame.

    Nerfstudio's dataparser re-orients the scene using the mean camera up vector.
    On a panoramic rig the eight views point in every direction, so that estimate
    does not recover gravity and a fixed axis swap cannot correct it. The rig
    itself does know: every perspective view was cut from a gravity-stabilised
    panorama at a known pitch, so the capture reports its own vertical.
    """
    dataset = ensure_within(scene_path / (run.selected_dataset or ""), scene_path)
    attempt = getattr(
        run.config.reconstruction, "fallback" if run.fallback_attempted else "primary"
    )
    rotation_matrix, translation, scale = load_dataparser_transform(training_config)
    up_colmap, gravity_metrics = rig_gravity(
        _selected_model_dir(dataset), attempt, allow_unsafe=allow_unsafe
    )
    safety_warnings = []
    if warning := gravity_metrics.get("deviation_warning"):
        safety_warnings.append(str(warning))

    transforms_path = dataset / "transforms.json"
    cameras_nerfstudio = camera_positions(transforms_path)
    cameras_model = to_model_frame(cameras_nerfstudio, rotation_matrix, translation, scale)
    gravity_metrics["pose_normalisation_peak"] = check_pose_normalisation(cameras_model)

    up_model = rotation_matrix @ (applied_transform(transforms_path) @ up_colmap)
    up_model = up_model / float(np.linalg.norm(up_model))
    trajectory = trajectory_frame(cameras_model, up_model)
    rotation = upright_rotation(up_model, trajectory["forward"])

    published_cameras = cameras_model @ rotation.T
    heights = published_cameras[:, 1]
    height_ratio = float((heights.max() - heights.min()) / (2.0 * trajectory["radius"]))
    height_gate_passed = height_ratio <= PUBLISHED_HEIGHT_SPAN_LIMIT
    if not height_gate_passed:
        warning = (
            f"Published cameras vary {height_ratio:.1%} in height across the capture, "
            f"above the {PUBLISHED_HEIGHT_SPAN_LIMIT:.0%} limit; the vertical axis is wrong"
        )
        if not allow_unsafe:
            raise RuntimeError(warning)
        safety_warnings.append(warning)
    return {
        "rotation": rotation,
        "cameras_model": cameras_model,
        "published_cameras": published_cameras,
        "trajectory_radius": trajectory["radius"],
        "dataparser": (rotation_matrix, translation, scale),
        "metrics": {
            **gravity_metrics,
            "up_in_model_frame": [float(value) for value in up_model],
            "published_height_span_ratio": height_ratio,
            "published_height_span_limit": PUBLISHED_HEIGHT_SPAN_LIMIT,
            "published_height_gate_passed": height_gate_passed,
            "unsafe_override_requested": allow_unsafe,
            "manual_review_required": bool(safety_warnings),
            "safety_warnings": safety_warnings,
            "trajectory_radius": trajectory["radius"],
        },
    }


def _unsafe_publish_marker(
    run: RunManifest, version: str, publish_metrics: dict[str, Any]
) -> str:
    """Human-readable stop sign placed beside every deliberately unsafe export."""
    warnings = publish_metrics.get("safety_warnings", [])
    lines = [
        "UNSAFE PUBLISH FRAME - MANUAL REVIEW REQUIRED",
        "=============================================",
        "",
        f"Run: {run.id}",
        f"Version: {version}",
        "",
        "This export explicitly bypassed publish-frame safety gates.",
        "Do not accept or distribute it until a viewer confirms orientation,",
        "camera alignment, scale, and scene usability.",
        "",
        "Triggered safety warnings:",
        *(f"- {warning}" for warning in warnings),
        "",
        "Measured publish-frame metrics:",
        json.dumps(publish_metrics, indent=2, ensure_ascii=False, sort_keys=True),
        "",
    ]
    return "\n".join(lines)


def export_run(
    scene_path: Path,
    run: RunManifest,
    version: str = "v001",
    resume: bool = False,
    cull: CullSettings | None = None,
    allow_unsafe_publish_frame: bool = False,
) -> RunManifest:
    if not VERSION_PATTERN.match(version):
        raise ValueError("Artifact version must look like v001")
    export_dir = scene_path / "exports" / version
    manifest_path = export_dir / "artifact.yaml"
    # A version with nothing on disk has not been published, whatever the manifest
    # remembers, so producing it adds artifacts rather than replacing any. That is
    # what lets a finished run publish a differently culled version without
    # retraining, and what lets a deleted export be regenerated. A version that does
    # have files still hits the not-empty guard below.
    republish = not (export_dir.is_dir() and any(export_dir.iterdir()))
    if not begin_stage(run, "export", resume=resume, force=republish):
        return run
    save_run(scene_path, run)
    work = ensure_work_dir(scene_path, run.id)
    log_dir = work / "logs"
    try:
        if export_dir.exists() and any(export_dir.iterdir()) and not resume:
            raise FileExistsError(f"Export directory is not empty: {export_dir}")
        export_dir.mkdir(parents=True, exist_ok=True)
        config = ensure_within(scene_path / run.metrics["train"]["config_path"], scene_path)
        y_up_export = getattr(run.config, "schema_version", 1) >= 2
        if y_up_export:
            staging = work / "export-staging" / version
            staging.mkdir(parents=True, exist_ok=True)
            canonical_ply = sorted(staging.glob("*.ply"))
            if not canonical_ply:
                run_logged(
                    [
                        "ns-export",
                        "gaussian-splat",
                        "--load-config",
                        str(config),
                        "--output-dir",
                        str(staging),
                    ],
                    log_dir / "export-ply.log",
                )
                canonical_ply = sorted(staging.glob("*.ply"))
            if len(canonical_ply) != 1:
                raise RuntimeError(
                    f"Expected one staged canonical PLY, found {len(canonical_ply)}"
                )
            frame = resolve_publish_frame(
                scene_path,
                run,
                config,
                allow_unsafe=allow_unsafe_publish_frame,
            )
            publish_metrics = dict(frame["metrics"])
            final_ply = export_dir / "splat-yup.ply"
            if not final_ply.is_file():
                spec = (
                    CullSpec(
                        cameras=frame["cameras_model"],
                        trajectory_radius=frame["trajectory_radius"],
                        distance_factor=cull.distance_factor,
                        scale_factor=cull.scale_factor,
                        max_removed_fraction=cull.max_removed_fraction,
                    )
                    if cull is not None and cull.enabled
                    else None
                )
                publish_metrics.update(
                    publish_gaussian_ply(canonical_ply[0], final_ply, frame["rotation"], spec)
                )
            ply_files = [final_ply]
        else:
            ply_files = sorted(export_dir.glob("*.ply"))
            if not ply_files:
                run_logged(
                    [
                        "ns-export",
                        "gaussian-splat",
                        "--load-config",
                        str(config),
                        "--output-dir",
                        str(export_dir),
                    ],
                    log_dir / "export-ply.log",
                )
                ply_files = sorted(export_dir.glob("*.ply"))
            if len(ply_files) != 1:
                raise RuntimeError(f"Expected one exported PLY, found {len(ply_files)}")

        preview = export_dir / "preview.mp4"
        if not preview.is_file():
            run_logged(
                _preview_render_command(config, preview),
                log_dir / "export-preview.log",
            )
        thumbnail = export_dir / "thumbnail.jpg"
        if not thumbnail.is_file():
            run_logged(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    "1",
                    "-i",
                    str(preview),
                    "-frames:v",
                    "1",
                    str(thumbnail),
                ],
                log_dir / "export-thumbnail.log",
            )

        dataset = ensure_within(scene_path / (run.selected_dataset or ""), scene_path)
        if y_up_export:
            # The reconstruction's transforms.json is in COLMAP space, but the PLY
            # comes out of training in the dataparser's normalised frame. Publishing
            # both untouched hands over two files that cannot be overlaid, so the
            # cameras are rewritten into the PLY's own frame and the raw pair is
            # kept alongside for audit.
            transform_names = (
                "transforms.json",
                "transforms-colmap.json",
                "dataparser_transforms.json",
            )
            shutil.copy2(dataset / "transforms.json", export_dir / "transforms-colmap.json")
            shutil.copy2(
                Path(config).parent / "dataparser_transforms.json",
                export_dir / "dataparser_transforms.json",
            )
            write_published_transforms(
                dataset / "transforms.json",
                export_dir / "transforms.json",
                *frame["dataparser"],
                frame["rotation"],
            )
            if frame["metrics"].get("manual_review_required"):
                (export_dir / UNSAFE_PUBLISH_MARKER).write_text(
                    _unsafe_publish_marker(run, version, frame["metrics"]),
                    encoding="utf-8",
                )
        else:
            transform_names = ("transforms.json", "dataparser_transforms.json")
            for name in transform_names:
                source = dataset / name
                if source.is_file():
                    shutil.copy2(source, export_dir / name)

        artifacts = [
            _artifact_record(ply_files[0], scene_path, "gaussian_ply", version),
            _artifact_record(preview, scene_path, "preview_video", version),
            _artifact_record(thumbnail, scene_path, "thumbnail", version),
        ]
        for name in transform_names:
            target = export_dir / name
            if target.is_file():
                artifacts.append(_artifact_record(target, scene_path, name.removesuffix(".json"), version))
        unsafe_marker = export_dir / UNSAFE_PUBLISH_MARKER
        if unsafe_marker.is_file():
            artifacts.append(
                _artifact_record(unsafe_marker, scene_path, "safety_warning", version)
            )
        count = gaussian_count(ply_files[0])
        export_metrics = run.metrics.setdefault("export", {})
        export_metrics["gaussian_count"] = count
        export_metrics["ply_axis"] = "y_up" if y_up_export else "z_up"
        export_metrics["gaussian_count_warning"] = (
            "below_2_million" if count < 2_000_000 else None
        )
        if y_up_export:
            cull_keys = {
                "input_gaussians",
                "published_gaussians",
                "removed_total",
                "removed_beyond_distance",
                "removed_above_scale",
                "removed_fraction",
                "distance_factor",
                "scale_factor",
                "max_distance",
                "max_scale",
                "max_removed_fraction",
                "camera_to_cloud_median",
                "trajectory_radius",
            }
            export_metrics["cull"] = {
                **(cull.as_metrics() if cull is not None else CullSettings(enabled=False).as_metrics()),
                **{key: value for key, value in publish_metrics.items() if key in cull_keys},
            }
            export_metrics["publish_frame"] = {
                key: value
                for key, value in publish_metrics.items()
                if key not in cull_keys and key != "rotation"
            }
            export_metrics["publish_frame"]["rotation"] = publish_metrics.get(
                "rotation", [[float(v) for v in row] for row in frame["rotation"]]
            )
        # Versions accumulate: publishing v003 must not erase the v002 record.
        kept = [item for item in run.artifacts if item.version != version]
        run.artifacts = kept + artifacts
        # The top-level block describes the newest export; the per-version copy is
        # what each artifact manifest gets stamped with, so v002 keeps describing
        # v002 after v003 is published.
        export_metrics.setdefault("versions", {})[version] = {
            key: value for key, value in export_metrics.items() if key != "versions"
        }
        _record_resources(run, work, extra_path=export_dir)
        artifact_manifest = ArtifactManifest(
            location_id=run.location_id,
            scene_id=run.scene_id,
            run_id=run.id,
            version=version,
            status=RunStatus.NEEDS_REVIEW,
            generated_at=utc_now(),
            artifacts=artifacts,
            qa_metrics=metrics_for_version(run, version),
        )
        save_yaml(manifest_path, artifact_manifest)
        complete_stage(
            run,
            "export",
            message=f"Exported {len(artifacts)} artifacts to {export_dir.relative_to(scene_path)}",
            log_path=log_dir.relative_to(scene_path).as_posix(),
        )
    except Exception as error:
        fail_stage(run, "export", str(error), log_dir.relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return run


def _artifact_path(run: RunManifest, scene_path: Path, kind: str) -> Path | None:
    for artifact in run.artifacts:
        if artifact.kind == kind:
            candidate = scene_path / artifact.relative_path
            if candidate.is_file():
                return candidate
    return None


def _transforms_payload(scene_path: Path, run: RunManifest) -> dict[str, Any]:
    path = _artifact_path(run, scene_path, "transforms")
    if path is None and run.selected_dataset:
        candidate = scene_path / run.selected_dataset / "transforms.json"
        path = candidate if candidate.is_file() else None
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def quality_snapshot(scene_path: Path, run: RunManifest) -> dict[str, Any]:
    selected_name = "fallback" if run.fallback_attempted else "primary"
    selected_attempt = getattr(run.config.reconstruction, selected_name)
    reconstruction = run.metrics.get("reconstruction", {}).get(selected_name, {})
    preprocess = run.metrics.get("preprocess", {})
    selected_preprocess = preprocess.get("selected_primary", preprocess)
    transforms = _transforms_payload(scene_path, run)
    width = int(transforms.get("w", getattr(selected_attempt, "projection_size", 0)))
    fov = projection_fov_degrees(selected_attempt)
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0)) if width else 0.0
    frame_centers: list[tuple[str, Any]] = []
    for frame in transforms.get("frames", []):
        matrix = frame.get("transform_matrix")
        if matrix and len(matrix) >= 3:
            frame_centers.append(
                (
                    str(frame.get("file_path", "")),
                    [float(matrix[index][3]) for index in range(3)],
                )
            )
    recorded_rig = reconstruction.get("rig", {}).get("post_bundle_adjustment", {})
    spread = (
        recorded_rig
        if recorded_rig
        else (center_spread_metrics(frame_centers) if frame_centers else {})
    )
    ply_path = _artifact_path(run, scene_path, "gaussian_ply")
    count = (
        int(run.metrics.get("export", {}).get("gaussian_count", 0))
        or (gaussian_count(ply_path) if ply_path else 0)
    )
    resources = run.metrics.get("resources", {})
    return {
        "angular_resolution_px_per_degree": focal * math.pi / 180.0,
        "candidate_frames": int(
            preprocess.get("candidate_frame_count", run.config.preprocess.target_frames)
        ),
        "selected_frames": int(selected_attempt.frame_count),
        "blur_median": float(selected_preprocess.get("blur_median", 0.0)),
        "registration_ratio": float(reconstruction.get("registration_ratio", 0.0)),
        "largest_component_coverage": float(
            reconstruction.get("largest_component_coverage", 0.0)
        ),
        "fixed_intrinsics": bool(reconstruction.get("fixed_intrinsics", False)),
        "rig_center_spread_p95": float(spread.get("center_spread_p95", 0.0)),
        "rig_spread_to_baseline": float(spread.get("p95_spread_to_baseline", 0.0)),
        "gaussian_count": count,
        "reconstruct_seconds": float(run.stages["reconstruct"].elapsed_seconds or 0.0),
        "train_seconds": float(run.stages["train"].elapsed_seconds or 0.0),
        "gpu_memory_peak_mib": float(resources.get("gpu_memory_observed_peak_mib", 0.0)),
        "disk_peak_gib": float(resources.get("disk_observed_peak_bytes", 0.0)) / 1024**3,
    }


def compare_quality_runs(
    scene_path: Path, current: RunManifest, baseline: RunManifest
) -> dict[str, Any]:
    current_values = quality_snapshot(scene_path, current)
    baseline_values = quality_snapshot(scene_path, baseline)
    comparison: dict[str, Any] = {
        "baseline_run_id": baseline.id,
        "current_run_id": current.id,
        "metrics": {},
    }
    for name, current_value in current_values.items():
        baseline_value = baseline_values.get(name)
        record: dict[str, Any] = {"baseline": baseline_value, "current": current_value}
        if (
            isinstance(current_value, (int, float))
            and not isinstance(current_value, bool)
            and isinstance(baseline_value, (int, float))
            and not isinstance(baseline_value, bool)
        ):
            record["delta"] = float(current_value) - float(baseline_value)
        comparison["metrics"][name] = record
    return comparison


def _format_comparison_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_qa_report(
    scene_path: Path,
    run: RunManifest,
    baseline: RunManifest | None = None,
    resume: bool = False,
) -> Path:
    if not begin_stage(run, "qa", resume=resume):
        return scene_path / "qa" / f"{run.id}.md"
    save_run(scene_path, run)
    report_path = scene_path / "qa" / f"{run.id}.md"
    try:
        reconstruction = run.metrics.get("reconstruction", {})
        resources = run.metrics.get("resources", {})
        selected_name = "fallback" if run.fallback_attempted else "primary"
        selected = reconstruction.get(selected_name, {})
        masking = run.metrics.get("masking", {}).get(selected_name, {})
        mask_validation = masking.get("validation", {})
        mask_vision = masking.get("vision_qa", {})
        snapshot = quality_snapshot(scene_path, run)
        run.metrics.setdefault("qa", {})["quality_snapshot"] = snapshot
        comparison = None
        if baseline is not None:
            comparison = compare_quality_runs(scene_path, run, baseline)
            run.metrics.setdefault("qa", {})["comparison"] = comparison
        lines = [
            f"# QA: {run.id}",
            "",
            f"- 状态：`needs_review`",
            f"- 配置哈希：`{run.config_hash}`",
            f"- 选中数据集：`{run.selected_dataset}`",
            f"- 注册率：{float(selected.get('registration_ratio', 0)):.2%}",
            f"- 最大连通模型覆盖：{float(selected.get('largest_component_coverage', 0)):.2%}",
            f"- 人像遮罩数：{int(mask_validation.get('mask_count', 0))}",
            f"- 遮罩最大像素占比：{float(mask_validation.get('max_masked_fraction', 0)):.2%}",
            f"- 遮罩确定性 QA：`{mask_validation.get('deterministic_qa', 'missing')}`",
            f"- DeepSeek 遮罩 QA：`{mask_vision.get('decision', mask_vision.get('status', 'missing'))}`",
            f"- 使用降级重建：{'是' if run.fallback_attempted else '否'}",
            f"- 使用 OOM 降采样重试：{'是' if run.metrics.get('train', {}).get('oom_retry') else '否'}",
            f"- 观测磁盘峰值：{float(resources.get('disk_observed_peak_bytes', 0)) / 1024**3:.2f} GiB",
            f"- 观测显存峰值：{float(resources.get('gpu_memory_observed_peak_mib', 0)):.0f} MiB",
            f"- 高斯数量：{int(snapshot['gaussian_count']):,}",
            f"- PLY 轴向：`{run.metrics.get('export', {}).get('ply_axis', 'unknown')}`",
            "",
        ]
        export_metrics = run.metrics.get("export", {})
        cull_metrics = export_metrics.get("cull", {})
        if cull_metrics.get("removed_total") is not None:
            lines.extend(
                [
                    "## 外围高斯剔除",
                    "",
                    f"- 剔除总数：{int(cull_metrics['removed_total']):,} / "
                    f"{int(cull_metrics['input_gaussians']):,}"
                    f"（{float(cull_metrics['removed_fraction']):.2%}）",
                    f"- 超出距离上限：{int(cull_metrics.get('removed_beyond_distance', 0)):,}"
                    f"（> {float(cull_metrics.get('max_distance', 0)):.3f}）",
                    f"- 超出尺度上限：{int(cull_metrics.get('removed_above_scale', 0)):,}"
                    f"（> {float(cull_metrics.get('max_scale', 0)):.3f}）",
                    f"- 剔除比例上限：{float(cull_metrics.get('max_removed_fraction', 0)):.2%}",
                    "",
                    "> 剔除只作用于发布产物，不改变训练结果；判据是到最近相机的距离与绝对尺度，"
                    "不是不透明度——外围高斯球本身是完全不透明的。",
                    "",
                ]
            )
        frame_metrics = export_metrics.get("publish_frame", {})
        if frame_metrics.get("up_vector") is not None:
            lines.extend(
                [
                    "## 发布坐标系",
                    "",
                    f"- rig 恢复的重力方向逐图偏差：中位 "
                    f"{float(frame_metrics.get('deviation_median_degrees', 0)):.2f}°、"
                    f"p95 {float(frame_metrics.get('deviation_p95_degrees', 0)):.2f}°"
                    f"（上限 {float(frame_metrics.get('deviation_limit_degrees', 0)):.1f}°）",
                    f"- 发布后相机高度跨度占比："
                    f"{float(frame_metrics.get('published_height_span_ratio', 0)):.2%}"
                    f"（上限 {float(frame_metrics.get('published_height_span_limit', 0)):.0%}）",
                    f"- 位姿归一化峰值："
                    f"{float(frame_metrics.get('pose_normalisation_peak', 0)):.4f}（应约等于 1.0）",
                    "",
                ]
            )
        eval_curve = run.metrics.get("train", {}).get("eval_curve", [])
        if eval_curve:
            lines.extend(["## 训练评估曲线", "", "| 步数 | PSNR | SSIM | LPIPS |", "| ---: | ---: | ---: | ---: |"])
            for point in eval_curve:
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            f"{int(point['step']):,}",
                            *(
                                f"{point[name]:.4f}" if name in point else "-"
                                for name in EVAL_SCALARS
                            ),
                        )
                    )
                    + " |"
                )
            lines.extend(["", "> 用于判断迭代步数的收益拐点；曲线走平之后的步数是可以省掉的。", ""])
        if int(snapshot["gaussian_count"]) < 2_000_000:
            lines.extend(
                [
                    "> 警告：高斯数量低于 2,000,000；只记录警告，不自动追加训练。",
                    "",
                ]
            )
        if comparison is not None:
            lines.extend(
                [
                    f"## 与基线 `{baseline.id}` 对比",
                    "",
                    "| 指标 | 基线 | 当前 | 差值 |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for name, values in comparison["metrics"].items():
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            name,
                            _format_comparison_value(values.get("baseline")),
                            _format_comparison_value(values.get("current")),
                            _format_comparison_value(values.get("delta", "—")),
                        )
                    )
                    + " |"
                )
            lines.extend(["", "## 最终资产人工检查清单（人物遮罩已在 mask 阶段门禁）", ""])
        else:
            lines.extend(["## 最终资产人工检查清单（人物遮罩已在 mask 阶段门禁）", ""])
        lines.extend([
            "- [ ] PLY 可在兼容查看器中打开",
            "- [ ] 预览视频不存在明显相机轨迹断裂",
            "- [ ] 记录漂浮噪点、几何断裂和灯光曝光伪影",
            "- [ ] 确认素材与场地使用权限",
            "- [ ] 决定 accepted 或 rejected",
            "",
            "## 说明",
            "",
            "此试点只验证流程，不保证度量尺度或正式参考库质量。",
        ])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        complete_stage(run, "qa", message=f"QA report: {report_path.relative_to(scene_path)}")
        for artifact_path in (scene_path / "exports").glob("*/artifact.yaml"):
            artifact = load_model(artifact_path, ArtifactManifest)
            if artifact.run_id == run.id:
                artifact.qa_metrics = metrics_for_version(run, artifact.version, artifact.qa_metrics)
                artifact.status = RunStatus.NEEDS_REVIEW
                save_yaml(artifact_path, artifact)
    except Exception as error:
        fail_stage(run, "qa", str(error))
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    return report_path


def review_run(scene_path: Path, run: RunManifest, accepted: bool, notes: str) -> RunManifest:
    if run.status != RunStatus.NEEDS_REVIEW:
        raise RuntimeError(f"Only needs_review runs can be reviewed; current status is {run.status.value}")
    run.status = RunStatus.ACCEPTED if accepted else RunStatus.REJECTED
    run.review_notes = notes
    for artifact_path in (scene_path / "exports").glob("*/artifact.yaml"):
        artifact = load_model(artifact_path, ArtifactManifest)
        if artifact.run_id == run.id:
            artifact.status = run.status
            save_yaml(artifact_path, artifact)
    save_run(scene_path, run)
    return run
