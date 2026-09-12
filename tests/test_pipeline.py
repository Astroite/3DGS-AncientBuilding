from pathlib import Path

import numpy as np
import pytest

from gsdb.manifests import canonical_hash
from gsdb.mask_finalize import MaskFinalizationMissingError
from gsdb.models import (
    ArtifactRecord,
    LegacyRunConfigV1,
    RunConfig,
    RunConfigV3,
    RunConfigV4,
    RunManifest,
    RunStatus,
    StageRecord,
    StageStatus,
    utc_now,
)
from gsdb.pipeline import (
    CullSettings,
    TrainSettings,
    _find_completed_training_config,
    _log_reports_nested_downscale_path_failure,
    _log_reports_memory_exhaustion,
    _preview_render_command,
    _train_command,
    _unsafe_publish_marker,
    compare_quality_runs,
    densification_schedule,
    metrics_for_version,
    export_run,
    mask_run,
    preprocess_run,
    read_eval_curve,
    reconstruct_run,
    training_image_count,
)
from gsdb.runs import begin_stage, create_run, load_run, save_run


def test_train_retry_uses_nerfstudio_dataparser_downscale() -> None:
    now = utc_now()
    run = RunManifest(
        id="20260823T000000Z-12345678",
        location_id="test-location",
        scene_id="test-scene",
        config_hash="12345678" * 8,
        config=RunConfig(capture_id="test-capture", input_sha256="a" * 64),
        created_at=now,
        updated_at=now,
    )

    command = _train_command(run, Path("/data/scene"), Path("/work/output"), downscale=2)

    parser_index = command.index("nerfstudio-data")
    assert command[parser_index + 1 :] == [
        "--data",
        str(Path("/data/scene")),
        "--downscale-factor",
        "2",
    ]
    assert command[:2] == ["ns-train", "splatfacto-big"]
    expected_options = {
        "--max-num-iterations": "100000",
        "--pipeline.datamanager.cache-images": "cpu",
        "--pipeline.datamanager.cache-images-type": "uint8",
        "--pipeline.model.use-scale-regularization": "True",
        "--pipeline.model.rasterize-mode": "classic",
        "--pipeline.model.use-bilateral-grid": "True",
        "--pipeline.model.camera-optimizer.mode": "SO3xR3",
    }
    for option, value in expected_options.items():
        assert command[command.index(option) + 1] == value


def test_training_config_is_complete_only_when_the_final_checkpoint_exists(
    tmp_path: Path,
) -> None:
    training = tmp_path / "attempt" / "unnamed" / "splatfacto" / "run"
    checkpoints = training / "nerfstudio_models"
    checkpoints.mkdir(parents=True)
    config = training / "config.yml"
    config.write_text("config\n", encoding="utf-8")

    assert _find_completed_training_config(tmp_path / "attempt", 100_000) is None
    (checkpoints / "step-000010000.ckpt").write_bytes(b"partial")
    assert _find_completed_training_config(tmp_path / "attempt", 100_000) is None
    (checkpoints / "step-000099999.ckpt").write_bytes(b"complete")
    assert _find_completed_training_config(tmp_path / "attempt", 100_000) == config


@pytest.mark.parametrize(
    "message",
    [
        "RuntimeError: CUDA out of memory",
        "OSError: [Errno 12] Cannot allocate memory",
        "allocator failed with std::bad_alloc",
    ],
)
def test_training_retry_recognizes_gpu_and_cpu_memory_exhaustion(
    tmp_path: Path, message: str
) -> None:
    log = tmp_path / "train.log"
    log.write_text(message, encoding="utf-8")
    assert _log_reports_memory_exhaustion(log)


def test_training_retry_recognizes_nerfstudio_nested_downscale_path_bug(
    tmp_path: Path,
) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'/data/images_2/frame_000001.jpg'",
        encoding="utf-8",
    )
    assert _log_reports_nested_downscale_path_failure(log)


def test_preview_render_uses_full_image_datamanager_compatible_path() -> None:
    config = Path("/work/config.yml")
    preview = Path("/exports/preview.mp4")

    command = _preview_render_command(config, preview)

    assert command[:2] == ["ns-render", "interpolate"]
    assert "--pose-source" in command
    assert command[command.index("--pose-source") + 1] == "eval"
    assert command[command.index("--interpolation-steps") + 1] == "1"
    assert "--seconds" not in command


def test_stage_start_is_saved_before_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    scene.mkdir(parents=True)
    run = create_run(
        scene,
        "site-001",
        "scene-001",
        RunConfig(capture_id="capture-001", input_sha256="f" * 64),
    )

    def interrupt() -> dict[str, str]:
        raise KeyboardInterrupt

    monkeypatch.setattr("gsdb.pipeline.collect_tool_versions", interrupt)
    with pytest.raises(KeyboardInterrupt):
        preprocess_run(scene, run)
    persisted = load_run(scene, run.id)
    assert persisted.active_stage == "preprocess"
    assert persisted.stages["preprocess"].status == StageStatus.PROCESSING


def test_fallback_waits_for_review_and_resumes_only_after_valid_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    config = RunConfigV3(
        capture_id="capture-001",
        input_dataset_sha256="b" * 64,
        prepared_relative_path="inputs/prepared/capture-001/hash",
        input={
            "source_kind": "equirect_sequence",
            "source_sha256": ["a" * 64],
            "source_probe": {
                "width": 40,
                "height": 20,
                "fps": 1,
                "frame_count": 3,
                "duration_seconds": 3,
            },
            "normalization": {"width": 40, "height": 20},
            "selection": {"end_seconds": 3},
            "candidate_frame_indices": [0, 1, 2],
        },
        preprocess={"target_frames": 3, "minimum_free_gib": 1},
        reconstruction={
            "primary": {
                "frame_count": 2,
                "images_per_equirect": 8,
                "projection_fov_degrees": 120,
                "projection_size": 256,
                "crop_bottom": 0.2,
            },
            "fallback": {
                "frame_count": 3,
                "images_per_equirect": 14,
                "projection_fov_degrees": 110,
                "projection_size": 256,
                "crop_bottom": 0.15,
            },
        },
    )
    now = utc_now()
    run = RunManifest(
        id="20260903T000000Z-fallback",
        location_id="site-001",
        scene_id="scene-001",
        config_hash=canonical_hash(config),
        config=config,
        created_at=now,
        updated_at=now,
    )
    run.stages["preprocess"] = StageRecord(status=StageStatus.SUCCEEDED)
    run.stages["mask"] = StageRecord(status=StageStatus.SUCCEEDED)
    work = scene / run.id
    work.mkdir(parents=True)
    records = [
        {
            "file": f"frame_{index:06d}.jpg",
            "timestamp_seconds": float(index),
            "selection_score": float(4 - index),
        }
        for index in range(1, 4)
    ]
    (work / "frame-metrics.jsonl").write_text(
        "".join(__import__("json").dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    save_run(scene, run)

    monkeypatch.setattr(
        "gsdb.pipeline.create_blur_aware_subset", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        "gsdb.pipeline.select_blur_aware_records",
        lambda items, target, **kwargs: items[:target],
    )
    monkeypatch.setattr(
        "gsdb.pipeline._prepare_masked_dataset",
        lambda *args, **kwargs: {"planar_images": 42},
    )
    alignment_calls: list[str] = []

    def fake_alignment(output: Path, *args, **kwargs) -> dict[str, float]:
        alignment_calls.append(output.name)
        score = 0.5 if output.name.endswith("primary") else 0.9
        return {"registration_ratio": score, "largest_component_coverage": score}

    monkeypatch.setattr("gsdb.pipeline.run_realityscan_alignment", fake_alignment)
    finalization = {"state": "missing"}

    def fake_finalization(path: Path, expected) -> dict[str, str]:
        if path.name.endswith("fallback"):
            if finalization["state"] == "missing":
                raise MaskFinalizationMissingError("fallback review is pending")
            if finalization["state"] == "corrupt":
                raise RuntimeError("mask-final is stale or corrupt")
        return {"status": "passed"}

    monkeypatch.setattr("gsdb.pipeline.validate_mask_finalization", fake_finalization)
    monkeypatch.setattr(
        "gsdb.pipeline.write_trajectory_qa",
        lambda *args, **kwargs: {"blocking": [], "warnings": [], "samples": []},
    )

    waiting = reconstruct_run(scene, run)
    assert waiting.status == RunStatus.WAITING_REVIEW
    assert waiting.active_stage == "reconstruct"
    assert waiting.stages["reconstruct"].status == StageStatus.PROCESSING
    assert waiting.fallback_attempted is True
    assert waiting.selected_dataset is None
    assert waiting.metrics["reconstruction"]["primary"]["registration_ratio"] == 0.5

    # Resume through the persisted manifest, as separate CLI invocations do.
    waiting = load_run(scene, run.id)
    finalization["state"] = "corrupt"
    with pytest.raises(RuntimeError, match="stale or corrupt"):
        reconstruct_run(scene, waiting, resume=True)
    failed = load_run(scene, run.id)
    assert failed.status == RunStatus.FAILED
    assert failed.stages["reconstruct"].status == StageStatus.FAILED

    finalization["state"] = "passed"
    completed = reconstruct_run(scene, failed, resume=True)
    assert completed.stages["reconstruct"].status == StageStatus.SUCCEEDED
    assert completed.selected_dataset.endswith("reconstruction-fallback")
    assert completed.metrics["reconstruction"]["fallback"]["registration_ratio"] == 0.9
    assert alignment_calls == [
        "reconstruction-primary",
        "reconstruction-primary",
        "reconstruction-primary",
        "reconstruction-fallback",
    ]


@pytest.mark.parametrize("marker", ["complete", "pending", "both"])
@pytest.mark.parametrize("statistics", ["missing", "truncated"])
@pytest.mark.parametrize("manual_review", [False, True])
def test_v4_mask_resume_recovers_without_metrics_source_or_models(
    tmp_path, monkeypatch, marker, statistics, manual_review,
):
    import json
    import cv2
    import numpy as np
    from gsdb.masking import filter_masked_images
    from gsdb.pipeline import _prepare_masked_dataset

    run = _v4_run()
    run.config.masking.mask_review_required = manual_review
    run.config.vision_qa.enabled = False
    work = tmp_path / "work"
    dataset = work / "reconstruction-fallback"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    records, rejected = [], {}
    for name, pixels in (("keep.jpg", 20), ("drop.jpg", 21)):
        image = dataset / "images" / name
        mask = dataset / "masks" / f"{name}.png"
        assert cv2.imwrite(str(image), np.full((20, 20, 3), 100, np.uint8))
        values = np.full((20, 20), 255, np.uint8)
        values.flat[:pixels] = 0
        assert cv2.imwrite(str(mask), values)
        records.append({"image": name, "mask": mask.name, "masked_fraction": pixels / 400})
        if name == "drop.jpg":
            rejected = {image: image.read_bytes(), mask: mask.read_bytes()}
    payload = filter_masked_images(dataset, records, 0.05)
    if manual_review:
        qa = dataset / "mask-qa"
        qa.mkdir()
        (qa / "mask-contact-01.jpg").write_bytes(b"existing QA artifact")
    if marker in {"pending", "both"}:
        (dataset / ".mask-filter.pending.json").write_text(json.dumps({**payload, "status": "pending"}))
    if marker == "pending":
        (dataset / "mask-filter.json").unlink()
    for path, content in rejected.items():
        path.write_bytes(content)
    if statistics == "truncated":
        (work / "mask-metrics-fallback.jsonl").write_text(
            json.dumps({"image": "keep.jpg", "detections": 3, "masked_fraction": 0.9}) + '\n{"image":'
        )
    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery must not project or instantiate segmentation")
    monkeypatch.setattr("gsdb.pipeline.project_equirectangular_frames", forbidden)
    monkeypatch.setattr("gsdb.pipeline.generate_person_masks", forbidden)
    monkeypatch.setattr("gsdb.masking.TorchvisionPersonSegmenter", forbidden)
    monkeypatch.setattr("gsdb.pipeline.create_mask_contact_sheets", forbidden)
    for _ in range(2):
        result = _prepare_masked_dataset(
            tmp_path, work, work / "missing-source", dataset,
            run.config.reconstruction.fallback, run, "fallback",
        )
        assert result["planar_images"] == result["projected_planar_images"] == 2
        assert result["reconstruction_input_images"] == 1
        assert result["automatic_rejected_fraction"] == 0.5
        assert result["validation"]["max_masked_fraction"] == 0.05
        assert result["detection"]["images_with_detection_metrics"] == (statistics == "truncated")
        assert result["detection"]["per_class"] == {}
        run.metrics.setdefault("masking", {})["fallback"] = result
    assert not any(path.exists() for path in rejected)
    assert (dataset / "mask-final.json").is_file() == (not manual_review)
    if manual_review:
        assert (dataset / "mask-qa/mask-contact-01.jpg").read_bytes() == b"existing QA artifact"


def _v4_run() -> RunManifest:
    config = RunConfigV4(
        capture_id="capture-001",
        input_dataset_sha256="b" * 64,
        prepared_relative_path="prepared/capture-001/hash",
        input={
            "source_kind": "equirect_sequence",
            "source_sha256": ["a" * 64],
            "source_probe": {
                "width": 40,
                "height": 20,
                "fps": 30,
                "frame_count": 72,
                "duration_seconds": 2.4,
            },
            "normalization": {"width": 40, "height": 20},
            "selection": {"end_seconds": 2.4},
            "candidate_frame_indices": list(range(12)),
            "candidate_fps": 5.0,
        },
        reconstruction={
            "primary": {
                "temporal_rank_limit": 1,
                "images_per_equirect": 8,
                "projection_fov_degrees": 120,
                "projection_size": 256,
                "crop_bottom": 0.2,
            },
            "fallback": {
                "temporal_rank_limit": 2,
                "images_per_equirect": 14,
                "projection_fov_degrees": 110,
                "projection_size": 256,
                "crop_bottom": 0.15,
            },
        },
    )
    now = utc_now()
    run = RunManifest(
        id="20260906T000000Z-v4",
        location_id="site-001",
        scene_id="scene-001",
        config_hash=canonical_hash(config),
        config=config,
        created_at=now,
        updated_at=now,
    )
    run.stages["preprocess"] = StageRecord(status=StageStatus.SUCCEEDED)
    run.stages["mask"] = StageRecord(status=StageStatus.SUCCEEDED)
    return run


def test_v4_reconstruction_fallback_uses_actual_filtered_inventories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    scene = tmp_path / "scene"
    run = _v4_run()
    work = scene / run.id
    work.mkdir(parents=True)
    (work / "equirect-selected").mkdir()
    (work / "equirect-fallback").mkdir()
    primary_records = [
        {
            "file": f"frame_{index:06d}.jpg",
            "timestamp_seconds": float(index) - 0.9,
            "time_bucket": index - 1,
            "temporal_rank": 1,
        }
        for index in range(1, 4)
    ]
    fallback_records = [
        {
            "file": f"frame_{index:06d}.jpg",
            "timestamp_seconds": 0.1 + (index - 1) * 0.4,
            "time_bucket": (index - 1) // 2,
            "temporal_rank": 1 + (index - 1) % 2,
        }
        for index in range(1, 7)
    ]
    for label, records in (("primary", primary_records), ("fallback", fallback_records)):
        (work / f"selected-{label}-metrics.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
        )
    for label in ("primary", "fallback"):
        (work / f"reconstruction-{label}").mkdir()
    run.metrics["masking"] = {
        "primary": {
            "projected_planar_images": 24,
            "automatic_rejected_images": 2,
            "reconstruction_input_images": 22,
        }
    }
    save_run(scene, run)

    primary_images = {"view_00/frame_000001.jpg", "view_00/frame_000002.jpg"}
    fallback_images = {
        f"view_00/frame_{index:06d}.jpg" for index in range(1, 7)
    }

    def filtered(dataset: Path, **kwargs: object) -> dict[str, object]:
        images = primary_images if dataset.name.endswith("primary") else fallback_images
        return {"accepted": [{"image": name} for name in sorted(images)], "rejected": []}

    def inventory(dataset: Path, run: RunManifest):
        images = primary_images if dataset.name.endswith("primary") else fallback_images
        return {"excluded_images": []}, images

    monkeypatch.setattr("gsdb.pipeline.validate_mask_filter", filtered)
    monkeypatch.setattr("gsdb.pipeline._v4_attempt_inventory", inventory)
    monkeypatch.setattr(
        "gsdb.pipeline._prepare_masked_dataset",
        lambda *args, **kwargs: {
            "projected_planar_images": 84,
            "automatic_rejected_images": 3,
            "reconstruction_input_images": 81,
        },
    )
    calls: list[tuple[str, set[str], int]] = []

    def align(dataset: Path, *args: object, **kwargs: object) -> dict[str, float]:
        included = set(kwargs["included_images"])
        calls.append((dataset.name, included, int(kwargs["projected_image_count"])))
        score = 0.5 if dataset.name.endswith("primary") else 0.9
        return {
            "registration_ratio": score,
            "largest_component_coverage": score,
            "registered_images": len(included),
        }

    monkeypatch.setattr("gsdb.pipeline.run_realityscan_alignment", align)
    observed: dict[str, object] = {}

    def trajectory(dataset: Path, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"blocking": [], "warnings": [], "samples": []}

    monkeypatch.setattr("gsdb.pipeline.write_trajectory_qa", trajectory)
    monkeypatch.setattr("gsdb.pipeline._record_resources", lambda *args, **kwargs: None)

    completed = reconstruct_run(scene, run)
    assert completed.stages["reconstruct"].status == StageStatus.SUCCEEDED
    assert completed.fallback_attempted is True
    assert completed.selected_dataset.endswith("reconstruction-fallback")
    assert calls == [
        ("reconstruction-primary", primary_images, 24),
        ("reconstruction-fallback", fallback_images, 84),
    ]
    assert observed["expected_frames"] == [1, 2, 3, 4, 5, 6]
    assert not (work / "equirect-selected").exists()
    assert not (work / "equirect-fallback").exists()


def test_v4_realityscan_tool_error_does_not_trigger_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    scene = tmp_path / "scene"
    run = _v4_run()
    work = scene / run.id
    primary = work / "reconstruction-primary"
    primary.mkdir(parents=True)
    records = [
        {
            "file": f"frame_{index:06d}.jpg",
            "timestamp_seconds": float(index),
            "time_bucket": index - 1,
            "temporal_rank": 1,
        }
        for index in range(1, 3)
    ]
    (work / "selected-primary-metrics.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    images = {"view_00/frame_000001.jpg", "view_00/frame_000002.jpg"}
    run.metrics["masking"] = {"primary": {"projected_planar_images": 16}}
    save_run(scene, run)
    monkeypatch.setattr(
        "gsdb.pipeline.validate_mask_filter",
        lambda *args, **kwargs: {
            "accepted": [{"image": name} for name in sorted(images)],
            "rejected": [],
        },
    )
    monkeypatch.setattr(
        "gsdb.pipeline._v4_attempt_inventory",
        lambda *args, **kwargs: ({"excluded_images": []}, images),
    )
    monkeypatch.setattr(
        "gsdb.pipeline.run_realityscan_alignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("license failure")),
    )

    with pytest.raises(RuntimeError, match="license failure"):
        reconstruct_run(scene, run)
    assert run.fallback_attempted is False
    assert not (work / "reconstruction-fallback").exists()


def test_qa_comparison_reports_baseline_deltas(tmp_path: Path) -> None:
    now = utc_now()

    def make_run(run_id: str, registration: float, gaussians: int) -> RunManifest:
        run = RunManifest(
            id=run_id,
            location_id="test-location",
            scene_id="test-scene",
            config_hash="1" * 64,
            config=RunConfig(capture_id="test-capture", input_sha256="a" * 64),
            created_at=now,
            updated_at=now,
        )
        run.metrics = {
            "preprocess": {
                "candidate_frame_count": 270,
                "selected_primary": {"blur_median": 20.0},
            },
            "reconstruction": {
                "primary": {
                    "registration_ratio": registration,
                    "largest_component_coverage": registration,
                    "fixed_intrinsics": True,
                }
            },
            "export": {"gaussian_count": gaussians},
        }
        return run

    baseline = make_run("baseline", 0.8, 500_000)
    current = make_run("current", 0.95, 2_500_000)
    comparison = compare_quality_runs(tmp_path, current, baseline)

    metrics = comparison["metrics"]
    assert metrics["registration_ratio"]["delta"] == pytest.approx(0.15)
    assert metrics["gaussian_count"]["delta"] == 2_000_000
    assert comparison["baseline_run_id"] == "baseline"


def _run(**overrides) -> RunManifest:
    now = utc_now()
    return RunManifest(
        id="20260823T000000Z-12345678",
        location_id="test-location",
        scene_id="test-scene",
        config_hash="12345678" * 8,
        config=RunConfig(capture_id="test-capture", input_sha256="a" * 64),
        created_at=now,
        updated_at=now,
        **overrides,
    )


@pytest.mark.parametrize(
    ("config", "removed"),
    [
        (RunConfig(capture_id="test-capture", input_sha256="a" * 64), True),
        (LegacyRunConfigV1(capture_id="test-capture", input_sha256="a" * 64), False),
    ],
)
def test_mask_cleanup_preserves_only_legacy_primary_equirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: RunConfig | LegacyRunConfigV1,
    removed: bool,
) -> None:
    scene = tmp_path / "scene"
    work = scene / "run-001"
    primary = work / "equirect-primary"
    primary.mkdir(parents=True)
    (primary / "frame_000001.jpg").write_bytes(b"frame")
    now = utc_now()
    run = RunManifest(
        id="run-001",
        location_id="test-location",
        scene_id="test-scene",
        config_hash=canonical_hash(config),
        config=config,
        created_at=now,
        updated_at=now,
    )
    run.stages["preprocess"] = StageRecord(status=StageStatus.SUCCEEDED)
    monkeypatch.setattr(
        "gsdb.pipeline._prepare_masked_dataset",
        lambda *args, **kwargs: {"planar_images": 16},
    )
    monkeypatch.setattr("gsdb.pipeline._record_resources", lambda *args, **kwargs: None)

    result = mask_run(scene, run)

    assert result.stages["mask"].status == StageStatus.SUCCEEDED
    assert primary.exists() is not removed


def test_export_cleanup_removes_superseded_version_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    run = _run()
    work = scene / run.id
    dataset = work / "reconstruction-primary"
    training = work / "training"
    staging = work / "export-staging" / "v001"
    export_dir = scene / "exports" / "v001"
    for path in (dataset, training, staging, export_dir):
        path.mkdir(parents=True, exist_ok=True)
    (dataset / "transforms.json").write_text("{}\n", encoding="utf-8")
    (training / "config.yml").write_text("config\n", encoding="utf-8")
    (training / "dataparser_transforms.json").write_text("{}\n", encoding="utf-8")
    (staging / "canonical.ply").write_bytes(b"staged")
    (export_dir / "preview.mp4").write_bytes(b"preview")
    (export_dir / "thumbnail.jpg").write_bytes(b"thumbnail")
    run.selected_dataset = dataset.relative_to(scene).as_posix()
    run.metrics["train"] = {
        "config_path": (training / "config.yml").relative_to(scene).as_posix()
    }
    for stage in ("preprocess", "mask", "reconstruct", "postshot_prepare", "train"):
        run.stages[stage] = StageRecord(status=StageStatus.SUCCEEDED)

    monkeypatch.setattr(
        "gsdb.pipeline.resolve_publish_frame",
        lambda *args, **kwargs: {
            "metrics": {},
            "rotation": np.eye(3),
            "cameras_model": np.empty((0, 3)),
            "trajectory_radius": 1.0,
            "dataparser": (np.eye(4), 1.0),
        },
    )

    def fake_publish(source: Path, target: Path, *args: object) -> dict[str, int]:
        target.write_bytes(b"published")
        return {"input_gaussians": 1, "published_gaussians": 1, "removed_total": 0}

    def fake_transforms(source: Path, target: Path, *args: object) -> None:
        target.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("gsdb.pipeline.publish_gaussian_ply", fake_publish)
    monkeypatch.setattr("gsdb.pipeline.write_published_transforms", fake_transforms)
    monkeypatch.setattr("gsdb.pipeline.gaussian_count", lambda path: 1)

    result = export_run(
        scene,
        run,
        version="v001",
        resume=True,
        cull=CullSettings(enabled=False),
    )

    assert result.stages["export"].status == StageStatus.SUCCEEDED
    assert not staging.exists()


def test_training_keeps_every_checkpoint_so_one_run_answers_the_step_question() -> None:
    command = _train_command(_run(), Path("/data"), Path("/out"))
    assert command[command.index("--save-only-latest-checkpoint") + 1] == "False"
    assert command[command.index("--steps-per-save") + 1] == "10000"
    assert command[command.index("--vis") + 1] == "tensorboard"

    custom = _train_command(
        _run(),
        Path("/data"),
        Path("/out"),
        settings=TrainSettings(steps_per_save=2500, vis="viewer+tensorboard"),
    )
    assert custom[custom.index("--steps-per-save") + 1] == "2500"
    assert custom[custom.index("--vis") + 1] == "viewer+tensorboard"

    # An invalid backend fails when the settings are built, not hours into a run.
    with pytest.raises(ValueError, match="vis must be one of"):
        TrainSettings(vis="none")
    with pytest.raises(ValueError, match="steps_per_save must be positive"):
        TrainSettings(steps_per_save=0)


def test_training_instrumentation_stays_out_of_the_config_hash(tmp_path: Path) -> None:
    # Checkpoint frequency and logging do not change the trained model, so turning
    # them on must not invalidate manifests that are already on disk.
    config = RunConfig(capture_id="test-capture", input_sha256="a" * 64)
    assert not hasattr(config.train, "steps_per_save")
    assert not hasattr(config.train, "vis")
    dumped = config.model_dump(mode="json")
    assert "steps_per_save" not in dumped["train"]
    assert "vis" not in dumped["train"]


def test_cull_settings_report_themselves_for_the_artifact_manifest() -> None:
    assert CullSettings().as_metrics() == {
        "enabled": True,
        "distance_factor": 3.0,
        "scale_factor": 1.0,
        "max_removed_fraction": 0.05,
    }
    assert CullSettings(enabled=False).as_metrics()["enabled"] is False


def test_unsafe_publish_marker_is_unmistakable_and_preserves_gate_evidence() -> None:
    run = _run()
    metrics = {
        "manual_review_required": True,
        "deviation_p95_degrees": 5.2594,
        "published_height_span_ratio": 0.1398,
        "safety_warnings": ["gravity gate failed", "height gate failed"],
    }

    marker = _unsafe_publish_marker(run, "v001", metrics)

    assert "UNSAFE PUBLISH FRAME - MANUAL REVIEW REQUIRED" in marker
    assert run.id in marker
    assert "gravity gate failed" in marker
    assert "height gate failed" in marker
    assert '"deviation_p95_degrees": 5.2594' in marker


def test_eval_curve_is_optional_instrumentation_not_a_hard_dependency(tmp_path: Path) -> None:
    # No event files at all: a run must not fail because a curve is unavailable.
    assert read_eval_curve(tmp_path) == []
    (tmp_path / "events.out.tfevents.123.host").write_bytes(b"not a real event file")
    assert read_eval_curve(tmp_path) == []


def _finished_run() -> RunManifest:
    run = _run(
        artifacts=[
            ArtifactRecord(
                kind="gaussian_ply",
                relative_path="exports/v002/splat-yup.ply",
                sha256="b" * 64,
                byte_size=10,
                version="v002",
            )
        ]
    )
    for name in ("preprocess", "mask", "reconstruct", "postshot_prepare", "train", "export"):
        run.stages[name] = StageRecord(status=StageStatus.SUCCEEDED)
    return run


def test_a_finished_export_stage_can_be_reopened_to_publish_another_version() -> None:
    run = _finished_run()
    assert begin_stage(run, "export", force=True)
    assert run.stages["export"].status == StageStatus.PROCESSING
    # Without force the completed stage still refuses, so nothing is republished by
    # accident.
    finished = _finished_run()
    with pytest.raises(RuntimeError, match="already succeeded"):
        begin_stage(finished, "export")
    assert begin_stage(_finished_run(), "export", resume=True) is False


def test_publishing_a_new_version_keeps_the_previously_published_artifacts() -> None:
    run = _finished_run()
    fresh = [
        ArtifactRecord(
            kind="gaussian_ply",
            relative_path="exports/v003/splat-yup.ply",
            sha256="c" * 64,
            byte_size=20,
            version="v003",
        )
    ]
    kept = [item for item in run.artifacts if item.version != "v003"]
    run.artifacts = kept + fresh
    assert sorted(item.version for item in run.artifacts) == ["v002", "v003"]

    # Re-publishing v003 replaces only v003's own records, never v002's.
    again = [item for item in run.artifacts if item.version != "v003"]
    assert [item.version for item in again] == ["v002"]


def test_republish_is_decided_by_what_is_on_disk_not_by_the_manifest(tmp_path: Path) -> None:
    # The manifest can outlive its files; an absent export directory means the
    # version is unpublished regardless of what the manifest remembers.
    export_dir = tmp_path / "exports" / "v003"
    assert not (export_dir.is_dir() and any(export_dir.iterdir()))
    export_dir.mkdir(parents=True)
    assert not (export_dir.is_dir() and any(export_dir.iterdir()))
    (export_dir / "splat-yup.ply").write_bytes(b"x")
    assert export_dir.is_dir() and any(export_dir.iterdir())


def test_each_published_version_keeps_its_own_export_metrics() -> None:
    # One run can publish several versions; an older one must not start claiming
    # the Gaussian count or cull settings of a later one.
    run = _run()
    run.metrics["export"] = {
        "gaussian_count": 860446,
        "cull": {"removed_total": 18021},
        "versions": {
            "v002": {"gaussian_count": 878467, "cull": {"enabled": False}},
            "v003": {"gaussian_count": 860446, "cull": {"removed_total": 18021}},
        },
    }
    run.metrics["reconstruction"] = {"primary": {"registration_ratio": 0.98}}

    assert metrics_for_version(run, "v003")["export"]["gaussian_count"] == 860446
    assert metrics_for_version(run, "v002")["export"]["gaussian_count"] == 878467
    assert metrics_for_version(run, "v002")["export"]["cull"] == {"enabled": False}
    # Everything outside the export block is genuinely run-wide and stays shared.
    assert metrics_for_version(run, "v002")["reconstruction"] == run.metrics["reconstruction"]
    assert "versions" not in metrics_for_version(run, "v002")["export"]


def test_a_version_published_before_per_version_metrics_keeps_what_it_recorded() -> None:
    run = _run()
    run.metrics["export"] = {"gaussian_count": 860446, "versions": {"v003": {"gaussian_count": 860446}}}
    legacy = {"export": {"gaussian_count": 878467, "ply_axis": "y_up"}}
    assert metrics_for_version(run, "v002", legacy)["export"] == legacy["export"]
    # With nothing to preserve, the block is simply absent rather than wrong.
    assert "export" not in metrics_for_version(run, "v002", None)


def test_densification_schedule_reproduces_nerfstudio_defaults_at_the_baseline_size() -> None:
    # The validated 90-second run is 135 frames x 8 views. Reproducing the upstream
    # defaults exactly there means that run's behaviour is untouched.
    assert densification_schedule(1080, 100_000) == {
        "warmup_length": 500,
        "stop_screen_size_at": 4000,
        "stop_split_at": 15000,
        "reset_alpha_every": 30,
        "refine_every": 100,
    }


def test_densification_schedule_keeps_the_reset_interval_ahead_of_the_refine_pause() -> None:
    # gsplat's DefaultStrategy pauses growth AND pruning for image_count + refine_every
    # steps after every opacity reset. If resets (reset_alpha_every * refine_every apart)
    # arrive faster than that pause expires, refinement locks up permanently -- this is
    # what actually happened on the 5,128-image Yunxiu run: it trained to completion with
    # 1,363,440 gaussians frozen since shortly after warmup, 98.6% of them decayed to
    # near-zero opacity and never pruned, and only nerfstudio's own export-time opacity
    # filter -- not gsdb's culling -- caught it, dropping to 19,705.
    for image_count in (1080, 2160, 2520, 5128, 20_000):
        schedule = densification_schedule(image_count, 100_000)
        reset_interval = schedule["reset_alpha_every"] * schedule["refine_every"]
        refine_pause = image_count + schedule["refine_every"]
        assert reset_interval > refine_pause, image_count
    # The baseline reproduces nerfstudio's own defaults unchanged.
    assert densification_schedule(1080, 100_000)["reset_alpha_every"] == 30
    # The Yunxiu run needed more than the default to stay safe.
    assert densification_schedule(5128, 100_000)["reset_alpha_every"] == 54


def test_densification_schedule_holds_per_view_coverage_as_a_scene_grows() -> None:
    baseline = densification_schedule(1080, 100_000)
    baseline_passes = baseline["stop_split_at"] / 1080
    for images in (2160, 2520, 5128):
        schedule = densification_schedule(images, 100_000)
        assert schedule["stop_split_at"] / images == pytest.approx(baseline_passes, rel=1e-3)
    # The 427-second capture is the case this exists for.
    assert densification_schedule(5128, 100_000)["stop_split_at"] == 71_222


def test_densification_always_leaves_a_refinement_tail() -> None:
    # A dataset large enough to want more splitting steps than the run has must
    # still stop splitting before the end, or nothing ever gets refined.
    schedule = densification_schedule(100_000, 10_000)
    assert schedule["stop_split_at"] <= 7_500
    assert schedule["stop_screen_size_at"] < schedule["stop_split_at"]
    assert schedule["warmup_length"] < schedule["stop_screen_size_at"]
    with pytest.raises(ValueError, match="must be positive"):
        densification_schedule(0, 100_000)


def test_densification_ordering_survives_a_tiny_smoke_run() -> None:
    schedule = densification_schedule(320, 5_000)
    assert schedule["warmup_length"] < schedule["stop_screen_size_at"] < schedule["stop_split_at"]
    assert schedule["stop_split_at"] < 5_000


def test_train_command_scales_densification_with_the_configured_view_count() -> None:
    run = _run()
    run.config.reconstruction.primary.frame_count = 641
    command = _train_command(run, Path("/data"), Path("/out"))
    assert command[command.index("--pipeline.model.stop-split-at") + 1] == "71222"
    assert command[command.index("--pipeline.model.stop-screen-size-at") + 1] == "18993"
    assert command[command.index("--pipeline.model.warmup-length") + 1] == "2374"
    # Scaled past nerfstudio's default so opacity resets cannot outrun the refine pause.
    assert command[command.index("--pipeline.model.reset-alpha-every") + 1] == "54"
    assert command[command.index("--pipeline.model.refine-every") + 1] == "100"
    # The baseline configuration still emits the upstream defaults.
    baseline = _train_command(_run(), Path("/data"), Path("/out"))
    assert baseline[baseline.index("--pipeline.model.stop-split-at") + 1] == "15000"
    assert baseline[baseline.index("--pipeline.model.reset-alpha-every") + 1] == "30"


def test_training_image_count_comes_from_configuration_not_from_reconstruction() -> None:
    run = _run()
    assert training_image_count(run) == 135 * 8
    # Registered-image counts vary with how reconstruction went; the schedule must
    # not depend on them or one config hash would stop meaning one model.
    run.metrics["reconstruction"] = {"primary": {"registered_images": 1067}}
    assert training_image_count(run) == 1080
    run.fallback_attempted = True
    assert training_image_count(run) == 180 * 14
