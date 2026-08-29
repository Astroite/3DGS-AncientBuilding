from pathlib import Path

import pytest

from gsdb.models import ArtifactRecord, RunConfig, RunManifest, StageRecord, StageStatus, utc_now
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
    preprocess_run,
    read_eval_curve,
    training_image_count,
)
from gsdb.runs import begin_stage, create_run, load_run


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
    (scene / "runs").mkdir(parents=True)
    (scene / "work").mkdir()
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
    for name in ("preprocess", "mask", "reconstruct", "train", "export"):
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
