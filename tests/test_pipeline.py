from pathlib import Path

import pytest

from gsdb.models import RunConfig, RunManifest, StageStatus, utc_now
from gsdb.pipeline import (
    _preview_render_command,
    _train_command,
    compare_quality_runs,
    preprocess_run,
)
from gsdb.runs import create_run, load_run


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
