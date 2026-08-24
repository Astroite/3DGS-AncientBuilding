from datetime import datetime, timezone
from pathlib import Path

import pytest

from gsdb.manifests import canonical_hash
from gsdb.models import LegacyRunConfigV1, RunConfig, RunStatus, StageStatus
from gsdb.runs import (
    begin_stage,
    complete_stage,
    create_run,
    fail_stage,
    load_run,
    save_run,
)


def test_run_id_and_stage_resume(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    (scene / "runs").mkdir(parents=True)
    (scene / "work").mkdir()
    config = RunConfig(capture_id="capture-001", input_sha256="a" * 64)
    run = create_run(
        scene,
        "site-001",
        "scene-001",
        config,
        now=datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc),
    )
    assert run.id == f"20260823T120000Z-{run.config_hash[:8]}"
    assert begin_stage(run, "preprocess") is True
    complete_stage(run, "preprocess")
    save_run(scene, run)
    loaded = load_run(scene, run.id)
    assert loaded.stages["preprocess"].status == StageStatus.SUCCEEDED
    assert loaded.stages["preprocess"].elapsed_seconds is not None
    assert begin_stage(loaded, "preprocess", resume=True) is False
    with pytest.raises(RuntimeError):
        begin_stage(loaded, "preprocess", resume=False)


def test_failed_stage_never_becomes_success_implicitly(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    (scene / "runs").mkdir(parents=True)
    (scene / "work").mkdir()
    run = create_run(
        scene,
        "site-001",
        "scene-001",
        RunConfig(capture_id="capture-001", input_sha256="b" * 64),
    )
    begin_stage(run, "preprocess")
    fail_stage(run, "preprocess", "decode failed")
    assert run.status == RunStatus.FAILED
    assert run.stages["preprocess"].status == StageStatus.FAILED


def test_changed_configuration_cannot_resume_existing_run(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    (scene / "runs").mkdir(parents=True)
    (scene / "work").mkdir()
    run = create_run(
        scene,
        "site-001",
        "scene-001",
        RunConfig(capture_id="capture-001", input_sha256="d" * 64),
    )
    run.config.preprocess.target_frames = 180
    save_run(scene, run)
    with pytest.raises(RuntimeError, match="configuration changes require a new run"):
        load_run(scene, run.id)


def test_reconstruction_requires_successful_mask_stage(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    (scene / "runs").mkdir(parents=True)
    (scene / "work").mkdir()
    run = create_run(
        scene,
        "site-001",
        "scene-001",
        RunConfig(capture_id="capture-001", input_sha256="e" * 64),
    )
    begin_stage(run, "preprocess")
    complete_stage(run, "preprocess")
    with pytest.raises(RuntimeError, match="requires mask=succeeded"):
        begin_stage(run, "reconstruct")
    begin_stage(run, "mask")
    complete_stage(run, "mask")
    assert begin_stage(run, "reconstruct") is True


def test_historical_v1_hash_loads_unchanged_and_v2_fields_enter_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    scene = root / "locations/yanguan-ancient-town-20260822/scenes/night-walk-4k"
    historical = load_run(scene, "20260824T022046Z-5679786b")
    assert isinstance(historical.config, LegacyRunConfigV1)
    assert historical.config_hash == (
        "5679786b71621b06191101aaa50fe27dd83d215193b6532ba7cad6eb12d62323"
    )
    assert "schema_version" not in historical.config.model_dump(mode="json")

    first = RunConfig(capture_id="capture-001", input_sha256="1" * 64)
    second = first.model_copy(deep=True)
    second.reconstruction.primary.projection_size = 2049
    assert first.schema_version == 2
    assert canonical_hash(first) != canonical_hash(second)
