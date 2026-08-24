from pathlib import Path

import pytest

from gsdb.models import RunConfig, RunManifest, StageStatus, utc_now
from gsdb.pipeline import _preview_render_command, _train_command, preprocess_run
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
