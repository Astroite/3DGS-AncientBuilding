from pathlib import Path

from gsdb.models import RunConfig, RunManifest, utc_now
from gsdb.pipeline import _train_command


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
