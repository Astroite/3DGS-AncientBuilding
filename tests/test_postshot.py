import struct
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb.models import RunConfig, RunManifest, StageStatus
from gsdb.postshot import (
    COLMAP_COUNT,
    COLMAP_IMAGE_HEADER,
    prepare_postshot_dataset,
    read_colmap_image_index,
    rewrite_colmap_images_binary,
    build_postshot_train_command,
    train_postshot,
    validate_postshot_dataset,
)
from gsdb import postshot
from gsdb.processes import CommandError


def _write_images_binary(
    path: Path, entries: list[tuple[int, str, list[tuple[float, float, int]]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(COLMAP_COUNT.pack(len(entries)))
        for image_id, name, points in entries:
            stream.write(
                COLMAP_IMAGE_HEADER.pack(
                    image_id,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    float(image_id),
                    2.0,
                    3.0,
                    image_id,
                )
            )
            stream.write(name.encode("utf-8") + b"\0")
            stream.write(COLMAP_COUNT.pack(len(points)))
            for x, y, point_id in points:
                stream.write(struct.pack("<ddq", x, y, point_id))


def _read_payload(path: Path) -> list[tuple[int, str, bytes, bytes]]:
    values = []
    with path.open("rb") as stream:
        count = COLMAP_COUNT.unpack(stream.read(8))[0]
        for _ in range(count):
            fixed = stream.read(COLMAP_IMAGE_HEADER.size)
            image_id = int(COLMAP_IMAGE_HEADER.unpack(fixed)[0])
            name = bytearray()
            while (character := stream.read(1)) != b"\0":
                name.extend(character)
            point_count_bytes = stream.read(8)
            point_count = COLMAP_COUNT.unpack(point_count_bytes)[0]
            points = stream.read(point_count * 24)
            values.append((image_id, name.decode("utf-8"), fixed, point_count_bytes + points))
    return values


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((16, 16, 3), value, dtype=np.uint8))


def _write_colmap_mask(path: Path, ignored: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.full((16, 16), 255, dtype=np.uint8)
    mask[ignored] = 0
    assert cv2.imwrite(str(path), mask)


def _run(run_id: str) -> RunManifest:
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    run = RunManifest(
        id=run_id,
        location_id="site-001",
        scene_id="scene-001",
        config_hash="0" * 64,
        config=RunConfig(capture_id="capture-001", input_sha256="a" * 64),
        created_at=now,
        updated_at=now,
        selected_dataset=f"work/{run_id}/reconstruction-primary",
    )
    run.stages["reconstruct"].status = StageStatus.SUCCEEDED
    return run


def test_rewrite_colmap_images_binary_changes_only_names(tmp_path: Path) -> None:
    source = tmp_path / "images.bin"
    destination = tmp_path / "rewritten.bin"
    _write_images_binary(
        source,
        [
            (7, "view_00/frame_000001.jpg", [(1.5, 2.5, 99)]),
            (12, "view_03/frame_000002.jpg", []),
        ],
    )
    records = read_colmap_image_index(source)

    rewrite_colmap_images_binary(source, destination, records)

    before = _read_payload(source)
    after = _read_payload(destination)
    assert [item[1] for item in after] == ["image_00000007.jpg", "image_00000012.jpg"]
    assert [(item[0], item[2], item[3]) for item in after] == [
        (item[0], item[2], item[3]) for item in before
    ]


def test_prepare_postshot_dataset_copies_registered_images_and_inverts_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    run_id = "run-001"
    dataset = scene / "work" / run_id / "reconstruction-primary"
    images = dataset / "images"
    masks = dataset / "masks"
    model = dataset / "colmap" / "attempt-001" / "rig-sparse"
    model.mkdir(parents=True)
    _write_image(images / "view_00" / "frame_000001.jpg", 80)
    _write_image(images / "view_03" / "frame_000002.jpg", 120)
    _write_colmap_mask(masks / "view_00" / "frame_000001.jpg.png", (2, 3))
    _write_colmap_mask(masks / "view_03" / "frame_000002.jpg.png", (4, 5))
    _write_images_binary(
        model / "images.bin",
        [
            (7, "view_00/frame_000001.jpg", [(1.0, 2.0, 1)]),
            (12, "view_03/frame_000002.jpg", []),
        ],
    )
    (model / "cameras.bin").write_bytes(COLMAP_COUNT.pack(2) + b"camera-data")
    (model / "points3D.bin").write_bytes(COLMAP_COUNT.pack(1) + b"point-data")
    monkeypatch.setattr("gsdb.postshot._selected_model_dir", lambda _: model)
    monkeypatch.setattr("gsdb.postshot._windows_argument", lambda path: str(path))
    target = tmp_path / "postshot"

    result = prepare_postshot_dataset(scene, _run(run_id), output=target)

    assert result["counts"]["images"] == 2
    assert result["counts"]["cameras"] == 2
    assert {item.name for item in (target / "images").iterdir()} == {
        "image_00000007.jpg",
        "image_00000012.jpg",
    }
    converted = cv2.imread(
        str(target / "masks" / "image_00000007.png"), cv2.IMREAD_GRAYSCALE
    )
    assert converted is not None
    assert converted[2, 3] == 255
    assert converted[0, 0] == 0
    assert [item.output_name for item in read_colmap_image_index(target / "colmap" / "images.bin")] == [
        "image_00000007.jpg",
        "image_00000012.jpg",
    ]
    assert validate_postshot_dataset(target)["validation"] == "passed"
    assert prepare_postshot_dataset(scene, _run(run_id), output=target)["reused"] is True

    # A resumed, already-complete .building tree should budget only bytes that
    # still need rewriting, not the full image/mask dataset a second time.
    naive_required = sum(
        item.stat().st_size
        for folder in (images, masks, model)
        for item in folder.rglob("*")
        if item.is_file()
    )
    building = target.with_name(target.name + ".building")
    target.replace(building)
    remaining: list[int] = []
    monkeypatch.setattr(
        "gsdb.postshot._check_disk_space",
        lambda _target, required: remaining.append(required),
    )
    assert prepare_postshot_dataset(
        scene, _run(run_id), output=target, resume=True
    )["reused"] is False
    assert remaining and remaining[0] < naive_required

    _write_image(target / "images" / "image_00000007.jpg", 33)
    with pytest.raises(RuntimeError, match="hash"):
        validate_postshot_dataset(target)


def test_partial_postshot_dataset_requires_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    run_id = "run-001"
    dataset = scene / "work" / run_id / "reconstruction-primary"
    model = dataset / "model"
    model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(COLMAP_COUNT.pack(0))
    (dataset / "images").mkdir()
    (dataset / "masks").mkdir()
    monkeypatch.setattr("gsdb.postshot._selected_model_dir", lambda _: model)
    monkeypatch.setattr("gsdb.postshot._windows_argument", lambda path: str(path))
    target = tmp_path / "postshot"
    target.with_name("postshot.building").mkdir()

    with pytest.raises(RuntimeError, match="use --resume"):
        prepare_postshot_dataset(scene, _run(run_id), output=target)


def test_postshot_command_uses_imported_poses_masks_and_quality_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("gsdb.postshot._windows_argument", lambda path: str(path))
    command = build_postshot_train_command(
        tmp_path / "postshot-cli.exe",
        tmp_path / "dataset",
        tmp_path / "model.psht",
    )
    assert command[1] == "train"
    assert command[command.index("--profile") + 1] == "Splat3"
    assert command[command.index("--image-select") + 1] == "all"
    assert command[command.index("--mask-mode") + 1] == "occluders"
    assert command[command.index("--max-image-size") + 1] == "0"
    assert "--store-training-context" in command
    assert str(tmp_path / "dataset/colmap") in command

    with pytest.raises(ValueError, match=r"\.psht"):
        build_postshot_train_command(
            tmp_path / "postshot-cli.exe",
            tmp_path / "dataset",
            tmp_path / "model.bin",
        )
    with pytest.raises(ValueError, match="only one"):
        build_postshot_train_command(
            tmp_path / "postshot-cli.exe",
            tmp_path / "dataset",
            tmp_path / "model.psht",
            export_ply=tmp_path / "model.ply",
            export_spz=tmp_path / "model.spz",
        )
    with pytest.raises(ValueError, match=r"\.ply"):
        build_postshot_train_command(
            tmp_path / "postshot-cli.exe",
            tmp_path / "dataset",
            tmp_path / "model.psht",
            export_ply=tmp_path / "mislabeled.spz",
        )
    with pytest.raises(ValueError, match=r"\.spz"):
        build_postshot_train_command(
            tmp_path / "postshot-cli.exe",
            tmp_path / "dataset",
            tmp_path / "model.psht",
            export_spz=tmp_path / "mislabeled.ply",
        )

    first_log, first_manifest = postshot._training_sidecars(tmp_path / "first.psht")
    second_log, second_manifest = postshot._training_sidecars(tmp_path / "second.psht")
    assert first_log != second_log
    assert first_manifest != second_manifest


def test_postshot_rejects_excluded_image_in_selected_model(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write_images_binary(
        model / "images.bin",
        [(7, "view_00/frame_000001.jpg", [])],
    )
    with pytest.raises(RuntimeError, match="Excluded masks leaked"):
        postshot._validate_colmap_exclusions(
            model,
            {"excluded_images": ["view_00/frame_000001.jpg"]},
        )


def test_postshot_dry_run_does_not_create_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    run = _run("run-001")
    dataset = scene / "run-001/postshot"
    dataset.mkdir(parents=True)
    (dataset / "dataset.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "gsdb.postshot.validate_postshot_dataset",
        lambda _: {"source": {"selected_dataset": run.selected_dataset}},
    )
    monkeypatch.setattr("gsdb.postshot._selected_model_dir", lambda _: tmp_path / "model")
    monkeypatch.setattr("gsdb.postshot._validate_postshot_source_lineage", lambda *args: None)
    executable = tmp_path / "postshot-cli.exe"
    executable.write_bytes(b"fake")
    monkeypatch.setattr("gsdb.postshot.postshot_executable", lambda: executable)
    monkeypatch.setattr("gsdb.postshot.postshot_version", lambda _: ((1, 1, 69), "1.1.69"))
    monkeypatch.setattr("gsdb.postshot.available_vram_mib", lambda _: 12000)
    monkeypatch.setattr("gsdb.postshot._windows_argument", lambda path: str(path))
    target = scene / "run-001/postshot-training/model.psht"
    result = train_postshot(scene, run, dry_run=True, output=target)
    assert result["dry_run"] is True
    assert not target.exists()
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing project")
    with pytest.raises(FileExistsError, match="already exists"):
        train_postshot(scene, run, dry_run=True, output=target)


def test_postshot_training_failure_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    run = _run("run-001")
    dataset = tmp_path / "native-dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("gsdb.postshot.validate_postshot_dataset", lambda _: {})
    monkeypatch.setattr("gsdb.postshot._selected_model_dir", lambda _: tmp_path / "model")
    monkeypatch.setattr("gsdb.postshot._validate_postshot_source_lineage", lambda *args: None)
    executable = tmp_path / "postshot-cli.exe"
    executable.write_bytes(b"fake")
    monkeypatch.setattr("gsdb.postshot.postshot_executable", lambda: executable)
    monkeypatch.setattr("gsdb.postshot.postshot_version", lambda _: ((1, 1, 69), "1.1.69"))
    monkeypatch.setattr("gsdb.postshot.available_vram_mib", lambda _: 12000)
    monkeypatch.setattr("gsdb.postshot._windows_argument", lambda path: str(path))
    target = tmp_path / "training/model.psht"

    def fail(command, log_path, **kwargs):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("failed", encoding="utf-8")
        raise CommandError(list(command), 7, log_path, {"elapsed_seconds": 1.0})

    monkeypatch.setattr("gsdb.postshot.run_logged", fail)
    with pytest.raises(CommandError):
        train_postshot(scene, run, dataset=dataset, output=target)
    payload = __import__("json").loads(
        (target.parent / "training.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["metrics"]["elapsed_seconds"] == 1.0


def test_postshot_missing_project_preserves_completed_process_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    run = _run("run-001")
    dataset = tmp_path / "native-dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("gsdb.postshot.validate_postshot_dataset", lambda _: {})
    monkeypatch.setattr("gsdb.postshot._selected_model_dir", lambda _: tmp_path / "model")
    monkeypatch.setattr("gsdb.postshot._validate_postshot_source_lineage", lambda *args: None)
    executable = tmp_path / "postshot-cli.exe"
    executable.write_bytes(b"fake")
    monkeypatch.setattr("gsdb.postshot.postshot_executable", lambda: executable)
    monkeypatch.setattr("gsdb.postshot.postshot_version", lambda _: ((1, 1, 69), "1.1.69"))
    monkeypatch.setattr("gsdb.postshot.available_vram_mib", lambda _: 12000)
    monkeypatch.setattr("gsdb.postshot._windows_argument", lambda path: str(path))
    target = tmp_path / "training/model.psht"

    def finish_without_project(command, log_path, **kwargs):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("finished", encoding="utf-8")
        return {"elapsed_seconds": 2.0, "gpu_memory_peak_mib": 4096.0}

    monkeypatch.setattr("gsdb.postshot.run_logged", finish_without_project)
    with pytest.raises(RuntimeError, match="without creating"):
        train_postshot(scene, run, dataset=dataset, output=target)
    payload = __import__("json").loads(
        (target.parent / "training.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["metrics"]["elapsed_seconds"] == 2.0
    assert payload["metrics"]["gpu_memory_peak_mib"] == 4096.0
