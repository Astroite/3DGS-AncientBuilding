import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gsdb.models import ReconstructionAttempt, ReconstructionConfig
from gsdb.reconstruction import _write_colmap_binary_model
from gsdb.reconstruction_realityscan import (
    build_realityscan_align_command,
    realityscan_reconstruction_metrics,
    run_realityscan_alignment,
)


def _attempt() -> ReconstructionAttempt:
    return ReconstructionAttempt(
        frame_count=2,
        images_per_equirect=8,
        projection_fov_degrees=120.0,
        projection_size=2048,
        crop_bottom=0.2,
    )


def test_realityscan_command_fixes_generated_pinhole_intrinsics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gsdb.reconstruction_realityscan._windows_argument", lambda path: str(path)
    )
    monkeypatch.setattr(
        "gsdb.reconstruction_realityscan.export_params_file",
        lambda: tmp_path / "params.xml",
    )

    command = build_realityscan_align_command(
        Path("RealityScan.exe"),
        tmp_path / "images",
        tmp_path / "crash",
        tmp_path / "export/images.txt",
        tmp_path / "project.rsproj",
        projection_fov_degrees=120.0,
    )

    edits = {
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "-editInputSelection"
    }
    assert "inpCalibrationGroup=0" in edits
    assert "inpCalibration=2" in edits
    assert "inpFocal=10.3923048454" in edits
    assert "inpPPX=0" in edits
    assert "inpPPY=0" in edits
    assert "inpLensGroup=0" in edits
    assert "inpDistortion=2" in edits
    assert "inpDistortionModel=0" in edits
    assert command.index("-deselectAllImages") < command.index("-align")
    selection_prompt_index = command.index("PUS-1-323292754=1")
    prompt_index = command.index("PUS-1-323301212=1")
    export_index = command.index("-exportRegistration")
    assert command[selection_prompt_index - 1] == "-set"
    assert command[prompt_index - 1] == "-set"
    assert selection_prompt_index < export_index
    assert prompt_index < export_index


def _write_realityscan_dataset(dataset: Path) -> None:
    names = [
        "view_00/frame_000001.jpg",
        "view_01/frame_000001.jpg",
    ]
    for name in names:
        image = dataset / "images" / name
        mask = dataset / "masks" / f"{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        mask.write_bytes(b"mask")

    camera = SimpleNamespace(
        model="PINHOLE",
        width=2048,
        height=2048,
        params=np.array([1024.0, 1024.0, 1024.0, 1024.0]),
    )
    images = {
        image_id: SimpleNamespace(
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([float(image_id), 0.0, 0.0]),
            camera_id=1,
            name=name,
            xys=np.empty((0, 2)),
            point3D_ids=np.empty(0, dtype=np.int64),
        )
        for image_id, name in enumerate(names, start=1)
    }
    model = dataset / "colmap"
    _write_colmap_binary_model(model, {1: camera}, images, {})
    (model / "selected-attempt.json").write_text(
        json.dumps(
            {
                "model": ".",
                "rig": {"enabled": False, "backend": "realityscan"},
            }
        ),
        encoding="utf-8",
    )
    (dataset / "transforms.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "file_path": f"./images/{name}",
                        "mask_path": f"./masks/{name}.png",
                    }
                    for name in names
                ]
            }
        ),
        encoding="utf-8",
    )
    (dataset / "mask-final.json").write_text(
        json.dumps({"excluded_images": ["view_07/frame_000002.jpg"]}),
        encoding="utf-8",
    )


def test_realityscan_reconstruction_metrics_reads_flat_single_component(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_realityscan_dataset(dataset)

    metrics = realityscan_reconstruction_metrics(
        dataset, _attempt(), ReconstructionConfig()
    )

    assert metrics == {
        "expected_planar_images": 15,
        "written_planar_images": 2,
        "excluded_images": 1,
        "registered_images": 2,
        "selected_component_images": 2,
        "registration_ratio": 2 / 15,
        "largest_component_coverage": 2 / 15,
        "component_count": 1,
        "component_sizes": [2],
        "sift_gpu": True,
        "fixed_intrinsics": True,
        "cross_view_pairs": 0,
        "rig": {"enabled": False, "backend": "realityscan"},
    }


def test_realityscan_metrics_rejects_missing_transforms(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="conversion did not create"):
        realityscan_reconstruction_metrics(tmp_path, _attempt())


def test_realityscan_metrics_wraps_mask_validation_failure(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_realityscan_dataset(dataset)
    (dataset / "masks" / "view_00" / "frame_000001.jpg.png").unlink()

    with pytest.raises(RuntimeError, match="frame-mask validation failed"):
        realityscan_reconstruction_metrics(dataset, _attempt())


def test_realityscan_metrics_rejects_missing_model_selection(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_realityscan_dataset(dataset)
    (dataset / "colmap" / "selected-attempt.json").unlink()

    with pytest.raises(RuntimeError, match="model selection is missing"):
        realityscan_reconstruction_metrics(dataset, _attempt())


def test_realityscan_resume_short_circuits_with_flat_model_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _write_realityscan_dataset(dataset)

    def unexpected_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("RealityScan should not relaunch for a reusable flat model")

    monkeypatch.setattr(
        "gsdb.reconstruction_realityscan.build_realityscan_align_command",
        unexpected_launch,
    )

    metrics = run_realityscan_alignment(
        dataset,
        _attempt(),
        tmp_path / "logs",
        ReconstructionConfig(registration_threshold=0.1),
    )

    assert metrics["registration_ratio"] == 2 / 15


def test_realityscan_removes_text_export_after_successful_binary_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _write_realityscan_dataset(dataset)
    transforms = dataset / "transforms.json"
    selection = dataset / "colmap" / "selected-attempt.json"
    transforms.unlink()
    selection.unlink()

    monkeypatch.setattr(
        "gsdb.reconstruction_realityscan.build_realityscan_align_command",
        lambda *args, **kwargs: ["RealityScan"],
    )
    monkeypatch.setattr(
        "gsdb.reconstruction_realityscan.realityscan_executable",
        lambda: Path("RealityScan.exe"),
    )

    def fake_run_logged(*args: object, **kwargs: object) -> None:
        export = dataset / "colmap" / "realityscan-export"
        export.mkdir(parents=True, exist_ok=True)
        (export / "images.txt").write_text("exported\n", encoding="utf-8")

    monkeypatch.setattr("gsdb.reconstruction_realityscan.run_logged", fake_run_logged)
    monkeypatch.setattr(
        "gsdb.reconstruction_realityscan.convert_colmap_text_to_binary",
        lambda *args, **kwargs: None,
    )

    def fake_colmap_to_json(*args: object, **kwargs: object) -> None:
        transforms.write_text(
            json.dumps(
                {
                    "frames": [
                        {"file_path": "./images/view_00/frame_000001.jpg"},
                        {"file_path": "./images/view_01/frame_000001.jpg"},
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "nerfstudio.process_data.colmap_utils.colmap_to_json", fake_colmap_to_json
    )

    metrics = run_realityscan_alignment(
        dataset, _attempt(), tmp_path / "logs", ReconstructionConfig()
    )

    assert metrics["registered_images"] == 2
    assert not (dataset / "colmap" / "realityscan-export").exists()


def test_realityscan_uses_hardlinked_final_inventory_and_dynamic_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _write_realityscan_dataset(dataset)
    (dataset / "transforms.json").unlink()
    (dataset / "colmap" / "selected-attempt.json").unlink()
    included = {"view_00/frame_000001.jpg"}
    captured: dict[str, Path] = {}

    def build(*args: object, **kwargs: object) -> list[str]:
        images_dir = Path(args[1])
        captured["images_dir"] = images_dir
        staged_files = list(images_dir.iterdir())
        assert len(staged_files) == 1
        staged = staged_files[0]
        assert staged.name == "gsdb_000000.jpg"
        assert staged.is_file()
        assert staged.samefile(dataset / "images/view_00/frame_000001.jpg")
        return ["RealityScan"]

    monkeypatch.setattr(
        "gsdb.reconstruction_realityscan.build_realityscan_align_command", build
    )
    monkeypatch.setattr(
        "gsdb.reconstruction_realityscan.realityscan_executable",
        lambda: Path("RealityScan.exe"),
    )

    def fake_run_logged(*args: object, **kwargs: object) -> None:
        export = dataset / "colmap/realityscan-export"
        export.mkdir(parents=True, exist_ok=True)
        (export / "images.txt").write_text("exported\n", encoding="utf-8")

    monkeypatch.setattr("gsdb.reconstruction_realityscan.run_logged", fake_run_logged)

    def fake_convert(*args: object, **kwargs: object) -> None:
        assert kwargs["image_name_map"] == {
            "gsdb_000000.jpg": "view_00/frame_000001.jpg"
        }
        camera = SimpleNamespace(
            model="PINHOLE",
            width=2048,
            height=2048,
            params=np.array([1024.0, 1024.0, 1024.0, 1024.0]),
        )
        image = SimpleNamespace(
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 0.0]),
            camera_id=1,
            name="view_00/frame_000001.jpg",
            xys=np.empty((0, 2)),
            point3D_ids=np.empty(0, dtype=np.int64),
        )
        _write_colmap_binary_model(dataset / "colmap", {1: camera}, {1: image}, {})

    monkeypatch.setattr(
        "gsdb.reconstruction_realityscan.convert_colmap_text_to_binary", fake_convert
    )

    def fake_colmap_to_json(*args: object, **kwargs: object) -> None:
        (dataset / "transforms.json").write_text(
            json.dumps(
                {"frames": [{"file_path": "./images/view_00/frame_000001.jpg"}]}
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "nerfstudio.process_data.colmap_utils.colmap_to_json", fake_colmap_to_json
    )

    metrics = run_realityscan_alignment(
        dataset,
        _attempt(),
        tmp_path / "logs",
        ReconstructionConfig(),
        included_images=included,
        projected_image_count=2,
    )

    assert metrics["expected_planar_images"] == 1
    assert metrics["written_planar_images"] == 2
    assert metrics["excluded_images"] == 1
    assert metrics["registration_ratio"] == 1.0
    assert not captured["images_dir"].exists()
