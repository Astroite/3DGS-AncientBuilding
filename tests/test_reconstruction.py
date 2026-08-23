import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb.models import ReconstructionAttempt
from gsdb.reconstruction import (
    attach_frame_masks,
    build_colmap_commands,
    build_image_pyramid,
    pinhole_camera_parameters,
    validate_frame_masks,
)


def test_colmap_commands_use_per_image_masks_cpu_and_projection_intrinsics(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    image_path = dataset / "images" / "frame_000001_0.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    attempt = ReconstructionAttempt(
        frame_count=1, images_per_equirect=8, crop_bottom=0.2
    )
    commands = build_colmap_commands(dataset, attempt)
    feature = commands[0]
    assert feature[feature.index("--ImageReader.mask_path") + 1] == str(dataset / "masks")
    assert feature[feature.index("--ImageReader.camera_model") + 1] == "PINHOLE"
    assert feature[feature.index("--SiftExtraction.use_gpu") + 1] == "0"
    assert commands[1][commands[1].index("--SiftMatching.use_gpu") + 1] == "0"
    assert commands[1][commands[1].index("--SequentialMatching.overlap") + 1] == "16"
    parameters = pinhole_camera_parameters(image_path, 120.0).split(",")
    assert abs(float(parameters[0]) - 28.867513) < 0.001


def test_transforms_receive_matching_per_frame_mask_path(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "frame_000001_0.jpg.png").write_bytes(b"present")
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "file_path": "./images/frame_000001_0.jpg",
                        "transform_matrix": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invalid mask_path"):
        validate_frame_masks(transforms, masks)
    attach_frame_masks(transforms, masks)
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    assert payload["frames"][0]["mask_path"] == "./masks/frame_000001_0.jpg.png"
    assert validate_frame_masks(transforms, masks) == 1
    assert not list(tmp_path.glob(".transforms.json.*.tmp"))


def test_mask_pyramid_keeps_binary_values_and_colmap_filename(tmp_path: Path) -> None:
    source = tmp_path / "masks"
    source.mkdir()
    mask = np.full((16, 16), 255, dtype=np.uint8)
    mask[4:12, 4:12] = 0
    assert cv2.imwrite(str(source / "view.jpg.png"), mask)
    build_image_pyramid(source, tmp_path, "masks", 2, is_mask=True)
    build_image_pyramid(source, tmp_path, "masks", 2, is_mask=True)
    for factor in (2, 4):
        result = cv2.imread(
            str(tmp_path / f"masks_{factor}" / "view.jpg.png"), cv2.IMREAD_GRAYSCALE
        )
        assert result is not None
        assert set(int(value) for value in np.unique(result)).issubset({0, 255})
