from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb.masking import (
    PersonPrediction,
    colmap_mask_from_ignored,
    generate_person_masks,
    mask_path_for_image,
    postprocess_person_mask,
    validate_mask_set,
)
from gsdb.models import MaskingConfig


def _write_image(path: Path, shape: tuple[int, int] = (32, 32)) -> None:
    image = np.full((*shape, 3), 100, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_person_mask_is_black_for_both_consumers(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    image_path = images / "frame_000001_0.jpg"
    _write_image(image_path)
    config = MaskingConfig(dilation_pixels=1, closing_pixels=0)

    def predictor(image: np.ndarray) -> PersonPrediction:
        prediction = np.zeros(image.shape[:2], dtype=bool)
        prediction[14:18, 14:18] = True
        return PersonPrediction(prediction, 1)

    records = generate_person_masks(
        images, masks, config, tmp_path / "metrics.jsonl", predictor=predictor
    )
    stored = cv2.imread(
        str(mask_path_for_image(masks, image_path)), cv2.IMREAD_GRAYSCALE
    )
    assert stored is not None
    assert stored[16, 16] == 0
    assert stored[0, 0] == 255
    assert records[0]["detections"] == 1
    assert not list(masks.glob(".*.tmp"))
    summary = validate_mask_set(images, masks, max_masked_fraction=0.45)
    assert summary["deterministic_qa"] == "passed"
    assert summary["masked_images"] == 1


def test_mask_resume_reuses_existing_file(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "view.jpg"
    _write_image(image_path)
    assert cv2.imwrite(
        str(mask_path_for_image(masks, image_path)),
        colmap_mask_from_ignored(np.zeros((32, 32), dtype=bool)),
    )

    def must_not_run(_: np.ndarray) -> PersonPrediction:
        raise AssertionError("predictor should not run for an existing valid mask")

    records = generate_person_masks(
        images,
        masks,
        MaskingConfig(dilation_pixels=0, closing_pixels=0),
        tmp_path / "metrics.jsonl",
        predictor=must_not_run,
    )
    assert records[0]["reused"] is True


def test_deterministic_qa_rejects_destructive_mask(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "view.jpg"
    _write_image(image_path)
    assert cv2.imwrite(
        str(mask_path_for_image(masks, image_path)), np.zeros((32, 32), dtype=np.uint8)
    )
    with pytest.raises(RuntimeError, match="above the configured"):
        validate_mask_set(images, masks, max_masked_fraction=0.45)


def test_postprocess_rejects_non_2d_mask() -> None:
    with pytest.raises(ValueError, match="2D"):
        postprocess_person_mask(
            np.zeros((4, 4, 1), dtype=np.uint8),
            MaskingConfig(dilation_pixels=0, closing_pixels=0),
        )


def test_nested_view_masks_mirror_image_relative_paths(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    view = images / "view_03"
    view.mkdir(parents=True)
    image_path = view / "frame_000007.jpg"
    _write_image(image_path)

    records = generate_person_masks(
        images,
        masks,
        MaskingConfig(enabled=False, dilation_pixels=0, closing_pixels=0),
        tmp_path / "metrics.jsonl",
    )

    assert records[0]["image"] == "view_03/frame_000007.jpg"
    assert records[0]["mask"] == "view_03/frame_000007.jpg.png"
    assert (masks / "view_03" / "frame_000007.jpg.png").is_file()
    assert validate_mask_set(images, masks, 0.45)["mask_count"] == 1
