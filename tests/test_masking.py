from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb.masking import (
    PersonPrediction,
    colmap_mask_from_ignored,
    generate_person_masks,
    image_dimensions,
    mask_path_for_image,
    postprocess_person_mask,
    validate_mask_set,
)
from gsdb.models import MaskingConfig, MaskingConfigV3


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


def test_multiclass_dynamic_masks_record_counts_and_area(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    image_path = images / "frame_000001.jpg"
    _write_image(image_path)
    config = MaskingConfigV3(
        classes=["person", "car"], dilation_pixels=0, closing_pixels=0
    )

    def predictor(image: np.ndarray) -> PersonPrediction:
        person = np.zeros(image.shape[:2], dtype=bool)
        car = np.zeros(image.shape[:2], dtype=bool)
        person[1:3, 1:3] = True
        car[10:13, 10:13] = True
        return PersonPrediction(
            mask=person | car,
            detections=3,
            detections_by_class={"person": 1, "car": 2},
            masks_by_class={"person": person, "car": car},
        )

    records = generate_person_masks(
        images, masks, config, tmp_path / "metrics.jsonl", predictor=predictor
    )
    assert records[0]["detections"] == 3
    assert records[0]["classes"] == {
        "person": {"detections": 1, "masked_fraction": round(4 / 1024, 8)},
        "car": {"detections": 2, "masked_fraction": round(9 / 1024, 8)},
    }


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


def test_image_dimensions_reads_headers_without_decoding(tmp_path: Path) -> None:
    jpeg = tmp_path / "frame.jpg"
    png = tmp_path / "mask.png"
    assert cv2.imwrite(str(jpeg), np.zeros((48, 96, 3), dtype=np.uint8))
    assert cv2.imwrite(str(png), np.zeros((17, 33), dtype=np.uint8))
    assert image_dimensions(jpeg) == (48, 96)
    assert image_dimensions(png) == (17, 33)


def test_image_dimensions_rejects_truncated_and_foreign_files(tmp_path: Path) -> None:
    jpeg = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(jpeg), np.zeros((48, 96, 3), dtype=np.uint8))
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(jpeg.read_bytes()[:4])
    with pytest.raises(RuntimeError):
        image_dimensions(truncated)
    foreign = tmp_path / "notes.txt"
    foreign.write_bytes(b"not an image at all")
    with pytest.raises(RuntimeError, match="Unsupported image header"):
        image_dimensions(foreign)
