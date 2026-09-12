from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb.mask_finalize import finalize_mask_dataset, validate_mask_finalization
from gsdb.mask_review import MaskReviewDataset


def _dataset(path: Path) -> Path:
    image_path = path / "images/view_00/frame_000001.jpg"
    mask_path = path / "masks/view_00/frame_000001.jpg.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image_path), np.full((16, 16, 3), 80, np.uint8))
    assert cv2.imwrite(str(mask_path), np.full((16, 16), 255, np.uint8))
    return path


def test_mask_review_becomes_stale_after_external_mask_edit(tmp_path: Path) -> None:
    path = _dataset(tmp_path / "dataset")
    review = MaskReviewDataset(path)
    review.update_review(1, "ok", "checked")
    assert review.review_for(1)["status"] == "ok"
    mask = np.full((16, 16), 255, np.uint8)
    mask[0, 0] = 0
    assert cv2.imwrite(str(path / "masks/view_00/frame_000001.jpg.png"), mask)
    assert MaskReviewDataset(path).review_for(1)["status"] == "stale"
    with pytest.raises(RuntimeError, match="stale"):
        finalize_mask_dataset(path)


def test_mask_finalize_allows_unreviewed_and_propagates_exclude(tmp_path: Path) -> None:
    path = _dataset(tmp_path / "dataset")
    review = MaskReviewDataset(path)
    review.update_review(1, "exclude", "moving tripod")
    result = finalize_mask_dataset(path)
    assert result["excluded_images"] == ["view_00/frame_000001.jpg"]
    assert validate_mask_finalization(path)["status"] == "passed"

    final_path = path / "mask-final.json"
    tampered = json.loads(final_path.read_text(encoding="utf-8"))
    tampered["mask_sha256"] = {}
    final_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mask_sha256"):
        validate_mask_finalization(path)


def test_mask_finalize_blocks_unresolved_issue(tmp_path: Path) -> None:
    path = _dataset(tmp_path / "dataset")
    MaskReviewDataset(path).update_review(1, "false_negative", "person")
    with pytest.raises(RuntimeError, match="false_negative"):
        finalize_mask_dataset(path)


def test_mask_finalize_requires_complete_expected_inventory(tmp_path: Path) -> None:
    path = _dataset(tmp_path / "dataset")
    expected = {
        "view_00/frame_000001.jpg",
        "view_01/frame_000001.jpg",
    }
    with pytest.raises(RuntimeError, match="inventory is incomplete"):
        finalize_mask_dataset(path, expected)


def test_mask_finalize_rejects_nonbinary_or_wrong_size_mask(tmp_path: Path) -> None:
    path = _dataset(tmp_path / "dataset")
    mask_path = path / "masks/view_00/frame_000001.jpg.png"
    assert cv2.imwrite(str(mask_path), np.full((16, 16), 127, np.uint8))
    with pytest.raises(RuntimeError, match="not binary"):
        finalize_mask_dataset(path)

    assert cv2.imwrite(str(mask_path), np.full((8, 8), 255, np.uint8))
    with pytest.raises(RuntimeError, match="size mismatch"):
        finalize_mask_dataset(path)


def test_mask_finalize_requires_explicit_exclusion_after_edit_above_v4_threshold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset"
    image_path = path / "images/view_00/frame_000001.jpg"
    mask_path = path / "masks/view_00/frame_000001.jpg.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image_path), np.full((20, 20, 3), 80, np.uint8))
    mask = np.full((20, 20), 255, np.uint8)
    mask.flat[:21] = 0
    assert cv2.imwrite(str(mask_path), mask)

    with pytest.raises(RuntimeError, match="explicitly exclude"):
        finalize_mask_dataset(
            path,
            {"view_00/frame_000001.jpg"},
            maximum_included_masked_fraction=0.05,
        )

    MaskReviewDataset(path).update_review(1, "exclude", "above 5% after edit")
    result = finalize_mask_dataset(
        path,
        {"view_00/frame_000001.jpg"},
        maximum_included_masked_fraction=0.05,
    )
    assert result["excluded_images"] == ["view_00/frame_000001.jpg"]
