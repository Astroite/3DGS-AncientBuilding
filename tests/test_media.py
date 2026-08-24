from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb.media import (
    analyze_frames,
    create_blur_aware_subset,
    select_blur_aware_records,
    summarize_frame_metrics,
    validate_equirectangular,
)


def test_equirectangular_validation() -> None:
    validate_equirectangular({"width": 7680, "height": 3840, "duration_seconds": 90.7})
    with pytest.raises(ValueError):
        validate_equirectangular({"width": 3840, "height": 2160, "duration_seconds": 10})


def test_frame_metrics_and_blur_aware_subset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        if index % 2:
            image[:, ::2] = 255
        assert cv2.imwrite(str(source / f"frame_{index + 1:06d}.jpg"), image)
    frames = sorted(source.glob("*.jpg"))
    metrics = analyze_frames(frames, 6.0, tmp_path / "metrics.jsonl", start_seconds=10.0)
    assert metrics[0]["timestamp_seconds"] == 10.5
    assert metrics[-1]["timestamp_seconds"] == 15.5
    summary = summarize_frame_metrics(metrics)
    assert summary["frame_count"] == 6
    subset = create_blur_aware_subset(source, metrics, tmp_path / "subset", 3)
    assert len(subset) == 3
    assert all(item.is_file() for item in subset)
    selected = select_blur_aware_records(metrics, 3)
    assert [item["file"] for item in selected] == [
        "frame_000002.jpg",
        "frame_000004.jpg",
        "frame_000006.jpg",
    ]
