from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb import media
from gsdb.media import (
    add_selection_scores,
    analyze_frames,
    create_blur_aware_subset,
    extract_indexed_frames,
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
            image[8:56, 16:112] = 255
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


def test_indexed_video_extraction_uses_exact_source_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    def fake_run_logged(command: list[str], log_path: Path) -> dict[str, float]:
        captured.extend(command)
        pattern = Path(command[-1])
        for index in range(1, 4):
            pattern.with_name(f"frame_{index:06d}.jpg").write_bytes(b"jpeg")
        return {}

    monkeypatch.setattr(media, "run_logged", fake_run_logged)
    frames = extract_indexed_frames(
        tmp_path / "input.mp4", tmp_path / "frames", [3, 17, 42], 95, tmp_path / "ffmpeg.log"
    )
    assert len(frames) == 3
    selection = captured[captured.index("-vf") + 1]
    assert selection == r"select=eq(n\,3)+eq(n\,17)+eq(n\,42)"
    assert captured[captured.index("-q:v") + 1] == "2"


def test_degenerate_tenengrad_does_not_fall_back_to_laplacian() -> None:
    records = [
        {
            "file": f"frame_{index:06d}.jpg",
            "timestamp_seconds": float(index),
            "tenengrad": 0.0,
            "blur_laplacian_variance": float(index * 100),
            "black_fraction": 0.0,
            "highlight_fraction": 0.0,
        }
        for index in range(4)
    ]
    add_selection_scores(records)
    assert [item["file"] for item in select_blur_aware_records(records, 2)] == [
        "frame_000000.jpg",
        "frame_000002.jpg",
    ]
