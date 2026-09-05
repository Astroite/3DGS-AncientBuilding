from __future__ import annotations

import numpy as np
import pytest

from gsdb.trajectory import (
    analyze_trajectory,
    trajectory_svg,
    validate_trajectory_qa,
    write_trajectory_qa,
)


def _sample(frame: int, x: float) -> dict[str, object]:
    return {
        "frame": frame,
        "timestamp_seconds": float(frame),
        "center": [x, 0, 0],
        "rotation": np.eye(3),
    }


def test_straight_trajectory_passes_and_svg_is_generated() -> None:
    report = analyze_trajectory(_sample(index, float(index)) for index in range(30))
    assert report["status"] == "passed"
    assert "<svg" in trajectory_svg(report)


def test_teleport_is_blocking() -> None:
    samples = [_sample(index, float(index)) for index in range(12)]
    samples[-1] = _sample(11, 100)
    report = analyze_trajectory(samples)
    assert report["status"] == "failed"
    assert any(item["type"] == "teleport" for item in report["blocking"])


def test_far_time_near_position_is_only_a_loop_warning() -> None:
    samples = [
        {
            **_sample(index, 0),
            "center": [
                float(np.cos(index * 2 * np.pi / 39)),
                0,
                float(np.sin(index * 2 * np.pi / 39)),
            ],
        }
        for index in range(40)
    ]
    report = analyze_trajectory(samples)
    assert report["status"] == "passed"
    assert any(item["type"] == "possible_loop" for item in report["warnings"])
    assert not any(item["type"] == "possible_loop" for item in report["blocking"])


def test_missing_expected_frame_is_blocking() -> None:
    report = analyze_trajectory(
        [_sample(1, 0), _sample(3, 1)], expected_frames=[1, 2, 3]
    )
    assert report["status"] == "failed"
    assert {"type": "missing_frames", "frames": [2]} in report["blocking"]


def test_trajectory_lineage_becomes_stale_when_poses_change(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    transforms = {
        "frames": [
            {
                "file_path": f"images/view_00/frame_{index:06d}.jpg",
                "transform_matrix": [
                    [1, 0, 0, float(index)],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
            for index in range(1, 4)
        ]
    }
    import json

    path = dataset / "transforms.json"
    path.write_text(json.dumps(transforms), encoding="utf-8")
    timestamps = {1: 0.0, 2: 1.0, 3: 2.0}
    incomplete = write_trajectory_qa(dataset, {1: 0.0, 3: 2.0}, [1, 2, 3])
    assert incomplete["status"] == "failed"
    assert {"type": "missing_timestamps", "frames": [2]} in incomplete["blocking"]
    write_trajectory_qa(dataset, timestamps, [1, 2, 3])
    validate_trajectory_qa(dataset, timestamps, [1, 2, 3])
    transforms["frames"][2]["transform_matrix"][0][3] = 99
    path.write_text(json.dumps(transforms), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale"):
        validate_trajectory_qa(dataset, timestamps, [1, 2, 3])


def test_nonfinite_pose_in_any_view_is_blocking(tmp_path) -> None:
    import json

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    valid = np.eye(4).tolist()
    invalid = np.eye(4)
    # A non-finite value anywhere in the 4x4 pose, even outside the 3x3
    # rotation/translation slices, must fail the trajectory gate.
    invalid[3, 3] = np.nan
    (dataset / "transforms.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "file_path": "images/view_00/frame_000001.jpg",
                        "transform_matrix": valid,
                    },
                    {
                        "file_path": "images/view_01/frame_000001.jpg",
                        "transform_matrix": invalid.tolist(),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report = write_trajectory_qa(dataset, {1: 0.0}, [1])
    assert report["status"] == "failed"
    assert any(item["type"] == "non_finite_pose" for item in report["blocking"])
