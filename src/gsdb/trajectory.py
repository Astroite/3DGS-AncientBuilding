from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .manifests import canonical_hash
from .media import sha256_file


FRAME_PATTERN = re.compile(r"frame_(\d+)")


def _rotation_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def analyze_trajectory(
    samples: Iterable[dict[str, Any]], expected_frames: Iterable[int] | None = None
) -> dict[str, Any]:
    values = list(samples)
    blocking: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    seen_timestamps: set[float] = set()
    previous_timestamp: float | None = None
    normalized: list[dict[str, Any]] = []
    for position, sample in enumerate(values):
        try:
            frame = int(sample["frame"])
            timestamp = float(sample["timestamp_seconds"])
            center = np.asarray(sample["center"], dtype=np.float64)
            rotation = np.asarray(sample["rotation"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            blocking.append({"type": "missing_pose", "index": position, "detail": str(error)})
            continue
        if center.shape != (3,) or rotation.shape != (3, 3):
            blocking.append({"type": "missing_pose", "frame": frame})
            continue
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(rotation)) or not math.isfinite(timestamp):
            blocking.append({"type": "non_finite_pose", "frame": frame})
            continue
        if frame in seen_frames or timestamp in seen_timestamps:
            blocking.append({"type": "duplicate_frame", "frame": frame, "timestamp_seconds": timestamp})
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            blocking.append({"type": "time_not_monotonic", "frame": frame, "timestamp_seconds": timestamp})
        seen_frames.add(frame)
        seen_timestamps.add(timestamp)
        previous_timestamp = timestamp
        normalized.append(
            {
                "frame": frame,
                "timestamp_seconds": timestamp,
                "center": center,
                "rotation": rotation,
            }
        )

    expected = set(int(frame) for frame in (expected_frames or []))
    if expected:
        missing = sorted(expected - seen_frames)
        unexpected = sorted(seen_frames - expected)
        if missing:
            blocking.append({"type": "missing_frames", "frames": missing})
        if unexpected:
            blocking.append({"type": "unexpected_frames", "frames": unexpected})

    steps = [
        float(np.linalg.norm(second["center"] - first["center"]))
        for first, second in zip(normalized, normalized[1:])
    ]
    nonzero_steps = [value for value in steps if value > 1e-12]
    median_step = float(np.median(nonzero_steps)) if nonzero_steps else 0.0
    rotations = [
        _rotation_angle_degrees(first["rotation"], second["rotation"])
        for first, second in zip(normalized, normalized[1:])
    ]
    nonzero_rotations = [value for value in rotations if value > 1e-9]
    median_rotation = (
        float(np.median(nonzero_rotations)) if nonzero_rotations else 0.0
    )
    heights = [float(item["center"][1]) for item in normalized]
    height_steps = [abs(second - first) for first, second in zip(heights, heights[1:])]
    nonzero_heights = [value for value in height_steps if value > 1e-12]
    median_height_step = (
        float(np.median(nonzero_heights)) if nonzero_heights else 0.0
    )
    for index, distance in enumerate(steps, start=1):
        if median_step > 0 and distance > median_step * 10.0:
            blocking.append(
                {
                    "type": "teleport",
                    "from_frame": normalized[index - 1]["frame"],
                    "to_frame": normalized[index]["frame"],
                    "distance": distance,
                    "median_nonzero_step": median_step,
                    "ratio": distance / median_step,
                }
            )
    for index, angle in enumerate(rotations, start=1):
        limit = max(90.0, median_rotation * 10.0)
        if angle > limit:
            warnings.append(
                {
                    "type": "rotation_jump",
                    "from_frame": normalized[index - 1]["frame"],
                    "to_frame": normalized[index]["frame"],
                    "degrees": angle,
                }
            )
    for index, delta in enumerate(height_steps, start=1):
        limit = max(median_step * 3.0, median_height_step * 10.0)
        if limit > 0 and delta > limit:
            warnings.append(
                {
                    "type": "height_jump",
                    "from_frame": normalized[index - 1]["frame"],
                    "to_frame": normalized[index]["frame"],
                    "delta": delta,
                }
            )
    separation = max(20, len(normalized) // 10)
    proximity = median_step * 2.0
    if proximity > 0:
        for first_index in range(len(normalized)):
            for second_index in range(first_index + separation, len(normalized)):
                distance = float(
                    np.linalg.norm(
                        normalized[second_index]["center"]
                        - normalized[first_index]["center"]
                    )
                )
                if distance <= proximity:
                    warnings.append(
                        {
                            "type": "possible_loop",
                            "from_frame": normalized[first_index]["frame"],
                            "to_frame": normalized[second_index]["frame"],
                            "distance": distance,
                        }
                    )
    return {
        "schema_version": 1,
        "status": "failed" if blocking else "passed",
        "sample_count": len(normalized),
        "expected_frame_count": len(expected) if expected else None,
        "median_nonzero_step": median_step,
        "median_rotation_degrees": median_rotation,
        "blocking": blocking,
        "warnings": warnings,
        "samples": [
            {
                "frame": item["frame"],
                "timestamp_seconds": item["timestamp_seconds"],
                "center": [float(value) for value in item["center"]],
            }
            for item in normalized
        ],
    }


def samples_from_transforms(
    path: Path, timestamps_by_frame: dict[int, float] | None = None
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for item in payload.get("frames", []):
        match = FRAME_PATTERN.search(str(item.get("file_path", "")))
        matrix = np.asarray(item.get("transform_matrix"), dtype=np.float64)
        if match is None or matrix.shape != (4, 4):
            raise RuntimeError("Trajectory frame is missing a frame number or 4x4 pose")
        rotation = matrix[:3, :3]
        if not np.all(np.isfinite(matrix)):
            # Preserve the sample so QA can emit a durable blocking report, but
            # propagate non-finiteness from any part of the full 4x4 pose.
            rotation = np.full((3, 3), np.nan, dtype=np.float64)
        grouped.setdefault(int(match.group(1)), []).append((matrix[:3, 3], rotation))
    if not grouped:
        raise RuntimeError("No trajectory poses were found in transforms.json")
    samples = []
    for frame, poses in sorted(grouped.items()):
        centers = np.asarray([item[0] for item in poses])
        rotation = poses[0][1]
        if any(
            not np.all(np.isfinite(center)) or not np.all(np.isfinite(item_rotation))
            for center, item_rotation in poses
        ):
            rotation = np.full((3, 3), np.nan, dtype=np.float64)
        samples.append(
            {
                "frame": frame,
                "timestamp_seconds": float(
                    (timestamps_by_frame or {}).get(frame, frame)
                ),
                "center": np.mean(centers, axis=0),
                "rotation": rotation,
            }
        )
    return samples


def timestamps_from_metrics(path: Path) -> dict[int, float]:
    if not path.is_file():
        raise RuntimeError(f"Selected-frame metrics are missing: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        index: float(record["timestamp_seconds"])
        for index, record in enumerate(records, start=1)
    }


def _trajectory_lineage(
    dataset: Path,
    timestamps_by_frame: dict[int, float],
    expected_frames: Iterable[int],
) -> dict[str, Any]:
    expected = sorted(set(int(frame) for frame in expected_frames))
    timestamps = {
        str(frame): float(timestamps_by_frame[frame])
        for frame in expected
        if frame in timestamps_by_frame
    }
    return {
        "transforms_sha256": sha256_file(dataset / "transforms.json"),
        "timestamps_sha256": canonical_hash(timestamps),
        "expected_frames_sha256": canonical_hash(expected),
        "expected_frames": expected,
    }


def trajectory_svg(report: dict[str, Any]) -> str:
    samples = report.get("samples", [])
    points = [(float(item["center"][0]), float(item["center"][2])) for item in samples]
    width = 900
    height = 600
    padding = 35
    if points:
        xs, ys = zip(*points)
        x_span = max(max(xs) - min(xs), 1e-9)
        y_span = max(max(ys) - min(ys), 1e-9)
        scale = min((width - 2 * padding) / x_span, (height - 2 * padding) / y_span)
        plotted = [
            (
                padding + (x - min(xs)) * scale,
                height - padding - (y - min(ys)) * scale,
            )
            for x, y in points
        ]
    else:
        plotted = []
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in plotted)
    status = html.escape(str(report.get("status", "unknown")))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#111827"/>\n'
        f'<text x="24" y="28" fill="#e5e7eb" font-family="sans-serif">Trajectory QA: {status}</text>\n'
        f'<polyline points="{polyline}" fill="none" stroke="#38bdf8" stroke-width="2"/>\n'
        + "".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#f8fafc"/>\n'
            for x, y in plotted
        )
        + "</svg>\n"
    )


def write_trajectory_qa(
    dataset: Path,
    timestamps_by_frame: dict[int, float] | None = None,
    expected_frames: Iterable[int] | None = None,
) -> dict[str, Any]:
    timestamps = timestamps_by_frame or {}
    expected = list(expected_frames or sorted(timestamps))
    report = analyze_trajectory(
        samples_from_transforms(
            dataset / "transforms.json", timestamps_by_frame=timestamps
        ),
        expected_frames=expected,
    )
    missing_timestamps = sorted(set(expected) - set(timestamps))
    if missing_timestamps:
        report["blocking"].append(
            {"type": "missing_timestamps", "frames": missing_timestamps}
        )
        report["status"] = "failed"
    report["lineage"] = _trajectory_lineage(dataset, timestamps, expected)
    (dataset / "trajectory-qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (dataset / "trajectory-top.svg").write_text(
        trajectory_svg(report), encoding="utf-8"
    )
    return report


def validate_trajectory_qa(
    dataset: Path,
    timestamps_by_frame: dict[int, float],
    expected_frames: Iterable[int],
) -> dict[str, Any]:
    report_path = dataset / "trajectory-qa.json"
    if not report_path.is_file():
        raise RuntimeError(f"Trajectory QA report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    current = _trajectory_lineage(dataset, timestamps_by_frame, expected_frames)
    if report.get("lineage") != current:
        raise RuntimeError("Trajectory QA is stale; reconstruct the run before training")
    if report.get("status") != "passed" or report.get("blocking"):
        raise RuntimeError("Training is blocked until trajectory QA passes")
    return report
