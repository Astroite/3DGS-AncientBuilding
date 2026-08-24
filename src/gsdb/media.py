from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .processes import run_logged


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _rational(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = json.loads(result.stdout)
    video_streams = [item for item in payload.get("streams", []) if item.get("codec_type") == "video"]
    if len(video_streams) != 1:
        raise ValueError(f"Expected exactly one stitched video stream, found {len(video_streams)}")
    stream = video_streams[0]
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": _rational(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "duration_seconds": float(duration),
        "codec": str(stream.get("codec_name") or stream.get("codec_tag_string") or "unknown"),
        "pixel_format": stream.get("pix_fmt"),
        "byte_size": path.stat().st_size,
    }


def validate_equirectangular(probe: dict[str, Any], tolerance: float = 0.01) -> None:
    width = int(probe["width"])
    height = int(probe["height"])
    if height <= 0 or abs(width / height - 2.0) > tolerance:
        raise ValueError(f"Expected a 2:1 equirectangular video, got {width}x{height}")
    if float(probe["duration_seconds"]) <= 0:
        raise ValueError("Video duration must be positive")


def required_free_bytes(video_size: int, reserve_gib: float) -> int:
    reserve = int(reserve_gib * 1024**3)
    estimated_work = max(video_size * 6, 20 * 1024**3)
    return reserve + estimated_work


def check_disk_budget(path: Path, video_size: int, reserve_gib: float) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    required = required_free_bytes(video_size, reserve_gib)
    if usage.free < required:
        raise RuntimeError(
            f"Insufficient disk space: {usage.free / 1024**3:.1f} GiB free, "
            f"{required / 1024**3:.1f} GiB required by the pilot estimate"
        )
    return {"free_bytes": usage.free, "required_bytes": required}


def extract_uniform_frames(
    video_path: Path,
    output_dir: Path,
    target_frames: int,
    duration_seconds: float,
    jpeg_quality: int,
    log_path: Path,
    start_seconds: float = 0.0,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing:
        if len(existing) != target_frames:
            raise RuntimeError(
                f"Found {len(existing)} existing frames but expected {target_frames}; create a new run"
            )
        return existing
    rate = target_frames / duration_seconds
    run_logged(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(video_path),
            "-ss",
            f"{start_seconds:.6f}",
            "-t",
            f"{duration_seconds:.6f}",
            "-vf",
            f"fps={rate:.12f}",
            "-frames:v",
            str(target_frames),
            "-q:v",
            str(jpeg_quality),
            "-pix_fmt",
            "yuvj420p",
            str(output_dir / "frame_%06d.jpg"),
        ],
        log_path,
    )
    frames = sorted(output_dir.glob("frame_*.jpg"))
    if len(frames) != target_frames:
        raise RuntimeError(f"FFmpeg produced {len(frames)} frames; expected {target_frames}")
    return frames


def _frame_statistics(path: Path, timestamp: float) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {path}")
    height, width = image.shape[:2]
    scale = min(1.0, 1024.0 / max(width, height))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "file": path.name,
        "timestamp_seconds": round(timestamp, 6),
        "width": width,
        "height": height,
        "blur_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "luma_mean": float(gray.mean()),
        "black_fraction": float(np.mean(gray <= 8)),
        "highlight_fraction": float(np.mean(gray >= 247)),
    }


def analyze_frames(
    frames: list[Path],
    duration_seconds: float,
    output_jsonl: Path,
    start_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    if output_jsonl.is_file():
        records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line]
        if len(records) == len(frames):
            return records
    records = [
        _frame_statistics(
            frame,
            start_seconds + (index + 0.5) * duration_seconds / len(frames),
        )
        for index, frame in enumerate(frames)
    ]
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    return records


def summarize_frame_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    def median(field: str) -> float:
        return float(np.median([float(item[field]) for item in records]))

    return {
        "frame_count": len(records),
        "blur_median": median("blur_laplacian_variance"),
        "luma_mean_median": median("luma_mean"),
        "black_fraction_median": median("black_fraction"),
        "highlight_fraction_median": median("highlight_fraction"),
    }


def select_blur_aware_records(
    metrics: list[dict[str, Any]], target_frames: int
) -> list[dict[str, Any]]:
    """Pick the sharpest frame in each of ``target_frames`` temporal buckets."""
    if target_frames > len(metrics):
        raise ValueError("Subset target cannot exceed the candidate frame count")
    if target_frames < 2:
        raise ValueError("Subset target must contain at least two frames")
    selected: list[dict[str, Any]] = []
    total = len(metrics)
    for index in range(target_frames):
        start = math.floor(index * total / target_frames)
        end = max(start + 1, math.floor((index + 1) * total / target_frames))
        bucket = metrics[start:end]
        selected.append(max(bucket, key=lambda item: float(item["blur_laplacian_variance"])))
    return selected


def create_blur_aware_subset(
    source_dir: Path,
    metrics: list[dict[str, Any]],
    output_dir: Path,
    target_frames: int,
) -> list[Path]:
    selected = select_blur_aware_records(metrics, target_frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if len(existing) == target_frames:
        return existing
    if existing:
        raise RuntimeError("Partial fallback frame directory exists; create a new run")
    outputs: list[Path] = []
    for index, item in enumerate(selected, start=1):
        source = source_dir / item["file"]
        destination = output_dir / f"frame_{index:06d}.jpg"
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        outputs.append(destination)
    return outputs
