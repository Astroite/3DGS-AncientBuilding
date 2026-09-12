from __future__ import annotations

from .gpu_lock import gpu_locked

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2

from .manifests import canonical_hash
from .media import (
    extract_indexed_frames,
    probe_video,
    sha256_file,
    validate_equirectangular,
)
from .models import CaptureManifestV2, SourceProbe
from .paths import host_path, wsl_to_windows


MEDIA_PROTOCOL_VERSION = 1
MEDIA_HELPER_ENV = "GSDB_MEDIA_HELPER"
MEDIA_SDK_ROOT_ENV = "INSTA360_MEDIA_SDK_ROOT"
ALLOW_FAKE_HELPER_ENV = "GSDB_ALLOW_FAKE_MEDIA_HELPER"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PUBLIC_SUPPORTED_CAMERAS = {
    "insta360 one x",
    "insta360 one r",
    "insta360 one rs",
    "insta360 one x2",
    "insta360 x3",
    "insta360 x4",
    "insta360 x4 air",
    "insta360 x5",
    "insta360 x6",
}


class MediaHelperError(RuntimeError):
    """A structured failure reported by the MediaSDK helper itself."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"MediaSDK {code}: {message}")


def _camera_key(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().casefold())
    return normalized.removeprefix("insta360 ")


@dataclass(frozen=True)
class CandidateFrameSet:
    path: Path
    manifest_path: Path
    dataset_sha256: str
    frame_paths: tuple[Path, ...]
    frame_sha256s: tuple[str, ...]
    frame_indices: tuple[int, ...]
    timestamps_seconds: tuple[float, ...]
    helper_version: str | None = None
    sdk_version: str | None = None
    source_probe: dict[str, Any] | None = None
    schema_version: int = 1
    candidate_fps: float | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _windows_visible(path: Path) -> str:
    absolute = path.absolute()
    value = wsl_to_windows(str(absolute))
    if os.name != "nt" and value == str(absolute):
        raise RuntimeError(
            f"MediaSDK request/output must be on a Windows-mounted project path: {path}"
        )
    if value.startswith("\\\\"):
        raise RuntimeError("MediaSDK request/output may not use a UNC path")
    return value


def _helper_visible(path: Path, helper: Path | None = None) -> str:
    selected = helper or media_helper_path()
    return str(path.absolute()) if selected.suffix.lower() == ".py" else _windows_visible(path)


def media_helper_path() -> Path:
    value = os.environ.get(MEDIA_HELPER_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"{MEDIA_HELPER_ENV} is not configured; legacy video and image-sequence "
            "inputs remain available"
        )
    path = host_path(value)
    if not path.is_file():
        raise RuntimeError(f"Configured MediaSDK helper is missing: {path}")
    if path.suffix.lower() == ".py":
        if os.environ.get(ALLOW_FAKE_HELPER_ENV) != "1":
            raise RuntimeError(
                "Python MediaSDK helpers are test-only; set GSDB_MEDIA_HELPER to the "
                "approved Windows x64 .exe"
            )
    elif path.suffix.lower() != ".exe":
        raise RuntimeError("GSDB_MEDIA_HELPER must be a Windows x64 .exe")
    return path


def invoke_media_helper(
    operation: str,
    payload: dict[str, Any],
    protocol_dir: Path,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    if operation not in {"capabilities", "probe", "export_frames"}:
        raise ValueError(f"Unsupported MediaSDK helper operation: {operation}")
    helper = media_helper_path()
    protocol_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    request_path = protocol_dir / f"request-{token}.json"
    response_path = protocol_dir / f"response-{token}.json"
    request = {
        "schema_version": MEDIA_PROTOCOL_VERSION,
        "operation": operation,
        **payload,
    }
    _atomic_json(request_path, request)
    try:
        command = [
                *([sys.executable] if helper.suffix.lower() == ".py" else []),
                str(helper),
                _helper_visible(request_path, helper),
                _helper_visible(response_path, helper),
            ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        if not response_path.is_file():
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"MediaSDK helper exited with {result.returncode} without a response"
                + (f": {detail}" if detail else "")
            )
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("MediaSDK helper response must be a JSON object")
        if response.get("schema_version") != MEDIA_PROTOCOL_VERSION:
            raise RuntimeError(
                f"Unsupported MediaSDK helper protocol: {response.get('schema_version')}"
            )
        for field in ("helper_version", "sdk_version"):
            if not isinstance(response.get(field), str) or not response[field].strip():
                raise RuntimeError(f"MediaSDK helper response omitted {field}")
        supported_models = response.get("supported_camera_models")
        if not isinstance(supported_models, list) or any(
            not isinstance(item, str) or not item.strip() for item in supported_models
        ):
            raise RuntimeError(
                "MediaSDK helper response has invalid supported_camera_models"
            )
        if response.get("ok", False) and not supported_models:
            raise RuntimeError("MediaSDK helper reports no supported camera models")
        if result.returncode != 0 or not response.get("ok", False):
            error = response.get("error") or {}
            code = error.get("code", "helper_failed")
            message = error.get("message", "MediaSDK helper failed")
            raise MediaHelperError(str(code), str(message))
        return response
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)


def media_capabilities(protocol_dir: Path) -> dict[str, Any]:
    return invoke_media_helper("capabilities", {}, protocol_dir)


def _sequence_index(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def ordered_sequence_files(paths: Iterable[Path], fps: float | None = None) -> list[Path]:
    files = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise RuntimeError("Image sequence contains no supported images")
    indexed = [(_sequence_index(path), path) for path in files]
    if any(index is None for index, _ in indexed):
        if fps is None:
            raise ValueError("Image filenames have no numeric index; provide --fps")
        return sorted(files, key=lambda path: path.name.casefold())
    indices = [int(index) for index, _ in indexed if index is not None]
    if len(set(indices)) != len(indices):
        raise ValueError("Image sequence contains duplicate numeric indices")
    return [path for _, path in sorted(indexed, key=lambda item: (int(item[0]), item[1].name))]


def probe_image_sequence(paths: Iterable[Path], fps: float | None = None) -> dict[str, Any]:
    ordered = ordered_sequence_files(paths, fps=fps)
    dimensions: set[tuple[int, int]] = set()
    for path in ordered:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode image sequence member: {path}")
        dimensions.add((int(image.shape[1]), int(image.shape[0])))
    if len(dimensions) != 1:
        raise ValueError(f"Image sequence dimensions are inconsistent: {sorted(dimensions)}")
    width, height = next(iter(dimensions))
    effective_fps = float(fps or 1.0)
    probe = {
        "width": width,
        "height": height,
        "fps": effective_fps,
        "frame_count": len(ordered),
        "duration_seconds": len(ordered) / effective_fps,
        "codec": "image-sequence",
    }
    validate_equirectangular(probe)
    return probe


def source_paths(capture: CaptureManifestV2) -> list[Path]:
    paths = [host_path(item.windows_path) for item in capture.source.files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Capture source files are missing: {missing}")
    return paths


def validate_source_fingerprints(capture: CaptureManifestV2) -> list[Path]:
    paths = source_paths(capture)
    for record, path in zip(capture.source.files, paths):
        if path.stat().st_size != record.byte_size or sha256_file(path) != record.sha256:
            raise RuntimeError(f"Capture source changed after registration: {path}")
    return paths


class SourceAdapter:
    """Normalize one capture source into a deterministic CandidateFrameSet."""

    kind: str

    def probe(
        self, capture: CaptureManifestV2, paths: list[Path], protocol_dir: Path
    ) -> dict[str, Any]:
        raise NotImplementedError

    def export(
        self,
        capture: CaptureManifestV2,
        paths: list[Path],
        probe: dict[str, Any],
        indices: list[int],
        output_paths: list[Path],
        building: Path,
        protocol_dir: Path,
        width: int,
        height: int,
        resume: bool,
    ) -> None:
        raise NotImplementedError


class EquirectVideoAdapter(SourceAdapter):
    kind = "equirect_video"

    def probe(
        self, capture: CaptureManifestV2, paths: list[Path], protocol_dir: Path
    ) -> dict[str, Any]:
        probe = probe_video(paths[0])
        validate_equirectangular(probe)
        frame_count = int(
            probe.get("frame_count")
            or round(float(probe["duration_seconds"]) * float(probe["fps"] or 0))
        )
        return {**probe, "frame_count": max(1, frame_count)}

    def export(
        self,
        capture: CaptureManifestV2,
        paths: list[Path],
        probe: dict[str, Any],
        indices: list[int],
        output_paths: list[Path],
        building: Path,
        protocol_dir: Path,
        width: int,
        height: int,
        resume: bool,
    ) -> None:
        valid: set[Path] = set()
        for path in output_paths:
            if not path.is_file():
                continue
            try:
                _validate_frame(path, width, height)
                valid.add(path)
            except RuntimeError:
                pass
        missing = [path for path in output_paths if path not in valid]
        if not missing:
            return
        if valid and not resume:
            raise RuntimeError("Partial video frame extraction exists; use --resume")

        # FFmpeg's legacy extraction command emits a complete, uniformly sampled
        # sequence.  During resume, stage that sequence separately and replace only
        # the absent or invalid destinations so already-verified candidates remain
        # byte-for-byte untouched.
        with tempfile.TemporaryDirectory(prefix=".video-extract-", dir=building) as temporary_dir:
            extracted_dir = Path(temporary_dir) / "frames"
            extract_indexed_frames(
                paths[0],
                extracted_dir,
                indices,
                capture.normalization.jpeg_quality,
                building / "export.log",
            )
            for destination in missing:
                extracted = extracted_dir / destination.name
                if (int(probe["width"]), int(probe["height"])) == (width, height):
                    os.replace(extracted, destination)
                    continue
                normalized = extracted.with_suffix(".normalized.jpg")
                _normalize_image(
                    extracted,
                    normalized,
                    width,
                    height,
                    capture.normalization.jpeg_quality,
                )
                os.replace(normalized, destination)


class EquirectSequenceAdapter(SourceAdapter):
    kind = "equirect_sequence"

    def probe(
        self, capture: CaptureManifestV2, paths: list[Path], protocol_dir: Path
    ) -> dict[str, Any]:
        return probe_image_sequence(paths, fps=capture.source.probe.fps)

    def export(
        self,
        capture: CaptureManifestV2,
        paths: list[Path],
        probe: dict[str, Any],
        indices: list[int],
        output_paths: list[Path],
        building: Path,
        protocol_dir: Path,
        width: int,
        height: int,
        resume: bool,
    ) -> None:
        ordered = ordered_sequence_files(paths, fps=float(probe["fps"]))
        for source_index, destination in zip(indices, output_paths):
            if destination.is_file():
                try:
                    _validate_frame(destination, width, height)
                    continue
                except RuntimeError:
                    destination.unlink()
            _normalize_image(
                ordered[source_index],
                destination,
                width,
                height,
                capture.normalization.jpeg_quality,
            )


class Insta360InsvAdapter(SourceAdapter):
    kind = "insta360_insv"

    def probe(
        self, capture: CaptureManifestV2, paths: list[Path], protocol_dir: Path
    ) -> dict[str, Any]:
        response = invoke_media_helper(
            "probe",
            {"source_files": [_helper_visible(path) for path in paths]},
            protocol_dir,
        )
        media = dict(response.get("media") or {})
        camera_model = str(media.get("camera_model") or capture.camera.model).strip()
        supported = {
            _camera_key(str(item))
            for item in response.get("supported_camera_models", [])
        }
        public_supported = {_camera_key(item) for item in PUBLIC_SUPPORTED_CAMERAS}
        declared_key = _camera_key(capture.camera.model)
        detected_key = _camera_key(camera_model)
        if detected_key not in supported or detected_key not in public_supported:
            raise RuntimeError(
                f"MediaSDK unsupported_camera: {camera_model}; use Insta360 Studio to export "
                "a 2:1 equirectangular video"
            )
        if declared_key not in {"", "unknown"} and declared_key != detected_key:
            raise RuntimeError(
                f"MediaSDK camera_mismatch: capture declares {capture.camera.model}, "
                f"but helper detected {camera_model}"
            )
        required = ("width", "height", "fps", "frame_count", "duration_seconds")
        if any(media.get(name) is None for name in required):
            raise RuntimeError(f"MediaSDK probe omitted required fields: {required}")
        media["helper_version"] = response.get("helper_version")
        media["sdk_version"] = response.get("sdk_version")
        media["camera_model"] = camera_model
        return media

    def export(
        self,
        capture: CaptureManifestV2,
        paths: list[Path],
        probe: dict[str, Any],
        indices: list[int],
        output_paths: list[Path],
        building: Path,
        protocol_dir: Path,
        width: int,
        height: int,
        resume: bool,
    ) -> None:
        missing = []
        requested_outputs: list[Path] = []
        for index, destination in zip(indices, output_paths):
            if destination.is_file():
                try:
                    _validate_frame(destination, width, height)
                    continue
                except RuntimeError:
                    destination.unlink()
            missing.append(
                {
                    "source_frame_index": index,
                    "output_file": _helper_visible(destination),
                }
            )
            requested_outputs.append(destination)
        if not missing:
            return
        response = invoke_media_helper(
            "export_frames",
            {
                "source_files": [_helper_visible(path) for path in paths],
                "frames": missing,
                "output": {
                    "width": width,
                    "height": height,
                    "format": "jpeg",
                    "jpeg_quality": capture.normalization.jpeg_quality,
                },
                "stitching": {
                    "mode": "dynamic",
                    "flowstate": True,
                    "direction_lock": False,
                    "ai_denoise": False,
                    "sharpening": False,
                    "color_processing": False,
                },
            },
            protocol_dir,
            timeout_seconds=3600,
        )
        validation_error: str | None = None
        if int(response.get("exported_count", -1)) != len(missing):
            validation_error = "MediaSDK helper exported an unexpected number of frames"
        elif response.get("helper_version") != probe.get("helper_version"):
            validation_error = "MediaSDK helper version changed between probe and export"
        elif response.get("sdk_version") != probe.get("sdk_version"):
            validation_error = "MediaSDK version changed between probe and export"
        if validation_error is not None:
            # The response cannot prove which bytes were produced under the
            # probed implementation. Never let a later --resume bless them.
            for path in requested_outputs:
                path.unlink(missing_ok=True)
            raise RuntimeError(validation_error)


SOURCE_ADAPTERS: dict[str, SourceAdapter] = {
    adapter.kind: adapter
    for adapter in (
        EquirectVideoAdapter(),
        EquirectSequenceAdapter(),
        Insta360InsvAdapter(),
    )
}


def source_adapter(capture: CaptureManifestV2) -> SourceAdapter:
    return SOURCE_ADAPTERS[capture.source.kind]


def probe_capture_source(capture: CaptureManifestV2, protocol_dir: Path) -> dict[str, Any]:
    paths = validate_source_fingerprints(capture)
    result = source_adapter(capture).probe(capture, paths, protocol_dir)
    validate_source_fingerprints(capture)
    return result


def uniform_frame_indices(
    frame_count: int,
    fps: float,
    start_seconds: float,
    end_seconds: float,
    target_frames: int,
) -> list[int]:
    if target_frames < 2:
        raise ValueError("At least two candidate frames are required")
    start = max(0, math.ceil(start_seconds * fps))
    stop = min(frame_count, math.ceil(end_seconds * fps))
    available = stop - start
    if available < target_frames:
        raise ValueError(
            f"Selection contains {available} source frames, fewer than target {target_frames}"
        )
    return [
        min(stop - 1, start + math.floor((index + 0.5) * available / target_frames))
        for index in range(target_frames)
    ]


def fixed_rate_frame_indices(
    frame_count: int,
    source_fps: float,
    start_seconds: float,
    end_seconds: float,
    candidate_fps: float,
) -> tuple[list[int], list[float]]:
    """Sample unique source frames at the centres of fixed-rate time cells."""

    if candidate_fps <= 0:
        raise ValueError("candidate_fps must be positive")
    if source_fps + 1e-9 < candidate_fps:
        raise ValueError(
            f"Source frame rate {source_fps:.6g} is below requested candidate rate "
            f"{candidate_fps:.6g}; duplicate source frames are not allowed"
        )
    if end_seconds <= start_seconds:
        raise ValueError("Selection end must be after its start")
    step = 1.0 / candidate_fps
    requested: list[float] = []
    sample = 0
    while True:
        timestamp = start_seconds + (sample + 0.5) * step
        if timestamp >= end_seconds - 1e-12:
            break
        requested.append(timestamp)
        sample += 1
    indices = [
        # Nearest source frame, resolving exact half-frame ties toward the
        # earlier timestamp so a source running exactly at candidate_fps keeps
        # each sample inside its intended one-second bucket.
        min(
            frame_count - 1,
            max(0, int(math.floor(timestamp * source_fps + 0.5 - 1e-12))),
        )
        for timestamp in requested
    ]
    if len(indices) < 2:
        raise ValueError("Selection contains fewer than two fixed-rate candidate frames")
    if len(set(indices)) != len(indices):
        raise ValueError(
            "Fixed-rate sampling mapped multiple timestamps to the same source frame"
        )
    return indices, [index / source_fps for index in indices]


def _validate_frame(path: Path, width: int, height: int) -> str:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xff\xd8\xff":
            raise RuntimeError(f"Prepared frame is not a JPEG file: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode prepared frame: {path}")
    if image.shape[1] != width or image.shape[0] != height:
        raise RuntimeError(
            f"Prepared frame has size {image.shape[1]}x{image.shape[0]}, expected {width}x{height}"
        )
    return sha256_file(path)


def _normalize_image(source: Path, destination: Path, width: int, height: int, quality: int) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode source image: {source}")
    source_height, source_width = image.shape[:2]
    if width > source_width or height > source_height:
        raise ValueError(
            f"Normalization may not upscale {source_width}x{source_height} to {width}x{height}"
        )
    if (source_width, source_height) != (width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ):
        raise RuntimeError(f"Cannot write normalized image: {destination}")


def _dataset_from_manifest(path: Path) -> CandidateFrameSet:
    manifest_path = path / "dataset.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0))
    if schema_version not in {1, 2} or payload.get("integrity") != "complete":
        raise RuntimeError(f"Prepared input is incomplete: {manifest_path}")
    records = payload.get("frames") or []
    lineage = payload["lineage"]
    if records != lineage.get("frames"):
        raise RuntimeError(f"Prepared input frame lineage is inconsistent: {manifest_path}")
    expected_count = (
        lineage.get("target_frames")
        if schema_version == 1
        else lineage.get("candidate_frame_count")
    )
    if len(records) != int(expected_count or -1):
        raise RuntimeError(f"Prepared input frame count is invalid: {manifest_path}")
    preparation = {key: value for key, value in lineage.items() if key != "frames"}
    preparation_hash = canonical_hash(preparation)
    directory_hash = path.name.removesuffix(".building")
    if (
        payload.get("preparation_hash") != preparation_hash
        or directory_hash != preparation_hash
    ):
        raise RuntimeError(f"Prepared input preparation hash is invalid: {manifest_path}")
    relative_files = []
    for record in records:
        relative = Path(str(record["file"]))
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
            raise RuntimeError(f"Prepared input contains an unsafe frame path: {relative}")
        relative_files.append(str(relative))
    if len(set(relative_files)) != len(relative_files):
        raise RuntimeError(f"Prepared input contains duplicate frame paths: {manifest_path}")
    frames = tuple(path / relative_file for relative_file in relative_files)
    for frame, record in zip(frames, payload["frames"]):
        if not frame.is_file():
            raise RuntimeError(f"Prepared input frame is missing or corrupt: {frame}")
        actual_sha256 = _validate_frame(frame, int(record["width"]), int(record["height"]))
        if actual_sha256 != record["sha256"]:
            raise RuntimeError(f"Prepared input frame is missing or corrupt: {frame}")
    expected = canonical_hash(lineage)
    if expected != payload.get("dataset_sha256"):
        raise RuntimeError("Prepared input dataset hash is invalid")
    return CandidateFrameSet(
        path=path,
        manifest_path=manifest_path,
        dataset_sha256=expected,
        frame_paths=frames,
        frame_sha256s=tuple(str(item["sha256"]) for item in records),
        frame_indices=tuple(int(item["source_frame_index"]) for item in payload["frames"]),
        timestamps_seconds=tuple(float(item["timestamp_seconds"]) for item in payload["frames"]),
        helper_version=lineage.get("helper_version"),
        sdk_version=lineage.get("sdk_version"),
        source_probe=dict(lineage["source_probe"]),
        schema_version=schema_version,
        candidate_fps=(
            float(lineage["sampling"]["candidate_fps"])
            if schema_version == 2
            else None
        ),
    )


def _trusted_cached_frame_hashes(path: Path, preparation_hash: str) -> dict[str, str]:
    """Read expected frame hashes without accepting any current frame bytes."""

    manifest_path = path / "dataset.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        lineage = payload["lineage"]
        records = payload["frames"]
        if (
            payload.get("schema_version") not in {1, 2}
            or payload.get("integrity") != "complete"
            or payload.get("preparation_hash") != preparation_hash
            or payload.get("dataset_sha256") != canonical_hash(lineage)
            or records != lineage.get("frames")
        ):
            return {}
        trusted: dict[str, str] = {}
        for record in records:
            relative = Path(str(record["file"]))
            if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
                return {}
            digest = str(record["sha256"])
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                return {}
            trusted[relative.as_posix()] = digest
        return trusted
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


@gpu_locked
def prepare_capture_input(
    scene_path: Path,
    capture: CaptureManifestV2,
    target_frames: int | None = None,
    candidate_fps: float | None = None,
    selection_end_seconds: float | None = None,
    resume: bool = False,
) -> CandidateFrameSet:
    protocol_dir = scene_path / "prepared" / ".protocol"
    probe = probe_capture_source(capture, protocol_dir)
    paths = source_paths(capture)
    effective_end = min(
        capture.selection.end_seconds,
        selection_end_seconds
        if selection_end_seconds is not None
        else capture.selection.end_seconds,
    )
    if effective_end > float(probe["duration_seconds"]) + 0.5:
        raise ValueError(
            f"Selected range ends at {effective_end:.3f}s but source is "
            f"{float(probe['duration_seconds']):.3f}s"
        )
    if (target_frames is None) == (candidate_fps is None):
        raise ValueError("Provide exactly one of target_frames or candidate_fps")
    if candidate_fps is not None:
        indices, timestamps = fixed_rate_frame_indices(
            int(probe["frame_count"]),
            float(probe["fps"]),
            capture.selection.start_seconds,
            effective_end,
            candidate_fps,
        )
        manifest_schema_version = 2
    else:
        assert target_frames is not None
        indices = uniform_frame_indices(
            int(probe["frame_count"]),
            float(probe["fps"]),
            capture.selection.start_seconds,
            effective_end,
            target_frames,
        )
        timestamps = [index / float(probe["fps"]) for index in indices]
        manifest_schema_version = 1
    width = int(capture.normalization.width or probe["width"])
    height = int(capture.normalization.height or probe["height"])
    if width != height * 2:
        raise ValueError(f"Prepared panorama must be 2:1, got {width}x{height}")
    if width > int(probe["width"]) or height > int(probe["height"]):
        raise ValueError("Normalization dimensions may not exceed source panorama dimensions")
    preparation = {
        "capture_id": capture.id,
        "source_kind": capture.source.kind,
        "source_files": [item.model_dump(mode="json") for item in capture.source.files],
        "selection": {
            "start_seconds": capture.selection.start_seconds,
            "end_seconds": effective_end,
        },
        "normalization": capture.normalization.model_dump(mode="json"),
        "source_probe": SourceProbe(
            **{
                name: value
                for name, value in probe.items()
                if name in SourceProbe.model_fields
            }
        ).model_dump(mode="json"),
        "source_frame_indices": indices,
        "helper_version": probe.get("helper_version"),
        "sdk_version": probe.get("sdk_version"),
    }
    if manifest_schema_version == 1:
        preparation["target_frames"] = len(indices)
    else:
        preparation["sampling"] = {
            "mode": "fixed_rate_v1",
            "candidate_fps": candidate_fps,
        }
        preparation["candidate_frame_count"] = len(indices)
    preparation_hash = canonical_hash(preparation)
    target = scene_path / "prepared" / capture.id / preparation_hash
    if (target / "dataset.json").is_file():
        try:
            return _dataset_from_manifest(target)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            if not resume:
                raise RuntimeError(
                    f"Prepared input cache is invalid; use --resume to repair it: {target}"
                ) from error
    required_bytes = len(indices) * width * height + (1 << 30)
    disk_path = scene_path
    while not disk_path.exists() and disk_path.parent != disk_path:
        disk_path = disk_path.parent
    free_bytes = shutil.disk_usage(disk_path).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient Windows-project disk space for prepared input: "
            f"{free_bytes / 1024**3:.1f} GiB free, "
            f"{required_bytes / 1024**3:.1f} GiB conservatively required"
        )
    building = target.with_name(target.name + ".building")
    if target.exists():
        if not resume:
            raise RuntimeError(f"Partial prepared input exists; use --resume: {target}")
        if building.exists():
            raise RuntimeError(
                f"Both invalid prepared input and partial repair exist; inspect them before resuming: "
                f"{target}, {building}"
            )
        target.replace(building)
    if building.exists() and not resume:
        raise RuntimeError(f"Partial prepared input exists; use --resume: {building}")
    frames_dir = building / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [frames_dir / f"frame_{index + 1:06d}.jpg" for index in range(len(indices))]

    # A completed cache carries trusted hashes.  On resume, invalidate only bytes
    # that no longer match that lineage; adapters then regenerate those candidates.
    if (building / "dataset.json").is_file():
        trusted_hashes = _trusted_cached_frame_hashes(building, preparation_hash)
        for output_path in output_paths:
            relative = output_path.relative_to(building).as_posix()
            if (
                output_path.is_file()
                and trusted_hashes.get(relative) != sha256_file(output_path)
            ):
                output_path.unlink()

    try:
        source_adapter(capture).export(
            capture,
            paths,
            probe,
            indices,
            output_paths,
            building,
            protocol_dir,
            width,
            height,
            resume,
        )
        # A native helper is an external process. Re-hash the registered source
        # before publishing the cache so even an accidental write to an INSV is
        # detected and can never enter prepared-input lineage unnoticed.
        validate_source_fingerprints(capture)

        frame_records = []
        for index, timestamp, path in zip(indices, timestamps, output_paths):
            frame_records.append(
                {
                    "file": path.relative_to(building).as_posix(),
                    "source_frame_index": index,
                    "timestamp_seconds": round(timestamp, 6),
                    "width": width,
                    "height": height,
                    "sha256": _validate_frame(path, width, height),
                }
            )
    except Exception as error:
        # Only an explicit helper interruption is trusted as resumable. Protocol
        # drift, malformed media, source mutation, and generic adapter failures
        # must not leave bytes that a later --resume could silently bless.
        if not isinstance(error, MediaHelperError) or error.code != "interrupted_export":
            shutil.rmtree(building, ignore_errors=True)
        raise
    lineage = {**preparation, "frames": frame_records}
    dataset_sha256 = canonical_hash(lineage)
    manifest = {
        "schema_version": manifest_schema_version,
        "generated_at": _utc_now(),
        "preparation_hash": preparation_hash,
        "dataset_sha256": dataset_sha256,
        "lineage": lineage,
        "frames": frame_records,
        "integrity": "complete",
    }
    _atomic_json(building / "dataset.json", manifest)
    _dataset_from_manifest(building)
    target.parent.mkdir(parents=True, exist_ok=True)
    building.replace(target)
    return _dataset_from_manifest(target)


def copy_candidate_frames(
    candidate_set: CandidateFrameSet, output_dir: Path, resume: bool = False
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"frame_{index + 1:06d}.jpg" for index in range(len(candidate_set.frame_paths))]
    invalid: list[tuple[Path, Path, str]] = []
    for source, destination, expected_sha256 in zip(
        candidate_set.frame_paths, outputs, candidate_set.frame_sha256s
    ):
        if sha256_file(source) != expected_sha256:
            raise RuntimeError(f"Prepared candidate changed after verification: {source}")
        if destination.is_file() and sha256_file(destination) == expected_sha256:
            continue
        invalid.append((source, destination, expected_sha256))
    if invalid and any(path.is_file() for path in outputs) and not resume:
        raise RuntimeError("Partial or corrupt candidate-frame directory exists; use --resume")
    for source, destination, expected_sha256 in invalid:
        temporary = destination.with_suffix(f".copying-{uuid.uuid4().hex}.jpg")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Candidate copy verification failed: {destination}")
        os.replace(temporary, destination)
    return outputs
