from __future__ import annotations

import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence, TypeVar

import cv2
import numpy as np

from .models import MaskingConfig


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_T = TypeVar("_T")


IO_WORKERS_ENV = "GSDB_IO_WORKERS"


def io_worker_count() -> int:
    """Threads used for JPEG/PNG decode, encode and write.

    OpenCV releases the GIL around codec and resize calls, so these stages scale
    with threads while the projection keeps its single CUDA stream serialized.
    """
    override = os.environ.get(IO_WORKERS_ENV, "").strip()
    if override:
        workers = int(override)
        if workers < 1:
            raise ValueError(f"{IO_WORKERS_ENV} must be at least 1, got {workers}")
        return workers
    return max(1, min(8, (os.cpu_count() or 1)))


def run_image_tasks(action: Callable[[_T], Any], items: Sequence[_T]) -> list[Any]:
    """Run ``action`` over ``items``, surfacing the first failure deterministically."""
    workers = min(io_worker_count(), len(items))
    if workers <= 1:
        return [action(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [future.result() for future in [pool.submit(action, item) for item in items]]


@dataclass(frozen=True)
class PersonPrediction:
    mask: np.ndarray
    detections: int


PersonPredictor = Callable[[np.ndarray], PersonPrediction]


def image_files(path: Path) -> list[Path]:
    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda item: item.relative_to(path).as_posix(),
    )


_JPEG_SIZE_MARKERS = frozenset(
    # SOF0-SOF15 carry the frame dimensions; DHT/JPG/DAC share the range and do not.
    value for value in range(0xC0, 0xD0) if value not in (0xC4, 0xC8, 0xCC)
)


def image_dimensions(path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` from the file header without decoding pixels.

    Validating a cached image set with ``cv2.imread`` costs a full JPEG decode per
    file — about 50 ms each over the 9p mount — which dominates every resume. The
    header carries the only property those checks actually test.
    """
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            header = stream.read(17)
            if len(header) < 17 or header[4:8] != b"IHDR":
                raise RuntimeError(f"Truncated PNG header: {path}")
            width, height = struct.unpack(">II", header[8:16])
            return int(height), int(width)
        if not prefix.startswith(b"\xff\xd8"):
            raise RuntimeError(f"Unsupported image header: {path}")
        stream.seek(2)
        while True:
            marker = stream.read(2)
            if len(marker) < 2 or marker[0] != 0xFF:
                raise RuntimeError(f"Truncated JPEG header: {path}")
            if marker[1] in _JPEG_SIZE_MARKERS:
                frame = stream.read(7)
                if len(frame) < 7:
                    raise RuntimeError(f"Truncated JPEG frame header: {path}")
                height, width = struct.unpack(">HH", frame[3:7])
                return int(height), int(width)
            length = stream.read(2)
            if len(length) < 2:
                raise RuntimeError(f"Truncated JPEG segment: {path}")
            stream.seek(struct.unpack(">H", length)[0] - 2, os.SEEK_CUR)


def mask_path_for_image(
    masks_dir: Path, image: Path, images_dir: Path | None = None
) -> Path:
    """COLMAP appends .png to the complete relative image name."""
    images_dir = images_dir or masks_dir.parent / "images"
    try:
        relative = image.relative_to(images_dir)
    except ValueError:
        relative = Path(image.name)
    return masks_dir / relative.parent / f"{relative.name}.png"


def atomic_imwrite(
    target: Path, image: np.ndarray, parameters: list[int] | None = None
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(target.suffix, image, parameters or [])
    if not encoded_ok:
        raise RuntimeError(f"Failed to encode image for {target}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def postprocess_person_mask(mask: np.ndarray, config: MaskingConfig) -> np.ndarray:
    ignored = np.asarray(mask, dtype=np.uint8)
    if ignored.ndim != 2:
        raise ValueError("Person mask must be a 2D array")
    ignored = (ignored > 0).astype(np.uint8)
    if config.closing_pixels > 1:
        kernel = np.ones((config.closing_pixels, config.closing_pixels), np.uint8)
        ignored = cv2.morphologyEx(ignored, cv2.MORPH_CLOSE, kernel)
    if config.dilation_pixels > 0:
        diameter = config.dilation_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        ignored = cv2.dilate(ignored, kernel, iterations=1)
    return ignored.astype(bool)


def colmap_mask_from_ignored(ignored: np.ndarray) -> np.ndarray:
    """Return a mask where white is usable and black is ignored by COLMAP/Nerfstudio."""
    return np.where(np.asarray(ignored, dtype=bool), 0, 255).astype(np.uint8)


def ignored_from_colmap_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Stored mask must be single-channel")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError(f"Stored mask is not binary: {sorted(values)}")
    return mask == 0


class TorchvisionPersonSegmenter:
    """Lazy Mask R-CNN wrapper; importing this module never downloads model weights."""

    def __init__(self, config: MaskingConfig):
        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Masking requested CUDA, but torch.cuda.is_available() is false")
        self._torch = torch
        self._config = config
        self._device = torch.device(config.device)
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._model = maskrcnn_resnet50_fpn_v2(weights=weights).to(self._device).eval()

    def __call__(self, image_bgr: np.ndarray) -> PersonPrediction:
        torch = self._torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self._config.inference_gamma != 1.0:
            rgb = np.power(rgb, self._config.inference_gamma)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self._device)
        with torch.inference_mode():
            output = self._model([tensor])[0]
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        selected = np.flatnonzero(
            (labels == self._config.person_class_id)
            & (scores >= self._config.score_threshold)
        )
        if len(selected) == 0:
            return PersonPrediction(
                mask=np.zeros(image_bgr.shape[:2], dtype=bool), detections=0
            )
        probabilities = output["masks"][selected, 0].detach().cpu().numpy()
        mask = np.any(probabilities >= self._config.probability_threshold, axis=0)
        return PersonPrediction(mask=mask, detections=len(selected))


def _read_existing_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray:
    stored = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if stored is None:
        raise RuntimeError(f"Cannot read existing mask: {mask_path}")
    if stored.shape != shape:
        raise RuntimeError(
            f"Existing mask size mismatch for {mask_path.name}: {stored.shape} != {shape}"
        )
    return ignored_from_colmap_mask(stored)


def generate_person_masks(
    images_dir: Path,
    masks_dir: Path,
    config: MaskingConfig,
    metrics_path: Path,
    predictor: PersonPredictor | None = None,
) -> list[dict[str, object]]:
    images = image_files(images_dir)
    if not images:
        raise RuntimeError(f"No planar images found in {images_dir}")
    masks_dir.mkdir(parents=True, exist_ok=True)
    if config.enabled and predictor is None:
        predictor = TorchvisionPersonSegmenter(config)

    previous_records: dict[str, dict[str, object]] = {}
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                previous = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(previous, dict) and "image" in previous:
                previous_records[str(previous["image"])] = previous
    records: list[dict[str, object]] = []
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_stream:
        for index, image_path in enumerate(images, start=1):
            image_name = image_path.relative_to(images_dir).as_posix()
            target = mask_path_for_image(masks_dir, image_path, images_dir)
            if target.exists():
                # Only the shape is needed to check a cached mask, so skip the decode.
                ignored = _read_existing_mask(target, image_dimensions(image_path))
                detections = previous_records.get(image_name, {}).get("detections")
                reused = True
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Cannot decode planar image: {image_path}")
                prediction = (
                    predictor(image)
                    if predictor is not None
                    else PersonPrediction(np.zeros(image.shape[:2], dtype=bool), 0)
                )
                if prediction.mask.shape != image.shape[:2]:
                    raise RuntimeError(
                        f"Segmenter mask size mismatch for {image_path.name}: "
                        f"{prediction.mask.shape} != {image.shape[:2]}"
                    )
                ignored = postprocess_person_mask(prediction.mask, config)
                detections = prediction.detections
                reused = False
                atomic_imwrite(target, colmap_mask_from_ignored(ignored))
            fraction = float(np.count_nonzero(ignored) / ignored.size)
            record: dict[str, object] = {
                "image": image_name,
                "mask": target.relative_to(masks_dir).as_posix(),
                "detections": detections,
                "masked_fraction": round(fraction, 8),
                "reused": reused,
            }
            records.append(record)
            metrics_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index == 1 or index % 50 == 0 or index == len(images):
                metrics_stream.flush()
                print(f"Person masks: {index}/{len(images)}", flush=True)
    return records


def validate_mask_set(
    images_dir: Path,
    masks_dir: Path,
    max_masked_fraction: float,
) -> dict[str, object]:
    images = image_files(images_dir)
    expected = {
        mask_path_for_image(masks_dir, image, images_dir).relative_to(masks_dir).as_posix(): image
        for image in images
    }
    actual = {
        item.relative_to(masks_dir).as_posix(): item
        for item in masks_dir.rglob("*.png")
        if item.is_file()
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise RuntimeError(f"Mask/image mismatch; missing={missing}, extra={extra}")
    def measure(item: tuple[str, Path]) -> float:
        name, image_path = item
        # The mask is decoded because its pixels drive the ratio; the image only has
        # to agree on shape, which the header already carries.
        mask = cv2.imread(str(actual[name]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot decode image/mask pair: {image_path.name}")
        if mask.shape != image_dimensions(image_path):
            raise RuntimeError(f"Mask size mismatch for {image_path.name}")
        ignored = ignored_from_colmap_mask(mask)
        fraction = float(np.count_nonzero(ignored) / ignored.size)
        if fraction > max_masked_fraction:
            raise RuntimeError(
                f"Mask for {image_path.name} hides {fraction:.1%}, above "
                f"the configured {max_masked_fraction:.1%} ceiling"
            )
        return fraction

    fractions: list[float] = run_image_tasks(measure, list(expected.items()))
    return {
        "image_count": len(images),
        "mask_count": len(actual),
        "masked_images": sum(value > 0 for value in fractions),
        "zero_mask_images": sum(value == 0 for value in fractions),
        "mean_masked_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "max_masked_fraction": max(fractions, default=0.0),
        "deterministic_qa": "passed",
    }


def select_qa_records(
    records: list[dict[str, object]], sample_count: int
) -> list[dict[str, object]]:
    if len(records) <= sample_count:
        return records
    half = max(1, sample_count // 2)
    by_fraction = sorted(records, key=lambda item: float(item["masked_fraction"]), reverse=True)
    chosen = {str(item["image"]): item for item in by_fraction[:half]}
    remaining_slots = sample_count - len(chosen)
    if remaining_slots > 0:
        step = (len(records) - 1) / max(1, remaining_slots - 1)
        for index in range(remaining_slots):
            item = records[round(index * step)]
            chosen[str(item["image"])] = item
    if len(chosen) < sample_count:
        for item in records:
            chosen.setdefault(str(item["image"]), item)
            if len(chosen) == sample_count:
                break
    return sorted(chosen.values(), key=lambda item: str(item["image"]))[:sample_count]


def create_mask_contact_sheets(
    images_dir: Path,
    masks_dir: Path,
    records: list[dict[str, object]],
    output_dir: Path,
    sample_count: int,
    max_sheets: int,
) -> list[Path]:
    selected = select_qa_records(records, min(sample_count, max_sheets * 8))
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[Path] = []
    tile_size = 320
    for sheet_index in range(0, len(selected), 8):
        batch = selected[sheet_index : sheet_index + 8]
        canvas = np.zeros((2 * (tile_size + 34), 4 * tile_size, 3), dtype=np.uint8)
        for tile_index, record in enumerate(batch):
            image_path = images_dir / Path(str(record["image"]))
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(
                str(mask_path_for_image(masks_dir, image_path, images_dir)),
                cv2.IMREAD_GRAYSCALE,
            )
            if image is None or mask is None:
                raise RuntimeError(f"Cannot build QA tile for {image_path.name}")
            ignored = mask == 0
            overlay = image.copy()
            overlay[ignored] = (
                overlay[ignored].astype(np.float32) * 0.35
                + np.array([0, 0, 255], dtype=np.float32) * 0.65
            ).astype(np.uint8)
            overlay = cv2.resize(overlay, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            row, column = divmod(tile_index, 4)
            y = row * (tile_size + 34)
            x = column * tile_size
            canvas[y : y + tile_size, x : x + tile_size] = overlay
            label = f"{record['image']}  mask={float(record['masked_fraction']):.1%}"
            cv2.putText(
                canvas,
                label[:43],
                (x + 4, y + tile_size + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        target = output_dir / f"mask-contact-{len(sheets) + 1:02d}.jpg"
        atomic_imwrite(target, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheets.append(target)
    return sheets


def summarize_detection_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    values = list(records)
    known_detections = [
        int(item["detections"]) for item in values if item.get("detections") is not None
    ]
    return {
        "images": len(values),
        "images_with_detection_metrics": len(known_detections),
        "detection_metrics_complete": len(known_detections) == len(values),
        "images_with_detections": sum(value > 0 for value in known_detections),
        "person_instances": sum(known_detections),
        "reused_masks": sum(bool(item.get("reused")) for item in values),
    }
