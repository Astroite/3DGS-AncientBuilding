from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Any

import cv2
import numpy as np

from .manifests import canonical_hash
from .masking import (
    IMAGE_SUFFIXES,
    atomic_imwrite,
    image_dimensions,
    image_files,
    io_worker_count,
    mask_path_for_image,
)
from .models import RunManifest, StageStatus
from .paths import ensure_within, ensure_work_dir, host_path, wsl_to_windows
from .mask_finalize import expected_reconstruction_images, validate_mask_finalization
from .processes import CommandError, run_logged
from .reconstruction import _selected_model_dir
from .trajectory import timestamps_from_metrics, validate_trajectory_qa


COLMAP_IMAGE_HEADER = struct.Struct("<i7di")
COLMAP_COUNT = struct.Struct("<Q")
COLMAP_POINT2D_BYTES = 24
COPY_CHUNK_BYTES = 8 << 20
POSTSHOT_SCHEMA_VERSION = 1
POSTSHOT_MINIMUM_VERSION = (1, 1, 69)
POSTSHOT_CLI_ENV = "GSDB_POSTSHOT_CLI"


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    source_name: str
    output_name: str
    point2d_count: int


def default_postshot_dataset(scene_path: Path, run: RunManifest) -> Path:
    """Keep v3 staging on a Windows-visible project volume; preserve legacy paths."""

    if getattr(run.config, "schema_version", 1) == 3:
        return scene_path / "inputs" / "postshot" / run.id
    return ensure_work_dir(scene_path, run.id) / "postshot"


def _validate_v3_quality_gates(
    scene_path: Path, run: RunManifest, dataset: Path
) -> dict[str, Any] | None:
    if getattr(run.config, "schema_version", 1) != 3:
        return None
    selected_label = "fallback" if run.fallback_attempted else "primary"
    attempt = getattr(run.config.reconstruction, selected_label)
    expected = expected_reconstruction_images(
        attempt.frame_count, attempt.images_per_equirect
    )
    mask_final = validate_mask_finalization(dataset, expected)
    timestamps = timestamps_from_metrics(
        ensure_work_dir(scene_path, run.id)
        / f"selected-{selected_label}-metrics.jsonl"
    )
    validate_trajectory_qa(
        dataset, timestamps, range(1, attempt.frame_count + 1)
    )
    return mask_final


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise RuntimeError(f"Truncated COLMAP images.bin while reading {label}")
    return value


def _read_c_string(stream: BinaryIO, label: str) -> str:
    value = bytearray()
    while True:
        character = stream.read(1)
        if not character:
            raise RuntimeError(f"Truncated COLMAP images.bin while reading {label}")
        if character == b"\0":
            break
        value.extend(character)
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"COLMAP image name is not UTF-8: {label}") from error


def _output_image_name(image_id: int, source_name: str) -> str:
    suffix = Path(source_name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise RuntimeError(f"Unsupported registered image extension: {source_name}")
    return f"image_{image_id:08d}{suffix}"


def read_colmap_image_index(path: Path) -> list[ColmapImage]:
    """Read only image IDs/names/counts, skipping the large point-observation payload."""
    file_size = path.stat().st_size
    records: list[ColmapImage] = []
    seen_ids: set[int] = set()
    seen_outputs: set[str] = set()
    with path.open("rb") as stream:
        image_count = COLMAP_COUNT.unpack(_read_exact(stream, 8, "image count"))[0]
        for index in range(image_count):
            fixed = _read_exact(stream, COLMAP_IMAGE_HEADER.size, f"image {index + 1} header")
            values = COLMAP_IMAGE_HEADER.unpack(fixed)
            image_id = int(values[0])
            if image_id <= 0 or image_id in seen_ids:
                raise RuntimeError(f"Invalid or duplicate COLMAP image ID: {image_id}")
            source_name = _read_c_string(stream, f"image {image_id} name")
            point2d_count = COLMAP_COUNT.unpack(
                _read_exact(stream, 8, f"image {image_id} point count")
            )[0]
            next_offset = stream.tell() + point2d_count * COLMAP_POINT2D_BYTES
            if next_offset > file_size:
                raise RuntimeError(
                    f"Truncated COLMAP images.bin in observations for image {image_id}"
                )
            stream.seek(next_offset)
            output_name = _output_image_name(image_id, source_name)
            if output_name in seen_outputs:
                raise RuntimeError(f"Duplicate Postshot image name: {output_name}")
            seen_ids.add(image_id)
            seen_outputs.add(output_name)
            records.append(
                ColmapImage(
                    image_id=image_id,
                    source_name=source_name,
                    output_name=output_name,
                    point2d_count=int(point2d_count),
                )
            )
        if stream.tell() != file_size:
            raise RuntimeError(
                f"COLMAP images.bin has {file_size - stream.tell()} unexpected trailing bytes"
            )
    return records


def _copy_exact(source: BinaryIO, destination: BinaryIO, size: int, label: str) -> None:
    remaining = size
    while remaining:
        block = source.read(min(remaining, COPY_CHUNK_BYTES))
        if not block:
            raise RuntimeError(f"Truncated COLMAP images.bin while copying {label}")
        destination.write(block)
        remaining -= len(block)


def rewrite_colmap_images_binary(
    source_path: Path, destination_path: Path, records: list[ColmapImage]
) -> None:
    """Rewrite only COLMAP image names while retaining poses and observations verbatim."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    try:
        with source_path.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            count_bytes = _read_exact(source, 8, "image count")
            count = COLMAP_COUNT.unpack(count_bytes)[0]
            if count != len(records):
                raise RuntimeError(
                    f"COLMAP image count changed during preparation: {count} != {len(records)}"
                )
            destination.write(count_bytes)
            for expected in records:
                fixed = _read_exact(source, COLMAP_IMAGE_HEADER.size, "image header")
                image_id = int(COLMAP_IMAGE_HEADER.unpack(fixed)[0])
                source_name = _read_c_string(source, f"image {image_id} name")
                if image_id != expected.image_id or source_name != expected.source_name:
                    raise RuntimeError("COLMAP images.bin changed during preparation")
                destination.write(fixed)
                destination.write(expected.output_name.encode("utf-8") + b"\0")
                point_count_bytes = _read_exact(source, 8, f"image {image_id} point count")
                point_count = COLMAP_COUNT.unpack(point_count_bytes)[0]
                if point_count != expected.point2d_count:
                    raise RuntimeError("COLMAP observations changed during preparation")
                destination.write(point_count_bytes)
                _copy_exact(
                    source,
                    destination,
                    point_count * COLMAP_POINT2D_BYTES,
                    f"image {image_id} observations",
                )
            if source.read(1):
                raise RuntimeError("COLMAP images.bin changed during preparation")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, destination_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(COPY_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _model_item_count(path: Path) -> int:
    with path.open("rb") as stream:
        return int(COLMAP_COUNT.unpack(_read_exact(stream, 8, path.name))[0])


def _registered_source_path(images_dir: Path, name: str) -> Path:
    normalized = Path(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise RuntimeError(f"COLMAP image path escapes the image directory: {name}")
    source = images_dir / normalized
    ensure_within(source, images_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Registered image is missing: {source}")
    return source


def _converted_occluder_mask(
    source_mask: Path, shape: tuple[int, int]
) -> tuple[np.ndarray, int]:
    stored = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
    if stored is None:
        raise RuntimeError(f"Cannot decode source mask: {source_mask}")
    if stored.shape != shape:
        raise RuntimeError(
            f"Mask size mismatch for {source_mask}: {stored.shape} != {shape}"
        )
    values = set(int(value) for value in np.unique(stored))
    if not values.issubset({0, 255}):
        raise RuntimeError(
            f"Source mask is not binary: {source_mask}; values={sorted(values)}"
        )
    ignored_pixels = int(np.count_nonzero(stored == 0))
    return np.where(stored == 0, 255, 0).astype(np.uint8), ignored_pixels


def _prepare_registered_image(
    record: ColmapImage,
    source_images: Path,
    source_masks: Path,
    target_images: Path,
    target_masks: Path,
) -> dict[str, Any]:
    source_image = _registered_source_path(source_images, record.source_name)
    source_mask = mask_path_for_image(source_masks, source_image, source_images)
    if not source_mask.is_file():
        raise FileNotFoundError(f"Registered image has no mask: {source_mask}")

    destination_image = target_images / record.output_name
    destination_mask = target_masks / f"{Path(record.output_name).stem}.png"
    shape = image_dimensions(source_image)
    source_image_sha256 = _sha256(source_image)

    if (
        not destination_image.is_file()
        or _sha256(destination_image) != source_image_sha256
    ):
        _atomic_copy(source_image, destination_image)
    if _sha256(destination_image) != source_image_sha256:
        raise RuntimeError(f"Copied image hash changed: {destination_image}")
    if image_dimensions(destination_image) != shape:
        raise RuntimeError(f"Copied image dimensions changed: {destination_image}")

    converted, ignored_pixels = _converted_occluder_mask(source_mask, shape)

    reusable_mask = False
    if destination_mask.is_file():
        existing = cv2.imread(str(destination_mask), cv2.IMREAD_GRAYSCALE)
        reusable_mask = existing is not None and np.array_equal(existing, converted)
    if not reusable_mask:
        atomic_imwrite(destination_mask, converted)

    source_mask_sha256 = _sha256(source_mask)
    output_mask_sha256 = _sha256(destination_mask)

    return {
        "image_id": record.image_id,
        "source_image": record.source_name,
        "output_image": record.output_name,
        "source_mask": source_mask.relative_to(source_masks).as_posix(),
        "output_mask": destination_mask.name,
        "height": shape[0],
        "width": shape[1],
        "ignored_pixels": ignored_pixels,
        "source_image_sha256": source_image_sha256,
        "source_mask_sha256": source_mask_sha256,
        "output_image_sha256": source_image_sha256,
        "output_mask_sha256": output_mask_sha256,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _write_image_map(path: Path, rows: list[dict[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "image_id",
                    "source_image",
                    "output_image",
                    "source_mask",
                    "output_mask",
                    "width",
                    "height",
                    "ignored_pixels",
                    "source_image_sha256",
                    "source_mask_sha256",
                    "output_image_sha256",
                    "output_mask_sha256",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_import_guide(path: Path, dataset: Path, image_count: int) -> None:
    windows_dataset = wsl_to_windows(str(dataset.absolute()))
    content = f"""# Postshot import

Prepared from a rig-constrained COLMAP reconstruction. Use Postshot v1.1.69 or newer.

1. Import `{windows_dataset}\\images` and `{windows_dataset}\\colmap` together.
2. Add `{windows_dataset}\\masks` under Image Masks.
3. Select `Remove Occluders`; white mask pixels are ignored.
4. Confirm Camera Poses is `Import`, Image Selection is `Use All`, and {image_count} images have poses.
5. Do not run camera tracking for this dataset.
"""
    destination = path / "IMPORT.md"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"Cannot find an existing parent for {path}")
        current = current.parent
    return current


def _check_disk_space(target: Path, required_bytes: int) -> None:
    free = shutil.disk_usage(_existing_parent(target)).free
    reserve = 1 << 30
    if free < required_bytes + reserve:
        raise RuntimeError(
            f"Insufficient disk space for Postshot dataset: {free / 1024**3:.1f} GiB free, "
            f"{(required_bytes + reserve) / 1024**3:.1f} GiB required"
        )


def _remaining_postshot_bytes(
    records: list[ColmapImage],
    source_images: Path,
    source_masks: Path,
    source_model: Path,
    building: Path,
) -> int:
    """Estimate bytes still needed, excluding verified resume artifacts."""

    required = 0
    target_images = building / "images"
    target_masks = building / "masks"
    for record in records:
        source_image = _registered_source_path(source_images, record.source_name)
        destination_image = target_images / record.output_name
        if (
            not destination_image.is_file()
            or _sha256(destination_image) != _sha256(source_image)
        ):
            required += source_image.stat().st_size

        source_mask = mask_path_for_image(source_masks, source_image, source_images)
        if not source_mask.is_file():
            raise FileNotFoundError(f"Registered image has no mask: {source_mask}")
        converted, _ = _converted_occluder_mask(
            source_mask, image_dimensions(source_image)
        )
        destination_mask = target_masks / f"{Path(record.output_name).stem}.png"
        existing = (
            cv2.imread(str(destination_mask), cv2.IMREAD_GRAYSCALE)
            if destination_mask.is_file()
            else None
        )
        if existing is None or not np.array_equal(existing, converted):
            # atomic_imwrite temporarily needs a complete replacement beside
            # any old file; an uncompressed byte/pixel estimate is conservative.
            required += max(source_mask.stat().st_size, int(converted.nbytes))

    target_model = building / "colmap"
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        source = source_model / name
        destination = target_model / name
        if (
            name == "images.bin"
            or not destination.is_file()
            or _sha256(destination) != _sha256(source)
        ):
            required += source.stat().st_size
    return required


def validate_postshot_dataset(path: Path) -> dict[str, Any]:
    manifest_path = path / "dataset.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Postshot dataset has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != POSTSHOT_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported Postshot dataset schema: {manifest.get('schema_version')}")

    model_dir = path / "colmap"
    model_files = [model_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if not all(item.is_file() for item in model_files):
        raise RuntimeError(f"Postshot dataset has an incomplete COLMAP model: {model_dir}")
    records = read_colmap_image_index(model_dir / "images.bin")
    expected_images = {record.output_name for record in records}
    actual_images = {item.name for item in image_files(path / "images")}
    expected_masks = {f"{Path(name).stem}.png" for name in expected_images}
    actual_masks = {item.name for item in image_files(path / "masks")}
    if actual_images != expected_images:
        raise RuntimeError("Postshot image files do not match images.bin")
    if actual_masks != expected_masks:
        raise RuntimeError("Postshot mask files do not match images.bin")
    if len(records) != int(manifest["counts"]["images"]):
        raise RuntimeError("Postshot manifest image count does not match images.bin")

    expected_dimensions = {
        (int(item["height"]), int(item["width"])) for item in manifest["image_dimensions"]
    }
    for image in image_files(path / "images"):
        if image_dimensions(image) not in expected_dimensions:
            raise RuntimeError(f"Unexpected Postshot image dimensions: {image}")
    for mask in image_files(path / "masks"):
        stored = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if stored is None:
            raise RuntimeError(f"Cannot decode Postshot mask: {mask}")
        if stored.shape not in expected_dimensions:
            raise RuntimeError(f"Unexpected Postshot mask dimensions: {mask}")
        values = set(int(value) for value in np.unique(stored))
        if not values.issubset({0, 255}):
            raise RuntimeError(
                f"Postshot mask is not binary: {mask}; values={sorted(values)}"
            )

    for model_file in model_files:
        expected_hash = manifest["model_sha256"][model_file.name]
        if _sha256(model_file) != expected_hash:
            raise RuntimeError(f"Postshot COLMAP model hash mismatch: {model_file}")
    file_records = manifest.get("files")
    if file_records is not None:
        if len(file_records) != len(records):
            raise RuntimeError("Postshot file-hash record count is invalid")
        if {str(item["output_image"]) for item in file_records} != expected_images:
            raise RuntimeError("Postshot file hashes do not match images.bin")
        if {str(item["output_mask"]) for item in file_records} != expected_masks:
            raise RuntimeError("Postshot mask hashes do not match images.bin")
        for item in file_records:
            image = path / "images" / str(item["output_image"])
            mask = path / "masks" / str(item["output_mask"])
            if _sha256(image) != item["output_image_sha256"]:
                raise RuntimeError(f"Postshot image hash mismatch: {image}")
            if _sha256(mask) != item["output_mask_sha256"]:
                raise RuntimeError(f"Postshot mask hash mismatch: {mask}")
        if _sha256(path / "image-map.csv") != manifest.get("image_map_sha256"):
            raise RuntimeError("Postshot image-map hash mismatch")
    return manifest


def _validate_postshot_source_lineage(
    scene_path: Path,
    run: RunManifest,
    selected_dataset: Path,
    source_model: Path,
    manifest: dict[str, Any],
    mask_final: dict[str, Any] | None,
) -> None:
    source = manifest.get("source", {})
    expected_identity = {
        "location_id": run.location_id,
        "scene_id": run.scene_id,
        "run_id": run.id,
    }
    actual_identity = {key: manifest.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise RuntimeError("Postshot dataset belongs to a different run")
    if source.get("selected_dataset") != run.selected_dataset:
        raise RuntimeError("Postshot dataset does not match the run's selected dataset")
    if mask_final is not None and source.get("mask_final_sha256") != mask_final.get(
        "finalization_sha256"
    ):
        raise RuntimeError("Postshot dataset was prepared from a different mask-final")
    expected_model_path = source_model.relative_to(scene_path).as_posix()
    if source.get("colmap_model") != expected_model_path:
        raise RuntimeError("Postshot dataset references a different COLMAP model")
    current_model_hashes = {
        name: _sha256(source_model / name)
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    if source.get("colmap_model_sha256") != current_model_hashes:
        raise RuntimeError("Selected COLMAP model changed after Postshot preparation")
    _validate_colmap_exclusions(source_model, mask_final)

    file_records = manifest.get("files")
    if file_records is None:
        raise RuntimeError("Postshot dataset lacks per-file source lineage; prepare it again")
    source_images = selected_dataset / "images"
    source_masks = selected_dataset / "masks"
    for item in file_records:
        source_image = _registered_source_path(
            source_images, str(item["source_image"])
        )
        source_mask = ensure_within(
            source_masks / str(item["source_mask"]), source_masks
        )
        if not source_mask.is_file():
            raise RuntimeError(f"Source mask is missing: {source_mask}")
        if _sha256(source_image) != item["source_image_sha256"]:
            raise RuntimeError(f"Source image changed: {source_image}")
        if _sha256(source_mask) != item["source_mask_sha256"]:
            raise RuntimeError(f"Source mask changed: {source_mask}")


def _validate_colmap_exclusions(
    source_model: Path, mask_final: dict[str, Any] | None
) -> None:
    if mask_final is None:
        return
    excluded = {
        str(name).replace("\\", "/")
        for name in mask_final.get("excluded_images", [])
    }
    registered = {
        record.source_name.replace("\\", "/")
        for record in read_colmap_image_index(source_model / "images.bin")
    }
    overlap = sorted(excluded & registered)
    if overlap:
        raise RuntimeError(
            "Excluded masks leaked into the selected COLMAP model: "
            f"{overlap[:10]}"
        )


def prepare_postshot_dataset(
    scene_path: Path,
    run: RunManifest,
    output: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    reconstruction = run.stages.get("reconstruct")
    if reconstruction is None or reconstruction.status != StageStatus.SUCCEEDED:
        raise RuntimeError("Postshot preparation requires reconstruct=succeeded")
    if not run.selected_dataset:
        raise RuntimeError("Run has no selected reconstruction dataset")

    dataset = ensure_within(scene_path / run.selected_dataset, scene_path)
    mask_final = _validate_v3_quality_gates(scene_path, run, dataset)
    source_images = dataset / "images"
    source_masks = dataset / "masks"
    source_model = _selected_model_dir(dataset)
    required_model = [source_model / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if not all(item.is_file() for item in required_model):
        raise RuntimeError(f"Selected COLMAP model is incomplete: {source_model}")
    _validate_colmap_exclusions(source_model, mask_final)

    target = output or default_postshot_dataset(scene_path, run)
    target = target.absolute()
    _windows_argument(target)
    if target.exists():
        manifest = validate_postshot_dataset(target)
        _validate_postshot_source_lineage(
            scene_path, run, dataset, source_model, manifest, mask_final
        )
        return {**manifest, "output_path": str(target), "reused": True}
    building = target.with_name(target.name + ".building")
    if building.exists() and not resume:
        raise RuntimeError(f"Partial Postshot dataset exists; use --resume: {building}")
    building.mkdir(parents=True, exist_ok=True)

    records = read_colmap_image_index(source_model / "images.bin")
    if not records:
        raise RuntimeError(f"Selected COLMAP model has no registered images: {source_model}")
    target_images = building / "images"
    target_masks = building / "masks"
    target_model = building / "colmap"
    required_bytes = _remaining_postshot_bytes(
        records, source_images, source_masks, source_model, building
    )
    _check_disk_space(building, required_bytes)

    target_images.mkdir(parents=True, exist_ok=True)
    target_masks.mkdir(parents=True, exist_ok=True)
    target_model.mkdir(parents=True, exist_ok=True)

    workers = min(io_worker_count(), len(records))
    def prepare_record(record: ColmapImage) -> dict[str, Any]:
        return _prepare_registered_image(
            record, source_images, source_masks, target_images, target_masks
        )

    if workers <= 1:
        rows = []
        for index, record in enumerate(records, start=1):
            rows.append(prepare_record(record))
            if index == 1 or index % 100 == 0 or index == len(records):
                print(f"Postshot images and masks: {index}/{len(records)}", flush=True)
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, row in enumerate(pool.map(prepare_record, records), start=1):
                rows.append(row)
                if index == 1 or index % 100 == 0 or index == len(records):
                    print(f"Postshot images and masks: {index}/{len(records)}", flush=True)

    rewrite_colmap_images_binary(
        source_model / "images.bin", target_model / "images.bin", records
    )
    for name in ("cameras.bin", "points3D.bin"):
        destination = target_model / name
        source_hash = _sha256(source_model / name)
        if (
            not destination.is_file()
            or _sha256(destination) != source_hash
        ):
            _atomic_copy(source_model / name, destination)
        if _sha256(destination) != source_hash:
            raise RuntimeError(f"Copied COLMAP model hash changed: {destination}")

    _write_image_map(building / "image-map.csv", rows)
    dimensions = sorted({(int(row["height"]), int(row["width"])) for row in rows})
    model_sha256 = {
        name: _sha256(target_model / name)
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    manifest: dict[str, Any] = {
        "schema_version": POSTSHOT_SCHEMA_VERSION,
        "format": "postshot-colmap",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "location_id": run.location_id,
        "scene_id": run.scene_id,
        "run_id": run.id,
        "source": {
            "selected_dataset": run.selected_dataset,
            "colmap_model": source_model.relative_to(scene_path).as_posix(),
            "colmap_model_sha256": {
                name: _sha256(source_model / name)
                for name in ("cameras.bin", "images.bin", "points3D.bin")
            },
            "mask_final_sha256": (
                mask_final.get("finalization_sha256") if mask_final else None
            ),
        },
        "counts": {
            "images": len(records),
            "masks": len(records),
            "cameras": _model_item_count(target_model / "cameras.bin"),
            "points3D": _model_item_count(target_model / "points3D.bin"),
        },
        "image_dimensions": [
            {"height": height, "width": width} for height, width in dimensions
        ],
        "image_naming": "image_<COLMAP_IMAGE_ID>.<source-extension>",
        "mask_semantics": "white_is_ignored",
        "paths": {
            "images": "images",
            "masks": "masks",
            "colmap": "colmap",
            "image_map": "image-map.csv",
        },
        "model_sha256": model_sha256,
        "image_map_sha256": _sha256(building / "image-map.csv"),
        "files": rows,
        "validation": "passed",
    }
    _write_json(building / "dataset.json", manifest)
    _write_import_guide(building, target, len(records))
    validate_postshot_dataset(building)
    _validate_postshot_source_lineage(
        scene_path, run, dataset, source_model, manifest, mask_final
    )
    building.replace(target)
    return {**manifest, "output_path": str(target), "reused": False}


def postshot_executable() -> Path:
    configured = os.environ.get(POSTSHOT_CLI_ENV, "").strip()
    candidates = []
    if configured:
        candidates.append(host_path(configured))
    if os.name == "nt":
        candidates.append(Path(r"C:\Program Files\Jawset Postshot\bin\postshot-cli.exe"))
    else:
        candidates.append(
            Path("/mnt/c/Program Files/Jawset Postshot/bin/postshot-cli.exe")
        )
    found = shutil.which("postshot-cli") or shutil.which("postshot-cli.exe")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Postshot CLI is unavailable; install Postshot or set GSDB_POSTSHOT_CLI"
    )


def postshot_version(executable: Path | None = None) -> tuple[tuple[int, int, int], str]:
    executable = executable or postshot_executable()
    result = subprocess.run(
        [str(executable), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    output = "\n".join((result.stdout, result.stderr)).strip()
    match = re.search(r"Postshot v(\d+)\.(\d+)\.(\d+)", output)
    if result.returncode != 0 or match is None:
        raise RuntimeError(f"Unable to determine Postshot version: {output[:300]}")
    version = tuple(int(value) for value in match.groups())
    return version, ".".join(str(value) for value in version)


def available_vram_mib(gpu: int = 0) -> float:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is unavailable; cannot verify Postshot VRAM")
    result = subprocess.run(
        [
            executable,
            f"--id={gpu}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    return float(result.stdout.strip().splitlines()[0])


def _windows_argument(path: Path) -> str:
    absolute = path.absolute().resolve(strict=False)
    converted = wsl_to_windows(str(absolute))
    if os.name != "nt" and converted == str(absolute):
        raise RuntimeError(f"Postshot paths must be Windows-accessible: {absolute}")
    if converted.startswith("\\\\"):
        raise RuntimeError(f"Postshot paths may not use UNC storage: {converted}")
    return converted


def build_postshot_train_command(
    executable: Path,
    dataset: Path,
    output: Path,
    *,
    profile: str = "Splat3",
    ksteps: int | None = None,
    max_splats: int | None = None,
    gpu: int = 0,
    store_training_context: bool = True,
    export_ply: Path | None = None,
    export_spz: Path | None = None,
) -> list[str]:
    if profile not in {"Splat ADC", "Splat MCMC", "Splat3"}:
        raise ValueError(f"Unsupported Postshot profile: {profile}")
    if output.suffix.casefold() != ".psht":
        raise ValueError("Postshot --output must use a .psht project path")
    if export_ply is not None and export_spz is not None:
        raise ValueError("Postshot accepts only one --export-splat target per training command")
    if export_ply is not None and export_ply.suffix.casefold() != ".ply":
        raise ValueError("--export-ply must use a .ply output path")
    if export_spz is not None and export_spz.suffix.casefold() != ".spz":
        raise ValueError("--export-spz must use a .spz output path")
    command = [
        str(executable),
        "train",
        "--import",
        _windows_argument(dataset / "images"),
        _windows_argument(dataset / "colmap"),
        "--import-masks",
        _windows_argument(dataset / "masks"),
        "--profile",
        profile,
        "--image-select",
        "all",
        "--max-image-size",
        "0",
        "--mask-mode",
        "occluders",
        "--gpu",
        str(gpu),
        "--output",
        _windows_argument(output),
    ]
    if store_training_context:
        command.append("--store-training-context")
    if ksteps is not None:
        if ksteps < 1:
            raise ValueError("Postshot kSteps must be positive")
        command.extend(["--train-steps-limit", str(ksteps)])
    if max_splats is not None:
        if max_splats < 1:
            raise ValueError("Postshot maximum splats must be positive")
        command.extend(["--max-num-splats", str(max_splats)])
    for target in (export_ply, export_spz):
        if target is not None:
            command.extend(["--export-splat", _windows_argument(target)])
    return command


def _training_targets(
    scene_path: Path, run: RunManifest, requested: Path | None
) -> tuple[Path, Path]:
    logical = (
        requested.absolute()
        if requested is not None
        else ensure_work_dir(scene_path, run.id)
        / "postshot-training"
        / "model.psht"
    )
    execution = logical
    if requested is None:
        try:
            _windows_argument(logical)
        except RuntimeError:
            execution = (
                scene_path
                / "inputs"
                / "postshot-training"
                / run.id
                / "model.psht"
            ).absolute()
    return logical, execution


def _publish_training_output(execution: Path, logical: Path) -> None:
    if execution.resolve() == logical.resolve(strict=False):
        return
    if logical.exists():
        if _sha256(logical) != _sha256(execution):
            raise RuntimeError(f"Logical Postshot output already differs: {logical}")
        return
    logical.parent.parent.mkdir(parents=True, exist_ok=True)
    try:
        logical.parent.symlink_to(execution.parent, target_is_directory=True)
    except OSError:
        logical.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy(execution, logical)


def _training_sidecars(target: Path) -> tuple[Path, Path]:
    """Return collision-free log and manifest paths for one output project."""

    if target.name.casefold() == "model.psht":
        return target.parent / "postshot-train.log", target.parent / "training.json"
    return (
        target.parent / f"{target.stem}.postshot-train.log",
        target.parent / f"{target.stem}.training.json",
    )


def train_postshot(
    scene_path: Path,
    run: RunManifest,
    *,
    dry_run: bool = False,
    allow_low_vram: bool = False,
    profile: str = "Splat3",
    ksteps: int | None = None,
    max_splats: int | None = None,
    gpu: int = 0,
    dataset: Path | None = None,
    output: Path | None = None,
    store_training_context: bool = True,
    export_ply: Path | None = None,
    export_spz: Path | None = None,
) -> dict[str, Any]:
    if not run.selected_dataset:
        raise RuntimeError("Run has no selected reconstruction dataset")
    dataset = (dataset or default_postshot_dataset(scene_path, run)).absolute()
    postshot_manifest = validate_postshot_dataset(dataset)
    selected = ensure_within(scene_path / run.selected_dataset, scene_path)
    final = _validate_v3_quality_gates(scene_path, run, selected)
    source_model = _selected_model_dir(selected)
    _validate_postshot_source_lineage(
        scene_path, run, selected, source_model, postshot_manifest, final
    )
    executable = postshot_executable()
    version_tuple, version_text = postshot_version(executable)
    if version_tuple < POSTSHOT_MINIMUM_VERSION:
        raise RuntimeError(
            f"Postshot {version_text} is too old; 1.1.69 or newer is required"
        )
    free_vram_mib = available_vram_mib(gpu)
    if free_vram_mib < 8192 and not allow_low_vram:
        raise RuntimeError(
            f"Only {free_vram_mib / 1024:.1f} GiB VRAM is free; use --allow-low-vram to override"
        )
    logical_target, execution_target = _training_targets(scene_path, run, output)
    command = build_postshot_train_command(
        executable,
        dataset,
        execution_target,
        profile=profile,
        ksteps=ksteps,
        max_splats=max_splats,
        gpu=gpu,
        store_training_context=store_training_context,
        export_ply=export_ply,
        export_spz=export_spz,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "postshot_version": version_text,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset / "dataset.json"),
        "available_vram_mib": free_vram_mib,
        "command": command,
        "output": str(logical_target),
        "execution_output": str(execution_target),
        "dry_run": dry_run,
    }
    result["lineage_sha256"] = canonical_hash(
        {
            "dataset_sha256": result["dataset_sha256"],
            "command": command,
            "postshot_version": version_text,
        }
    )
    conflicting_targets = {
        target for target in (logical_target, execution_target) if target.exists()
    }
    if conflicting_targets:
        raise FileExistsError(
            "Postshot output already exists; choose a new --output: "
            + ", ".join(sorted(str(target) for target in conflicting_targets))
        )
    if dry_run:
        return result
    execution_target.parent.mkdir(parents=True, exist_ok=True)
    log_path, manifest_path = _training_sidecars(execution_target)
    try:
        metrics = run_logged(
            command, log_path, monitor_gpu=True, gpu_index=gpu
        )
        result["metrics"] = metrics
        if not execution_target.is_file():
            raise RuntimeError(
                "Postshot exited successfully without creating the .psht project"
            )
        exported: dict[str, dict[str, str]] = {}
        for format_name, export_path in (("ply", export_ply), ("spz", export_spz)):
            if export_path is None:
                continue
            if not export_path.is_file():
                raise RuntimeError(
                    f"Postshot did not create the requested {format_name.upper()} export"
                )
            exported[format_name] = {
                "path": str(export_path.absolute()),
                "sha256": _sha256(export_path),
            }
        _publish_training_output(execution_target, logical_target)
        result.update(
            {
                "status": "succeeded",
                "completed_at": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "output_sha256": _sha256(execution_target),
                "log": str(log_path),
                "exports": exported,
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "error": str(error),
                "log": str(log_path),
            }
        )
        if isinstance(error, CommandError):
            result["metrics"] = error.metrics
        if execution_target.is_file():
            result["output_sha256"] = _sha256(execution_target)
        partial_exports: dict[str, dict[str, str]] = {}
        for format_name, export_path in (("ply", export_ply), ("spz", export_spz)):
            if export_path is not None and export_path.is_file():
                partial_exports[format_name] = {
                    "path": str(export_path.absolute()),
                    "sha256": _sha256(export_path),
                }
        if partial_exports:
            result["exports"] = partial_exports
        _write_json(manifest_path, result)
        raise
    _write_json(manifest_path, result)
    return result
