from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .masking import atomic_imwrite, image_files
from .models import ReconstructionAttempt
from .processes import run_logged


def expected_planar_images(attempt: ReconstructionAttempt) -> int:
    return attempt.frame_count * attempt.images_per_equirect


def projection_fov_degrees(attempt: ReconstructionAttempt) -> float:
    return 120.0 if attempt.images_per_equirect == 8 else 110.0


def project_equirectangular_frames(
    source: Path,
    dataset: Path,
    attempt: ReconstructionAttempt,
) -> list[Path]:
    expected = expected_planar_images(attempt)
    target = dataset / "images"
    if target.is_dir():
        existing = image_files(target)
        if len(existing) != expected:
            raise RuntimeError(
                f"Existing planar image set is incomplete ({len(existing)}/{expected}); "
                "create a new run instead of overwriting partial projection output"
            )
        return existing
    if target.exists():
        raise RuntimeError(f"Planar image target is not a directory: {target}")

    from nerfstudio.process_data.equirect_utils import (
        compute_resolution_from_equirect,
        generate_planar_projections_from_equirectangular,
    )

    generated = source / "planar_projections"
    if generated.exists():
        existing = image_files(generated) if generated.is_dir() else []
        if len(existing) != expected:
            raise RuntimeError(
                f"Found incomplete projection scratch data ({len(existing)}/{expected}) in {generated}"
            )
    else:
        resolution = compute_resolution_from_equirect(
            source, attempt.images_per_equirect
        )
        generated = generate_planar_projections_from_equirectangular(
            source,
            resolution,
            attempt.images_per_equirect,
            crop_factor=(0.0, attempt.crop_bottom, 0.0, 0.0),
        )
    produced = image_files(generated)
    if len(produced) != expected:
        raise RuntimeError(
            f"Projection produced {len(produced)} images; expected {expected}"
        )
    dataset.mkdir(parents=True, exist_ok=True)
    generated.replace(target)
    return image_files(target)


def build_image_pyramid(
    source_dir: Path,
    dataset: Path,
    prefix: str,
    num_downscales: int,
    is_mask: bool = False,
) -> None:
    sources = image_files(source_dir)
    for level in range(1, num_downscales + 1):
        factor = 2**level
        target_dir = dataset / f"{prefix}_{factor}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            target = target_dir / source.name
            flag = cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR
            image = cv2.imread(str(source), flag)
            if image is None:
                raise RuntimeError(f"Cannot decode pyramid source: {source}")
            height, width = image.shape[:2]
            expected_shape = (max(1, height // factor), max(1, width // factor))
            if target.exists():
                existing = cv2.imread(str(target), flag)
                if existing is None or existing.shape[:2] != expected_shape:
                    raise RuntimeError(
                        f"Existing pyramid file is invalid: {target}; "
                        f"expected shape {expected_shape}"
                    )
                if is_mask and not set(int(value) for value in np.unique(existing)).issubset(
                    {0, 255}
                ):
                    raise RuntimeError(f"Existing mask pyramid file is not binary: {target}")
                continue
            interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
            resized = cv2.resize(
                image,
                (expected_shape[1], expected_shape[0]),
                interpolation=interpolation,
            )
            arguments = [] if is_mask else [cv2.IMWRITE_JPEG_QUALITY, 95]
            atomic_imwrite(target, resized, arguments)
        actual = image_files(target_dir)
        if {item.name for item in actual} != {item.name for item in sources}:
            raise RuntimeError(
                f"Pyramid {target_dir.name} does not exactly match its source file set"
            )


def pinhole_camera_parameters(image_path: Path, fov_degrees: float) -> str:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot determine planar image dimensions: {image_path}")
    height, width = image.shape
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return f"{focal:.10f},{focal:.10f},{width / 2.0:.10f},{height / 2.0:.10f}"


def build_colmap_commands(
    dataset: Path,
    attempt: ReconstructionAttempt,
    colmap_attempt_dir: Path | None = None,
) -> list[list[str]]:
    images_dir = dataset / "images"
    masks_dir = dataset / "masks"
    images = image_files(images_dir)
    if not images:
        raise RuntimeError(f"No planar images found in {images_dir}")
    colmap_attempt_dir = colmap_attempt_dir or dataset / "colmap" / "attempt-001"
    database = colmap_attempt_dir / "database.db"
    sparse = colmap_attempt_dir / "sparse"
    camera_params = pinhole_camera_parameters(images[0], projection_fov_degrees(attempt))
    return [
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--ImageReader.mask_path",
            str(masks_dir),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_params",
            camera_params,
            "--SiftExtraction.use_gpu",
            "0",
        ],
        [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            str(max(10, attempt.images_per_equirect * 2)),
            "--SiftMatching.use_gpu",
            "0",
        ],
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse),
        ],
    ]


def _colmap_components(sparse_root: Path) -> list[tuple[Path, dict[int, Any]]]:
    from nerfstudio.process_data.colmap_utils import read_images_binary

    components: list[tuple[Path, dict[int, Any]]] = []
    for candidate in sorted(item for item in sparse_root.iterdir() if item.is_dir()):
        required = [candidate / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
        if all(path.is_file() for path in required):
            components.append((candidate, read_images_binary(candidate / "images.bin")))
    return components


def _selected_sparse_root(dataset: Path) -> Path:
    selection = dataset / "colmap" / "selected-attempt.json"
    if selection.is_file():
        payload = json.loads(selection.read_text(encoding="utf-8"))
        selected = dataset / "colmap" / str(payload["attempt"]) / "sparse"
        if selected.is_dir():
            return selected
    completed = sorted(
        path.parent / "sparse"
        for path in (dataset / "colmap").glob("attempt-*/.mapping-complete")
    )
    if completed:
        return completed[-1]
    raise RuntimeError("No completed COLMAP attempt is recorded")


def validate_frame_masks(transforms_path: Path, masks_dir: Path) -> int:
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise RuntimeError(f"No registered frames in {transforms_path}")
    for frame in frames:
        image_name = Path(frame["file_path"]).name
        mask = masks_dir / f"{image_name}.png"
        if not mask.is_file():
            raise RuntimeError(f"Registered image has no person mask: {image_name}")
        expected = f"./masks/{mask.name}"
        if frame.get("mask_path") != expected:
            raise RuntimeError(f"Registered image has an invalid mask_path: {image_name}")
    return len(frames)


def attach_frame_masks(transforms_path: Path, masks_dir: Path) -> None:
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise RuntimeError(f"No registered frames in {transforms_path}")
    for frame in frames:
        image_name = Path(frame["file_path"]).name
        mask = masks_dir / f"{image_name}.png"
        if not mask.is_file():
            raise RuntimeError(f"Registered image has no person mask: {image_name}")
        frame["mask_path"] = f"./masks/{mask.name}"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=transforms_path.parent,
        prefix=f".{transforms_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(transforms_path)
    validate_frame_masks(transforms_path, masks_dir)


def reconstruction_metrics(
    dataset: Path, attempt: ReconstructionAttempt
) -> dict[str, Any]:
    transforms_path = dataset / "transforms.json"
    if not transforms_path.is_file():
        raise RuntimeError(f"COLMAP conversion did not create {transforms_path}")
    validate_frame_masks(transforms_path, dataset / "masks")
    components = _colmap_components(_selected_sparse_root(dataset))
    component_sizes = [len(images) for _, images in components]
    unique_names = {
        image.name for _, images in components for image in images.values()
    }
    expected = expected_planar_images(attempt)
    selected_registered = len(
        json.loads(transforms_path.read_text(encoding="utf-8")).get("frames", [])
    )
    return {
        "expected_planar_images": expected,
        "written_planar_images": len(image_files(dataset / "images")),
        "registered_images": len(unique_names),
        "selected_component_images": selected_registered,
        "registration_ratio": len(unique_names) / expected if expected else 0.0,
        "largest_component_coverage": max(component_sizes, default=0) / expected
        if expected
        else 0.0,
        "component_count": len(components),
        "component_sizes": sorted(component_sizes, reverse=True),
    }


def run_masked_colmap(
    dataset: Path,
    attempt: ReconstructionAttempt,
    log_dir: Path,
) -> dict[str, Any]:
    transforms = dataset / "transforms.json"
    if transforms.is_file():
        try:
            return reconstruction_metrics(dataset, attempt)
        except (RuntimeError, json.JSONDecodeError, KeyError, TypeError):
            # A process may have stopped after colmap_to_json wrote transforms.json but
            # before per-frame mask paths were attached. Reuse the completed sparse model
            # below, regenerate transforms, and never allow unmasked training to continue.
            pass

    colmap_root = dataset / "colmap"
    colmap_root.mkdir(parents=True, exist_ok=True)
    existing_attempts = sorted(
        item for item in colmap_root.glob("attempt-*") if item.is_dir()
    )
    if existing_attempts and not (existing_attempts[-1] / ".mapping-complete").is_file():
        next_number = int(existing_attempts[-1].name.rsplit("-", 1)[1]) + 1
        colmap_attempt = colmap_root / f"attempt-{next_number:03d}"
    elif existing_attempts:
        colmap_attempt = existing_attempts[-1]
    else:
        colmap_attempt = colmap_root / "attempt-001"
    sparse_root = colmap_attempt / "sparse"
    colmap_attempt.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)
    commands = build_colmap_commands(dataset, attempt, colmap_attempt)
    steps = (
        ("features", commands[0]),
        ("matching", commands[1]),
        ("mapping", commands[2]),
    )
    for name, command in steps:
        marker = colmap_attempt / f".{name}-complete"
        if marker.is_file():
            continue
        run_logged(command, log_dir / colmap_attempt.name / f"colmap-{name}.log")
        if name != "mapping":
            marker.write_text("complete\n", encoding="utf-8")

    components = _colmap_components(sparse_root)
    if not components:
        raise RuntimeError("COLMAP mapper completed without a valid sparse model")
    (colmap_attempt / ".mapping-complete").write_text("complete\n", encoding="utf-8")
    selected_dir, _ = max(components, key=lambda item: len(item[1]))
    (colmap_root / "selected-attempt.json").write_text(
        json.dumps(
            {"attempt": colmap_attempt.name, "component": selected_dir.name}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    from nerfstudio.process_data.colmap_utils import colmap_to_json

    colmap_to_json(selected_dir, dataset, use_single_camera_mode=True)
    attach_frame_masks(transforms, dataset / "masks")
    (colmap_attempt / ".conversion-complete").write_text("complete\n", encoding="utf-8")
    return reconstruction_metrics(dataset, attempt)
