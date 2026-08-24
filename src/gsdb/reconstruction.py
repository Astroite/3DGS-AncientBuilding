from __future__ import annotations

import json
import math
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .masking import atomic_imwrite, image_files
from .models import (
    LegacyReconstructionAttemptV1,
    LegacyReconstructionConfigV1,
    ReconstructionAttempt,
    ReconstructionConfig,
)
from .processes import run_logged


AttemptConfig = ReconstructionAttempt | LegacyReconstructionAttemptV1
ReconstructionSettings = ReconstructionConfig | LegacyReconstructionConfigV1


def expected_planar_images(attempt: AttemptConfig) -> int:
    return attempt.frame_count * attempt.images_per_equirect


def projection_fov_degrees(attempt: AttemptConfig) -> float:
    return float(
        getattr(
            attempt,
            "projection_fov_degrees",
            120.0 if attempt.images_per_equirect == 8 else 110.0,
        )
    )


def _crop_vertical_bounds(
    bounds: list[float], fov_degrees: float, crop_bottom: float
) -> list[float | None]:
    """Mirror Nerfstudio 1.1.5's bottom-crop pitch adjustment without private imports."""
    adjusted: list[float | None] = list(bounds)
    if crop_bottom <= 0:
        return adjusted
    degrees_chopped = 180.0 * crop_bottom
    new_top_start = -90.0 + degrees_chopped + fov_degrees / 2.0
    for index, value in enumerate(adjusted):
        assert value is not None
        if value < new_top_start - fov_degrees / 2.0:
            adjusted[index] = None
        elif value < new_top_start:
            difference = new_top_start - value
            adjusted[index] = new_top_start
            for later in range(index + 1, len(adjusted)):
                assert adjusted[later] is not None
                adjusted[later] += difference / (2 ** (later - index))
            break
    return adjusted


def projection_view_specs(attempt: AttemptConfig) -> list[tuple[float, float]]:
    """Return deterministic ``(yaw, pitch)`` pairs in Nerfstudio's view order."""
    bounds = _crop_vertical_bounds(
        [-45.0, 0.0, 45.0], projection_fov_degrees(attempt), attempt.crop_bottom
    )
    middle_step, outer_step = (
        (90, 180) if attempt.images_per_equirect == 8 else (60, 90)
    )
    pairs: list[tuple[float, float]] = []
    for bound_index, step in ((1, middle_step), (2, outer_step), (0, outer_step)):
        pitch = bounds[bound_index]
        if pitch is not None:
            pairs.extend((float(yaw), float(pitch)) for yaw in range(-180, 180, step))
    if len(pairs) != attempt.images_per_equirect:
        raise RuntimeError(
            f"Projection crop leaves {len(pairs)} views; expected {attempt.images_per_equirect}"
        )
    return pairs


def project_equirectangular_frames(
    source: Path,
    dataset: Path,
    attempt: AttemptConfig,
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

    if isinstance(attempt, LegacyReconstructionAttemptV1):
        from nerfstudio.process_data.equirect_utils import (
            compute_resolution_from_equirect,
            generate_planar_projections_from_equirectangular,
        )

        generated = source / "planar_projections"
        if generated.exists():
            existing = image_files(generated) if generated.is_dir() else []
            if len(existing) != expected:
                raise RuntimeError(
                    f"Found incomplete projection scratch data ({len(existing)}/{expected}) "
                    f"in {generated}"
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

    import torch
    from nerfstudio.process_data.equirect_utils import equirect2persp

    frames = image_files(source)
    if len(frames) != attempt.frame_count:
        raise RuntimeError(
            f"Projection source has {len(frames)} frames; expected {attempt.frame_count}"
        )
    target.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    size = attempt.projection_size
    specs = projection_view_specs(attempt)
    for view_index in range(len(specs)):
        (target / f"view_{view_index:02d}").mkdir(parents=True, exist_ok=False)
    for frame_index, frame in enumerate(frames, start=1):
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode equirectangular frame: {frame}")
        tensor = torch.tensor(image, dtype=torch.float32, device=device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        for view_index, (yaw, pitch) in enumerate(specs):
            perspective = equirect2persp(
                tensor,
                attempt.projection_fov_degrees,
                yaw,
                pitch,
                size,
                size,
            )
            output = (
                (perspective * 255.0)
                .clamp(0, 255)
                .squeeze(0)
                .permute(1, 2, 0)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            atomic_imwrite(
                target / f"view_{view_index:02d}" / f"frame_{frame_index:06d}.jpg",
                output,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
        print(f"Planar projection: {frame_index}/{len(frames)}", flush=True)
    produced = image_files(target)
    if len(produced) != expected:
        raise RuntimeError(f"Projection produced {len(produced)} images; expected {expected}")
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
            relative = source.relative_to(source_dir)
            target = target_dir / relative
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
        if {item.relative_to(target_dir) for item in actual} != {
            item.relative_to(source_dir) for item in sources
        }:
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
    attempt: AttemptConfig,
    settings: ReconstructionSettings | None = None,
    colmap_attempt_dir: Path | None = None,
    mapper_num_threads: int | None = None,
) -> list[list[str]]:
    images_dir = dataset / "images"
    masks_dir = dataset / "masks"
    images = image_files(images_dir)
    if not images:
        raise RuntimeError(f"No planar images found in {images_dir}")
    colmap_attempt_dir = colmap_attempt_dir or dataset / "colmap" / "attempt-001"
    database = colmap_attempt_dir / "database.db"
    sparse = colmap_attempt_dir / "sparse"
    image_list = colmap_attempt_dir / "image-list.txt"
    image_list.parent.mkdir(parents=True, exist_ok=True)
    names = interleaved_image_names(images_dir)
    image_list.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    camera_params = pinhole_camera_parameters(images[0], projection_fov_degrees(attempt))
    is_v2 = isinstance(attempt, ReconstructionAttempt)
    use_gpu = bool(getattr(settings, "use_gpu_sift", is_v2))
    gpu_index = int(getattr(settings, "gpu_index", 0))
    fix_intrinsics = bool(getattr(settings, "fix_intrinsics", is_v2))
    commands = [
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--image_list_path",
            str(image_list),
            "--ImageReader.mask_path",
            str(masks_dir),
            "--ImageReader.camera_model",
            "PINHOLE",
            (
                "--ImageReader.single_camera_per_folder"
                if is_v2
                else "--ImageReader.single_camera"
            ),
            "1",
            "--ImageReader.camera_params",
            camera_params,
            "--SiftExtraction.use_gpu",
            "1" if use_gpu else "0",
        ],
        [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            str(max(10, attempt.images_per_equirect * 2)),
            "--SiftMatching.use_gpu",
            "1" if use_gpu else "0",
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
    if use_gpu:
        commands[0].extend(["--SiftExtraction.gpu_index", str(gpu_index)])
        commands[1].extend(["--SiftMatching.gpu_index", str(gpu_index)])
    if fix_intrinsics:
        commands[2].extend(
            [
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
            ]
        )
    if mapper_num_threads is not None:
        commands[2].extend(["--Mapper.num_threads", str(mapper_num_threads)])
    return commands


_NESTED_FRAME_PATTERN = re.compile(r"^view_(\d+)/frame_(\d+)\.[^.]+$")
_LEGACY_FRAME_PATTERN = re.compile(r"^frame_(\d+)_([0-9]+)\.[^.]+$")


def _image_order_key(name: str) -> tuple[int, int, str]:
    normalized = name.replace("\\", "/")
    nested = _NESTED_FRAME_PATTERN.match(normalized)
    if nested:
        view, frame = (int(value) for value in nested.groups())
        return frame, view, normalized
    legacy = _LEGACY_FRAME_PATTERN.match(normalized)
    if legacy:
        frame, view = (int(value) for value in legacy.groups())
        return frame, view, normalized
    return 2**31 - 1, 2**31 - 1, normalized


def interleaved_image_names(images_dir: Path) -> list[str]:
    """Frame-major order keeps sequential matching aligned across view folders."""
    names = [path.relative_to(images_dir).as_posix() for path in image_files(images_dir)]
    return sorted(names, key=_image_order_key)


def reorder_database_image_ids(database: Path, ordered_names: list[str]) -> bool:
    """Transactionally enforce frame-major IDs before sequential matching.

    COLMAP 3.8 sorts ``image_list_path`` lexically before inserting rows, so a
    per-folder camera layout otherwise becomes view-major despite an interleaved
    list. Feature tables can be safely re-keyed before any matches exist.
    """
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT image_id, name FROM images").fetchall()
        actual_names = {str(name) for _, name in rows}
        if actual_names != set(ordered_names) or len(rows) != len(ordered_names):
            raise RuntimeError(
                f"COLMAP database image set differs from image-list.txt: "
                f"database={len(rows)}, expected={len(ordered_names)}"
            )
        for table in ("matches", "two_view_geometries"):
            count = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            if count:
                raise RuntimeError(
                    f"Refusing to reorder COLMAP image IDs after {table} has rows"
                )
        desired = {name: index for index, name in enumerate(ordered_names, start=1)}
        current = {str(name): int(image_id) for image_id, name in rows}
        if current == desired:
            return False
        offset = max(current.values()) + len(current) + 1000
        connection.execute("BEGIN IMMEDIATE")
        for table in ("images", "keypoints", "descriptors"):
            connection.execute(f"UPDATE {table} SET image_id = image_id + ?", (offset,))
        for name, new_id in desired.items():
            temporary_id = current[name] + offset
            for table in ("images", "keypoints", "descriptors"):
                connection.execute(
                    f"UPDATE {table} SET image_id = ? WHERE image_id = ?",
                    (new_id, temporary_id),
                )
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone():
            connection.execute(
                "UPDATE sqlite_sequence SET seq = ? WHERE name = 'images'",
                (len(ordered_names),),
            )
        connection.commit()
        verified = connection.execute(
            "SELECT name FROM images ORDER BY image_id"
        ).fetchall()
    if [str(row[0]) for row in verified] != ordered_names:
        raise RuntimeError("COLMAP database image-ID reordering verification failed")
    return True


def validate_folder_camera_ids(
    database: Path, expected_view_count: int
) -> dict[str, int]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
    grouped: dict[str, set[int]] = {}
    for name, camera_id in rows:
        normalized = str(name).replace("\\", "/")
        prefix = normalized.split("/", 1)[0] + "/"
        grouped.setdefault(prefix, set()).add(int(camera_id))
    expected_prefixes = {f"view_{index:02d}/" for index in range(expected_view_count)}
    if set(grouped) != expected_prefixes:
        raise RuntimeError(
            f"COLMAP camera folders mismatch: expected={sorted(expected_prefixes)}, "
            f"actual={sorted(grouped)}"
        )
    if any(len(camera_ids) != 1 for camera_ids in grouped.values()):
        raise RuntimeError("COLMAP did not assign exactly one camera ID per view folder")
    mapping = {prefix: next(iter(camera_ids)) for prefix, camera_ids in grouped.items()}
    if len(set(mapping.values())) != expected_view_count:
        raise RuntimeError("COLMAP view folders do not have distinct camera IDs")
    return dict(sorted(mapping.items()))


def _rodrigues(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    angle = math.radians(angle_degrees)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def projection_world_from_camera(yaw: float, pitch: float) -> np.ndarray:
    """Rotation from COLMAP camera axes to the equirectangular world axes."""
    yaw_rotation = _rodrigues(np.array((0.0, 0.0, 1.0)), yaw)
    pitch_axis = yaw_rotation @ np.array((0.0, 1.0, 0.0))
    pitch_rotation = _rodrigues(pitch_axis, -pitch)
    # Nerfstudio rays are [forward, right, up]; COLMAP uses [right, down, forward].
    nerfstudio_from_colmap = np.array(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    )
    return pitch_rotation @ yaw_rotation @ nerfstudio_from_colmap


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = math.sqrt(
            1.0 + matrix[index, index] - matrix[next_index, next_index] - matrix[last_index, last_index]
        ) * 2.0
        quaternion = np.zeros(4, dtype=np.float64)
        quaternion[index + 1] = 0.25 * scale
        quaternion[0] = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        quaternion[next_index + 1] = (
            matrix[next_index, index] + matrix[index, next_index]
        ) / scale
        quaternion[last_index + 1] = (
            matrix[last_index, index] + matrix[index, last_index]
        ) / scale
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return [float(value) for value in quaternion]


def build_rig_config(
    camera_ids: dict[str, int], attempt: ReconstructionAttempt
) -> list[dict[str, Any]]:
    specs = projection_view_specs(attempt)
    reference_prefix = "view_00/"
    reference_world_from_camera = projection_world_from_camera(*specs[0])
    cameras: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        prefix = f"view_{index:02d}/"
        entry: dict[str, Any] = {
            "camera_id": camera_ids[prefix],
            "image_prefix": prefix,
        }
        if prefix != reference_prefix:
            world_from_camera = projection_world_from_camera(*spec)
            camera_from_rig = world_from_camera.T @ reference_world_from_camera
            entry["cam_from_rig_rotation"] = rotation_matrix_to_quaternion_wxyz(
                camera_from_rig
            )
            entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
        cameras.append(entry)
    return [{"ref_camera_id": camera_ids[reference_prefix], "cameras": cameras}]


def write_rig_config(
    path: Path, camera_ids: dict[str, int], attempt: ReconstructionAttempt
) -> list[dict[str, Any]]:
    payload = build_rig_config(camera_ids, attempt)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def select_colmap_attempt(colmap_root: Path) -> tuple[Path, int | None]:
    """Select an attempt and request a safe mapper retry when prerequisites are reusable."""
    existing_attempts = sorted(
        item for item in colmap_root.glob("attempt-*") if item.is_dir()
    )
    if not existing_attempts:
        return colmap_root / "attempt-001", None

    latest = existing_attempts[-1]
    if (latest / ".mapping-complete").is_file():
        return latest, None

    prerequisites_complete = all(
        (latest / f".{name}-complete").is_file()
        for name in ("features", "matching")
    )
    sparse_root = latest / "sparse"
    sparse_has_output = sparse_root.is_dir() and any(sparse_root.iterdir())
    if prerequisites_complete and not sparse_has_output:
        return latest, 1

    next_number = int(latest.name.rsplit("-", 1)[1]) + 1
    return colmap_root / f"attempt-{next_number:03d}", None


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


def _selected_model_dir(dataset: Path) -> Path:
    selection = dataset / "colmap" / "selected-attempt.json"
    payload = json.loads(selection.read_text(encoding="utf-8"))
    model = dataset / "colmap" / str(payload["attempt"]) / str(payload["model"])
    required = [model / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"Selected COLMAP model is incomplete: {model}")
    return model


def _relative_image_path(file_path: str) -> Path:
    normalized = file_path.replace("\\", "/").removeprefix("./")
    if normalized.startswith("images/"):
        normalized = normalized.removeprefix("images/")
    return Path(normalized)


def validate_frame_masks(transforms_path: Path, masks_dir: Path) -> int:
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise RuntimeError(f"No registered frames in {transforms_path}")
    for frame in frames:
        image_relative = _relative_image_path(str(frame["file_path"]))
        mask_relative = image_relative.parent / f"{image_relative.name}.png"
        mask = masks_dir / mask_relative
        if not mask.is_file():
            raise RuntimeError(f"Registered image has no person mask: {image_relative.as_posix()}")
        expected = f"./masks/{mask_relative.as_posix()}"
        if frame.get("mask_path") != expected:
            raise RuntimeError(
                f"Registered image has an invalid mask_path: {image_relative.as_posix()}"
            )
    return len(frames)


def attach_frame_masks(transforms_path: Path, masks_dir: Path) -> None:
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise RuntimeError(f"No registered frames in {transforms_path}")
    for frame in frames:
        image_relative = _relative_image_path(str(frame["file_path"]))
        mask_relative = image_relative.parent / f"{image_relative.name}.png"
        mask = masks_dir / mask_relative
        if not mask.is_file():
            raise RuntimeError(f"Registered image has no person mask: {image_relative.as_posix()}")
        frame["mask_path"] = f"./masks/{mask_relative.as_posix()}"
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


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec /= np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array(
        (
            (1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y),
            (2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x),
            (2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y),
        ),
        dtype=np.float64,
    )


def _frame_number(name: str) -> int:
    normalized = name.replace("\\", "/")
    nested = _NESTED_FRAME_PATTERN.match(normalized)
    if nested:
        return int(nested.group(2))
    legacy = _LEGACY_FRAME_PATTERN.match(Path(normalized).name)
    if legacy:
        return int(legacy.group(1))
    raise ValueError(f"Unrecognized projected image name: {name}")


def center_spread_metrics(
    named_centers: list[tuple[str, np.ndarray]]
) -> dict[str, float | int]:
    grouped: dict[int, list[np.ndarray]] = {}
    for name, center in named_centers:
        grouped.setdefault(_frame_number(name), []).append(np.asarray(center, dtype=np.float64))
    centroids: list[tuple[int, np.ndarray]] = []
    spreads: list[float] = []
    for frame, centers in grouped.items():
        matrix = np.vstack(centers)
        centroid = matrix.mean(axis=0)
        centroids.append((frame, centroid))
        spreads.extend(float(value) for value in np.linalg.norm(matrix - centroid, axis=1))
    centroids.sort(key=lambda item: item[0])
    baselines = [
        float(np.linalg.norm(second[1] - first[1]))
        for first, second in zip(centroids, centroids[1:])
    ]
    median_baseline = float(np.median(baselines)) if baselines else 0.0
    p95_spread = float(np.percentile(spreads, 95)) if spreads else 0.0
    return {
        "frame_count": len(grouped),
        "center_spread_median": float(np.median(spreads)) if spreads else 0.0,
        "center_spread_p95": p95_spread,
        "center_spread_max": max(spreads, default=0.0),
        "median_interframe_baseline": median_baseline,
        "p95_spread_to_baseline": p95_spread / median_baseline if median_baseline else 0.0,
    }


def sparse_model_metrics(model_dir: Path) -> dict[str, Any]:
    from nerfstudio.process_data.colmap_utils import (
        read_cameras_binary,
        read_images_binary,
    )

    images = read_images_binary(model_dir / "images.bin")
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    named_centers: list[tuple[str, np.ndarray]] = []
    for image in images.values():
        rotation = _qvec_to_rotation(image.qvec)
        center = -(rotation.T @ np.asarray(image.tvec, dtype=np.float64))
        named_centers.append((image.name, center))
    intrinsics = {
        str(camera_id): {
            "model": camera.model,
            "width": camera.width,
            "height": camera.height,
            "params": [float(value) for value in camera.params],
        }
        for camera_id, camera in sorted(cameras.items())
    }
    return {**center_spread_metrics(named_centers), "intrinsics": intrinsics}


def reconstruction_metrics(
    dataset: Path,
    attempt: AttemptConfig,
    settings: ReconstructionSettings | None = None,
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
    selection_path = dataset / "colmap" / "selected-attempt.json"
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.is_file()
        else {}
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
        "sift_gpu": bool(getattr(settings, "use_gpu_sift", False)),
        "fixed_intrinsics": bool(getattr(settings, "fix_intrinsics", False)),
        "rig": selection.get("rig", {"enabled": False}),
    }


def run_masked_colmap(
    dataset: Path,
    attempt: AttemptConfig,
    log_dir: Path,
    settings: ReconstructionSettings | None = None,
) -> dict[str, Any]:
    transforms = dataset / "transforms.json"
    if transforms.is_file():
        try:
            return reconstruction_metrics(dataset, attempt, settings)
        except (RuntimeError, json.JSONDecodeError, KeyError, TypeError):
            # A process may have stopped after colmap_to_json wrote transforms.json but
            # before per-frame mask paths were attached. Reuse the completed sparse model
            # below, regenerate transforms, and never allow unmasked training to continue.
            pass

    colmap_root = dataset / "colmap"
    colmap_root.mkdir(parents=True, exist_ok=True)
    colmap_attempt, mapper_num_threads = select_colmap_attempt(colmap_root)
    sparse_root = colmap_attempt / "sparse"
    colmap_attempt.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)
    commands = build_colmap_commands(
        dataset,
        attempt,
        settings,
        colmap_attempt,
        mapper_num_threads=mapper_num_threads,
    )
    rig_enabled = isinstance(attempt, ReconstructionAttempt) and attempt.use_rig
    rig_config_path = colmap_attempt / "rig-config.json"
    feature_marker = colmap_attempt / ".features-complete"
    if not feature_marker.is_file():
        run_logged(
            commands[0], log_dir / colmap_attempt.name / "colmap-features.log"
        )
        if isinstance(attempt, ReconstructionAttempt):
            reorder_database_image_ids(
                colmap_attempt / "database.db", interleaved_image_names(dataset / "images")
            )
        feature_marker.write_text("complete\n", encoding="utf-8")

    matching_marker = colmap_attempt / ".matching-complete"
    if not matching_marker.is_file():
        run_logged(
            commands[1], log_dir / colmap_attempt.name / "colmap-matching.log"
        )
        matching_marker.write_text("complete\n", encoding="utf-8")

    if rig_enabled:
        camera_ids = validate_folder_camera_ids(
            colmap_attempt / "database.db", attempt.images_per_equirect
        )
        write_rig_config(rig_config_path, camera_ids, attempt)

    mapping_marker = colmap_attempt / ".mapping-complete"
    if not mapping_marker.is_file():
        run_logged(commands[2], log_dir / colmap_attempt.name / "colmap-mapping.log")

    components = _colmap_components(sparse_root)
    if not components:
        raise RuntimeError("COLMAP mapper completed without a valid sparse model")
    mapping_marker.write_text("complete\n", encoding="utf-8")
    selected_dir, _ = max(components, key=lambda item: len(item[1]))
    selected_model = selected_dir
    rig_metrics: dict[str, Any] = {"enabled": False}
    if rig_enabled:
        pre_rig = sparse_model_metrics(selected_dir)
        rig_marker = colmap_attempt / ".rig-complete"
        if rig_marker.is_file():
            rig_output_name = rig_marker.read_text(encoding="utf-8").strip() or "rig-sparse"
            rig_output = colmap_attempt / rig_output_name
        else:
            first_output = colmap_attempt / "rig-sparse"
            if not first_output.exists() or not any(first_output.iterdir()):
                rig_output = first_output
            else:
                retries = sorted(colmap_attempt.glob("rig-sparse-attempt-*"))
                rig_output = colmap_attempt / f"rig-sparse-attempt-{len(retries) + 2:03d}"
            rig_output.mkdir(parents=True, exist_ok=True)
            rig_command = [
                "colmap",
                "rig_bundle_adjuster",
                "--input_path",
                str(selected_dir),
                "--output_path",
                str(rig_output),
                "--rig_config_path",
                str(rig_config_path),
                "--estimate_rig_relative_poses",
                "0",
                "--RigBundleAdjustment.refine_relative_poses",
                "0",
                "--BundleAdjustment.refine_focal_length",
                "0",
                "--BundleAdjustment.refine_principal_point",
                "0",
                "--BundleAdjustment.refine_extra_params",
                "0",
            ]
            run_logged(
                rig_command,
                log_dir / colmap_attempt.name / "colmap-rig-bundle-adjustment.log",
            )
            required = [rig_output / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
            if not all(path.is_file() for path in required):
                raise RuntimeError("COLMAP rig bundle adjustment produced an incomplete model")
            rig_marker.write_text(rig_output.name + "\n", encoding="utf-8")
        required = [rig_output / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"Recorded rig model is incomplete: {rig_output}")
        selected_model = rig_output
        post_rig = sparse_model_metrics(selected_model)
        limit = float(getattr(settings, "rig_center_spread_ratio_limit", 0.001))
        observed = float(post_rig["p95_spread_to_baseline"])
        if observed > limit:
            raise RuntimeError(
                f"Rig center-spread gate failed: {observed:.6f} > {limit:.6f}"
            )
        rig_metrics = {
            "enabled": True,
            "config_path": rig_config_path.relative_to(dataset).as_posix(),
            "pre_bundle_adjustment": pre_rig,
            "post_bundle_adjustment": post_rig,
            "p95_spread_ratio_limit": limit,
            "gate": "passed",
        }
    (colmap_root / "selected-attempt.json").write_text(
        json.dumps(
            {
                "attempt": colmap_attempt.name,
                "component": selected_dir.name,
                "model": selected_model.relative_to(colmap_attempt).as_posix(),
                "rig": rig_metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    from nerfstudio.process_data.colmap_utils import colmap_to_json

    colmap_to_json(
        selected_model,
        dataset,
        use_single_camera_mode=not isinstance(attempt, ReconstructionAttempt),
    )
    attach_frame_masks(transforms, dataset / "masks")
    (colmap_attempt / ".conversion-complete").write_text("complete\n", encoding="utf-8")
    return reconstruction_metrics(dataset, attempt, settings)
