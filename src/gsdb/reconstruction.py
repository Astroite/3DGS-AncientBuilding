from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import tempfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .masking import (
    atomic_imwrite,
    image_dimensions,
    image_files,
    io_worker_count,
    run_image_tasks,
)
from .models import (
    LegacyReconstructionAttemptV1,
    LegacyReconstructionConfigV1,
    ReconstructionAttempt,
    ReconstructionConfig,
)
from .processes import run_logged


AttemptConfig = ReconstructionAttempt | LegacyReconstructionAttemptV1
ReconstructionSettings = ReconstructionConfig | LegacyReconstructionConfigV1

class _WriteQueue:
    """Bounded background encode/write queue that keeps the GPU loop moving.

    Each pending 2048x2048 BGR frame holds about 12 MiB, so the queue is capped
    rather than unbounded.
    """

    def __init__(self) -> None:
        self._workers = io_worker_count()
        self._pool = ThreadPoolExecutor(max_workers=self._workers)
        self._pending: deque[Future[Any]] = deque()
        self._limit = max(2, self._workers * 4)

    def submit(self, action: Callable[..., Any], *arguments: Any) -> None:
        while len(self._pending) >= self._limit:
            self._pending.popleft().result()
        self._pending.append(self._pool.submit(action, *arguments))

    def drain(self) -> None:
        while self._pending:
            self._pending.popleft().result()

    def __enter__(self) -> "_WriteQueue":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: Any) -> None:
        try:
            if exc_type is None:
                self.drain()
            else:
                # Let the original failure surface instead of a queued write's error.
                for future in self._pending:
                    future.cancel()
        finally:
            self._pool.shutdown(wait=True)


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


def validate_existing_projection_set(
    target: Path,
    frame_count: int,
    view_count: int,
    projection_size: int,
) -> set[Path]:
    """Validate and return reusable relative paths from a partial v2 projection."""
    if not target.exists():
        return set()
    if not target.is_dir():
        raise RuntimeError(f"Planar image target is not a directory: {target}")
    expected = {
        Path(f"view_{view_index:02d}") / f"frame_{frame_index:06d}.jpg"
        for frame_index in range(1, frame_count + 1)
        for view_index in range(view_count)
    }
    actual = {path.relative_to(target) for path in image_files(target)}
    unexpected = sorted(actual - expected)
    if unexpected:
        raise RuntimeError(
            "Existing planar projection contains unexpected images: "
            + ", ".join(path.as_posix() for path in unexpected[:5])
        )

    def check(relative: Path) -> None:
        try:
            shape = image_dimensions(target / relative)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Existing planar projection is invalid: {relative.as_posix()}; {error}"
            ) from error
        if shape != (projection_size, projection_size):
            raise RuntimeError(
                f"Existing planar projection is invalid: {relative.as_posix()}; "
                f"expected {projection_size}x{projection_size}"
            )

    run_image_tasks(check, sorted(actual))
    return actual


def project_equirectangular_frames(
    source: Path,
    dataset: Path,
    attempt: AttemptConfig,
) -> list[Path]:
    expected = expected_planar_images(attempt)
    target = dataset / "images"
    if target.is_dir():
        existing = image_files(target)
        if len(existing) == expected:
            return existing
        if isinstance(attempt, LegacyReconstructionAttemptV1):
            raise RuntimeError(
                f"Existing planar image set is incomplete ({len(existing)}/{expected}); "
                "create a new run instead of overwriting partial projection output"
            )
    if target.exists():
        if not target.is_dir():
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
    device = torch.device("cuda")
    size = attempt.projection_size
    specs = projection_view_specs(attempt)
    reusable = validate_existing_projection_set(
        target, attempt.frame_count, len(specs), size
    )
    target.mkdir(parents=True, exist_ok=True)
    for view_index in range(len(specs)):
        (target / f"view_{view_index:02d}").mkdir(parents=True, exist_ok=True)

    def decode(frame_path: Path) -> np.ndarray:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode equirectangular frame: {frame_path}")
        return image

    with ThreadPoolExecutor(max_workers=1) as reader, _WriteQueue() as writes:
        prefetched: Future[np.ndarray] | None = None
        for frame_index, frame in enumerate(frames, start=1):
            frame_targets = [
                Path(f"view_{view_index:02d}") / f"frame_{frame_index:06d}.jpg"
                for view_index in range(len(specs))
            ]
            if all(relative in reusable for relative in frame_targets):
                print(
                    f"Planar projection: {frame_index}/{len(frames)} (reused)",
                    flush=True,
                )
                continue
            image = prefetched.result() if prefetched is not None else decode(frame)
            prefetched = None
            # Overlap the next equirectangular decode with this frame's GPU work.
            for later_index in range(frame_index, len(frames)):
                later_targets = [
                    Path(f"view_{view_index:02d}") / f"frame_{later_index + 1:06d}.jpg"
                    for view_index in range(len(specs))
                ]
                if not all(relative in reusable for relative in later_targets):
                    prefetched = reader.submit(decode, frames[later_index])
                    break
            tensor = torch.tensor(image, dtype=torch.float32, device=device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
            for view_index, (yaw, pitch) in enumerate(specs):
                relative = frame_targets[view_index]
                if relative in reusable:
                    continue
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
                writes.submit(
                    atomic_imwrite,
                    target / relative,
                    output,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
            print(f"Planar projection: {frame_index}/{len(frames)}", flush=True)
    produced = image_files(target)
    if len(produced) != expected:
        raise RuntimeError(f"Projection produced {len(produced)} images; expected {expected}")
    return image_files(target)


def pyramid_level_marker(dataset: Path, prefix: str, factor: int) -> Path:
    """Marker recording that a level was written and fully validated once."""
    return dataset / f".pyramid-complete-{prefix}_{factor}"


def _pyramid_level_is_current(
    target_dir: Path,
    relatives: Sequence[Path],
    source_shapes: dict[Path, tuple[int, int]],
    factor: int,
) -> bool:
    """Header-only recheck of a level a previous run already validated in full.

    Every file was written through ``atomic_imwrite``, so a name that exists holds
    complete content; the header confirms it is the level this factor expects.
    """
    if {item.relative_to(target_dir) for item in image_files(target_dir)} != set(relatives):
        return False

    def check(relative: Path) -> None:
        height, width = source_shapes[relative]
        expected = (max(1, height // factor), max(1, width // factor))
        if image_dimensions(target_dir / relative) != expected:
            raise RuntimeError(
                f"Existing pyramid file is invalid: {target_dir / relative}; "
                f"expected shape {expected}"
            )

    try:
        run_image_tasks(check, relatives)
    except (OSError, RuntimeError):
        return False
    return True


def build_image_pyramid(
    source_dir: Path,
    dataset: Path,
    prefix: str,
    num_downscales: int,
    is_mask: bool = False,
) -> None:
    if num_downscales < 1:
        return
    sources = image_files(source_dir)
    if not sources:
        raise RuntimeError(f"No pyramid source images found in {source_dir}")
    relatives = [source.relative_to(source_dir) for source in sources]
    flag = cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    arguments: list[int] = [] if is_mask else [cv2.IMWRITE_JPEG_QUALITY, 95]

    source_shapes: dict[Path, tuple[int, int]] | None = None
    pending: list[tuple[int, Path]] = []
    for level in range(1, num_downscales + 1):
        factor = 2**level
        target_dir = dataset / f"{prefix}_{factor}"
        if pyramid_level_marker(dataset, prefix, factor).is_file():
            if source_shapes is None:
                source_shapes = dict(
                    zip(
                        relatives,
                        run_image_tasks(
                            lambda relative: image_dimensions(source_dir / relative),
                            relatives,
                        ),
                    )
                )
            if _pyramid_level_is_current(target_dir, relatives, source_shapes, factor):
                continue
        pending.append((factor, target_dir))
    if not pending:
        return
    for _, target_dir in pending:
        target_dir.mkdir(parents=True, exist_ok=True)

    def build(relative: Path) -> None:
        source = source_dir / relative
        image = cv2.imread(str(source), flag)
        if image is None:
            raise RuntimeError(f"Cannot decode pyramid source: {source}")
        height, width = image.shape[:2]
        for factor, target_dir in pending:
            target = target_dir / relative
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
            resized = cv2.resize(
                image,
                (expected_shape[1], expected_shape[0]),
                interpolation=interpolation,
            )
            atomic_imwrite(target, resized, arguments)

    run_image_tasks(build, relatives)
    for factor, target_dir in pending:
        actual = image_files(target_dir)
        if {item.relative_to(target_dir) for item in actual} != set(relatives):
            raise RuntimeError(
                f"Pyramid {target_dir.name} does not exactly match its source file set"
            )
        pyramid_level_marker(dataset, prefix, factor).write_text(
            "complete\n", encoding="utf-8"
        )


def pinhole_camera_parameters(image_path: Path, fov_degrees: float) -> str:
    height, width = image_dimensions(image_path)
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return f"{focal:.10f},{focal:.10f},{width / 2.0:.10f},{height / 2.0:.10f}"


def cross_view_pair_names(attempt: AttemptConfig) -> list[tuple[str, str]]:
    """Every within-frame view pair of one panorama, in deterministic order.

    COLMAP 3.8's sequential matcher orders images by *name*, so ``view_XX/`` folders
    are matched as independent temporal chains and the eight views of a single
    panorama never see each other. The mapper then splits the scene into one
    component per view. These pairs supply the missing links: they share an optical
    center, so their two-view geometry is a homography that cannot be triangulated
    on its own, but they join feature tracks that neighbouring frames with a real
    baseline do triangulate. Non-overlapping pairs such as opposing views are
    rejected by COLMAP's own geometric verification, so the full set is safe to emit.
    """
    views = attempt.images_per_equirect
    return [
        (
            f"view_{first:02d}/frame_{frame:06d}.jpg",
            f"view_{second:02d}/frame_{frame:06d}.jpg",
        )
        for frame in range(1, attempt.frame_count + 1)
        for first in range(views)
        for second in range(first + 1, views)
    ]


def write_cross_view_pair_list(path: Path, attempt: AttemptConfig) -> int:
    pairs = cross_view_pair_names(attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{first} {second}\n" for first, second in pairs), encoding="utf-8"
    )
    return len(pairs)


def build_colmap_commands(
    dataset: Path,
    attempt: AttemptConfig,
    settings: ReconstructionSettings | None = None,
    colmap_attempt_dir: Path | None = None,
    mapper_num_threads: int | None = None,
) -> dict[str, list[str]]:
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
            "matches_importer",
            "--database_path",
            str(database),
            "--match_list_path",
            str(colmap_attempt_dir / "cross-view-pairs.txt"),
            "--match_type",
            "pairs",
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
    named = dict(
        zip(("features", "matching", "cross_view", "mapping"), commands)
    )
    if not is_v2:
        # Legacy v1 datasets are flat, so there are no per-view folders to link.
        del named["cross_view"]
    if use_gpu:
        named["features"].extend(["--SiftExtraction.gpu_index", str(gpu_index)])
        for key in ("matching", "cross_view"):
            if key in named:
                named[key].extend(["--SiftMatching.gpu_index", str(gpu_index)])
    if fix_intrinsics:
        named["mapping"].extend(
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
        named["mapping"].extend(["--Mapper.num_threads", str(mapper_num_threads)])
    return named


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
    """Frame-major order for the COLMAP image list and rig-facing outputs.

    This is presentation order only. COLMAP 3.8's sequential matcher orders
    images by name regardless of image ID, so cross-view links come from
    :func:`cross_view_pair_names` rather than from the ID layout.
    """
    names = [path.relative_to(images_dir).as_posix() for path in image_files(images_dir)]
    return sorted(names, key=_image_order_key)


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
    available = [
        (index, spec, f"view_{index:02d}/")
        for index, spec in enumerate(specs)
        if f"view_{index:02d}/" in camera_ids
    ]
    if not available:
        raise RuntimeError("Cannot build a rig config without model cameras")
    _, reference_spec, reference_prefix = available[0]
    reference_world_from_camera = projection_world_from_camera(*reference_spec)
    cameras: list[dict[str, Any]] = []
    for _, spec, prefix in available:
        entry: dict[str, Any] = {
            "camera_id": camera_ids[prefix],
            "image_prefix": prefix,
            # COLMAP 3.8 flips estimate_rig_relative_poses back on when either
            # field is missing from any camera, including the reference camera.
            "rel_qvec": [1.0, 0.0, 0.0, 0.0],
            "rel_tvec": [0.0, 0.0, 0.0],
        }
        if prefix != reference_prefix:
            world_from_camera = projection_world_from_camera(*spec)
            camera_from_rig = world_from_camera.T @ reference_world_from_camera
            # COLMAP 3.8's rig JSON schema uses rel_qvec/rel_tvec. Newer
            # cam_from_rig_* names are silently ignored by the locked binary.
            entry["rel_qvec"] = rotation_matrix_to_quaternion_wxyz(
                camera_from_rig
            )
        cameras.append(entry)
    return [{"ref_camera_id": camera_ids[reference_prefix], "cameras": cameras}]


def camera_ids_for_model(
    model_dir: Path, database_camera_ids: dict[str, int]
) -> dict[str, int]:
    """Return only database camera folders present in a sparse component."""
    from nerfstudio.process_data.colmap_utils import read_cameras_binary

    model_camera_ids = set(read_cameras_binary(model_dir / "cameras.bin"))
    selected = {
        prefix: camera_id
        for prefix, camera_id in database_camera_ids.items()
        if camera_id in model_camera_ids
    }
    if set(selected.values()) != model_camera_ids:
        raise RuntimeError(
            "Sparse model cameras do not match the database folder-camera mapping"
        )
    if not selected:
        raise RuntimeError("Selected sparse model does not contain a rig camera")
    return selected


def _average_rotations(
    rotations: list[np.ndarray], determinant: int = 1
) -> np.ndarray:
    if not rotations:
        raise ValueError("Cannot average an empty rotation set")
    if determinant not in (-1, 1):
        raise ValueError("Orthogonal-matrix determinant must be -1 or 1")
    left, _, right = np.linalg.svd(np.sum(rotations, axis=0))
    correction = np.eye(3)
    correction[2, 2] = determinant * np.linalg.det(left @ right)
    return left @ correction @ right


def _component_rig_poses(
    images: dict[int, Any], attempt: ReconstructionAttempt
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Infer one rig center/orientation per frame from a sparse component."""
    specs = projection_view_specs(attempt)
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for image in images.values():
        match = _NESTED_FRAME_PATTERN.match(image.name.replace("\\", "/"))
        if match is None:
            raise RuntimeError(f"Cannot infer rig pose from image name: {image.name}")
        view_index, frame = (int(value) for value in match.groups())
        if view_index >= len(specs):
            raise RuntimeError(f"Projected view index is out of range: {image.name}")
        camera_from_world = _qvec_to_rotation(image.qvec)
        world_from_camera = camera_from_world.T
        center = -(world_from_camera @ np.asarray(image.tvec, dtype=np.float64))
        rig_from_camera = projection_world_from_camera(*specs[view_index])
        world_from_rig = world_from_camera @ rig_from_camera.T
        grouped.setdefault(frame, []).append((center, world_from_rig))
    return {
        frame: (
            np.mean([item[0] for item in values], axis=0),
            # Nerfstudio's equirectangular [forward, right, up] convention is
            # left-handed, so these per-frame rig transforms have det=-1.
            _average_rotations([item[1] for item in values], determinant=-1),
        )
        for frame, values in grouped.items()
    }


def _rig_component_alignment(
    reference: dict[int, tuple[np.ndarray, np.ndarray]],
    moving: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Estimate ``reference = scale * rotation * moving + translation``."""
    common = sorted(set(reference) & set(moving))
    if len(common) < 6:
        raise RuntimeError(
            f"Rig components share only {len(common)} frames; at least 6 are required"
        )
    rotations = [reference[frame][1] @ moving[frame][1].T for frame in common]
    rotation = _average_rotations(rotations)
    moving_centers = np.vstack([moving[frame][0] for frame in common])
    reference_centers = np.vstack([reference[frame][0] for frame in common])
    rotated = moving_centers @ rotation.T
    moving_mean = rotated.mean(axis=0)
    reference_mean = reference_centers.mean(axis=0)
    centered_moving = rotated - moving_mean
    centered_reference = reference_centers - reference_mean
    denominator = float(np.sum(centered_moving * centered_moving))
    if denominator <= np.finfo(np.float64).eps:
        raise RuntimeError("Rig component trajectory has no usable translation baseline")
    scale = float(np.sum(centered_moving * centered_reference) / denominator)
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Rig component alignment produced invalid scale: {scale}")
    translation = reference_mean - scale * moving_mean
    aligned_centers = scale * rotated + translation
    residuals = np.linalg.norm(aligned_centers - reference_centers, axis=1)
    baselines = np.linalg.norm(np.diff(reference_centers, axis=0), axis=1)
    positive_baselines = baselines[baselines > np.finfo(np.float64).eps]
    median_baseline = (
        float(np.median(positive_baselines)) if positive_baselines.size else 0.0
    )
    if median_baseline <= 0:
        raise RuntimeError("Reference rig trajectory has no usable inter-frame baseline")
    trajectory_extent = float(
        np.percentile(
            np.linalg.norm(reference_centers - reference_mean, axis=1), 95
        )
    )
    if trajectory_extent <= 0:
        raise RuntimeError("Reference rig trajectory has no usable spatial extent")
    rotation_errors = []
    for frame in common:
        delta = rotation @ moving[frame][1] @ reference[frame][1].T
        cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(math.degrees(math.acos(cosine)))
    metrics: dict[str, float | int] = {
        "common_frames": len(common),
        "scale": scale,
        "center_residual_median": float(np.median(residuals)),
        "center_residual_p95": float(np.percentile(residuals, 95)),
        "median_interframe_baseline": median_baseline,
        "center_residual_p95_to_baseline": float(np.percentile(residuals, 95))
        / median_baseline,
        "reference_trajectory_extent_p95": trajectory_extent,
        "center_residual_p95_to_trajectory_extent": float(
            np.percentile(residuals, 95)
        )
        / trajectory_extent,
        "rotation_error_median_degrees": float(np.median(rotation_errors)),
        "rotation_error_p95_degrees": float(np.percentile(rotation_errors, 95)),
    }
    return scale, rotation, translation, metrics


def _write_colmap_binary_model(
    output: Path,
    cameras: dict[int, Any],
    images: dict[int, Any],
    points: dict[int, Any],
) -> None:
    """Write the COLMAP 3.8 binary subset used by the rig merge."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(cameras)))
        for camera_id, camera in sorted(cameras.items()):
            if camera.model != "PINHOLE" or len(camera.params) != 4:
                raise RuntimeError(
                    f"Rig merge supports only four-parameter PINHOLE cameras, got {camera.model}"
                )
            stream.write(
                struct.pack(
                    "<iiQQ4d",
                    int(camera_id),
                    1,
                    int(camera.width),
                    int(camera.height),
                    *(float(value) for value in camera.params),
                )
            )
    with (output / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(images)))
        for image_id, image in sorted(images.items()):
            stream.write(
                struct.pack(
                    "<i7di",
                    int(image_id),
                    *(float(value) for value in image.qvec),
                    *(float(value) for value in image.tvec),
                    int(image.camera_id),
                )
            )
            stream.write(image.name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", len(image.xys)))
            for xy, point_id in zip(image.xys, image.point3D_ids):
                stream.write(
                    struct.pack(
                        "<ddq", float(xy[0]), float(xy[1]), int(point_id)
                    )
                )
    with (output / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, point in sorted(points.items()):
            error = float(np.asarray(point.error).reshape(-1)[0])
            stream.write(
                struct.pack(
                    "<Q3d3BdQ",
                    int(point_id),
                    *(float(value) for value in point.xyz),
                    *(int(value) for value in point.rgb),
                    error,
                    len(point.image_ids),
                )
            )
            for image_id, point2d_index in zip(point.image_ids, point.point2D_idxs):
                stream.write(struct.pack("<ii", int(image_id), int(point2d_index)))


def merge_rig_components(
    sparse_root: Path,
    output: Path,
    attempt: ReconstructionAttempt,
) -> dict[str, Any]:
    """Align disconnected COLMAP components using shared-frame virtual-rig poses."""
    from nerfstudio.process_data.colmap_utils import (
        read_cameras_binary,
        read_images_binary,
        read_points3D_binary,
    )

    components: list[tuple[Path, dict[int, Any]]] = []
    for candidate, images in _colmap_components(sparse_root):
        if candidate.name.isdigit():
            components.append((candidate, images))
    if len(components) < 2:
        raise RuntimeError("Rig component merge requires at least two mapper components")
    poses = {
        candidate: _component_rig_poses(images, attempt)
        for candidate, images in components
    }
    reference_path, reference_images = max(
        components,
        key=lambda item: (
            len(poses[item[0]]),
            len(item[1]),
            any(image.name.replace("\\", "/").startswith("view_00/") for image in item[1].values()),
        ),
    )
    alignment_parts: list[dict[str, Any]] = [
        {
            "path": reference_path,
            "frames": set(poses[reference_path]),
            "transform": (1.0, np.eye(3), np.zeros(3)),
            "label": reference_path.name,
        }
    ]
    accepted_paths = {reference_path}
    alignment_metrics: dict[str, Any] = {
        reference_path.name: {"reference": True, "frames": len(poses[reference_path])}
    }
    rejected: dict[str, str] = {}

    def gate_error(metrics: dict[str, float | int]) -> str | None:
        if float(metrics["center_residual_p95_to_trajectory_extent"]) > 0.055:
            return "p95 center residual exceeds 5.5% of the trajectory extent"
        if float(metrics["rotation_error_p95_degrees"]) > 5.0:
            return "p95 rig rotation residual exceeds 5 degrees"
        return None

    for candidate, _ in components:
        if candidate == reference_path:
            continue
        try:
            scale, rotation, translation, metrics = _rig_component_alignment(
                poses[reference_path], poses[candidate]
            )
            failure = gate_error(metrics)
            if failure is None:
                alignment_parts.append(
                    {
                        "path": candidate,
                        "frames": set(poses[candidate]),
                        "transform": (scale, rotation, translation),
                        "label": candidate.name,
                    }
                )
                accepted_paths.add(candidate)
                alignment_metrics[candidate.name] = metrics
                continue

            # Monocular trajectories can accumulate scale drift in one interval
            # while remaining accurate elsewhere. Recover only independently
            # gated, non-overlapping 30-frame windows when orientation is sound.
            if float(metrics["rotation_error_p95_degrees"]) > 5.0:
                raise RuntimeError(failure)
            common = sorted(set(poses[reference_path]) & set(poses[candidate]))
            runs: list[list[int]] = []
            for frame in common:
                if not runs or frame != runs[-1][-1] + 1:
                    runs.append([])
                runs[-1].append(frame)
            segment_metrics: dict[str, Any] = {}
            for run in runs:
                for start in range(0, len(run), 30):
                    frames = run[start : start + 30]
                    if len(frames) < 10:
                        continue
                    segment_reference = {
                        frame: poses[reference_path][frame] for frame in frames
                    }
                    segment_moving = {frame: poses[candidate][frame] for frame in frames}
                    try:
                        segment_scale, segment_rotation, segment_translation, segment = (
                            _rig_component_alignment(segment_reference, segment_moving)
                        )
                    except RuntimeError:
                        continue
                    if gate_error(segment) is not None:
                        continue
                    label = f"{candidate.name}:frames-{frames[0]:06d}-{frames[-1]:06d}"
                    alignment_parts.append(
                        {
                            "path": candidate,
                            "frames": set(frames),
                            "transform": (
                                segment_scale,
                                segment_rotation,
                                segment_translation,
                            ),
                            "label": label,
                        }
                    )
                    segment_metrics[label] = segment
            if not segment_metrics:
                raise RuntimeError(failure)
            accepted_paths.add(candidate)
            alignment_metrics[candidate.name] = {
                "global_rejection": failure,
                "segments": segment_metrics,
            }
        except RuntimeError as error:
            rejected[candidate.name] = str(error)
    if len(alignment_parts) < 2:
        raise RuntimeError(
            "No additional COLMAP component passed the virtual-rig alignment gates"
        )

    merged_cameras: dict[int, Any] = {}
    merged_images: dict[int, Any] = {}
    merged_points: dict[int, Any] = {}
    seen_names: set[str] = set()
    next_point_id = 1
    duplicate_images = 0
    contributing_components: list[str] = []
    images_by_path = dict(components)
    alignment_parts.sort(
        key=lambda part: (
            part["path"] != reference_path,
            -len(part["frames"]),
            int(part["path"].name),
            part["label"],
        )
    )
    camera_cache: dict[Path, dict[int, Any]] = {}
    point_cache: dict[Path, dict[int, Any]] = {}
    for part in alignment_parts:
        candidate = part["path"]
        component_images = {
            image_id: image
            for image_id, image in images_by_path[candidate].items()
            if _frame_number(image.name) in part["frames"]
        }
        for image_id, image in component_images.items():
            previous = merged_images.get(image_id)
            if previous is not None and previous.name != image.name:
                raise RuntimeError(
                    f"COLMAP image ID {image_id} names differ across components"
                )
        included_image_ids = {
            int(image_id)
            for image_id, image in component_images.items()
            if image.name not in seen_names and image_id not in merged_images
        }
        duplicate_images += len(component_images) - len(included_image_ids)
        if not included_image_ids:
            continue
        contributing_components.append(str(part["label"]))
        if candidate not in camera_cache:
            camera_cache[candidate] = read_cameras_binary(candidate / "cameras.bin")
        if candidate not in point_cache:
            point_cache[candidate] = read_points3D_binary(candidate / "points3D.bin")
        component_cameras = camera_cache[candidate]
        component_points = point_cache[candidate]
        for camera_id, camera in component_cameras.items():
            previous = merged_cameras.get(camera_id)
            if previous is not None and (
                previous.model != camera.model
                or previous.width != camera.width
                or previous.height != camera.height
                or not np.allclose(previous.params, camera.params)
            ):
                raise RuntimeError(f"Camera {camera_id} differs across sparse components")
            merged_cameras[camera_id] = camera
        point_id_map: dict[int, int] = {}
        filtered_tracks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for old_id, point in sorted(component_points.items()):
            keep = np.isin(point.image_ids, list(included_image_ids))
            if not np.any(keep):
                continue
            point_id_map[int(old_id)] = next_point_id
            next_point_id += 1
            filtered_tracks[int(old_id)] = (
                np.asarray(point.image_ids)[keep],
                np.asarray(point.point2D_idxs)[keep],
            )
        scale, alignment_rotation, translation = part["transform"]
        for old_id, point in component_points.items():
            if int(old_id) not in point_id_map:
                continue
            new_id = point_id_map[int(old_id)]
            xyz = scale * (alignment_rotation @ np.asarray(point.xyz)) + translation
            image_ids, point2d_indices = filtered_tracks[int(old_id)]
            merged_points[new_id] = point._replace(
                id=new_id,
                xyz=xyz,
                image_ids=image_ids,
                point2D_idxs=point2d_indices,
            )
        for image_id, image in component_images.items():
            if int(image_id) not in included_image_ids:
                continue
            camera_from_component = _qvec_to_rotation(image.qvec)
            component_from_camera = camera_from_component.T
            component_center = -(component_from_camera @ np.asarray(image.tvec))
            merged_center = scale * (alignment_rotation @ component_center) + translation
            merged_from_camera = alignment_rotation @ component_from_camera
            camera_from_merged = merged_from_camera.T
            merged_tvec = -(camera_from_merged @ merged_center)
            merged_qvec = np.asarray(
                rotation_matrix_to_quaternion_wxyz(camera_from_merged),
                dtype=np.float64,
            )
            merged_point_ids = np.asarray(
                [
                    point_id_map.get(int(point_id), -1)
                    if int(point_id) >= 0
                    else -1
                    for point_id in image.point3D_ids
                ],
                dtype=np.int64,
            )
            merged_images[image_id] = image._replace(
                qvec=merged_qvec,
                tvec=merged_tvec,
                point3D_ids=merged_point_ids,
            )
            seen_names.add(image.name)

    _write_colmap_binary_model(output, merged_cameras, merged_images, merged_points)
    verified_images = read_images_binary(output / "images.bin")
    verified_points = read_points3D_binary(output / "points3D.bin")
    if set(verified_images) != set(merged_images) or {
        image.name for image in verified_images.values()
    } != seen_names:
        raise RuntimeError("Rig-merged COLMAP image verification failed")
    if len(verified_points) != len(merged_points):
        raise RuntimeError("Rig-merged COLMAP point verification failed")
    metrics = {
        "merge_algorithm_version": 2,
        "enabled": True,
        "reference_component": reference_path.name,
        "source_component_count": len(components),
        "accepted_component_count": len(accepted_paths),
        "accepted_alignment_part_count": len(alignment_parts),
        "contributing_components": contributing_components,
        "rejected_components": rejected,
        "duplicate_images_deduplicated": duplicate_images,
        "merged_images": len(merged_images),
        "merged_points": len(merged_points),
        "alignments": alignment_metrics,
    }
    (output / "merge-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


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
    named_centers: list[tuple[str, np.ndarray]],
    required_prefix: str | None = None,
) -> dict[str, float | int]:
    grouped: dict[int, list[tuple[str, np.ndarray]]] = {}
    for name, center in named_centers:
        grouped.setdefault(_frame_number(name), []).append(
            (name.replace("\\", "/"), np.asarray(center, dtype=np.float64))
        )
    centroids: list[tuple[int, np.ndarray]] = []
    spreads: list[float] = []
    for frame, named_group in grouped.items():
        if required_prefix is not None:
            if len(named_group) < 2 or not any(
                name.startswith(required_prefix) for name, _ in named_group
            ):
                # COLMAP 3.8 leaves frames without the rig's reference camera
                # as independent poses, so they are outside the rig gate.
                continue
        matrix = np.vstack([center for _, center in named_group])
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
        "frame_count": len(centroids),
        "center_spread_median": float(np.median(spreads)) if spreads else 0.0,
        "center_spread_p95": p95_spread,
        "center_spread_max": max(spreads, default=0.0),
        "median_interframe_baseline": median_baseline,
        "p95_spread_to_baseline": p95_spread / median_baseline if median_baseline else 0.0,
    }


def sparse_model_metrics(
    model_dir: Path, required_prefix: str | None = None
) -> dict[str, Any]:
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
    return {
        **center_spread_metrics(named_centers, required_prefix=required_prefix),
        "intrinsics": intrinsics,
    }


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
        "cross_view_pairs": _recorded_cross_view_pairs(dataset),
        "rig": selection.get("rig", {"enabled": False}),
    }


def _recorded_cross_view_pairs(dataset: Path) -> int:
    """Pairs imported by the within-frame view matcher for the selected attempt."""
    selection_path = dataset / "colmap" / "selected-attempt.json"
    attempt_name = None
    if selection_path.is_file():
        attempt_name = json.loads(selection_path.read_text(encoding="utf-8")).get("attempt")
    candidates = (
        [dataset / "colmap" / str(attempt_name)]
        if attempt_name
        else sorted((dataset / "colmap").glob("attempt-*"))
    )
    for candidate in candidates:
        marker = candidate / ".cross-view-matching-complete"
        if marker.is_file():
            return int(marker.read_text(encoding="utf-8").strip() or 0)
    return 0


def run_masked_colmap(
    dataset: Path,
    attempt: AttemptConfig,
    log_dir: Path,
    settings: ReconstructionSettings | None = None,
) -> dict[str, Any]:
    transforms = dataset / "transforms.json"
    if transforms.is_file():
        try:
            existing_metrics = reconstruction_metrics(dataset, attempt, settings)
            threshold = float(getattr(settings, "registration_threshold", 0.70))
            needs_rig_merge = (
                isinstance(attempt, ReconstructionAttempt)
                and attempt.use_rig
                and float(existing_metrics["largest_component_coverage"])
                < threshold
            )
            if not needs_rig_merge:
                return existing_metrics
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
            commands["features"], log_dir / colmap_attempt.name / "colmap-features.log"
        )
        feature_marker.write_text("complete\n", encoding="utf-8")

    matching_marker = colmap_attempt / ".matching-complete"
    if not matching_marker.is_file():
        run_logged(
            commands["matching"], log_dir / colmap_attempt.name / "colmap-matching.log"
        )
        matching_marker.write_text("complete\n", encoding="utf-8")

    cross_view_marker = colmap_attempt / ".cross-view-matching-complete"
    if "cross_view" in commands and not cross_view_marker.is_file():
        pair_count = write_cross_view_pair_list(
            colmap_attempt / "cross-view-pairs.txt", attempt
        )
        run_logged(
            commands["cross_view"],
            log_dir / colmap_attempt.name / "colmap-cross-view-matching.log",
        )
        cross_view_marker.write_text(f"{pair_count}\n", encoding="utf-8")

    if rig_enabled:
        database_camera_ids = validate_folder_camera_ids(
            colmap_attempt / "database.db", attempt.images_per_equirect
        )

    mapping_marker = colmap_attempt / ".mapping-complete"
    if not mapping_marker.is_file():
        run_logged(commands["mapping"], log_dir / colmap_attempt.name / "colmap-mapping.log")

    components = _colmap_components(sparse_root)
    if not components:
        raise RuntimeError("COLMAP mapper completed without a valid sparse model")
    mapping_marker.write_text("complete\n", encoding="utf-8")
    mapper_components = [item for item in components if item[0].name.isdigit()]
    selection_pool = mapper_components or components
    selected_dir, _ = max(selection_pool, key=lambda item: len(item[1]))
    component_merge: dict[str, Any] = {"enabled": False}
    if rig_enabled:
        raw_components = mapper_components
        raw_names = {
            image.name for _, images in raw_components for image in images.values()
        }
        largest_raw_count = max(len(images) for _, images in raw_components)
        if len(raw_components) > 1 and len(raw_names) > max(
            len(images) for _, images in raw_components
        ):
            merge_marker = colmap_attempt / ".component-merge-complete"
            merged_dir: Path | None = None
            if merge_marker.is_file():
                recorded = merge_marker.read_text(encoding="utf-8").strip()
                candidate = colmap_attempt / recorded
                required = [
                    candidate / name
                    for name in ("cameras.bin", "images.bin", "points3D.bin")
                ]
                if all(path.is_file() for path in required):
                    from nerfstudio.process_data.colmap_utils import read_images_binary

                    recorded_images = read_images_binary(candidate / "images.bin")
                    recorded_names = {
                        image.name for image in recorded_images.values()
                    }
                    if (
                        recorded_names.issubset(raw_names)
                        and len(recorded_names) > largest_raw_count
                    ):
                        metrics_path = candidate / "merge-metrics.json"
                        if metrics_path.is_file():
                            recorded_metrics = json.loads(
                                metrics_path.read_text(encoding="utf-8")
                            )
                            if recorded_metrics.get("merge_algorithm_version") == 2:
                                merged_dir = candidate
                                component_merge = recorded_metrics
            if merged_dir is None:
                first_merged = sparse_root / "rig-merged"
                existing_merged = [
                    first_merged,
                    *sorted(sparse_root.glob("rig-merged-attempt-*")),
                ]
                for existing in reversed(existing_merged):
                    required = [
                        existing / name
                        for name in ("cameras.bin", "images.bin", "points3D.bin")
                    ]
                    metrics_path = existing / "merge-metrics.json"
                    if not all(path.is_file() for path in required) or not metrics_path.is_file():
                        continue
                    from nerfstudio.process_data.colmap_utils import read_images_binary

                    existing_images = read_images_binary(existing / "images.bin")
                    existing_names = {image.name for image in existing_images.values()}
                    existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if (
                        existing_metrics.get("merge_algorithm_version") == 2
                        and existing_names.issubset(raw_names)
                        and len(existing_names) > largest_raw_count
                    ):
                        merged_dir = existing
                        component_merge = existing_metrics
                        break
                if merged_dir is None:
                    if not first_merged.exists() or not any(first_merged.iterdir()):
                        candidate = first_merged
                    else:
                        retries = sorted(sparse_root.glob("rig-merged-attempt-*"))
                        candidate = sparse_root / f"rig-merged-attempt-{len(retries) + 2:03d}"
                    try:
                        component_merge = merge_rig_components(
                            sparse_root, candidate, attempt
                        )
                        from nerfstudio.process_data.colmap_utils import read_images_binary

                        candidate_images = read_images_binary(candidate / "images.bin")
                        candidate_names = {
                            image.name for image in candidate_images.values()
                        }
                        if (
                            not candidate_names.issubset(raw_names)
                            or len(candidate_names) <= largest_raw_count
                        ):
                            raise RuntimeError(
                                "Rig component merge did not improve mapper coverage"
                            )
                        merged_dir = candidate
                    except RuntimeError as error:
                        component_merge = {"enabled": False, "reason": str(error)}
                        print(f"Rig component merge skipped: {error}", flush=True)
                if merged_dir is not None:
                    merge_marker.write_text(
                        merged_dir.relative_to(colmap_attempt).as_posix() + "\n",
                        encoding="utf-8",
                    )
            if merged_dir is not None:
                from nerfstudio.process_data.colmap_utils import read_images_binary

                merged_images = read_images_binary(merged_dir / "images.bin")
                threshold = float(getattr(settings, "registration_threshold", 0.70))
                if len(merged_images) >= math.ceil(
                    expected_planar_images(attempt) * threshold
                ):
                    selected_dir = merged_dir
    selected_model = selected_dir
    rig_metrics: dict[str, Any] = {"enabled": False}
    if rig_enabled:
        model_camera_ids = camera_ids_for_model(selected_dir, database_camera_ids)
        write_rig_config(rig_config_path, model_camera_ids, attempt)
        reference_prefix = next(iter(model_camera_ids))
        pre_rig = sparse_model_metrics(
            selected_dir, required_prefix=reference_prefix
        )
        rig_marker = colmap_attempt / ".rig-complete"
        from nerfstudio.process_data.colmap_utils import read_images_binary

        selected_image_names = {
            image.name
            for image in read_images_binary(selected_dir / "images.bin").values()
        }
        reuse_rig_output = False
        if rig_marker.is_file():
            rig_output_name = rig_marker.read_text(encoding="utf-8").strip() or "rig-sparse"
            rig_output = colmap_attempt / rig_output_name
            required = [
                rig_output / name for name in ("cameras.bin", "images.bin", "points3D.bin")
            ]
            if all(path.is_file() for path in required):
                reuse_rig_output = {
                    image.name
                    for image in read_images_binary(rig_output / "images.bin").values()
                } == selected_image_names
        if not reuse_rig_output:
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
        required = [rig_output / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"Recorded rig model is incomplete: {rig_output}")
        selected_model = rig_output
        post_rig = sparse_model_metrics(
            selected_model, required_prefix=reference_prefix
        )
        limit = float(getattr(settings, "rig_center_spread_ratio_limit", 0.001))
        observed = float(post_rig["p95_spread_to_baseline"])
        if observed > limit:
            raise RuntimeError(
                f"Rig center-spread gate failed: {observed:.6f} > {limit:.6f}"
            )
        rig_marker.write_text(rig_output.name + "\n", encoding="utf-8")
        rig_metrics = {
            "enabled": True,
            "config_path": rig_config_path.relative_to(dataset).as_posix(),
            "pre_bundle_adjustment": pre_rig,
            "post_bundle_adjustment": post_rig,
            "p95_spread_ratio_limit": limit,
            "gate": "passed",
            "component_merge": component_merge,
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
