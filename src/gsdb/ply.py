"""Validated Gaussian Splatting PLY inspection and axis conversion."""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FLOAT_PROPERTIES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *(f"f_rest_{index}" for index in range(45)),
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)

ROTATE_X_MINUS_90 = np.array(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    dtype=np.float64,
)


def read_ply_header(path: Path) -> tuple[bytes, int, list[str]]:
    with path.open("rb") as stream:
        header = bytearray()
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header has no end_header marker")
            header.extend(line)
            if line.strip() == b"end_header":
                break
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("Only binary_little_endian PLY 1.0 is supported")
    match = re.search(r"^element vertex (\d+)$", text, re.MULTILINE)
    if match is None:
        raise ValueError("PLY header has no vertex count")
    properties = re.findall(r"^property float (\S+)$", text, re.MULTILINE)
    return bytes(header), int(match.group(1)), properties


def gaussian_count(path: Path) -> int:
    """Read the vertex count without loading the potentially multi-GB body."""
    _, count, _ = read_ply_header(path)
    return count


def _updated_header(header: bytes, rotation: np.ndarray, vertex_count: int) -> bytes:
    """Rewrite the axis comment and the vertex count to match what is written."""
    text = header.decode("ascii")
    for axis in ("x", "y", "z"):
        text = text.replace(f"comment Vertical Axis: {axis}", "comment Vertical Axis: y")
    text = re.sub(
        r"^element vertex \d+$",
        f"element vertex {vertex_count}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    rows = "; ".join(
        "(" + ", ".join(f"{value:.9f}" for value in row) + ")" for row in rotation
    )
    note = f"comment Axis transform rows: {rows}\n"
    return text.replace("end_header", note + "end_header", 1).encode("ascii")


_PAIRWISE_BLOCK_BYTES = 64 << 20


def _nearest_camera_distance(points: np.ndarray, cameras: np.ndarray) -> np.ndarray:
    """Distance from each point to the closest camera, in memory-bounded blocks.

    The full pairwise matrix for a published scene is roughly 878k x 1067, so it is
    evaluated in slices sized from the reference count rather than a fixed row
    count — the same helper is called the other way round to check that the cameras
    sit inside the cloud, where a fixed 20k rows would allocate gigabytes.
    Deliberately plain NumPy: the environment has no SciPy, and a KD-tree would not
    pay for itself at this size.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    cameras = np.ascontiguousarray(cameras, dtype=np.float32)
    if not len(cameras):
        raise ValueError("Nearest-camera distance needs at least one camera")
    rows = max(1, _PAIRWISE_BLOCK_BYTES // (len(cameras) * 3 * 4))
    distances = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), rows):
        block = points[start : start + rows]
        gaps = block[:, None, :] - cameras[None, :, :]
        distances[start : start + rows] = np.sqrt(np.einsum("ijk,ijk->ij", gaps, gaps)).min(axis=1)
    return distances


def _sh_basis(directions: np.ndarray) -> np.ndarray:
    x, y, z = directions.T
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = np.array(
        [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396,
        ]
    )
    c3 = np.array(
        [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435,
        ]
    )
    basis = np.empty((directions.shape[0], 16), dtype=np.float64)
    basis[:, 0] = c0
    basis[:, 1] = -c1 * y
    basis[:, 2] = c1 * z
    basis[:, 3] = -c1 * x
    basis[:, 4] = c2[0] * xy
    basis[:, 5] = c2[1] * yz
    basis[:, 6] = c2[2] * (2.0 * zz - xx - yy)
    basis[:, 7] = c2[3] * xz
    basis[:, 8] = c2[4] * (xx - yy)
    basis[:, 9] = c3[0] * y * (3.0 * xx - yy)
    basis[:, 10] = c3[1] * xy * z
    basis[:, 11] = c3[2] * y * (4.0 * zz - xx - yy)
    basis[:, 12] = c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
    basis[:, 13] = c3[4] * x * (4.0 * zz - xx - yy)
    basis[:, 14] = c3[5] * z * (xx - yy)
    basis[:, 15] = c3[6] * x * (xx - 3.0 * yy)
    return basis


def _sh_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    sample_count = 256
    indices = np.arange(sample_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / sample_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    directions_new = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    directions_old = directions_new @ rotation
    basis_new = _sh_basis(directions_new)
    basis_old = _sh_basis(directions_old)
    transform = np.zeros((15, 15), dtype=np.float64)
    for start, stop in ((1, 4), (4, 9), (9, 16)):
        band_transform, *_ = np.linalg.lstsq(
            basis_new[:, start:stop], basis_old[:, start:stop], rcond=None
        )
        transform[start - 1 : stop - 1, start - 1 : stop - 1] = band_transform
    error = np.max(np.abs(basis_new[:, 1:] @ transform - basis_old[:, 1:]))
    if error > 1e-10:
        raise RuntimeError(f"SH rotation validation failed: maximum error {error}")
    return transform


def rotation_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Unit ``(w, x, y, z)`` for a proper rotation matrix, via the stable branch."""
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
        axis = int(np.argmax(np.diag(matrix)))
        other = [(axis + 1) % 3, (axis + 2) % 3]
        scale = (
            math.sqrt(1.0 + matrix[axis, axis] - matrix[other[0], other[0]] - matrix[other[1], other[1]])
            * 2.0
        )
        quaternion = np.empty(4)
        quaternion[0] = (matrix[other[1], other[0]] - matrix[other[0], other[1]]) / scale
        quaternion[1 + axis] = 0.25 * scale
        quaternion[1 + other[0]] = (matrix[other[0], axis] + matrix[axis, other[0]]) / scale
        quaternion[1 + other[1]] = (matrix[other[1], axis] + matrix[axis, other[1]]) / scale
    return quaternion / np.linalg.norm(quaternion)


def _rotate_quaternions_wxyz(quaternions: np.ndarray, rotation: np.ndarray) -> None:
    """Left-multiply each Gaussian orientation by an arbitrary world rotation."""
    rw, rx, ry, rz = rotation_to_quaternion_wxyz(rotation)
    w, x, y, z = quaternions.astype(np.float64, copy=True).T
    quaternions[:, 0] = rw * w - rx * x - ry * y - rz * z
    quaternions[:, 1] = rw * x + rx * w + ry * z - rz * y
    quaternions[:, 2] = rw * y - rx * z + ry * w + rz * x
    quaternions[:, 3] = rw * z + rx * y - ry * x + rz * w


@dataclass(frozen=True, eq=False)
class CullSpec:
    """Which Gaussians to drop when publishing, and how far the cull may go.

    Peripheral Gaussians in a night panorama capture are fully opaque — the median
    opacity beyond twice the capture radius measured 1.000 — so an opacity
    threshold removes fine nearby detail while leaving every floater in place.
    Distance to the capture path and absolute size are what actually separate them.
    """

    cameras: np.ndarray
    trajectory_radius: float
    distance_factor: float = 3.0
    scale_factor: float = 1.0
    max_removed_fraction: float = 0.05

    def limits(self) -> tuple[float, float]:
        return (
            self.distance_factor * self.trajectory_radius,
            self.scale_factor * self.trajectory_radius,
        )


def _cull_mask(data: np.ndarray, spec: CullSpec) -> tuple[np.ndarray, dict[str, Any]]:
    xyz = np.column_stack((data["x"], data["y"], data["z"]))
    distance = _nearest_camera_distance(xyz, spec.cameras)
    largest_scale = np.exp(
        np.column_stack((data["scale_0"], data["scale_1"], data["scale_2"])).astype(np.float64)
    ).max(axis=1)
    max_distance, max_scale = spec.limits()

    # Guard against a camera set that is not in the PLY's frame at all: if it were,
    # the cameras would sit far outside the cloud and this would delete the scene.
    camera_gap = float(np.median(_nearest_camera_distance(spec.cameras, xyz)))
    if camera_gap > spec.trajectory_radius:
        raise RuntimeError(
            f"Cameras sit a median {camera_gap:.3f} from the nearest Gaussian, beyond the "
            f"{spec.trajectory_radius:.3f} capture radius; they are not in the PLY's frame"
        )

    too_far = distance > max_distance
    too_large = largest_scale > max_scale
    removed = too_far | too_large
    fraction = float(removed.mean())
    metrics: dict[str, Any] = {
        "input_gaussians": int(len(data)),
        "removed_total": int(removed.sum()),
        "removed_beyond_distance": int(too_far.sum()),
        "removed_above_scale": int(too_large.sum()),
        "removed_fraction": fraction,
        "distance_factor": spec.distance_factor,
        "scale_factor": spec.scale_factor,
        "trajectory_radius": spec.trajectory_radius,
        "max_distance": max_distance,
        "max_scale": max_scale,
        "camera_to_cloud_median": camera_gap,
        "max_removed_fraction": spec.max_removed_fraction,
    }
    if fraction > spec.max_removed_fraction:
        raise RuntimeError(
            f"Culling would remove {fraction:.2%} of the Gaussians, above the "
            f"{spec.max_removed_fraction:.2%} ceiling; refusing to publish a scene "
            "this different from what was trained"
        )
    return ~removed, metrics


def publish_gaussian_ply(
    source: Path,
    destination: Path,
    rotation: np.ndarray,
    cull: CullSpec | None = None,
) -> dict[str, Any]:
    """Cull, rotate, and write a Gaussian Splatting PLY in one pass.

    Both steps happen against a single in-memory copy; writing an intermediate
    file would cost another couple of hundred megabytes for no benefit.
    """
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if source == destination:
        raise ValueError("Destination must differ from the input PLY")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
        raise ValueError("Publishing requires a proper 3x3 rotation matrix")
    header, vertex_count, properties = read_ply_header(source)
    if tuple(properties) != tuple(FLOAT_PROPERTIES):
        raise ValueError("PLY properties do not match the expected Gaussian Splatting layout")
    dtype = np.dtype([(name, "<f4") for name in properties])
    data = np.fromfile(source, dtype=dtype, count=vertex_count, offset=len(header))
    if data.size != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices, read {data.size}")
    expected_size = len(header) + vertex_count * dtype.itemsize
    if source.stat().st_size != expected_size:
        raise ValueError("Unexpected trailing data or record size in PLY")

    metrics: dict[str, Any] = {"input_gaussians": int(vertex_count)}
    if cull is not None:
        keep, cull_metrics = _cull_mask(data, cull)
        metrics.update(cull_metrics)
        data = data[keep]

    xyz = np.column_stack((data["x"], data["y"], data["z"]))
    data["x"], data["y"], data["z"] = (xyz @ rotation.T).T
    normals = np.column_stack((data["nx"], data["ny"], data["nz"]))
    data["nx"], data["ny"], data["nz"] = (normals @ rotation.T).T
    quaternions = np.column_stack(tuple(data[f"rot_{index}"] for index in range(4)))
    _rotate_quaternions_wxyz(quaternions, rotation)
    for index in range(4):
        data[f"rot_{index}"] = quaternions[:, index]
    sh_transform = _sh_rotation_matrix(rotation)
    for channel in range(3):
        start = channel * 15
        coefficients = np.column_stack(
            tuple(data[f"f_rest_{start + index}"] for index in range(15))
        )
        coefficients = coefficients @ sh_transform.T
        for index in range(15):
            data[f"f_rest_{start + index}"] = coefficients[:, index]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_updated_header(header, rotation, int(data.size)))
            data.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    metrics["published_gaussians"] = int(data.size)
    metrics["rotation"] = [[float(value) for value in row] for row in rotation]
    return metrics


def rotate_gaussian_ply_y_up(source: Path, destination: Path) -> int:
    """Publish with the fixed Z-up to Y-up rotation and no culling."""
    return publish_gaussian_ply(source, destination, ROTATE_X_MINUS_90)["published_gaussians"]
