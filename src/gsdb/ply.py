"""Validated Gaussian Splatting PLY inspection and axis conversion."""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path

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


def _updated_header(header: bytes) -> bytes:
    text = header.decode("ascii")
    text = text.replace("comment Vertical Axis: z", "comment Vertical Axis: y")
    note = "comment Axis transform: (x, y, z) -> (x, z, -y); Rx(-90 deg)\n"
    return text.replace("end_header", note + "end_header", 1).encode("ascii")


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


def _rotate_quaternions_wxyz(quaternions: np.ndarray) -> None:
    half_sqrt = math.sqrt(0.5)
    old = quaternions.astype(np.float64, copy=True)
    w, x, y, z = old.T
    quaternions[:, 0] = half_sqrt * (w + x)
    quaternions[:, 1] = half_sqrt * (x - w)
    quaternions[:, 2] = half_sqrt * (y + z)
    quaternions[:, 3] = half_sqrt * (z - y)


def rotate_gaussian_ply_y_up(source: Path, destination: Path) -> int:
    """Bake Z-up to Y-up into positions, normals, quaternions, and real SH."""
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if source == destination:
        raise ValueError("Destination must differ from the input PLY")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
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

    xyz = np.column_stack((data["x"], data["y"], data["z"]))
    data["x"], data["y"], data["z"] = (xyz @ ROTATE_X_MINUS_90.T).T
    normals = np.column_stack((data["nx"], data["ny"], data["nz"]))
    data["nx"], data["ny"], data["nz"] = (normals @ ROTATE_X_MINUS_90.T).T
    quaternions = np.column_stack(tuple(data[f"rot_{index}"] for index in range(4)))
    _rotate_quaternions_wxyz(quaternions)
    for index in range(4):
        data[f"rot_{index}"] = quaternions[:, index]
    sh_transform = _sh_rotation_matrix(ROTATE_X_MINUS_90)
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
            stream.write(_updated_header(header))
            data.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return vertex_count
