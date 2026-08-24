from pathlib import Path

import numpy as np
import pytest

from gsdb.ply import (
    FLOAT_PROPERTIES,
    ROTATE_X_MINUS_90,
    _sh_basis,
    gaussian_count,
    read_ply_header,
    rotate_gaussian_ply_y_up,
)


def _write_gaussian_ply(path: Path) -> np.ndarray:
    dtype = np.dtype([(name, "<f4") for name in FLOAT_PROPERTIES])
    data = np.zeros(1, dtype=dtype)
    data["x"], data["y"], data["z"] = 1.0, 2.0, 3.0
    data["nx"], data["ny"], data["nz"] = 0.0, 1.0, 0.0
    data["rot_0"] = 1.0
    generator = np.random.default_rng(7)
    for index in range(45):
        data[f"f_rest_{index}"] = generator.normal()
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment Vertical Axis: z",
            "element vertex 1",
            *(f"property float {name}" for name in FLOAT_PROPERTIES),
            "end_header",
            "",
        ]
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        data.tofile(stream)
    return data


def test_gaussian_ply_rotation_preserves_layout_orientation_and_sh(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    destination = tmp_path / "splat-yup.ply"
    before = _write_gaussian_ply(source)

    assert rotate_gaussian_ply_y_up(source, destination) == 1
    header, count, properties = read_ply_header(destination)
    dtype = np.dtype([(name, "<f4") for name in properties])
    after = np.fromfile(destination, dtype=dtype, count=count, offset=len(header))

    assert gaussian_count(destination) == 1
    assert b"comment Vertical Axis: y" in header
    assert np.array([after["x"][0], after["y"][0], after["z"][0]]) == pytest.approx(
        [1.0, 3.0, -2.0]
    )
    assert np.array([after["nx"][0], after["ny"][0], after["nz"][0]]) == pytest.approx(
        [0.0, 0.0, -1.0]
    )
    quaternion = np.array([after[f"rot_{index}"][0] for index in range(4)])
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert quaternion == pytest.approx([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0])

    direction_old = np.array([[0.2, -0.3, 0.9327379]])
    direction_old /= np.linalg.norm(direction_old, axis=1, keepdims=True)
    direction_new = direction_old @ ROTATE_X_MINUS_90.T
    basis_old = _sh_basis(direction_old)[0, 1:]
    basis_new = _sh_basis(direction_new)[0, 1:]
    for channel in range(3):
        start = channel * 15
        coeff_before = np.array(
            [before[f"f_rest_{start + index}"][0] for index in range(15)]
        )
        coeff_after = np.array(
            [after[f"f_rest_{start + index}"][0] for index in range(15)]
        )
        assert basis_new @ coeff_after == pytest.approx(
            basis_old @ coeff_before, rel=1e-5, abs=1e-5
        )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        rotate_gaussian_ply_y_up(source, destination)
