from pathlib import Path

import numpy as np
import pytest

from gsdb.ply import (
    FLOAT_PROPERTIES,
    ROTATE_X_MINUS_90,
    CullSpec,
    _sh_basis,
    gaussian_count,
    publish_gaussian_ply,
    read_ply_header,
    rotate_gaussian_ply_y_up,
    rotation_to_quaternion_wxyz,
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


def _write_scene_ply(
    path: Path, positions: np.ndarray, scales: np.ndarray | None = None
) -> np.ndarray:
    """A PLY with one Gaussian per position, tagged so survivors can be identified."""
    dtype = np.dtype([(name, "<f4") for name in FLOAT_PROPERTIES])
    data = np.zeros(len(positions), dtype=dtype)
    data["x"], data["y"], data["z"] = np.asarray(positions, dtype=np.float32).T
    data["rot_0"] = 1.0
    data["opacity"] = 9.0  # sigmoid(9) is effectively 1.0, as the real floaters are
    log_scales = np.log(np.full(len(positions), 0.01) if scales is None else np.asarray(scales))
    for axis in range(3):
        data[f"scale_{axis}"] = log_scales
    data["f_dc_0"] = np.arange(len(positions), dtype=np.float32)  # identity tag
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment Vertical Axis: z",
            f"element vertex {len(positions)}",
            *(f"property float {name}" for name in FLOAT_PROPERTIES),
            "end_header",
            "",
        ]
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        data.tofile(stream)
    return data


def _read_back(path: Path) -> np.ndarray:
    header, count, properties = read_ply_header(path)
    dtype = np.dtype([(name, "<f4") for name in properties])
    return np.fromfile(path, dtype=dtype, count=count, offset=len(header))


def test_culling_drops_distant_and_oversized_gaussians_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    cameras = np.array([(value, 0.0, 0.0) for value in np.linspace(-1.0, 1.0, 5)])
    positions = np.array(
        [
            (0.0, 0.0, 0.0),      # 0 core
            (0.5, 0.2, 0.1),      # 1 core
            (-1.0, 0.3, 0.0),     # 2 near the path end
            (0.0, 0.0, 12.0),     # 3 far floater
            (30.0, 0.0, 0.0),     # 4 far floater
            (0.2, 0.1, 0.0),      # 5 core but enormous
        ]
    )
    scales = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 4.0])
    source = tmp_path / "source.ply"
    _write_scene_ply(source, positions, scales)

    destination = tmp_path / "culled.ply"
    spec = CullSpec(
        cameras=cameras,
        trajectory_radius=1.0,
        distance_factor=3.0,
        scale_factor=1.0,
        # Half of this six-Gaussian scene is deliberately junk; the production
        # ceiling has its own test below.
        max_removed_fraction=1.0,
    )
    metrics = publish_gaussian_ply(source, destination, np.eye(3), spec)

    assert metrics["input_gaussians"] == 6
    assert metrics["published_gaussians"] == 3
    assert metrics["removed_beyond_distance"] == 2
    assert metrics["removed_above_scale"] == 1
    assert metrics["removed_total"] == 3
    # The header must agree with the body, or every downstream reader mis-parses it.
    assert gaussian_count(destination) == 3
    after = _read_back(destination)
    assert len(after) == 3
    assert sorted(int(value) for value in after["f_dc_0"]) == [0, 1, 2]


def test_culling_is_skipped_entirely_when_no_spec_is_given(tmp_path: Path) -> None:
    positions = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 500.0)])
    source = tmp_path / "source.ply"
    _write_scene_ply(source, positions)
    destination = tmp_path / "kept.ply"
    metrics = publish_gaussian_ply(source, destination, np.eye(3), None)
    assert metrics["published_gaussians"] == 2
    assert "removed_total" not in metrics


def test_culling_refuses_to_remove_more_than_the_ceiling(tmp_path: Path) -> None:
    cameras = np.array([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
    positions = np.vstack(
        (np.zeros((10, 3)), np.column_stack((np.full(10, 50.0), np.zeros(10), np.zeros(10))))
    )
    source = tmp_path / "source.ply"
    _write_scene_ply(source, positions)
    spec = CullSpec(cameras=cameras, trajectory_radius=1.0, max_removed_fraction=0.05)
    with pytest.raises(RuntimeError, match="above the 5.00% ceiling"):
        publish_gaussian_ply(source, tmp_path / "out.ply", np.eye(3), spec)
    assert not (tmp_path / "out.ply").exists()


def test_culling_refuses_cameras_that_are_not_in_the_ply_frame(tmp_path: Path) -> None:
    # Exactly the shipped defect: transforms.json was 5.14x larger than its own PLY.
    positions = np.random.default_rng(3).normal(scale=0.3, size=(200, 3))
    cameras = np.array([(value * 5.14, 0.0, 0.0) for value in np.linspace(-1.0, 1.0, 5)]) + 40.0
    source = tmp_path / "source.ply"
    _write_scene_ply(source, positions)
    spec = CullSpec(cameras=cameras, trajectory_radius=1.0)
    with pytest.raises(RuntimeError, match="not in the PLY's frame"):
        publish_gaussian_ply(source, tmp_path / "out.ply", np.eye(3), spec)


def test_publishing_rejects_anything_that_is_not_a_proper_rotation(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    _write_scene_ply(source, np.zeros((2, 3)))
    for bad in (np.eye(3) * 2.0, np.diag((1.0, 1.0, -1.0)), np.eye(4)):
        with pytest.raises(ValueError, match="proper 3x3 rotation"):
            publish_gaussian_ply(source, tmp_path / "out.ply", bad, None)


def test_arbitrary_rotation_moves_positions_normals_and_orientation_together(
    tmp_path: Path,
) -> None:
    angle = np.radians(37.0)
    axis = np.array((0.3, -0.5, 0.81))
    axis = axis / np.linalg.norm(axis)
    skew = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)

    source = tmp_path / "source.ply"
    before = _write_gaussian_ply(source)
    destination = tmp_path / "rotated.ply"
    publish_gaussian_ply(source, destination, rotation, None)
    after = _read_back(destination)

    assert np.array([after["x"][0], after["y"][0], after["z"][0]]) == pytest.approx(
        rotation @ np.array([1.0, 2.0, 3.0]), rel=1e-5
    )
    assert np.array([after["nx"][0], after["ny"][0], after["nz"][0]]) == pytest.approx(
        rotation @ np.array([0.0, 1.0, 0.0]), rel=1e-5, abs=1e-6
    )
    quaternion = np.array([after[f"rot_{index}"][0] for index in range(4)])
    assert np.linalg.norm(quaternion) == pytest.approx(1.0, rel=1e-6)
    assert quaternion == pytest.approx(rotation_to_quaternion_wxyz(rotation), rel=1e-5, abs=1e-6)

    # The rotated SH must evaluate to the same radiance in the rotated direction.
    direction_old = np.array([[0.2, -0.3, 0.9327379]])
    direction_old /= np.linalg.norm(direction_old, axis=1, keepdims=True)
    basis_old = _sh_basis(direction_old)[0, 1:]
    basis_new = _sh_basis(direction_old @ rotation.T)[0, 1:]
    for channel in range(3):
        start = channel * 15
        coeff_before = np.array([before[f"f_rest_{start + i}"][0] for i in range(15)])
        coeff_after = np.array([after[f"f_rest_{start + i}"][0] for i in range(15)])
        assert basis_new @ coeff_after == pytest.approx(
            basis_old @ coeff_before, rel=1e-5, abs=1e-5
        )
