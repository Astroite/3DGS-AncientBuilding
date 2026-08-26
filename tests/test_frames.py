import json
import math
from pathlib import Path

import numpy as np
import pytest

from gsdb.frames import (
    MAX_GRAVITY_DEVIATION_DEGREES,
    POSE_NORMALISATION_TOLERANCE,
    applied_transform,
    camera_positions,
    check_pose_normalisation,
    load_dataparser_transform,
    to_model_frame,
    trajectory_frame,
    upright_rotation,
    write_published_transforms,
)


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(degrees)
    skew = np.array(
        ((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)),
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _write_dataparser(tmp_path: Path, rotation: np.ndarray, translation, scale: float) -> Path:
    config = tmp_path / "config.yml"
    config.write_text("method_name: splatfacto\n", encoding="utf-8")
    payload = {
        "transform": np.column_stack((rotation, translation)).tolist(),
        "scale": scale,
    }
    (tmp_path / "dataparser_transforms.json").write_text(json.dumps(payload), encoding="utf-8")
    return config


def test_dataparser_transform_round_trips_into_the_model_frame(tmp_path: Path) -> None:
    rotation = _rotation(np.array((0.2, 1.0, -0.3)), 37.0)
    translation = np.array((0.5, -1.25, 2.0))
    config = _write_dataparser(tmp_path, rotation, translation, 0.25)
    loaded_rotation, loaded_translation, scale = load_dataparser_transform(config)
    assert np.allclose(loaded_rotation, rotation)
    assert np.allclose(loaded_translation, translation)
    assert scale == 0.25
    points = np.array(((1.0, 2.0, 3.0), (-4.0, 0.5, 6.0)))
    expected = (points @ rotation.T + translation) * 0.25
    assert np.allclose(to_model_frame(points, loaded_rotation, loaded_translation, scale), expected)


def test_missing_dataparser_transform_is_a_clear_failure(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    config.write_text("method_name: splatfacto\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="dataparser_transforms.json"):
        load_dataparser_transform(config)


def test_upright_rotation_sends_the_measured_vertical_to_plus_y() -> None:
    up = np.array((-0.0012, 1.0, -0.0023))
    up = up / np.linalg.norm(up)
    forward = np.array((1.0, 0.0, 0.0))
    rotation = upright_rotation(up, forward)
    assert np.allclose(rotation @ up, (0.0, 1.0, 0.0), atol=1e-9)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_upright_rotation_is_deterministic_and_pins_the_remaining_spin() -> None:
    up = np.array((0.0, 0.0, 1.0))
    forward = np.array((0.3, 0.9, 0.4))
    first = upright_rotation(up, forward)
    assert all(np.array_equal(first, upright_rotation(up, forward)) for _ in range(5))
    # The hint's horizontal part becomes +X, so the spin about up is fixed.
    horizontal = forward - up * float(forward @ up)
    horizontal = horizontal / np.linalg.norm(horizontal)
    assert np.allclose(first @ horizontal, (1.0, 0.0, 0.0), atol=1e-9)
    # A hint parallel to up carries no direction, so it must fall back, not divide by zero.
    assert np.isclose(np.linalg.det(upright_rotation(up, up)), 1.0)


def test_upright_rotation_handles_a_steeply_tilted_vertical() -> None:
    up = np.array((-0.0001, -0.9047, 0.4261))
    up = up / np.linalg.norm(up)
    rotation = upright_rotation(up, np.array((1.0, 0.0, 0.0)))
    assert np.allclose(rotation @ up, (0.0, 1.0, 0.0), atol=1e-9)


def test_pose_normalisation_check_catches_a_mismatched_transform() -> None:
    good = np.array(((1.0, 0.0, 0.0), (-0.4, 0.2, 0.0), (0.0, 0.0, 0.5)))
    assert check_pose_normalisation(good) == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="peak at"):
        check_pose_normalisation(good * 5.14)
    # The tolerance is a band, not an exact match.
    assert check_pose_normalisation(good * (1.0 + POSE_NORMALISATION_TOLERANCE / 2))


def test_trajectory_frame_describes_a_level_walking_path() -> None:
    up = np.array((0.0, 1.0, 0.0))
    steps = np.linspace(-1.0, 1.0, 50)
    cameras = np.column_stack((steps, np.full_like(steps, 0.01), steps * 0.1))
    frame = trajectory_frame(cameras, up)
    assert frame["radius"] == pytest.approx(np.linalg.norm(cameras - cameras.mean(0), axis=1).max())
    assert frame["height_span"] == pytest.approx(0.0, abs=1e-9)
    assert abs(float(frame["forward"] @ up)) < 1e-9
    assert np.allclose(np.abs(frame["forward"]), np.abs(np.array((1.0, 0.0, 0.1)) / np.linalg.norm((1.0, 0.0, 0.1))), atol=1e-6)


def test_published_transforms_land_in_the_ply_frame(tmp_path: Path) -> None:
    rotation = _rotation(np.array((0.1, 0.9, 0.2)), 21.0)
    translation = np.array((0.3, -0.7, 1.1))
    scale = 0.19451
    publish = _rotation(np.array((1.0, 0.2, 0.0)), 12.0)
    frames = []
    for index in range(4):
        matrix = np.eye(4)
        matrix[:3, :3] = _rotation(np.array((0.0, 1.0, 0.0)), 30.0 * index)
        matrix[:3, 3] = (index - 1.5, 0.2, 0.4 * index)
        frames.append({"file_path": f"images/{index}.jpg", "transform_matrix": matrix.tolist()})
    source = tmp_path / "transforms.json"
    source.write_text(
        json.dumps({"applied_transform": np.eye(4)[:3].tolist(), "frames": frames}),
        encoding="utf-8",
    )
    destination = tmp_path / "published.json"
    assert write_published_transforms(source, destination, rotation, translation, scale, publish) == 4

    original = camera_positions(source)
    published = camera_positions(destination)
    expected = to_model_frame(original, rotation, translation, scale) @ publish.T
    assert np.allclose(published, expected)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["gsdb_published_frame"]["dataparser_scale"] == pytest.approx(scale)
    # Orientations rotate too; a uniform scale must not leak into them.
    first = np.asarray(payload["frames"][0]["transform_matrix"])[:3, :3]
    assert np.allclose(first @ first.T, np.eye(3), atol=1e-9)
    assert np.allclose(applied_transform(source), np.eye(3))


def test_gravity_deviation_limit_is_tight_enough_to_catch_an_unstabilised_source() -> None:
    # The measured p95 on a stabilised capture was 0.83 degrees, so the limit has
    # a wide margin over real data while still rejecting an uncorrected source.
    assert 1.0 < MAX_GRAVITY_DEVIATION_DEGREES < 15.0
