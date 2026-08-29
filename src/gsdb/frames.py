"""Coordinate frames between COLMAP, Nerfstudio training space, and published PLY.

Three frames are involved in publishing a trained scene, and they are not
interchangeable:

``COLMAP``
    What the reconstruction stage produces. ``transforms.json`` records the
    ``applied_transform`` that Nerfstudio's ``colmap_to_json`` already baked into
    the camera poses it writes.
``model``
    What ``ns-export`` writes. Nerfstudio's dataparser re-orients, re-centers and
    re-scales the poses before training, so the trained Gaussians live in a frame
    that is typically several times smaller than the COLMAP one. The mapping is
    recorded in ``dataparser_transforms.json`` beside the training ``config.yml``.
``published``
    The frame of the exported PLY, which this module defines as gravity-up along
    ``+Y``.

Publishing a PLY and its cameras together requires all three, and culling by
distance to the capture path requires camera positions expressed in the model
frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# Nerfstudio's equirectangular rig frame is [forward, right, up], so column 2 of
# ``world_from_rig`` points along the panorama's own vertical.
RIG_UP_COLUMN = 2

# A gravity-stabilised panorama should agree with itself across the whole capture.
# Anything looser means the source was not stabilised and the derived vertical
# cannot be trusted.
MAX_GRAVITY_DEVIATION_DEGREES = 5.0

# ``auto_scale_poses`` normalises the largest absolute camera coordinate to 1.0,
# which makes it a free end-to-end check that the transform chain is wired right.
POSE_NORMALISATION_TOLERANCE = 0.05


def load_dataparser_transform(training_config: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Read the Nerfstudio dataparser transform beside a training ``config.yml``."""
    path = Path(training_config).parent / "dataparser_transforms.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Training output has no dataparser_transforms.json: {path}. "
            "Without it the exported PLY cannot be related to the camera poses."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform = np.asarray(payload["transform"], dtype=np.float64)
    if transform.shape != (3, 4):
        raise ValueError(f"Expected a 3x4 dataparser transform, got {transform.shape}")
    scale = float(payload["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Dataparser scale must be positive and finite, got {scale}")
    return transform[:, :3], transform[:, 3], scale


def to_model_frame(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float
) -> np.ndarray:
    """Map Nerfstudio-space points into the trained model's frame."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) point array, got {points.shape}")
    return (points @ rotation.T + translation) * scale


def camera_positions(transforms_path: Path) -> np.ndarray:
    """Camera centers from a Nerfstudio ``transforms.json``, in Nerfstudio space."""
    payload = json.loads(Path(transforms_path).read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"No frames in {transforms_path}")
    return np.array(
        [np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3] for frame in frames]
    )


def applied_transform(transforms_path: Path) -> np.ndarray:
    """The COLMAP-to-Nerfstudio rotation already baked into ``transforms.json``."""
    payload = json.loads(Path(transforms_path).read_text(encoding="utf-8"))
    if "applied_transform" not in payload:
        raise ValueError(f"{transforms_path} has no applied_transform")
    return np.asarray(payload["applied_transform"], dtype=np.float64)[:3, :3]


def rig_gravity(
    model_dir: Path, attempt: Any, allow_unsafe: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover the capture's vertical in COLMAP space from the panorama rig.

    Every perspective view was cut from a gravity-stabilised equirectangular frame
    at a known yaw and pitch, so each registered image independently reports where
    "up" is. Averaging over the whole capture is far more reliable than inferring
    orientation from camera poses alone, which is what Nerfstudio's dataparser
    attempts and gets wrong on a panoramic rig.
    """
    from nerfstudio.process_data.colmap_utils import read_images_binary

    from .reconstruction import _component_rig_poses

    poses = _component_rig_poses(read_images_binary(model_dir / "images.bin"), attempt)
    if not poses:
        raise RuntimeError(f"No rig poses recoverable from {model_dir}")
    ups = np.array([world_from_rig[:, RIG_UP_COLUMN] for _, world_from_rig in poses.values()])
    mean = ups.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        raise RuntimeError(
            "Per-frame rig verticals cancel out; the capture is not gravity-stabilised"
        )
    up = mean / norm
    angles = np.degrees(np.arccos(np.clip(ups @ up, -1.0, 1.0)))
    metrics = {
        "frames": int(len(poses)),
        "up_vector": [float(value) for value in up],
        "deviation_median_degrees": float(np.median(angles)),
        "deviation_p95_degrees": float(np.percentile(angles, 95)),
        "deviation_max_degrees": float(angles.max()),
        "deviation_limit_degrees": MAX_GRAVITY_DEVIATION_DEGREES,
    }
    deviation_passed = (
        metrics["deviation_p95_degrees"] <= MAX_GRAVITY_DEVIATION_DEGREES
    )
    metrics["deviation_gate_passed"] = deviation_passed
    if not deviation_passed:
        warning = (
            "Per-frame rig verticals disagree by "
            f"{metrics['deviation_p95_degrees']:.2f} degrees at p95, above the "
            f"{MAX_GRAVITY_DEVIATION_DEGREES} degree limit; the source panoramas are "
            "not gravity-stabilised and the published axis would be wrong"
        )
        metrics["deviation_warning"] = warning
        if not allow_unsafe:
            raise RuntimeError(warning)
    return up, metrics


def upright_rotation(up: np.ndarray, forward_hint: np.ndarray | None = None) -> np.ndarray:
    """Rotation taking ``up`` to ``+Y``, with the remaining spin pinned down.

    Sending one axis to another leaves a free rotation about it. That freedom is
    resolved with ``forward_hint`` — in practice the capture path's dominant
    horizontal direction — so the same scene always publishes in the same pose.
    """
    up = np.asarray(up, dtype=np.float64)
    norm = float(np.linalg.norm(up))
    if norm < 1e-9:
        raise ValueError("Up vector must be non-zero")
    up = up / norm

    if forward_hint is None:
        # Pick the axis least parallel to up, so the fallback is still deterministic.
        forward_hint = np.eye(3)[int(np.argmin(np.abs(up)))]
    forward = np.asarray(forward_hint, dtype=np.float64)
    forward = forward - up * float(forward @ up)
    if float(np.linalg.norm(forward)) < 1e-9:
        forward = np.eye(3)[int(np.argmin(np.abs(up)))]
        forward = forward - up * float(forward @ up)
    forward = forward / float(np.linalg.norm(forward))

    # Rows are the published axes expressed in the source frame: X=forward, Y=up,
    # and X cross Y completes a right-handed basis.
    rotation = np.vstack((forward, up, np.cross(forward, up)))
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-9):
        raise RuntimeError(f"Upright rotation is not a proper rotation: det={determinant}")
    return rotation


def trajectory_frame(cameras: np.ndarray, up: np.ndarray) -> dict[str, Any]:
    """Descriptive geometry of the capture path, used for culling and for checks."""
    cameras = np.asarray(cameras, dtype=np.float64)
    centroid = cameras.mean(axis=0)
    centered = cameras - centroid
    radius = float(np.linalg.norm(centered, axis=1).max())
    if radius <= 0:
        raise RuntimeError("Capture path has no spatial extent")
    horizontal = centered - np.outer(centered @ up, up)
    # Principal horizontal direction of travel; sign fixed so the result is stable.
    _, _, components = np.linalg.svd(horizontal, full_matrices=False)
    forward = components[0]
    if float(forward @ horizontal[np.argmax(np.linalg.norm(horizontal, axis=1))]) < 0:
        forward = -forward
    heights = centered @ up
    return {
        "centroid": centroid,
        "radius": radius,
        "forward": forward,
        "height_span": float(heights.max() - heights.min()),
        "height_span_ratio": float((heights.max() - heights.min()) / (2.0 * radius)),
    }


def write_published_transforms(
    source: Path,
    destination: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
    publish_rotation: np.ndarray,
) -> int:
    """Rewrite camera poses into the published PLY's frame.

    Positions pick up the dataparser's scale; orientations do not, since a uniform
    scale leaves rotations untouched.
    """
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    combined = publish_rotation @ rotation
    for frame in payload.get("frames", []):
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        matrix[:3, 3] = publish_rotation @ ((rotation @ matrix[:3, 3] + translation) * scale)
        matrix[:3, :3] = combined @ matrix[:3, :3]
        frame["transform_matrix"] = [[float(value) for value in row] for row in matrix]
    payload["gsdb_published_frame"] = {
        "note": "Camera poses match the exported PLY; y is up.",
        "dataparser_scale": float(scale),
        "publish_rotation": [[float(value) for value in row] for row in publish_rotation],
        "source": "transforms-colmap.json",
    }
    Path(destination).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return len(payload.get("frames", []))


def check_pose_normalisation(cameras: np.ndarray) -> float:
    """Confirm the transform chain landed in the model frame.

    Nerfstudio's ``auto_scale_poses`` divides by the largest absolute pose
    coordinate, so a correctly transformed camera set peaks at 1.0. A wrong chain
    misses by orders of magnitude, which is exactly the failure that shipped a
    ``transforms.json`` five times larger than its own PLY.
    """
    peak = float(np.abs(np.asarray(cameras, dtype=np.float64)).max())
    if abs(peak - 1.0) > POSE_NORMALISATION_TOLERANCE:
        raise RuntimeError(
            f"Cameras in the model frame peak at {peak:.4f} instead of 1.0; the "
            "dataparser transform does not match these poses"
        )
    return peak
