import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb.models import ReconstructionAttempt, ReconstructionConfig
from gsdb.reconstruction import (
    attach_frame_masks,
    build_colmap_commands,
    build_image_pyramid,
    build_rig_config,
    center_spread_metrics,
    interleaved_image_names,
    pinhole_camera_parameters,
    select_colmap_attempt,
    projection_view_specs,
    reorder_database_image_ids,
    validate_folder_camera_ids,
    validate_frame_masks,
)


def _attempt(images_per_equirect: int = 8) -> ReconstructionAttempt:
    return ReconstructionAttempt(
        frame_count=2,
        images_per_equirect=images_per_equirect,
        projection_fov_degrees=120.0 if images_per_equirect == 8 else 110.0,
        projection_size=2048 if images_per_equirect == 8 else 1746,
        crop_bottom=0.2 if images_per_equirect == 8 else 0.15,
    )


def test_colmap_commands_use_gpu_fixed_intrinsics_rig_cameras_and_image_list(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    for view in range(8):
        (dataset / "images" / f"view_{view:02d}").mkdir(parents=True)
    (dataset / "masks").mkdir()
    for frame in range(1, 3):
        for view in range(8):
            image_path = dataset / "images" / f"view_{view:02d}" / f"frame_{frame:06d}.jpg"
            assert cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    image_path = dataset / "images" / "view_00" / "frame_000001.jpg"
    attempt = _attempt()
    commands = build_colmap_commands(dataset, attempt, ReconstructionConfig())
    feature = commands[0]
    assert feature[feature.index("--ImageReader.mask_path") + 1] == str(dataset / "masks")
    assert feature[feature.index("--ImageReader.camera_model") + 1] == "PINHOLE"
    assert feature[feature.index("--ImageReader.single_camera_per_folder") + 1] == "1"
    assert feature[feature.index("--SiftExtraction.use_gpu") + 1] == "1"
    assert feature[feature.index("--SiftExtraction.gpu_index") + 1] == "0"
    assert commands[1][commands[1].index("--SiftMatching.use_gpu") + 1] == "1"
    assert commands[1][commands[1].index("--SequentialMatching.overlap") + 1] == "16"
    retry_commands = build_colmap_commands(
        dataset, attempt, ReconstructionConfig(), mapper_num_threads=1
    )
    assert retry_commands[2][retry_commands[2].index("--Mapper.num_threads") + 1] == "1"
    for option in (
        "--Mapper.ba_refine_focal_length",
        "--Mapper.ba_refine_principal_point",
        "--Mapper.ba_refine_extra_params",
    ):
        assert retry_commands[2][retry_commands[2].index(option) + 1] == "0"
    image_list = Path(feature[feature.index("--image_list_path") + 1])
    assert image_list.read_text(encoding="utf-8").splitlines()[:9] == [
        *(f"view_{view:02d}/frame_000001.jpg" for view in range(8)),
        "view_00/frame_000002.jpg",
    ]
    parameters = pinhole_camera_parameters(image_path, 120.0).split(",")
    assert abs(float(parameters[0]) - 28.867513) < 0.001


def test_projection_specs_match_nerfstudio_orientation_counts_and_crop_math() -> None:
    primary = projection_view_specs(_attempt(8))
    fallback = projection_view_specs(_attempt(14))
    assert len(primary) == 8
    assert len(fallback) == 14
    assert [yaw for yaw, _ in primary[:4]] == [-180.0, -90.0, 0.0, 90.0]
    assert primary[0][1] == pytest.approx(25.5)
    assert primary[4][1] == pytest.approx(57.75)
    assert primary[6][1] == pytest.approx(6.0)
    assert [yaw for yaw, _ in fallback[:6]] == [
        -180.0,
        -120.0,
        -60.0,
        0.0,
        60.0,
        120.0,
    ]


def test_rig_config_has_distinct_cameras_zero_centers_and_unit_quaternions() -> None:
    camera_ids = {f"view_{index:02d}/": index + 1 for index in range(8)}
    payload = build_rig_config(camera_ids, _attempt())
    rig = payload[0]
    assert rig["ref_camera_id"] == 1
    assert len(rig["cameras"]) == 8
    assert "cam_from_rig_translation" not in rig["cameras"][0]
    for camera in rig["cameras"][1:]:
        assert camera["cam_from_rig_translation"] == [0.0, 0.0, 0.0]
        assert np.linalg.norm(camera["cam_from_rig_rotation"]) == pytest.approx(1.0)


def test_database_camera_ids_are_one_per_view_folder(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER)")
        for frame in range(2):
            for view in range(8):
                connection.execute(
                    "INSERT INTO images VALUES (?, ?, ?)",
                    (frame * 8 + view + 1, f"view_{view:02d}/frame_{frame + 1:06d}.jpg", view + 11),
                )
    mapping = validate_folder_camera_ids(database, 8)
    assert mapping["view_00/"] == 11
    assert len(set(mapping.values())) == 8


def test_database_image_ids_are_transactionally_reordered_frame_major(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database.db"
    lexical_names = [
        "view_00/frame_000001.jpg",
        "view_00/frame_000002.jpg",
        "view_01/frame_000001.jpg",
        "view_01/frame_000002.jpg",
    ]
    desired = [
        "view_00/frame_000001.jpg",
        "view_01/frame_000001.jpg",
        "view_00/frame_000002.jpg",
        "view_01/frame_000002.jpg",
    ]
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, camera_id INTEGER)"
        )
        connection.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER)")
        connection.execute("CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER)")
        connection.execute("CREATE TABLE matches (pair_id INTEGER)")
        connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER)")
        for image_id, name in enumerate(lexical_names, start=1):
            connection.execute("INSERT INTO images VALUES (?, ?, ?)", (image_id, name, 1))
            connection.execute("INSERT INTO keypoints VALUES (?, ?)", (image_id, image_id * 10))
            connection.execute("INSERT INTO descriptors VALUES (?, ?)", (image_id, image_id * 10))

    assert reorder_database_image_ids(database, desired) is True
    with sqlite3.connect(database) as connection:
        names = [row[0] for row in connection.execute("SELECT name FROM images ORDER BY image_id")]
        keypoint_rows = [
            row[0] for row in connection.execute("SELECT rows FROM keypoints ORDER BY image_id")
        ]
    assert names == desired
    assert keypoint_rows == [10, 30, 20, 40]
    assert reorder_database_image_ids(database, desired) is False


def test_center_spread_gate_metric_is_zero_for_coincident_rig_views() -> None:
    centers = []
    for frame in range(1, 4):
        center = np.array((float(frame), 0.0, 0.0))
        for view in range(8):
            centers.append((f"view_{view:02d}/frame_{frame:06d}.jpg", center.copy()))
    metrics = center_spread_metrics(centers)
    assert metrics["center_spread_p95"] == pytest.approx(0.0)
    assert metrics["median_interframe_baseline"] == pytest.approx(1.0)
    assert metrics["p95_spread_to_baseline"] == pytest.approx(0.0)


def test_colmap_mapper_retry_reuses_completed_database_single_threaded(
    tmp_path: Path,
) -> None:
    colmap_root = tmp_path / "colmap"
    attempt_dir = colmap_root / "attempt-001"
    (attempt_dir / "sparse").mkdir(parents=True)
    (attempt_dir / ".features-complete").write_text("complete\n", encoding="utf-8")
    (attempt_dir / ".matching-complete").write_text("complete\n", encoding="utf-8")

    selected, mapper_num_threads = select_colmap_attempt(colmap_root)

    assert selected == attempt_dir
    assert mapper_num_threads == 1


def test_colmap_mapper_retry_does_not_overwrite_partial_sparse_output(
    tmp_path: Path,
) -> None:
    colmap_root = tmp_path / "colmap"
    attempt_dir = colmap_root / "attempt-001"
    sparse_component = attempt_dir / "sparse" / "0"
    sparse_component.mkdir(parents=True)
    (sparse_component / "images.bin").write_bytes(b"partial")
    (attempt_dir / ".features-complete").write_text("complete\n", encoding="utf-8")
    (attempt_dir / ".matching-complete").write_text("complete\n", encoding="utf-8")

    selected, mapper_num_threads = select_colmap_attempt(colmap_root)

    assert selected == colmap_root / "attempt-002"
    assert mapper_num_threads is None


def test_transforms_receive_matching_per_frame_mask_path(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    (masks / "view_00").mkdir(parents=True)
    (masks / "view_00" / "frame_000001.jpg.png").write_bytes(b"present")
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "file_path": "./images/view_00/frame_000001.jpg",
                        "transform_matrix": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invalid mask_path"):
        validate_frame_masks(transforms, masks)
    attach_frame_masks(transforms, masks)
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    assert payload["frames"][0]["mask_path"] == "./masks/view_00/frame_000001.jpg.png"
    assert validate_frame_masks(transforms, masks) == 1
    assert not list(tmp_path.glob(".transforms.json.*.tmp"))


def test_mask_pyramid_keeps_binary_values_and_colmap_filename(tmp_path: Path) -> None:
    source = tmp_path / "masks"
    (source / "view_00").mkdir(parents=True)
    mask = np.full((16, 16), 255, dtype=np.uint8)
    mask[4:12, 4:12] = 0
    assert cv2.imwrite(str(source / "view_00" / "view.jpg.png"), mask)
    build_image_pyramid(source, tmp_path, "masks", 2, is_mask=True)
    build_image_pyramid(source, tmp_path, "masks", 2, is_mask=True)
    for factor in (2, 4):
        result = cv2.imread(
            str(tmp_path / f"masks_{factor}" / "view_00" / "view.jpg.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        assert result is not None
        assert set(int(value) for value in np.unique(result)).issubset({0, 255})
