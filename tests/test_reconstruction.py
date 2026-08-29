import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gsdb.models import (
    LegacyReconstructionAttemptV1,
    LegacyReconstructionConfigV1,
    ReconstructionAttempt,
    ReconstructionConfig,
)
from gsdb.reconstruction import (
    _average_rotations,
    _component_rig_poses,
    _rig_component_alignment,
    attach_frame_masks,
    build_colmap_commands,
    build_image_pyramid,
    build_rig_config,
    center_spread_metrics,
    cross_view_pair_names,
    interleaved_image_names,
    latest_mapper_snapshot,
    pinhole_camera_parameters,
    pyramid_level_marker,
    select_colmap_attempt,
    projection_view_specs,
    projection_world_from_camera,
    rotation_matrix_to_quaternion_wxyz,
    validate_existing_projection_set,
    validate_folder_camera_ids,
    validate_frame_masks,
    write_cross_view_pair_list,
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
    feature = commands["features"]
    assert feature[feature.index("--ImageReader.mask_path") + 1] == str(dataset / "masks")
    assert feature[feature.index("--ImageReader.camera_model") + 1] == "PINHOLE"
    assert feature[feature.index("--ImageReader.single_camera_per_folder") + 1] == "1"
    assert feature[feature.index("--SiftExtraction.use_gpu") + 1] == "1"
    assert feature[feature.index("--SiftExtraction.gpu_index") + 1] == "0"
    matching = commands["matching"]
    assert matching[matching.index("--SiftMatching.use_gpu") + 1] == "1"
    assert matching[matching.index("--SequentialMatching.overlap") + 1] == "16"
    cross_view = commands["cross_view"]
    assert cross_view[1] == "matches_importer"
    assert cross_view[cross_view.index("--match_type") + 1] == "pairs"
    assert cross_view[cross_view.index("--SiftMatching.gpu_index") + 1] == "0"
    fresh_mapping = commands["mapping"]
    assert fresh_mapping[fresh_mapping.index("--output_path") + 1] == str(
        dataset / "colmap" / "attempt-001" / "sparse"
    )
    assert "--input_path" not in fresh_mapping
    assert fresh_mapping[fresh_mapping.index("--Mapper.snapshot_path") + 1] == str(
        dataset / "colmap" / "attempt-001" / "snapshots"
    )
    assert int(fresh_mapping[fresh_mapping.index("--Mapper.snapshot_images_freq") + 1]) > 0
    # Global BA ran to the 50-iteration cap on every trigger because COLMAP leaves the
    # cost-based convergence test switched off; these are the knobs that stop it.
    assert float(
        fresh_mapping[fresh_mapping.index("--Mapper.ba_global_function_tolerance") + 1]
    ) > 0
    assert int(
        fresh_mapping[fresh_mapping.index("--Mapper.ba_global_max_refinements") + 1]
    ) < 5
    assert float(
        fresh_mapping[fresh_mapping.index("--Mapper.ba_global_images_ratio") + 1]
    ) > 1.1
    # The frequency triggers are independent of the ratio and become the binding one
    # if they stay at their defaults.
    assert int(fresh_mapping[fresh_mapping.index("--Mapper.ba_global_images_freq") + 1]) > 500
    retry_commands = build_colmap_commands(
        dataset, attempt, ReconstructionConfig(), mapper_num_threads=1
    )
    mapping = retry_commands["mapping"]
    assert mapping[mapping.index("--Mapper.num_threads") + 1] == "1"
    for option in (
        "--Mapper.ba_refine_focal_length",
        "--Mapper.ba_refine_principal_point",
        "--Mapper.ba_refine_extra_params",
    ):
        assert mapping[mapping.index(option) + 1] == "0"
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


def test_partial_projection_set_is_validated_for_resume(tmp_path: Path) -> None:
    target = tmp_path / "images"
    first = target / "view_00" / "frame_000001.jpg"
    second = target / "view_01" / "frame_000001.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(first), image)
    assert cv2.imwrite(str(second), image)

    reusable = validate_existing_projection_set(target, 2, 2, 32)

    assert reusable == {
        Path("view_00/frame_000001.jpg"),
        Path("view_01/frame_000001.jpg"),
    }


def test_rig_config_has_distinct_cameras_zero_centers_and_unit_quaternions() -> None:
    camera_ids = {f"view_{index:02d}/": index + 1 for index in range(8)}
    payload = build_rig_config(camera_ids, _attempt())
    rig = payload[0]
    assert rig["ref_camera_id"] == 1
    assert len(rig["cameras"]) == 8
    assert rig["cameras"][0]["rel_qvec"] == [1.0, 0.0, 0.0, 0.0]
    for camera in rig["cameras"]:
        assert camera["rel_tvec"] == [0.0, 0.0, 0.0]
        assert np.linalg.norm(camera["rel_qvec"]) == pytest.approx(1.0)


def test_rig_config_accepts_sparse_component_camera_subset() -> None:
    camera_ids = {"view_02/": 3, "view_03/": 4}

    payload = build_rig_config(camera_ids, _attempt())

    rig = payload[0]
    assert rig["ref_camera_id"] == 3
    assert [camera["camera_id"] for camera in rig["cameras"]] == [3, 4]
    assert rig["cameras"][0]["rel_qvec"] == [1.0, 0.0, 0.0, 0.0]
    assert np.linalg.norm(rig["cameras"][1]["rel_qvec"]) == pytest.approx(1.0)


def test_rig_component_alignment_preserves_left_handed_panorama_basis() -> None:
    attempt = ReconstructionAttempt(
        frame_count=8,
        images_per_equirect=8,
        projection_fov_degrees=120.0,
        projection_size=2048,
        crop_bottom=0.2,
    )
    specs = projection_view_specs(attempt)
    reference_view = 0
    moving_view = 3
    reference_projection = projection_world_from_camera(*specs[reference_view])
    moving_projection = projection_world_from_camera(*specs[moving_view])
    assert np.linalg.det(reference_projection) == pytest.approx(-1.0)
    assert np.linalg.det(
        _average_rotations([reference_projection], determinant=-1)
    ) == pytest.approx(-1.0)

    angle = np.deg2rad(23.0)
    alignment_rotation = np.array(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    scale = 1.7
    translation = np.array((2.0, -3.0, 0.5))
    reference_images = {}
    moving_images = {}
    for frame in range(1, 9):
        rig_angle = np.deg2rad(frame * 2.0)
        reference_world_from_camera = np.array(
            (
                (np.cos(rig_angle), 0.0, np.sin(rig_angle)),
                (0.0, 1.0, 0.0),
                (-np.sin(rig_angle), 0.0, np.cos(rig_angle)),
            )
        )
        reference_center = np.array(
            (float(frame), 0.1 * frame * frame, 0.2 * np.sin(frame))
        )
        moving_center = alignment_rotation.T @ (
            (reference_center - translation) / scale
        )
        moving_world_from_camera = (
            alignment_rotation.T
            @ reference_world_from_camera
            @ reference_projection.T
            @ moving_projection
        )
        reference_camera_from_world = reference_world_from_camera.T
        moving_camera_from_world = moving_world_from_camera.T
        reference_images[frame] = SimpleNamespace(
            name=f"view_{reference_view:02d}/frame_{frame:06d}.jpg",
            qvec=np.asarray(
                rotation_matrix_to_quaternion_wxyz(reference_camera_from_world)
            ),
            tvec=-(reference_camera_from_world @ reference_center),
        )
        moving_images[frame] = SimpleNamespace(
            name=f"view_{moving_view:02d}/frame_{frame:06d}.jpg",
            qvec=np.asarray(
                rotation_matrix_to_quaternion_wxyz(moving_camera_from_world)
            ),
            tvec=-(moving_camera_from_world @ moving_center),
        )

    estimated_scale, estimated_rotation, estimated_translation, metrics = (
        _rig_component_alignment(
            _component_rig_poses(reference_images, attempt),
            _component_rig_poses(moving_images, attempt),
        )
    )

    assert estimated_scale == pytest.approx(scale)
    assert estimated_rotation == pytest.approx(alignment_rotation, abs=1e-10)
    assert estimated_translation == pytest.approx(translation, abs=1e-10)
    assert metrics["center_residual_p95_to_trajectory_extent"] < 1e-10
    assert metrics["rotation_error_p95_degrees"] < 1e-5


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


def test_center_spread_gate_excludes_frames_without_reference_camera() -> None:
    centers = [
        ("view_00/frame_000001.jpg", np.array((0.0, 0.0, 0.0))),
        ("view_01/frame_000001.jpg", np.array((0.0, 0.0, 0.0))),
        ("view_00/frame_000002.jpg", np.array((1.0, 0.0, 0.0))),
        ("view_01/frame_000002.jpg", np.array((1.0, 0.0, 0.0))),
        ("view_01/frame_000003.jpg", np.array((2.0, 0.0, 0.0))),
        ("view_02/frame_000003.jpg", np.array((9.0, 0.0, 0.0))),
    ]

    metrics = center_spread_metrics(centers, required_prefix="view_00/")

    assert metrics["frame_count"] == 2
    assert metrics["center_spread_p95"] == pytest.approx(0.0)
    assert metrics["median_interframe_baseline"] == pytest.approx(1.0)


def _mapper_snapshot(attempt_dir: Path, name: str, image_count: int) -> Path:
    snapshot = attempt_dir / "snapshots" / name
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "cameras.bin").write_bytes(b"cameras")
    (snapshot / "points3D.bin").write_bytes(b"points")
    (snapshot / "images.bin").write_bytes(image_count.to_bytes(8, "little") + b"payload")
    return snapshot


def test_mapper_snapshot_selection_prefers_registered_count_over_directory_name(
    tmp_path: Path,
) -> None:
    attempt_dir = tmp_path / "attempt-001"
    _mapper_snapshot(attempt_dir, "500", 500)
    furthest = _mapper_snapshot(attempt_dir, "1000", 1000)
    (attempt_dir / "snapshots" / "partial").mkdir()

    assert latest_mapper_snapshot(attempt_dir) == furthest
    assert latest_mapper_snapshot(tmp_path / "attempt-002") is None


def test_colmap_mapper_retry_keeps_all_threads_and_resumes_after_a_snapshot(
    tmp_path: Path,
) -> None:
    colmap_root = tmp_path / "colmap"
    attempt_dir = colmap_root / "attempt-001"
    (attempt_dir / "sparse").mkdir(parents=True)
    (attempt_dir / ".features-complete").write_text("complete\n", encoding="utf-8")
    (attempt_dir / ".matching-complete").write_text("complete\n", encoding="utf-8")
    snapshot = _mapper_snapshot(attempt_dir, "2500", 2500)

    selected, mapper_num_threads = select_colmap_attempt(colmap_root)

    assert selected == attempt_dir
    # A mapper that snapshotted was working; it was killed from outside, and one
    # thread would be the wrong answer to that.
    assert mapper_num_threads is None
    assert latest_mapper_snapshot(selected) == snapshot


def test_resumed_mapping_command_continues_the_snapshot_into_a_component_directory(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images" / "view_00").mkdir(parents=True)
    (dataset / "masks").mkdir()
    assert cv2.imwrite(
        str(dataset / "images" / "view_00" / "frame_000001.jpg"),
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    attempt_dir = dataset / "colmap" / "attempt-001"
    snapshot = _mapper_snapshot(attempt_dir, "2500", 2500)
    (attempt_dir / ".features-complete").write_text("complete\n", encoding="utf-8")
    (attempt_dir / ".matching-complete").write_text("complete\n", encoding="utf-8")

    commands = build_colmap_commands(
        dataset,
        _attempt(),
        ReconstructionConfig(),
        attempt_dir,
        mapper_input_path=snapshot,
    )

    mapping = commands["mapping"]
    assert mapping[mapping.index("--input_path") + 1] == str(snapshot)
    # COLMAP writes a continued model flat into --output_path, so the resumed run has
    # to name the component directory that _colmap_components looks for, and it has to
    # exist before the mapper starts.
    assert mapping[mapping.index("--output_path") + 1] == str(
        attempt_dir / "sparse" / "0"
    )
    assert (attempt_dir / "sparse" / "0").is_dir()
    # The empty component directory must not read as output worth preserving, or the
    # next retry throws away hours of completed features and matches.
    selected, mapper_num_threads = select_colmap_attempt(dataset / "colmap")
    assert selected == attempt_dir
    assert mapper_num_threads is None


def test_colmap_mapper_retry_reuses_completed_database_single_threaded(
    tmp_path: Path,
) -> None:
    colmap_root = tmp_path / "colmap"
    attempt_dir = colmap_root / "attempt-001"
    (attempt_dir / "sparse").mkdir(parents=True)
    # An empty snapshot directory is what a mapper that died before its first 500
    # registrations leaves behind, which is the shape of a reproducible crash.
    (attempt_dir / "snapshots").mkdir(parents=True)
    (attempt_dir / ".features-complete").write_text("complete\n", encoding="utf-8")
    (attempt_dir / ".matching-complete").write_text("complete\n", encoding="utf-8")

    selected, mapper_num_threads = select_colmap_attempt(colmap_root)

    assert selected == attempt_dir
    assert mapper_num_threads == 1


def test_colmap_mapper_retry_keeps_all_threads_for_an_attempt_that_predates_snapshots(
    tmp_path: Path,
) -> None:
    colmap_root = tmp_path / "colmap"
    attempt_dir = colmap_root / "attempt-001"
    (attempt_dir / "sparse").mkdir(parents=True)
    (attempt_dir / ".features-complete").write_text("complete\n", encoding="utf-8")
    (attempt_dir / ".matching-complete").write_text("complete\n", encoding="utf-8")

    selected, mapper_num_threads = select_colmap_attempt(colmap_root)

    assert selected == attempt_dir
    # No snapshot directory at all is no evidence, not evidence of a crash.
    assert mapper_num_threads is None


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


def test_cross_view_pairs_link_every_view_within_one_frame_only(tmp_path: Path) -> None:
    attempt = _attempt(8)
    pairs = cross_view_pair_names(attempt)
    # Every unordered view pair, once per frame, and never across frames.
    assert len(pairs) == attempt.frame_count * 8 * 7 // 2
    assert len(set(pairs)) == len(pairs)
    for first, second in pairs:
        assert first.split("/")[1] == second.split("/")[1]
        assert first < second
    listed = tmp_path / "cross-view-pairs.txt"
    assert write_cross_view_pair_list(listed, attempt) == len(pairs)
    lines = listed.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "view_00/frame_000001.jpg view_01/frame_000001.jpg"
    assert len(lines) == len(pairs)
    assert write_cross_view_pair_list(listed, attempt) == len(pairs)


def test_cross_view_matching_is_skipped_for_legacy_flat_datasets(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    assert cv2.imwrite(
        str(dataset / "images" / "frame_000001_00.jpg"),
        np.zeros((64, 64, 3), dtype=np.uint8),
    )
    legacy = LegacyReconstructionAttemptV1(frame_count=2, images_per_equirect=8, crop_bottom=0.2)
    commands = build_colmap_commands(dataset, legacy, LegacyReconstructionConfigV1())
    assert "cross_view" not in commands
    assert set(commands) == {"features", "matching", "mapping"}


def test_pyramid_marker_allows_header_only_revalidation(tmp_path: Path) -> None:
    source = tmp_path / "images"
    (source / "view_00").mkdir(parents=True)
    image_path = source / "view_00" / "frame_000001.jpg"
    assert cv2.imwrite(str(image_path), np.full((64, 64, 3), 120, dtype=np.uint8))
    build_image_pyramid(source, tmp_path, "images", 2)
    marker = pyramid_level_marker(tmp_path, "images", 2)
    assert marker.is_file()
    downscaled = tmp_path / "images_2" / "view_00" / "frame_000001.jpg"
    original = downscaled.read_bytes()

    calls: list[str] = []
    real_imread = cv2.imread

    def counting_imread(path: str, *arguments: object) -> object:
        calls.append(path)
        return real_imread(path, *arguments)

    cv2.imread = counting_imread  # type: ignore[assignment]
    try:
        build_image_pyramid(source, tmp_path, "images", 2)
    finally:
        cv2.imread = real_imread  # type: ignore[assignment]
    assert calls == []
    assert downscaled.read_bytes() == original

    # A level whose contents no longer match the factor is still rejected.
    assert cv2.imwrite(str(downscaled), np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="Existing pyramid file is invalid"):
        build_image_pyramid(source, tmp_path, "images", 2)


def test_pyramid_without_marker_still_fully_validates_existing_masks(tmp_path: Path) -> None:
    source = tmp_path / "masks"
    (source / "view_00").mkdir(parents=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:32] = 255
    assert cv2.imwrite(str(source / "view_00" / "view.jpg.png"), mask)
    target_dir = tmp_path / "masks_2" / "view_00"
    target_dir.mkdir(parents=True)
    assert cv2.imwrite(
        str(target_dir / "view.jpg.png"), np.full((32, 32), 128, dtype=np.uint8)
    )
    with pytest.raises(RuntimeError, match="not binary"):
        build_image_pyramid(source, tmp_path, "masks", 1, is_mask=True)


def test_interleaved_image_names_orders_frame_major_for_the_colmap_image_list(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    for view in range(3):
        (images / f"view_{view:02d}").mkdir(parents=True)
        for frame in (1, 2, 10):
            assert cv2.imwrite(
                str(images / f"view_{view:02d}" / f"frame_{frame:06d}.jpg"),
                np.zeros((8, 8, 3), dtype=np.uint8),
            )
    # Frame-major, and frame 10 sorts after frame 2 rather than lexically before it.
    assert interleaved_image_names(images) == [
        f"view_{view:02d}/frame_{frame:06d}.jpg"
        for frame in (1, 2, 10)
        for view in range(3)
    ]
