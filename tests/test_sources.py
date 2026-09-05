from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb.media import sha256_file
from gsdb.models import (
    Camera,
    CaptureManifestV2,
    NormalizationSettings,
    PanoramaSource,
    SourceFile,
    SourceProbe,
    TimeSelection,
)
from gsdb.sources import (
    ALLOW_FAKE_HELPER_ENV,
    copy_candidate_frames,
    invoke_media_helper,
    ordered_sequence_files,
    prepare_capture_input,
    probe_image_sequence,
    uniform_frame_indices,
)


@pytest.fixture(autouse=True)
def _allow_ci_fake_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_FAKE_HELPER_ENV, "1")


def _write_image(path: Path, value: int) -> None:
    assert cv2.imwrite(str(path), np.full((20, 40, 3), value, dtype=np.uint8))


def _source_file(path: Path) -> SourceFile:
    return SourceFile(
        windows_path=str(path), sha256=sha256_file(path), byte_size=path.stat().st_size
    )


def test_image_sequence_order_dimensions_and_preparation(tmp_path: Path) -> None:
    source = tmp_path / "sequence"
    source.mkdir()
    for index in (10, 2, 1, 3):
        _write_image(source / f"pano_{index}.jpg", index)
    paths = list(source.iterdir())
    assert [item.stem for item in ordered_sequence_files(paths)] == [
        "pano_1",
        "pano_2",
        "pano_3",
        "pano_10",
    ]
    assert probe_image_sequence(paths, fps=2)["duration_seconds"] == 2
    capture = CaptureManifestV2(
        id="capture-001",
        location_id="site-001",
        scene_id="scene-001",
        source=PanoramaSource(
            kind="equirect_sequence",
            projection="equirectangular",
            files=[_source_file(item) for item in ordered_sequence_files(paths)],
            probe=SourceProbe(fps=2),
        ),
        camera=Camera(make="Insta360", model="X5"),
        selection=TimeSelection(end_seconds=2),
        normalization=NormalizationSettings(width=40, height=20),
    )
    scene = tmp_path / "scene"
    prepared = prepare_capture_input(scene, capture, target_frames=3)
    assert len(prepared.frame_paths) == 3
    assert prepared.manifest_path.is_file()
    assert prepare_capture_input(scene, capture, target_frames=3).dataset_sha256 == prepared.dataset_sha256

    original_hash = prepared.frame_sha256s[0]
    _write_image(prepared.frame_paths[0], 250)
    with pytest.raises(RuntimeError, match="use --resume"):
        prepare_capture_input(scene, capture, target_frames=3)
    assert prepared.path.is_dir()
    assert not prepared.path.with_name(prepared.path.name + ".building").exists()
    repaired = prepare_capture_input(scene, capture, target_frames=3, resume=True)
    assert repaired.frame_sha256s[0] == original_hash
    assert sha256_file(repaired.frame_paths[0]) == original_hash

    work_candidates = tmp_path / "work-candidates"
    copied = copy_candidate_frames(repaired, work_candidates)
    _write_image(copied[0], 251)
    with pytest.raises(RuntimeError, match="use --resume"):
        copy_candidate_frames(repaired, work_candidates)
    copy_candidate_frames(repaired, work_candidates, resume=True)
    assert sha256_file(copied[0]) == original_hash
    assert sha256_file(repaired.frame_paths[0]) == original_hash


def test_sequence_without_numeric_index_requires_fps(tmp_path: Path) -> None:
    first = tmp_path / "alpha.jpg"
    second = tmp_path / "beta.jpg"
    _write_image(first, 1)
    _write_image(second, 2)
    with pytest.raises(ValueError, match="provide --fps"):
        ordered_sequence_files([first, second])


def test_uniform_indices_are_deterministic_and_bounded() -> None:
    assert uniform_frame_indices(100, 10, 2, 8, 3) == [30, 50, 70]


def test_fake_mediasdk_exports_only_requested_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "capture.insv"
    raw.write_bytes(b"fake insv")
    raw.with_suffix(".insv.probe.json").write_text(
        json.dumps(
            {
                "width": 40,
                "height": 20,
                "fps": 10,
                "frame_count": 100,
                "duration_seconds": 10,
                "codec": "fake",
                "camera_model": "Insta360 X5",
            }
        ),
        encoding="utf-8",
    )
    helper = Path(__file__).resolve().parents[1] / "tools/mediasdk-helper/fake-helper.py"
    monkeypatch.setenv("GSDB_MEDIA_HELPER", str(helper))
    capture = CaptureManifestV2(
        id="capture-001",
        location_id="site-001",
        scene_id="scene-001",
        source=PanoramaSource(
            kind="insta360_insv",
            projection="dual_fisheye",
            files=[_source_file(raw)],
        ),
        camera=Camera(make="Insta360", model="X5"),
        selection=TimeSelection(end_seconds=10),
        normalization=NormalizationSettings(width=40, height=20),
    )
    prepared = prepare_capture_input(tmp_path / "scene", capture, target_frames=5)
    assert len(prepared.frame_paths) == 5
    assert not list((prepared.path / "frames").glob("*.png"))
    assert all(item.is_file() for item in prepared.frame_paths)
    assert prepared.source_probe is not None
    assert prepared.source_probe["camera_model"] == "Insta360 X5"


def test_python_media_helper_requires_explicit_test_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = Path(__file__).resolve().parents[1] / "tools/mediasdk-helper/fake-helper.py"
    monkeypatch.setenv("GSDB_MEDIA_HELPER", str(helper))
    monkeypatch.delenv(ALLOW_FAKE_HELPER_ENV)
    with pytest.raises(RuntimeError, match="test-only"):
        invoke_media_helper("capabilities", {}, tmp_path / "protocol")


def test_x6_is_accepted_by_mediasdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "capture.insv"
    raw.write_bytes(b"fake x6")
    raw.with_suffix(".insv.probe.json").write_text(
        json.dumps(
            {
                "width": 40,
                "height": 20,
                "fps": 10,
                "frame_count": 100,
                "duration_seconds": 10,
                "codec": "fake",
                "camera_model": "Insta360 X6",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "GSDB_MEDIA_HELPER",
        str(Path(__file__).resolve().parents[1] / "tools/mediasdk-helper/fake-helper.py"),
    )
    capture = CaptureManifestV2(
        id="capture-001",
        location_id="site-001",
        scene_id="scene-001",
        source=PanoramaSource(
            kind="insta360_insv",
            projection="dual_fisheye",
            files=[_source_file(raw)],
        ),
        camera=Camera(make="Insta360", model="X6"),
        selection=TimeSelection(end_seconds=10),
        normalization=NormalizationSettings(width=40, height=20),
    )
    prepared = prepare_capture_input(tmp_path / "scene", capture, target_frames=5)
    assert len(prepared.frame_paths) == 5
    assert prepared.source_probe is not None
    assert prepared.source_probe["camera_model"] == "Insta360 X6"


def test_fake_mediasdk_supports_utf8_paths_and_explicit_dual_insv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "全景素材"
    source_dir.mkdir()
    first = source_dir / "VID_001_00.insv"
    second = source_dir / "VID_001_10.insv"
    first.write_bytes(b"front")
    second.write_bytes(b"back")
    first.with_suffix(".insv.probe.json").write_text(
        json.dumps(
            {
                "width": 40,
                "height": 20,
                "fps": 10,
                "frame_count": 100,
                "duration_seconds": 10,
                "camera_model": "Insta360 X5",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "GSDB_MEDIA_HELPER",
        str(Path(__file__).resolve().parents[1] / "tools/mediasdk-helper/fake-helper.py"),
    )
    capture = CaptureManifestV2(
        id="capture-001",
        location_id="site-001",
        scene_id="scene-001",
        source=PanoramaSource(
            kind="insta360_insv",
            projection="dual_fisheye",
            files=[_source_file(first), _source_file(second)],
        ),
        camera=Camera(make="Insta360", model="X5"),
        selection=TimeSelection(end_seconds=10),
        normalization=NormalizationSettings(width=40, height=20),
    )
    prepared = prepare_capture_input(tmp_path / "scene", capture, target_frames=5)
    payload = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert len(payload["lineage"]["source_files"]) == 2
    assert "全景素材" in payload["lineage"]["source_files"][0]["windows_path"]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("no_response", "without a response"),
        ("wrong_schema", "Unsupported MediaSDK helper protocol"),
        ("missing_versions", "omitted helper_version"),
        ("error", "forced_failure"),
    ],
)
def test_mediasdk_protocol_failures_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    message: str,
) -> None:
    helper = Path(__file__).resolve().parents[1] / "tools/mediasdk-helper/fake-helper.py"
    monkeypatch.setenv("GSDB_MEDIA_HELPER", str(helper))
    monkeypatch.setenv("GSDB_FAKE_HELPER_MODE", mode)
    with pytest.raises(RuntimeError, match=message):
        invoke_media_helper("capabilities", {}, tmp_path / "协议缓存")


def test_interrupted_insv_export_resumes_only_missing_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "capture.insv"
    raw.write_bytes(b"fake insv")
    raw.with_suffix(".insv.probe.json").write_text(
        json.dumps(
            {
                "width": 40,
                "height": 20,
                "fps": 10,
                "frame_count": 100,
                "duration_seconds": 10,
                "camera_model": "Insta360 X5",
            }
        ),
        encoding="utf-8",
    )
    helper = Path(__file__).resolve().parents[1] / "tools/mediasdk-helper/fake-helper.py"
    monkeypatch.setenv("GSDB_MEDIA_HELPER", str(helper))
    monkeypatch.setenv("GSDB_FAKE_HELPER_FAIL_AFTER", "2")
    capture = CaptureManifestV2(
        id="capture-001",
        location_id="site-001",
        scene_id="scene-001",
        source=PanoramaSource(
            kind="insta360_insv",
            projection="dual_fisheye",
            files=[_source_file(raw)],
        ),
        camera=Camera(make="Insta360", model="X5"),
        selection=TimeSelection(end_seconds=10),
        normalization=NormalizationSettings(width=40, height=20),
    )
    scene = tmp_path / "scene"
    with pytest.raises(RuntimeError, match="interrupted_export"):
        prepare_capture_input(scene, capture, target_frames=5)
    building = next((scene / "prepared/capture-001").glob("*.building"))
    assert len(list((building / "frames").glob("*.jpg"))) == 2

    monkeypatch.delenv("GSDB_FAKE_HELPER_FAIL_AFTER")
    prepared = prepare_capture_input(scene, capture, target_frames=5, resume=True)
    assert len(prepared.frame_paths) == 5
    assert not building.exists()


@pytest.mark.parametrize(
    ("environment_name", "message"),
    [
        ("GSDB_FAKE_HELPER_EXPORT_VERSION", "version changed"),
        ("GSDB_FAKE_HELPER_PNG_AS_JPEG", "not a JPEG"),
        ("GSDB_FAKE_HELPER_MUTATE_SOURCE", "source changed"),
    ],
)
def test_insv_export_rejects_protocol_or_source_integrity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    message: str,
) -> None:
    raw = tmp_path / "capture.insv"
    raw.write_bytes(b"fake insv")
    raw.with_suffix(".insv.probe.json").write_text(
        json.dumps(
            {
                "width": 40,
                "height": 20,
                "fps": 10,
                "frame_count": 100,
                "duration_seconds": 10,
                "camera_model": "Insta360 X5",
            }
        ),
        encoding="utf-8",
    )
    helper = Path(__file__).resolve().parents[1] / "tools/mediasdk-helper/fake-helper.py"
    monkeypatch.setenv("GSDB_MEDIA_HELPER", str(helper))
    monkeypatch.setenv(environment_name, "fake-2.0" if "VERSION" in environment_name else "1")
    capture = CaptureManifestV2(
        id="capture-001",
        location_id="site-001",
        scene_id="scene-001",
        source=PanoramaSource(
            kind="insta360_insv",
            projection="dual_fisheye",
            files=[_source_file(raw)],
        ),
        camera=Camera(make="Insta360", model="X5"),
        selection=TimeSelection(end_seconds=10),
        normalization=NormalizationSettings(width=40, height=20),
    )
    with pytest.raises(RuntimeError, match=message):
        prepare_capture_input(tmp_path / "scene", capture, target_frames=5)
    capture_cache = tmp_path / "scene/prepared/capture-001"
    assert not any(
        item.is_dir() and not item.name.endswith(".building")
        for item in capture_cache.iterdir()
    )
    assert not list(capture_cache.glob("*.building/frames/*.jpg"))


def test_source_manifest_rejects_duplicate_or_wrong_extension(tmp_path: Path) -> None:
    raw = tmp_path / "capture.insv"
    raw.write_bytes(b"raw")
    record = _source_file(raw)
    with pytest.raises(ValueError, match="distinct paths"):
        PanoramaSource(
            kind="insta360_insv",
            projection="dual_fisheye",
            files=[record, record],
        )
    with pytest.raises(ValueError, match="distinct paths"):
        PanoramaSource(
            kind="insta360_insv",
            projection="dual_fisheye",
            files=[
                SourceFile(
                    windows_path=r"D:\Media\same.insv",
                    sha256="a" * 64,
                    byte_size=1,
                ),
                SourceFile(
                    windows_path="D:/Media/same.insv",
                    sha256="a" * 64,
                    byte_size=1,
                ),
            ],
        )
    with pytest.raises(ValueError, match=r"\.insv extension"):
        PanoramaSource(
            kind="insta360_insv",
            projection="dual_fisheye",
            files=[
                SourceFile(
                    windows_path=str(tmp_path / "capture.mp4"),
                    sha256="a" * 64,
                    byte_size=1,
                )
            ],
        )
