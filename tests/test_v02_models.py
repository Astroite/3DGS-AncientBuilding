from __future__ import annotations

from pathlib import Path

import pytest

from gsdb.manifests import canonical_hash, load_capture_manifest, save_yaml
from gsdb.models import (
    Camera,
    CaptureManifestV2,
    NormalizationSettings,
    PanoramaSource,
    PreparedInputConfig,
    RunConfigV3,
    SourceFile,
    TimeSelection,
)


def _capture(kind: str = "equirect_video") -> CaptureManifestV2:
    suffix = {
        "equirect_video": ".mp4",
        "equirect_sequence": ".jpg",
        "insta360_insv": ".insv",
    }[kind]
    return CaptureManifestV2(
        id="capture-001",
        location_id="site-001",
        scene_id="scene-001",
        source=PanoramaSource(
            kind=kind,
            projection="dual_fisheye" if kind == "insta360_insv" else "equirectangular",
            files=[
                SourceFile(
                    windows_path=rf"D:\capture{suffix}",
                    sha256="a" * 64,
                    byte_size=123,
                )
            ],
        ),
        camera=Camera(make="Insta360", model="X5"),
        selection=TimeSelection(end_seconds=10),
        normalization=NormalizationSettings(width=7680, height=3840),
    )


@pytest.mark.parametrize(
    "kind", ["equirect_video", "equirect_sequence", "insta360_insv"]
)
def test_capture_v2_three_source_kinds_round_trip(tmp_path: Path, kind: str) -> None:
    capture = _capture(kind)
    path = tmp_path / "capture.yaml"
    save_yaml(path, capture)
    assert load_capture_manifest(path) == capture


def test_run_config_v3_hash_includes_input_adapter_lineage() -> None:
    capture = _capture()
    input_config = PreparedInputConfig(
        source_kind="equirect_video",
        source_sha256=["a" * 64],
        source_probe={
            "width": 7680,
            "height": 3840,
            "fps": 29.97,
            "frame_count": 300,
            "duration_seconds": 10,
            "codec": "h264",
        },
        normalization=capture.normalization,
        selection=capture.selection,
        candidate_frame_indices=list(range(4)),
    )
    first = RunConfigV3(
        capture_id=capture.id,
        input_dataset_sha256="b" * 64,
        prepared_relative_path="inputs/prepared/capture-001/hash",
        input=input_config,
        preprocess={"target_frames": 4},
        reconstruction={
            "primary": {
                "frame_count": 2,
                "images_per_equirect": 8,
                "projection_fov_degrees": 120,
                "projection_size": 512,
                "crop_bottom": 0.2,
            },
            "fallback": {
                "frame_count": 3,
                "images_per_equirect": 14,
                "projection_fov_degrees": 110,
                "projection_size": 512,
                "crop_bottom": 0.15,
            },
        },
    )
    second = first.model_copy(deep=True)
    assert canonical_hash(first) == canonical_hash(second)
    second.input.candidate_frame_indices[0] = 9
    assert canonical_hash(first) != canonical_hash(second)
