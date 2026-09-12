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
    PreparedInputConfigV2,
    RunConfigV3,
    RunConfigV4,
    SourceProbe,
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


def _run_config_v4(**overrides: object) -> RunConfigV4:
    capture = _capture()
    values = {
        "capture_id": capture.id,
        "input_dataset_sha256": "b" * 64,
        "prepared_relative_path": "prepared/capture-001/hash",
        "input": PreparedInputConfigV2(
            source_kind="equirect_video",
            source_sha256=["a" * 64],
            source_probe=SourceProbe(
                width=7680,
                height=3840,
                fps=30,
                frame_count=72,
                duration_seconds=2.4,
                codec="h264",
            ),
            normalization=capture.normalization,
            selection=TimeSelection(end_seconds=2.4),
            candidate_frame_indices=list(range(12)),
            candidate_fps=5.0,
        ),
    }
    values.update(overrides)
    return RunConfigV4(**values)


def test_run_config_v4_defaults_to_five_two_primary_one_fallback_two() -> None:
    config = _run_config_v4()
    assert config.preprocess.candidate_fps == 5.0
    assert config.preprocess.selected_per_second == 2
    assert config.masking.mask_discard_threshold == 0.05
    assert config.masking.mask_review_required is False
    assert config.reconstruction.primary.temporal_rank_limit == 1
    assert config.reconstruction.primary.images_per_equirect == 8
    assert config.reconstruction.fallback.temporal_rank_limit == 2
    assert config.reconstruction.fallback.images_per_equirect == 14


def test_v5_upgrade_preserves_legacy_serialization_and_changes_new_defaults():
    from gsdb.models import RunConfigV5
    legacy=_run_config_v4()
    serialized=legacy.model_dump(mode='json')
    assert RunConfigV4.model_validate(serialized).model_dump(mode='json')==serialized
    inputs={k:serialized[k] for k in ('capture_id','input_dataset_sha256','prepared_relative_path','input')}
    new=RunConfigV5(**inputs)
    assert new.reconstruction.primary.images_per_equirect==14
    assert new.masking.mask_discard_threshold==0.005
    assert legacy.reconstruction.primary.images_per_equirect==8
    assert legacy.masking.mask_discard_threshold==0.05
    invalid=new.model_dump(mode='json')
    invalid['reconstruction']['repair_attempts']=2
    with pytest.raises(ValueError):
        RunConfigV5.model_validate(invalid)


def test_run_config_v4_validates_temporal_density_chain() -> None:
    with pytest.raises(ValueError, match="primary <= fallback"):
        _run_config_v4(
            preprocess={"candidate_fps": 5.0, "selected_per_second": 2},
            reconstruction={
                "primary": {
                    "temporal_rank_limit": 2,
                    "images_per_equirect": 8,
                    "projection_fov_degrees": 120,
                    "projection_size": 512,
                    "crop_bottom": 0.2,
                },
                "fallback": {
                    "temporal_rank_limit": 1,
                    "images_per_equirect": 14,
                    "projection_fov_degrees": 110,
                    "projection_size": 512,
                    "crop_bottom": 0.15,
                },
            },
        )
