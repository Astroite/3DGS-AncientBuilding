from gsdb.cli import SMOKE_PROFILE, apply_smoke_profile
from gsdb.models import ReconstructionAttempt, RunConfig
from gsdb.media import select_temporal_records
from gsdb.sources import fixed_rate_frame_indices


def _defaults() -> dict[str, int | None]:
    return {
        "target_frames": 270,
        "primary_frames": 135,
        "fallback_frames": 180,
        "primary_projection_size": 2048,
        "fallback_projection_size": 1746,
        "mask_qa_sample_count": 16,
        "train_iterations": None,
    }


def test_smoke_profile_shrinks_every_expensive_setting() -> None:
    reduced = apply_smoke_profile(_defaults())
    assert reduced == {
        "target_frames": SMOKE_PROFILE["target_frames"],
        "primary_frames": SMOKE_PROFILE["primary_frames"],
        "fallback_frames": SMOKE_PROFILE["fallback_frames"],
        "primary_projection_size": SMOKE_PROFILE["projection_size"],
        "fallback_projection_size": SMOKE_PROFILE["projection_size"],
        "mask_qa_sample_count": SMOKE_PROFILE["mask_qa_sample_count"],
        "train_iterations": SMOKE_PROFILE["train_iterations"],
    }
    defaults = _defaults()
    for key, value in reduced.items():
        baseline = defaults[key]
        assert baseline is None or int(value or 0) < baseline


def test_smoke_profile_is_a_ceiling_and_never_raises_an_explicit_setting() -> None:
    smaller = {
        "target_frames": 20,
        "primary_frames": 10,
        "fallback_frames": 12,
        "primary_projection_size": 512,
        "fallback_projection_size": 512,
        "mask_qa_sample_count": 4,
        "train_iterations": 500,
    }
    assert apply_smoke_profile(smaller) == smaller


def _config(settings: dict[str, int | None]) -> RunConfig:
    """Build the run config exactly the way the preprocess command does."""
    config = RunConfig(
        capture_id="capture-009-4k",
        input_sha256="a" * 64,
        masking={"qa_sample_count": int(settings["mask_qa_sample_count"] or 16)},
        reconstruction={
            "primary": {
                "frame_count": int(settings["primary_frames"] or 0),
                "images_per_equirect": 8,
                "projection_fov_degrees": 120.0,
                "projection_size": int(settings["primary_projection_size"] or 0),
                "crop_bottom": 0.20,
                "use_rig": True,
            },
            "fallback": {
                "frame_count": int(settings["fallback_frames"] or 0),
                "images_per_equirect": 14,
                "projection_fov_degrees": 110.0,
                "projection_size": int(settings["fallback_projection_size"] or 0),
                "crop_bottom": 0.15,
                "use_rig": True,
            },
        },
    )
    config.preprocess.target_frames = int(settings["target_frames"] or 0)
    if settings["train_iterations"] is not None:
        config.train.max_iterations = int(settings["train_iterations"])
    return config


def test_smoke_profile_produces_a_valid_and_distinct_run_config() -> None:
    smoke = _config(apply_smoke_profile(_defaults()))
    full = _config(_defaults())
    assert smoke.reconstruction.primary.frame_count == SMOKE_PROFILE["primary_frames"]
    assert smoke.train.max_iterations == SMOKE_PROFILE["train_iterations"]
    # A different config must never collide with the full-quality run's identity.
    assert smoke.model_dump(mode="json") != full.model_dump(mode="json")


def test_smoke_projection_size_still_satisfies_the_reconstruction_bounds() -> None:
    reduced = apply_smoke_profile(_defaults())
    attempt = ReconstructionAttempt(
        frame_count=int(reduced["primary_frames"]),
        images_per_equirect=8,
        projection_fov_degrees=120.0,
        projection_size=int(reduced["primary_projection_size"]),
        crop_bottom=0.20,
    )
    assert attempt.projection_size == SMOKE_PROFILE["projection_size"]
    assert attempt.frame_count <= SMOKE_PROFILE["target_frames"]


def test_v4_smoke_keeps_five_two_density_for_exact_25_10_40_counts() -> None:
    _, timestamps = fixed_rate_frame_indices(300, 30.0, 0.0, 5.0, 5.0)
    candidates = [
        {
            "file": f"frame_{index:06d}.jpg",
            "timestamp_seconds": timestamp,
            "selection_score": float(index % 5),
        }
        for index, timestamp in enumerate(timestamps, start=1)
    ]
    selected = select_temporal_records(candidates, 0.0, 2)
    primary = [item for item in selected if item["temporal_rank"] == 1]
    assert len(candidates) == 25
    assert len(selected) == 10
    assert len(primary) * 8 == 40
