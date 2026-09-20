"""Training instrumentation and publish-time culling knobs.

Kept out of RunConfig on purpose: these change how a run is observed or
published, not what is learned, so they must not invalidate config hashes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Splatfacto densification is written in steps; what matters is per-view sampling
# budget before splitting stops. Defaults match the validated ~1,080-view baseline.
DENSIFICATION_BASELINE_IMAGES = 1080
DENSIFICATION_BASELINE_STEPS = {
    "warmup_length": 500,
    "stop_screen_size_at": 4000,
    "stop_split_at": 15000,
}
DENSIFICATION_MAX_FRACTION = 0.75
SPLATFACTO_REFINE_EVERY = 100
SPLATFACTO_DEFAULT_RESET_ALPHA_EVERY = 30


def densification_schedule(image_count: int, max_iterations: int) -> dict[str, int]:
    """Step bounds that keep densification coverage constant as a scene grows."""
    if image_count < 1:
        raise ValueError(f"Image count must be positive, got {image_count}")
    ratio = image_count / DENSIFICATION_BASELINE_IMAGES
    schedule = {
        name: max(1, round(steps * ratio))
        for name, steps in DENSIFICATION_BASELINE_STEPS.items()
    }
    schedule["stop_split_at"] = min(
        schedule["stop_split_at"], max(1, int(max_iterations * DENSIFICATION_MAX_FRACTION))
    )
    schedule["stop_screen_size_at"] = min(
        schedule["stop_screen_size_at"], max(1, schedule["stop_split_at"] - 1)
    )
    schedule["warmup_length"] = min(
        schedule["warmup_length"], max(1, schedule["stop_screen_size_at"] - 1)
    )
    minimum_reset_alpha_every = (
        math.ceil((image_count + SPLATFACTO_REFINE_EVERY) / SPLATFACTO_REFINE_EVERY) + 1
    )
    schedule["reset_alpha_every"] = max(
        SPLATFACTO_DEFAULT_RESET_ALPHA_EVERY, minimum_reset_alpha_every
    )
    schedule["refine_every"] = SPLATFACTO_REFINE_EVERY
    return schedule


@dataclass(frozen=True)
class TrainSettings:
    """Instrumentation around training that does not change what is learned."""

    VIS_CHOICES = (
        "viewer",
        "wandb",
        "tensorboard",
        "comet",
        "viewer+tensorboard",
        "viewer+wandb",
    )

    steps_per_save: int = 10_000
    vis: str = "tensorboard"

    def __post_init__(self) -> None:
        if self.vis not in self.VIS_CHOICES:
            raise ValueError(f"vis must be one of {self.VIS_CHOICES}, got {self.vis!r}")
        if self.steps_per_save < 1:
            raise ValueError(f"steps_per_save must be positive, got {self.steps_per_save}")

    def command_arguments(self) -> list[str]:
        return [
            "--save-only-latest-checkpoint",
            "False",
            "--steps-per-save",
            str(self.steps_per_save),
            "--vis",
            self.vis,
        ]

    def as_metrics(self) -> dict[str, Any]:
        return {"steps_per_save": self.steps_per_save, "vis": self.vis}


@dataclass(frozen=True)
class CullSettings:
    """Publish-time culling knobs; not part of the run config hash."""

    enabled: bool = True
    distance_factor: float = 3.0
    scale_factor: float = 1.0
    max_removed_fraction: float = 0.05

    def as_metrics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "distance_factor": self.distance_factor,
            "scale_factor": self.scale_factor,
            "max_removed_fraction": self.max_removed_fraction,
        }
