"""Central Run schema predicates so pipeline/CLI stop scattering magic numbers."""
from __future__ import annotations

from typing import Any


def schema_version(run_or_config: Any) -> int:
    config = getattr(run_or_config, "config", run_or_config)
    value = getattr(config, "schema_version", None)
    return int(value) if value is not None else 1


def uses_run_owned_inputs(version: int) -> bool:
    """Schema 6 keeps prepared candidates inside the Run, never Capture-shared."""
    return version >= 6


def uses_temporal_segment_pipeline(version: int) -> bool:
    """Schema 5/6 reconstruct once, then at most one targeted repair + segment QA."""
    return version >= 5


def uses_rate_sampled_prepared_input(version: int) -> bool:
    """Schema 4+ require rate-sampled prepared manifests rather than uniform dumps."""
    return version >= 4


def uses_legacy_nerfstudio_export(version: int) -> bool:
    """Only pre-schema-5 runs still publish through the Nerfstudio export path."""
    return version < 5


def allows_segment_train(version: int) -> bool:
    return version in (5, 6)


def allows_repair_attempt(version: int) -> bool:
    return version in (5, 6)


def preprocess_handlers() -> dict[str, str]:
    """Documentation map: schema generation -> preprocess implementation module."""
    return {
        "6": "pipeline_v6.preprocess_v6",
        "1-5": "pipeline.preprocess_run",
    }


def reconstruct_handlers() -> dict[str, str]:
    """Documentation map: schema generation -> reconstruct implementation module."""
    return {
        "5-6": "pipeline_v5.reconstruct_v5",
        "4": "pipeline._reconstruct_run_v4",
        "1-3": "pipeline.reconstruct_run",
    }
