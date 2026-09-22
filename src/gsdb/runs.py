from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .manifests import canonical_hash, load_model, save_yaml
from .models import (
    STAGES,
    RunConfig,
    RunManifest,
    RunStatus,
    StageRecord,
    StageStatus,
    utc_now,
)
from .paths import ensure_run_dir


STAGE_ORDER = STAGES


def run_manifest_path(scene_path: Path, run_id: str) -> Path:
    return scene_path / run_id / "manifest.yaml"


def load_run(scene_path: Path, run_id: str) -> RunManifest:
    run = load_model(run_manifest_path(scene_path, run_id), RunManifest)
    current_hash = canonical_hash(run.config)
    if current_hash != run.config_hash:
        raise RuntimeError(
            f"Run config hash mismatch for {run_id}; configuration changes require a new run"
        )
    return run


def save_run(scene_path: Path, run: RunManifest) -> None:
    run.updated_at = utc_now()
    save_yaml(run_manifest_path(scene_path, run.id), run)


def create_run(
    scene_path: Path,
    location_id: str,
    scene_id: str,
    config: RunConfig,
    now: datetime | None = None,
    run_id: str | None = None,
) -> RunManifest:
    created_at = now or utc_now()
    digest = canonical_hash(config)
    identifier = run_id or f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{digest[:8]}"
    if (scene_path / identifier).exists():
        raise FileExistsError(f"Run already exists: {identifier}")
    run = RunManifest(
        id=identifier,
        location_id=location_id,
        scene_id=scene_id,
        config_hash=digest,
        config=config,
        created_at=created_at,
        updated_at=created_at,
    )
    ensure_run_dir(scene_path, identifier, exist_ok=True)
    save_run(scene_path, run)
    return run


def require_previous_stages(run: RunManifest, stage: str) -> None:
    index = STAGE_ORDER.index(stage)
    for previous in STAGE_ORDER[:index]:
        if previous == "qa":
            continue
        status = run.stages[previous].status
        if status != StageStatus.SUCCEEDED:
            raise RuntimeError(
                f"Stage {stage} requires {previous}=succeeded; current status is {status.value}"
            )


def begin_stage(
    run: RunManifest, stage: str, resume: bool = False, force: bool = False
) -> bool:
    """Open a stage for work, or report that a completed one can be reused."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage: {stage}")
    if stage not in ("preprocess", "qa"):
        require_previous_stages(run, stage)
    current = run.stages.get(stage, StageRecord())
    if current.status == StageStatus.SUCCEEDED and not force:
        if resume:
            return False
        raise RuntimeError(f"Stage {stage} already succeeded; use --resume to reuse it")
    if (
        current.status in (StageStatus.PROCESSING, StageStatus.FAILED)
        and not resume
        and not force
    ):
        raise RuntimeError(f"Stage {stage} was already attempted; use --resume or create a new run")
    run.stages[stage] = StageRecord(status=StageStatus.PROCESSING, started_at=utc_now())
    run.status = RunStatus.PROCESSING
    run.active_stage = stage
    return True


def complete_stage(
    run: RunManifest, stage: str, message: str | None = None, log_path: str | None = None
) -> None:
    record = run.stages[stage]
    record.status = StageStatus.SUCCEEDED
    record.completed_at = utc_now()
    if record.started_at:
        record.elapsed_seconds = (record.completed_at - record.started_at).total_seconds()
    record.message = message
    record.log_path = log_path
    run.active_stage = None
    if stage == "qa":
        run.status = RunStatus.NEEDS_REVIEW


def fail_stage(
    run: RunManifest, stage: str, message: str, log_path: str | None = None
) -> None:
    record = run.stages[stage]
    record.status = StageStatus.FAILED
    record.completed_at = utc_now()
    if record.started_at:
        record.elapsed_seconds = (record.completed_at - record.started_at).total_seconds()
    record.message = message
    record.log_path = log_path
    run.status = RunStatus.FAILED
    run.active_stage = None
