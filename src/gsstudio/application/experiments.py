"""Shared write operations for the CLI and the GS-Studio workbench.

The methods here return domain objects.  They never infer success from an
artifact's existence, and callers must not turn an exception into success.
"""
from __future__ import annotations

import json
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from gsstudio.infrastructure.persistence.manifests import load_capture_manifest, load_model, save_yaml
from gsstudio.pipeline.masks.finalize import MaskFinalizationMissingError, finalize_mask_dataset, validate_mask_finalization
from gsstudio.pipeline.masks.masking import validate_mask_filter
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.domain.models import (
    IMAGE_SUFFIXES, Camera, CaptureManifest, CaptureSource, LocationManifest,
    NormalizationSettings, PreparedInputConfig, Region, Rights, RunConfig,
    RunStatus, SceneManifest, SourceFile, SourceProbe, TimeSelection, utc_now,
)
from gsstudio.infrastructure.paths import host_path, location_dir, scene_dir
from gsstudio.pipeline.editor.ply import FLOAT_PROPERTIES, read_ply_header
from gsstudio.pipeline.stages import (
    ingest_capture, mask_run, preprocess_run, reconstruct_run, review_run,
    write_qa_report,
)
from gsstudio.pipeline.retention import auto_cleanup, cleanup_run
from gsstudio.infrastructure.runtime.run_lock import run_session
from gsstudio.infrastructure.persistence.run_repository import create_run, load_run, save_run
from gsstudio.pipeline.input.sources import load_prepared_input, prepare_capture_input, probe_capture_source
from gsstudio.pipeline.training.runner import train_package
from gsstudio.pipeline.training.data import prepare_segment


from gsstudio.application.contracts import EventSink, _emit
from gsstudio.application._shared import _scene, _selected_dataset, _verify_model
from gsstudio.application.captures import verify_capture_sources


def segment_choices(root: Path, location_id: str, scene_id: str,
                    run_id: str) -> dict[str, Any]:
    scene = _scene(root, location_id, scene_id)
    run = load_run(scene, run_id)
    if not run.selected_dataset or run.stages["reconstruct"].status.value != "succeeded":
        raise RuntimeError("No verified reconstruction is selected")
    dataset = _selected_dataset(scene, run)
    report = json.loads((dataset / "segments.json").read_text(encoding="utf-8"))
    return {
        "training_status": report["training_status"],
        "coverage_status": report["coverage_status"],
        "segments": [item for item in report["segments"] if item["status"] == "passed"],
        "report": str(dataset / "segments.json"),
    }


def train_segment(root: Path, location_id: str, scene_id: str, run_id: str,
                  segment: str, *, backend: str = "gsplat", steps: int | None = None,
                  resume: bool = False, sink: EventSink | None = None,
                  output: Path | None = None, photo_comp: bool = True,
                  use_bilateral_grid: bool = True,
                  use_sparse_depth: bool = True, dry_run: bool = False,
                  duration_seconds: float | None = None,
                  preview_every: int = 0) -> dict[str, Any]:
    scene = _scene(root, location_id, scene_id)
    with run_session(scene / run_id):
        run = load_run(scene, run_id)
        if run.status != RunStatus.ACCEPTED:
            raise RuntimeError("QA must be accepted before GUI training")
        capture = load_capture_manifest(scene / "captures" / f"{run.config.capture_id}.yaml")
        verify_capture_sources(capture)
        choices = segment_choices(root, location_id, scene_id, run_id)
        if segment not in {item["id"] for item in choices["segments"]}:
            raise ValueError("Select a passed segment")
        label = run.metrics["selected_attempt"]
        work = scene / run_id
        records = [json.loads(line) for line in
                   (work / f"selected-{label}-metrics.jsonl").read_text(encoding="utf-8").splitlines()
                   if line]
        package_name = segment if duration_seconds is None else f"{segment}-first-{duration_seconds:g}s"
        package = work / "training-data" / package_name
        dataset = _selected_dataset(scene, run)
        _emit(sink, "started", "train", "正在准备训练数据", run_id, segment=segment)
        prepare_segment(dataset, records, run.config.reconstruction.primary,
                        run.config.segment_qa, segment, package,
                        duration_seconds=duration_seconds, reuse_files=True)
        experiments = run.metrics.setdefault("training_experiments", [])
        previous = next((item for item in reversed(experiments)
                         if item.get("segment") == segment and
                         item.get("backend") == backend and
                         (output is None or item.get("output") == str(output.resolve())) and
                         item.get("status") in {"failed", "preparing"}), None)
        if resume and previous:
            experiment = previous
            output = Path(experiment["output"])
            steps = experiment.get("steps", steps)
            dispatch = output / "dispatch.json"
            if dispatch.is_file():
                recorded = json.loads(dispatch.read_text(encoding="utf-8"))
                if recorded.get("status") == "succeeded":
                    if (recorded.get("backend") != backend or
                            recorded.get("package_sha256") != sha256_file(package / "dataset.json") or
                            (backend == "gsplat" and json.loads(
                                (output / "training.json").read_text(encoding="utf-8")
                            ).get("status") != "succeeded")):
                        raise RuntimeError("Completed training evidence differs from the experiment")
                    _verify_model(output / "model.ply", recorded["model_sha256"])
                    experiment.update(recorded)
                    save_run(scene, run)
                    _emit(sink, "completed", "train", "已恢复完成的训练记录", run_id,
                          output=str(output / "model.ply"))
                    return {**recorded, "output": str(output / "model.ply")}
        else:
            if resume:
                raise ValueError("No failed or interrupted experiment exists to resume")
            output = output.resolve() if output is not None else (
                work / "experiments" / f"{backend}-{segment}-{uuid.uuid4().hex[:8]}"
            )
            experiment = {"segment": segment, "backend": backend,
                          "output": str(output), "steps": steps,
                          "status": "preparing"}
            experiments.append(experiment)
        _emit(sink, "progress", "train", "训练日志已就绪", run_id,
              log_path=str(output / "train.log"), output=str(output))
        save_run(scene, run)
        try:
            result = train_package(
                package, output, backend, steps, photo_comp, resume, dry_run,
                use_bilateral_grid=use_bilateral_grid,
                use_sparse_depth=use_sparse_depth,
                preview_every=preview_every,
            )
            experiment.update(result)
            if dry_run:
                _emit(sink, "completed", "train", "训练输入已准备；尚未训练", run_id)
                return {**result, "output": str(output)}
            if result.get("status") != "succeeded" or not result.get("model_sha256"):
                raise RuntimeError("Training did not publish a verified PLY")
            _verify_model(output / "model.ply", result["model_sha256"])
            _emit(sink, "completed", "train", "PLY 已校验", run_id,
                  output=str(output / "model.ply"))
            return {**result, "output": str(output / "model.ply")}
        except Exception as error:
            experiment.update(status="failed", error=str(error))
            _emit(sink, "failed", "train", str(error), run_id)
            raise
        finally:
            save_run(scene, run)
