"""Isolated JSON-lines operation worker for the Qt workbench.

The GUI sends one JSON request on stdin. Stdout contains only structured
events; complete engine output remains in the Run's own log files.
"""
from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from gsstudio.application import operations as ops
from gsstudio.application.errors import classify

_PROTOCOL_STREAM = sys.stdout


def emit(kind: str, action: str, message: str, run_id: str | None = None,
         **detail: Any) -> None:
    _PROTOCOL_STREAM.write(json.dumps({"kind": kind, "action": action, "message": message,
                                       "run_id": run_id, "detail": detail}, ensure_ascii=False) + "\n")
    _PROTOCOL_STREAM.flush()


def _event(event: ops.OperationEvent) -> None:
    emit(event.kind, event.action, event.message, event.run_id, **(event.detail or {}))


def _stop_requested(request: dict[str, Any]) -> bool:
    value = request.get("stop_file")
    return bool(value and Path(value).is_file())


def execute(request: dict[str, Any]) -> dict[str, Any]:
    action = request["action"]
    root = Path(request["root"])
    location = request.get("location_id")
    scene = request.get("scene_id")
    run_id = request.get("run_id")
    if action == "create_location":
        result = ops.create_location(root, location, request["name"])
        return {"location_id": result.id}
    if action == "create_scene":
        result = ops.create_scene(root, location, scene, request["name"])
        return {"scene_id": result.id}
    if action == "create_capture":
        result = ops.create_capture(
            root, location, scene, request["capture_id"], request["source_type"],
            [Path(value) for value in request["sources"]],
            camera_model=request.get("camera_model", "unknown"),
            fps=request.get("fps"), start_seconds=request.get("start_seconds", 0.0),
            end_seconds=request.get("end_seconds"),
            output_width=request.get("output_width"),
            output_height=request.get("output_height"),
            horizontal_fov=request.get("horizontal_fov", 84.0),
        )
        return {"capture_id": result.id}
    if action == "relink_capture":
        result = ops.relink_capture_sources(
            root, location, scene, request["capture_id"],
            [Path(value) for value in request["sources"]],
        )
        return {"capture_id": result.id}
    if action == "probe_capture":
        return ops.probe_capture(root, location, scene, request["capture_id"])
    if action == "ingest":
        result = ops.ingest(root, location, scene, request["capture_id"])
        return {"capture_id": result.id}
    if action == "create_run":
        run_id = ops.create_and_preprocess(
            root, location, scene, request["capture_id"],
            candidate_fps=request.get("candidate_fps", 1.0),
            selected_per_second=request.get("selected_per_second", 1),
            primary_per_second=request.get("primary_per_second", 1),
            mask_review_gate=True, sink=_event,
        )
        return ops.advance_run(root, location, scene, run_id,
                               stop_requested=lambda: _stop_requested(request), sink=_event)
    if action == "resume_run":
        return ops.advance_run(root, location, scene, run_id,
                               stop_requested=lambda: _stop_requested(request), sink=_event)
    if action == "finalize_masks":
        result = ops.finalize_masks(root, location, scene, run_id,
                                    attempt=request.get("attempt", "primary"))
        emit("completed", "finalize_masks", "遮罩已固定", run_id)
        return {"finalization_sha256": result["finalization_sha256"]}
    if action == "review_qa":
        result = ops.review_qa(root, location, scene, run_id,
                               request["accepted"], request["notes"])
        return {"status": result.status.value}
    if action == "segment_choices":
        return ops.segment_choices(root, location, scene, run_id)
    if action == "train_segment":
        return ops.train_segment(root, location, scene, run_id,
                                 request["segment"],
                                 backend=request.get("backend", "gsplat"),
                                 steps=request.get("steps"),
                                 resume=request.get("resume", False), sink=_event,
                                 output=Path(request["output"]) if request.get("output") else None,
                                 preview_every=500 if request.get("backend", "gsplat") == "gsplat" else 0)
    if action == "cleanup":
        return ops.cleanup(root, location, scene, run_id,
                           apply=request.get("apply", False))
    if action == "verified_models":
        return {"models": ops.verified_models(root, location, scene, run_id)}
    raise ValueError(f"Unknown operation: {action}")


def main() -> int:
    try:
        payload = json.loads(sys.stdin.readline())
        if not isinstance(payload, dict):
            raise ValueError("Operation request must be an object")
        request = ops.OperationRequest.from_mapping(payload)
        with redirect_stdout(sys.stderr):
            result = ops.OperationResult(request.action, execute(request.to_mapping()))
        emit("result", result.action, "操作完成", result.data.get("run_id"), result=result.data)
        return 0
    except Exception as error:
        classified = classify(error)
        emit("error", "worker", classified.message,
             code=classified.code, path=classified.path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
