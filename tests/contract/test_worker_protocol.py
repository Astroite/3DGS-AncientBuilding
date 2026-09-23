"""Worker contract: Run advancement delegates to application and emits JSON lines."""
from __future__ import annotations

import io
import json

from gsstudio.application.contracts import OperationEvent, OperationRequest
from gsstudio.interfaces.worker import main as worker


def test_resume_run_uses_application_and_serializes_progress(tmp_path, monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(worker, "_PROTOCOL_STREAM", stream)

    def advance(root, location_id, scene_id, run_id, *, stop_requested, sink):
        assert root == tmp_path
        assert (location_id, scene_id, run_id) == ("loc", "scene", "run")
        assert stop_requested() is False
        sink(OperationEvent("waiting", "mask_review", "请审核遮罩", run_id))
        return {"run_id": run_id, "next": "mask_review"}

    monkeypatch.setattr(worker.ops, "advance_run", advance)
    payload = {"action": "resume_run", "root": str(tmp_path),
               "location_id": "loc", "scene_id": "scene", "run_id": "run"}
    request = OperationRequest.from_mapping(payload)
    result = worker.execute(request.to_mapping())

    assert result == {"run_id": "run", "next": "mask_review"}
    assert json.loads(stream.getvalue()) == {
        "kind": "waiting", "action": "mask_review", "message": "请审核遮罩",
        "run_id": "run", "detail": {},
    }
