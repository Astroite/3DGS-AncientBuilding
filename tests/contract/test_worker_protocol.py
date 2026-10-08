"""Worker contract: Run advancement delegates to application and emits JSON lines."""
from __future__ import annotations

import io
import json

import pytest
from PySide6.QtWidgets import QApplication

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


def test_external_relink_worker_uses_shared_operation(tmp_path, monkeypatch):
    def relink(root, import_id, *, new_root, mapping_file):
        assert (root, import_id, new_root, mapping_file) == (
            tmp_path, "import-one", tmp_path / "moved", None,
        )
        return {"status": "prepared", "id": import_id}

    monkeypatch.setattr(worker.ops, "relink_external_import", relink)
    assert worker.execute({"action": "relink_external_import", "root": str(tmp_path),
                           "import_id": "import-one", "new_root": str(tmp_path / "moved")}) == {
        "status": "prepared", "id": "import-one",
    }


def test_missing_frozen_worker_does_not_leave_operation_busy(monkeypatch):
    from gsstudio.interfaces.desktop import workflow

    app = QApplication.instance() or QApplication([])
    client = workflow.OperationClient()

    def missing():
        raise FileNotFoundError("Missing frozen worker")

    monkeypatch.setattr(workflow, "worker_command", missing)
    with pytest.raises(FileNotFoundError, match="Missing frozen worker"):
        client.start({"action": "list_external_imports", "root": "C:/Data"})
    assert not client.busy
    assert client._control is None
    assert client._stop_path is None
