"""The desktop comparison never presents an earlier camera as the current one."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.interfaces.desktop.compare import TrainingCompare
from gsstudio.pipeline.training.preview import publish


def test_camera_switch_rejects_stale_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    package = tmp_path / "package"
    output = tmp_path / "experiment"
    package.mkdir()
    output.mkdir()
    rows = [{"image": f"images/{index}.png", "source_image": f"{index}.png",
             "split": "validation"} for index in range(2)]
    (package / "dataset.json").write_text(json.dumps({"images": rows}), encoding="utf-8")
    digest = sha256_file(package / "dataset.json")
    (output / "dispatch.json").write_text(json.dumps({"package_sha256": digest}), encoding="utf-8")
    view = TrainingCompare()
    view.configure(output, package, digest, "gsplat")
    first = view.request_id
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    publish(output, request_id=first, row=rows[0], step=1, target=2,
            source_bgr=image, render_rgb=image)
    view.camera.setCurrentIndex(1)
    view.refresh()
    assert view.request_id != first
    assert view.render.pixmap.isNull()
    publish(output, request_id=view.request_id, row=rows[1], step=2, target=2,
            source_bgr=image, render_rgb=image)
    view.refresh()
    assert not view.render.pixmap.isNull()

    # The producer may reuse this slot immediately after the UI reads it.
    # Decoding must use the exact bytes whose digest was checked.
    view.shown = None
    render_path = output / json.loads((output / "preview.json").read_text(encoding="utf-8"))["render_file"]
    ok, encoded = cv2.imencode(".png", np.full((4, 4, 3), 255, dtype=np.uint8))
    assert ok
    original_read = Path.read_bytes
    changed = False

    def replace_after_read(path: Path) -> bytes:
        nonlocal changed
        content = original_read(path)
        if path == render_path and not changed:
            changed = True
            render_path.write_bytes(encoded.tobytes())
        return content

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    view.refresh()
    assert changed
    assert view.render.pixmap.toImage().pixelColor(0, 0).red() == 0
    view.close()
