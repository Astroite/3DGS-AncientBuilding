"""Same-camera training comparison; camera requests never alter training settings."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.training.data import json_write


class _ImagePane(QWidget):
    def __init__(self, owner: "TrainingCompare", title: str):
        super().__init__()
        self.owner = owner
        self.title = title
        self.pixmap = QPixmap()
        self._last: QPointF | None = None
        self.setMinimumSize(280, 240)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self.pixmap.isNull():
            fit = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
            scale = fit * self.owner.zoom
            width, height = self.pixmap.width() * scale, self.pixmap.height() * scale
            target = QRectF((self.width() - width) / 2 + self.owner.pan.x(),
                            (self.height() - height) / 2 + self.owner.pan.y(), width, height)
            painter.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(8, 20, self.title)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._last is not None:
            delta = event.position() - self._last
            self.owner.pan += delta
            self._last = event.position()
            self.owner.update_panes()

    def mouseReleaseEvent(self, _event) -> None:
        self._last = None

    def wheelEvent(self, event) -> None:
        self.owner.zoom = min(16.0, max(1.0, self.owner.zoom * (1.15 if event.angleDelta().y() > 0 else 1 / 1.15)))
        self.owner.update_panes()
        event.accept()


class TrainingCompare(QWidget):
    def __init__(self):
        super().__init__()
        self.output: Path | None = None
        self.package_sha256: str | None = None
        self.request_id: str | None = None
        self.request_written = False
        self.shown: tuple[str, int] | None = None
        self.zoom = 1.0
        self.pan = QPointF()
        self.camera = QComboBox()
        self.camera.currentIndexChanged.connect(self._camera_changed)
        self.status = QLabel("预览尚未生成")
        self.source = _ImagePane(self, "同视角源图")
        self.render = _ImagePane(self, "训练渲染")
        images = QHBoxLayout()
        images.addWidget(self.source, 1)
        images.addWidget(self.render, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.camera)
        layout.addWidget(self.status)
        layout.addLayout(images, 1)

    def update_panes(self) -> None:
        self.source.update()
        self.render.update()

    def clear(self) -> None:
        self.output = None
        self.package_sha256 = None
        self.request_id = None
        self.request_written = False
        self.shown = None
        self.camera.clear()
        self.camera.setEnabled(False)
        self.source.pixmap = QPixmap()
        self.render.pixmap = QPixmap()
        self.status.setText("预览尚未生成")
        self.update_panes()

    def configure(self, output: Path, package: Path | None, digest: str | None,
                  backend: str) -> None:
        self.output = output
        self.request_id = None
        self.request_written = False
        self.shown = None
        self.package_sha256 = None
        self.source.pixmap = QPixmap()
        self.render.pixmap = QPixmap()
        self.camera.blockSignals(True)
        self.camera.clear()
        self.camera.blockSignals(False)
        self.update_panes()
        if backend != "gsplat":
            self.status.setText("Postshot 不提供训练中的交互式相机预览")
            self.camera.setEnabled(False)
            return
        try:
            valid_package = package is not None and digest is not None and sha256_file(package / "dataset.json") == digest
        except OSError:
            valid_package = False
        if not valid_package:
            self.status.setText("训练包身份无法核对；预览不可用")
            self.camera.setEnabled(False)
            return
        self.package_sha256 = digest
        try:
            rows = json.loads((package / "dataset.json").read_text(encoding="utf-8"))["images"]
        except (OSError, ValueError, KeyError, TypeError):
            self.status.setText("训练包相机清单无法读取")
            self.camera.setEnabled(False)
            return
        self.camera.blockSignals(True)
        for row in rows:
            self.camera.addItem(f"{'验' if row['split'] == 'validation' else '训'} · {row['source_image']}", row["image"])
        first = next((index for index, row in enumerate(rows) if row["split"] == "validation"), 0)
        self.camera.setCurrentIndex(first)
        self.camera.blockSignals(False)
        self.camera.setEnabled(bool(rows))
        if rows:
            self._camera_changed(first)

    def _camera_changed(self, index: int) -> None:
        if index < 0 or self.output is None or self.package_sha256 is None:
            return
        self.zoom = 1.0
        self.pan = QPointF()
        self.source.pixmap = QPixmap()
        self.render.pixmap = QPixmap()
        self.update_panes()
        self.request_id = uuid.uuid4().hex
        self.request_written = False
        self.status.setText("当前相机渲染更新中…")
        self.shown = None
        self._write_request()

    def _write_request(self) -> None:
        if (self.output is None or not (self.output / "dispatch.json").is_file() or self.request_written or
                self.request_id is None or self.package_sha256 is None):
            return
        try:
            dispatch = json.loads((self.output / "dispatch.json").read_text(encoding="utf-8"))
            if dispatch.get("package_sha256") != self.package_sha256:
                raise ValueError("实验训练包身份与相机请求不一致")
            json_write(self.output / "preview-request.json", {
                "schema_version": 1, "package_sha256": self.package_sha256,
                "request_id": self.request_id, "image": self.camera.currentData(),
            })
            self.request_written = True
        except (OSError, ValueError) as error:
            self.status.setText(f"相机请求写入失败：{error}")

    def refresh(self) -> None:
        if self.output is None or self.request_id is None:
            return
        self._write_request()
        index = self.output / "preview.json"
        if not index.is_file():
            return
        try:
            report = json.loads(index.read_text(encoding="utf-8"))
            if (report.get("schema_version") != 1 or
                    report.get("request_id") != self.request_id or
                    report.get("image") != self.camera.currentData()):
                return
            token = (report["request_id"], report["step"])
            if self.shown == token:
                return
            pixmaps = []
            for key in ("source_file", "render_file"):
                name = report[key]
                if not isinstance(name, str) or Path(name).name != name or not name.startswith("preview-"):
                    return
                content = (self.output / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != report[key.removesuffix("_file") + "_sha256"]:
                    return
                pixmap = QPixmap()
                if not pixmap.loadFromData(content):
                    return
                pixmaps.append(pixmap)
            self.source.pixmap, self.render.pixmap = pixmaps
            self.shown = token
            self.status.setText(f"同视角 · step {report['step']} / {report['target']}")
            self.update_panes()
        except (OSError, KeyError, ValueError, TypeError):
            self.status.setText("预览文件正在更新…")
