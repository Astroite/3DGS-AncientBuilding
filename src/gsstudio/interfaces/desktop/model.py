"""Qt viewer and non-destructive SH3 Gaussian editing workbench."""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.editor.runtime import Runtime
from gsstudio.pipeline.training.data import json_write


class _Viewport(QLabel):
    def __init__(self, owner: "ModelWorkbench"):
        super().__init__("打开已验证的模型后显示预览")
        self.owner = owner
        self.setMinimumSize(520, 380)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self._start = None
        self._last = None
        self._button = None
        self._selection: QRect | None = None

    def mousePressEvent(self, event) -> None:
        self._start = event.position().toPoint()
        self._last = self._start
        self._button = event.button()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._last is None:
            return
        point = event.position().toPoint()
        if self.owner.select_mode.isChecked() and self._button == Qt.MouseButton.LeftButton:
            self._selection = QRect(self._start, point).normalized()
            self.update()
        else:
            self.owner.drag_view(point.x() - self._last.x(), point.y() - self._last.y(),
                                 self._button)
        self._last = point

    def mouseReleaseEvent(self, event) -> None:
        if self._selection is not None:
            self.owner.select_rectangle(self._selection)
            self._selection = None
            self.update()
        self._last = self._start = self._button = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        self.owner.zoom(event.angleDelta().y())
        event.accept()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._selection is not None:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
            painter.drawRect(self._selection)


class _NumbersDialog(QDialog):
    def __init__(self, title: str, names: list[str], values: list[float], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.fields = []
        form = QFormLayout()
        for name, value in zip(names, values):
            field = QDoubleSpinBox()
            field.setRange(-1_000_000_000, 1_000_000_000)
            field.setDecimals(6)
            field.setValue(float(value))
            form.addRow(name, field)
            self.fields.append(field)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> list[float]:
        return [field.value() for field in self.fields]


class ModelWorkbench(QWidget):
    """Uses the existing gsplat renderer and Editor without changing a checkpoint."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runtime = Runtime()
        self.editor = None
        self.source: Path | None = None
        self.source_sha256: str | None = None
        self.center = np.zeros(3)
        self.distance = 5.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.render_size = (640, 480)
        self.viewport = _Viewport(self)
        self.title = QLabel("模型预览与编辑", objectName="pageTitle")
        self.status = QLabel("尚未打开模型", objectName="caption")
        self.select_mode = QCheckBox("矩形选择")
        self.through = QCheckBox("穿透选择")
        fit = QPushButton("适配视角")
        fit.clicked.connect(self.fit)
        clear = QPushButton("清空选择")
        clear.clicked.connect(lambda: self._edit("select", self.camera(), [-2, -2, -1, -1], True))
        delete = QPushButton("删除选中")
        delete.clicked.connect(lambda: self._edit("delete"))
        crop = QPushButton("裁剪框")
        crop.clicked.connect(self._crop)
        transform = QPushButton("变换")
        transform.clicked.connect(self._transform)
        undo = QPushButton("撤销")
        undo.clicked.connect(lambda: self._edit("undo"))
        redo = QPushButton("重做")
        redo.clicked.connect(lambda: self._edit("redo"))
        save = QPushButton("保存编辑状态")
        save.clicked.connect(self.save_state)
        export = QPushButton("导出编辑 PLY")
        export.clicked.connect(self.export)
        buttons = QVBoxLayout()
        for widget in (self.select_mode, self.through, fit, clear, delete,
                       crop, transform, undo, redo, save, export):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        body = QHBoxLayout()
        body.addWidget(self.viewport, 1)
        body.addLayout(buttons)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.status)
        layout.addLayout(body, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

    def open_model(self, path: Path, digest: str) -> None:
        self.source = path
        self.source_sha256 = digest
        self.editor = None
        self.status.setText(f"正在读取 {path.name}…")

        def load():
            self.runtime.editor = None
            if sha256_file(path) != digest:
                raise RuntimeError("训练模型在打开前已改变")
            pointer = path.parent / "edits" / "current.json"
            state = None
            if pointer.is_file():
                record = json.loads(pointer.read_text(encoding="utf-8"))
                if record.get("source_sha256") != digest:
                    raise RuntimeError("编辑状态与原始模型身份不匹配")
                state_name = record.get("state")
                if not isinstance(state_name, str) or Path(state_name).name != state_name:
                    raise RuntimeError("编辑状态路径无效")
                state = path.parent / "edits" / state_name
                if sha256_file(state) != record["state_sha256"]:
                    raise RuntimeError("编辑状态文件已改变")
            self.runtime.load_model(path, state=state)
        self.runtime.submit(load)

    def _poll(self) -> None:
        while not self.runtime.events.empty():
            kind, payload = self.runtime.events.get_nowait()
            if kind == "editor":
                self.editor = payload["editor"]
                if payload.get("fit", True):
                    self.fit()
                self.status.setText(
                    f"保留 {int(self.editor.keep.sum()):,} · 已选 {int(self.editor.selected.sum()):,} Gaussian"
                )
            elif kind == "preview":
                pixels = payload["pixels"]
                height, width = pixels.shape[:2]
                self.render_size = (width, height)
                image = QImage(pixels.tobytes(), width, height, 3 * width,
                               QImage.Format.Format_RGB888).copy()
                self.viewport.setPixmap(QPixmap.fromImage(image).scaled(
                    self.viewport.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
            elif kind in {"notice", "error"}:
                self.status.setText(payload.get("message", kind))
            elif kind == "status" and payload.get("state") == "viewing":
                self.status.setText("模型可查看；原始 PLY 保持只读")

    def fit(self) -> None:
        if self.editor is None:
            return
        xyz = self.editor.xyz()[self.editor.keep]
        low, high = xyz.min(axis=0), xyz.max(axis=0)
        self.center = (low + high) / 2
        self.distance = max(float(np.linalg.norm(high - low)) * 1.2, 0.1)
        self.request_camera()

    def camera(self) -> dict:
        view_width = max(1, self.viewport.width())
        view_height = max(1, self.viewport.height())
        aspect = view_width / view_height
        width, height = ((640, max(1, round(640 / aspect))) if aspect >= 1 else
                         (max(1, round(640 * aspect)), 640))
        forward = np.array([
            math.sin(self.yaw) * math.cos(self.pitch), math.sin(self.pitch),
            math.cos(self.yaw) * math.cos(self.pitch),
        ])
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        eye = self.center - forward * self.distance
        rotation = np.array([right, down, forward])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = -rotation @ eye
        return {"world_to_camera": matrix.tolist(),
                "K": [[width * .9, 0, width / 2], [0, width * .9, height / 2], [0, 0, 1]],
                "width": width, "height": height}

    def request_camera(self) -> None:
        if self.editor is not None:
            self.runtime.set_camera(self.camera())

    def drag_view(self, dx: int, dy: int, button) -> None:
        if button == Qt.MouseButton.LeftButton:
            self.yaw += dx * .006
            self.pitch = float(np.clip(self.pitch + dy * .006, -1.5, 1.5))
        elif button == Qt.MouseButton.RightButton:
            rotation = np.asarray(self.camera()["world_to_camera"])[:3, :3]
            self.center += (-dx * rotation[0] - dy * rotation[1]) * self.distance * .002
        self.request_camera()

    def zoom(self, delta: int) -> None:
        self.distance = max(1e-5, self.distance * math.exp(-delta / 120 * .12))
        self.request_camera()

    def select_rectangle(self, rect: QRect) -> None:
        if self.editor is None:
            return
        width, height = self.render_size
        scale = min(self.viewport.width() / width, self.viewport.height() / height)
        offset_x = (self.viewport.width() - width * scale) / 2
        offset_y = (self.viewport.height() - height * scale) / 2
        box = [(rect.left() - offset_x) / scale, (rect.top() - offset_y) / scale,
               (rect.right() - offset_x) / scale, (rect.bottom() - offset_y) / scale]
        self._edit("select", self.camera(), box, self.through.isChecked())

    def _edit(self, method: str, *args) -> None:
        if self.editor is None:
            return
        self.runtime.submit(self.runtime.edit, method, *args)

    def _crop(self) -> None:
        if self.editor is None:
            return
        xyz = self.editor.xyz()[self.editor.keep]
        low, high = xyz.min(axis=0), xyz.max(axis=0)
        dialog = _NumbersDialog("三维裁剪框", ["最小 X", "最小 Y", "最小 Z", "最大 X", "最大 Y", "最大 Z"],
                                [*low, *high], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            values = dialog.values()
            self._edit("crop", values[:3], values[3:], True, True)

    def _transform(self) -> None:
        if self.editor is None:
            return
        dialog = _NumbersDialog("模型变换", ["平移 X", "平移 Y", "平移 Z", "旋转 X", "旋转 Y", "旋转 Z", "统一缩放"],
                                [0, 0, 0, 0, 0, 0, 1], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            values = dialog.values()
            self._edit("transform", values[:3], values[3:6], values[6])

    def _persist_state(self) -> None:
        if self.editor is None or self.source is None or self.source_sha256 is None:
            return
        if sha256_file(self.source) != self.source_sha256:
            raise RuntimeError("原始训练 PLY 已改变；拒绝保存编辑状态")
        directory = self.source.parent / "edits"
        directory.mkdir(exist_ok=True)
        name = f"state-{uuid.uuid4().hex}.npz"
        state = directory / name
        self.editor.save_state(state)
        json_write(directory / "current.json", {
            "schema_version": 1, "source": self.source.name,
            "source_sha256": self.source_sha256,
            "state": name, "state_sha256": sha256_file(state),
        })
        self.runtime.emit("notice", message=f"编辑状态已保存：{state}")

    def save_state(self) -> None:
        self.runtime.submit(self._persist_state)

    def export(self) -> None:
        if self.editor is None or self.source is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出编辑 PLY", str(self.source.parent / "edit.ply"),
                                              "Gaussian PLY (*.ply)")
        if not path:
            return
        destination = Path(path)
        if destination == self.source or destination.exists() or destination.with_suffix(".edit.json").exists():
            QMessageBox.warning(self, "保护原始模型", "请选择一个尚不存在的新 PLY 路径。")
            return

        def write():
            self._persist_state()
            self.editor.export(destination)
            json_write(destination.with_suffix(".edit.json"), {
                "schema_version": 1, "source": str(self.source),
                "source_sha256": self.source_sha256,
                "output": str(destination), "output_sha256": sha256_file(destination),
            })
            self.runtime.emit("notice", message=f"编辑 PLY 已导出：{destination}")
        self.runtime.submit(write)

    def shutdown(self) -> None:
        self.runtime.quit.set()
