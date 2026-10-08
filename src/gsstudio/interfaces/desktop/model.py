"""Qt viewer and non-destructive SH3 Gaussian editing workbench."""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QRect, QPointF
from PySide6.QtGui import QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.editor.runtime import Runtime
from gsstudio.pipeline.editor.edit import project_points
from gsstudio.pipeline.training.data import json_write


class _Viewport(QLabel):
    def __init__(self, owner: "ModelWorkbench"):
        super().__init__("打开已验证的模型后显示预览")
        self.owner = owner
        self.setMinimumSize(520, 380)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._start = None
        self._last = None
        self._button = None
        self._selection: QRect | None = None
        self._handle: str | None = None

    def mousePressEvent(self, event) -> None:
        self._start = event.position().toPoint()
        self._last = self._start
        self._button = event.button()
        if event.button() == Qt.MouseButton.LeftButton:
            self._handle = self.owner.hit_handle(event.position())
            if self._handle:
                self.owner.begin_handle(self._handle, event.position())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._last is None:
            return
        point = event.position().toPoint()
        if self._handle:
            self.owner.move_handle(event.position())
        elif self.owner.select_mode.isChecked() and self._button == Qt.MouseButton.LeftButton:
            self._selection = QRect(self._start, point).normalized()
            self.update()
        else:
            self.owner.drag_view(point.x() - self._last.x(), point.y() - self._last.y(),
                                 self._button)
        self._last = point

    def mouseReleaseEvent(self, event) -> None:
        if self._handle:
            self.owner.end_handle()
            self._handle = None
        elif self._selection is not None:
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
            painter.end()
        self.owner.draw_gizmo(self)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.owner.rescale_preview()
        self.owner.request_camera()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._selection = None
            self._handle = None
            self.owner._cancel_pending()
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)


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
        self.last_image: QPixmap | None = None
        self.display_camera: dict | None = None
        self._handle_start: QPointF | None = None
        self._handle_key: str | None = None
        self._handle_moved = False
        self._handle_values: list[float] = []
        self.pivot = np.zeros(3)
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
        crop.clicked.connect(lambda: self.tool.setCurrentIndex(1))
        transform = QPushButton("变换")
        transform.clicked.connect(lambda: self.tool.setCurrentIndex(2))
        undo = QPushButton("撤销")
        undo.clicked.connect(lambda: self._edit("undo"))
        redo = QPushButton("重做")
        redo.clicked.connect(lambda: self._edit("redo"))
        save = QPushButton("保存编辑状态")
        save.clicked.connect(self.save_state)
        export = QPushButton("导出编辑 PLY")
        export.clicked.connect(self.export)
        self.tool = QComboBox()
        for label, value in (("观察", "view"), ("裁剪框", "crop"),
                             ("平移", "move"), ("旋转", "rotate"), ("统一缩放", "scale")):
            self.tool.addItem(label, value)
        self.crop_keep = QCheckBox("保留框内")
        self.crop_keep.setChecked(True)
        self.crop_keep.toggled.connect(self._preview_pending)
        self.crop_fields = self._fields(["最小 X", "最小 Y", "最小 Z", "最大 X", "最大 Y", "最大 Z"],
                                         [0, 0, 0, 1, 1, 1])
        self.transform_fields = self._fields(["平移 X", "平移 Y", "平移 Z", "旋转 X", "旋转 Y", "旋转 Z", "统一缩放"],
                                              [0, 0, 0, 0, 0, 0, 1])
        self.crop_form = self._field_form(self.crop_fields, ["最小 X", "最小 Y", "最小 Z", "最大 X", "最大 Y", "最大 Z"])
        self.transform_form = self._field_form(self.transform_fields, ["平移 X", "平移 Y", "平移 Z", "旋转 X", "旋转 Y", "旋转 Z", "统一缩放"])
        apply = QPushButton("应用当前编辑")
        apply.clicked.connect(self._apply_pending)
        cancel = QPushButton("取消预览")
        cancel.clicked.connect(self._cancel_pending)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self._send_preview)
        buttons = QVBoxLayout()
        for widget in (self.select_mode, self.through, fit, clear, delete,
                       crop, transform, self.tool, self.crop_keep,
                       self.crop_form, self.transform_form, apply, cancel,
                       undo, redo, save, export):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        controls = QWidget()
        controls.setLayout(buttons)
        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setMinimumWidth(240)
        sidebar.setWidget(controls)
        body = QHBoxLayout()
        body.addWidget(self.viewport, 1)
        body.addWidget(sidebar)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.status)
        layout.addLayout(body, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self.cancel_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.cancel_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.cancel_shortcut.activated.connect(self._cancel_pending)
        self.tool.currentIndexChanged.connect(self._tool_changed)
        self._tool_changed()

    def open_model(self, path: Path, digest: str) -> None:
        self._cancel_pending()
        self.source = path
        self.source_sha256 = digest
        self.editor = None
        self.last_image = None
        self.display_camera = None
        self.viewport.setPixmap(QPixmap())
        self.viewport.setText("模型读取中…")
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
                if self.source is None or payload.get("model_identity") != str(self.source.resolve()):
                    continue
                self.editor = payload["editor"]
                if payload.get("fit", True):
                    self.fit()
                self.status.setText(
                    f"保留 {int(self.editor.keep.sum()):,} · 已选 {int(self.editor.selected.sum()):,} Gaussian"
                )
            elif kind == "preview":
                if self.source is None or payload.get("model_identity") != str(self.source.resolve()):
                    continue
                if payload.get("revision") != self.runtime.preview_revision:
                    continue
                pixels = payload["pixels"]
                height, width = pixels.shape[:2]
                self.render_size = (width, height)
                image = QImage(pixels.tobytes(), width, height, 3 * width,
                               QImage.Format.Format_RGB888).copy()
                self.last_image = QPixmap.fromImage(image)
                self.display_camera = payload["camera"]
                self.rescale_preview()
            elif kind in {"notice", "error"}:
                self.status.setText(payload.get("message", kind))
            elif kind == "status" and payload.get("state") == "viewing":
                self.status.setText("模型可查看；原始 PLY 保持只读")

    def rescale_preview(self) -> None:
        if self.last_image is not None and not self.last_image.isNull():
            self.viewport.setPixmap(self.last_image.scaled(
                self.viewport.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            self.viewport.update()

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
        if self.editor is None or self.display_camera is None:
            return
        width, height = self.render_size
        pixmap = self.viewport.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        scale = pixmap.width() / width
        offset_x = (self.viewport.width() - pixmap.width()) / 2
        offset_y = (self.viewport.height() - pixmap.height()) / 2
        box = [(rect.left() - offset_x) / scale, (rect.top() - offset_y) / scale,
               (rect.right() - offset_x) / scale, (rect.bottom() - offset_y) / scale]
        self._edit("select", self.display_camera, box, self.through.isChecked())

    def _edit(self, method: str, *args) -> None:
        if self.editor is None:
            return
        self.runtime.submit(self.runtime.edit, method, *args)

    def _fields(self, names: list[str], values: list[float]) -> list[QDoubleSpinBox]:
        fields = []
        for name, value in zip(names, values):
            field = QDoubleSpinBox()
            field.setRange(0.000001 if name == "统一缩放" else -1_000_000_000, 1_000_000_000)
            field.setDecimals(6)
            field.setValue(value)
            field.valueChanged.connect(self._preview_pending)
            fields.append(field)
        return fields

    @staticmethod
    def _field_form(fields: list[QDoubleSpinBox], names: list[str]) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        for name, field in zip(names, fields):
            form.addRow(name, field)
        return widget

    def _field_values(self, mode: str | None = None) -> list[float]:
        fields = self.crop_fields if (mode or self.tool.currentData()) == "crop" else self.transform_fields
        return [field.value() for field in fields]

    @staticmethod
    def _set_values(fields: list[QDoubleSpinBox], values) -> None:
        for field, value in zip(fields, values):
            field.blockSignals(True)
            field.setValue(float(value))
            field.blockSignals(False)

    def _tool_changed(self, *_args) -> None:
        self._preview_timer.stop()
        self._handle_start = None
        self._handle_key = None
        self._handle_moved = False
        self.runtime.submit(self.runtime.clear_edit_preview)
        mode = self.tool.currentData()
        self.crop_form.setVisible(mode == "crop")
        self.crop_keep.setVisible(mode == "crop")
        self.transform_form.setVisible(mode in {"move", "rotate", "scale"})
        if self.editor is not None:
            xyz = self.editor.xyz()[self.editor.keep]
            self.pivot = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
            if mode == "crop":
                self._set_values(self.crop_fields, [*xyz.min(axis=0), *xyz.max(axis=0)])
            else:
                self._set_values(self.transform_fields, [0, 0, 0, 0, 0, 0, 1])
        self.viewport.update()

    def _preview_pending(self, *_args) -> None:
        if self.editor is not None and self.tool.currentData() != "view":
            self._preview_timer.start()
            self.viewport.update()

    def _send_preview(self) -> None:
        mode = self.tool.currentData()
        if self.editor is None:
            return
        values = self._field_values()
        if mode == "crop":
            if any(values[i] >= values[i + 3] for i in range(3)):
                self.status.setText("裁剪框最小值必须小于最大值")
                return
            self.runtime.submit(self.runtime.set_edit_preview,
                                crop=(values[:3], values[3:], self.crop_keep.isChecked()))
        elif mode in {"move", "rotate", "scale"}:
            self.runtime.submit(self.runtime.set_edit_preview,
                                transform=(values[:3], values[3:6], values[6], self.pivot))

    def _apply_pending(self) -> None:
        if self.editor is None:
            return
        mode = self.tool.currentData()
        values = self._field_values()
        self._preview_timer.stop()
        if mode == "crop":
            if any(values[i] >= values[i + 3] for i in range(3)):
                self.status.setText("裁剪框最小值必须小于最大值")
                return
            self._edit("crop", values[:3], values[3:], self.crop_keep.isChecked(), True)
        elif mode in {"move", "rotate", "scale"}:
            if any(abs(value) > 1e-9 for value in values[:6]) or abs(values[6] - 1) > 1e-9:
                self._edit("transform", values[:3], values[3:6], values[6], self.pivot)
        self.tool.setCurrentIndex(0)

    def _cancel_pending(self) -> None:
        if hasattr(self, "_preview_timer"):
            self._preview_timer.stop()
            self._handle_start = None
            self._handle_key = None
            self._handle_moved = False
            self.viewport._handle = None
            self.viewport._selection = None
            self.runtime.submit(self.runtime.clear_edit_preview)
            self.tool.setCurrentIndex(0)
            self.viewport.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.tool.currentData() != "view":
            self._cancel_pending()
            event.accept()
            return
        super().keyPressEvent(event)

    def _project(self, points) -> np.ndarray:
        camera = self.display_camera or self.camera()
        xy, _ = project_points(np.asarray(points, dtype=float).reshape(-1, 3), camera)
        pixmap = self.viewport.pixmap()
        width = pixmap.width() if pixmap is not None and not pixmap.isNull() else self.viewport.width()
        height = pixmap.height() if pixmap is not None and not pixmap.isNull() else self.viewport.height()
        sx = width / max(1, camera["width"])
        sy = height / max(1, camera["height"])
        xy[:, 0] = xy[:, 0] * sx + (self.viewport.width() - width) / 2
        xy[:, 1] = xy[:, 1] * sy + (self.viewport.height() - height) / 2
        return xy

    def _gizmo_points(self) -> dict[str, np.ndarray]:
        mode = self.tool.currentData()
        if self.editor is None or mode == "view":
            return {}
        points = {}
        if mode == "crop":
            values = self._field_values("crop")
            low, high = np.asarray(values[:3]), np.asarray(values[3:])
            for bits in range(8):
                points[f"corner:{bits}"] = np.where([(bits >> axis) & 1 for axis in range(3)], high, low)
            for axis in range(3):
                for side in range(2):
                    point = (low + high) / 2
                    point[axis] = high[axis] if side else low[axis]
                    points[f"face:{axis}:{side}"] = point
        else:
            length = max(self.distance * .15, .05)
            if mode == "move":
                for axis in range(3):
                    point = self.pivot.copy()
                    point[axis] += length
                    points[f"axis:{axis}"] = point
                for first, second in ((0, 1), (0, 2), (1, 2)):
                    point = self.pivot.copy()
                    point[first] += length * .55
                    point[second] += length * .55
                    points[f"plane:{first}:{second}"] = point
            elif mode == "scale":
                points["scale"] = self.pivot + length * np.array([1., 1., 1.])
        return points

    def draw_gizmo(self, viewport: _Viewport) -> None:
        if not hasattr(self, "tool"):
            return
        mode = self.tool.currentData()
        if self.editor is None or mode == "view":
            return
        painter = QPainter(viewport)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        points = self._gizmo_points()
        if mode == "crop" and points:
            painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
            for bits in range(8):
                for axis in range(3):
                    other = bits ^ (1 << axis)
                    if bits < other:
                        a, b = self._project([points[f"corner:{bits}"], points[f"corner:{other}"]])
                        painter.drawLine(QPointF(*a), QPointF(*b))
        elif mode == "rotate":
            center = self._project([self.pivot])[0]
            for axis, color in enumerate((Qt.GlobalColor.red, Qt.GlobalColor.green, Qt.GlobalColor.blue)):
                painter.setPen(QPen(color, 3))
                radius = 35 + axis * 14
                painter.drawEllipse(QPointF(*center), radius, radius)
        elif points:
            center = self._project([self.pivot])[0]
            for key, point in points.items():
                endpoint = self._project([point])[0]
                painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
                painter.drawLine(QPointF(*center), QPointF(*endpoint))
        for key, point in points.items():
            screen = self._project([point])[0]
            painter.setPen(QPen(Qt.GlobalColor.yellow if key == self._handle_key else Qt.GlobalColor.cyan, 2))
            painter.drawEllipse(QPointF(*screen), 6, 6)
        painter.end()

    def hit_handle(self, position: QPointF) -> str | None:
        if self.editor is None or self.tool.currentData() == "view":
            return None
        if self.tool.currentData() == "rotate":
            center = self._project([self.pivot])[0]
            radius = float(np.linalg.norm(np.array([position.x(), position.y()]) - center))
            return next((f"ring:{axis}" for axis in range(3)
                         if abs(radius - (35 + axis * 14)) <= 7), None)
        handles = self._gizmo_points()
        if not handles:
            return None
        target = np.array([position.x(), position.y()])
        distances = [(float(np.linalg.norm(self._project([point])[0] - target)), key)
                     for key, point in handles.items()]
        distance, key = min(distances)
        return key if distance <= 12 else None

    def begin_handle(self, key: str, position: QPointF) -> None:
        self._handle_key = key
        self._handle_start = position
        self._handle_values = self._field_values()
        self._handle_moved = False
        self.viewport.setFocus()

    def _world_delta(self, delta: np.ndarray, axes: list[int]) -> np.ndarray:
        step = max(self.distance * .05, .001)
        origin = self._project([self.pivot])[0]
        columns = []
        for axis in axes:
            point = self.pivot.copy()
            point[axis] += step
            columns.append((self._project([point])[0] - origin) / step)
        basis = np.column_stack(columns)
        if np.linalg.norm(basis) < 1e-6:
            return np.zeros(len(axes))
        return np.linalg.pinv(basis) @ delta

    def move_handle(self, position: QPointF) -> None:
        if self._handle_start is None or self._handle_key is None:
            return
        delta = np.array([position.x() - self._handle_start.x(),
                          position.y() - self._handle_start.y()])
        if np.linalg.norm(delta) < 2:
            return
        self._handle_moved = True
        key = self._handle_key
        values = self._handle_values.copy()
        if key.startswith("face:"):
            _, axis, side = key.split(":")
            axis, side = int(axis), int(side)
            index = axis + 3 * side
            values[index] += self._world_delta(delta, [axis])[0]
            other = values[axis + 3 * (1 - side)]
            values[index] = min(values[index], other - .000001) if side == 0 else max(values[index], other + .000001)
        elif key.startswith("corner:"):
            bits = int(key.split(":")[1])
            changes = self._world_delta(delta, [0, 1, 2])
            for axis in range(3):
                index = axis + 3 * ((bits >> axis) & 1)
                other = values[axis + 3 * (1 - ((bits >> axis) & 1))]
                proposed = values[index] + changes[axis]
                values[index] = (min(proposed, other - .000001) if index < 3 else
                                 max(proposed, other + .000001))
        elif key.startswith("axis:") or key.startswith("plane:"):
            axes = [int(value) for value in key.split(":")[1:]]
            for axis, change in zip(axes, self._world_delta(delta, axes)):
                values[axis] += change
        elif key.startswith("ring:"):
            center = self._project([self.pivot])[0]
            start = np.array([self._handle_start.x(), self._handle_start.y()]) - center
            end = np.array([position.x(), position.y()]) - center
            angle = math.degrees(math.atan2(start[0] * end[1] - start[1] * end[0],
                                            float(start @ end)))
            values[3 + int(key.split(":")[1])] += angle
        elif key == "scale":
            center = self._project([self.pivot])[0]
            start = np.linalg.norm(np.array([self._handle_start.x(), self._handle_start.y()]) - center)
            end = np.linalg.norm(np.array([position.x(), position.y()]) - center)
            values[6] = max(.000001, values[6] * end / max(start, 1))
        self._set_values(self.crop_fields if self.tool.currentData() == "crop" else self.transform_fields, values)
        self._preview_pending()

    def end_handle(self) -> None:
        moved = self._handle_moved and any(
            abs(current - initial) > 1e-6
            for current, initial in zip(self._field_values(), self._handle_values)
        )
        self._handle_start = None
        self._handle_key = None
        self._handle_moved = False
        if moved:
            self._apply_pending()
        self.viewport.update()

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
