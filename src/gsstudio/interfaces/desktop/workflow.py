"""Native dialogs and the isolated operation client for GS-Studio."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QSpinBox, QTextEdit, QVBoxLayout,
)

from gsstudio.pipeline.masks.review import MaskReviewDataset


class OperationClient(QObject):
    event = Signal(object)
    finished = Signal(int)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._buffer = b""
        self._saw_error = False
        self._saw_result = False
        self._stderr_tail = b""
        self._control: tempfile.TemporaryDirectory | None = None
        self._stop_path: Path | None = None

    @property
    def busy(self) -> bool:
        return self._process is not None

    def start(self, request: dict) -> None:
        if self.busy:
            raise RuntimeError("Another GS-Studio operation is still running")
        self._control = tempfile.TemporaryDirectory(prefix="gs-studio-")
        self._stop_path = Path(self._control.name) / "stop"
        request = {**request, "stop_file": str(self._stop_path)}
        process = QProcess(self)
        self._process = process
        self._buffer = b""
        self._saw_error = False
        self._saw_result = False
        self._stderr_tail = b""
        process.readyReadStandardOutput.connect(self._read_output)
        process.readyReadStandardError.connect(self._read_error)
        process.finished.connect(self._finished)
        process.start(sys.executable, ["-m", "gsstudio.interfaces.worker.main"])
        if not process.waitForStarted(3000):
            self._cleanup()
            raise RuntimeError("Could not start the GS-Studio operation worker")
        process.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        process.closeWriteChannel()

    def request_stop(self) -> None:
        if self._stop_path and self.busy:
            self._stop_path.write_text("stop\n", encoding="utf-8")

    def _read_output(self) -> None:
        if self._process is None:
            return
        self._buffer += bytes(self._process.readAllStandardOutput())
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.event.emit({"kind": "error", "message": "Worker emitted invalid structured output"})
                self._saw_error = True
                continue
            if event.get("kind") == "error":
                self._saw_error = True
            elif event.get("kind") == "result":
                self._saw_result = True
            self.event.emit(event)

    def _read_error(self) -> None:
        if self._process is not None:
            self._stderr_tail = (self._stderr_tail +
                                 bytes(self._process.readAllStandardError()))[-2000:]

    def _finished(self, exit_code: int, _status) -> None:
        self._read_output()
        self._read_error()
        failed = bool(exit_code or _status == QProcess.ExitStatus.CrashExit or
                      not self._saw_result)
        if failed and not self._saw_error:
            message = self._stderr_tail.decode("utf-8", "replace").strip()
            self.event.emit({"kind": "error", "message": message or
                             "工作进程中断且未返回结构化结果；请检查 Run 清单与日志。"})
        self._cleanup()
        self.finished.emit(1 if failed else 0)

    def _cleanup(self) -> None:
        if self._process is not None:
            self._process.deleteLater()
            self._process = None
        if self._control is not None:
            self._control.cleanup()
            self._control = None
        self._stop_path = None


class CaptureDialog(QDialog):
    SOURCE_TYPES = (
        ("全景视频", "equirect_video"),
        ("全景图片序列", "equirect_sequence"),
        ("Insta360 INSV", "insta360_insv"),
        ("透视视频", "perspective_video"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建 Capture")
        self.setMinimumWidth(540)
        self.sources: list[str] = []
        self.capture_id = QLineEdit()
        self.capture_id.setPlaceholderText("例如 capture-001")
        self.source_type = QComboBox()
        for label, value in self.SOURCE_TYPES:
            self.source_type.addItem(label, value)
        self.source_text = QLineEdit()
        self.source_text.setReadOnly(True)
        browse = QPushButton("选择…")
        browse.clicked.connect(self._browse)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_text, 1)
        source_row.addWidget(browse)
        self.camera_model = QLineEdit("unknown")
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.0, 120.0)
        self.fps.setDecimals(3)
        self.fps.setSpecialValueText("自动 / 不适用")
        self.start_seconds = QDoubleSpinBox()
        self.start_seconds.setRange(0.0, 1_000_000.0)
        self.end_seconds = QDoubleSpinBox()
        self.end_seconds.setRange(0.0, 1_000_000.0)
        self.end_seconds.setSpecialValueText("素材结束")
        self.width = QSpinBox()
        self.width.setRange(0, 32768)
        self.width.setSpecialValueText("原始尺寸")
        self.height = QSpinBox()
        self.height.setRange(0, 16384)
        self.height.setSpecialValueText("原始尺寸")
        self.fov = QDoubleSpinBox()
        self.fov.setRange(1.0, 179.0)
        self.fov.setValue(84.0)
        form = QFormLayout()
        form.addRow("Capture ID", self.capture_id)
        form.addRow("素材类型", self.source_type)
        form.addRow("原片", source_row)
        form.addRow("相机型号", self.camera_model)
        form.addRow("图片序列 FPS", self.fps)
        form.addRow("起始秒", self.start_seconds)
        form.addRow("结束秒", self.end_seconds)
        form.addRow("全景输出宽", self.width)
        form.addRow("全景输出高", self.height)
        form.addRow("透视水平 FOV", self.fov)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        kind = self.source_type.currentData()
        if kind == "equirect_sequence":
            path = QFileDialog.getExistingDirectory(self, "选择图片序列目录")
            self.sources = [path] if path else []
        elif kind == "insta360_insv":
            paths, _ = QFileDialog.getOpenFileNames(self, "选择一或两个 INSV 文件", "", "INSV (*.insv)")
            self.sources = paths
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频")
            self.sources = [path] if path else []
        self.source_text.setText("; ".join(self.sources))

    def _accept(self) -> None:
        if not self.capture_id.text().strip() or not self.sources:
            QMessageBox.warning(self, "缺少输入", "请填写 Capture ID 并选择素材。")
            return
        if self.source_type.currentData() == "equirect_sequence" and self.fps.value() == 0:
            QMessageBox.warning(self, "缺少 FPS", "图片序列需要明确的 FPS。")
            return
        if bool(self.width.value()) != bool(self.height.value()):
            QMessageBox.warning(self, "尺寸不完整", "全景宽和高需要同时填写。")
            return
        self.accept()

    def request(self) -> dict:
        return {
            "capture_id": self.capture_id.text().strip(),
            "source_type": self.source_type.currentData(),
            "sources": self.sources,
            "camera_model": self.camera_model.text().strip() or "unknown",
            "fps": self.fps.value() or None,
            "start_seconds": self.start_seconds.value(),
            "end_seconds": self.end_seconds.value() or None,
            "output_width": self.width.value() or None,
            "output_height": self.height.value() or None,
            "horizontal_fov": self.fov.value(),
        }


class RunDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建 Run")
        self.candidate = QDoubleSpinBox()
        self.candidate.setRange(0.1, 120.0)
        self.candidate.setValue(1.0)
        self.selected = QSpinBox()
        self.selected.setRange(1, 120)
        self.selected.setValue(1)
        self.primary = QSpinBox()
        self.primary.setRange(1, 120)
        self.primary.setValue(1)
        form = QFormLayout()
        form.addRow("候选 FPS", self.candidate)
        form.addRow("每秒保留", self.selected)
        form.addRow("初次对齐每秒", self.primary)
        form.addRow("遮罩审核", QLabel("启用；审核后继续"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        if self.primary.value() > self.selected.value() or self.selected.value() > self.candidate.value():
            QMessageBox.warning(self, "抽帧参数", "需要满足 初次对齐 ≤ 每秒保留 ≤ 候选 FPS。")
            return
        self.accept()

    def request(self) -> dict:
        return {"candidate_fps": self.candidate.value(),
                "selected_per_second": self.selected.value(),
                "primary_per_second": self.primary.value()}


class MaskReviewDialog(QDialog):
    """Review the exact image/mask identity used by mask finalization."""

    def __init__(self, dataset: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"遮罩审核 · {dataset.name}")
        self.resize(980, 680)
        self.dataset = MaskReviewDataset(dataset)
        self.items = QListWidget()
        for item in self.dataset.items:
            status = self.dataset.review_for(item.image_id)["status"]
            self.items.addItem(f"{item.image_id:05d}  {item.source_image}  {item.masked_fraction:.2%}  [{status}]")
        self.preview = QLabel("选择图片")
        self.preview.setMinimumSize(340, 320)
        self.preview.setScaledContents(False)
        self.status = QComboBox()
        for value in ("unreviewed", "ok", "false_positive", "false_negative", "exclude"):
            self.status.addItem(value, value)
        self.note = QTextEdit()
        self.note.setMaximumHeight(90)
        save = QPushButton("保存当前审核")
        save.clicked.connect(self._save_review)
        done = QPushButton("关闭并返回 Run")
        done.clicked.connect(self.accept)
        right = QVBoxLayout()
        right.addWidget(self.preview, 1)
        right.addWidget(QLabel("审核状态"))
        right.addWidget(self.status)
        right.addWidget(QLabel("备注"))
        right.addWidget(self.note)
        right.addWidget(save)
        right.addWidget(done)
        body = QHBoxLayout(self)
        body.addWidget(self.items, 2)
        body.addLayout(right, 3)
        self.items.currentRowChanged.connect(self._select)
        if self.dataset.items:
            self.items.setCurrentRow(0)

    def _select(self, row: int) -> None:
        if row < 0:
            return
        item = self.dataset.items[row]
        pixmap = QPixmap()
        pixmap.loadFromData(self.dataset.thumbnail(item.image_id))
        self.preview.setPixmap(pixmap)
        review = self.dataset.review_for(item.image_id)
        index = self.status.findData(review["status"])
        self.status.setCurrentIndex(max(index, 0))
        self.note.setPlainText(review["note"])

    def _save_review(self) -> None:
        row = self.items.currentRow()
        if row < 0:
            return
        item = self.dataset.items[row]
        try:
            self.dataset.update_review(item.image_id, self.status.currentData(),
                                       self.note.toPlainText())
        except Exception as error:
            QMessageBox.warning(self, "审核未保存", str(error))
            return
        self.items.item(row).setText(
            f"{item.image_id:05d}  {item.source_image}  {item.masked_fraction:.2%}  [{self.status.currentData()}]"
        )
