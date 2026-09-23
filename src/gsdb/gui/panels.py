"""P1 只读面板：项目树、Run 状态、证据、日志、依赖检查。

所有内容来自 :mod:`gsdb.services`。界面只负责呈现：状态文字取自清单与阶段记录，
证据只显示文件是否存在，问题列表给出可定位的清单/日志/结果路径。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..services import browse, run_status
from ..services.deps import collect_checks
from . import theme

ROLE_PAYLOAD = Qt.ItemDataRole.UserRole
ROLE_PLACEHOLDER = "placeholder"


def make_table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.setWordWrap(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def fill_table(table: QTableWidget, rows: list[list[str]], tooltips: list[list[str]] | None = None) -> None:
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            cell = QTableWidgetItem(value)
            if tooltips:
                cell.setToolTip(tooltips[row_index][column_index])
            table.setItem(row_index, column_index, cell)
    table.resizeColumnsToContents()


def _bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KiB"
    return f"{size / 1024**2:.1f} MiB"


def entry_status_text(entry) -> str:
    text = theme.run_status_text(browse.status_label(entry))
    if entry.problems:
        return f"{text}（{len(entry.problems)} 个问题）"
    return text


class ProjectTree(QTreeWidget):
    """地点 / 场景 / Capture / Run 树。状态来自清单，不按目录猜测。"""

    entry_selected = Signal(object)
    entry_problems = Signal(object)

    def __init__(self, data_root: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self._data_root = data_root
        self.setHeaderLabels(["名称", "状态"])
        self.setColumnWidth(0, 200)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.itemExpanded.connect(self._expand)
        self.itemSelectionChanged.connect(self._select)

    def reload(self) -> None:
        self.clear()
        try:
            locations = browse.list_locations(self._data_root)
        except Exception as error:  # missing or unreadable data root
            problem = QTreeWidgetItem([f"! {error}", ""])
            problem.setData(0, ROLE_PAYLOAD, {"kind": "problem"})
            self.addTopLevelItem(problem)
            self.entry_problems.emit(
                [browse.Problem("unreadable_manifest", str(error), str(self._data_root))]
            )
            return
        problems: list[browse.Problem] = []
        for entry in locations:
            problems.extend(entry.problems)
            item = self._entry_item(
                entry, {"kind": "location", "location": entry.id, "entry": entry}
            )
            item.addChild(self._placeholder())
            self.addTopLevelItem(item)
        self.entry_problems.emit(problems)

    def _entry_item(self, entry, payload: dict[str, Any]) -> QTreeWidgetItem:
        item = QTreeWidgetItem([entry.id, entry_status_text(entry)])
        item.setData(0, ROLE_PAYLOAD, payload)
        item.setToolTip(0, entry.manifest_path)
        item.setToolTip(1, entry_status_text(entry))
        return item

    def _placeholder(self) -> QTreeWidgetItem:
        item = QTreeWidgetItem(["加载中…", ""])
        item.setData(0, ROLE_PAYLOAD, {"kind": ROLE_PLACEHOLDER})
        return item

    def _group(self, title: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([title, ""])
        item.setData(0, ROLE_PAYLOAD, {"kind": "group"})
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        return item

    def _expand(self, item: QTreeWidgetItem) -> None:
        payload = item.data(0, ROLE_PAYLOAD) or {}
        if item.childCount() != 1:
            return
        first = item.child(0)
        if (first.data(0, ROLE_PAYLOAD) or {}).get("kind") != ROLE_PLACEHOLDER:
            return
        item.takeChildren()
        try:
            if payload.get("kind") == "location":
                self._add_scenes(item, payload["location"])
            elif payload.get("kind") == "scene":
                self._add_scene_children(item, payload["location"], payload["scene"])
        except Exception as error:
            item.addChild(QTreeWidgetItem([f"! {error}", ""]))

    def _add_scenes(self, parent: QTreeWidgetItem, location_id: str) -> None:
        for entry in browse.list_scenes(self._data_root, location_id):
            item = self._entry_item(
                entry,
                {
                    "kind": "scene",
                    "location": location_id,
                    "scene": entry.id,
                    "entry": entry,
                },
            )
            item.addChild(self._placeholder())
            parent.addChild(item)

    def _add_scene_children(self, parent: QTreeWidgetItem, location_id: str, scene_id: str) -> None:
        captures = browse.list_captures(self._data_root, location_id, scene_id)
        runs = browse.list_runs(self._data_root, location_id, scene_id)
        capture_group = self._group(f"Capture（{len(captures)}）")
        for entry in captures:
            capture_group.addChild(
                self._entry_item(
                    entry,
                    {
                        "kind": "capture",
                        "location": location_id,
                        "scene": scene_id,
                        "capture": entry.id,
                        "entry": entry,
                    },
                )
            )
        run_group = self._group(f"Run（{len(runs)}）")
        for entry in runs:
            run_group.addChild(
                self._entry_item(
                    entry,
                    {
                        "kind": "run",
                        "location": location_id,
                        "scene": scene_id,
                        "run": entry.id,
                        "entry": entry,
                    },
                )
            )
        parent.addChild(capture_group)
        parent.addChild(run_group)

    def _select(self) -> None:
        selected = self.selectedItems()
        if not selected:
            return
        payload = selected[0].data(0, ROLE_PAYLOAD) or {}
        if payload.get("kind") in {"location", "scene", "capture", "run"}:
            self.entry_selected.emit(payload)


class RunDetailPanel(QWidget):
    """一个 Run 的阶段、配置、证据、日志与训练实验。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = QLabel("尚未选择 Run", objectName="panelTitle")
        self.subtitle = QLabel("", objectName="caption")
        self.stages = make_table(["阶段", "状态", "耗时", "说明", "日志"])
        self.config = make_table(["配置项", "值"])
        self.evidence = make_table(["证据", "存在", "路径"])
        self.logs = make_table(["日志", "大小", "路径"])
        self.experiments = make_table(["分段", "后端", "状态", "输出", "错误"])

        tabs = QTabWidget()
        tabs.addTab(self.config, "配置")
        tabs.addTab(self.evidence, "证据")
        tabs.addTab(self.logs, "日志")
        tabs.addTab(self.experiments, "训练实验")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACE_PANEL, theme.SPACE_PANEL, theme.SPACE_PANEL, theme.SPACE_PANEL)
        layout.setSpacing(theme.SPACE_IN_GROUP)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(QLabel("阶段", objectName="panelTitle"))
        layout.addWidget(self.stages, 2)
        layout.addWidget(tabs, 3)

    def clear(self, message: str) -> None:
        self.title.setText(message)
        self.subtitle.setText("")
        for table in (self.stages, self.config, self.evidence, self.logs, self.experiments):
            table.setRowCount(0)

    def show_detail(self, detail: run_status.RunDetail) -> None:
        entry = detail.entry
        self.title.setText(f"Run {entry.id}")
        selected = entry.selected_dataset or "—"
        self.subtitle.setText(
            f"状态：{theme.run_status_text(detail.status_label)} · 场景：{entry.location_id}/{entry.scene_id} · "
            f"选中数据集：{selected} · 配置哈希：{entry.config_hash or '—'}"
        )
        self.subtitle.setToolTip(self.subtitle.text())

        rows: list[list[str]] = []
        tooltips: list[list[str]] = []
        for stage in detail.stages:
            elapsed = "—" if stage.elapsed_seconds is None else f"{stage.elapsed_seconds:.1f} s"
            log = stage.log_path if stage.log_path and stage.log_present else "—"
            rows.append(
                [stage.name, theme.stage_status_text(stage.status), elapsed, stage.message or "—", log]
            )
            tooltips.append(["", "", "", stage.message or "", stage.log_path or ""])
        fill_table(self.stages, rows, tooltips)

        fill_table(
            self.config,
            [[key, _render_value(value)] for key, value in detail.config_summary.items()],
            [["", _render_value(value)] for key, value in detail.config_summary.items()],
        )
        fill_table(
            self.evidence,
            [
                [item.name, "✓ 存在" if item.present else "! 缺失", item.path]
                for item in detail.evidence
            ],
            [["", "", item.path] for item in detail.evidence],
        )
        fill_table(
            self.logs,
            [[item.name, _bytes(item.size_bytes), item.path] for item in detail.logs],
            [["", "", item.path] for item in detail.logs],
        )
        fill_table(
            self.experiments,
            [
                [
                    item.segment or "—",
                    item.backend or "—",
                    theme.run_status_text(item.status) if item.status else "—",
                    item.output or "—",
                    item.error or "—",
                ]
                for item in detail.experiments
            ],
            [["", "", "", item.output or "", item.error or ""] for item in detail.experiments],
        )


def _render_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, ensure_ascii=False)
    return str(value)


class LocationDetailPanel(QWidget):
    """一个地点下的场景列表。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = QLabel("尚未选择地点", objectName="panelTitle")
        self.scenes = make_table(["场景", "名称", "状态", "质量目标", "米制尺度"])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACE_PANEL, theme.SPACE_PANEL, theme.SPACE_PANEL, theme.SPACE_PANEL)
        layout.setSpacing(theme.SPACE_IN_GROUP)
        layout.addWidget(self.title)
        layout.addWidget(QLabel("场景", objectName="panelTitle"))
        layout.addWidget(self.scenes)

    def clear(self, message: str) -> None:
        self.title.setText(message)
        self.scenes.setRowCount(0)

    def show_location(self, entry, scenes: list[browse.SceneEntry]) -> None:
        self.title.setText(f"{entry.id}（{entry.display_name or '未命名'}）")
        rows = []
        tooltips = []
        for item in scenes:
            rows.append(
                [
                    item.id,
                    item.display_name or "—",
                    entry_status_text(item),
                    item.quality_target or "—",
                    "是" if item.metric_scale else "否",
                ]
            )
            tooltips.append([item.manifest_path, "", "", "", ""])
        fill_table(self.scenes, rows, tooltips)


class SceneDetailPanel(QWidget):
    """一个场景下的 Capture 与 Run 列表。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = QLabel("尚未选择场景", objectName="panelTitle")
        self.captures = make_table(["Capture", "来源", "源文件", "媒体", "时长", "状态"])
        self.runs = make_table(["Run", "状态", "阶段", "选中数据集", "更新时间"])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACE_PANEL, theme.SPACE_PANEL, theme.SPACE_PANEL, theme.SPACE_PANEL)
        layout.setSpacing(theme.SPACE_IN_GROUP)
        layout.addWidget(self.title)
        layout.addWidget(QLabel("Capture", objectName="panelTitle"))
        layout.addWidget(self.captures, 1)
        layout.addWidget(QLabel("Run", objectName="panelTitle"))
        layout.addWidget(self.runs, 2)

    def clear(self, message: str) -> None:
        self.title.setText(message)
        self.captures.setRowCount(0)
        self.runs.setRowCount(0)

    def show_scene(
        self,
        location_id: str,
        scene_id: str,
        captures: list[browse.CaptureEntry],
        runs: list[browse.RunEntry],
    ) -> None:
        self.title.setText(f"{location_id} / {scene_id}")
        rows = []
        tooltips = []
        for item in captures:
            media = item.media or {}
            size = f"{media.get('width')}×{media.get('height')}" if media.get("width") else "—"
            duration = f"{media.get('duration_seconds')} s" if media.get("duration_seconds") else "—"
            rows.append(
                [
                    item.id,
                    item.source_kind or "—",
                    f"{item.files_present}/{item.files_total}",
                    size,
                    duration,
                    entry_status_text(item),
                ]
            )
            tooltips.append([item.manifest_path, "", "", "", "", ""])
        fill_table(self.captures, rows, tooltips)

        rows = []
        tooltips = []
        for item in runs:
            stages = ", ".join(f"{name}={theme.stage_status_text(value)}" for name, value in item.stages.items())
            rows.append(
                [
                    item.id,
                    entry_status_text(item),
                    stages or "—",
                    item.selected_dataset or "—",
                    item.updated_at or "—",
                ]
            )
            tooltips.append([item.manifest_path, "", stages, item.selected_dataset or "", ""])
        fill_table(self.runs, rows, tooltips)


class InspectorPanel(QWidget):
    """选中对象摘要、问题定位与依赖检查。"""

    check_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.summary = make_table(["项", "值"])
        self.problems = QListWidget()
        self.checks = make_table(["检查", "结果", "说明"])
        self.check_state = QLabel("依赖检查未运行。", objectName="caption")
        self.check_button = QPushButton("运行依赖检查")
        self.check_button.clicked.connect(self.check_requested.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACE_PANEL, theme.SPACE_PANEL, theme.SPACE_PANEL, theme.SPACE_PANEL)
        layout.setSpacing(theme.SPACE_IN_GROUP)
        layout.addWidget(QLabel("选中对象", objectName="panelTitle"))
        layout.addWidget(self.summary, 2)
        layout.addWidget(QLabel("问题", objectName="panelTitle"))
        layout.addWidget(self.problems, 2)
        layout.addWidget(QLabel("依赖检查", objectName="panelTitle"))
        layout.addWidget(self.check_state)
        layout.addWidget(self.check_button)
        layout.addWidget(self.checks, 2)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def show_summary(self, rows: list[tuple[str, str]], tooltips: list[str] | None = None) -> None:
        fill_table(self.summary, [[key, value] for key, value in rows], [["", t] for t in (tooltips or [""] * len(rows))])

    def show_problems(self, problems: list[browse.Problem]) -> None:
        self.problems.clear()
        for problem in problems:
            item = QListWidgetItem(f"{problem.code}：{problem.message}")
            item.setToolTip(problem.path or "")
            item.setForeground(_color(theme.WARNING if problem.code != "missing_source" else theme.ERROR))
            self.problems.addItem(item)
        if not problems:
            self.problems.addItem(QListWidgetItem("没有发现问题。"))

    def set_checks_running(self, running: bool) -> None:
        self.check_button.setEnabled(not running)
        self.check_state.setText("依赖检查执行中…" if running else "依赖检查未运行。")

    def show_checks(self, ok: bool, views) -> None:
        self.check_state.setText("✓ 依赖检查通过" if ok else "! 依赖检查存在未通过项")
        self.check_state.setObjectName("caption" if ok else "warning")
        self.check_state.style().unpolish(self.check_state)
        self.check_state.style().polish(self.check_state)
        fill_table(
            self.checks,
            [[view.name, view.label, view.detail] for view in views],
            [["", "", view.detail] for view in views],
        )

    def show_check_error(self, message: str) -> None:
        self.check_state.setText(f"! 依赖检查失败：{message}")
        self.check_state.setObjectName("error")
        self.check_state.style().unpolish(self.check_state)
        self.check_state.style().polish(self.check_state)


def _color(value: str):
    from PySide6.QtGui import QColor

    return QColor(value)


class _ChecksSignals(QObject):
    finished = Signal(object, bool)
    failed = Signal(str)


class ChecksWorker(QRunnable):
    """在工作线程运行 doctor，避免占用 GPU 锁时冻结界面。"""

    def __init__(self, data_root: Path, backend: str = "postshot"):
        super().__init__()
        self.signals = _ChecksSignals()
        self._data_root = data_root
        self._backend = backend

    def run(self) -> None:
        try:
            ok, views = collect_checks(self._data_root, backend=self._backend)
        except Exception as error:
            self.signals.failed.emit(str(error))
            return
        self.signals.finished.emit(views, ok)


def run_checks_async(
    data_root: Path,
    on_finished: Callable[[object, bool], None],
    on_failed: Callable[[str], None],
    backend: str = "postshot",
) -> ChecksWorker:
    worker = ChecksWorker(data_root, backend=backend)
    worker.signals.finished.connect(on_finished)
    worker.signals.failed.connect(on_failed)
    QThreadPool.globalInstance().start(worker)
    return worker
