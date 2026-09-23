"""GS-Studio 主窗口（P1：只读项目浏览、状态、证据、日志与依赖检查）。

布局按 A 方向「深色专业工作台」：顶部项目身份条、左项目树、中状态区、右检查器。
所有内容来自 :mod:`gsdb.services`；界面不判定阶段成功，也不绕过锁或校验。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..paths import find_data_root
from . import theme
from .panels import (
    InspectorPanel,
    LocationDetailPanel,
    ProjectTree,
    RunDetailPanel,
    SceneDetailPanel,
    entry_status_text,
    run_checks_async,
)

MIN_WIDTH_FULL = 1600
WIDTH_SIDEBAR = 284
WIDTH_INSPECTOR = 324
HEIGHT_TOP_BAR = 72


def shorten_path(value: str, head: int = 2, tail: int = 2) -> str:
    """按规范只显示开头与末尾，完整路径放在提示与检查器里。"""
    parts = [item for item in value.replace("/", "\\").split("\\") if item]
    if len(parts) <= head + tail:
        return value
    return "\\".join(parts[:head]) + "\\…\\" + "\\".join(parts[-tail:])


class MainWindow(QMainWindow):
    def __init__(self, data_root: Path):
        super().__init__()
        self._data_root = data_root
        self.setWindowTitle("GS-Studio")
        self.resize(1600, 900)
        self.setMinimumSize(960, 600)

        self.tree = ProjectTree(data_root)
        self.location_panel = LocationDetailPanel()
        self.scene_panel = SceneDetailPanel()
        self.run_panel = RunDetailPanel()
        self.empty = self._empty_state()
        self.center = QStackedWidget()
        for widget in (self.empty, self.location_panel, self.scene_panel, self.run_panel):
            self.center.addWidget(widget)
        self.inspector = InspectorPanel()

        self._sidebar_button = QPushButton("侧栏")
        self._sidebar_button.setCheckable(True)
        self._sidebar_button.setChecked(True)
        self.inspector_button = QPushButton("检查器")
        self.inspector_button.setCheckable(True)
        self.inspector_button.setChecked(True)
        self._sidebar_button.toggled.connect(lambda shown: self._set_column(0, shown))
        self.inspector_button.toggled.connect(lambda shown: self._set_column(2, shown))
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._reload)

        top = QFrame()
        top.setFixedHeight(HEIGHT_TOP_BAR)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(
            theme.SPACE_PANEL, theme.SPACE_GROUP, theme.SPACE_PANEL, theme.SPACE_GROUP
        )
        top_layout.setSpacing(theme.SPACE_IN_GROUP)
        identity = QVBoxLayout()
        identity.setSpacing(0)
        name = QLabel("GS-Studio", objectName="panelTitle")
        root_text = str(data_root)
        root = QLabel(f"数据根：{shorten_path(root_text)}", objectName="caption")
        root.setToolTip(root_text)
        identity.addWidget(name)
        identity.addWidget(root)
        top_layout.addLayout(identity, 1)
        top_layout.addWidget(self._sidebar_button)
        top_layout.addWidget(self.inspector_button)
        top_layout.addWidget(refresh)

        left = QFrame(objectName="panel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(theme.SPACE_GROUP, theme.SPACE_GROUP, theme.SPACE_GROUP, theme.SPACE_GROUP)
        left_layout.setSpacing(theme.SPACE_GROUP)
        left_layout.addWidget(QLabel("项目", objectName="panelTitle"))
        left_layout.addWidget(self.tree)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(left)
        self._splitter.addWidget(self.center)
        self._splitter.addWidget(self.inspector)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([WIDTH_SIDEBAR, 900, WIDTH_INSPECTOR])

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(theme.SPACE_BASE, 0, theme.SPACE_BASE, 0)
        body_layout.setSpacing(theme.SPACE_AREA)
        body_layout.addWidget(top)
        body_layout.addWidget(self._splitter, 1)
        self.setCentralWidget(body)

        self._problem_count = QLabel("问题：0")
        self._problem_count.setObjectName("caption")
        self.statusBar().addPermanentWidget(self._problem_count)
        self.statusBar().showMessage("就绪")

        self.tree.entry_selected.connect(self._on_entry_selected)
        self.tree.entry_problems.connect(self._on_problems)
        self.inspector.check_requested.connect(self._run_checks)
        self._reload()

    def _empty_state(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(theme.SPACE_SECTION, theme.SPACE_SECTION, theme.SPACE_SECTION, theme.SPACE_SECTION)
        layout.setSpacing(theme.SPACE_IN_GROUP)
        title = QLabel("GS-Studio 项目浏览", objectName="pageTitle")
        hint = QLabel(
            "在左侧选择地点、场景、Capture 或 Run，查看阶段状态、证据、日志和问题定位。\n"
            "当前界面是只读视图：状态只来自清单与证据，导出文件或目录的存在都不表示阶段成功。",
            objectName="muted",
        )
        hint.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _set_column(self, index: int, shown: bool) -> None:
        self._splitter.widget(index).setVisible(shown)

    def _reload(self) -> None:
        self.statusBar().showMessage("正在读取项目清单…")
        self.tree.reload()
        self.center.setCurrentWidget(self.empty)
        self.statusBar().showMessage("就绪")

    def _run_checks(self) -> None:
        self.inspector.set_checks_running(True)
        self.statusBar().showMessage("依赖检查执行中（会占用 GPU 锁）…")
        run_checks_async(
            self._data_root,
            self._checks_finished,
            self._checks_failed,
        )

    def _checks_finished(self, views, ok: bool) -> None:
        self.inspector.set_checks_running(False)
        self.inspector.show_checks(ok, views)
        self.statusBar().showMessage("✓ 依赖检查通过" if ok else "! 依赖检查存在未通过项")

    def _checks_failed(self, message: str) -> None:
        self.inspector.set_checks_running(False)
        self.inspector.show_check_error(message)
        self.statusBar().showMessage(f"! 依赖检查失败：{message}")

    def _on_problems(self, problems) -> None:
        self._problem_count.setText(f"问题：{len(problems)}")

    def _on_entry_selected(self, payload: dict) -> None:
        kind = payload.get("kind")
        entry = payload.get("entry")
        location = payload.get("location")
        scene = payload.get("scene")
        self.inspector.show_summary(self._summary_rows(kind, entry))
        self.inspector.show_problems(list(getattr(entry, "problems", []) or []))
        try:
            if kind == "location":
                scenes = self._service(lambda: self._scenes_of(location), "场景列表")
                if scenes is None:
                    return
                self.location_panel.show_location(entry, scenes)
                self.center.setCurrentWidget(self.location_panel)
            elif kind == "scene":
                captures = self._service(lambda: self._captures(location, scene), "Capture 列表")
                runs = self._service(lambda: self._runs(location, scene), "Run 列表")
                if captures is None or runs is None:
                    return
                self.scene_panel.show_scene(location, scene, captures, runs)
                self.center.setCurrentWidget(self.scene_panel)
            elif kind == "run":
                detail = self._service(
                    lambda: self._detail(location, scene, payload["run"]), "Run 状态"
                )
                if detail is None:
                    return
                self.run_panel.show_detail(detail)
                self.center.setCurrentWidget(self.run_panel)
                self.inspector.show_problems(list(detail.problems))
            elif kind == "capture":
                self.center.setCurrentWidget(self.empty)
            self.statusBar().showMessage(f"已选择 {entry.id}" if entry is not None else "就绪")
        except Exception as error:
            self._report_error(error)

    def _service(self, call, label: str):
        self.statusBar().showMessage(f"正在读取{label}…")
        try:
            return call()
        except Exception as error:
            self._report_error(error, label)
            return None

    def _report_error(self, error: Exception, label: str = "状态") -> None:
        from ..services import classify, browse as browse_service

        classified = classify(error)
        message = f"{label}读取失败：{classified.message}"
        self.statusBar().showMessage(f"! {message}")
        self.inspector.show_problems(
            [
                browse_service.Problem(
                    code=classified.code,
                    message=classified.message,
                    path=classified.path,
                )
            ]
        )
        # 失败显示成可定位问题，不用模态对话框打断浏览。
        self._problem_count.setText("问题：1")

    def _scenes_of(self, location: str):
        from ..services import browse

        return browse.list_scenes(self._data_root, location)

    def _captures(self, location: str, scene: str):
        from ..services import browse

        return browse.list_captures(self._data_root, location, scene)

    def _runs(self, location: str, scene: str):
        from ..services import browse

        return browse.list_runs(self._data_root, location, scene)

    def _detail(self, location: str, scene: str, run: str):
        from ..services import run_status

        return run_status.read_run_detail(self._data_root, location, scene, run)

    @staticmethod
    def _summary_rows(kind: str, entry) -> list[tuple[str, str]]:
        if entry is None:
            return []
        rows: list[tuple[str, str]] = [("类型", kind or "—"), ("标识", entry.id)]
        if kind == "location":
            rows += [
                ("名称", entry.display_name or "—"),
                ("状态", entry_status_text(entry)),
                ("拍摄日期", entry.capture_date or "—"),
                ("省市", " / ".join(item for item in (entry.province, entry.city) if item) or "—"),
                ("标签", ", ".join(entry.tags) or "—"),
                ("场景数", str(entry.scene_count)),
            ]
        elif kind == "scene":
            rows += [
                ("名称", entry.display_name or "—"),
                ("状态", entry_status_text(entry)),
                ("质量目标", entry.quality_target or "—"),
                ("米制尺度", "是" if entry.metric_scale else "否"),
            ]
        elif kind == "capture":
            media = entry.media or {}
            rows += [
                ("来源", entry.source_kind or "—"),
                ("投影", entry.projection or "—"),
                ("相机", entry.camera or "—"),
                ("源文件", f"{entry.files_present}/{entry.files_total} 可用"),
                (
                    "媒体",
                    f"{media.get('width')}×{media.get('height')} · {media.get('fps')} fps"
                    if media.get("width")
                    else "—",
                ),
                ("状态", entry_status_text(entry)),
            ]
        elif kind == "run":
            rows += [
                ("状态", entry_status_text(entry)),
                ("活动阶段", entry.active_stage or "—"),
                ("选中数据集", entry.selected_dataset or "—"),
                ("配置哈希", entry.config_hash or "—"),
                ("创建", entry.created_at or "—"),
                ("更新", entry.updated_at or "—"),
            ]
        rows.append(("清单", entry.manifest_path))
        return rows


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("GS-Studio")
    application.setStyleSheet(theme.build_stylesheet())
    try:
        data_root = find_data_root()
    except Exception as error:
        QMessageBox.critical(None, "GS-Studio", f"找不到数据根目录：\n{error}")
        return 1
    window = MainWindow(data_root)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
