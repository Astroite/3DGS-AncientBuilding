"""GS-Studio 主窗口：项目操作、阶段审核、训练和模型工作台。

布局按 A 方向「深色专业工作台」：顶部项目身份条、左项目树、中状态区、右检查器。
业务操作由 :mod:`gsstudio.application` 执行；界面不自行判定阶段成功。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gsstudio.infrastructure.paths import find_data_root
from gsstudio.interfaces.desktop import theme
from gsstudio.interfaces.desktop.panels import (
    InspectorPanel,
    LocationDetailPanel,
    ProjectTree,
    RunDetailPanel,
    SceneDetailPanel,
    entry_status_text,
    run_checks_async,
)
from gsstudio.interfaces.desktop.workflow import CaptureDialog, MaskReviewDialog, OperationClient, RunDialog
from gsstudio.interfaces.desktop.model import ModelWorkbench

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
        self._selection: dict = {}
        self._last_request: dict | None = None
        self._last_result: dict | None = None
        self._pending_followup: dict | None = None
        self.operations = OperationClient(self)
        self.operations.event.connect(self._operation_event)
        self.operations.finished.connect(self._operation_finished)
        self._live_log_path: Path | None = None
        self._live_preview_output: Path | None = None
        self._live_log_position = 0
        self._live_log_timer = QTimer(self)
        self._live_log_timer.setInterval(1000)
        self._live_log_timer.timeout.connect(self._poll_live_log)
        self.setWindowTitle("GS-Studio")
        self.resize(1600, 900)
        self.setMinimumSize(960, 600)

        self.tree = ProjectTree(data_root)
        self.location_panel = LocationDetailPanel()
        self.scene_panel = SceneDetailPanel()
        self.run_panel = RunDetailPanel()
        self.model_panel = ModelWorkbench()
        self.empty = self._empty_state()
        self.center = QStackedWidget()
        for widget in (self.empty, self.location_panel, self.scene_panel, self.run_panel, self.model_panel):
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
        self.actions_button = QPushButton("操作")
        menu = QMenu(self.actions_button)
        self.actions_button.setMenu(menu)
        self._actions = {}
        for key, label, handler in (
            ("create_location", "新建地点", self._create_location),
            ("create_scene", "新建场景", self._create_scene),
            ("create_capture", "新建 Capture", self._create_capture),
            ("probe_capture", "检查素材", lambda: self._context_action("probe_capture")),
            ("ingest", "更新素材探测", lambda: self._context_action("ingest")),
            ("relink_capture", "重新定位原片", self._relink_capture),
            ("create_run", "新建并启动 Run", self._create_run),
            ("resume_run", "继续到下一审核点", lambda: self._context_action("resume_run")),
            ("review_masks", "审核当前遮罩", self._review_masks),
            ("finalize_masks", "固定当前遮罩", self._finalize_masks),
            ("qa_accept", "接受 QA", lambda: self._review_qa(True)),
            ("qa_reject", "拒绝 QA", lambda: self._review_qa(False)),
            ("train_segment", "选择分段并训练", self._train_segment),
            ("resume_training", "检查并恢复训练", self._resume_training),
            ("verified_models", "打开训练模型", lambda: self._context_action("verified_models")),
            ("cleanup", "预览清理", lambda: self._context_action("cleanup")),
        ):
            action = menu.addAction(label)
            action.triggered.connect(handler)
            self._actions[key] = action
        self.stop_button = QPushButton("阶段后停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.operations.request_stop)

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
        top_layout.addWidget(self.actions_button)
        top_layout.addWidget(self.stop_button)
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
        self._update_actions()
        self._reload()

    def _empty_state(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(theme.SPACE_SECTION, theme.SPACE_SECTION, theme.SPACE_SECTION, theme.SPACE_SECTION)
        layout.setSpacing(theme.SPACE_IN_GROUP)
        title = QLabel("GS-Studio 项目工作台", objectName="pageTitle")
        hint = QLabel(
            "在左侧选择地点、场景、Capture 或 Run，通过“操作”创建素材、推进流水线、审核并训练。\n"
            "阶段状态来自 Run 清单；导出文件或目录的存在不表示阶段成功。",
            objectName="muted",
        )
        hint.setWordWrap(True)
        self.empty_title = title
        self.empty_hint = hint
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _set_column(self, index: int, shown: bool) -> None:
        self._splitter.widget(index).setVisible(shown)

    def _reload(self) -> None:
        context = dict(self._selection)
        self.statusBar().showMessage("正在读取项目清单…")
        self.tree.reload()
        self.center.setCurrentWidget(self.empty)
        if context:
            self.tree.select_context(
                context.get("location"), context.get("scene"),
                context.get("capture"), context.get("run"),
            )
        self._update_actions()
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
        self._selection = payload
        self._update_actions()
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
                self.empty_title.setText(f"Capture {entry.id}")
                self.empty_hint.setText(
                    "使用“操作”检查或重新定位素材；创建 Run 后会自动推进到遮罩审核。"
                )
                self.center.setCurrentWidget(self.empty)
            self.statusBar().showMessage(f"已选择 {entry.id}" if entry is not None else "就绪")
        except Exception as error:
            self._report_error(error)

    def _update_actions(self) -> None:
        kind = self._selection.get("kind")
        entry = self._selection.get("entry")
        run_status_value = getattr(entry, "status", None)
        busy = self.operations.busy
        allowed = {
            "create_location": True,
            "create_scene": kind == "location",
            "create_capture": kind == "scene",
            "probe_capture": kind == "capture",
            "ingest": kind == "capture",
            "relink_capture": kind == "capture",
            "create_run": kind == "capture",
            "resume_run": kind == "run",
            "review_masks": kind == "run" and run_status_value == "waiting_review",
            "finalize_masks": kind == "run" and run_status_value == "waiting_review",
            "qa_accept": kind == "run" and run_status_value == "needs_review",
            "qa_reject": kind == "run" and run_status_value == "needs_review",
            "train_segment": kind == "run" and run_status_value == "accepted",
            "resume_training": kind == "run" and run_status_value == "accepted",
            "verified_models": kind == "run",
            "cleanup": kind == "run",
        }
        for key, action in self._actions.items():
            action.setEnabled(not busy and allowed[key])
        self.stop_button.setEnabled(busy)

    def _start_operation(self, request: dict) -> None:
        request = {"root": str(self._data_root), **request}
        self._last_request = request
        self._last_result = None
        self._had_worker_error = False
        try:
            self.operations.start(request)
        except Exception as error:
            QMessageBox.warning(self, "操作未启动", str(error))
            return
        self.statusBar().showMessage(f"正在执行 {request['action']}…")
        self._update_actions()

    def _context_action(self, action: str, **extra) -> None:
        payload = self._selection
        if not payload:
            return
        request = {
            "action": action,
            "location_id": payload.get("location"),
            "scene_id": payload.get("scene"),
            "capture_id": payload.get("capture"),
            "run_id": payload.get("run"),
            **extra,
        }
        self._start_operation(request)

    def _create_location(self) -> None:
        location_id, ok = QInputDialog.getText(self, "新建地点", "地点 ID（小写字母、数字、短横线）")
        if not ok or not location_id.strip():
            return
        name, ok = QInputDialog.getText(self, "新建地点", "显示名称")
        if ok and name.strip():
            self._start_operation({"action": "create_location", "location_id": location_id.strip(),
                                   "name": name.strip()})

    def _create_scene(self) -> None:
        scene_id, ok = QInputDialog.getText(self, "新建场景", "场景 ID（小写字母、数字、短横线）")
        if not ok or not scene_id.strip():
            return
        name, ok = QInputDialog.getText(self, "新建场景", "显示名称")
        if ok and name.strip():
            self._context_action("create_scene", scene_id=scene_id.strip(), name=name.strip())

    def _create_capture(self) -> None:
        dialog = CaptureDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._context_action("create_capture", **dialog.request())

    def _relink_capture(self) -> None:
        kind = self._selection["entry"].source_kind
        if kind == "equirect_sequence":
            path = QFileDialog.getExistingDirectory(self, "重新定位图片序列")
            paths = [path] if path else []
        elif kind == "insta360_insv":
            paths, _ = QFileDialog.getOpenFileNames(self, "重新定位 INSV 文件", "", "INSV (*.insv)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "重新定位原片")
            paths = [path] if path else []
        if paths:
            self._context_action("relink_capture", sources=paths)

    def _create_run(self) -> None:
        dialog = RunDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._context_action("create_run", **dialog.request())

    def _mask_attempts(self) -> list[str]:
        payload = self._selection
        work = self._data_root / payload["location"] / payload["scene"] / payload["run"]
        return [label for label in ("primary", "repair")
                if (work / f"reconstruction-{label}" / "images").is_dir()]

    def _choose_mask_attempt(self) -> str | None:
        attempts = self._mask_attempts()
        if not attempts:
            QMessageBox.information(self, "遮罩审核", "当前 Run 尚无可审核的遮罩数据。")
            return None
        preferred = "repair" if "repair" in attempts else attempts[0]
        choice, ok = QInputDialog.getItem(self, "遮罩 attempt", "选择要审核的 attempt",
                                          attempts, attempts.index(preferred), False)
        return choice if ok else None

    def _review_masks(self) -> None:
        attempt = self._choose_mask_attempt()
        if not attempt:
            return
        payload = self._selection
        dataset = (self._data_root / payload["location"] / payload["scene"] /
                   payload["run"] / f"reconstruction-{attempt}")
        try:
            MaskReviewDialog(dataset, self).exec()
        except Exception as error:
            QMessageBox.warning(self, "遮罩审核未打开", str(error))

    def _finalize_masks(self) -> None:
        attempt = self._choose_mask_attempt()
        if attempt:
            self._context_action("finalize_masks", attempt=attempt)

    def _review_qa(self, accepted: bool) -> None:
        notes, ok = QInputDialog.getMultiLineText(
            self, "接受 QA" if accepted else "拒绝 QA", "人工结论与依据"
        )
        if ok and notes.strip():
            self._context_action("review_qa", accepted=accepted, notes=notes.strip())

    def _train_segment(self) -> None:
        from gsstudio.application.operations import segment_choices

        payload = self._selection
        try:
            choices = segment_choices(self._data_root, payload["location"],
                                      payload["scene"], payload["run"])
        except Exception as error:
            QMessageBox.warning(self, "无法选择分段", str(error))
            return
        segments = [item["id"] for item in choices["segments"]]
        if not segments:
            QMessageBox.information(self, "分段 QA", "没有通过 QA 的可训练分段。")
            return
        segment, ok = QInputDialog.getItem(
            self, "选择训练分段",
            f"可训练：{choices['training_status']}；全路线覆盖：{choices['coverage_status']}",
            segments, 0, False,
        )
        if not ok:
            return
        backend, ok = QInputDialog.getItem(self, "训练后端", "选择后端",
                                           ["gsplat", "postshot"], 0, False)
        if ok:
            self._context_action("train_segment", segment=segment, backend=backend)

    def _resume_training(self) -> None:
        from gsstudio.infrastructure.persistence.run_repository import load_run

        payload = self._selection
        try:
            scene = self._data_root / payload["location"] / payload["scene"]
            run = load_run(scene, payload["run"])
            candidates = [item for item in run.metrics.get("training_experiments", [])
                          if item.get("backend") == "gsplat" and
                          item.get("status") in {"failed", "preparing"}]
        except Exception as error:
            QMessageBox.warning(self, "训练恢复", str(error))
            return
        if not candidates:
            QMessageBox.information(self, "训练恢复", "没有可检查的 gsplat 失败或中断实验。")
            return
        labels = [f"{item['segment']} · {item['output']} · {item['status']}"
                  for item in candidates]
        chosen, ok = QInputDialog.getItem(self, "训练恢复", "选择实验", labels,
                                          len(labels) - 1, False)
        if ok:
            item = candidates[labels.index(chosen)]
            self._context_action("train_segment", segment=item["segment"],
                                 backend="gsplat", output=item["output"], resume=True)

    def _operation_event(self, event: dict) -> None:
        kind = event.get("kind", "event")
        message = event.get("message", "")
        action = event.get("action", "")
        line = f"[{kind}] {action}: {message}"
        self.run_panel.append_event(line)
        self.statusBar().showMessage(message or action)
        log_path = (event.get("detail") or {}).get("log_path")
        if log_path:
            self._live_log_path = Path(log_path)
            output = (event.get("detail") or {}).get("output")
            self._live_preview_output = Path(output) if output else None
            self._live_log_position = 0
            self._live_log_timer.start()
        if kind == "result":
            self._last_result = (event.get("detail") or {}).get("result")
        elif kind == "error":
            self._had_worker_error = True
            from gsstudio.application.browse import Problem

            detail = event.get("detail") or {}
            self.inspector.show_problems([
                Problem(detail.get("code", "error"), message, detail.get("path"))
            ])
            QMessageBox.warning(self, "操作失败", message)

    def _operation_finished(self, exit_code: int) -> None:
        self._poll_live_log()
        self._live_log_timer.stop()
        self._live_log_path = None
        self._live_preview_output = None
        request = self._last_request or {}
        result = self._last_result or {}
        action = request.get("action")
        if exit_code and not self._had_worker_error:
            QMessageBox.warning(self, "操作中断", "工作进程意外退出；请检查 Run 清单与日志后继续。")
        if not exit_code:
            if action == "create_location":
                self._selection = {"location": result.get("location_id")}
            elif action == "create_scene":
                self._selection = {"location": request.get("location_id"),
                                   "scene": result.get("scene_id")}
            elif action == "create_capture":
                self._selection = {"location": request.get("location_id"),
                                   "scene": request.get("scene_id"),
                                   "capture": result.get("capture_id")}
            elif action == "create_run":
                self._selection = {"location": request.get("location_id"),
                                   "scene": request.get("scene_id"),
                                   "run": result.get("run_id")}
            elif action == "verified_models":
                models = result.get("models", [])
                if models:
                    labels = [f"{item['segment']} · {item['backend']} · {item['path']}"
                              for item in models]
                    label, chosen = QInputDialog.getItem(self, "打开模型", "已验证的训练成果",
                                                          labels, len(labels) - 1, False)
                    if chosen:
                        model = models[labels.index(label)]
                        self.model_panel.open_model(Path(model["path"]), model["sha256"])
                        self.center.setCurrentWidget(self.model_panel)
                else:
                    QMessageBox.information(self, "模型", "没有哈希校验通过的训练 PLY。")
            elif action == "probe_capture":
                QMessageBox.information(
                    self, "素材探测", json.dumps(result, ensure_ascii=False, indent=2)[:5000]
                )
            elif action == "cleanup" and not request.get("apply"):
                summary = (
                    f"计划文件：{result.get('planned_files', 0)}\n"
                    f"预计可回收：{result.get('reclaimable_gib', 0)} GiB\n"
                    f"保留原因：{'; '.join(result.get('retained', []))[:800] or '无'}\n\n"
                    "执行仍会重新核验 Run 锁、消费者和每个文件的身份。"
                )
                if result.get("planned_files", 0) and QMessageBox.question(
                    self, "清理预览", summary + "\n\n执行清理？"
                ) == QMessageBox.StandardButton.Yes:
                    self._pending_followup = {**request, "apply": True}
                elif not result.get("planned_files", 0):
                    QMessageBox.information(self, "清理预览", summary)
            elif action == "cleanup" and request.get("apply"):
                QMessageBox.information(self, "清理结果", f"状态：{result.get('status')}；文件：{result.get('planned_files', 0)}")
            if action == "finalize_masks" or (action == "review_qa" and result.get("status") == "accepted"):
                self._pending_followup = {"action": "resume_run",
                                          "location_id": request.get("location_id"),
                                          "scene_id": request.get("scene_id"),
                                          "run_id": request.get("run_id")}
            if action == "train_segment" and result.get("status") == "succeeded":
                self._pending_followup = {"action": "verified_models",
                                          "location_id": request.get("location_id"),
                                          "scene_id": request.get("scene_id"),
                                          "run_id": request.get("run_id")}
        if action != "verified_models":
            self._reload()
        self._update_actions()
        if self._pending_followup:
            followup = self._pending_followup
            self._pending_followup = None
            QTimer.singleShot(0, lambda: self._start_operation(followup))

    def _poll_live_log(self) -> None:
        if self._live_preview_output is not None:
            self.run_panel.show_training_preview(self._live_preview_output)
        path = self._live_log_path
        if path is None or not path.is_file():
            return
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(self._live_log_position)
            content = stream.read()
            self._live_log_position = stream.tell()
        for line in content.splitlines()[-40:]:
            if not line.strip():
                continue
            self.run_panel.append_event(line[:500])
            try:
                progress = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(progress, dict) and "step" in progress:
                self.statusBar().showMessage(
                    f"训练 step {progress['step']} · loss {progress.get('loss', '—')} · "
                    f"Gaussian {progress.get('splats', '—')}"
                )

    def closeEvent(self, event) -> None:
        if self.operations.busy:
            QMessageBox.information(self, "任务仍在运行", "已请求在当前阶段结束后停止；完成前请保持窗口打开。")
            self.operations.request_stop()
            event.ignore()
            return
        self.model_panel.shutdown()
        super().closeEvent(event)

    def _service(self, call, label: str):
        self.statusBar().showMessage(f"正在读取{label}…")
        try:
            return call()
        except Exception as error:
            self._report_error(error, label)
            return None

    def _report_error(self, error: Exception, label: str = "状态") -> None:
        from gsstudio.application import classify, browse as browse_service

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
        from gsstudio.application import browse

        return browse.list_scenes(self._data_root, location)

    def _captures(self, location: str, scene: str):
        from gsstudio.application import browse

        return browse.list_captures(self._data_root, location, scene)

    def _runs(self, location: str, scene: str):
        from gsstudio.application import browse

        return browse.list_runs(self._data_root, location, scene)

    def _detail(self, location: str, scene: str, run: str):
        from gsstudio.application import run_status

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
