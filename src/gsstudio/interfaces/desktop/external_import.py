"""Qt inspection and training entry for independent external COLMAP sources."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
    QVBoxLayout,
)

from gsstudio.interfaces.desktop.compare import TrainingCompare
from gsstudio.interfaces.desktop.workflow import OperationClient


class ExternalImportDialog(QDialog):
    """One visible path from inspection to an independently identified PLY."""

    model_requested = Signal(str, str)

    def __init__(self, data_root: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle('外部 COLMAP 导入与训练')
        self.resize(1120, 820)
        self.data_root = Path(data_root)
        self.client = OperationClient(self)
        self.client.event.connect(self._event)
        self.client.finished.connect(self._finished)
        self._action = ''
        self._inspection: dict | None = None
        self._result: dict | None = None
        self._cameras: list[dict] = []
        self._log_path: Path | None = None
        self._log_position = 0
        self._preview_output: Path | None = None
        self._training_backend: str | None = None
        self._training_output: Path | None = None
        self._training_package_sha256: str | None = None
        self._training_control_status: str | None = None
        self._training_pending_id: str | None = None
        self._training_pending_action: str | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._poll)

        self.import_id = QLineEdit()
        self.import_id.setPlaceholderText('例如 courtyard-colmap')
        self.package = QLineEdit()
        self.images = QLineEdit()
        self.model = QLineEdit()
        self.masks = QLineEdit()
        self.group_file = QLineEdit()
        self.independent = QCheckBox('我确认每张图片是独立采样组')
        self.white_ignore = QCheckBox('源遮罩白色表示忽略')
        form = QFormLayout()
        form.addRow('新项目 ID', self.import_id)
        for label, field, file_mode in (
            ('现有共享包（可选）', self.package, False),
            ('images', self.images, False),
            ('COLMAP BIN/TXT 模型', self.model, False),
            ('masks（可选）', self.masks, False),
            ('图片分组 JSON（可选）', self.group_file, True),
        ):
            select = QPushButton('选择…')
            select.clicked.connect(lambda checked=False, f=field, file=file_mode:
                                   self._choose(f, file))
            line = QHBoxLayout()
            line.addWidget(field, 1)
            line.addWidget(select)
            form.addRow(label, line)
            field.textChanged.connect(self._invalidate_inspection)
        self.independent.toggled.connect(self._invalidate_inspection)
        self.white_ignore.toggled.connect(self._invalidate_inspection)
        form.addRow('分组决策', self.independent)
        form.addRow('遮罩极性', self.white_ignore)

        self.inspect_button = QPushButton('检查输入')
        self.inspect_button.clicked.connect(self._inspect)
        self.prepare_button = QPushButton('固定训练包')
        self.prepare_button.setEnabled(False)
        self.prepare_button.clicked.connect(self._prepare)
        self.status = QLabel('先选择 images 和 COLMAP 模型，再检查输入。')
        self.status.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.inspect_button)
        row.addWidget(self.prepare_button)
        row.addWidget(self.status, 1)

        self.issues = QPlainTextEdit()
        self.issues.setReadOnly(True)
        self.issues.setMaximumHeight(110)
        self.camera = QComboBox()
        self.camera.currentIndexChanged.connect(self._show_camera)
        self.image = QLabel('选择相机以查看源图与稀疏点')
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(220)
        self.image.setStyleSheet('background: #111; color: #ddd;')

        self.existing = QComboBox()
        self.existing.currentIndexChanged.connect(self._existing_changed)
        self.reload_button = QPushButton('刷新项目')
        self.reload_button.clicked.connect(self._reload)
        self.relink_button = QPushButton('重新定位源文件')
        self.relink_button.clicked.connect(self._relink)
        self.backend = QComboBox()
        self.backend.addItems(['gsplat', 'postshot'])
        self.steps = QSpinBox()
        self.steps.setRange(0, 10_000_000)
        self.steps.setSpecialValueText('正式默认预算')
        self.train_button = QPushButton('创建训练实验')
        self.train_button.clicked.connect(self._train)
        self.resume_button = QPushButton('恢复失败实验')
        self.resume_button.clicked.connect(self._resume)
        self.open_button = QPushButton('打开已验证 PLY')
        self.open_button.clicked.connect(self._open_model)
        self.pause_button = QPushButton('暂停')
        self.pause_button.clicked.connect(lambda: self._control_training('pause'))
        self.continue_button = QPushButton('继续')
        self.continue_button.clicked.connect(lambda: self._control_training('resume'))
        self.checkpoint_button = QPushButton('保存检查点')
        self.checkpoint_button.clicked.connect(lambda: self._control_training('checkpoint'))
        self.stop_button = QPushButton('提前结束')
        self.stop_button.clicked.connect(lambda: self._control_training('stop'))
        train = QHBoxLayout()
        for widget in (self.existing, self.reload_button, self.relink_button, self.backend, self.steps,
                       self.train_button, self.resume_button, self.open_button):
            train.addWidget(widget)
        controls = QHBoxLayout()
        for button in (self.pause_button, self.continue_button,
                       self.checkpoint_button, self.stop_button):
            controls.addWidget(button)
        self.activity = QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setMaximumHeight(120)
        self.compare = TrainingCompare()

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(row)
        layout.addWidget(QLabel('检查结果与阻断原因'))
        layout.addWidget(self.issues)
        layout.addWidget(self.camera)
        layout.addWidget(self.image, 1)
        layout.addWidget(QLabel('独立外部项目；不表示 Run QA 已通过'))
        layout.addLayout(train)
        layout.addLayout(controls)
        layout.addWidget(self.compare, 2)
        layout.addWidget(self.activity)
        self._reload()
        self._update_training_controls()

    def _choose(self, field: QLineEdit, file: bool) -> None:
        if file:
            value, _ = QFileDialog.getOpenFileName(self, '选择分组 JSON', '', 'JSON (*.json)')
        else:
            value = QFileDialog.getExistingDirectory(self, '选择目录')
        if value:
            field.setText(value)

    def _invalidate_inspection(self) -> None:
        self._inspection = None
        self.prepare_button.setEnabled(False)

    def _options(self) -> dict:
        package = self.package.text().strip()
        if package:
            return {'source': package}
        return {'images': self.images.text().strip() or None,
                'model': self.model.text().strip() or None,
                'masks': self.masks.text().strip() or None,
                'group_file': self.group_file.text().strip() or None,
                'independent_images': self.independent.isChecked(),
                'white_ignore': self.white_ignore.isChecked()}

    def _start(self, action: str, **payload) -> None:
        if self.client.busy:
            return
        self._action = action
        self._result = None
        try:
            self.client.start({'action': action, 'root': str(self.data_root), **payload})
        except Exception as error:
            QMessageBox.warning(self, '操作未启动', str(error))
            return
        self.status.setText(f'正在执行 {action}…')
        self._set_busy(True)

    def _set_busy(self, busy: bool) -> None:
        for control in (self.inspect_button, self.prepare_button, self.train_button,
                        self.resume_button, self.reload_button, self.relink_button,
                        self.open_button):
            control.setEnabled(not busy)
        if not busy:
            self.prepare_button.setEnabled(bool(self._inspection and not self._inspection['errors']))
        self._update_training_controls()

    def _update_training_controls(self) -> None:
        active = (self.client.busy and self._action == 'train_external_import' and
                  self._training_backend == 'gsplat' and self._training_output is not None)
        pending = self._training_pending_id is not None
        state = self._training_control_status
        self.pause_button.setEnabled(active and state == 'running' and not pending)
        self.continue_button.setEnabled(active and state == 'paused' and not pending)
        self.checkpoint_button.setEnabled(active and state in {'running', 'paused'} and not pending)
        self.stop_button.setEnabled(active and state in {'waiting_gpu', 'running', 'paused'} and
                                    self._training_pending_action != 'stop')

    def _control_training(self, action: str) -> None:
        if not self._training_output or self._training_backend != 'gsplat':
            return
        from gsstudio.application.operations import control_training
        try:
            result = control_training(self.data_root, self._training_output, action)
            self._training_pending_id = result['request_id']
            self._training_pending_action = action
            self.activity.appendPlainText(f'[训练控制] 已请求 {action}；等待训练步边界确认')
            self._update_training_controls()
        except Exception as error:
            QMessageBox.warning(self, '训练控制失败', str(error))

    def _inspect(self) -> None:
        self._invalidate_inspection()
        self._start('inspect_external_import', **self._options())

    def _prepare(self) -> None:
        if self._inspection is None or self._inspection['errors']:
            return
        import_id = self.import_id.text().strip()
        if not import_id:
            QMessageBox.warning(self, '缺少项目 ID', '请为这次外部导入填写项目 ID。')
            return
        self._start('prepare_external_import', import_id=import_id,
                    expected_input_identity=self._inspection['input_identity'],
                    **self._options())

    def _train(self) -> None:
        import_id = self.existing.currentData()
        if import_id:
            self._start('train_external_import', import_id=import_id,
                        backend=self.backend.currentText(), steps=self.steps.value() or None)

    def _relink(self) -> None:
        import_id = self.existing.currentData()
        if not import_id:
            return
        from PySide6.QtWidgets import QInputDialog
        modes = ['按原目录结构重新定位', '使用逐文件路径映射 JSON']
        selected, ok = QInputDialog.getItem(self, '重新定位外部源', '选择定位方式',
                                            modes, 0, False)
        if not ok:
            return
        if selected == modes[0]:
            path = QFileDialog.getExistingDirectory(self, '选择新的源根目录')
            key = 'new_root'
        else:
            path, _ = QFileDialog.getOpenFileName(self, '选择路径映射 JSON', '', 'JSON (*.json)')
            key = 'mapping_file'
        if path:
            self._start('relink_external_import', import_id=import_id, **{key: path})

    def _resume(self) -> None:
        record = self.existing.currentData(Qt.ItemDataRole.UserRole + 1) or {}
        candidates = [item for item in record.get('experiments', [])
                      if item.get('status') in {'failed', 'preparing', 'stopped'} and
                      item.get('backend') == 'gsplat']
        if not candidates:
            QMessageBox.information(self, '恢复实验', '当前项目没有可恢复的 gsplat 实验。')
            return
        from PySide6.QtWidgets import QInputDialog
        names = [item['id'] for item in candidates]
        selected, ok = QInputDialog.getItem(self, '恢复实验', '选择实验', names, len(names)-1, False)
        if ok:
            self._start('train_external_import', import_id=record['id'], backend='gsplat',
                        resume=True, experiment_id=selected)

    def _open_model(self) -> None:
        record = self.existing.currentData(Qt.ItemDataRole.UserRole + 1) or {}
        experiments = [item for item in record.get('experiments', [])
                       if item.get('status') == 'succeeded' and item.get('model_sha256')]
        if not experiments:
            QMessageBox.information(self, 'PLY', '当前项目没有已验证的模型记录。')
            return
        from PySide6.QtWidgets import QInputDialog
        names = [item['id'] for item in experiments]
        selected, ok = QInputDialog.getItem(self, '打开模型', '选择实验', names, len(names)-1, False)
        if ok:
            item = experiments[names.index(selected)]
            output = Path(record['project']) / 'experiments' / item['id']
            path = output / 'model.ply'
            from gsstudio.application._shared import _verify_model
            try:
                dispatch = json.loads((output / 'dispatch.json').read_text(encoding='utf-8'))
                if (dispatch.get('status') != 'succeeded' or
                        dispatch.get('package_sha256') != record.get('package_sha256') or
                        dispatch.get('model_sha256') != item['model_sha256']):
                    raise RuntimeError('训练实验记录与 PLY 身份不一致')
                _verify_model(path, item['model_sha256'])
            except Exception as error:
                QMessageBox.warning(self, 'PLY 身份错误', str(error))
                return
            self.model_requested.emit(str(path), item['model_sha256'])
            self.accept()

    def _reload(self) -> None:
        self._start('list_external_imports')

    def _existing_changed(self) -> None:
        record = self.existing.currentData(Qt.ItemDataRole.UserRole + 1) or {}
        self.train_button.setEnabled(bool(record and record.get('status') == 'prepared'))

    def _event(self, event: dict) -> None:
        kind, message = event.get('kind'), event.get('message', '')
        self.activity.appendPlainText(f'[{kind}] {message}')
        self.status.setText(message)
        if kind == 'result':
            self._result = (event.get('detail') or {}).get('result')
        elif kind == 'error':
            QMessageBox.warning(self, '外部导入失败', message)
        detail = event.get('detail') or {}
        if detail.get('log_path'):
            self._log_path = Path(detail['log_path'])
            self._log_position = 0
            self._preview_output = Path(detail['output'])
            self._training_output = self._preview_output
            self._training_backend = detail.get('backend')
            self._training_package_sha256 = detail.get('package_sha256')
            self._training_control_status = None
            self._training_pending_id = None
            self._training_pending_action = None
            self.compare.configure(self._preview_output, Path(detail['package']),
                                   detail.get('package_sha256'), detail.get('backend', ''))
            self.timer.start()
            self._update_training_controls()

    def _finished(self, exit_code: int) -> None:
        self._poll()
        self.timer.stop()
        self._log_path = None
        self._preview_output = None
        self._training_output = None
        self._training_backend = None
        self._training_control_status = None
        self._training_pending_id = None
        self._training_pending_action = None
        self._set_busy(False)
        if exit_code or self._result is None:
            return
        if self._action == 'inspect_external_import':
            self._inspection = self._result
            info = self._inspection
            self.issues.setPlainText(
                f"注册相机 {info['registered_images']} / 图片 {info['image_inventory']}；"
                f"初始化点 {info['initial_points']}；遮罩 {info['masks']}；"
                f"模型候选 {len(info['model_candidates'])}；相机 {', '.join(info['camera_models']) or '—'}；"
                f"分组来源 {info['group_source']}\n" +
                '\n'.join('模型：' + item for item in info['model_candidates']) + '\n' +
                '\n'.join('! ' + item for item in info['errors']) + '\n' +
                '\n'.join('· ' + item for item in info['warnings']))
            self._cameras = info['cameras']
            self.camera.blockSignals(True)
            self.camera.clear()
            for row in self._cameras:
                self.camera.addItem(f"{row['source_image']} · {row['camera_model']} · "
                                    f"{row['split']} · 组 {row['group']}")
            self.camera.blockSignals(False)
            self._show_camera(0)
            self.prepare_button.setEnabled(not info['errors'])
            self.status.setText('发现阻断问题' if info['errors'] else '输入检查通过，可固定训练包')
        elif self._action == 'list_external_imports':
            previous = self.existing.currentData()
            self.existing.blockSignals(True)
            self.existing.clear()
            for item in self._result['imports']:
                self.existing.addItem(f"{item['id']} · {item['status']}", item['id'])
                self.existing.setItemData(self.existing.count()-1, item,
                                          Qt.ItemDataRole.UserRole + 1)
            index = self.existing.findData(previous)
            self.existing.setCurrentIndex(max(index, 0))
            self.existing.blockSignals(False)
            self._existing_changed()
        elif self._action in {'prepare_external_import', 'train_external_import',
                              'relink_external_import'}:
            if self._action == 'prepare_external_import':
                self.import_id.setText('')
            self._reload()

    def _show_camera(self, index: int) -> None:
        if index < 0 or index >= len(self._cameras):
            self.image.setText('没有可查看的相机')
            return
        row = self._cameras[index]
        source = QPixmap(row['source_path'])
        if source.isNull():
            self.image.setText(f"图片无法显示：{row['source_image']}")
            return
        canvas = source.scaled(760, 300, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(canvas)
        painter.setPen(QPen(QColor('#00e5ff'), 2))
        sx, sy = canvas.width() / row['width'], canvas.height() / row['height']
        for x, y in row['sparse_points']:
            painter.drawEllipse(int(x * sx) - 2, int(y * sy) - 2, 4, 4)
        painter.end()
        self.image.setPixmap(canvas)
        self.image.setToolTip(row['source_path'])

    def _poll(self) -> None:
        if self._training_output is not None and self._training_backend == 'gsplat':
            state_path = self._training_output / 'control-state.json'
            if state_path.is_file():
                try:
                    state = json.loads(state_path.read_text(encoding='utf-8'))
                    if state.get('package_sha256') == self._training_package_sha256:
                        if state.get('request_id') == self._training_pending_id:
                            self._training_pending_id = None
                            self._training_pending_action = None
                        status = state.get('status')
                        if status != self._training_control_status:
                            self._training_control_status = status
                            self.activity.appendPlainText(
                                f"[训练控制] {status} · step {state.get('step', '—')}")
                        self._update_training_controls()
                except (OSError, ValueError, TypeError):
                    pass
        if self._preview_output is not None:
            self.compare.refresh()
        if self._log_path is None or not self._log_path.is_file():
            return
        with self._log_path.open('r', encoding='utf-8', errors='replace') as stream:
            stream.seek(self._log_position)
            lines = stream.read().splitlines()
            self._log_position = stream.tell()
        for line in lines[-30:]:
            self.activity.appendPlainText(line[:500])
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and 'step' in payload:
                self.status.setText(f"训练 step {payload['step']} · loss {payload.get('loss', '—')}")

    def closeEvent(self, event) -> None:
        if self.client.busy:
            if self._action == 'train_external_import' and self._training_backend == 'gsplat':
                if (self._training_control_status not in {'stopping', 'stopped'} and
                        self._training_pending_action != 'stop'):
                    self._control_training('stop')
                message = '已请求在训练步边界保存并结束；完成前请保持窗口打开。'
            else:
                message = '当前操作仍在运行；请等待工作进程完成后再关闭。'
            QMessageBox.information(self, '任务进行中', message)
            event.ignore()
            return
        super().closeEvent(event)
