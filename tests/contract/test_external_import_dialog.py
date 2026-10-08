"""The Qt entry exposes external grouping and an image/point preview."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')

from PySide6.QtWidgets import QApplication  # noqa: E402
from gsstudio.interfaces.desktop.external_import import ExternalImportDialog  # noqa: E402
from gsstudio.pipeline.editor.data import write_image  # noqa: E402


def test_dialog_requires_visible_group_choice_and_draws_camera(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ExternalImportDialog, '_reload', lambda self: None)
    dialog = ExternalImportDialog(tmp_path)
    dialog.images.setText(str(tmp_path / 'images'))
    dialog.model.setText(str(tmp_path / 'sparse' / '0'))
    assert not dialog._options()['independent_images']
    dialog.independent.setChecked(True)
    assert dialog._options()['independent_images']
    assert dialog._inspection is None
    source = tmp_path / '测试.png'
    write_image(source, np.full((10, 10, 3), 128, np.uint8))
    dialog._cameras = [{'source_path': str(source), 'source_image': source.name,
                        'width': 10, 'height': 10, 'sparse_points': [[5, 5]]}]
    dialog._show_camera(0)
    assert not dialog.image.pixmap().isNull()
    dialog.close()
    assert app is not None
