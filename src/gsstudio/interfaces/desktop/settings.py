"""First-launch Data selection for the desktop application."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog

from gsstudio.infrastructure.paths import find_data_root, save_data_root


def data_root_for_desktop() -> Path | None:
    try:
        return find_data_root()
    except (OSError, ValueError, RuntimeError):
        selected = QFileDialog.getExistingDirectory(
            None, "选择 GS Studio Data 目录", str(Path.home())
        )
        return save_data_root(Path(selected)) if selected else None
