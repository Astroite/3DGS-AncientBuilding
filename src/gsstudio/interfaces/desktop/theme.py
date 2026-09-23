"""A 方向设计令牌与样式表（docs/GS-Studio-Plan/design/specification-a/VISUAL-SPEC.md）。

数值直接取自已通过评审的规范：颜色角色、字体回退、逻辑字号、间距与控件高度。
样式表用像素字号以保持与设计逻辑像素 1:1；Qt 高 DPI 缩放负责物理换算。
"""
from __future__ import annotations

APP_BG = "#121719"
PANEL_BG = "#1B2225"
SURFACE = "#263135"
VIEW_BG = "#101516"
BORDER = "#354247"
TEXT = "#EAF1F0"
TEXT_MUTED = "#A5B4B6"
ACCENT = "#77D6C3"
ACCENT_BG = "#233D39"
ACCENT_TEXT = "#102A24"
WARNING = "#EDC17C"
ERROR = "#FAADB1"
AXIS_Z = "#8DAFF7"

FONT_FAMILY = "Segoe UI, 'Microsoft YaHei UI', 'Microsoft YaHei', sans-serif"

SIZE_PAGE_TITLE = 32
SIZE_PANEL_TITLE = 20
SIZE_PROGRESS = 26
SIZE_BODY = 17
SIZE_LABEL = 15
SIZE_CAPTION = 13

SPACE_BASE = 8
SPACE_GROUP = 12
SPACE_IN_GROUP = 16
SPACE_PANEL = 24
SPACE_AREA = 24
SPACE_SECTION = 32

HEIGHT_BUTTON = 42
HEIGHT_PRIMARY = 46
HEIGHT_NUMBER_INPUT = 44
HEIGHT_PATH_INPUT = 52

RADIUS_WIDGET = 7
RADIUS_PREVIEW = 4
FOCUS_WIDTH = 2

# RunStatus / StageStatus 的显示文字。状态用「符号 + 文字」，不只靠颜色。
RUN_STATUS_TEXT = {
    "draft": "草稿",
    "processing": "处理中",
    "needs_inspection": "待检查",
    "failed": "! 失败",
    "waiting_review": "等待复核",
    "needs_review": "待复核",
    "accepted": "✓ 已接受",
    "rejected": "! 已拒绝",
    "unreadable": "! 不可读",
}

STAGE_STATUS_TEXT = {
    "pending": "待执行",
    "processing": "处理中",
    "succeeded": "✓ 完成",
    "failed": "! 失败",
    "skipped": "跳过",
}


def run_status_text(status: str | None) -> str:
    return RUN_STATUS_TEXT.get(status or "", status or "—")


def stage_status_text(status: str | None) -> str:
    return STAGE_STATUS_TEXT.get(status or "", status or "—")


def build_stylesheet() -> str:
    return f"""
    QWidget {{
        color: {TEXT};
        background: {APP_BG};
        font-family: {FONT_FAMILY};
        font-size: {SIZE_LABEL}px;
    }}
    QMainWindow::separator {{ background: {BORDER}; width: 1px; height: 1px; }}
    QFrame#panel {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: {RADIUS_WIDGET}px; }}
    QFrame#view {{ background: {VIEW_BG}; border: 1px solid {BORDER}; border-radius: {RADIUS_PREVIEW}px; }}
    QLabel#pageTitle {{ font-size: {SIZE_PAGE_TITLE}px; font-weight: 600; color: {TEXT}; }}
    QLabel#panelTitle {{ font-size: {SIZE_PANEL_TITLE}px; font-weight: 600; color: {TEXT}; }}
    QLabel#caption {{ font-size: {SIZE_CAPTION}px; color: {TEXT_MUTED}; }}
    QLabel#muted {{ color: {TEXT_MUTED}; }}
    QLabel#error {{ color: {ERROR}; }}
    QLabel#warning {{ color: {WARNING}; }}
    QLabel#accent {{ color: {ACCENT}; }}
    QTreeView, QTableView, QTreeWidget, QTableWidget, QListWidget, QTextEdit, QPlainTextEdit {{
        background: {PANEL_BG};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_WIDGET}px;
        selection-background-color: {ACCENT_BG};
        selection-color: {TEXT};
        alternate-background-color: {SURFACE};
        font-size: {SIZE_LABEL}px;
    }}
    QTreeView::item, QTableView::item, QListWidget::item {{ padding: {SPACE_BASE}px {SPACE_GROUP}px; }}
    QHeaderView::section {{
        background: {SURFACE};
        color: {TEXT_MUTED};
        border: none;
        border-bottom: 1px solid {BORDER};
        padding: {SPACE_BASE}px {SPACE_GROUP}px;
        font-size: {SIZE_CAPTION}px;
    }}
    QTabWidget::pane {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: {RADIUS_WIDGET}px; }}
    QTabBar::tab {{
        background: {SURFACE};
        color: {TEXT_MUTED};
        padding: {SPACE_GROUP}px {SPACE_IN_GROUP}px;
        border: 1px solid {BORDER};
        border-bottom: none;
        border-top-left-radius: {RADIUS_WIDGET}px;
        border-top-right-radius: {RADIUS_WIDGET}px;
    }}
    QTabBar::tab:selected {{ background: {ACCENT_BG}; color: {TEXT}; }}
    QPushButton {{
        background: {SURFACE};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_WIDGET}px;
        padding: 0 {SPACE_IN_GROUP}px;
        min-height: {HEIGHT_BUTTON}px;
        font-size: {SIZE_BODY}px;
    }}
    QPushButton:hover {{ border-color: {ACCENT}; }}
    QPushButton:disabled {{ color: {TEXT_MUTED}; border-color: {BORDER}; background: {PANEL_BG}; }}
    QPushButton#primary {{
        background: {ACCENT};
        color: {ACCENT_TEXT};
        border: 1px solid {ACCENT};
        min-height: {HEIGHT_PRIMARY}px;
        font-weight: 600;
    }}
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_WIDGET}px;
        padding: 0 {SPACE_GROUP}px;
        min-height: {HEIGHT_NUMBER_INPUT}px;
        selection-background-color: {ACCENT_BG};
    }}
    QLineEdit#path {{ min-height: {HEIGHT_PATH_INPUT}px; font-size: {SIZE_CAPTION}px; }}
    QStatusBar {{ background: {PANEL_BG}; color: {TEXT_MUTED}; border-top: 1px solid {BORDER}; }}
    QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; }}
    QSplitter::handle {{ background: {BORDER}; }}
    """
