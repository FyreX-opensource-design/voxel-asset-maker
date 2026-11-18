"""Dark editor theme."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

STYLESHEET = """
QToolTip { color: #f4f4f4; background: #2c2a24; border: 1px solid #c4843a; padding: 4px; }
QMainWindow, QWidget#panel { background: #1c1c20; color: #ececec; }
QSplitter::handle { background: #121214; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #1c1c20; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3c3c44; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar:horizontal { background: #1c1c20; height: 10px; }
QScrollBar::handle:horizontal { background: #3c3c44; min-width: 24px; }
QGroupBox {
    border: 1px solid #34343a;
    margin-top: 12px;
    padding: 8px 6px 6px 6px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #e0a45a;
}
QLabel#hint { color: #a3a3ab; }
QLabel#error { color: #f0a8a0; }
QToolBar { background: #1c1c20; border: none; border-bottom: 1px solid #34343a; spacing: 6px; padding: 3px; }
QStatusBar { background: #16161a; color: #c8c8cc; }
QStatusBar::item { border: none; }
QMenuBar { background: #1c1c20; color: #ececec; }
QMenuBar::item:selected { background: #3a2c1c; }
QMenu { background: #242428; color: #ececec; border: 1px solid #3a3a40; }
QMenu::item:selected { background: #6a4a1e; }
QPushButton {
    background: #2c2c32;
    color: #f0f0f0;
    border: 1px solid #3f3f46;
    padding: 4px 8px;
}
QPushButton:hover { border-color: #e0a45a; }
QPushButton:checked { background: #6a4516; border: 1px solid #e0a45a; }
QPushButton:disabled { color: #8a8a92; }
QToolButton {
    background: #2c2c32;
    border: 1px solid #3a3a40;
    padding: 0;
}
QToolButton:hover { border-color: #e0a45a; }
QListWidget { background: #17171a; color: #f0f0f0; border: 1px solid #34343a; }
QListWidget::item { padding: 3px 4px; }
QListWidget::item:selected { background: #3a2e1c; color: #fff8ea; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #141418;
    color: #f4f4f4;
    border: 1px solid #3f3f46;
    padding: 2px 4px;
    selection-background-color: #6a4a1e;
}
QSlider { min-height: 22px; }
QSlider::groove:horizontal { height: 6px; background: #3a3a42; margin: 0 2px; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #e0a45a;
    border: 1px solid #f3d7a6;
    width: 14px;
    margin: -6px 0;
    border-radius: 3px;
}
QSlider::handle:horizontal:disabled { background: #5a5a62; border-color: #5a5a62; }
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1c1c20"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#ececec"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#17171a"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#242428"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#ececec"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2e2e34"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ececec"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#a56b22"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#2a2a24"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f4f4f4"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8a8a92"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#8a8a92"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#8a8a92"))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
