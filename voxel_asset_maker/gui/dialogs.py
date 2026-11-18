"""Modal dialogs for new textures and canvas size."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..model import TILE, Color, Document

PRESETS: tuple[tuple[str, int | None, int | None], ...] = (
    ("16 × 16 block", 16, 16),
    ("32 × 32", 32, 32),
    ("64 × 64", 64, 64),
    ("128 × 128", 128, 128),
    ("256 × 256", 256, 256),
    ("16 × 32", 16, 32),
    ("32 × 16", 32, 16),
    ("16 × 48 strip", 16, 48),
    ("Custom", None, None),
)


class NewTextureDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New texture")
        self.result_size: tuple[int, int] | None = None
        self.result_name = "texture"
        self.result_fill: Color | None = None
        self._fill: Color = (127, 127, 127, 255)
        self._use_fill = False

        self.name = QLineEdit("texture")
        self.preset = QComboBox()
        for label, _w, _h in PRESETS:
            self.preset.addItem(label)
        self.width = _size_box()
        self.height = _size_box()
        self.fill_button = QPushButton("Transparent")
        self.fill_button.clicked.connect(self._pick_fill)
        hint = QLabel("Width and height are multiples of 16, matching block and item slots.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Preset", self.preset)
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)
        form.addRow("Background", self.fill_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)

        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.width.valueChanged.connect(self._mark_custom)
        self.height.valueChanged.connect(self._mark_custom)
        self._apply_preset(0)
        self.resize(360, 240)

    def _apply_preset(self, index: int) -> None:
        _label, width, height = PRESETS[index]
        if width is None or height is None:
            return
        self.width.blockSignals(True)
        self.height.blockSignals(True)
        self.width.setValue(width)
        self.height.setValue(height)
        self.width.blockSignals(False)
        self.height.blockSignals(False)

    def _mark_custom(self) -> None:
        label, width, height = PRESETS[self.preset.currentIndex()]
        if width is None:
            return
        if self.width.value() != width or self.height.value() != height:
            custom = len(PRESETS) - 1
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(custom)
            self.preset.blockSignals(False)
            del label

    def _pick_fill(self) -> None:
        initial = QColor(*self._fill)
        chosen = QColorDialog.getColor(
            initial,
            self,
            "Background",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not chosen.isValid():
            self._use_fill = False
            self.fill_button.setText("Transparent")
            return
        self._fill = (chosen.red(), chosen.green(), chosen.blue(), chosen.alpha())
        self._use_fill = True
        self.fill_button.setText(
            f"#{chosen.red():02x}{chosen.green():02x}{chosen.blue():02x}{chosen.alpha():02x}"
        )

    def _accept(self) -> None:
        width = self.width.value()
        height = self.height.value()
        if width % TILE or height % TILE:
            QMessageBox.warning(
                self,
                "Size",
                "Width and height have to be multiples of 16.",
            )
            return
        self.result_size = (width, height)
        self.result_name = self.name.text().strip() or "texture"
        self.result_fill = self._fill if self._use_fill else None
        self.accept()


class ResizeDialog(QDialog):
    def __init__(self, doc: Document, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resize canvas")
        self.result_size: tuple[int, int] | None = None
        self.result_anchor = "topleft"
        self.width = _size_box()
        self.height = _size_box()
        self.width.setValue(doc.width)
        self.height.setValue(doc.height)
        self.anchor = QComboBox()
        self.anchor.addItem("Top left", "topleft")
        self.anchor.addItem("Center", "center")
        hint = QLabel("Existing pixels stay put. New space is transparent.")
        hint.setObjectName("hint")
        form = QFormLayout()
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)
        form.addRow("Anchor", self.anchor)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        width = self.width.value()
        height = self.height.value()
        if width % TILE or height % TILE:
            QMessageBox.warning(self, "Size", "Width and height have to be multiples of 16.")
            return
        self.result_size = (width, height)
        self.result_anchor = str(self.anchor.currentData())
        self.accept()


def _size_box() -> QSpinBox:
    box = QSpinBox()
    box.setRange(16, 4096)
    box.setSingleStep(16)
    box.setValue(16)
    box.setSuffix(" px")
    return box


def ask_text(parent: QWidget, title: str, label: str, text: str) -> str | None:
    from PySide6.QtWidgets import QInputDialog

    value, ok = QInputDialog.getText(parent, title, label, text=text)
    if not ok:
        return None
    cleaned = value.strip()
    return cleaned or None
