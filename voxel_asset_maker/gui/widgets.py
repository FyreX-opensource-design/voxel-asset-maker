"""Small controls used by the texture editor window."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..model import Color


class Swatch(QWidget):
    """A color chip drawn over a checkerboard so alpha stays visible."""

    clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.color: Color = (0, 0, 0, 255)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_color(self, color: Color) -> None:
        self.color = color
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(1)
        elif event.button() == Qt.MouseButton.RightButton:
            self.clicked.emit(2)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        cell = 4
        for y in range(0, self.height(), cell):
            for x in range(0, self.width(), cell):
                on = ((x // cell) + (y // cell)) % 2 == 0
                painter.fillRect(x, y, cell, cell, QColor("#3a3a40") if on else QColor("#222226"))
        red, green, blue, alpha = self.color
        painter.fillRect(self.rect(), QColor(red, green, blue, alpha))
        painter.setPen(QColor("#111111"))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        painter.setPen(QColor("#e0a45a"))
        painter.drawRect(1, 1, self.width() - 3, self.height() - 3)
        painter.end()


class PaletteChip(QWidget):
    picked = Signal(str, int)

    def __init__(self, hex_color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.hex_color = hex_color
        self.setFixedSize(18, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{hex_color}\nLeft click sets the front color. Right click sets the back color.")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.picked.emit(self.hex_color, 1)
        elif event.button() == Qt.MouseButton.RightButton:
            self.picked.emit(self.hex_color, 2)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        color = QColor(self.hex_color)
        painter.fillRect(self.rect(), color)
        painter.setPen(QPen(QColor("#111111")))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        painter.end()


def tool_icon(kind: str) -> QIcon:
    pixmap = QPixmap(18, 18)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    pen = QPen(QColor("#f2f2f2"))
    pen.setWidth(1)
    pen.setCosmetic(True)
    painter.setPen(pen)
    if kind == "pencil":
        painter.drawLine(4, 14, 13, 5)
        painter.drawLine(3, 15, 5, 13)
    elif kind == "eraser":
        painter.drawRect(4, 4, 10, 10)
    elif kind == "fill":
        painter.drawRect(3, 3, 6, 6)
        painter.drawLine(10, 10, 15, 15)
        painter.drawPoint(14, 12)
    elif kind == "picker":
        painter.drawLine(4, 14, 12, 6)
        painter.drawRect(11, 3, 4, 4)
    elif kind == "line":
        painter.drawLine(3, 14, 14, 3)
    elif kind == "rect":
        painter.drawRect(3, 3, 11, 11)
    elif kind == "select":
        dash = QPen(QColor("#f2f2f2"))
        dash.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(dash)
        painter.drawRect(3, 3, 11, 11)
    painter.end()
    return QIcon(pixmap)
