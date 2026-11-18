"""Pixel-accurate texture canvas."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .images import numpy_to_qimage

ZOOMS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


class CanvasWidget(QWidget):
    """Draws the texture with nearest-neighbor zoom, a grid, and the selection."""

    pressed = Signal(int, int, int)
    moved = Signal(int, int)
    released = Signal(int, int)
    hovered = Signal(int, int)
    zoomChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("canvas")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMinimumSize(240, 240)
        self._pixels: np.ndarray | None = None
        self._zoom_index = ZOOMS.index(16)
        self.zoom = ZOOMS[self._zoom_index]
        self.pan_x = 40.0
        self.pan_y = 40.0
        self.show_minor_grid = True
        self.show_major_grid = True
        self.selection: tuple[int, int, int, int] | None = None
        self.selection_offset = (0, 0)
        self._space = False
        self._panning = False
        self._dragging = False
        self._pan_origin = QPointF()
        self._pan_start = (0.0, 0.0)
        self._hover = (-1, -1)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_pixels(self, arr: np.ndarray) -> None:
        self._pixels = arr
        self.update()

    def set_selection(self, rect: tuple[int, int, int, int] | None, offset: tuple[int, int]) -> None:
        self.selection = rect
        self.selection_offset = offset
        self.update()

    def set_space(self, held: bool) -> None:
        self._space = held
        if not held:
            self._panning = False
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif not self._panning:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def pixel_to_widget(self, x: int, y: int) -> QPoint:
        px = int(round(self.pan_x + (x + 0.5) * self.zoom))
        py = int(round(self.pan_y + (y + 0.5) * self.zoom))
        return QPoint(px, py)

    def cell_at(self, pos: QPointF) -> tuple[int, int]:
        x = math.floor((pos.x() - self.pan_x) / self.zoom)
        y = math.floor((pos.y() - self.pan_y) / self.zoom)
        return int(x), int(y)

    def fit(self) -> None:
        if self._pixels is None or self.width() < 2 or self.height() < 2:
            return
        height, width = self._pixels.shape[:2]
        target = min((self.width() - 48) / width, (self.height() - 48) / height)
        index = 0
        for i, zoom in enumerate(ZOOMS):
            if zoom <= target + 1e-6:
                index = i
        self._zoom_index = index
        self.zoom = ZOOMS[index]
        self.center()
        self.zoomChanged.emit(self.zoom)

    def center(self) -> None:
        if self._pixels is None:
            return
        height, width = self._pixels.shape[:2]
        self.pan_x = (self.width() - width * self.zoom) / 2
        self.pan_y = (self.height() - height * self.zoom) / 2
        self._snap_pan()
        self.update()

    def zoom_by(self, direction: int, anchor: QPointF | None = None) -> None:
        index = max(0, min(len(ZOOMS) - 1, self._zoom_index + direction))
        self._set_zoom_index(index, anchor)

    def set_zoom(self, zoom: int, anchor: QPointF | None = None) -> None:
        if zoom not in ZOOMS:
            return
        self._set_zoom_index(ZOOMS.index(zoom), anchor)

    def _set_zoom_index(self, index: int, anchor: QPointF | None) -> None:
        old = self.zoom
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        img_x = (anchor.x() - self.pan_x) / old
        img_y = (anchor.y() - self.pan_y) / old
        self._zoom_index = index
        self.zoom = ZOOMS[index]
        self.pan_x = anchor.x() - img_x * self.zoom
        self.pan_y = anchor.y() - img_y * self.zoom
        self._snap_pan()
        self.zoomChanged.emit(self.zoom)
        self.update()

    def _snap_pan(self) -> None:
        dpr = self.devicePixelRatioF() or 1.0
        self.pan_x = round(self.pan_x * dpr) / dpr
        self.pan_y = round(self.pan_y * dpr) / dpr

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#141416"))
        if self._pixels is None:
            painter.end()
            return
        height, width = self._pixels.shape[:2]
        x0, y0, x1, y1 = self._visible_cells(width, height)
        if x0 < x1 and y0 < y1:
            self._draw_image(painter, x0, y0, x1, y1, checker=True)
            self._draw_image(painter, x0, y0, x1, y1, checker=False)
            self._draw_grid(painter, x0, y0, x1, y1)
        self._draw_selection(painter)
        self._draw_hover(painter)
        painter.end()

    def _visible_cells(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0 = max(0, math.floor(-self.pan_x / self.zoom) - 1)
        y0 = max(0, math.floor(-self.pan_y / self.zoom) - 1)
        x1 = min(width, math.ceil((self.width() - self.pan_x) / self.zoom) + 1)
        y1 = min(height, math.ceil((self.height() - self.pan_y) / self.zoom) + 1)
        return x0, y0, x1, y1

    def _draw_image(self, painter: QPainter, x0: int, y0: int, x1: int, y1: int, checker: bool) -> None:
        if checker:
            ys, xs = np.ogrid[y0:y1, x0:x1]
            mask = ((xs + ys) & 1) == 0
            chunk = np.empty((y1 - y0, x1 - x0, 4), dtype=np.uint8)
            chunk[mask] = (48, 48, 54, 255)
            chunk[~mask] = (28, 28, 32, 255)
        else:
            assert self._pixels is not None
            chunk = self._pixels[y0:y1, x0:x1]
        image = numpy_to_qimage(chunk)
        dpr = self.devicePixelRatioF() or 1.0
        dest_w = max(1, int(round((x1 - x0) * self.zoom * dpr)))
        dest_h = max(1, int(round((y1 - y0) * self.zoom * dpr)))
        scaled = image.scaled(dest_w, dest_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
        scaled.setDevicePixelRatio(dpr)
        painter.drawImage(QPointF(self.pan_x + x0 * self.zoom, self.pan_y + y0 * self.zoom), scaled)

    def _draw_grid(self, painter: QPainter, x0: int, y0: int, x1: int, y1: int) -> None:
        top = self.pan_y + y0 * self.zoom
        bottom = self.pan_y + y1 * self.zoom
        left = self.pan_x + x0 * self.zoom
        right = self.pan_x + x1 * self.zoom
        if self.show_minor_grid and self.zoom >= 8:
            pen = QPen(QColor(255, 255, 255, 40))
            pen.setCosmetic(True)
            painter.setPen(pen)
            for x in range(x0, x1 + 1):
                px = self.pan_x + x * self.zoom
                painter.drawLine(QPointF(px, top), QPointF(px, bottom))
            for y in range(y0, y1 + 1):
                py = self.pan_y + y * self.zoom
                painter.drawLine(QPointF(left, py), QPointF(right, py))
        if self.show_major_grid:
            pen = QPen(QColor(224, 164, 90, 180))
            pen.setCosmetic(True)
            painter.setPen(pen)
            start_x = (x0 // 16) * 16
            start_y = (y0 // 16) * 16
            for x in range(start_x, x1 + 1, 16):
                px = self.pan_x + x * self.zoom
                painter.drawLine(QPointF(px, top), QPointF(px, bottom))
            for y in range(start_y, y1 + 1, 16):
                py = self.pan_y + y * self.zoom
                painter.drawLine(QPointF(left, py), QPointF(right, py))

    def _draw_selection(self, painter: QPainter) -> None:
        if self.selection is None:
            return
        sx, sy, sw, sh = self.selection
        ox, oy = self.selection_offset
        rect_x = self.pan_x + (sx + ox) * self.zoom
        rect_y = self.pan_y + (sy + oy) * self.zoom
        rect_w = sw * self.zoom
        rect_h = sh * self.zoom
        solid = QPen(QColor(12, 12, 12))
        solid.setCosmetic(True)
        painter.setPen(solid)
        painter.drawRect(QRectF(rect_x, rect_y, rect_w, rect_h))
        dashed = QPen(QColor(245, 245, 245))
        dashed.setStyle(Qt.PenStyle.DashLine)
        dashed.setCosmetic(True)
        painter.setPen(dashed)
        painter.drawRect(QRectF(rect_x, rect_y, rect_w, rect_h))

    def _draw_hover(self, painter: QPainter) -> None:
        x, y = self._hover
        if self._pixels is None:
            return
        height, width = self._pixels.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        pen = QPen(QColor(255, 255, 255, 200))
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(QRectF(self.pan_x + x * self.zoom, self.pan_y + y * self.zoom, self.zoom, self.zoom))

    def wheelEvent(self, event) -> None:  # noqa: N802
        direction = 1 if event.angleDelta().y() > 0 else -1
        self.zoom_by(direction, event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space
        ):
            self._panning = True
            self._pan_origin = event.position()
            self._pan_start = (self.pan_x, self.pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        button = _button_code(event.button())
        if not button:
            return
        x, y = self.cell_at(event.position())
        self._dragging = True
        if QGuiApplication.platformName() != "offscreen":
            self.grabMouse()
        self.pressed.emit(x, y, button)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            delta = event.position() - self._pan_origin
            self.pan_x = self._pan_start[0] + delta.x()
            self.pan_y = self._pan_start[1] + delta.y()
            self._snap_pan()
            self.update()
            event.accept()
            return
        x, y = self.cell_at(event.position())
        if (x, y) != self._hover:
            self._hover = (x, y)
            self.hovered.emit(x, y)
            self.update()
        if self._dragging:
            self.moved.emit(x, y)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space else Qt.CursorShape.CrossCursor)
            event.accept()
            return
        if self._dragging and _button_code(event.button()):
            self._dragging = False
            if self.mouseGrabber() is self:
                self.releaseMouse()
            x, y = self.cell_at(event.position())
            self.released.emit(x, y)
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_space(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_space(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = (-1, -1)
        self.hovered.emit(-1, -1)
        self.update()
        super().leaveEvent(event)


def _button_code(button: Qt.MouseButton) -> int:
    if button == Qt.MouseButton.LeftButton:
        return 1
    if button == Qt.MouseButton.RightButton:
        return 2
    return 0
