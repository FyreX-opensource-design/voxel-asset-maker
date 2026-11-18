"""Convert between RGBA arrays and QImage."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    data = np.ascontiguousarray(arr, dtype=np.uint8)
    height, width = data.shape[:2]
    raw = data.tobytes()
    image = QImage(raw, width, height, width * 4, QImage.Format.Format_RGBA8888)
    return image.copy()


def qimage_to_numpy(image: QImage) -> np.ndarray | None:
    if image.isNull():
        return None
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    width = converted.width()
    height = converted.height()
    bits = converted.constBits()
    raw = np.frombuffer(bytes(bits), dtype=np.uint8)
    stride = converted.bytesPerLine()
    rows = raw.reshape((height, stride))
    return rows[:, : width * 4].reshape((height, width, 4)).copy()
