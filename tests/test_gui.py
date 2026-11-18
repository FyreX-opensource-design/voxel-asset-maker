"""Offscreen checks that the window paints, edits, and exports."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from voxel_asset_maker.gui.images import numpy_to_qimage, qimage_to_numpy  # noqa: E402
from voxel_asset_maker.gui.window import MainWindow  # noqa: E402


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_image_roundtrip(self) -> None:
        import numpy as np

        source = np.zeros((2, 2, 4), dtype=np.uint8)
        source[0, 1] = (12, 34, 56, 78)
        restored = qimage_to_numpy(numpy_to_qimage(source))
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(tuple(restored[0, 1]), (12, 34, 56, 78))

    def test_click_paints_and_zip_exports(self) -> None:
        window = MainWindow()
        window.session.history.mark_clean()
        try:
            window.show()
            self.app.processEvents()
            window.session.new_document(16, 16, name="stone", fill=(90, 140, 60, 255))
            window.sync_all()
            window.canvas.fit()
            self.app.processEvents()
            window.session.fg = (200, 30, 30, 255)
            window.sync_colors()
            point = window.canvas.pixel_to_widget(2, 3)
            QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
            self.app.processEvents()
            painted = tuple(int(channel) for channel in window.session.doc.active_layer.pixels[3, 2])
            self.assertEqual(painted, (200, 30, 30, 255))

            window.hex_field.setFocus()
            self.app.processEvents()
            window.hex_field.clear()
            QTest.keyClicks(window.hex_field, "abc")
            self.app.processEvents()
            self.assertEqual(window.session.tool, "pencil")
            self.assertIn("abc", window.hex_field.text())

            window.session.begin_edit()
            window.session.set_script_id("noise")
            window.session.set_script_param("count", 3)
            window.session.set_script_param("sigma", 22)
            window.session.set_script_param("seed", 9)
            window.session.end_edit()
            window.sync_all()
            self.assertEqual(window.session.variant_count(), 3)
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "stone.zip"
                names, errors = window.session.export_zip(target)
                self.assertEqual(errors, [])
                self.assertEqual(len(names), 3)
                with zipfile.ZipFile(target) as archive:
                    self.assertIn("manifest.json", archive.namelist())
                    self.assertEqual(len([name for name in archive.namelist() if name.endswith(".png")]), 3)
            self.assertIn("3 texture", window.session.message)
        finally:
            window.session.history.mark_clean()
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_noise_slider_changes_the_result(self) -> None:
        import numpy as np

        window = MainWindow()
        window.session.history.mark_clean()
        try:
            window.resize(1100, 720)
            window.show()
            self.app.processEvents()
            window.session.new_document(16, 16, name="stone", fill=(80, 140, 60, 255))
            window.session.set_script_id("noise")
            window.session.set_script_param("sigma", 28)
            window.session.set_script_param("seed", 11)
            window.sync_all()
            self.app.processEvents()
            self.assertGreater(window.variant_slider.maximum(), 1)
            self.assertTrue(window.variant_slider.isEnabled())
            first = window.session.display_array().copy()
            window.variant_slider.setValue(window.variant_slider.maximum())
            self.app.processEvents()
            second = window.session.display_array().copy()
            self.assertFalse(np.array_equal(first, second))
            self.assertEqual(window.session.variant_index, window.variant_slider.maximum())
            self.assertEqual(window.variant_spin.value(), window.variant_slider.value())
        finally:
            window.session.history.mark_clean()
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_clipboard_image_paste(self) -> None:
        window = MainWindow()
        window.session.history.mark_clean()
        try:
            image = QImage(4, 4, QImage.Format.Format_RGBA8888)
            image.fill(QColor(1, 2, 3, 255))
            QApplication.clipboard().setImage(image)
            window.session.paste_pixels(qimage_to_numpy(QApplication.clipboard().image()))
            pixel = tuple(int(channel) for channel in window.session.doc.active_layer.pixels[0, 0])
            self.assertEqual(pixel, (1, 2, 3, 255))
        finally:
            window.session.history.mark_clean()
            window.close()
            window.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
