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

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QEventPoint, QImage, QPointingDevice  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
)

from voxel_asset_maker.gui.dialogs import NewTextureDialog  # noqa: E402
from voxel_asset_maker.gui.images import numpy_to_qimage, qimage_to_numpy  # noqa: E402
from voxel_asset_maker.gui.window import MainWindow  # noqa: E402
from voxel_asset_maker.definitions import DefinitionLibrary  # noqa: E402
from voxel_asset_maker.model import BLOCK_SHEET  # noqa: E402


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
            window.session.set_script_id("palette")
            window.sync_all()
            self.app.processEvents()
            add = next(
                button
                for button in window.form_host.findChildren(QPushButton)
                if button.text() == "Add color"
            )
            before = window.session.current_binding().params["colors"].split()
            add.click()
            self.app.processEvents()
            after = window.session.current_binding().params["colors"].split()
            self.assertEqual(len(after), len(before) + 1)
        finally:
            window.session.history.mark_clean()
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_script_list_and_asset_preview(self) -> None:
        window = MainWindow()
        window.session.history.mark_clean()
        try:
            window.resize(1100, 820)
            window.show()
            self.app.processEvents()
            window.session.new_document(16, 16, name="copper_block", fill=(180, 90, 40, 255))
            window.session.add_script("hsb")
            window.session.add_script("noise")
            window.sync_all()
            self.app.processEvents()
            self.assertEqual(window.script_list.count(), 2)
            self.assertGreater(window.session.variant_count(), 1)
            self.assertIn("Texture", window.asset_caption.text())
            self.assertEqual(window.asset_kind.text(), "Texture")
            dialog = NewTextureDialog(kind="block")
            self.assertEqual(dialog.kind.currentData(), "block")
            self.assertEqual(dialog.name.text(), "block")
            self.assertEqual((dialog.width.value(), dialog.height.value()), BLOCK_SHEET)
            self.assertFalse(dialog.width.isEnabled())
            crop_dialog = NewTextureDialog(kind="crop")
            self.assertEqual(crop_dialog.kind.currentData(), "crop")
            self.assertEqual(crop_dialog.name.text(), "crop")
            self.assertEqual(crop_dialog.stages.value(), 4)
            self.assertEqual(crop_dialog.windowTitle(), "New crop")
            self.assertEqual((crop_dialog.width.value(), crop_dialog.height.value()), (64, 16))
            self.assertFalse(crop_dialog.width.isEnabled())
            file_menu = next(action.menu() for action in window.menuBar().actions() if action.menu() and action.text() == "File")
            new_menu = next(action.menu() for action in file_menu.actions() if action.menu() and action.text() == "New")
            self.assertIn("Texture…", [action.text() for action in new_menu.actions()])
            block_game = next(action.menu() for action in new_menu.actions() if action.menu() and action.text() == "Block game")
            assets_menu = next(action.menu() for action in block_game.actions() if action.menu() and action.text() == "Assets")
            self.assertEqual(
                [action.text() for action in assets_menu.actions()],
                ["Block…", "Item…", "Crop…", "Door…", "Trapdoor…"],
            )
            sets_menu = next(action.menu() for action in block_game.actions() if action.menu() and action.text() == "Sets")
            self.assertEqual([action.text() for action in sets_menu.actions()], ["Wood…"])
            minecraft = next(action.menu() for action in new_menu.actions() if action.menu() and action.text() == "Minecraft")
            self.assertEqual([action.text() for action in minecraft.actions()], ["Block…"])
            wood_dialog = NewTextureDialog(kind="set")
            self.assertEqual(wood_dialog.kind.currentData(), "set")
            self.assertEqual(wood_dialog.name.text(), "wood")
            self.assertEqual(wood_dialog.windowTitle(), "New wood set")
            self.assertFalse(wood_dialog.width.isEnabled())
            window.session.new_document(64, 16, name="wheat", kind="crop", stages=4)
            window.sync_all()
            self.app.processEvents()
            self.assertEqual(len(window.loot_rows), 2)
            self.assertEqual(window.loot_rows[0].item.text(), "")
            self.assertEqual(window.loot_rows[0].item.placeholderText(), "wheat")
            self.assertEqual(window.loot_rows[1].item.placeholderText(), "wheat_seeds")
            window.session.new_document(16, 16, name="copper_block", fill=(180, 90, 40, 255), kind="block")
            window.sync_all()
            self.app.processEvents()
            self.assertEqual(window.session.doc.kind, "block")
            self.assertEqual(window.asset_kind.text(), "Block")
            self.assertEqual(len(window.canvas.face_boxes), 6)
            self.assertEqual(window.face_mode.currentData(), "six")
            self.assertTrue(window.face_mode.isVisible())
            window.face_mode.setCurrentIndex(window.face_mode.findData("all"))
            self.app.processEvents()
            self.assertEqual([box[0] for box in window.canvas.face_boxes], ["all"])
            self.assertEqual((window.session.doc.width, window.session.doc.height), (16, 16))
            window.face_mode.setCurrentIndex(window.face_mode.findData("column"))
            self.app.processEvents()
            self.assertEqual([box[0] for box in window.canvas.face_boxes], ["top", "side", "bottom"])
            window.face_mode.setCurrentIndex(window.face_mode.findData("six"))
            self.app.processEvents()
            self.assertTrue(window.light_host.isVisible())
            self.assertEqual(tuple(window.light_spins), ("r", "g", "b", "uv", "ir"))
            window.light_spins["r"].setValue(14)
            window.light_spins["g"].setValue(11)
            window.light_spins["b"].setValue(4)
            window.light_spins["ir"].setValue(10)
            self.app.processEvents()
            self.assertEqual(window.session.doc.traits.light, (14, 11, 4, 0, 10))
            self.assertIn("ultraviolet 0", window.asset_caption.text())
            self.assertIn("infrared 10", window.asset_caption.text())
            for spin in window.light_spins.values():
                spin.setValue(0)
            self.app.processEvents()
            self.assertIn("south", window.asset_caption.text())
            pixmap = window.asset_preview.pixmap()
            self.assertIsNotNone(pixmap)
            assert pixmap is not None
            self.assertFalse(pixmap.isNull())
            self.assertGreater(pixmap.width(), 16)
            window.preview_right.click()
            self.app.processEvents()
            self.assertIn("west", window.asset_caption.text())
            window.session.new_document(16, 16, name="oak_door", kind="door")
            window.sync_all()
            self.app.processEvents()
            self.assertEqual(window.asset_kind.text(), "Door")
            self.assertEqual([box[0] for box in window.canvas.face_boxes], ["lower", "upper", "edge"])
            self.assertFalse(window.rotation_combo.isVisible())
            self.assertFalse(window.face_mode.isVisible())
            self.assertFalse(window.light_host.isVisible())
            window.session.new_document(16, 16, name="kiln", kind="block")
            window.sync_all()
            window.station_combo.setCurrentIndex(window.station_combo.findData("furnace"))
            self.app.processEvents()
            self.assertTrue(window.recipe_edit.isVisible())
            self.assertTrue(window.cook_spin.isVisible())
            self.assertEqual(window.session.doc.traits.station, "furnace")
            self.assertEqual(window.session.doc.traits.rotation, "horizontal")
            window.session.new_document(16, 16, name="pine", kind="set")
            window.sync_all()
            self.app.processEvents()
            self.assertEqual(window.asset_kind.text(), "Wood set")
            self.assertEqual(len(window.canvas.face_boxes), 11)
            self.assertTrue(window.set_member.isVisible())
            self.assertTrue(window.export_definition_button.isEnabled())
            self.assertFalse(window.block_extra.isVisible())
            self.assertIn("pine_log", window.asset_caption.text())
            preview = window.asset_preview.pixmap()
            self.assertIsNotNone(preview)
            assert preview is not None
            self.assertFalse(preview.isNull())
            window.set_member.setCurrentIndex(window.set_member.findData("stick"))
            self.app.processEvents()
            self.assertIn("pine_stick", window.asset_caption.text())
            self.assertFalse(window.preview_turn.isVisible())
            window.session.new_document(16, 16, name="copper_nugget", kind="item")
            window.sync_all()
            self.app.processEvents()
            self.assertIn("Item", window.asset_caption.text())
            help_menu = next(action.menu() for action in window.menuBar().actions() if action.menu() and action.text() == "Help")
            self.assertIn("Layer scripts", [action.text() for action in help_menu.actions()])
            self.assertIn("Custom definitions", [action.text() for action in help_menu.actions()])
            window.session.definitions = DefinitionLibrary(folders=[])
            window.session.definitions.load_file(ROOT / "examples" / "lantern.py")
            window._rebuild_definition_menus()
            menus = [action.text() for action in new_menu.actions() if action.menu()]
            self.assertIn("Lanterns", menus)
            window.session.new_document(16, 16, name="oak_lantern", definition="lantern", asset="lantern")
            window.sync_all()
            self.app.processEvents()
            self.assertEqual(window.asset_kind.text(), "Lanterns")
            self.assertEqual([box[0] for box in window.canvas.face_boxes], ["front", "top"])
            self.assertIn("oak_lantern", window.asset_caption.text())
            spin = window.definition_host.findChild(QSpinBox)
            self.assertIsNotNone(spin)
            assert spin is not None
            self.assertEqual(spin.value(), 12)
            spin.setValue(8)
            self.app.processEvents()
            self.assertEqual(window.session.doc.fields["light"], 8)
            window.session.add_script("hsb")
            window.sync_all()
            self.app.processEvents()
            front = window.form_host.findChild(QCheckBox, "part-front")
            top = window.form_host.findChild(QCheckBox, "part-top")
            self.assertIsNotNone(front)
            self.assertIsNotNone(top)
            assert front is not None
            front.setChecked(True)
            self.app.processEvents()
            self.assertEqual(window.session.current_binding().parts, ("front",))
            self.assertIn("front", window.script_list.item(0).text())
            item = window.asset_preview.pixmap()
            self.assertIsNotNone(item)
            assert item is not None
            self.assertFalse(item.isNull())
        finally:
            window.session.history.mark_clean()
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_minecraft_frames_grow_the_sheet(self) -> None:
        window = MainWindow()
        window.session.history.mark_clean()
        try:
            window.resize(1100, 820)
            window.show()
            self.app.processEvents()
            window.session.new_document(16, 16, name="glowstone", definition="minecraft", asset="block")
            window.sync_all()
            self.app.processEvents()
            self.assertEqual(window.asset_kind.text(), "Minecraft")
            self.assertTrue(window.preview_turn.isVisible())
            spin = _form_spin(window, "Frames")
            self.assertIsNotNone(spin)
            assert isinstance(spin, QSpinBox)
            self.assertEqual(spin.value(), 1)
            spin.setValue(3)
            self.app.processEvents()
            self.assertEqual((window.session.doc.width, window.session.doc.height), (16, 48))
            self.assertEqual([box[0] for box in window.canvas.face_boxes], ["1", "2", "3"])
            preview = window.asset_preview.pixmap()
            self.assertIsNotNone(preview)
            assert preview is not None
            self.assertFalse(preview.isNull())
        finally:
            window.session.history.mark_clean()
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_touch_stroke_skips_the_device_filter_and_the_preview(self) -> None:
        window = MainWindow()
        window.session.history.mark_clean()
        try:
            window.session.new_document(16, 16, name="swatch", kind="texture")
            window.sync_all()
            device = QPointingDevice()
            self.assertFalse(window.eventFilter(device, QEvent(QEvent.Type.TouchUpdate)))
            previews: list[int] = []
            window.refresh_preview = lambda: previews.append(1)  # type: ignore[method-assign]
            canvas = window.canvas
            canvas.pan_x = 0
            canvas.pan_y = 0
            canvas.zoom = 16

            class _Point:
                def __init__(self, state: QEventPoint.State, x: float) -> None:
                    self._state = state
                    self._pos = QPointF(x, 8)

                def id(self) -> int:
                    return 7

                def state(self) -> QEventPoint.State:
                    return self._state

                def position(self) -> QPointF:
                    return self._pos

            class _Touch:
                def __init__(self, kind: QEvent.Type, sample: _Point) -> None:
                    self._kind = kind
                    self._sample = sample

                def type(self) -> QEvent.Type:
                    return self._kind

                def points(self) -> list[_Point]:
                    return [self._sample]

                def accept(self) -> None:
                    return None

            begin = _Touch(QEvent.Type.TouchBegin, _Point(QEventPoint.State.Pressed, 8))
            move = _Touch(QEvent.Type.TouchUpdate, _Point(QEventPoint.State.Updated, 40))
            end = _Touch(QEvent.Type.TouchEnd, _Point(QEventPoint.State.Released, 40))
            self.assertTrue(canvas.event(begin))
            self.assertTrue(canvas.event(move))
            self.assertEqual(previews, [])
            pixel = tuple(int(channel) for channel in window.session.doc.active_layer.pixels[0, 2])
            self.assertEqual(pixel, (232, 232, 232, 255))
            self.assertTrue(canvas.event(end))
            self.assertEqual(previews, [1])
        finally:
            window._preview_timer.stop()
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


def _form_spin(window, label_text: str):
    form = window.definition_form
    for row in range(form.rowCount()):
        label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        field_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
        if label_item is None or field_item is None:
            continue
        label = label_item.widget()
        if label is not None and label.text() == label_text:
            return field_item.widget()
    return None


if __name__ == "__main__":
    unittest.main()
