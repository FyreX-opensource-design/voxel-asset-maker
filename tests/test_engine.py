"""Tests for painting, scripts, compositing, and export."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voxel_asset_maker.model import Document, validate_size  # noqa: E402
from voxel_asset_maker.pixels import flood_fill  # noqa: E402
from voxel_asset_maker.processors import ScriptLibrary  # noqa: E402
from voxel_asset_maker.render import render_variant  # noqa: E402
from voxel_asset_maker.session import Session  # noqa: E402
from voxel_asset_maker.storage import export_variant_zip, load_project, save_project  # noqa: E402


class SizeTests(unittest.TestCase):
    def test_rejects_non_multiples_of_16(self) -> None:
        with self.assertRaises(ValueError):
            validate_size(24, 16)
        with self.assertRaises(ValueError):
            Document(15, 16)


class PaintTests(unittest.TestCase):
    def test_pencil_line_fill_and_selection_clip(self) -> None:
        session = Session()
        session.view_mode = "layer"
        session.fg = (10, 20, 30, 255)
        session.pointer_down(1, 1, 1)
        session.pointer_move(1, 4)
        session.pointer_up(1, 4)
        column = session.doc.active_layer.pixels[1:5, 1]
        self.assertTrue(np.all(column == (10, 20, 30, 255)))

        session.tool = "fill"
        session.fg = (1, 2, 3, 255)
        session.pointer_down(0, 0, 1)
        session.pointer_up(0, 0)
        self.assertEqual(tuple(session.doc.active_layer.pixels[0, 0]), (1, 2, 3, 255))
        self.assertEqual(tuple(session.doc.active_layer.pixels[1, 1]), (10, 20, 30, 255))

        session.undo()
        self.assertEqual(tuple(session.doc.active_layer.pixels[0, 0]), (0, 0, 0, 0))
        self.assertEqual(tuple(session.doc.active_layer.pixels[1, 1]), (10, 20, 30, 255))
        session.redo()
        self.assertEqual(tuple(session.doc.active_layer.pixels[0, 0]), (1, 2, 3, 255))

        session.tool = "pencil"
        session.fg = (9, 9, 9, 255)
        session.selection = (0, 0, 1, 1)
        session.pointer_down(5, 5, 1)
        session.pointer_up(5, 5)
        self.assertEqual(tuple(session.doc.active_layer.pixels[5, 5]), (1, 2, 3, 255))
        session.pointer_down(0, 0, 1)
        session.pointer_up(0, 0)
        self.assertEqual(tuple(session.doc.active_layer.pixels[0, 0]), (9, 9, 9, 255))

    def test_selection_limits_the_pencil(self) -> None:
        session = Session()
        session.view_mode = "layer"
        session.fg = (9, 9, 9, 255)
        session.selection = (2, 2, 2, 2)
        session.pointer_down(0, 0, 1)
        session.pointer_up(0, 0)
        session.pointer_down(2, 2, 1)
        session.pointer_up(2, 2)
        self.assertEqual(int(session.doc.active_layer.pixels[0, 0, 3]), 0)
        self.assertEqual(tuple(session.doc.active_layer.pixels[2, 2]), (9, 9, 9, 255))

    def test_flood_fill_stops_at_a_different_color(self) -> None:
        session = Session()
        pixels = session.doc.active_layer.pixels
        pixels[0, :] = (255, 0, 0, 255)
        flood_fill(pixels, 3, 3, (0, 0, 255, 255))
        self.assertEqual(tuple(pixels[3, 3]), (0, 0, 255, 255))
        self.assertEqual(tuple(pixels[0, 0]), (255, 0, 0, 255))


class CompositeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = ScriptLibrary()

    def test_layers_blend_and_honor_visibility_and_offset(self) -> None:
        doc = Document(16, 16)
        doc.layers[0].pixels[:, :] = (200, 0, 0, 255)
        session = Session(self.library)
        session.replace_document(doc, None)
        session.add_layer()
        top = session.doc.layers[1]
        top.pixels[:, :] = (0, 0, 200, 255)
        top.opacity = 0.5
        result = render_variant(session.doc, 1, self.library).image
        pixel = result.getpixel((0, 0))
        self.assertGreater(pixel[0], 70)
        self.assertGreater(pixel[2], 70)
        self.assertEqual(pixel[1], 0)
        self.assertEqual(pixel[3], 255)

        top.visible = False
        hidden = render_variant(session.doc, 1, self.library).image
        self.assertEqual(hidden.getpixel((0, 0))[:3], (200, 0, 0))

        top.visible = True
        top.opacity = 1
        top.pixels[:, :] = (0, 0, 0, 0)
        top.pixels[0, 0] = (0, 255, 0, 255)
        top.offset = (4, 1)
        shifted = render_variant(session.doc, 1, self.library).image
        self.assertEqual(shifted.getpixel((4, 1))[:3], (0, 255, 0))
        self.assertEqual(shifted.getpixel((0, 0))[:3], (200, 0, 0))

    def test_noise_starts_with_more_than_one_variant(self) -> None:
        session = Session(self.library)
        session.doc.layers[0].pixels[:, :] = (80, 140, 60, 255)
        session.set_script_id("noise")
        self.assertGreater(session.variant_count(), 1)

    def test_noise_variants_are_stable_and_different(self) -> None:
        session = Session(self.library)
        session.doc.layers[0].pixels[:, :] = (80, 140, 60, 255)
        session.doc.layers[0].script.script_id = "noise"
        session.doc.layers[0].script.params = {"count": 3, "sigma": 24, "seed": 7, "mode": "mono"}
        first = render_variant(session.doc, 1, self.library).image
        again = render_variant(session.doc, 1, self.library).image
        second = render_variant(session.doc, 2, self.library).image
        self.assertTrue(np.array_equal(np.array(first), np.array(again)))
        self.assertFalse(np.array_equal(np.array(first), np.array(second)))
        self.assertEqual(first.getpixel((0, 0))[3], 255)
        self.assertEqual(session.variant_count(), 3)

    def test_transparent_pixels_stay_transparent_under_noise(self) -> None:
        session = Session(self.library)
        session.doc.layers[0].pixels[0, 0] = (10, 20, 30, 0)
        session.doc.layers[0].script.script_id = "noise"
        session.doc.layers[0].script.params = {"count": 1, "sigma": 40, "seed": 1}
        image = render_variant(session.doc, 1, self.library).image
        self.assertEqual(image.getpixel((0, 0))[3], 0)

    def test_hue_steps_change_across_variants(self) -> None:
        session = Session(self.library)
        session.doc.layers[0].pixels[:, :] = (220, 20, 20, 255)
        session.doc.layers[0].script.script_id = "hsb"
        session.doc.layers[0].script.params = {"count": 3, "hue_step": 120, "hue_shift": 0}
        first = render_variant(session.doc, 1, self.library).image.getpixel((0, 0))
        second = render_variant(session.doc, 2, self.library).image.getpixel((0, 0))
        self.assertGreater(first[0], first[1])
        self.assertGreater(second[1], second[0])

    def test_edge_shade_darkens_the_outline_only(self) -> None:
        session = Session(self.library)
        pixels = session.doc.layers[0].pixels
        pixels[:, :] = (0, 0, 0, 0)
        pixels[2:14, 2:14] = (200, 200, 200, 255)
        session.doc.layers[0].script.script_id = "script:edge_shade"
        session.doc.layers[0].script.params = {"strength": 0.5, "passes": 1}
        image = render_variant(session.doc, 1, self.library).image
        self.assertEqual(image.getpixel((0, 0))[3], 0)
        self.assertLess(image.getpixel((2, 2))[0], 200)
        self.assertEqual(image.getpixel((8, 8))[0], 200)

    def test_variant_zip_and_project_roundtrip(self) -> None:
        session = Session(self.library)
        session.doc.name = "stone"
        session.doc.layers[0].pixels[:, :] = (90, 80, 70, 255)
        session.doc.palette = ["#112233"]
        session.set_script_id("noise")
        session.set_script_param("count", 3)
        session.set_script_param("sigma", 18)
        session.set_script_param("seed", 4)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            names, errors = export_variant_zip(session.doc, folder / "out.zip", self.library)
            self.assertEqual(errors, [])
            self.assertEqual(names, ["stone_1.png", "stone_2.png", "stone_3.png"])
            with zipfile.ZipFile(folder / "out.zip") as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["count"], 3)
                images = [Image.open(io.BytesIO(archive.read(name))) for name in names]
                self.assertEqual(images[0].size, (16, 16))
                self.assertFalse(np.array_equal(np.array(images[0]), np.array(images[1])))
            project = folder / "stone.vam"
            save_project(session.doc, project)
            loaded = load_project(project)
            self.assertEqual(loaded.name, "stone")
            self.assertEqual(loaded.layers[0].script.script_id, "noise")
            self.assertEqual(loaded.layers[0].script.params["seed"], 4)
            self.assertEqual(loaded.palette, ["#112233"])
            self.assertTrue(np.array_equal(loaded.layers[0].pixels, session.doc.layers[0].pixels))


class CliTests(unittest.TestCase):
    def test_noise_and_aliases_still_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pattern = str(Path(tmp) / "grass_{i}.png")
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/texture_noise.py",
                    "--color",
                    "#4a752c",
                    "--size",
                    "16x16",
                    "-n",
                    "2",
                    "--seed",
                    "3",
                    "-o",
                    pattern,
                    "-q",
                ],
                cwd=ROOT,
            )
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 2)
        subprocess.check_call([sys.executable, "scripts/texture_hsb.py", "--help"], cwd=ROOT, stdout=subprocess.DEVNULL)
        subprocess.check_call(
            [sys.executable, "scripts/texture_composite.py", "--help"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    unittest.main()
