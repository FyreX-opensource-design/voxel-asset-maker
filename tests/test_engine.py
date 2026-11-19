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

from voxel_asset_maker.blockgame import (  # noqa: E402
    PublishError,
    block_preview,
    block_view,
    cube_preview,
    export_definition,
    slice_faces,
)
from voxel_asset_maker.definitions import DefinitionError, DefinitionLibrary, export_custom  # noqa: E402
from voxel_asset_maker.model import (  # noqa: E402
    BLOCK_FACE_ORIGIN,
    BLOCK_SHEET,
    WOOD_SHEET,
    Document,
    ctm_label,
    ctm_origin,
    empty_pixels,
    read_ctm,
    read_face,
    read_stage,
    validate_size,
    write_face,
)
from voxel_asset_maker.pixels import flood_fill  # noqa: E402
from voxel_asset_maker.processors import ScriptLibrary  # noqa: E402
from voxel_asset_maker.render import render_variant  # noqa: E402
from voxel_asset_maker.session import Session  # noqa: E402
from voxel_asset_maker.storage import export_variant_zip, load_project, png_bytes, save_project  # noqa: E402


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

    def test_palette_noise_picks_from_the_given_colors(self) -> None:
        session = Session(self.library)
        session.new_document(16, 16, name="stone", fill=(255, 0, 0, 255))
        session.doc.layers[0].pixels[0, 0] = (255, 0, 0, 0)
        session.set_script_id("palette")
        session.set_script_param("colors", "#000000 #ffffff")
        session.set_script_param("spread", 0)
        session.set_script_param("seed", 3)
        session.set_script_param("count", 2)
        first = render_variant(session.doc, 1, self.library)
        again = render_variant(session.doc, 1, self.library).image
        second = render_variant(session.doc, 2, self.library).image
        self.assertEqual(first.errors, [])
        self.assertTrue(np.array_equal(np.array(first.image), np.array(again)))
        self.assertFalse(np.array_equal(np.array(first.image), np.array(second)))
        colors = {first.image.getpixel((x, y))[:3] for x in range(16) for y in range(16) if not (x == 0 and y == 0)}
        self.assertEqual(colors, {(0, 0, 0), (255, 255, 255)})
        self.assertEqual(first.image.getpixel((0, 0))[3], 0)
        session.set_script_param("spread", 12)
        drifted = np.array(render_variant(session.doc, 1, self.library).image)
        opaque = drifted[1:, :, :3].reshape(-1, 3)
        self.assertTrue(np.all(opaque[:, 0] == opaque[:, 1]))
        self.assertTrue(np.all(opaque[:, 1] == opaque[:, 2]))
        distance = np.minimum(opaque[:, 0], 255 - opaque[:, 0])
        self.assertLessEqual(int(distance.max()), 12)
        self.assertGreater(int(distance.max()), 0)
        session.set_script_param("colors", "")
        missed = render_variant(session.doc, 1, self.library)
        self.assertTrue(any("Palette noise" in line for line in missed.errors))
        self.assertEqual(missed.image.getpixel((4, 4))[:3], (255, 0, 0))
        wood = Session(self.library)
        wood.new_document(16, 16, name="pine", kind="set", fill=(20, 20, 20, 255))
        wood.set_script_id("palette")
        wood.set_script_param("colors", "#000000 #ffffff")
        wood.set_script_param("spread", 0)
        wood.set_script_param("seed", 5)
        wood.set_script_parts(["log_top", "log_side"])
        sheet = np.array(render_variant(wood.doc, 1, self.library).image)
        self.assertFalse(np.array_equal(sheet[:, 0:16], sheet[:, 16:32]))
        self.assertEqual(tuple(int(channel) for channel in sheet[4, 40, :3]), (20, 20, 20))

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

    def test_scripts_run_in_order_and_roundtrip(self) -> None:
        session = Session(self.library)
        session.doc.layers[0].pixels[:, :] = (220, 20, 20, 255)
        session.set_script_id("hsb")
        session.set_script_param("hue_shift", 120)
        session.set_script_param("count", 2)
        session.add_script("hsb")
        session.set_script_param("hue_shift", 120)
        session.set_script_param("count", 5)
        self.assertEqual(session.variant_count(), 5)
        self.assertEqual(
            [binding.script_id for binding in session.doc.layers[0].scripts],
            ["hsb", "hsb"],
        )
        pixel = render_variant(session.doc, 1, self.library).image.getpixel((0, 0))
        self.assertGreater(pixel[2], pixel[0])
        self.assertGreater(pixel[2], pixel[1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.vam"
            save_project(session.doc, path)
            loaded = load_project(path)
            self.assertEqual(len(loaded.layers[0].scripts), 2)
            self.assertEqual(loaded.layers[0].scripts[1].params["hue_shift"], 120)

    def test_legacy_project_keeps_its_single_script(self) -> None:
        pixels = np.zeros((16, 16, 4), dtype=np.uint8)
        pixels[:, :] = (9, 8, 7, 255)
        payload = {
            "format": 1,
            "name": "old",
            "width": 16,
            "height": 16,
            "active": 0,
            "layers": [
                {
                    "id": "base",
                    "name": "base",
                    "script": {"id": "hsb", "path": "", "params": {"hue_shift": 15}},
                    "file": "layers/0.png",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.vam"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("project.json", json.dumps(payload))
                archive.writestr("layers/0.png", png_bytes(pixels))
            loaded = load_project(path)
        self.assertEqual(loaded.layers[0].scripts[0].script_id, "hsb")
        self.assertEqual(loaded.layers[0].scripts[0].params["hue_shift"], 15)


class BlockGameTests(unittest.TestCase):
    def test_strip_splits_into_three_faces(self) -> None:
        image = np.zeros((48, 16, 4), dtype=np.uint8)
        image[0:16, :, 0] = 255
        image[16:32, :, 1] = 255
        image[32:48, :, 2] = 255
        image[..., 3] = 255
        faces = slice_faces(image)
        self.assertFalse(faces.single)
        self.assertEqual(int(faces.top[0, 0, 0]), 255)
        self.assertEqual(int(faces.side[0, 0, 1]), 255)
        self.assertEqual(int(faces.bottom[0, 0, 2]), 255)

    def test_cube_preview_puts_the_top_above_the_sides(self) -> None:
        top = np.zeros((16, 16, 4), dtype=np.uint8)
        top[..., 0] = 220
        top[..., 3] = 255
        side = np.zeros((16, 16, 4), dtype=np.uint8)
        side[..., 1] = 200
        side[..., 3] = 255
        preview = cube_preview(top, side, scale=2)
        red = np.argwhere((preview[..., 0] > 180) & (preview[..., 1] < 40))
        green = np.argwhere((preview[..., 1] > 100) & (preview[..., 0] < 40))
        self.assertGreater(len(red), 40)
        self.assertGreater(len(green), 40)
        self.assertLess(float(red[:, 0].mean()), float(green[:, 0].mean()))
        self.assertEqual(int(preview[0, 0, 3]), 255)
        edge = cube_preview(top, side, scale=4)
        solid = edge[..., :3].max(axis=2) > 90
        crest = []
        for column in range(edge.shape[1]):
            rows = np.nonzero(solid[:, column])[0]
            if len(rows):
                crest.append(int(rows[0]))
        steps = np.abs(np.diff(crest))
        self.assertGreater(len(crest), 20)
        self.assertLessEqual(int(steps.max()), 2)

    def test_preview_turns_to_each_side_and_the_underside(self) -> None:
        def solid(red: int, green: int, blue: int) -> np.ndarray:
            tile = np.zeros((16, 16, 4), dtype=np.uint8)
            tile[..., 0] = red
            tile[..., 1] = green
            tile[..., 2] = blue
            tile[..., 3] = 255
            return tile

        south = solid(0, 0, 255)
        east = solid(255, 0, 0)
        top = solid(0, 255, 0)
        west = solid(255, 255, 0)
        north = solid(255, 0, 255)
        bottom = solid(0, 255, 255)

        def center(preview: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
            ys, xs = np.nonzero(mask)
            self.assertGreater(len(xs), 20)
            return float(xs.mean()), float(ys.mean())

        front = cube_preview(
            top, north, left=south, right=east, bottom=bottom, north=north, west=west, scale=1
        )
        blue_x, blue_y = center(front, (front[..., 2] > 160) & (front[..., 0] < 50) & (front[..., 1] < 50))
        red_x, red_y = center(front, (front[..., 0] > 140) & (front[..., 1] < 50) & (front[..., 2] < 50))
        _green_x, green_y = center(front, (front[..., 1] > 200) & (front[..., 0] < 40) & (front[..., 2] < 40))
        self.assertLess(blue_x, red_x)
        self.assertLess(green_y, blue_y)
        self.assertLess(green_y, red_y)
        self.assertEqual(block_view(0, 0), ("top", "south", "east"))

        turned = cube_preview(
            top, north, left=south, right=east, bottom=bottom, north=north, west=west, scale=1, yaw=1
        )
        yellow_x, _yellow_y = center(
            turned, (turned[..., 0] > 180) & (turned[..., 1] > 180) & (turned[..., 2] < 40)
        )
        moved_x, _moved_y = center(turned, (turned[..., 2] > 140) & (turned[..., 0] < 50) & (turned[..., 1] < 50))
        self.assertLess(yellow_x, moved_x)
        self.assertEqual(block_view(1, 0), ("top", "west", "south"))

        under = cube_preview(
            top, north, left=south, right=east, bottom=bottom, north=north, west=west, scale=1, pitch=1
        )
        green = (under[..., 1] > 200) & (under[..., 0] < 40) & (under[..., 2] < 40)
        self.assertEqual(int(np.count_nonzero(green)), 0)
        cyan_x, cyan_y = center(under, (under[..., 1] > 120) & (under[..., 2] > 120) & (under[..., 0] < 40))
        side_x, side_y = center(under, (under[..., 2] > 140) & (under[..., 0] < 50) & (under[..., 1] < 50))
        self.assertLess(cyan_y, side_y)
        del cyan_x, side_x
        self.assertEqual(block_view(0, 1), ("bottom", "south", "east"))

    def test_preview_shows_connected_tiles_and_glow(self) -> None:
        def solid(red: int, green: int, blue: int) -> np.ndarray:
            tile = np.zeros((16, 16, 4), dtype=np.uint8)
            tile[..., 0] = red
            tile[..., 1] = green
            tile[..., 2] = blue
            tile[..., 3] = 255
            return tile

        faces = np.zeros((48, 64, 4), dtype=np.uint8)
        for name in BLOCK_FACE_ORIGIN:
            origin_x, origin_y = BLOCK_FACE_ORIGIN[name]
            faces[origin_y : origin_y + 16, origin_x : origin_x + 16] = solid(20, 20, 20)
        ctm = np.zeros((64, 64, 4), dtype=np.uint8)
        left_x, left_y = ctm_origin(4)
        right_x, right_y = ctm_origin(8)
        ctm[left_y : left_y + 16, left_x : left_x + 16] = solid(0, 0, 255)
        ctm[right_y : right_y + 16, right_x : right_x + 16] = solid(255, 0, 0)
        preview = block_preview(faces, ctm=ctm, scale=1)
        red = (preview[..., 0] > 200) & (preview[..., 2] < 40)
        blue = (preview[..., 2] > 200) & (preview[..., 0] < 40)
        self.assertGreater(int(np.count_nonzero(red)), 20)
        self.assertGreater(int(np.count_nonzero(blue)), 20)
        red_x = float(np.argwhere(red)[:, 1].min())
        blue_x = float(np.argwhere(blue)[:, 1].max())
        self.assertLess(red_x, blue_x)

        glow = solid(0, 255, 80)
        plain = block_preview(faces, scale=1)
        glowing = block_preview(faces, glow=glow, glow_strength=1.0, scale=1)
        plain_green = int(np.count_nonzero(plain[..., 1] > 150))
        glow_green = int(np.count_nonzero(glowing[..., 1] > 150))
        self.assertEqual(plain_green, 0)
        self.assertGreater(glow_green, 40)
        colored = block_preview(faces, glow_color="#00ff55", glow_strength=1.0, scale=1)
        self.assertGreater(int(np.count_nonzero(colored[..., 1] > 150)), 40)

    def test_each_block_face_paints_and_exports_on_its_own(self) -> None:
        library = ScriptLibrary()
        session = Session(library)
        session.new_document(16, 16, name="Copper Brick", kind="block", fill=(180, 90, 40, 255))
        self.assertEqual((session.doc.width, session.doc.height), (64, 48))
        sx, sy = BLOCK_FACE_ORIGIN["south"]
        session.fg = (0, 0, 255, 255)
        session.pointer_down(sx + 2, sy + 3, 1)
        session.pointer_up(sx + 2, sy + 3)
        self.assertEqual(tuple(int(channel) for channel in session.doc.layers[0].pixels[sy + 3, sx + 2]), (0, 0, 255, 255))
        nx, ny = BLOCK_FACE_ORIGIN["north"]
        self.assertEqual(tuple(int(channel) for channel in session.doc.layers[0].pixels[ny + 3, nx + 2])[:3], (180, 90, 40))
        session.pointer_down(1, 1, 1)
        session.pointer_up(1, 1)
        self.assertEqual(int(session.doc.layers[0].pixels[1, 1, 3]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plain = Session(library)
            plain.new_document(16, 16, name="texture")
            with self.assertRaises(PublishError):
                export_definition(plain.doc, library, folder / "nope.zip")
            exported = export_definition(session.doc, library, folder / "copper.zip")
            self.assertEqual(exported.errors, [])
            self.assertFalse((folder / "assets").exists())
            with zipfile.ZipFile(exported.path) as archive:
                names = archive.namelist()
                self.assertIn("copper_brick.json", names)
                self.assertFalse(any(name.startswith("assets/") or name.endswith("en.json") for name in names))
                block = json.loads(archive.read("copper_brick.json"))
                self.assertEqual(block["model"], "cube")
                self.assertIn("south", block["textures"])
                south = Image.open(io.BytesIO(archive.read(block["textures"]["south"] + ".png")))
                self.assertEqual(south.getpixel((2, 3))[:3], (0, 0, 255))
                north = Image.open(io.BytesIO(archive.read(block["textures"]["north"] + ".png")))
                self.assertEqual(north.getpixel((2, 3))[:3], (180, 90, 40))
            session.set_block_trait(faces="all")
            self.assertEqual((session.doc.width, session.doc.height), (16, 16))
            one = export_definition(session.doc, library, folder / "one.zip")
            with zipfile.ZipFile(one.path) as archive:
                one_block = json.loads(archive.read("copper_brick.json"))
                self.assertEqual(set(one_block["textures"].values()), {one_block["textures"]["top"]})
                self.assertEqual(one_block["textures"]["top"], one_block["textures"]["side"])
            session.set_block_trait(faces="column")
            self.assertEqual((session.doc.width, session.doc.height), (48, 16))
            session.doc.layers[0].pixels[:, 16:32] = (1, 2, 3, 255)
            column = export_definition(session.doc, library, folder / "column.zip")
            with zipfile.ZipFile(column.path) as archive:
                column_block = json.loads(archive.read("copper_brick.json"))
                self.assertEqual(set(column_block["textures"]), {"top", "side", "bottom"})
                side = Image.open(io.BytesIO(archive.read(column_block["textures"]["side"] + ".png")))
                self.assertEqual(side.getpixel((4, 4))[:3], (1, 2, 3))
                self.assertNotEqual(column_block["textures"]["top"], column_block["textures"]["side"])
            session.set_block_trait(faces="six")
            self.assertEqual(tuple(int(channel) for channel in read_face(session.doc.layers[0].pixels, "north")[0, 0])[:3], (1, 2, 3))
            self.assertEqual(tuple(int(channel) for channel in read_face(session.doc.layers[0].pixels, "east")[0, 0])[:3], (1, 2, 3))
            save_project(session.doc, folder / "copper.vam")
            loaded = load_project(folder / "copper.vam")
            self.assertEqual(loaded.traits.faces, "six")
            session.new_document(16, 16, name="copper_nugget", kind="item", fill=(200, 120, 40, 255))
            item = export_definition(session.doc, library, folder / "nugget.zip")
            with zipfile.ZipFile(item.path) as archive:
                payload = json.loads(archive.read("copper_nugget.json"))
                self.assertEqual(payload["model"], "flat")
                self.assertEqual(payload["texture"], "copper_nugget")
                self.assertIn("copper_nugget.png", archive.namelist())
            project = folder / "brick.vam"
            session.set_kind("block")
            save_project(session.doc, project)
            loaded = load_project(project)
            self.assertEqual(loaded.kind, "block")
            self.assertEqual((loaded.width, loaded.height), (64, 48))

    def test_ctm_and_fluorescence_paint_and_export(self) -> None:
        library = ScriptLibrary()
        session = Session(library)
        session.new_document(16, 16, name="Amber Glass", kind="block", fill=(40, 80, 60, 255))
        session.set_ctm(True)
        self.assertEqual(session.surface, "ctm")
        self.assertEqual(ctm_label(0), "alone")
        self.assertEqual(ctm_label(1 | 4), "up+left")
        session.fg = (10, 20, 240, 255)
        session.pointer_down(2, 3, 1)
        session.pointer_up(2, 3)
        self.assertEqual(tuple(int(channel) for channel in session.doc.ctm_pixels[3, 2]), (10, 20, 240, 255))
        self.assertEqual(int(session.doc.ctm_pixels[3, 16, 3]), 0)
        session.undo()
        self.assertEqual(int(session.doc.ctm_pixels[3, 2, 3]), 0)
        session.redo()
        session.set_fluor_enabled(True)
        self.assertEqual(session.surface, "glow")
        session.fg = (180, 255, 40, 255)
        session.pointer_down(4, 5, 1)
        session.pointer_up(4, 5)
        session.set_fluor_range(4, 10)
        session.set_fluor_strength(1.0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exported = export_definition(session.doc, library, folder / "amber.zip")
            with zipfile.ZipFile(exported.path) as archive:
                block = json.loads(archive.read("amber_glass.json"))
                self.assertTrue(block["ctm"])
                self.assertEqual(block["fluorescence"]["start"], 4)
                self.assertEqual(block["fluorescence"]["full"], 10)
                self.assertEqual(block["fluorescence"]["texture"], "amber_glass_glow")
                self.assertIn("ctm_0", block["textures"])
                self.assertIn("ctm_15", block["textures"])
                alone = Image.open(io.BytesIO(archive.read(block["textures"]["ctm_0"] + ".png")))
                self.assertEqual(alone.getpixel((2, 3))[:3], (10, 20, 240))
                glow = Image.open(io.BytesIO(archive.read("amber_glass_glow.png")))
                self.assertEqual(glow.getpixel((4, 5))[:3], (180, 255, 40))
            session.set_fluor_mode("color")
            session.set_fluor_color("#112233")
            colored = export_definition(session.doc, library, folder / "amber_color.zip")
            with zipfile.ZipFile(colored.path) as archive:
                block = json.loads(archive.read("amber_glass.json"))
                self.assertEqual(block["fluorescence"]["color"], "#112233")
                self.assertNotIn("texture", block["fluorescence"])
            project = folder / "amber.vam"
            save_project(session.doc, project)
            loaded = load_project(project)
            self.assertTrue(loaded.ctm)
            self.assertEqual(tuple(int(channel) for channel in read_ctm(loaded.ctm_pixels, 0)[3, 2]), (10, 20, 240, 255))
            self.assertTrue(loaded.fluor.enabled)
            self.assertEqual(loaded.fluor.mode, "color")
            self.assertEqual(loaded.fluor.color, "#112233")
            self.assertEqual(loaded.fluor.start, 4)

    def test_shapes_preview_and_block_attributes(self) -> None:
        def solid(red: int, green: int, blue: int) -> np.ndarray:
            tile = np.zeros((16, 16, 4), dtype=np.uint8)
            tile[..., 0] = red
            tile[..., 1] = green
            tile[..., 2] = blue
            tile[..., 3] = 255
            return tile

        sheet = empty_pixels(*BLOCK_SHEET)
        write_face(sheet, "top", solid(0, 220, 0))
        write_face(sheet, "bottom", solid(0, 220, 220))
        write_face(sheet, "south", solid(0, 0, 220))
        write_face(sheet, "east", solid(220, 0, 0))
        write_face(sheet, "north", solid(220, 0, 220))
        write_face(sheet, "west", solid(220, 220, 0))

        def green_rows(shape: str) -> np.ndarray:
            preview = block_preview(sheet, shape=shape, scale=1)
            mask = (preview[..., 1] > 180) & (preview[..., 0] < 50) & (preview[..., 2] < 50)
            rows = np.nonzero(mask)[0]
            self.assertGreater(len(rows), 8)
            return rows

        cube_rows = green_rows("cube")
        slab_rows = green_rows("slab")
        stair_rows = green_rows("stairs")
        self.assertGreater(float(slab_rows.mean()), float(cube_rows.mean()) + 4)
        self.assertGreater(int(stair_rows.max() - stair_rows.min()), int(slab_rows.max() - slab_rows.min()))
        vertical = block_preview(sheet, shape="vertical_slab", scale=1)
        east = (vertical[..., 0] > 140) & (vertical[..., 1] < 40) & (vertical[..., 2] < 40)
        cube = block_preview(sheet, shape="cube", scale=1)
        cube_east = (cube[..., 0] > 140) & (cube[..., 1] < 40) & (cube[..., 2] < 40)
        self.assertLess(int(np.count_nonzero(east)), int(np.count_nonzero(cube_east)))
        fence = block_preview(sheet, shape="fence", scale=1)
        wall = block_preview(sheet, shape="wall", scale=1)
        colored = lambda image: int(np.count_nonzero(image[..., :3].max(axis=2) > 80))
        self.assertLess(colored(fence), colored(cube) // 2)
        self.assertGreater(colored(wall), colored(fence))

        library = ScriptLibrary()
        session = Session(library)
        session.new_document(16, 16, name="Oak", kind="block")
        session.begin_edit()
        session.set_block_trait(
            shape="vertical_slab",
            tab="natural",
            hardness=1.5,
            tool="axe",
            mining_level=1,
            sound="wood",
            render="cutout",
            solid=False,
            gravity=True,
            light=7,
            full_block="Oak Planks",
        )
        self.assertTrue(session.end_edit())
        self.assertEqual(session.doc.traits.full_block, "oak_planks")
        session.undo()
        self.assertEqual(session.doc.traits.shape, "cube")
        session.redo()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exported = export_definition(session.doc, library, folder / "oak.zip")
            with zipfile.ZipFile(exported.path) as archive:
                block = json.loads(archive.read("oak.json"))
            self.assertEqual(block["type"], "vertical_slab")
            self.assertEqual(block["model"], "vertical_slab")
            self.assertEqual(block["tab"], "natural")
            self.assertEqual(block["hardness"], 1.5)
            self.assertEqual(block["tool"], "axe")
            self.assertEqual(block["mining_level"], 1)
            self.assertEqual(block["sound"], "wood")
            self.assertEqual(block["render"], "cutout")
            self.assertFalse(block["solid"])
            self.assertTrue(block["gravity"])
            self.assertEqual(block["light"], 7)
            self.assertEqual(block["full_block"], "oak_planks")
            session.set_block_trait(shape="fence", solid=True, gravity=False, light=0, render="solid")
            fenced = export_definition(session.doc, library, folder / "fence.zip")
            with zipfile.ZipFile(fenced.path) as archive:
                fence_block = json.loads(archive.read("oak.json"))
            self.assertEqual(fence_block["type"], "fence")
            self.assertEqual(fence_block["model"], "fence_post")
            self.assertTrue(fence_block["properties"]["fence"])
            self.assertIn("texture", fence_block["textures"])
            self.assertNotIn("solid", fence_block)
            project = folder / "oak.vam"
            save_project(session.doc, project)
            loaded = load_project(project)
            self.assertEqual(loaded.traits.shape, "fence")
            self.assertEqual(loaded.traits.sound, "wood")
            self.assertEqual(loaded.traits.full_block, "oak_planks")

    def test_plant_cross_and_arbitrary_sound(self) -> None:
        tile = np.zeros((16, 16, 4), dtype=np.uint8)
        tile[:, :8, 0] = 220
        tile[:, 8:, 2] = 220
        tile[..., 3] = 255
        tile[0, 0, 3] = 0
        sheet = empty_pixels(*BLOCK_SHEET)
        write_face(sheet, "south", tile)
        preview = block_preview(sheet, shape="cross", scale=1)
        red = (preview[..., 0] > 160) & (preview[..., 1] < 50) & (preview[..., 2] < 50)
        blue = (preview[..., 2] > 160) & (preview[..., 0] < 50) & (preview[..., 1] < 50)
        self.assertGreater(int(np.count_nonzero(red)), 20)
        self.assertGreater(int(np.count_nonzero(blue)), 20)
        self.assertLess(int(preview[0, 0, :3].max()), 80)
        turned = block_preview(sheet, shape="cross", scale=1, yaw=1)
        self.assertGreater(int(np.count_nonzero((turned[..., 0] > 160) & (turned[..., 2] < 50))), 20)

        library = ScriptLibrary()
        session = Session(library)
        session.new_document(16, 16, name="Fern", kind="block")
        session.set_block_trait(shape="cross", sound="Custom Metal", solid=True)
        self.assertEqual(session.doc.traits.sound, "custom_metal")
        session.fg = (9, 8, 7, 255)
        session.pointer_down(16, 16, 1)
        session.pointer_up(16, 16)
        self.assertEqual(int(session.doc.layers[0].pixels[16, 16, 3]), 0)
        session.pointer_down(48, 16, 1)
        session.pointer_up(48, 16)
        self.assertEqual(tuple(int(channel) for channel in session.doc.layers[0].pixels[16, 48]), (9, 8, 7, 255))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exported = export_definition(session.doc, library, folder / "fern.zip")
            with zipfile.ZipFile(exported.path) as archive:
                block = json.loads(archive.read("fern.json"))
            self.assertEqual(block["model"], "cross")
            self.assertNotIn("type", block)
            self.assertTrue(block["plant"])
            self.assertTrue(block["fluid_breakable"])
            self.assertEqual(block["render"], "cutout")
            self.assertTrue(block["solid"])
            self.assertEqual(block["sound"], "custom_metal")
            self.assertIn("cross", block["textures"])
            save_project(session.doc, folder / "fern.vam")
            loaded = load_project(folder / "fern.vam")
            self.assertEqual(loaded.traits.shape, "cross")
            self.assertEqual(loaded.traits.sound, "custom_metal")

    def test_crop_definition_and_arbitrary_tabs(self) -> None:
        library = ScriptLibrary()
        session = Session(library)
        session.new_document(16, 16, name="Wheat", kind="crop", stages=3)
        self.assertEqual((session.doc.width, session.doc.height), (48, 16))
        session.set_crop(
            ground="farmland, lichen_farmland",
            seed_tab="Magic Crops",
            produce_tab="pantry",
            loot=(
                {"item": "Glow Berry", "min": 2, "max": 2, "chance": 1},
                {"item": "wheat_seeds", "min": 1, "max": 3, "chance": 0.25},
            ),
            min_time=200,
            max_time=120,
        )
        self.assertEqual(session.doc.crop.loot[0].item, "glow_berry")
        self.assertEqual(session.doc.crop.loot[0].minimum, 2)
        self.assertEqual(session.doc.crop.loot[1].chance, 0.25)
        self.assertEqual(session.doc.crop.ground, "farmland lichen_farmland")
        self.assertEqual(session.doc.crop.seed_tab, "magic_crops")
        self.assertEqual(session.doc.crop.min_time, 120.0)
        self.assertEqual(session.doc.crop.max_time, 200.0)
        session.fg = (10, 180, 20, 255)
        session.pointer_down(20, 4, 1)
        session.pointer_up(20, 4)
        self.assertEqual(session.stage_index, 1)
        self.assertEqual(tuple(int(channel) for channel in session.doc.layers[0].pixels[4, 20]), (10, 180, 20, 255))
        self.assertEqual(int(session.doc.layers[0].pixels[4, 4, 3]), 0)
        session.pointer_down(60, 4, 1)
        session.pointer_up(60, 4)
        self.assertEqual(int(session.doc.layers[0].pixels[4, 4, 3]), 0)
        session.begin_edit()
        session.set_crop(stages=5)
        self.assertTrue(session.end_edit())
        self.assertEqual(session.doc.width, 80)
        self.assertEqual(tuple(int(channel) for channel in session.doc.layers[0].pixels[4, 20]), (10, 180, 20, 255))
        session.undo()
        self.assertEqual(session.doc.crop.stages, 3)
        self.assertEqual(session.doc.width, 48)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exported = export_definition(session.doc, library, folder / "wheat.zip")
            with zipfile.ZipFile(exported.path) as archive:
                crop = json.loads(archive.read("wheat.json"))
                stage = Image.open(io.BytesIO(archive.read("wheat_1.png")))
                seeds = json.loads(archive.read("wheat_seeds.json"))
                produce = json.loads(archive.read("wheat.item.json"))
                ripe = json.loads(archive.read("loot/crops/wheat_ripe.json"))
                young = json.loads(archive.read("loot/crops/wheat_young.json"))
            self.assertEqual(crop["stages"], 3)
            self.assertEqual(crop["ground"], ["farmland", "lichen_farmland"])
            self.assertEqual(crop["seed_tab"], "magic_crops")
            self.assertEqual(crop["produce_tab"], "pantry")
            self.assertNotIn("food", crop)
            self.assertEqual(crop["loot_table"], "crops/wheat_ripe")
            self.assertEqual(crop["seeds"], "wheat_seeds")
            self.assertEqual(crop["produce"], "wheat")
            self.assertTrue(crop["needs_water"])
            self.assertEqual(stage.getpixel((4, 4))[:3], (10, 180, 20))
            self.assertEqual(seeds["tab"], "magic_crops")
            self.assertEqual(seeds["block"], "wheat_0")
            self.assertEqual(produce["tab"], "pantry")
            self.assertEqual(produce["food"], 0)
            self.assertEqual(produce["right"], "none")
            self.assertEqual(
                ripe["pools"][0]["entries"],
                [
                    {"type": "item", "name": "glow_berry", "count": 2},
                    {"type": "item", "name": "wheat_seeds", "count": [1, 3], "chance": 0.25},
                ],
            )
            self.assertEqual(
                young["pools"][0]["entries"],
                [{"type": "item", "name": "wheat_seeds", "count": 1}],
            )
            save_project(session.doc, folder / "wheat.vam")
            loaded = load_project(folder / "wheat.vam")
            self.assertEqual(loaded.kind, "crop")
            self.assertEqual(loaded.crop.stages, 3)
            self.assertEqual(loaded.crop.produce_tab, "pantry")
            self.assertEqual(loaded.crop.loot[0].item, "glow_berry")
            self.assertEqual(loaded.crop.loot[1].maximum, 3)
            self.assertEqual(tuple(int(channel) for channel in read_stage(loaded.layers[0].pixels, 1)[4, 4]), (10, 180, 20, 255))
        session.new_document(16, 16, name="shelf", kind="block")
        session.set_block_trait(tab="Odd Shelves")
        with tempfile.TemporaryDirectory() as tmp:
            exported = export_definition(session.doc, library, Path(tmp) / "shelf.zip")
            with zipfile.ZipFile(exported.path) as archive:
                block = json.loads(archive.read("shelf.json"))
            self.assertEqual(block["tab"], "odd_shelves")
        session.new_document(16, 16, name="spoon", kind="item")
        session.set_block_trait(tab="tools")
        with tempfile.TemporaryDirectory() as tmp:
            exported = export_definition(session.doc, library, Path(tmp) / "spoon.zip")
            with zipfile.ZipFile(exported.path) as archive:
                item = json.loads(archive.read("spoon.json"))
            self.assertEqual(item["tab"], "tools")

    def test_rotation_station_door_and_trapdoor(self) -> None:
        library = ScriptLibrary()
        session = Session(library)
        session.new_document(16, 16, name="Kiln", kind="block")
        session.set_block_trait(
            rotation="horizontal",
            station="furnace",
            recipe_types="Kilning, blasting",
            cook_speed=2,
        )
        self.assertEqual(session.doc.traits.rotation, "horizontal")
        self.assertEqual(session.doc.traits.recipe_types, "kilning blasting")
        self.assertEqual(session.doc.traits.cook_speed, 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exported = export_definition(session.doc, library, folder / "kiln.zip")
            with zipfile.ZipFile(exported.path) as archive:
                block = json.loads(archive.read("kiln.json"))
            self.assertEqual(block["properties"]["rotation"], "horizontal")
            self.assertEqual(block["station"], "furnace")
            self.assertEqual(block["recipe_types"], ["kilning", "blasting"])
            self.assertEqual(block["cook_speed"], 2)
            save_project(session.doc, folder / "kiln.vam")
            loaded = load_project(folder / "kiln.vam")
            self.assertEqual(loaded.traits.station, "furnace")
            self.assertEqual(loaded.traits.recipe_types, "kilning blasting")

        session.new_document(16, 16, name="Oak Door", kind="door")
        self.assertEqual((session.doc.width, session.doc.height), (48, 16))
        self.assertEqual(session.doc.traits.tool, "axe")
        self.assertEqual(session.doc.traits.render, "cutout")
        self.assertEqual(session.doc.traits.sound, "wood")
        session.fg = (20, 140, 40, 255)
        session.pointer_down(4, 4, 1)
        session.pointer_up(4, 4)
        session.fg = (180, 40, 20, 255)
        session.pointer_down(20, 4, 1)
        session.pointer_up(20, 4)
        self.assertEqual(int(session.doc.layers[0].pixels[4, 36, 3]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exported = export_definition(session.doc, library, folder / "oak_door.zip")
            with zipfile.ZipFile(exported.path) as archive:
                block = json.loads(archive.read("oak_door.json"))
                behavior = json.loads(archive.read("doors/oak_door.json"))
                item = json.loads(archive.read("items/oak_door.json"))
                loot = json.loads(archive.read("loot/blocks/oak_door.json"))
                lower = Image.open(io.BytesIO(archive.read(f"{block['textures']['lower']}.png")))
                upper = Image.open(io.BytesIO(archive.read(f"{block['textures']['upper']}.png")))
            self.assertEqual(block["type"], "door")
            self.assertEqual(block["model"], "door_lower")
            self.assertEqual(block["properties"]["door"], True)
            self.assertEqual(set(block["textures"]), {"lower", "upper", "edge"})
            self.assertEqual(behavior["item"], "oak_door")
            self.assertTrue(behavior["openable_by_hand"])
            self.assertEqual(item["right"], "place")
            self.assertEqual(item["block"], "oak_door")
            self.assertEqual(loot["pools"][0]["entries"][0]["name"], "oak_door")
            self.assertEqual(lower.getpixel((4, 4))[:3], (20, 140, 40))
            self.assertEqual(upper.getpixel((4, 4))[:3], (180, 40, 20))
            save_project(session.doc, folder / "oak_door.vam")
            loaded = load_project(folder / "oak_door.vam")
            self.assertEqual(loaded.kind, "door")
            self.assertEqual(tuple(int(channel) for channel in loaded.layers[0].pixels[4, 4]), (20, 140, 40, 255))

        session.new_document(16, 16, name="oak_trapdoor", kind="trapdoor")
        self.assertEqual((session.doc.width, session.doc.height), (32, 16))
        session.fg = (10, 20, 200, 255)
        session.pointer_down(2, 2, 1)
        session.pointer_up(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            exported = export_definition(session.doc, library, Path(tmp) / "oak_trapdoor.zip")
            with zipfile.ZipFile(exported.path) as archive:
                block = json.loads(archive.read("oak_trapdoor.json"))
                behavior = json.loads(archive.read("trapdoors/oak_trapdoor.json"))
                panel = Image.open(io.BytesIO(archive.read(f"{block['textures']['texture']}.png")))
            self.assertEqual(block["type"], "trapdoor")
            self.assertEqual(block["model"], "trapdoor")
            self.assertEqual(set(block["textures"]), {"texture", "edge"})
            self.assertEqual(behavior["loot_table"], "blocks/oak_trapdoor")
            self.assertEqual(panel.getpixel((2, 2))[:3], (10, 20, 200))

    def test_wood_set_exports_the_family(self) -> None:
        library = ScriptLibrary()
        session = Session(library)
        session.new_document(16, 16, name="Pine", kind="set")
        self.assertEqual(session.doc.kind, "set")
        self.assertEqual(session.doc.family, "wood")
        self.assertEqual((session.doc.width, session.doc.height), WOOD_SHEET)
        session.fg = (9, 8, 7, 255)
        session.pointer_down(1, 1, 1)
        session.pointer_up(1, 1)
        pixels = session.doc.layers[0].pixels
        self.assertEqual(tuple(int(channel) for channel in pixels[1, 1]), (9, 8, 7, 255))
        self.assertEqual(int(pixels[1, 17, 3]), 0)

        def fill(index: int, color: tuple[int, int, int, int], spot: tuple[int, int] | None = None) -> None:
            origin = index * 16
            if spot is None:
                pixels[:, origin : origin + 16] = color
            else:
                row, col = spot
                pixels[row, origin + col] = color

        fill(0, (120, 72, 32, 255))
        fill(1, (80, 48, 24, 255))
        fill(2, (200, 160, 80, 255))
        fill(3, (0, 0, 0, 0))
        fill(3, (210, 170, 90, 255), (0, 0))
        fill(4, (180, 140, 60, 255))
        fill(5, (40, 90, 140, 255))
        fill(6, (140, 90, 40, 255))
        fill(7, (20, 140, 40, 255))
        fill(8, (180, 40, 20, 255))
        fill(9, (30, 30, 30, 255))
        fill(10, (10, 20, 200, 255))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exported = export_definition(session.doc, library, folder / "pine.zip", "pine")
            with zipfile.ZipFile(exported.path) as archive:
                manifest = json.loads(archive.read("sets/pine.json"))
                log = json.loads(archive.read("blocks/pine_log.json"))
                wood = json.loads(archive.read("blocks/pine_wood.json"))
                planks = json.loads(archive.read("blocks/pine_planks.json"))
                slab = json.loads(archive.read("blocks/pine_planks_slab.json"))
                upright = json.loads(archive.read("blocks/pine_planks_vertical_slab.json"))
                stairs = json.loads(archive.read("blocks/pine_planks_stairs.json"))
                vertical = json.loads(archive.read("blocks/vertical_pine_planks.json"))
                vertical_slab = json.loads(archive.read("blocks/vertical_pine_planks_vertical_slab.json"))
                table = json.loads(archive.read("blocks/pine_crafting_table.json"))
                door = json.loads(archive.read("blocks/pine_door.json"))
                trap = json.loads(archive.read("blocks/pine_trapdoor.json"))
                door_file = json.loads(archive.read("doors/pine_door.json"))
                trap_file = json.loads(archive.read("trapdoors/pine_trapdoor.json"))
                stick = json.loads(archive.read("items/pine_stick.json"))
                log_item = json.loads(archive.read("items/pine_log.json"))
                loot = json.loads(archive.read("loot/blocks/pine_planks.json"))
                top = np.array(Image.open(io.BytesIO(archive.read(f"textures/{log['textures']['top']}.png"))))
                side = np.array(Image.open(io.BytesIO(archive.read(f"textures/{log['textures']['side']}.png"))))
                plank_tex = np.array(Image.open(io.BytesIO(archive.read(f"textures/{planks['textures']['top']}.png"))))
                turned = np.array(Image.open(io.BytesIO(archive.read(f"textures/{vertical['textures']['top']}.png"))))
                lower = Image.open(io.BytesIO(archive.read(f"textures/{door['textures']['lower']}.png")))
            self.assertEqual(manifest["set"], "wood")
            self.assertIn("pine_log", manifest["members"])
            self.assertIn("vertical_pine_planks_stairs", manifest["members"])
            self.assertIn("pine_trapdoor", manifest["members"])
            self.assertEqual(log["properties"]["rotation"], "axis")
            self.assertEqual(log["textures"]["bottom"], log["textures"]["top"])
            self.assertNotEqual(log["textures"]["top"], log["textures"]["side"])
            self.assertEqual(tuple(top[4, 4][:3]), (120, 72, 32))
            self.assertEqual(tuple(side[4, 4][:3]), (80, 48, 24))
            self.assertEqual(wood["textures"], {"top": log["textures"]["side"], "side": log["textures"]["side"], "bottom": log["textures"]["side"]})
            self.assertEqual(wood["tab"], ["building", "natural"])
            self.assertEqual(slab["type"], "slab")
            self.assertEqual(slab["hardness"], 1)
            self.assertEqual(upright["type"], "vertical_slab")
            self.assertEqual(upright["full_block"], "pine_planks")
            self.assertEqual(stairs["type"], "stairs")
            self.assertEqual(planks["textures"]["side"], planks["textures"]["top"])
            self.assertTrue(np.array_equal(turned, np.rot90(plank_tex, 1)))
            self.assertFalse(np.array_equal(turned, plank_tex))
            self.assertEqual(vertical_slab["full_block"], "vertical_pine_planks")
            self.assertEqual(table["station"], "crafting_table")
            self.assertEqual(table["hardness"], 2.5)
            self.assertEqual(table["textures"]["down"], log["textures"]["top"])
            self.assertEqual(table["textures"]["north"], table["textures"]["south"])
            self.assertNotEqual(table["textures"]["up"], table["textures"]["north"])
            self.assertEqual(door["type"], "door")
            self.assertEqual(trap["type"], "trapdoor")
            self.assertEqual(trap["textures"]["edge"], door["textures"]["edge"])
            self.assertEqual(door_file["item"], "pine_door")
            self.assertEqual(trap_file["textures"]["edge"], door["textures"]["edge"])
            self.assertEqual(lower.getpixel((4, 4))[:3], (20, 140, 40))
            self.assertEqual(stick["model"], "flat")
            self.assertEqual(stick["right"], "none")
            self.assertEqual(stick["tab"], "materials")
            self.assertEqual(log_item["model"], "block")
            self.assertEqual(log_item["right"], "place")
            self.assertEqual(loot["pools"][0]["entries"][0]["name"], "pine_planks")
            save_project(session.doc, folder / "pine.vam")
            loaded = load_project(folder / "pine.vam")
            self.assertEqual(loaded.kind, "set")
            self.assertEqual(loaded.family, "wood")
            self.assertEqual(tuple(int(channel) for channel in loaded.layers[0].pixels[0, 48]), (210, 170, 90, 255))

    def test_custom_definition_exports_and_clips(self) -> None:
        definitions = DefinitionLibrary(folders=[])
        definitions.load_file(ROOT / "examples" / "lantern.py")
        session = Session(definitions=definitions)
        session.new_document(16, 16, name="Oak Lantern", definition="lantern", asset="lantern")
        self.assertEqual(session.doc.kind, "custom")
        self.assertEqual(session.doc.definition, "lantern")
        self.assertEqual((session.doc.width, session.doc.height), (32, 16))
        self.assertEqual(session.doc.fields["light"], 12)
        self.assertIs(session.doc.fields["hang"], True)
        session.fg = (1, 2, 3, 255)
        session.pointer_down(2, 2, 1)
        session.pointer_up(2, 2)
        pixels = session.doc.layers[0].pixels
        self.assertEqual(tuple(int(channel) for channel in pixels[2, 2]), (1, 2, 3, 255))
        self.assertEqual(int(pixels[2, 18, 3]), 0)
        pixels[:, 0:16] = (20, 30, 40, 255)
        pixels[:, 16:32] = (50, 60, 70, 255)
        session.set_definition_field("light", 8)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exported = export_custom(
                session.doc, definitions, session.library, folder / "lantern.zip", "oak_lantern"
            )
            with zipfile.ZipFile(exported.path) as archive:
                body = json.loads(archive.read("lanterns/oak_lantern.json"))
                front = Image.open(io.BytesIO(archive.read("textures/oak_lantern_front.png")))
                top = Image.open(io.BytesIO(archive.read("textures/oak_lantern_top.png")))
            self.assertEqual(body["light"], 8)
            self.assertIs(body["hang"], True)
            self.assertEqual(body["textures"]["front"], "oak_lantern_front")
            self.assertEqual(front.getpixel((4, 4))[:3], (20, 30, 40))
            self.assertEqual(top.getpixel((4, 4))[:3], (50, 60, 70))
            save_project(session.doc, folder / "lantern.vam")
            loaded = load_project(folder / "lantern.vam")
            self.assertEqual(loaded.definition, "lantern")
            self.assertEqual(loaded.asset, "lantern")
            self.assertEqual(loaded.fields["light"], 8)
            self.assertEqual(tuple(int(channel) for channel in loaded.layers[0].pixels[0, 20]), (50, 60, 70, 255))
        broken = DefinitionLibrary(folders=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.py"
            path.write_text("ASSETS = [\n", encoding="utf-8")
            library = DefinitionLibrary(folders=[Path(tmp)])
            self.assertTrue(any("broken.py" in line for line in library.errors))
        session.doc.definition = "missing"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DefinitionError):
                export_custom(session.doc, broken, session.library, Path(tmp) / "nope.zip")

    def test_layer_scripts_and_definitions_use_different_folders(self) -> None:
        from voxel_asset_maker.definitions import definition_directories, definition_from_path
        from voxel_asset_maker.processors import ProcessError, processor_from_path

        with self.assertRaises(ProcessError) as script_error:
            processor_from_path(ROOT / "examples" / "lantern.py")
        self.assertIn("user_scripts/definitions", str(script_error.exception))
        with self.assertRaises(DefinitionError) as definition_error:
            definition_from_path(ROOT / "user_scripts" / "brightness_steps.py")
        self.assertIn("layer script", str(definition_error.exception))
        folders = [path.name for path in definition_directories()]
        self.assertIn("definitions", folders)
        self.assertTrue(any(path.parent.name == "user_scripts" for path in definition_directories()))

    def test_minecraft_block_exports_frames_variants_and_kubejs(self) -> None:
        session = Session()
        self.assertIsNotNone(session.definitions.get("minecraft"))
        session.new_document(16, 16, name="glowstone", definition="minecraft", asset="block")
        self.assertEqual((session.doc.width, session.doc.height), (16, 16))
        session.set_definition_field("frames", 3)
        session.set_definition_field("frametime", 4)
        session.set_definition_field("tool", "pickaxe")
        session.set_definition_field("requires_tool", True)
        self.assertEqual((session.doc.width, session.doc.height), (16, 48))
        self.assertEqual([part.id for part in session.sheet_parts()], ["all_1", "all_2", "all_3"])
        pixels = session.doc.layers[0].pixels
        pixels[0:16, :] = (10, 20, 30, 255)
        pixels[16:32, :] = (40, 50, 60, 255)
        pixels[32:48, :] = (70, 80, 90, 255)
        session.set_script_id("hsb")
        session.set_script_param("count", 2)
        session.set_script_param("hue_step", 120)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exported = export_custom(session.doc, session.definitions, session.library, folder / "glowstone.zip")
            self.assertEqual(exported.errors, [])
            with zipfile.ZipFile(exported.path) as archive:
                names = set(archive.namelist())
                script = archive.read("startup_scripts/glowstone.js").decode("utf-8")
                meta = json.loads(archive.read("assets/kubejs/textures/block/glowstone.png.mcmeta"))
                state = json.loads(archive.read("assets/kubejs/blockstates/glowstone.json"))
                model = json.loads(archive.read("assets/kubejs/models/block/glowstone.json"))
                first = Image.open(io.BytesIO(archive.read("assets/kubejs/textures/block/glowstone.png")))
                second = Image.open(io.BytesIO(archive.read("assets/kubejs/textures/block/glowstone_2.png")))
            self.assertIn("assets/kubejs/textures/block/glowstone_2.png.mcmeta", names)
            self.assertIn("StartupEvents.registry('block'", script)
            self.assertIn("event.create('glowstone')", script)
            self.assertIn(".requiresTool(true)", script)
            self.assertIn("minecraft:mineable/pickaxe", script)
            self.assertEqual(meta["animation"]["frametime"], 4)
            self.assertNotIn("interpolate", meta["animation"])
            self.assertEqual(first.size, (16, 48))
            self.assertEqual(first.getpixel((4, 4))[:3], (10, 20, 30))
            self.assertEqual(first.getpixel((4, 20))[:3], (40, 50, 60))
            self.assertNotEqual(first.getpixel((4, 4))[:3], second.getpixel((4, 4))[:3])
            self.assertEqual(
                state["variants"][""],
                [
                    {"model": "kubejs:block/glowstone"},
                    {"model": "kubejs:block/glowstone_2"},
                ],
            )
            self.assertEqual(model["parent"], "minecraft:block/cube_all")
            self.assertEqual(model["textures"]["all"], "kubejs:block/glowstone")
        session.set_definition_field("frames", 1)
        session.set_definition_field("faces", "column")
        self.assertEqual((session.doc.width, session.doc.height), (48, 16))
        self.assertEqual([part.id for part in session.sheet_parts()], ["top", "side", "bottom"])
        with tempfile.TemporaryDirectory() as tmp:
            exported = export_custom(
                session.doc, session.definitions, session.library, Path(tmp) / "column.zip"
            )
            with zipfile.ZipFile(exported.path) as archive:
                names = set(archive.namelist())
                column = json.loads(archive.read("assets/kubejs/models/block/glowstone.json"))
            self.assertNotIn("assets/kubejs/textures/block/glowstone_top.png.mcmeta", names)
            self.assertEqual(column["parent"], "minecraft:block/cube_bottom_top")
            self.assertEqual(column["textures"]["side"], "kubejs:block/glowstone_side")

    def test_script_can_target_one_part(self) -> None:
        definitions = DefinitionLibrary(folders=[])
        definitions.load_file(ROOT / "examples" / "lantern.py")
        session = Session(definitions=definitions)
        session.new_document(16, 16, name="Oak Lantern", definition="lantern", asset="lantern")
        pixels = session.doc.layers[0].pixels
        pixels[:, 0:16] = (200, 100, 50, 255)
        pixels[:, 16:32] = (10, 20, 30, 255)
        session.add_script("hsb")
        session.set_script_param("brightness", 0)
        session.set_script_parts(["front"])
        limited = render_variant(session.doc, 1, session.library, session.definitions)
        self.assertEqual(limited.image.getpixel((4, 4))[:3], (0, 0, 0))
        self.assertEqual(limited.image.getpixel((20, 4))[:3], (10, 20, 30))
        self.assertEqual(limited.errors, [])
        session.set_script_parts([])
        whole = render_variant(session.doc, 1, session.library, session.definitions).image
        self.assertEqual(whole.getpixel((4, 4))[:3], (0, 0, 0))
        self.assertEqual(whole.getpixel((20, 4))[:3], (0, 0, 0))
        session.set_script_parts(["top", "front"])
        self.assertEqual(session.current_binding().parts, ("front", "top"))
        session.doc.layers[0].scripts[0].parts = ("missing",)
        missed = render_variant(session.doc, 1, session.library, session.definitions)
        self.assertEqual(missed.image.getpixel((4, 4))[:3], (200, 100, 50))
        self.assertTrue(any("chosen parts" in line for line in missed.errors))
        wood = Session()
        wood.new_document(16, 16, name="pine", kind="set")
        self.assertIn("log_top", [part.id for part in wood.sheet_parts()])
        with tempfile.TemporaryDirectory() as tmp:
            session.set_script_parts(["front"])
            path = Path(tmp) / "lantern.vam"
            save_project(session.doc, path)
            loaded = load_project(path)
            self.assertEqual(loaded.layers[0].scripts[0].parts, ("front",))


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
