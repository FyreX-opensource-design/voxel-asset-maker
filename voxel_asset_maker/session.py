"""Editing session: tools, layers, and the pixels they change.

The window is a view over this. Tests drive the session directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .definitions import DefinitionError, DefinitionLibrary
from .history import History
from .model import (
    BLOCK_FACE_ORIGIN,
    BLOCK_SHEET,
    DOOR_SHEET,
    TRAPDOOR_SHEET,
    WOOD_PARTS,
    WOOD_SHEET,
    TILE,
    BlockTraits,
    Color,
    CropTraits,
    Document,
    Layer,
    ScriptBinding,
    block_face_origins,
    block_sheet_from_image,
    block_sheet_size,
    ctm_at,
    ctm_origin,
    empty_pixels,
    face_at,
    face_on_block,
    mask_block_faces,
    part_at,
    read_face,
    relayout_block_pixels,
    remap_block_parts,
    resolve_cube_faces,
    refit,
    stage_at,
    validate_size,
    write_face,
)
from .pixels import (
    Rect,
    blit,
    clear_region,
    extract,
    flip_horizontal,
    flip_vertical,
    flood_fill,
    normalize_rect,
    point_in_rect,
    rotate_cw,
    stamp_brush,
    stroke_line,
    stroke_rect,
)
from .processors import ScriptLibrary
from .render import place, render_variant, stack_variant_count, variant_sources
from .storage import (
    document_from_image,
    export_png,
    export_variant_zip,
    layer_pixels_from_image,
    load_project,
    save_project,
)


class Session:
    def __init__(
        self,
        library: ScriptLibrary | None = None,
        definitions: DefinitionLibrary | None = None,
    ) -> None:
        self.library = library or ScriptLibrary()
        self.definitions = definitions if definitions is not None else DefinitionLibrary()
        self.doc = Document(16, 16, name="texture")
        self.history = History()
        self.selection: Rect | None = None
        self.tool = "pencil"
        self.fg: Color = (232, 232, 232, 255)
        self.bg: Color = (0, 0, 0, 0)
        self.brush_size = 1
        self.filled_shapes = False
        self.variant_index = 1
        self.script_index = 0
        self.view_mode = "composite"
        self.surface = "faces"
        self.stage_index = 0
        self.show_minor_grid = True
        self.show_major_grid = True
        self.path: Path | None = None
        self.message = ""
        self.script_errors: list[str] = []
        self._stroke: dict[str, Any] | None = None
        self._display_cache: np.ndarray | None = None
        self._clipboard: np.ndarray | None = None

    # --- document lifecycle -------------------------------------------------

    def replace_document(self, doc: Document, path: Path | None, message: str = "") -> None:
        self.doc = doc
        self._apply_definition_sheet()
        self.path = path
        self.selection = None
        self.surface = "faces"
        self.stage_index = 0
        self.variant_index = 1
        self.script_index = 0
        self._stroke = None
        self.history.reset()
        self.invalidate()
        self._remember_script_paths()
        if message:
            self.message = message

    def new_document(
        self,
        width: int,
        height: int,
        name: str = "texture",
        fill: Color | None = None,
        kind: str = "texture",
        stages: int = 4,
        definition: str = "",
        asset: str = "",
    ) -> None:
        kind = (kind or "texture").strip().lower() or "texture"
        if definition or kind == "custom":
            spec = self.definitions.get(definition)
            if spec is None:
                raise ValueError(f"Definition '{definition}' isn't loaded.")
            try:
                desc = spec.asset(asset)
            except DefinitionError as exc:
                raise ValueError(str(exc)) from exc
            pixels = empty_pixels(desc.width, desc.height, fill)
            doc = Document(
                desc.width,
                desc.height,
                name=name.strip() or desc.label,
                layers=[Layer("base", pixels)],
                kind="custom",
                definition=spec.id,
                asset=desc.id,
                fields=spec.resolved_fields(desc, {}),
            )
            self.replace_document(doc, None)
            return
        if kind == "crop":
            count = int(np.clip(int(stages), 2, 16))
            width, height = count * TILE, TILE
            pixels = empty_pixels(width, height, fill)
            doc = Document(
                width,
                height,
                name=name.strip() or "crop",
                layers=[Layer("base", pixels)],
                kind="crop",
                crop=CropTraits(stages=count),
            )
            self.replace_document(doc, None)
            return
        if kind == "set":
            width, height = WOOD_SHEET
            pixels = empty_pixels(width, height, fill)
            doc = Document(
                width,
                height,
                name=name.strip() or "wood",
                layers=[Layer("base", pixels)],
                kind="set",
                family="wood",
            )
            self.replace_document(doc, None)
            return
        if kind in ("door", "trapdoor"):
            width, height = DOOR_SHEET if kind == "door" else TRAPDOOR_SHEET
            pixels = empty_pixels(width, height, fill)
            doc = Document(
                width,
                height,
                name=name.strip() or kind,
                layers=[Layer("base", pixels)],
                kind=kind,
                traits=BlockTraits(
                    tab="building",
                    hardness=3.0,
                    tool="axe",
                    sound="wood",
                    render="cutout",
                    solid=True,
                ),
            )
            self.replace_document(doc, None)
            return
        if kind == "block":
            width, height = BLOCK_SHEET
            pixels = empty_pixels(width, height)
            if fill is not None:
                tile = np.zeros((TILE, TILE, 4), dtype=np.uint8)
                tile[:, :] = fill
                for face in BLOCK_FACE_ORIGIN:
                    write_face(pixels, face, tile)
        else:
            width, height = validate_size(width, height)
            pixels = empty_pixels(width, height, fill)
        doc = Document(
            width,
            height,
            name=name.strip() or "texture",
            layers=[Layer("base", pixels)],
            kind=kind,
        )
        if kind == "item":
            doc.traits.tab = "materials"
            doc.traits.__post_init__()
        self.replace_document(doc, None)

    def set_kind(self, kind: str) -> None:
        """Switch what this file is. Block lays the canvas out as six faces."""
        kind = (kind or "texture").strip().lower() or "texture"
        if kind == self.doc.kind:
            return
        self.begin_edit()
        previous = self.doc.kind
        if kind == "block":
            for layer in self.doc.layers:
                layer.pixels = block_sheet_from_image(layer.pixels)
                layer.offset = (0, 0)
            self.doc.width, self.doc.height = BLOCK_SHEET
        elif previous == "block" and kind == "item":
            for layer in self.doc.layers:
                layer.pixels = resolve_cube_faces(layer.pixels, self.doc.traits.faces)["south"]
                layer.offset = (0, 0)
            self.doc.width, self.doc.height = TILE, TILE
        self.doc.kind = kind
        self.selection = None
        self._clamp_surface()
        self.end_edit()
        self.invalidate()

    def set_surface(self, surface: str) -> None:
        """Choose faces, the connected-texture sheet, or the glow map."""
        surface = surface if surface in ("faces", "ctm", "glow") else "faces"
        if surface == self.surface:
            self._clamp_surface()
            return
        self.surface = surface
        self.selection = None
        self._clamp_surface()
        self.invalidate()

    def set_ctm(self, enabled: bool) -> None:
        enabled = bool(enabled) and self.doc.kind == "block"
        if enabled == self.doc.ctm:
            return
        self.doc.ctm = enabled
        if enabled:
            self.surface = "ctm"
        self.selection = None
        self._clamp_surface()
        self.invalidate()

    def set_fluor_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled) and self.doc.kind == "block"
        if enabled == self.doc.fluor.enabled:
            return
        self.doc.fluor.enabled = enabled
        if enabled and self.doc.fluor.mode == "texture":
            self.surface = "glow"
        self.selection = None
        self._clamp_surface()
        self.invalidate()

    def set_fluor_mode(self, mode: str) -> None:
        mode = mode if mode in ("texture", "color") else "texture"
        if mode == self.doc.fluor.mode:
            return
        self.doc.fluor.mode = mode
        if mode == "texture" and self.doc.fluor.enabled:
            self.surface = "glow"
        self._clamp_surface()
        self.invalidate()

    def set_fluor_color(self, color: str) -> None:
        self.doc.fluor.color = color
        self.doc.fluor.__post_init__()

    def set_fluor_range(self, start: int, full: int) -> None:
        self.doc.fluor.start = int(start)
        self.doc.fluor.full = int(full)
        self.doc.fluor.__post_init__()

    def set_fluor_strength(self, strength: float) -> None:
        self.doc.fluor.strength = float(strength)
        self.doc.fluor.__post_init__()

    def set_block_trait(self, **changes: object) -> None:
        traits = self.doc.traits
        allowed = (
            "shape",
            "tab",
            "hardness",
            "tool",
            "mining_level",
            "sound",
            "render",
            "solid",
            "gravity",
            "light",
            "full_block",
            "rotation",
            "station",
            "recipe_types",
            "cook_speed",
            "faces",
        )
        previous = traits.faces
        for key, value in changes.items():
            if key in allowed:
                setattr(traits, key, value)
        traits.__post_init__()
        if self.doc.kind == "block" and traits.faces != previous:
            shape = traits.shape
            for layer in self.doc.layers:
                layer.pixels = relayout_block_pixels(layer.pixels, previous, traits.faces, shape)
                for binding in layer.scripts:
                    binding.parts = remap_block_parts(binding.parts, traits.faces)
            self.doc.width, self.doc.height = block_sheet_size(traits.faces)
            self.selection = None
        self.invalidate()

    def set_crop(self, **changes: object) -> None:
        crop = self.doc.crop
        allowed = (
            "stages",
            "min_time",
            "max_time",
            "needs_water",
            "water_radius",
            "water_vertical",
            "seeds",
            "produce",
            "loot",
            "ground",
            "seed_tab",
            "produce_tab",
        )
        for key, value in changes.items():
            if key in allowed:
                setattr(crop, key, value)
        crop.__post_init__()
        if self.doc.kind == "crop":
            width = crop.stages * TILE
            if self.doc.width != width or self.doc.height != TILE:
                from .model import refit

                for layer in self.doc.layers:
                    layer.pixels = refit(layer.pixels, width, TILE, "topleft")
                self.doc.width = width
                self.doc.height = TILE
                self.selection = None
        if self.stage_index >= crop.stages:
            self.stage_index = crop.stages - 1
        self.invalidate()

    def save(self, path: Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("This texture doesn't have a file yet.")
        if target.suffix.lower() != ".vam":
            target = target.with_suffix(".vam")
        self.doc.name = target.stem or self.doc.name
        self._store_script_paths()
        save_project(self.doc, target)
        self.path = target
        self.history.mark_clean()
        self.message = f"Saved {target.name}"
        return target

    def open_project(self, path: Path) -> None:
        path = Path(path)
        doc = load_project(path)
        self.replace_document(doc, path, message=f"Opened {path.name}")

    def open_image(self, path: Path) -> None:
        doc, note = document_from_image(path)
        self.replace_document(doc, None, message=note or f"Opened {Path(path).name}")

    def import_layer(self, path: Path) -> None:
        pixels = layer_pixels_from_image(path, self.doc.width, self.doc.height)
        self.begin_edit()
        layer = Layer(Path(path).stem or "import", pixels)
        index = self.doc.active_index + 1
        self.doc.layers.insert(index, layer)
        self.doc.active_index = index
        self.end_edit()
        self.invalidate()
        self.message = f"Imported {Path(path).name} as a layer."

    def resize(self, width: int, height: int, anchor: str = "topleft") -> None:
        from .model import refit

        if self.doc.kind == "block":
            self.message = (
                "Each block tile stays 16×16. Faces in the asset panel switches between "
                "six faces, one texture, and a column."
            )
            return
        if self.doc.kind == "crop":
            self.message = "A crop grows by its stage count. Change Stages in the Asset panel."
            return
        width, height = validate_size(width, height)
        self.begin_edit()
        for layer in self.doc.layers:
            layer.pixels = refit(layer.pixels, width, height, anchor)
        self.doc.width = width
        self.doc.height = height
        self.selection = None
        self.end_edit()
        self.invalidate()

    def set_name(self, name: str) -> None:
        cleaned = name.strip() or "texture"
        if cleaned == self.doc.name:
            return
        self.begin_edit()
        self.doc.name = cleaned
        self.end_edit()

    # --- history ------------------------------------------------------------

    def begin_edit(self) -> None:
        self.history.begin(self.doc)

    def end_edit(self) -> bool:
        changed = self.history.commit(self.doc)
        if changed:
            self.invalidate()
        return changed

    def undo(self) -> bool:
        if self._stroke is not None:
            return False
        if not self.history.undo(self.doc):
            return False
        self.selection = None
        self._clamp_surface()
        self._clamp_variant()
        self.invalidate()
        return True

    def redo(self) -> bool:
        if self._stroke is not None:
            return False
        if not self.history.redo(self.doc):
            return False
        self._clamp_surface()
        self.selection = None
        self._clamp_variant()
        self.invalidate()
        return True

    # --- view ---------------------------------------------------------------

    def invalidate(self) -> None:
        self._display_cache = None

    def variant_count(self) -> int:
        return stack_variant_count(self.doc, self.library)

    def variant_labels(self) -> list[tuple[str, int]]:
        return variant_sources(self.doc, self.library)

    def set_variant(self, index: int) -> None:
        self.variant_index = max(1, min(int(index), self.variant_count()))
        self.invalidate()

    def display_array(self) -> np.ndarray:
        if self._display_cache is not None:
            return self._display_cache
        if self._on_extra():
            self.script_errors = []
            self._display_cache = self._pixels()
            return self._display_cache
        if self.view_mode == "layer":
            self.script_errors = []
            shown = self._pixels()
        else:
            try:
                result = render_variant(self.doc, self.variant_index, self.library, self.definitions)
                self.script_errors = list(result.errors)
                shown = np.array(result.image, dtype=np.uint8)
            except Exception as exc:
                self.script_errors = [str(exc)]
                shown = self._pixels()
        if self.doc.kind == "block":
            shown = mask_block_faces(shown)
        self._display_cache = shown
        return self._display_cache

    def selection_offset(self) -> tuple[int, int]:
        if self._on_extra():
            return (0, 0)
        if self.view_mode == "composite":
            ox, oy = self.doc.active_layer.offset
            return int(ox), int(oy)
        return (0, 0)

    # --- layers -------------------------------------------------------------

    def add_layer(self) -> None:
        self.begin_edit()
        layer = Layer(
            f"layer {len(self.doc.layers) + 1}",
            empty_pixels(self.doc.width, self.doc.height),
        )
        index = self.doc.active_index + 1
        self.doc.layers.insert(index, layer)
        self.doc.active_index = index
        self.end_edit()
        self.invalidate()

    def duplicate_layer(self) -> None:
        self.begin_edit()
        copy = self.doc.active_layer.copy()
        copy.name = f"{self.doc.active_layer.name} copy"
        index = self.doc.active_index + 1
        self.doc.layers.insert(index, copy)
        self.doc.active_index = index
        self.end_edit()
        self.invalidate()

    def delete_layer(self) -> None:
        self.begin_edit()
        if len(self.doc.layers) == 1:
            self.doc.layers[0].pixels[:] = 0
            self.message = "Cleared the only layer."
        else:
            index = self.doc.active_index
            del self.doc.layers[index]
            self.doc.active_index = min(index, len(self.doc.layers) - 1)
        self.selection = None
        self.end_edit()
        self.invalidate()

    def move_layer(self, delta: int) -> None:
        index = self.doc.active_index
        target = index + int(delta)
        if not 0 <= target < len(self.doc.layers):
            return
        self.begin_edit()
        layers = self.doc.layers
        layers[index], layers[target] = layers[target], layers[index]
        self.doc.active_index = target
        self.end_edit()
        self.invalidate()

    def merge_down(self) -> bool:
        import texture_layer

        index = self.doc.active_index
        if index <= 0:
            self.message = "The bottom layer has nothing under it."
            return False
        self.begin_edit()
        upper = self.doc.layers[index]
        lower = self.doc.layers[index - 1]
        base = place(lower.pixels, self.doc.width, self.doc.height, lower.offset, lower.opacity)
        over = place(upper.pixels, self.doc.width, self.doc.height, upper.offset, upper.opacity)
        mode = upper.blend if upper.blend else "normal"
        lower.pixels = texture_layer.blend_arrays(base, over, mode=mode, opacity=1.0, offset=(0, 0))
        lower.offset = (0, 0)
        lower.opacity = 1.0
        del self.doc.layers[index]
        self.doc.active_index = index - 1
        self.selection = None
        self.end_edit()
        self.invalidate()
        self.message = f"Merged onto {lower.name}."
        return True

    def set_active(self, index: int) -> None:
        if 0 <= index < len(self.doc.layers) and index != self.doc.active_index:
            self.doc.active_index = index
            self.script_index = 0
            self.selection = None
            self.invalidate()

    def set_active_id(self, layer_id: str) -> None:
        for index, layer in enumerate(self.doc.layers):
            if layer.id == layer_id:
                self.set_active(index)
                return

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        for layer in self.doc.layers:
            if layer.id == layer_id:
                layer.visible = bool(visible)
                self.invalidate()
                return

    def set_opacity(self, opacity: float) -> None:
        self.doc.active_layer.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.invalidate()

    def set_blend(self, mode: str) -> None:
        self.doc.active_layer.blend = mode
        self.invalidate()

    def set_offset(self, x: int, y: int) -> None:
        if self.doc.kind in ("block", "crop"):
            return
        self.doc.active_layer.offset = (int(x), int(y))
        self.invalidate()

    def current_binding(self) -> ScriptBinding | None:
        scripts = self.doc.active_layer.scripts
        if not scripts:
            return None
        self.script_index = max(0, min(self.script_index, len(scripts) - 1))
        return scripts[self.script_index]

    def set_script_id(self, script_id: str, script_path: str = "") -> None:
        """Set the type of the script currently selected on the active layer."""
        layer = self.doc.active_layer
        if not layer.scripts:
            layer.scripts.append(ScriptBinding())
            self.script_index = 0
        binding = self.current_binding()
        assert binding is not None
        if binding.script_id != script_id:
            binding.params = {}
        binding.script_id = script_id
        if script_path:
            binding.script_path = script_path
        proc = self.library.get(script_id)
        for param in proc.params:
            binding.params.setdefault(param.key, param.default)
        self._clamp_variant()
        self.invalidate()

    def sheet_parts(self):
        from .model import sheet_parts

        tiles = self.definitions.tiles_for(self.doc) or ()
        return sheet_parts(self.doc, tiles)

    def set_script_parts(self, parts: list[str] | tuple[str, ...]) -> None:
        """Limit the selected script to these part ids. Empty means the whole sheet."""
        binding = self.current_binding()
        if binding is None:
            return
        known = [part.id for part in self.sheet_parts()]
        chosen: list[str] = []
        for part_id in parts:
            token = str(part_id)
            if token in known and token not in chosen:
                chosen.append(token)
        ordered = tuple(part_id for part_id in known if part_id in chosen)
        if ordered == tuple(binding.parts):
            return
        binding.parts = ordered
        self.invalidate()

    def set_script_param(self, key: str, value: Any) -> None:
        binding = self.current_binding()
        if binding is None:
            return
        binding.params[key] = value
        self._clamp_variant()
        self.invalidate()

    def add_script(self, script_id: str = "none", script_path: str = "") -> None:
        """Append a script. It runs after the ones already on the layer."""
        binding = ScriptBinding(script_id, {}, script_path)
        proc = self.library.get(script_id)
        for param in proc.params:
            binding.params.setdefault(param.key, param.default)
        self.doc.active_layer.scripts.append(binding)
        self.script_index = len(self.doc.active_layer.scripts) - 1
        self._clamp_variant()
        self.invalidate()

    def remove_script(self) -> None:
        scripts = self.doc.active_layer.scripts
        if not scripts:
            return
        index = max(0, min(self.script_index, len(scripts) - 1))
        del scripts[index]
        self.script_index = min(index, max(0, len(scripts) - 1))
        self._clamp_variant()
        self.invalidate()

    def move_script(self, delta: int) -> None:
        """Move the selected script earlier (negative) or later (positive) in the chain."""
        scripts = self.doc.active_layer.scripts
        if not scripts:
            return
        index = max(0, min(self.script_index, len(scripts) - 1))
        target = index + int(delta)
        if not 0 <= target < len(scripts):
            return
        scripts[index], scripts[target] = scripts[target], scripts[index]
        self.script_index = target
        self.invalidate()

    def randomize_seed(self) -> None:
        import random

        self.begin_edit()
        self.set_script_param("seed", random.randint(0, 2_147_483_647))
        self.end_edit()

    def reload_scripts(self) -> None:
        self.library.reload()
        self.definitions.reload()
        self._remember_script_paths()
        self._clamp_variant()
        self.invalidate()

    def set_definition_field(self, key: str, value: object) -> None:
        if self.doc.kind != "custom":
            return
        spec = self.definitions.get(self.doc.definition)
        if spec is None:
            return
        try:
            asset = spec.asset(self.doc.asset)
        except DefinitionError:
            return
        fields = spec.resolved_fields(asset, {**self.doc.fields, key: value})
        if fields == self.doc.fields:
            return
        self.doc.fields = fields
        self._apply_definition_sheet()
        self.invalidate()

    def _apply_definition_sheet(self) -> None:
        """Grow or shrink a custom sheet when its definition layout changes."""
        if self.doc.kind != "custom":
            return
        spec = self.definitions.get(self.doc.definition)
        if spec is None:
            return
        try:
            asset = spec.asset(self.doc.asset)
            fields = spec.resolved_fields(asset, self.doc.fields)
            sheet = spec.sheet(asset, fields)
        except DefinitionError:
            return
        self.doc.fields = fields
        if sheet.width == self.doc.width and sheet.height == self.doc.height:
            return
        for layer in self.doc.layers:
            layer.pixels = refit(layer.pixels, sheet.width, sheet.height, "topleft")
        self.doc.width = sheet.width
        self.doc.height = sheet.height
        self.selection = None

    # --- painting -----------------------------------------------------------

    def pointer_down(self, x: int, y: int, button: int) -> None:
        if button not in (1, 2) or self.tool == "picker":
            if self.tool == "picker" and button in (1, 2):
                self._pick(x, y, button)
            return
        self.begin_edit()
        lx, ly = self.layer_xy(x, y)
        if self.doc.kind == "crop":
            index = stage_at(lx, ly, self.doc.crop.stages)
            if index is not None:
                self.stage_index = index
        clip = self._paint_clip(lx, ly)
        if self._paint_is_blocked(clip):
            self.history.abandon()
            return
        target = self._pixels()
        self._stroke = {
            "tool": self.tool,
            "button": button,
            "anchor": (lx, ly),
            "last": (lx, ly),
            "before": target.copy(),
            "moved": False,
            "clip": clip,
        }
        if self.tool in ("pencil", "eraser"):
            self._stamp(lx, ly, button)
        elif self.tool == "fill":
            flood_fill(target, lx, ly, self._color(button), clip)
        elif self.tool == "line":
            self._draw_line((lx, ly), (lx, ly), button)
        elif self.tool == "rect":
            self._draw_rect((lx, ly), (lx, ly), button)
        elif self.tool == "select":
            self._select_down(lx, ly)
        self.invalidate()

    def pointer_move(self, x: int, y: int) -> bool:
        if not self._stroke:
            return False
        lx, ly = self.layer_xy(x, y)
        tool = self._stroke["tool"]
        button = self._stroke["button"]
        if tool in ("pencil", "eraser"):
            last = self._stroke["last"]
            self._stroke_between(last, (lx, ly), button)
        elif tool == "line":
            self._pixels()[:] = self._stroke["before"]
            self._draw_line(self._stroke["anchor"], (lx, ly), button)
        elif tool == "rect":
            self._pixels()[:] = self._stroke["before"]
            self._draw_rect(self._stroke["anchor"], (lx, ly), button)
        elif tool == "select":
            self._select_move(lx, ly)
        self._stroke["last"] = (lx, ly)
        self._stroke["moved"] = True
        self.invalidate()
        return True

    def pointer_up(self, x: int, y: int) -> bool:
        if not self._stroke:
            return False
        try:
            self.pointer_move(x, y)
        finally:
            self._stroke = None
            changed = self.end_edit()
            self.invalidate()
        return changed

    def cancel_stroke(self) -> None:
        if not self._stroke:
            return
        self._pixels()[:] = self._stroke["before"]
        self._stroke = None
        self.history.abandon()
        self.invalidate()

    def layer_xy(self, x: int, y: int) -> tuple[int, int]:
        if self._on_extra():
            return int(x), int(y)
        if self.view_mode == "composite":
            ox, oy = self.doc.active_layer.offset
            return int(x) - int(ox), int(y) - int(oy)
        return int(x), int(y)

    def swap_colors(self) -> None:
        self.fg, self.bg = self.bg, self.fg

    def nudge(self, dx: int, dy: int) -> None:
        if self.selection is None or self._stroke is not None:
            return
        sx, sy, sw, sh = self.selection
        self.begin_edit()
        lifted = extract(self._pixels(), sx, sy, sw, sh)
        clear_region(self._pixels(), sx, sy, sw, sh)
        blit(self._pixels(), lifted, sx + dx, sy + dy)
        self.selection = (sx + dx, sy + dy, sw, sh)
        self.end_edit()
        self.invalidate()

    def select_all(self) -> None:
        image = self._pixels()
        self.selection = (0, 0, int(image.shape[1]), int(image.shape[0]))

    def deselect(self) -> None:
        self.selection = None

    def copy_selection(self) -> np.ndarray:
        pixels = self._pixels()
        if self.selection is None:
            copied = pixels.copy()
        else:
            x, y, w, h = self.selection
            copied = extract(pixels, x, y, w, h)
        self._clipboard = copied
        return copied

    def cut_selection(self) -> np.ndarray | None:
        if self.selection is None:
            return None
        copied = self.copy_selection()
        self.begin_edit()
        x, y, w, h = self.selection
        clear_region(self._pixels(), x, y, w, h)
        self.end_edit()
        self.invalidate()
        return copied

    def paste_pixels(self, src: np.ndarray | None = None) -> bool:
        image = self._clipboard if src is None else src
        if image is None:
            return False
        pasted = np.array(image, dtype=np.uint8, copy=True)
        if pasted.ndim == 2:
            pasted = np.stack([pasted, pasted, pasted, np.full_like(pasted, 255)], axis=-1)
        elif pasted.shape[-1] == 3:
            alpha = np.full((*pasted.shape[:2], 1), 255, dtype=np.uint8)
            pasted = np.concatenate([pasted, alpha], axis=-1)
        self.begin_edit()
        if self.selection is not None:
            dest_x, dest_y = self.selection[0], self.selection[1]
        else:
            dest_x, dest_y = 0, 0
        blit(self._pixels(), pasted, dest_x, dest_y)
        self.selection = (dest_x, dest_y, pasted.shape[1], pasted.shape[0])
        self._clipboard = pasted
        self.end_edit()
        self.invalidate()
        return True

    def delete_selection(self) -> None:
        if self.selection is None:
            return
        self.begin_edit()
        x, y, w, h = self.selection
        clear_region(self._pixels(), x, y, w, h)
        self.end_edit()
        self.invalidate()

    def clear_layer(self) -> None:
        self.begin_edit()
        if self.selection is not None:
            x, y, w, h = self.selection
            clear_region(self._pixels(), x, y, w, h)
        else:
            self._pixels()[:] = 0
        self.end_edit()
        self.invalidate()

    def flip_horizontal(self) -> None:
        self.begin_edit()
        flip_horizontal(self._pixels(), self.selection)
        self.end_edit()
        self.invalidate()

    def flip_vertical(self) -> None:
        self.begin_edit()
        flip_vertical(self._pixels(), self.selection)
        self.end_edit()
        self.invalidate()

    def rotate_cw(self) -> None:
        self.begin_edit()
        updated = rotate_cw(self._pixels(), self.selection)
        if self.selection is not None and updated is not None:
            self.selection = updated
        self.end_edit()
        self.invalidate()

    def add_swatch(self, color: Color | None = None) -> None:
        text = _hex(color or self.fg)
        if text not in self.doc.palette:
            self.begin_edit()
            self.doc.palette.append(text)
            self.end_edit()

    # --- export -------------------------------------------------------------

    def export_png(self, path: Path) -> list[str]:
        errors = export_png(self.doc, path, self.variant_index, self.library, self.definitions)
        count = self.variant_count()
        if count > 1:
            self.message = f"Exported variant {self.variant_index} of {count} to {Path(path).name}"
        else:
            self.message = f"Exported {Path(path).name}"
        return errors

    def export_zip(self, path: Path) -> tuple[list[str], list[str]]:
        names, errors = export_variant_zip(self.doc, path, self.library, self.definitions)
        self.message = f"Wrote {len(names)} texture{'s' if len(names) != 1 else ''} to {Path(path).name}"
        return names, errors

    # --- internals ----------------------------------------------------------

    def _color(self, button: int) -> Color:
        if self.tool == "eraser":
            return (0, 0, 0, 0)
        return self.bg if button == 2 else self.fg

    def _on_extra(self) -> bool:
        if self.doc.kind != "block":
            return False
        if self.surface == "ctm" and self.doc.ctm:
            return True
        return self.surface == "glow" and self.doc.fluor.enabled and self.doc.fluor.mode == "texture"

    def _pixels(self) -> np.ndarray:
        if self.doc.kind == "block" and self.surface == "ctm" and self.doc.ctm:
            return self.doc.ctm_pixels
        if self.doc.kind == "block" and self.surface == "glow" and self.doc.fluor.enabled and self.doc.fluor.mode == "texture":
            return self.doc.fluor.pixels
        return self.doc.active_layer.pixels

    def _clamp_surface(self) -> None:
        if self.surface == "ctm" and not (self.doc.kind == "block" and self.doc.ctm):
            self.surface = "faces"
            self.selection = None
        elif self.surface == "glow" and not (
            self.doc.kind == "block" and self.doc.fluor.enabled and self.doc.fluor.mode == "texture"
        ):
            self.surface = "faces"
            self.selection = None

    def _paint_is_blocked(self, clip: Rect | None) -> bool:
        """Faces and connection tiles ignore a stroke that starts off a cell."""
        if self.tool == "select" or clip is not None:
            return False
        if self.surface == "ctm" and self.doc.kind == "block" and self.doc.ctm:
            return True
        if self.doc.kind in ("crop", "door", "trapdoor", "set"):
            return True
        tiles = self.definitions.tiles_for(self.doc)
        if tiles:
            return True
        return self.doc.kind == "block" and self.surface == "faces"

    def _paint_clip(self, x: int, y: int):
        """Selection, limited to the block face or connection tile under the pointer."""
        if self.surface == "ctm" and self.doc.kind == "block" and self.doc.ctm:
            index = ctm_at(x, y)
            if index is None:
                return None
            origin_x, origin_y = ctm_origin(index)
            cell = (origin_x, origin_y, TILE, TILE)
            if self.selection is None:
                return cell
            return _intersect(self.selection, cell)
        if self.surface == "glow":
            return self.selection
        if self.doc.kind == "crop":
            index = stage_at(x, y, self.doc.crop.stages)
            if index is None:
                return None
            cell = (index * TILE, 0, TILE, TILE)
            if self.selection is None:
                return cell
            return _intersect(self.selection, cell)
        tiles = self.definitions.tiles_for(self.doc)
        if tiles:
            for tile in tiles:
                if tile.contains(x, y):
                    cell = (tile.x, tile.y, tile.w, tile.h)
                    if self.selection is None:
                        return cell
                    return _intersect(self.selection, cell)
            return None
        if self.doc.kind in ("door", "trapdoor", "set"):
            count = len(WOOD_PARTS) if self.doc.kind == "set" else (3 if self.doc.kind == "door" else 2)
            index = part_at(x, y, count)
            if index is None:
                return None
            cell = (index * TILE, 0, TILE, TILE)
            if self.selection is None:
                return cell
            return _intersect(self.selection, cell)
        if self.doc.kind != "block":
            return self.selection
        mode = self.doc.traits.faces
        name = face_on_block(x, y, mode)
        plant = self.doc.traits.shape == "cross"
        if name is None or (plant and mode == "six" and name != "south") or (plant and mode == "column" and name != "side"):
            return None
        origin_x, origin_y = block_face_origins(mode)[name]
        face = (origin_x, origin_y, TILE, TILE)
        if self.selection is None:
            return face
        return _intersect(self.selection, face)

    def _clip(self):
        if self._stroke is not None and "clip" in self._stroke:
            return self._stroke["clip"]
        return self.selection

    def _stamp(self, x: int, y: int, button: int) -> None:
        stamp_brush(
            self._pixels(),
            x,
            y,
            self._color(button),
            self.brush_size,
            self._clip(),
        )

    def _stroke_between(self, start: tuple[int, int], end: tuple[int, int], button: int) -> None:
        stroke_line(
            self._pixels(),
            start[0],
            start[1],
            end[0],
            end[1],
            self._color(button),
            self.brush_size,
            self._clip(),
        )

    def _draw_line(self, start: tuple[int, int], end: tuple[int, int], button: int) -> None:
        self._stroke_between(start, end, button)

    def _draw_rect(self, start: tuple[int, int], end: tuple[int, int], button: int) -> None:
        stroke_rect(
            self._pixels(),
            start[0],
            start[1],
            end[0],
            end[1],
            self._color(button),
            self.brush_size,
            self.filled_shapes,
            self._clip(),
        )

    def _select_down(self, x: int, y: int) -> None:
        pixels = self._pixels()
        if self.selection is not None and point_in_rect(x, y, self.selection):
            sx, sy, sw, sh = self.selection
            lifted = extract(pixels, sx, sy, sw, sh)
            clear_region(pixels, sx, sy, sw, sh)
            assert self._stroke is not None
            self._stroke["mode"] = "move"
            self._stroke["lifted"] = lifted
            self._stroke["base"] = pixels.copy()
            self._stroke["sel_origin"] = (sx, sy)
        else:
            assert self._stroke is not None
            self._stroke["mode"] = "band"
            self.selection = normalize_rect(x, y, x, y)

    def _select_move(self, x: int, y: int) -> None:
        assert self._stroke is not None
        if self._stroke.get("mode") == "move":
            ax, ay = self._stroke["anchor"]
            sx, sy = self._stroke["sel_origin"]
            lifted = self._stroke["lifted"]
            self._pixels()[:] = self._stroke["base"]
            blit(self._pixels(), lifted, sx + (x - ax), sy + (y - ay))
            self.selection = (sx + (x - ax), sy + (y - ay), lifted.shape[1], lifted.shape[0])
        else:
            ax, ay = self._stroke["anchor"]
            self.selection = normalize_rect(ax, ay, x, y)

    def _pick(self, x: int, y: int, button: int) -> None:
        image = self.display_array()
        height, width = image.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        color = tuple(int(channel) for channel in image[y, x])
        if button == 2:
            self.bg = color  # type: ignore[assignment]
        else:
            self.fg = color  # type: ignore[assignment]

    def _clamp_variant(self) -> None:
        self.variant_index = max(1, min(self.variant_index, self.variant_count()))

    def _store_script_paths(self) -> None:
        for layer in self.doc.layers:
            for binding in layer.scripts:
                proc = self.library.get(binding.script_id)
                if proc.path is not None:
                    binding.script_path = str(proc.path)

    def _remember_script_paths(self) -> None:
        for layer in self.doc.layers:
            for binding in layer.scripts:
                script_id = binding.script_id
                if self.library.known(script_id) or not binding.script_path:
                    continue
                path = Path(binding.script_path)
                if path.is_file():
                    try:
                        proc = self.library.load_file(path)
                        binding.script_id = proc.id
                    except Exception as exc:
                        self.script_errors.append(f"{layer.name}: {exc}")


def _intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    return (x0, y0, x1 - x0, y1 - y0)


def _hex(color: Color) -> str:
    r, g, b, a = color
    if a == 255:
        return f"#{r:02x}{g:02x}{b:02x}"
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
