"""Editing session: tools, layers, and the pixels they change.

The window is a view over this. Tests drive the session directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .history import History
from .model import Color, Document, Layer, empty_pixels, validate_size
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
    def __init__(self, library: ScriptLibrary | None = None) -> None:
        self.library = library or ScriptLibrary()
        self.doc = Document(16, 16, name="texture")
        self.history = History()
        self.selection: Rect | None = None
        self.tool = "pencil"
        self.fg: Color = (232, 232, 232, 255)
        self.bg: Color = (0, 0, 0, 0)
        self.brush_size = 1
        self.filled_shapes = False
        self.variant_index = 1
        self.view_mode = "composite"
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
        self.path = path
        self.selection = None
        self.variant_index = 1
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
    ) -> None:
        width, height = validate_size(width, height)
        pixels = empty_pixels(width, height, fill)
        doc = Document(width, height, name=name.strip() or "texture", layers=[Layer("base", pixels)])
        self.replace_document(doc, None)

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
        self._clamp_variant()
        self.invalidate()
        return True

    def redo(self) -> bool:
        if self._stroke is not None:
            return False
        if not self.history.redo(self.doc):
            return False
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
        if self.view_mode == "layer":
            self.script_errors = []
            self._display_cache = self.doc.active_layer.pixels
            return self._display_cache
        try:
            result = render_variant(self.doc, self.variant_index, self.library)
            self.script_errors = list(result.errors)
            self._display_cache = np.array(result.image, dtype=np.uint8)
        except Exception as exc:
            self.script_errors = [str(exc)]
            self._display_cache = self.doc.active_layer.pixels
        return self._display_cache

    def selection_offset(self) -> tuple[int, int]:
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
        self.doc.active_layer.offset = (int(x), int(y))
        self.invalidate()

    def set_script_id(self, script_id: str, script_path: str = "") -> None:
        binding = self.doc.active_layer.script
        binding.script_id = script_id
        if script_path:
            binding.script_path = script_path
        proc = self.library.get(script_id)
        for param in proc.params:
            binding.params.setdefault(param.key, param.default)
        self._clamp_variant()
        self.invalidate()

    def set_script_param(self, key: str, value: Any) -> None:
        self.doc.active_layer.script.params[key] = value
        self._clamp_variant()
        self.invalidate()

    def randomize_seed(self) -> None:
        import random

        self.begin_edit()
        self.set_script_param("seed", random.randint(0, 2_147_483_647))
        self.end_edit()

    def reload_scripts(self) -> None:
        self.library.reload()
        self._remember_script_paths()
        self._clamp_variant()
        self.invalidate()

    # --- painting -----------------------------------------------------------

    def pointer_down(self, x: int, y: int, button: int) -> None:
        if button not in (1, 2) or self.tool == "picker":
            if self.tool == "picker" and button in (1, 2):
                self._pick(x, y, button)
            return
        self.begin_edit()
        lx, ly = self.layer_xy(x, y)
        layer = self.doc.active_layer
        self._stroke = {
            "tool": self.tool,
            "button": button,
            "anchor": (lx, ly),
            "last": (lx, ly),
            "before": layer.pixels.copy(),
            "moved": False,
        }
        if self.tool in ("pencil", "eraser"):
            self._stamp(lx, ly, button)
        elif self.tool == "fill":
            flood_fill(layer.pixels, lx, ly, self._color(button), self.selection)
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
            self.doc.active_layer.pixels[:] = self._stroke["before"]
            self._draw_line(self._stroke["anchor"], (lx, ly), button)
        elif tool == "rect":
            self.doc.active_layer.pixels[:] = self._stroke["before"]
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
        self.doc.active_layer.pixels[:] = self._stroke["before"]
        self._stroke = None
        self.history.abandon()
        self.invalidate()

    def layer_xy(self, x: int, y: int) -> tuple[int, int]:
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
        lifted = extract(self.doc.active_layer.pixels, sx, sy, sw, sh)
        clear_region(self.doc.active_layer.pixels, sx, sy, sw, sh)
        blit(self.doc.active_layer.pixels, lifted, sx + dx, sy + dy)
        self.selection = (sx + dx, sy + dy, sw, sh)
        self.end_edit()
        self.invalidate()

    def select_all(self) -> None:
        self.selection = (0, 0, self.doc.width, self.doc.height)

    def deselect(self) -> None:
        self.selection = None

    def copy_selection(self) -> np.ndarray:
        pixels = self.doc.active_layer.pixels
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
        clear_region(self.doc.active_layer.pixels, x, y, w, h)
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
        blit(self.doc.active_layer.pixels, pasted, dest_x, dest_y)
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
        clear_region(self.doc.active_layer.pixels, x, y, w, h)
        self.end_edit()
        self.invalidate()

    def clear_layer(self) -> None:
        self.begin_edit()
        if self.selection is not None:
            x, y, w, h = self.selection
            clear_region(self.doc.active_layer.pixels, x, y, w, h)
        else:
            self.doc.active_layer.pixels[:] = 0
        self.end_edit()
        self.invalidate()

    def flip_horizontal(self) -> None:
        self.begin_edit()
        flip_horizontal(self.doc.active_layer.pixels, self.selection)
        self.end_edit()
        self.invalidate()

    def flip_vertical(self) -> None:
        self.begin_edit()
        flip_vertical(self.doc.active_layer.pixels, self.selection)
        self.end_edit()
        self.invalidate()

    def rotate_cw(self) -> None:
        self.begin_edit()
        updated = rotate_cw(self.doc.active_layer.pixels, self.selection)
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
        errors = export_png(self.doc, path, self.variant_index, self.library)
        count = self.variant_count()
        if count > 1:
            self.message = f"Exported variant {self.variant_index} of {count} to {Path(path).name}"
        else:
            self.message = f"Exported {Path(path).name}"
        return errors

    def export_zip(self, path: Path) -> tuple[list[str], list[str]]:
        names, errors = export_variant_zip(self.doc, path, self.library)
        self.message = f"Wrote {len(names)} texture{'s' if len(names) != 1 else ''} to {Path(path).name}"
        return names, errors

    # --- internals ----------------------------------------------------------

    def _color(self, button: int) -> Color:
        if self.tool == "eraser":
            return (0, 0, 0, 0)
        return self.bg if button == 2 else self.fg

    def _stamp(self, x: int, y: int, button: int) -> None:
        stamp_brush(
            self.doc.active_layer.pixels,
            x,
            y,
            self._color(button),
            self.brush_size,
            self.selection,
        )

    def _stroke_between(self, start: tuple[int, int], end: tuple[int, int], button: int) -> None:
        stroke_line(
            self.doc.active_layer.pixels,
            start[0],
            start[1],
            end[0],
            end[1],
            self._color(button),
            self.brush_size,
            self.selection,
        )

    def _draw_line(self, start: tuple[int, int], end: tuple[int, int], button: int) -> None:
        self._stroke_between(start, end, button)

    def _draw_rect(self, start: tuple[int, int], end: tuple[int, int], button: int) -> None:
        stroke_rect(
            self.doc.active_layer.pixels,
            start[0],
            start[1],
            end[0],
            end[1],
            self._color(button),
            self.brush_size,
            self.filled_shapes,
            self.selection,
        )

    def _select_down(self, x: int, y: int) -> None:
        pixels = self.doc.active_layer.pixels
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
            self.doc.active_layer.pixels[:] = self._stroke["base"]
            blit(self.doc.active_layer.pixels, lifted, sx + (x - ax), sy + (y - ay))
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
            proc = self.library.get(layer.script.script_id)
            if proc.path is not None:
                layer.script.script_path = str(proc.path)

    def _remember_script_paths(self) -> None:
        for layer in self.doc.layers:
            script_id = layer.script.script_id
            if self.library.known(script_id) or not layer.script.script_path:
                continue
            path = Path(layer.script.script_path)
            if path.is_file():
                try:
                    proc = self.library.load_file(path)
                    layer.script.script_id = proc.id
                except Exception as exc:
                    self.script_errors.append(f"{layer.name}: {exc}")


def _hex(color: Color) -> str:
    r, g, b, a = color
    if a == 255:
        return f"#{r:02x}{g:02x}{b:02x}"
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
