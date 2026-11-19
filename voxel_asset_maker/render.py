"""Run each layer's script and composite the stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import bootstrap  # noqa: F401
import texture_layer

from .model import Document, Layer, SheetPart, mask_block_faces, refit, sheet_parts
from .processors import (
    ProcessError,
    Processor,
    ScriptContext,
    ScriptLibrary,
    array_to_image,
    image_to_array,
    resolve_params,
)


@dataclass
class RenderResult:
    image: Image.Image
    errors: list[str]
    sources: list[tuple[str, int]]


def variant_sources(doc: Document, library: ScriptLibrary) -> list[tuple[str, int]]:
    """Scripts whose count is above 1, in layer order then script order."""
    sources: list[tuple[str, int]] = []
    for layer in doc.layers:
        for binding in layer.scripts:
            count = _binding_count(binding, library)
            if count > 1:
                label = library.get(binding.script_id).label
                sources.append((f"{layer.name} · {label}", count))
    return sources


def stack_variant_count(doc: Document, library: ScriptLibrary) -> int:
    count = 1
    for layer in doc.layers:
        count = max(count, _layer_count(layer, library))
    return count


def render_variant(doc: Document, export_index: int, library: ScriptLibrary, definitions=None) -> RenderResult:
    """Composite every visible layer for one variant index (1-based)."""
    export_count = stack_variant_count(doc, library)
    export_index = max(1, min(int(export_index), export_count))
    sources = variant_sources(doc, library)
    tiles = ()
    if definitions is not None:
        found = definitions.tiles_for(doc)
        tiles = tuple(found) if found else ()
    parts = sheet_parts(doc, tiles)
    acc: np.ndarray | None = None
    errors: list[str] = []
    for layer in doc.layers:
        if not layer.visible:
            continue
        processed, error = _process_layer(layer, doc, export_index, export_count, library, parts)
        if error:
            errors.append(error)
        if doc.kind == "block":
            processed = mask_block_faces(processed)
            offset = (0, 0)
        else:
            offset = layer.offset
        placed = place(processed, doc.width, doc.height, offset, layer.opacity)
        if acc is None:
            acc = placed
        else:
            mode = layer.blend if layer.blend else "normal"
            try:
                acc = texture_layer.blend_arrays(acc, placed, mode=mode, opacity=1.0, offset=(0, 0))
            except Exception as exc:
                errors.append(f"{layer.name}: {exc}")
    if acc is None:
        acc = np.zeros((doc.height, doc.width, 4), dtype=np.uint8)
    return RenderResult(Image.fromarray(acc, mode="RGBA"), errors, sources)


def place(
    pixels: np.ndarray,
    width: int,
    height: int,
    offset: tuple[int, int],
    opacity: float,
) -> np.ndarray:
    """Paste a layer onto a transparent canvas, then scale its alpha."""
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    src_h, src_w = pixels.shape[:2]
    ox, oy = int(offset[0]), int(offset[1])
    x0 = max(0, ox)
    y0 = max(0, oy)
    x1 = min(width, ox + src_w)
    y1 = min(height, oy + src_h)
    if x0 < x1 and y0 < y1:
        patch = np.array(pixels[y0 - oy : y1 - oy, x0 - ox : x1 - ox], copy=True)
        if opacity < 0.999:
            alpha = patch[..., 3].astype(np.float32) * float(opacity)
            patch[..., 3] = np.clip(np.round(alpha), 0, 255).astype(np.uint8)
        canvas[y0:y1, x0:x1] = patch
    return canvas


def _process_layer(
    layer: Layer,
    doc: Document,
    export_index: int,
    export_count: int,
    library: ScriptLibrary,
    parts: tuple[SheetPart, ...] = (),
) -> tuple[np.ndarray, str | None]:
    """Run this layer's scripts from first to last. Each sees the previous result."""
    pixels = layer.pixels
    errors: list[str] = []
    ran = False
    for binding in layer.scripts:
        if not binding.script_id or binding.script_id == "none":
            continue
        ran = True
        pixels, error = _apply_binding(
            pixels, binding, layer, doc, export_index, export_count, library, parts
        )
        if error:
            errors.append(error)
    if not ran:
        return np.array(layer.pixels, copy=True), None
    return pixels, "\n".join(errors) if errors else None


def _apply_binding(
    pixels: np.ndarray,
    binding,
    layer: Layer,
    doc: Document,
    export_index: int,
    export_count: int,
    library: ScriptLibrary,
    parts: tuple[SheetPart, ...] = (),
) -> tuple[np.ndarray, str | None]:
    if not library.known(binding.script_id):
        return np.array(pixels, copy=True), f"{layer.name}: missing script {binding.script_id}"
    chosen = tuple(getattr(binding, "parts", ()) or ())
    if not chosen or not parts:
        return _run_script(pixels, binding, layer, doc, export_index, export_count, library, "")
    by_id = {part.id: part for part in parts}
    matched = [by_id[part_id] for part_id in chosen if part_id in by_id]
    missing = [part_id for part_id in chosen if part_id not in by_id]
    if not matched:
        return np.array(pixels, copy=True), f"{layer.name}: none of the chosen parts are on this sheet"
    out = np.array(pixels, copy=True)
    notes: list[str] = []
    for part in matched:
        y0 = max(0, part.y)
        x0 = max(0, part.x)
        y1 = min(out.shape[0], part.y + part.h)
        x1 = min(out.shape[1], part.x + part.w)
        if y0 >= y1 or x0 >= x1:
            notes.append(f"{layer.name}: part {part.id} is outside the layer")
            continue
        crop = np.array(out[y0:y1, x0:x1], copy=True)
        rendered, error = _run_script(
            crop, binding, layer, doc, export_index, export_count, library, part.id, x1 - x0, y1 - y0
        )
        if error:
            notes.append(error)
            continue
        out[y0:y1, x0:x1] = refit(rendered, x1 - x0, y1 - y0)
    if missing:
        notes.append(f"{layer.name}: missing parts {', '.join(missing)}")
    return out, "\n".join(notes) if notes else None


def _run_script(
    pixels: np.ndarray,
    binding,
    layer: Layer,
    doc: Document,
    export_index: int,
    export_count: int,
    library: ScriptLibrary,
    part: str,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray, str | None]:
    proc = library.get(binding.script_id)
    label = proc.label
    try:
        params = resolve_params(proc, binding.params)
        own = proc.variant_count(params)
        local = ((export_index - 1) % own) + 1
        context = ScriptContext(
            variant_index=local,
            variant_count=own,
            export_index=export_index,
            export_count=export_count,
            width=int(width if width is not None else pixels.shape[1]),
            height=int(height if height is not None else pixels.shape[0]),
            layer_name=layer.name,
            part=part,
        )
        image = array_to_image(pixels)
        output = proc.apply(image, binding.params, context)
        if isinstance(output, list):
            if not output:
                raise ProcessError("script returned no images")
            output = output[(local - 1) % len(output)]
        return image_to_array(output), None
    except Exception as exc:
        where = f" · {part}" if part else ""
        return np.array(pixels, copy=True), f"{layer.name} · {label}{where}: {exc}"


def _layer_count(layer: Layer, library: ScriptLibrary) -> int:
    count = 1
    for binding in layer.scripts:
        count = max(count, _binding_count(binding, library))
    return count


def _binding_count(binding, library: ScriptLibrary) -> int:
    if not binding.script_id or binding.script_id == "none":
        return 1
    if not library.known(binding.script_id):
        return 1
    proc: Processor = library.get(binding.script_id)
    try:
        return proc.variant_count(binding.params)
    except Exception:
        return 1
