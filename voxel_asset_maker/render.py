"""Run each layer's script and composite the stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import bootstrap  # noqa: F401
import texture_layer

from .model import Document, Layer
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
    """Layers whose script count is above 1, in stack order."""
    sources: list[tuple[str, int]] = []
    for layer in doc.layers:
        count = _layer_count(layer, library)
        if count > 1:
            sources.append((layer.name, count))
    return sources


def stack_variant_count(doc: Document, library: ScriptLibrary) -> int:
    count = 1
    for layer in doc.layers:
        count = max(count, _layer_count(layer, library))
    return count


def render_variant(doc: Document, export_index: int, library: ScriptLibrary) -> RenderResult:
    """Composite every visible layer for one variant index (1-based)."""
    export_count = stack_variant_count(doc, library)
    export_index = max(1, min(int(export_index), export_count))
    sources = variant_sources(doc, library)
    acc: np.ndarray | None = None
    errors: list[str] = []
    for layer in doc.layers:
        if not layer.visible:
            continue
        processed, error = _process_layer(layer, doc, export_index, export_count, library)
        if error:
            errors.append(error)
        placed = place(processed, doc.width, doc.height, layer.offset, layer.opacity)
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
) -> tuple[np.ndarray, str | None]:
    proc = library.get(layer.script.script_id)
    if layer.script.script_id != "none" and not library.known(layer.script.script_id):
        return layer.pixels.copy(), f"{layer.name}: missing script {layer.script.script_id}"
    try:
        params = resolve_params(proc, layer.script.params)
        own = proc.variant_count(params)
        local = ((export_index - 1) % own) + 1
        context = ScriptContext(
            variant_index=local,
            variant_count=own,
            export_index=export_index,
            export_count=export_count,
            width=doc.width,
            height=doc.height,
            layer_name=layer.name,
        )
        image = array_to_image(layer.pixels)
        output = proc.apply(image, layer.script.params, context)
        if isinstance(output, list):
            if not output:
                raise ProcessError("script returned no images")
            output = output[(local - 1) % len(output)]
        return image_to_array(output), None
    except Exception as exc:
        return layer.pixels.copy(), f"{layer.name}: {exc}"


def _layer_count(layer: Layer, library: ScriptLibrary) -> int:
    proc: Processor = library.get(layer.script.script_id)
    if not library.known(layer.script.script_id):
        return 1
    try:
        return proc.variant_count(layer.script.params)
    except Exception:
        return 1
