"""Project files and export."""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .model import BLEND_MODES, Document, Layer, ScriptBinding, refit, validate_size
from .processors import ScriptLibrary
from .render import render_variant, stack_variant_count

PROJECT_FORMAT = 1


def png_bytes(arr: np.ndarray) -> bytes:
    image = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def array_from_png_bytes(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        rgba = image.convert("RGBA")
        return np.array(rgba, dtype=np.uint8)


def save_project(doc: Document, path: Path) -> None:
    path = Path(path)
    payload = {
        "format": PROJECT_FORMAT,
        "name": doc.name,
        "width": doc.width,
        "height": doc.height,
        "active": doc.active_index,
        "palette": list(doc.palette),
        "layers": [
            {
                "id": layer.id,
                "name": layer.name,
                "visible": layer.visible,
                "opacity": layer.opacity,
                "blend": layer.blend,
                "offset": [int(layer.offset[0]), int(layer.offset[1])],
                "script": {
                    "id": layer.script.script_id,
                    "path": layer.script.script_path,
                    "params": layer.script.params,
                },
                "file": f"layers/{index}.png",
            }
            for index, layer in enumerate(doc.layers)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(payload, indent=2))
        for index, layer in enumerate(doc.layers):
            archive.writestr(f"layers/{index}.png", png_bytes(layer.pixels))
    os.replace(tmp, path)


def load_project(path: Path) -> Document:
    path = Path(path)
    with zipfile.ZipFile(path, "r") as archive:
        payload = json.loads(archive.read("project.json"))
        layers: list[Layer] = []
        width, height = validate_size(payload["width"], payload["height"])
        for index, entry in enumerate(payload["layers"]):
            raw = archive.read(entry.get("file") or f"layers/{index}.png")
            pixels = refit(array_from_png_bytes(raw), width, height)
            script = entry.get("script") or {}
            blend = entry.get("blend", "normal")
            if blend not in BLEND_MODES:
                blend = "normal"
            offset = entry.get("offset") or [0, 0]
            layers.append(
                Layer(
                    name=str(entry.get("name") or f"layer {index + 1}"),
                    pixels=pixels,
                    id=str(entry.get("id") or f"layer{index}"),
                    visible=bool(entry.get("visible", True)),
                    opacity=float(entry.get("opacity", 1.0)),
                    blend=blend,
                    offset=(int(offset[0]), int(offset[1])),
                    script=ScriptBinding(
                        script_id=str(script.get("id") or "none"),
                        params=dict(script.get("params") or {}),
                        script_path=str(script.get("path") or ""),
                    ),
                )
            )
    doc = Document(
        width=width,
        height=height,
        name=str(payload.get("name") or path.stem),
        layers=layers,
        active_index=int(payload.get("active") or 0),
        palette=[str(color) for color in payload.get("palette") or []],
    )
    return doc


def document_from_image(path: Path) -> tuple[Document, str]:
    """Open a picture as a one-layer texture, padding up to a multiple of 16."""
    path = Path(path)
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        src = np.array(rgba, dtype=np.uint8)
    src_h, src_w = src.shape[:2]
    width = _pad_dim(src_w)
    height = _pad_dim(src_h)
    pixels = refit(src, width, height)
    note = ""
    if (width, height) != (src_w, src_h):
        note = f"Padded {src_w}×{src_h} up to {width}×{height}."
    if src_w > width or src_h > height:
        note = f"Cropped to {width}×{height}."
    doc = Document(width, height, name=_safe_stem(path.stem), layers=[Layer("base", pixels)])
    return doc, note


def layer_pixels_from_image(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        src = np.array(rgba, dtype=np.uint8)
    return refit(src, width, height)


def export_png(doc: Document, path: Path, variant_index: int, library: ScriptLibrary) -> list[str]:
    path = Path(path)
    result = render_variant(doc, variant_index, library)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(path, format="PNG")
    return result.errors


def export_variant_zip(doc: Document, path: Path, library: ScriptLibrary) -> tuple[list[str], list[str]]:
    """Write one composited PNG per variant, plus a manifest."""
    path = Path(path)
    count = stack_variant_count(doc, library)
    base = _safe_stem(doc.name)
    errors: list[str] = []
    names: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(1, count + 1):
            result = render_variant(doc, index, library)
            if index == 1:
                errors.extend(result.errors)
            filename = f"{base}.png" if count == 1 else f"{base}_{index}.png"
            buffer = io.BytesIO()
            result.image.save(buffer, format="PNG")
            archive.writestr(filename, buffer.getvalue())
            names.append(filename)
        manifest = {
            "name": doc.name,
            "width": doc.width,
            "height": doc.height,
            "count": count,
            "files": names,
            "layers": [
                {
                    "name": layer.name,
                    "visible": layer.visible,
                    "opacity": layer.opacity,
                    "blend": layer.blend,
                    "offset": [int(layer.offset[0]), int(layer.offset[1])],
                    "script": layer.script.script_id,
                    "params": layer.script.params,
                }
                for layer in doc.layers
            ],
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    os.replace(tmp, path)
    return names, errors


def _pad_dim(value: int) -> int:
    if value <= 0:
        return 16
    padded = ((value + 15) // 16) * 16
    return max(16, min(4096, padded))


def _safe_stem(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "texture"
