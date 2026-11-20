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

from .model import (
    BLEND_MODES,
    CTM_SHEET,
    TILE,
    BlockTraits,
    CropTraits,
    Document,
    Fluorescence,
    Layer,
    ScriptBinding,
    empty_pixels,
    refit,
    validate_size,
)
from .processors import ScriptLibrary
from .render import render_variant, stack_variant_count

PROJECT_FORMAT = 2


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
        "kind": doc.kind,
        "set": doc.family,
        "definition": doc.definition,
        "asset": doc.asset,
        "fields": dict(doc.fields),
        "width": doc.width,
        "height": doc.height,
        "active": doc.active_index,
        "palette": list(doc.palette),
        "ctm": bool(doc.ctm),
        "fluorescence": {
            "enabled": bool(doc.fluor.enabled),
            "mode": doc.fluor.mode,
            "color": doc.fluor.color,
            "start": int(doc.fluor.start),
            "full": int(doc.fluor.full),
            "strength": float(doc.fluor.strength),
        },
        "block": {
            "shape": doc.traits.shape,
            "tab": doc.traits.tab,
            "hardness": doc.traits.hardness,
            "tool": doc.traits.tool,
            "mining_level": int(doc.traits.mining_level),
            "sound": doc.traits.sound,
            "render": doc.traits.render,
            "solid": bool(doc.traits.solid),
            "gravity": bool(doc.traits.gravity),
            "light": list(doc.traits.light),
            "full_block": doc.traits.full_block,
            "rotation": doc.traits.rotation,
            "station": doc.traits.station,
            "recipe_types": doc.traits.recipe_types,
            "cook_speed": float(doc.traits.cook_speed),
            "faces": doc.traits.faces,
        },
        "crop": {
            "stages": int(doc.crop.stages),
            "min_time": doc.crop.min_time,
            "max_time": doc.crop.max_time,
            "needs_water": bool(doc.crop.needs_water),
            "water_radius": int(doc.crop.water_radius),
            "water_vertical": int(doc.crop.water_vertical),
            "seeds": doc.crop.seeds,
            "produce": doc.crop.produce,
            "loot": [
                {
                    "item": drop.item,
                    "min": int(drop.minimum),
                    "max": int(drop.maximum),
                    "chance": float(drop.chance),
                }
                for drop in doc.crop.loot
            ],
            "ground": doc.crop.ground,
            "seed_tab": doc.crop.seed_tab,
            "produce_tab": doc.crop.produce_tab,
        },
        "layers": [
            {
                "id": layer.id,
                "name": layer.name,
                "visible": layer.visible,
                "opacity": layer.opacity,
                "blend": layer.blend,
                "offset": [int(layer.offset[0]), int(layer.offset[1])],
                "script": _script_dict(layer.scripts[0]) if layer.scripts else _script_dict(None),
                "scripts": [_script_dict(binding) for binding in layer.scripts if binding.script_id != "none"],
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
        archive.writestr("ctm.png", png_bytes(doc.ctm_pixels))
        archive.writestr("fluorescence.png", png_bytes(doc.fluor.pixels))
    os.replace(tmp, path)


def load_project(path: Path) -> Document:
    path = Path(path)
    with zipfile.ZipFile(path, "r") as archive:
        payload = json.loads(archive.read("project.json"))
        ctm_pixels = (
            refit(array_from_png_bytes(archive.read("ctm.png")), CTM_SHEET[0], CTM_SHEET[1])
            if "ctm.png" in archive.namelist()
            else None
        )
        fluor_pixels = (
            refit(array_from_png_bytes(archive.read("fluorescence.png")), TILE, TILE)
            if "fluorescence.png" in archive.namelist()
            else None
        )
        layers: list[Layer] = []
        width, height = validate_size(payload["width"], payload["height"])
        for index, entry in enumerate(payload["layers"]):
            raw = archive.read(entry.get("file") or f"layers/{index}.png")
            pixels = refit(array_from_png_bytes(raw), width, height)
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
                    scripts=_scripts_from_entry(entry),
                )
            )
    doc = Document(
        width=width,
        height=height,
        name=str(payload.get("name") or path.stem),
        layers=layers,
        kind=str(payload.get("kind") or "texture"),
        family=str(payload.get("set") or ""),
        definition=str(payload.get("definition") or ""),
        asset=str(payload.get("asset") or ""),
        fields=payload.get("fields") if isinstance(payload.get("fields"), dict) else {},
        active_index=int(payload.get("active") or 0),
        palette=[str(color) for color in payload.get("palette") or []],
        ctm=bool(payload.get("ctm", False)),
        ctm_pixels=ctm_pixels if ctm_pixels is not None else _blank_ctm(),
        fluor=_fluorescence_from_payload(payload.get("fluorescence"), fluor_pixels),
        traits=_traits_from_payload(payload.get("block")),
        crop=_crop_from_payload(payload.get("crop")),
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


def export_png(doc: Document, path: Path, variant_index: int, library: ScriptLibrary, definitions=None) -> list[str]:
    path = Path(path)
    result = render_variant(doc, variant_index, library, definitions)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(path, format="PNG")
    return result.errors


def export_variant_zip(doc: Document, path: Path, library: ScriptLibrary, definitions=None) -> tuple[list[str], list[str]]:
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
            result = render_variant(doc, index, library, definitions)
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
                    "script": layer.scripts[0].script_id if layer.scripts else "none",
                    "scripts": [
                        {"id": binding.script_id, "params": binding.params}
                        for binding in layer.scripts
                        if binding.script_id != "none"
                    ],
                }
                for layer in doc.layers
            ],
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    os.replace(tmp, path)
    return names, errors


def _script_dict(binding: ScriptBinding | None) -> dict:
    if binding is None or binding.script_id == "none":
        return {"id": "none", "path": "", "params": {}}
    body = {"id": binding.script_id, "path": binding.script_path, "params": binding.params}
    if binding.parts:
        body["parts"] = list(binding.parts)
    return body


def _scripts_from_entry(entry: dict) -> list[ScriptBinding]:
    """Read the script stack. Older projects stored a single ``script`` object."""
    if "scripts" in entry and isinstance(entry["scripts"], list):
        bindings = [_binding_from_dict(item) for item in entry["scripts"] if isinstance(item, dict)]
    else:
        raw = entry.get("script") or {}
        bindings = [_binding_from_dict(raw)] if isinstance(raw, dict) else []
    return [binding for binding in bindings if binding.script_id and binding.script_id != "none"]


def _binding_from_dict(raw: dict) -> ScriptBinding:
    listed = raw.get("parts") or []
    if isinstance(listed, str):
        listed = listed.replace(",", " ").split()
    parts: list[str] = []
    if isinstance(listed, (list, tuple)):
        for item in listed:
            token = str(item).strip()
            if token and token not in parts:
                parts.append(token)
    return ScriptBinding(
        script_id=str(raw.get("id") or "none"),
        params=dict(raw.get("params") or {}),
        script_path=str(raw.get("path") or ""),
        parts=tuple(parts),
    )


def _blank_ctm() -> np.ndarray:
    return empty_pixels(*CTM_SHEET)


def _crop_from_payload(raw: object) -> CropTraits:
    if not isinstance(raw, dict):
        return CropTraits()
    return CropTraits(
        stages=int(raw.get("stages", 4)),
        min_time=float(raw.get("min_time", 180.0)),
        max_time=float(raw.get("max_time", 340.0)),
        needs_water=bool(raw.get("needs_water", True)),
        water_radius=int(raw.get("water_radius", 4)),
        water_vertical=int(raw.get("water_vertical", 1)),
        seeds=str(raw.get("seeds") or ""),
        produce=str(raw.get("produce") or ""),
        ground=str(raw.get("ground") or "farmland"),
        loot=raw.get("loot") if isinstance(raw.get("loot"), list) else None,
        seed_tab=str(raw.get("seed_tab") or "natural"),
        produce_tab=str(raw.get("produce_tab") or "materials"),
    )


def _traits_from_payload(raw: object) -> BlockTraits:
    if not isinstance(raw, dict):
        return BlockTraits()
    return BlockTraits(
        shape=str(raw.get("shape") or "cube"),
        tab=str(raw.get("tab") or "building"),
        hardness=float(raw.get("hardness", 1.0)),
        tool=str(raw.get("tool") or "pickaxe"),
        mining_level=int(raw.get("mining_level", 0)),
        sound=str(raw.get("sound") or "stone"),
        render=str(raw.get("render") or "solid"),
        solid=bool(raw.get("solid", True)),
        gravity=bool(raw.get("gravity", False)),
        light=raw.get("light", 0),
        full_block=str(raw.get("full_block") or ""),
        rotation=str(raw.get("rotation") or ""),
        station=str(raw.get("station") or ""),
        recipe_types=str(raw.get("recipe_types") or ""),
        cook_speed=float(raw.get("cook_speed", 1.0)),
        faces=str(raw.get("faces") or "six"),
    )


def _fluorescence_from_payload(raw: object, pixels: np.ndarray | None) -> Fluorescence:
    data = raw if isinstance(raw, dict) else {}
    return Fluorescence(
        enabled=bool(data.get("enabled", False)),
        mode=str(data.get("mode") or "texture"),
        color=str(data.get("color") or "#9dff57"),
        start=int(data.get("start", 1) or 0),
        full=int(data.get("full", 8) or 0),
        strength=float(data.get("strength", 1) or 0),
        pixels=pixels if pixels is not None else empty_pixels(TILE, TILE),
    )


def _pad_dim(value: int) -> int:
    if value <= 0:
        return 16
    padded = ((value + 15) // 16) * 16
    return max(16, min(4096, padded))


def _safe_stem(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "texture"
