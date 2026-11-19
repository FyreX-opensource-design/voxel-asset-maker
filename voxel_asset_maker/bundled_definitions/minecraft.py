"""Minecraft block for KubeJS.

File → New → Minecraft → Block. Each texture is a column. An animation is
that column stacked in frames, and each frame is as tall as the texture is
wide. Export writes a startup script, the textures, models, and a blockstate.
More than one frame also writes a .mcmeta file beside each texture. Layer
script variants become extra blockstate models.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from voxel_asset_maker.blockgame import cube_preview, display_name

ID = "minecraft"
NAME = "Minecraft"
BLURB = "A cube block for KubeJS, with animation frames and blockstate variants."

_FACES = {
    "all": (("all", "all", "All faces"),),
    "column": (
        ("top", "top", "Top"),
        ("side", "side", "Side"),
        ("bottom", "bottom", "Bottom"),
    ),
    "cube": (
        ("top", "top", "Top"),
        ("bottom", "bottom", "Bottom"),
        ("north", "north", "North"),
        ("south", "south", "South"),
        ("east", "east", "East"),
        ("west", "west", "West"),
    ),
}

ASSETS = [
    {
        "id": "block",
        "label": "Block",
        "blurb": (
            "One texture on every face, or top, side, and bottom, or all six. "
            "Frames stack down each texture, one width tall. "
            "Export writes a KubeJS startup script, the textures, and an mcmeta file when Frames is above 1. "
            "Layer-script variants become blockstate variants."
        ),
        "width": 16,
        "height": 16,
        "turn": True,
        "tiles": [
            {"id": "all", "label": "all", "caption": "All faces", "x": 0, "y": 0, "w": 16, "h": 16},
        ],
        "fields": [
            {
                "key": "faces",
                "label": "Faces",
                "type": "choice",
                "default": "all",
                "choices": ["all", "column", "cube"],
                "help": "all is one texture. column is top, side, and bottom. cube is the six faces.",
            },
            {
                "key": "resolution",
                "label": "Resolution",
                "type": "choice",
                "default": "16",
                "choices": ["16", "32"],
                "help": "Width of each texture. A frame is this many pixels tall.",
            },
            {
                "key": "frames",
                "label": "Frames",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 32,
                "step": 1,
                "help": "How many frames stack down each texture. 1 is a still texture.",
            },
            {
                "key": "frametime",
                "label": "Frame time",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 240,
                "step": 1,
                "help": "Ticks each frame stays up. Used when Frames is above 1.",
            },
            {
                "key": "interpolate",
                "label": "Blend frames",
                "type": "bool",
                "default": False,
                "help": "Minecraft fades between frames. Used when Frames is above 1.",
            },
            {
                "key": "namespace",
                "label": "Namespace",
                "type": "text",
                "default": "kubejs",
                "help": "Resource namespace. kubejs matches a normal KubeJS pack.",
            },
            {
                "key": "sound",
                "label": "Sound",
                "type": "text",
                "default": "stone",
                "help": "Minecraft sound type, such as stone, wood, gravel, or wool.",
            },
            {
                "key": "map_color",
                "label": "Map color",
                "type": "text",
                "default": "stone",
                "help": "Map color name, such as stone, wood, or grass.",
            },
            {
                "key": "hardness",
                "label": "Hardness",
                "type": "float",
                "default": 1.5,
                "min": 0,
                "max": 1000,
                "step": 0.1,
            },
            {
                "key": "resistance",
                "label": "Resistance",
                "type": "float",
                "default": 6,
                "min": 0,
                "max": 1000,
                "step": 0.1,
            },
            {
                "key": "tool",
                "label": "Tool",
                "type": "choice",
                "default": "none",
                "choices": ["none", "pickaxe", "axe", "shovel", "hoe"],
            },
            {
                "key": "requires_tool",
                "label": "Requires tool",
                "type": "bool",
                "default": False,
            },
            {
                "key": "render",
                "label": "Render",
                "type": "choice",
                "default": "solid",
                "choices": ["solid", "cutout", "translucent"],
            },
            {
                "key": "light",
                "label": "Light",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 15,
                "step": 1,
            },
        ],
    },
]


def layout(asset, fields):
    del asset
    resolution = _resolution(fields)
    frames = _frames(fields)
    faces = _faces(fields)
    tiles = []
    single = len(faces) == 1
    for index, (face_id, label, caption) in enumerate(faces):
        for frame in range(frames):
            if frames == 1:
                tile_id, short, full = face_id, label, caption
            elif single:
                tile_id, short, full = f"{face_id}_{frame + 1}", str(frame + 1), f"Frame {frame + 1}"
            else:
                tile_id = f"{face_id}_{frame + 1}"
                short, full = f"{label} {frame + 1}", f"{caption} {frame + 1}"
            tiles.append(
                {
                    "id": tile_id,
                    "label": short,
                    "caption": full,
                    "x": index * resolution,
                    "y": frame * resolution,
                    "w": resolution,
                    "h": resolution,
                }
            )
    return {"width": len(faces) * resolution, "height": frames * resolution, "tiles": tiles}


def preview(image, asset, fields, context):
    del asset
    resolution = _resolution(fields)
    faces = _faces(fields)
    tiles = {}
    for index, (face_id, _label, _caption) in enumerate(faces):
        tiles[face_id] = _square(image[0:resolution, index * resolution : (index + 1) * resolution])

    def face(name):
        if name in tiles:
            return tiles[name]
        if "all" in tiles:
            return tiles["all"]
        if name in ("north", "south", "east", "west") and "side" in tiles:
            return tiles["side"]
        return next(iter(tiles.values()))

    return cube_preview(
        face("top"),
        face("south"),
        left=face("south"),
        right=face("east"),
        bottom=face("bottom"),
        north=face("north"),
        west=face("west"),
        yaw=context.yaw,
        pitch=context.pitch,
        scale=6,
    )


def export(image, asset, fields, context):
    del asset
    namespace = _token(fields.get("namespace"), "kubejs")
    asset_id = context.asset_id
    frames = _frames(fields)
    resolution = _resolution(fields)
    faces = _faces(fields)
    sheets = context.variants if context.variants else (np.asarray(image, dtype=np.uint8),)
    count = max(1, len(sheets))
    files: dict = {}
    single = len(faces) == 1 and faces[0][0] == "all"
    for variant_index, sheet in enumerate(sheets, start=1):
        model_stem = _variant_stem(asset_id, variant_index, count)
        textures = {}
        for index, (face_id, _label, _caption) in enumerate(faces):
            stem = _texture_stem(asset_id, face_id, variant_index, count, single)
            column = _column(sheet, index, resolution, frames)
            files[f"assets/{namespace}/textures/block/{stem}.png"] = column
            if frames > 1:
                files[f"assets/{namespace}/textures/block/{stem}.png.mcmeta"] = _mcmeta(fields)
            textures[face_id] = f"{namespace}:block/{stem}"
        files[f"assets/{namespace}/models/block/{model_stem}.json"] = _model(textures)
    first = asset_id
    listed = []
    for variant_index in range(1, count + 1):
        listed.append({"model": f"{namespace}:block/{_variant_stem(asset_id, variant_index, count)}"})
    files[f"assets/{namespace}/blockstates/{asset_id}.json"] = {
        "variants": {"": listed[0] if count == 1 else listed}
    }
    files[f"assets/{namespace}/models/item/{asset_id}.json"] = {"parent": f"{namespace}:block/{first}"}
    files[f"startup_scripts/{asset_id}.js"] = _startup(asset_id, fields, namespace)
    return files


def _resolution(fields) -> int:
    try:
        size = int(fields.get("resolution", 16))
    except (TypeError, ValueError):
        size = 16
    return size if size in (16, 32) else 16


def _frames(fields) -> int:
    try:
        count = int(fields.get("frames", 1))
    except (TypeError, ValueError):
        count = 1
    return max(1, min(32, count))


def _faces(fields):
    mode = str(fields.get("faces") or "all")
    return _FACES.get(mode, _FACES["all"])


def _token(value, default: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return text or default


def _variant_stem(asset_id: str, variant_index: int, variant_count: int) -> str:
    if variant_count > 1 and variant_index > 1:
        return f"{asset_id}_{variant_index}"
    return asset_id


def _texture_stem(asset_id: str, face_id: str, variant_index: int, variant_count: int, single: bool) -> str:
    base = asset_id if single else f"{asset_id}_{face_id}"
    if variant_count > 1 and variant_index > 1:
        return f"{base}_{variant_index}"
    return base


def _column(image, index: int, resolution: int, frames: int) -> np.ndarray:
    height = frames * resolution
    out = np.zeros((height, resolution, 4), dtype=np.uint8)
    array = np.asarray(image, dtype=np.uint8)
    x = index * resolution
    y1 = min(array.shape[0], height)
    x1 = min(array.shape[1], x + resolution)
    if y1 > 0 and x1 > x:
        patch = array[0:y1, x:x1]
        out[: patch.shape[0], : patch.shape[1]] = patch
    return out


def _square(tile: np.ndarray) -> np.ndarray:
    array = np.asarray(tile, dtype=np.uint8)
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 1:
        return np.zeros((16, 16, 4), dtype=np.uint8)
    if array.shape[0] == 16 and array.shape[1] == 16:
        return np.ascontiguousarray(array)
    image = Image.fromarray(array, mode="RGBA").resize((16, 16), Image.Resampling.NEAREST)
    return np.array(image, dtype=np.uint8)


def _mcmeta(fields) -> dict:
    try:
        frametime = int(fields.get("frametime", 2))
    except (TypeError, ValueError):
        frametime = 2
    animation = {"frametime": max(1, min(240, frametime))}
    if bool(fields.get("interpolate")):
        animation["interpolate"] = True
    return {"animation": animation}


def _model(textures: dict[str, str]) -> dict:
    if "all" in textures and len(textures) == 1:
        return {"parent": "minecraft:block/cube_all", "textures": {"all": textures["all"]}}
    if set(textures) == {"top", "side", "bottom"}:
        return {
            "parent": "minecraft:block/cube_bottom_top",
            "textures": {
                "top": textures["top"],
                "bottom": textures["bottom"],
                "side": textures["side"],
            },
        }
    particle = textures.get("north") or textures.get("side") or next(iter(textures.values()))
    return {
        "parent": "minecraft:block/cube",
        "textures": {
            "particle": particle,
            "down": textures.get("bottom", particle),
            "up": textures.get("top", particle),
            "north": textures.get("north", particle),
            "south": textures.get("south", particle),
            "east": textures.get("east", particle),
            "west": textures.get("west", particle),
        },
    }


def _js_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", " ")


def _js_number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if number == int(number):
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _startup(asset_id: str, fields, namespace: str) -> str:
    sound = _token(fields.get("sound"), "stone")
    map_color = _token(fields.get("map_color"), "")
    tool = str(fields.get("tool") or "none")
    if tool not in ("pickaxe", "axe", "shovel", "hoe"):
        tool = "none"
    render = str(fields.get("render") or "solid")
    if render not in ("cutout", "translucent"):
        render = "solid"
    try:
        light = int(fields.get("light", 0))
    except (TypeError, ValueError):
        light = 0
    light = max(0, min(15, light))
    lines = [
        "// Unzip this archive into the kubejs folder.",
        "// startup_scripts/ lands next to the kubejs startup scripts.",
        f"// assets/ uses the {namespace} namespace. The block id is kubejs:{asset_id}.",
        "// A texture with more than one frame uses the .mcmeta file beside the PNG.",
        "// Blockstate variants, when a layer script has Count above 1, are in blockstates/.",
        "StartupEvents.registry('block', event => {",
        f"  event.create('{_js_string(asset_id)}')",
        f"    .displayName('{_js_string(display_name(asset_id))}')",
        f"    .soundType('{sound}')",
    ]
    if map_color:
        lines.append(f"    .mapColor('{map_color}')")
    lines.append(f"    .hardness({_js_number(fields.get('hardness', 1.5))})")
    lines.append(f"    .resistance({_js_number(fields.get('resistance', 6))})")
    if bool(fields.get("requires_tool")):
        lines.append("    .requiresTool(true)")
    if tool != "none":
        lines.append(f"    .tagBlock('minecraft:mineable/{tool}')")
    if render != "solid":
        lines.append(f"    .renderType('{render}')")
    if light:
        lines.append(f"    .lightLevel({light}/15)")
    lines.append("})")
    lines.append("")
    return "\n".join(lines)
