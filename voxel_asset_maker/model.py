"""Layered texture document.

Textures are sized in multiples of 16 so they line up with block and item
slots in games like Minecraft, block-game, and Lunati.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TILE = 16
MAX_SIZE = 4096

BLEND_MODES = (
    "normal",
    "multiply",
    "screen",
    "overlay",
    "add",
    "darken",
    "lighten",
    "mask",
)

DEFAULT_PALETTE = (
    "#000000",
    "#1c1c1c",
    "#3c3c3c",
    "#7f7f7f",
    "#b5b5b5",
    "#ffffff",
    "#5a3818",
    "#8b5a2b",
    "#c4a574",
    "#e6d3a3",
    "#3d6b2f",
    "#5d9b3f",
    "#8fce6b",
    "#245c45",
    "#4a5560",
    "#7d848c",
    "#b7c0c7",
    "#d9d3c7",
    "#243e86",
    "#3f6fd0",
    "#8ec8f0",
    "#8f2d1f",
    "#d4532a",
    "#f0a040",
    "#f2d15a",
    "#6b3fa0",
    "#c44b8a",
    "#f2c2d4",
)

Color = tuple[int, int, int, int]

# What a document is. Block and item export a block-game definition.
# A set paints one strip and exports a whole family of blocks and items.
# Further kinds can be added without changing ones that are already saved.
ASSET_KINDS = ("texture", "block", "item", "crop", "door", "trapdoor", "set", "custom")


def asset_token(value: object, fallback: str = "") -> str:
    """A block-game id: lowercase letters, digits, and underscores."""
    text = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value).strip().lower())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")[:64] or fallback


def asset_token_list(value: object, fallback: str) -> str:
    """Space-separated ids. Commas are separators too."""
    parts: list[str] = []
    for chunk in str(value).replace(",", " ").split():
        token = asset_token(chunk, "")
        if token and token not in parts:
            parts.append(token)
    return " ".join(parts) if parts else fallback

# Unfolded cube. Each face is one 16×16 tile:
#        top
# west north east south
#       bottom
BLOCK_FACE_ORIGIN: dict[str, tuple[int, int]] = {
    "top": (TILE, 0),
    "west": (0, TILE),
    "north": (TILE, TILE),
    "east": (TILE * 2, TILE),
    "south": (TILE * 3, TILE),
    "bottom": (TILE, TILE * 2),
}
BLOCK_SHEET = (TILE * 4, TILE * 3)
# How a block's textures are laid out. six is the unfolded cube. all is one
# tile on every face. column is top, side, and bottom.
BLOCK_FACE_MODES = ("six", "all", "column")

# A door is the lower half, the upper half, and the thin edge. A trapdoor
# is the panel and that same kind of edge. Each part is one 16×16 tile.
DOOR_PARTS = ("lower", "upper", "edge")
TRAPDOOR_PARTS = ("panel", "edge")
DOOR_SHEET = (TILE * len(DOOR_PARTS), TILE)
TRAPDOOR_SHEET = (TILE * len(TRAPDOOR_PARTS), TILE)

# A wood set paints these tiles, left to right. The short label sits on the
# canvas. The caption is the status-bar name. Log top and log side build the
# log and the bark wood block. Planks build the planks, both slabs, the
# stairs, and the vertical-plank versions (that texture turned a quarter
# turn). The door edge is also the trapdoor edge.
WOOD_PARTS: tuple[tuple[str, str, str], ...] = (
    ("log_top", "log top", "Log top"),
    ("log_side", "log side", "Log side"),
    ("stick", "stick", "Stick"),
    ("planks", "planks", "Planks"),
    ("table_top", "table", "Table top"),
    ("table_front", "front", "Table front"),
    ("table_side", "t. side", "Table side"),
    ("door_lower", "lower", "Door lower"),
    ("door_upper", "upper", "Door upper"),
    ("door_edge", "edge", "Door edge"),
    ("trapdoor", "trap", "Trapdoor"),
)
WOOD_SHEET = (TILE * len(WOOD_PARTS), TILE)

# Preview order. The id is ``{wood}_{key}``, except vertical planks, which
# are ``vertical_{wood}_planks`` and so on, matching block-game.
WOOD_MEMBERS: tuple[tuple[str, str], ...] = (
    ("log", "Log"),
    ("wood", "Wood"),
    ("stick", "Stick"),
    ("planks", "Planks"),
    ("planks_slab", "Slab"),
    ("planks_vertical_slab", "Vertical slab"),
    ("planks_stairs", "Stairs"),
    ("vertical_planks", "Vertical planks"),
    ("vertical_planks_slab", "Vertical planks slab"),
    ("vertical_planks_vertical_slab", "Vertical planks vertical slab"),
    ("vertical_planks_stairs", "Vertical planks stairs"),
    ("crafting_table", "Crafting table"),
    ("door", "Door"),
    ("trapdoor", "Trapdoor"),
)
SET_FAMILIES = ("wood",)

# Connected textures. Sixteen tiles in a 4×4 sheet. The index is a mask of
# which edges of the face touch the same block: up=1, down=2, left=4, right=8.
# Up, down, left, and right follow the face itself, the same directions the
# block-game uses when it picks ctm_0 … ctm_15.
CTM_SHEET = (TILE * 4, TILE * 4)


def ctm_origin(index: int) -> tuple[int, int]:
    index = int(index) % 16
    return (index % 4) * TILE, (index // 4) * TILE


def ctm_at(x: int, y: int) -> int | None:
    if not (0 <= x < CTM_SHEET[0] and 0 <= y < CTM_SHEET[1]):
        return None
    return (int(y) // TILE) * 4 + (int(x) // TILE)


def ctm_label(index: int) -> str:
    """Words for a connection mask: 'up+left', or 'alone' when nothing connects."""
    parts = []
    if index & 1:
        parts.append("up")
    if index & 2:
        parts.append("down")
    if index & 4:
        parts.append("left")
    if index & 8:
        parts.append("right")
    return "+".join(parts) if parts else "alone"


def ctm_short(index: int) -> str:
    text = ""
    if index & 1:
        text += "U"
    if index & 2:
        text += "D"
    if index & 4:
        text += "L"
    if index & 8:
        text += "R"
    return text or "·"


def part_at(x: int, y: int, count: int) -> int | None:
    """Which door or trapdoor tile contains this pixel, or None off the strip."""
    if count < 1 or not (0 <= y < TILE and 0 <= x < count * TILE):
        return None
    return int(x) // TILE


def read_part(image: np.ndarray, index: int) -> np.ndarray:
    """One 16×16 tile from a horizontal strip."""
    out = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    origin_x = int(index) * TILE
    height, width = image.shape[:2]
    if origin_x + TILE <= width and TILE <= height:
        out[:, :] = image[0:TILE, origin_x : origin_x + TILE]
    return out


def read_ctm(image: np.ndarray, index: int) -> np.ndarray:
    origin_x, origin_y = ctm_origin(index)
    out = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    height, width = image.shape[:2]
    if origin_y + TILE <= height and origin_x + TILE <= width:
        out[:, :] = image[origin_y : origin_y + TILE, origin_x : origin_x + TILE]
    return out


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def validate_size(width: int, height: int) -> tuple[int, int]:
    """Return width and height when both are legal texture dimensions."""
    width = int(width)
    height = int(height)
    if width < TILE or height < TILE:
        raise ValueError(f"Texture size has to be at least {TILE}×{TILE}.")
    if width > MAX_SIZE or height > MAX_SIZE:
        raise ValueError(f"Texture size has to be at most {MAX_SIZE}px on a side.")
    if width % TILE != 0 or height % TILE != 0:
        raise ValueError(
            f"Texture size has to be a multiple of {TILE} pixels (got {width}×{height})."
        )
    return width, height


def empty_pixels(width: int, height: int, color: Color | None = None) -> np.ndarray:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    if color is not None:
        arr[:, :] = color
    return arr


def parse_hex_color(text: str) -> Color:
    """Parse #RGB, #RRGGBB, or #RRGGBBAA."""
    raw = text.strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) == 4:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) not in (6, 8) or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
        raise ValueError(f"Can't read color '{text}'.")
    r = int(raw[0:2], 16)
    g = int(raw[2:4], 16)
    b = int(raw[4:6], 16)
    a = int(raw[6:8], 16) if len(raw) == 8 else 255
    return (r, g, b, a)


def format_hex_color(color: Color) -> str:
    r, g, b, a = color
    if a == 255:
        return f"#{r:02x}{g:02x}{b:02x}"
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"


def face_at(x: int, y: int) -> str | None:
    """The unfolded-cube face that contains this pixel, if any."""
    return face_on_block(x, y, "six")


def block_face_origins(mode: str) -> dict[str, tuple[int, int]]:
    """Where each paintable tile sits for a block face layout."""
    if mode == "all":
        return {"all": (0, 0)}
    if mode == "column":
        return {"top": (0, 0), "side": (TILE, 0), "bottom": (TILE * 2, 0)}
    return dict(BLOCK_FACE_ORIGIN)


def block_sheet_size(mode: str) -> tuple[int, int]:
    origins = block_face_origins(mode)
    width = max(origin_x for origin_x, _origin_y in origins.values()) + TILE
    height = max(origin_y for _origin_x, origin_y in origins.values()) + TILE
    return width, height


def face_on_block(x: int, y: int, mode: str) -> str | None:
    """The tile that contains this pixel for a block face layout."""
    for name, (origin_x, origin_y) in block_face_origins(mode).items():
        if origin_x <= x < origin_x + TILE and origin_y <= y < origin_y + TILE:
            return name
    return None


def _tile_at(image: np.ndarray, origin_x: int, origin_y: int) -> np.ndarray:
    out = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    height, width = image.shape[:2]
    if origin_y < height and origin_x < width:
        patch = image[origin_y : origin_y + TILE, origin_x : origin_x + TILE]
        out[: patch.shape[0], : patch.shape[1]] = patch
    return out


def resolve_cube_faces(image: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    """The six cube faces, however the sheet is laid out."""
    array = np.asarray(image, dtype=np.uint8)
    if mode == "all":
        tile = _tile_at(array, 0, 0)
        return {name: np.array(tile, copy=True) for name in BLOCK_FACE_ORIGIN}
    if mode == "column":
        top = _tile_at(array, 0, 0)
        side = _tile_at(array, TILE, 0)
        bottom = _tile_at(array, TILE * 2, 0)
        faces = {name: np.array(side, copy=True) for name in BLOCK_FACE_ORIGIN}
        faces["top"] = top
        faces["bottom"] = bottom
        return faces
    return {name: read_face(array, name) for name in BLOCK_FACE_ORIGIN}


def expand_block_sheet(image: np.ndarray, mode: str) -> np.ndarray:
    """Lay one texture or a column out onto the unfolded cube."""
    if mode not in ("all", "column"):
        array = np.asarray(image, dtype=np.uint8)
        if array.shape[1] == BLOCK_SHEET[0] and array.shape[0] == BLOCK_SHEET[1]:
            return np.array(array, copy=True)
        return block_sheet_from_image(array)
    sheet = np.zeros((BLOCK_SHEET[1], BLOCK_SHEET[0], 4), dtype=np.uint8)
    for name, tile in resolve_cube_faces(image, mode).items():
        write_face(sheet, name, tile)
    return sheet


def relayout_block_pixels(pixels: np.ndarray, old: str, new: str, shape: str = "cube") -> np.ndarray:
    """Move painted faces from one block layout onto another."""
    old = old if old in BLOCK_FACE_MODES else "six"
    new = new if new in BLOCK_FACE_MODES else "six"
    width, height = block_sheet_size(new)
    if old == new and pixels.shape[1] == width and pixels.shape[0] == height:
        return np.array(pixels, copy=True)
    faces = resolve_cube_faces(pixels, old)
    sheet = np.zeros((height, width, 4), dtype=np.uint8)
    if new == "all":
        if old == "column":
            tile = faces["north"]
        elif shape == "cross":
            tile = faces["south"]
        else:
            tile = faces["top"]
        sheet[:, :] = tile
        return sheet
    if new == "column":
        side = faces["south"] if shape == "cross" else faces["north"]
        sheet[0:TILE, 0:TILE] = faces["top"]
        sheet[0:TILE, TILE : TILE * 2] = side
        sheet[0:TILE, TILE * 2 : TILE * 3] = faces["bottom"]
        return sheet
    for name, tile in faces.items():
        write_face(sheet, name, tile)
    return sheet


def remap_block_parts(parts: tuple[str, ...] | list[str], mode: str) -> tuple[str, ...]:
    """Keep a script's part list pointed at tiles that exist in this layout."""
    if not parts:
        return ()
    mode = mode if mode in BLOCK_FACE_MODES else "six"
    if mode == "all":
        return ("all",)
    chosen: list[str] = []

    def add(name: str) -> None:
        if name not in chosen:
            chosen.append(name)

    if mode == "column":
        for part in parts:
            if part in ("top", "bottom", "side"):
                add(part)
            elif part in ("north", "south", "east", "west", "all"):
                add("side")
        return tuple(chosen)
    for part in parts:
        if part == "all":
            names = list(BLOCK_FACE_ORIGIN)
        elif part == "side":
            names = ["north", "south", "east", "west"]
        elif part in BLOCK_FACE_ORIGIN:
            names = [part]
        else:
            names = []
        for name in names:
            add(name)
    return tuple(chosen)


def stage_at(x: int, y: int, stages: int) -> int | None:
    """Index of the growth stage under a pixel, or None off the strip."""
    if y < 0 or y >= TILE or x < 0:
        return None
    index = int(x) // TILE
    if index >= int(stages):
        return None
    return index


def read_stage(image: np.ndarray, index: int) -> np.ndarray:
    """Copy one 16×16 growth stage out of a crop strip."""
    tile = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    origin = int(index) * TILE
    height, width = image.shape[:2]
    if origin + TILE <= width and TILE <= height:
        tile[:, :] = image[0:TILE, origin : origin + TILE]
    return tile


def read_face(image: np.ndarray, name: str) -> np.ndarray:
    """Copy one 16×16 face out of a block sheet."""
    origin_x, origin_y = BLOCK_FACE_ORIGIN[name]
    out = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    height, width = image.shape[:2]
    if origin_y + TILE <= height and origin_x + TILE <= width:
        out[:, :] = image[origin_y : origin_y + TILE, origin_x : origin_x + TILE]
    return out


def write_face(image: np.ndarray, name: str, tile: np.ndarray) -> None:
    origin_x, origin_y = BLOCK_FACE_ORIGIN[name]
    image[origin_y : origin_y + TILE, origin_x : origin_x + TILE] = tile[:TILE, :TILE]


def block_sheet_from_image(image: np.ndarray) -> np.ndarray:
    """Lay an existing texture onto the six faces of a block sheet.

    A 16×16 image is copied onto every face. A horizontal or vertical strip
    of tiles is read as top, side, then bottom, and the side is copied onto
    north, south, east, and west. A sheet that is already the block size
    keeps the pixels that sit on each face.
    """
    array = np.asarray(image, dtype=np.uint8)
    sheet = np.zeros((BLOCK_SHEET[1], BLOCK_SHEET[0], 4), dtype=np.uint8)
    height, width = array.shape[:2]
    if (width, height) == BLOCK_SHEET:
        for name in BLOCK_FACE_ORIGIN:
            write_face(sheet, name, read_face(array, name))
        return sheet

    def tile(col: int, row: int) -> np.ndarray:
        out = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        x = col * TILE
        y = row * TILE
        if y < height and x < width:
            patch = array[y : y + TILE, x : x + TILE]
            out[: patch.shape[0], : patch.shape[1]] = patch
        return out

    cols = width // TILE
    rows = height // TILE
    aligned = width % TILE == 0 and height % TILE == 0
    if aligned and cols >= 2 and rows == 1:
        top = tile(0, 0)
        side = tile(1, 0)
        bottom = tile(2, 0) if cols >= 3 else top.copy()
    elif aligned and rows >= 2 and cols == 1:
        top = tile(0, 0)
        side = tile(0, 1)
        bottom = tile(0, 2) if rows >= 3 else top.copy()
    else:
        top = side = bottom = tile(0, 0)
    for name, src in (
        ("top", top),
        ("bottom", bottom),
        ("north", side),
        ("south", side.copy()),
        ("east", side.copy()),
        ("west", side.copy()),
    ):
        write_face(sheet, name, src)
    return sheet


def mask_block_faces(image: np.ndarray) -> np.ndarray:
    """Clear every pixel that is not on a block face."""
    out = np.array(image, copy=True)
    if out.shape[1] != BLOCK_SHEET[0] or out.shape[0] != BLOCK_SHEET[1]:
        return out
    keep = np.zeros(out.shape[:2], dtype=bool)
    for origin_x, origin_y in BLOCK_FACE_ORIGIN.values():
        keep[origin_y : origin_y + TILE, origin_x : origin_x + TILE] = True
    out[~keep] = 0
    return out


def refit(pixels: np.ndarray, width: int, height: int, anchor: str = "topleft") -> np.ndarray:
    """Copy pixels into a new canvas, keeping the overlapping region."""
    new = np.zeros((height, width, 4), dtype=np.uint8)
    src_h, src_w = pixels.shape[:2]
    if anchor == "center":
        ox = (width - src_w) // 2
        oy = (height - src_h) // 2
    else:
        ox = 0
        oy = 0
    src_x0 = max(0, -ox)
    src_y0 = max(0, -oy)
    dst_x0 = max(0, ox)
    dst_y0 = max(0, oy)
    copy_w = min(src_w - src_x0, width - dst_x0)
    copy_h = min(src_h - src_y0, height - dst_y0)
    if copy_w > 0 and copy_h > 0:
        new[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = pixels[
            src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
        ]
    return new


@dataclass(frozen=True)
class SheetPart:
    """One paintable cell a layer script can be limited to."""

    id: str
    label: str
    x: int
    y: int
    w: int
    h: int


def sheet_parts(doc: Document, tiles: tuple | list = ()) -> tuple[SheetPart, ...]:
    """Named cells on this sheet. A plain texture or item has none.

    ``tiles`` supplies a custom definition's cells. Built-in sheets use
    their own layout.
    """
    if doc.kind == "custom":
        parts: list[SheetPart] = []
        for tile in tiles:
            parts.append(
                SheetPart(
                    str(getattr(tile, "id", "")),
                    str(getattr(tile, "label", "") or getattr(tile, "id", "")),
                    int(getattr(tile, "x", 0)),
                    int(getattr(tile, "y", 0)),
                    int(getattr(tile, "w", TILE)),
                    int(getattr(tile, "h", TILE)),
                )
            )
        return tuple(part for part in parts if part.id and part.w > 0 and part.h > 0)
    if doc.kind == "set":
        return tuple(
            SheetPart(key, label, index * TILE, 0, TILE, TILE)
            for index, (key, label, _caption) in enumerate(WOOD_PARTS)
        )
    if doc.kind == "door":
        return tuple(SheetPart(name, name, index * TILE, 0, TILE, TILE) for index, name in enumerate(DOOR_PARTS))
    if doc.kind == "trapdoor":
        return tuple(
            SheetPart(name, name, index * TILE, 0, TILE, TILE) for index, name in enumerate(TRAPDOOR_PARTS)
        )
    if doc.kind == "crop":
        return tuple(
            SheetPart(str(index + 1), str(index + 1), index * TILE, 0, TILE, TILE)
            for index in range(doc.crop.stages)
        )
    if doc.kind == "block":
        return tuple(
            SheetPart(name, name, origin_x, origin_y, TILE, TILE)
            for name, (origin_x, origin_y) in block_face_origins(doc.traits.faces).items()
        )
    return ()


@dataclass
class ScriptBinding:
    script_id: str = "none"
    params: dict[str, Any] = field(default_factory=dict)
    script_path: str = ""
    parts: tuple[str, ...] = ()

    def copy(self) -> ScriptBinding:
        return ScriptBinding(self.script_id, deepcopy(self.params), self.script_path, tuple(self.parts))


@dataclass
class Layer:
    name: str
    pixels: np.ndarray
    id: str = field(default_factory=new_id)
    visible: bool = True
    opacity: float = 1.0
    blend: str = "normal"
    offset: tuple[int, int] = (0, 0)
    scripts: list[ScriptBinding] = field(default_factory=list)

    @property
    def script(self) -> ScriptBinding:
        """The first script on this layer, created if the layer has none yet."""
        if not self.scripts:
            self.scripts.append(ScriptBinding())
        return self.scripts[0]

    def copy(self) -> Layer:
        """A new layer with the same pixels and settings."""
        return Layer(
            name=self.name,
            pixels=self.pixels.copy(),
            id=new_id(),
            visible=self.visible,
            opacity=self.opacity,
            blend=self.blend,
            offset=(int(self.offset[0]), int(self.offset[1])),
            scripts=[binding.copy() for binding in self.scripts],
        )

    def clone_state(self) -> Layer:
        """Deep copy that keeps the id, for undo history."""
        return Layer(
            name=self.name,
            pixels=self.pixels.copy(),
            id=self.id,
            visible=self.visible,
            opacity=float(self.opacity),
            blend=self.blend,
            offset=(int(self.offset[0]), int(self.offset[1])),
            scripts=[binding.copy() for binding in self.scripts],
        )


@dataclass
class Fluorescence:
    """How a block answers ultraviolet light.

    Texture mode paints a glow map. Color mode is one flat color. Start is the
    UV level where the glow begins, full is where it reaches strength.
    """

    enabled: bool = False
    mode: str = "texture"
    color: str = "#9dff57"
    start: int = 1
    full: int = 8
    strength: float = 1.0
    pixels: np.ndarray = field(default_factory=lambda: empty_pixels(TILE, TILE))

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if self.mode not in ("texture", "color"):
            self.mode = "texture"
        try:
            red, green, blue, _alpha = parse_hex_color(self.color)
            self.color = f"#{red:02x}{green:02x}{blue:02x}"
        except ValueError:
            self.color = "#9dff57"
        self.start = int(np.clip(int(self.start), 0, 15))
        self.full = int(np.clip(int(self.full), 0, 15))
        if self.full < self.start:
            self.start, self.full = self.full, self.start
        self.strength = float(np.clip(float(self.strength), 0.0, 4.0))
        if not isinstance(self.pixels, np.ndarray) or self.pixels.shape[:2] != (TILE, TILE):
            source = self.pixels if isinstance(self.pixels, np.ndarray) else empty_pixels(TILE, TILE)
            self.pixels = refit(source, TILE, TILE)
        else:
            self.pixels = np.ascontiguousarray(self.pixels, dtype=np.uint8)

    def copy(self) -> Fluorescence:
        return Fluorescence(
            self.enabled,
            self.mode,
            self.color,
            self.start,
            self.full,
            self.strength,
            self.pixels.copy(),
        )


BLOCK_SHAPES = ("cube", "slab", "vertical_slab", "stairs", "fence", "wall", "cross")
BLOCK_ROTATIONS = ("", "axis", "horizontal")
BLOCK_STATIONS = ("", "crafting_table", "furnace")
BLOCK_TABS = ("building", "natural", "machines")
BLOCK_TOOLS = ("none", "pickaxe", "axe", "shovel")
BLOCK_SOUNDS = ("stone", "wood", "grass", "dirt", "gravel", "soil")
BLOCK_RENDERS = ("solid", "cutout", "translucent")
SHAPE_LABELS = {
    "cube": "Cube",
    "slab": "Slab",
    "vertical_slab": "Vertical slab",
    "stairs": "Stairs",
    "fence": "Fence",
    "wall": "Wall",
    "cross": "Plant",
}
ROTATION_LABELS = {
    "": "None",
    "axis": "Axis",
    "horizontal": "Horizontal",
}
STATION_LABELS = {
    "": "None",
    "crafting_table": "Crafting table",
    "furnace": "Furnace",
}


@dataclass
class BlockTraits:
    """Fields written into a block-game block definition."""

    shape: str = "cube"
    tab: str = "building"
    hardness: float = 1.0
    tool: str = "pickaxe"
    mining_level: int = 0
    sound: str = "stone"
    render: str = "solid"
    solid: bool = True
    gravity: bool = False
    light: int = 0
    full_block: str = ""
    rotation: str = ""
    station: str = ""
    recipe_types: str = ""
    cook_speed: float = 1.0
    faces: str = "six"

    def __post_init__(self) -> None:
        self.shape = self.shape if self.shape in BLOCK_SHAPES else "cube"
        self.tab = asset_token(self.tab, "building")
        self.tool = self.tool if self.tool in BLOCK_TOOLS else "pickaxe"
        self.sound = asset_token(self.sound, "stone")
        self.render = self.render if self.render in BLOCK_RENDERS else "solid"
        self.hardness = round(float(np.clip(float(self.hardness), 0.0, 100.0)), 2)
        self.mining_level = int(np.clip(int(self.mining_level), 0, 4))
        self.light = int(np.clip(int(self.light), 0, 15))
        self.solid = bool(self.solid)
        self.gravity = bool(self.gravity)
        self.full_block = asset_token(self.full_block, "")
        self.rotation = self.rotation if self.rotation in BLOCK_ROTATIONS else ""
        self.station = self.station if self.station in BLOCK_STATIONS else ""
        self.recipe_types = asset_token_list(self.recipe_types, "")
        self.cook_speed = round(float(np.clip(float(self.cook_speed), 0.25, 16.0)), 2)
        self.faces = self.faces if self.faces in BLOCK_FACE_MODES else "six"

    def copy(self) -> BlockTraits:
        return BlockTraits(
            self.shape,
            self.tab,
            self.hardness,
            self.tool,
            self.mining_level,
            self.sound,
            self.render,
            self.solid,
            self.gravity,
            self.light,
            self.full_block,
            self.rotation,
            self.station,
            self.recipe_types,
            self.cook_speed,
            self.faces,
        )


@dataclass(frozen=True)
class LootDrop:
    """One entry in a crop's ripe loot table.

    An empty item means the produce on the first row and the seeds on the
    second. ``chance`` is 0 to 1; 1 always drops.
    """

    item: str = ""
    minimum: int = 1
    maximum: int = 1
    chance: float = 1.0


DEFAULT_CROP_LOOT = (
    LootDrop(),
    LootDrop(maximum=2),
)


def normalize_loot(raw: object) -> tuple[LootDrop, ...]:
    """Turn a saved list, or a row of drops, into clamped loot entries."""
    if raw is None:
        return DEFAULT_CROP_LOOT
    if isinstance(raw, LootDrop):
        raw = (raw,)
    if not isinstance(raw, (list, tuple)):
        return DEFAULT_CROP_LOOT
    drops: list[LootDrop] = []
    for entry in raw:
        if isinstance(entry, LootDrop):
            item, minimum, maximum, chance = entry.item, entry.minimum, entry.maximum, entry.chance
        elif isinstance(entry, dict):
            minimum = entry.get("minimum", entry.get("min", 1))
            maximum = entry.get("maximum", entry.get("max", minimum))
            item = entry.get("item", entry.get("name", ""))
            chance = entry.get("chance", 1.0)
        else:
            continue
        try:
            low = int(minimum)
            high = int(maximum)
            odds = float(chance)
        except (TypeError, ValueError):
            continue
        low = int(np.clip(low, 1, 64))
        high = int(np.clip(high, 1, 64))
        if high < low:
            low, high = high, low
        odds = round(float(np.clip(odds, 0.0, 1.0)), 2)
        drops.append(LootDrop(asset_token(str(item or ""), ""), low, high, odds))
    return tuple(drops)


@dataclass
class CropTraits:
    """Fields written into a block-game crop definition.

    A crop is a row of plant sprites, one per growth stage, plus the soil
    it can root in. Stage blocks are named ``{id}_0`` onward. ``loot`` is
    what a ripe plant drops. An unripe plant drops one seed.
    """

    stages: int = 4
    min_time: float = 180.0
    max_time: float = 340.0
    needs_water: bool = True
    water_radius: int = 4
    water_vertical: int = 1
    seeds: str = ""
    produce: str = ""
    ground: str = "farmland"
    seed_tab: str = "natural"
    produce_tab: str = "materials"
    loot: tuple[LootDrop, ...] | None = None

    def __post_init__(self) -> None:
        self.stages = int(np.clip(int(self.stages), 2, 16))
        self.min_time = round(float(np.clip(float(self.min_time), 1.0, 10000.0)), 1)
        self.max_time = round(float(np.clip(float(self.max_time), 1.0, 10000.0)), 1)
        if self.max_time < self.min_time:
            self.min_time, self.max_time = self.max_time, self.min_time
        self.needs_water = bool(self.needs_water)
        self.water_radius = int(np.clip(int(self.water_radius), 0, 16))
        self.water_vertical = int(np.clip(int(self.water_vertical), 0, 8))
        self.seeds = asset_token(self.seeds, "")
        self.produce = asset_token(self.produce, "")
        self.ground = asset_token_list(self.ground, "farmland")
        self.seed_tab = asset_token(self.seed_tab, "natural")
        self.produce_tab = asset_token(self.produce_tab, "materials")
        self.loot = normalize_loot(self.loot)

    def copy(self) -> CropTraits:
        return CropTraits(
            self.stages,
            self.min_time,
            self.max_time,
            self.needs_water,
            self.water_radius,
            self.water_vertical,
            self.seeds,
            self.produce,
            self.ground,
            self.seed_tab,
            self.produce_tab,
            self.loot,
        )


@dataclass
class Document:
    width: int
    height: int
    name: str = "texture"
    layers: list[Layer] = field(default_factory=list)
    active_index: int = 0
    palette: list[str] = field(default_factory=list)
    kind: str = "texture"
    family: str = ""
    definition: str = ""
    asset: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    ctm: bool = False
    ctm_pixels: np.ndarray = field(default_factory=lambda: empty_pixels(*CTM_SHEET))
    fluor: Fluorescence = field(default_factory=Fluorescence)
    traits: BlockTraits = field(default_factory=BlockTraits)
    crop: CropTraits = field(default_factory=CropTraits)

    def __post_init__(self) -> None:
        self.kind = (self.kind or "texture").strip().lower() or "texture"
        self.family = asset_token(self.family, "")
        self.definition = asset_token(self.definition, "")
        self.asset = asset_token(self.asset, "")
        self.fields = _plain_fields(self.fields)
        if not isinstance(self.crop, CropTraits):
            self.crop = CropTraits()
        else:
            self.crop.__post_init__()
        if self.kind == "crop":
            self.width = self.crop.stages * TILE
            self.height = TILE
            self.family = ""
            self._clear_definition()
        elif self.kind == "set":
            if self.family not in SET_FAMILIES:
                self.family = "wood"
            self.width, self.height = WOOD_SHEET
            self._clear_definition()
        elif self.kind == "custom":
            self.family = ""
            self.width, self.height = validate_size(self.width, self.height)
        else:
            self.family = ""
            self._clear_definition()
            self.width, self.height = validate_size(self.width, self.height)
        if not self.name.strip():
            self.name = "texture"
        if not self.layers:
            self.layers.append(Layer("base", empty_pixels(self.width, self.height)))
        if not self.palette:
            self.palette = list(DEFAULT_PALETTE)
        self.active_index = max(0, min(self.active_index, len(self.layers) - 1))
        for layer in self.layers:
            if layer.pixels.shape[0] != self.height or layer.pixels.shape[1] != self.width:
                layer.pixels = refit(layer.pixels, self.width, self.height)
            if layer.blend not in BLEND_MODES:
                layer.blend = "normal"
            layer.offset = (int(layer.offset[0]), int(layer.offset[1]))
            layer.opacity = float(np.clip(layer.opacity, 0.0, 1.0))
        self.ctm = bool(self.ctm)
        if not isinstance(self.ctm_pixels, np.ndarray) or self.ctm_pixels.shape[:2] != (CTM_SHEET[1], CTM_SHEET[0]):
            source = self.ctm_pixels if isinstance(self.ctm_pixels, np.ndarray) else empty_pixels(*CTM_SHEET)
            self.ctm_pixels = refit(source, CTM_SHEET[0], CTM_SHEET[1])
        else:
            self.ctm_pixels = np.ascontiguousarray(self.ctm_pixels, dtype=np.uint8)
        if not isinstance(self.fluor, Fluorescence):
            self.fluor = Fluorescence()
        if not isinstance(self.traits, BlockTraits):
            self.traits = BlockTraits()

    def _clear_definition(self) -> None:
        self.definition = ""
        self.asset = ""
        self.fields = {}

    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active_index]


def _plain_fields(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if type(value) in (str, int, float, bool)}
