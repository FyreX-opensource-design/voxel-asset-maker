"""Block-game definitions exported from a texture, block, or item.

A block document is an unfolded cube: top, bottom, and four sides, each
16×16. Export writes a zip the game can take pieces from. It does not
modify a block-game checkout. Item documents export one flat icon and an
item definition. Other document kinds can be added later and simply do
not export until they grow their own definition.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .model import BLOCK_FACE_ORIGIN, TILE, Document, read_ctm, read_face, read_stage
from .processors import ScriptLibrary
from .render import render_variant, stack_variant_count

ASSET_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class PublishError(Exception):
    """The texture can't be written as a block-game asset."""


@dataclass
class Faces:
    top: np.ndarray
    side: np.ndarray
    bottom: np.ndarray
    single: bool


@dataclass
class ExportedDefinition:
    kind: str
    asset_id: str
    path: Path
    names: list[str]
    errors: list[str]


def sanitize_id(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "asset"
    if not cleaned[0].isalpha():
        cleaned = "a_" + cleaned
    return cleaned[:64]


def display_name(asset_id: str) -> str:
    return " ".join(part.capitalize() for part in asset_id.split("_") if part) or asset_id


def slice_faces(image: np.ndarray) -> Faces:
    """Split a texture into top, side, and bottom tiles."""
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    cols = width // TILE
    rows = height // TILE
    aligned = width % TILE == 0 and height % TILE == 0 and cols >= 1 and rows >= 1

    def tile(col: int, row: int) -> np.ndarray:
        if not aligned:
            patch = array[:TILE, :TILE]
        else:
            x = col * TILE
            y = row * TILE
            patch = array[y : y + TILE, x : x + TILE]
        out = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        copy_h = min(TILE, patch.shape[0])
        copy_w = min(TILE, patch.shape[1])
        if copy_h and copy_w:
            out[:copy_h, :copy_w] = patch[:copy_h, :copy_w]
        return out

    if aligned and cols >= 2 and rows == 1:
        top = tile(0, 0)
        side = tile(1, 0)
        bottom = tile(2, 0) if cols >= 3 else top.copy()
        return Faces(top, side, bottom, False)
    if aligned and rows >= 2 and cols == 1:
        top = tile(0, 0)
        side = tile(0, 1)
        bottom = tile(0, 2) if rows >= 3 else top.copy()
        return Faces(top, side, bottom, False)
    face = tile(0, 0)
    return Faces(face, face.copy(), face.copy(), True)


def _turn_xz(x: int, z: int, yaw: int, last: int) -> tuple[int, int]:
    """Turn a block point clockwise, one quarter turn at a time, seen from above."""
    for _ in range(int(yaw) % 4):
        x, z = z, last - x
    return x, z


def block_view(yaw: int = 0, pitch: int = 0) -> tuple[str, str, str]:
    """The cap, the left face, and the right face at this preview angle.

    Yaw 0 looks from the southeast: south is on the left, east is on the right,
    and the cap is the top. Each step of yaw walks one face to the right.
    An odd pitch looks from underneath: the bottom face is the cap, and each
    side hangs from the edge that meets it.
    """
    last = 2
    samples = {"east": (last, 1), "west": (0, 1), "south": (1, last), "north": (1, 0)}
    left = "south"
    right = "east"
    for name, (x, z) in samples.items():
        turned_x, turned_z = _turn_xz(x, z, yaw, last)
        if turned_z == last:
            left = name
        if turned_x == last:
            right = name
    cap = "bottom" if int(pitch) % 2 else "top"
    return cap, left, right


def _face_xyz(name: str, col: int, row: int, last: int) -> tuple[int, int, int]:
    """Map a texture pixel onto the block.

    Column 0 is the left of the face and row 0 is its top, matching the
    block-game cube: image-right follows the face's right vector, and image-up
    follows its up vector. On the top face, row 0 is north.
    """
    if name == "east":
        return last, last - row, last - col
    if name == "west":
        return 0, last - row, col
    if name == "south":
        return col, last - row, last
    if name == "north":
        return last - col, last - row, 0
    if name == "bottom":
        return col, 0, last - row
    return col, last, row


def cube_preview(
    top: np.ndarray,
    side: np.ndarray,
    scale: int = 4,
    *,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    bottom: np.ndarray | None = None,
    north: np.ndarray | None = None,
    west: np.ndarray | None = None,
    yaw: int = 0,
    pitch: int = 0,
) -> np.ndarray:
    """An isometric cube from the southeast, turned by yaw and pitch.

    ``side`` fills any face that is not passed on its own. ``left`` is the
    south face and ``right`` is the east face, which is the yaw-0 view.
    """
    tiles = {
        "top": _square(top),
        "bottom": _square(side if bottom is None else bottom),
        "south": _square(side if left is None else left),
        "east": _square(side if right is None else right),
        "north": _square(side if north is None else north),
        "west": _square(side if west is None else west),
    }
    n = int(tiles["top"].shape[0])
    last = n - 1
    cap, left_name, right_name = block_view(yaw, pitch)
    shades = {cap: 1.0 if cap == "top" else 0.62, left_name: 0.92, right_name: 0.72}
    canvas = np.zeros((2 * n, 2 * n, 4), dtype=np.uint8)
    depth = np.full(canvas.shape[:2], -1, dtype=np.int16)
    for name in (right_name, left_name, cap):
        tile = _fit_tile(tiles[name], n)
        factor = shades[name]
        for row in range(n):
            for col in range(n):
                color = tile[row, col]
                if int(color[3]) == 0:
                    continue
                x, y, z = _face_xyz(name, col, row, last)
                x, z = _turn_xz(x, z, yaw, last)
                sx = (x - z) + n
                # From above, world-up points up the image. From below, the
                # underside is the cap and the sides hang beneath it.
                if int(pitch) % 2:
                    sy = (x + z) // 2 + y
                else:
                    sy = (x + z) // 2 + (last - y)
                if not (0 <= sy < canvas.shape[0] and 0 <= sx < canvas.shape[1]):
                    continue
                near = (x + z) * 2 + (n if name == cap else 0)
                if near < int(depth[sy, sx]):
                    continue
                depth[sy, sx] = near
                if factor == 1.0:
                    canvas[sy, sx] = color
                else:
                    rgb = np.clip(np.round(color[:3].astype(np.float32) * factor), 0, 255)
                    canvas[sy, sx, :3] = rgb.astype(np.uint8)
                    canvas[sy, sx, 3] = color[3]
    preview = np.repeat(np.repeat(canvas, max(1, int(scale)), axis=0), max(1, int(scale)), axis=1)
    return on_checker(preview, cell=max(4, max(1, int(scale)) * 2))


# Connection bits follow the face, matching block-game: up=1, down=2, left=4, right=8.
# Up and right are the face's own axes, not world up and world east.
_FACE_UP_RIGHT: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "top": ((0, 0, -1), (1, 0, 0)),
    "bottom": ((0, 0, 1), (1, 0, 0)),
    "south": ((0, 1, 0), (1, 0, 0)),
    "north": ((0, 1, 0), (-1, 0, 0)),
    "east": ((0, 1, 0), (0, 0, -1)),
    "west": ((0, 1, 0), (0, 0, 1)),
}
_FACE_OUT: dict[str, tuple[int, int, int]] = {
    "top": (0, 1, 0),
    "bottom": (0, -1, 0),
    "south": (0, 0, 1),
    "north": (0, 0, -1),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
}
_CLUSTER = ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1))


def _connection_mask(cells: set[tuple[int, int, int]], pos: tuple[int, int, int], face: str) -> int:
    up, right = _FACE_UP_RIGHT[face]
    down = (-up[0], -up[1], -up[2])
    left = (-right[0], -right[1], -right[2])
    mask = 0
    for bit, step in ((1, up), (2, down), (4, left), (8, right)):
        neighbor = (pos[0] + step[0], pos[1] + step[1], pos[2] + step[2])
        if neighbor in cells:
            mask |= bit
    return mask


def _face_hidden(cells: set[tuple[int, int, int]], pos: tuple[int, int, int], face: str) -> bool:
    step = _FACE_OUT[face]
    return (pos[0] + step[0], pos[1] + step[1], pos[2] + step[2]) in cells


def _color_tile(color: str) -> np.ndarray:
    text = color.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    tile = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    if len(text) < 6:
        return tile
    try:
        tile[..., 0] = int(text[0:2], 16)
        tile[..., 1] = int(text[2:4], 16)
        tile[..., 2] = int(text[4:6], 16)
    except ValueError:
        return np.zeros((TILE, TILE, 4), dtype=np.uint8)
    tile[..., 3] = 255
    return tile


def _with_glow(tile: np.ndarray, glow: np.ndarray | None, strength: float, shade: float) -> np.ndarray:
    """Shade the face, then add the fluorescence on top so it still reads as light."""
    base = np.array(tile, dtype=np.float32, copy=True)
    if shade != 1.0:
        base[..., :3] *= float(shade)
    if glow is not None and strength > 0:
        add = np.asarray(glow, dtype=np.float32)
        if add.shape[0] != base.shape[0] or add.shape[1] != base.shape[1]:
            add = add[: base.shape[0], : base.shape[1]]
        add_rgb = add[..., :3] * (add[..., 3:4] / 255.0) * float(strength)
        base[..., :3] += add_rgb
        base[..., 3] = np.maximum(base[..., 3], add[..., 3])
    out = np.clip(np.round(base), 0, 255).astype(np.uint8)
    return out


# Boxes are the block-game model elements, in the 0–16 grid. UVs match those
# models. Fence and wall use the inventory models so the rails are visible.
# Faces omitted from a box (the stair step has no underside) are not drawn.
_UV = tuple[int, int, int, int]
_FULL: _UV = (0, 0, 16, 16)
_SLAB_SIDE: _UV = (0, 8, 16, 16)
_SHAPE_BOXES: dict[str, tuple[tuple, ...]] = {
    "slab": (
        (0, 0, 0, 16, 8, 16, {
            "north": _SLAB_SIDE, "south": _SLAB_SIDE, "east": _SLAB_SIDE, "west": _SLAB_SIDE,
            "top": _FULL, "bottom": _FULL,
        }),
    ),
    "vertical_slab": (
        (0, 0, 8, 16, 16, 16, {
            "east": (0, 0, 8, 16),
            "west": (8, 0, 16, 16),
            "top": (0, 8, 16, 16),
            "bottom": (0, 0, 16, 8),
            "south": _FULL,
            "north": _FULL,
        }),
    ),
    "stairs": (
        (0, 0, 0, 16, 8, 16, {
            "north": _SLAB_SIDE, "south": _SLAB_SIDE, "east": _SLAB_SIDE, "west": _SLAB_SIDE,
            "top": _FULL, "bottom": _FULL,
        }),
        (0, 8, 0, 16, 16, 8, {
            "top": (0, 0, 16, 8),
            "north": (0, 0, 16, 8),
            "south": (0, 0, 16, 8),
            "west": (0, 0, 8, 8),
            "east": (8, 0, 16, 8),
        }),
    ),
    "fence": (
        (0, 0, 6, 4, 16, 10, {
            "bottom": (0, 6, 4, 10), "top": (0, 6, 4, 10),
            "north": (0, 0, 4, 16), "south": (0, 0, 4, 16),
            "west": (6, 0, 10, 16), "east": (6, 0, 10, 16),
        }),
        (12, 0, 6, 16, 16, 10, {
            "bottom": (12, 6, 16, 10), "top": (12, 6, 16, 10),
            "north": (12, 0, 16, 16), "south": (12, 0, 16, 16),
            "west": (6, 0, 10, 16), "east": (6, 0, 10, 16),
        }),
        (4, 12, 7, 12, 15, 9, {
            "bottom": (4, 7, 12, 9), "top": (4, 7, 12, 9),
            "north": (4, 1, 12, 4), "south": (4, 1, 12, 4),
            "west": (7, 1, 9, 4), "east": (7, 1, 9, 4),
        }),
        (4, 6, 7, 12, 9, 9, {
            "bottom": (4, 7, 12, 9), "top": (4, 7, 12, 9),
            "north": (4, 7, 12, 10), "south": (4, 7, 12, 10),
            "west": (7, 7, 9, 10), "east": (7, 7, 9, 10),
        }),
    ),
    "wall": (
        (4, 0, 4, 12, 16, 12, {
            "bottom": (4, 4, 12, 12), "top": (4, 4, 12, 12),
            "north": (4, 0, 12, 16), "south": (4, 0, 12, 16),
            "west": (4, 0, 12, 16), "east": (4, 0, 12, 16),
        }),
        (5, 0, 0, 11, 13, 16, {
            "bottom": (5, 0, 11, 16), "top": (5, 0, 11, 16),
            "north": (5, 3, 11, 16), "south": (5, 3, 11, 16),
            "west": (0, 3, 16, 16), "east": (0, 3, 16, 16),
        }),
    ),
}
_SHAPE_EXPORT = {
    "cube": ("cube", ""),
    "slab": ("slab", "slab"),
    "vertical_slab": ("vertical_slab", "vertical_slab"),
    "stairs": ("stairs", "stairs"),
    "fence": ("fence_post", "fence"),
    "wall": ("wall_post", "wall"),
    "cross": ("cross", ""),
}
_FACE_STEP = {
    "top": (0, 1, 0),
    "bottom": (0, -1, 0),
    "south": (0, 0, 1),
    "north": (0, 0, -1),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
}


def _texel(tile: np.ndarray, u: float, v: float) -> np.ndarray:
    height, width = tile.shape[:2]
    col = min(width - 1, max(0, int(u)))
    row = min(height - 1, max(0, int(v)))
    return tile[row, col]


def _box_uv(face: str, x: int, y: int, z: int, bounds: tuple[int, ...], uv: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, z0, x1, y1, z1 = bounds
    u0, v0, u1, v1 = uv
    span_x = max(1, x1 - x0)
    span_y = max(1, y1 - y0)
    span_z = max(1, z1 - z0)
    if face == "south":
        across = (x + 0.5 - x0) / span_x
        down = 1.0 - (y + 0.5 - y0) / span_y
    elif face == "north":
        across = (x1 - (x + 0.5)) / span_x
        down = 1.0 - (y + 0.5 - y0) / span_y
    elif face == "east":
        across = (z1 - (z + 0.5)) / span_z
        down = 1.0 - (y + 0.5 - y0) / span_y
    elif face == "west":
        across = (z + 0.5 - z0) / span_z
        down = 1.0 - (y + 0.5 - y0) / span_y
    elif face == "top":
        across = (x + 0.5 - x0) / span_x
        down = (z + 0.5 - z0) / span_z
    else:
        across = (x + 0.5 - x0) / span_x
        down = (z1 - (z + 0.5)) / span_z
    return u0 + across * (u1 - u0), v0 + down * (v1 - v0)


def _face_cells(face: str, bounds: tuple[int, ...]):
    x0, y0, z0, x1, y1, z1 = bounds
    if face == "south":
        z = z1 - 1
        for x in range(x0, x1):
            for y in range(y0, y1):
                yield x, y, z
    elif face == "north":
        z = z0
        for x in range(x0, x1):
            for y in range(y0, y1):
                yield x, y, z
    elif face == "east":
        x = x1 - 1
        for y in range(y0, y1):
            for z in range(z0, z1):
                yield x, y, z
    elif face == "west":
        x = x0
        for y in range(y0, y1):
            for z in range(z0, z1):
                yield x, y, z
    elif face == "top":
        y = y1 - 1
        for x in range(x0, x1):
            for z in range(z0, z1):
                yield x, y, z
    else:
        y = y0
        for x in range(x0, x1):
            for z in range(z0, z1):
                yield x, y, z


def _shape_preview(
    faces: dict[str, np.ndarray],
    boxes: tuple,
    *,
    yaw: int,
    pitch: int,
    glow: np.ndarray | None,
    glow_strength: float,
    scale: int,
) -> np.ndarray:
    """Draw one block made of the model's boxes, from the same camera as a cube."""
    n = TILE
    last = n - 1
    cap, left_name, right_name = block_view(yaw, pitch)
    visible = {cap, left_name, right_name}
    shades = {cap: 1.0 if cap == "top" else 0.62, left_name: 0.92, right_name: 0.72}
    occupied: set[tuple[int, int, int]] = set()
    for box in boxes:
        x0, y0, z0, x1, y1, z1, _uvs = box
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    occupied.add((x, y, z))
    canvas = np.zeros((n * 2, n * 2, 4), dtype=np.uint8)
    depth = np.full(canvas.shape[:2], -1, dtype=np.int16)
    below = int(pitch) % 2 == 1
    for box in boxes:
        bounds = box[:6]
        uvs = box[6]
        for name in (right_name, left_name, cap):
            if name not in uvs or name not in visible:
                continue
            step = _FACE_STEP[name]
            shade = shades[name]
            for x, y, z in _face_cells(name, bounds):
                outward = (x + step[0], y + step[1], z + step[2])
                if outward in occupied:
                    continue
                u, v = _box_uv(name, x, y, z, bounds, uvs[name])
                sample = _texel(faces[name], u, v).astype(np.float32)
                if shade != 1.0:
                    sample[:3] *= shade
                if glow is not None and glow_strength > 0:
                    add = _texel(glow, u, v).astype(np.float32)
                    sample[:3] += add[:3] * (add[3] / 255.0) * float(glow_strength)
                    sample[3] = max(float(sample[3]), float(add[3]))
                if sample[3] <= 0:
                    continue
                color = np.clip(np.round(sample), 0, 255).astype(np.uint8)
                turned_x, turned_z = _turn_xz(x, z, yaw, last)
                sx = (turned_x - turned_z) + n
                if below:
                    sy = (turned_x + turned_z) // 2 + y
                else:
                    sy = (turned_x + turned_z) // 2 + (last - y)
                if not (0 <= sy < canvas.shape[0] and 0 <= sx < canvas.shape[1]):
                    continue
                near = (turned_x + turned_z) * 2 + (n if name == cap else 0)
                if near < int(depth[sy, sx]):
                    continue
                depth[sy, sx] = near
                canvas[sy, sx] = color
    preview = np.repeat(np.repeat(canvas, max(1, int(scale)), axis=0), max(1, int(scale)), axis=1)
    return on_checker(preview, cell=max(4, max(1, int(scale)) * 2))


def _plant_tile(faces: dict[str, np.ndarray]) -> np.ndarray:
    """The plant sprite is the south face, or the first face that has paint."""
    for name in ("south", "north", "east", "west", "top", "bottom"):
        tile = faces[name]
        if int(tile[..., 3].max()) > 0:
            return tile
    return faces["south"]


def _cross_preview(
    tile: np.ndarray,
    *,
    yaw: int,
    pitch: int,
    glow: np.ndarray | None,
    glow_strength: float,
    scale: int,
) -> np.ndarray:
    """Two crossed planes, the way a plant stands in the world.

    The camera sits a little off the diagonal so both planes stay visible.
    Texture row 0 is the top of the plant. One plane runs northwest to
    southeast, the other northeast to southwest.
    """
    angle = int(yaw) * math.pi / 2.0 + math.radians(22)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    below = int(pitch) % 2 == 1
    canvas = np.zeros((48, 48, 4), dtype=np.uint8)
    depth = np.full(canvas.shape[:2], -1e9, dtype=np.float32)

    def project(x: float, y: float, z: float) -> tuple[float, float, float]:
        rx = (x - 7.5) * cos_a - (z - 7.5) * sin_a
        rz = (x - 7.5) * sin_a + (z - 7.5) * cos_a
        sx = (rx - rz) + 24.0
        if below:
            sy = (rx + rz) * 0.5 + y + 8.0
        else:
            sy = (rx + rz) * 0.5 + ((TILE - 1) - y) + 8.0
        return sx, sy, rx + rz

    def paint(sx: int, sy: int, near: float, color: np.ndarray) -> None:
        if not (0 <= sy < canvas.shape[0] and 0 <= sx < canvas.shape[1]):
            return
        if near < float(depth[sy, sx]):
            return
        depth[sy, sx] = near
        canvas[sy, sx] = color

    shades = (0.82, 1.0)
    for plane in (0, 1):
        shade = shades[plane]
        for row in range(TILE):
            for col in range(TILE):
                sample = tile[row, col].astype(np.float32)
                if shade != 1.0:
                    sample[:3] *= shade
                if glow is not None and glow_strength > 0:
                    add = _texel(glow, col, row).astype(np.float32)
                    sample[:3] += add[:3] * (add[3] / 255.0) * float(glow_strength)
                    sample[3] = max(float(sample[3]), float(add[3]))
                if sample[3] <= 0:
                    continue
                color = np.clip(np.round(sample), 0, 255).astype(np.uint8)
                y0 = float((TILE - 1) - row)
                y1 = y0 - 1.0
                if plane == 0:
                    corners = ((col, y0, col), (col + 1, y0, col + 1), (col + 1, y1, col + 1), (col, y1, col))
                else:
                    z0 = (TILE - 1) - col
                    corners = (
                        (col, y0, z0),
                        (col + 1, y0, z0 - 1),
                        (col + 1, y1, z0 - 1),
                        (col, y1, z0),
                    )
                projected = [project(x, y, z) for x, y, z in corners]
                xs = [point[0] for point in projected]
                ys = [point[1] for point in projected]
                near = max(point[2] for point in projected) + plane * 0.01
                min_x = max(0, int(math.floor(min(xs))))
                max_x = min(canvas.shape[1] - 1, int(math.ceil(max(xs))))
                min_y = max(0, int(math.floor(min(ys))))
                max_y = min(canvas.shape[0] - 1, int(math.ceil(max(ys))))
                if max_x < min_x or max_y < min_y:
                    sx, sy, _near = projected[0]
                    paint(int(round(sx)), int(round(sy)), near, color)
                    continue
                for sy in range(min_y, max_y + 1):
                    for sx in range(min_x, max_x + 1):
                        paint(sx, sy, near, color)
    preview = np.repeat(np.repeat(canvas, max(1, int(scale)), axis=0), max(1, int(scale)), axis=1)
    return on_checker(preview, cell=max(4, max(1, int(scale)) * 2))


def block_preview(
    image: np.ndarray,
    *,
    yaw: int = 0,
    pitch: int = 0,
    ctm: np.ndarray | None = None,
    glow: np.ndarray | None = None,
    glow_color: str | None = None,
    glow_strength: float = 0.0,
    scale: int | None = None,
    shape: str = "cube",
) -> np.ndarray:
    """The block as it sits in the world.

    A cube with connected textures draws four blocks that share edges, and
    each face uses the tile for those neighbors. Slabs, stairs, fences, and
    walls draw that model's boxes. Fluorescence is added at full ultraviolet,
    scaled by strength. A flat glow color fills every face.
    """
    array = np.asarray(image, dtype=np.uint8)
    faces = {name: read_face(array, name) for name in BLOCK_FACE_ORIGIN}
    if glow is None and glow_color and glow_strength > 0:
        glow = _color_tile(glow_color)
    glow_tile = None if glow is None or glow_strength <= 0 else _fit_tile(np.asarray(glow, dtype=np.uint8), TILE)
    if shape == "cross":
        chosen = 4 if scale is None else max(1, int(scale))
        return _cross_preview(
            _plant_tile(faces),
            yaw=yaw,
            pitch=pitch,
            glow=glow_tile,
            glow_strength=glow_strength,
            scale=chosen,
        )
    if shape in _SHAPE_BOXES:
        chosen = 4 if scale is None else max(1, int(scale))
        return _shape_preview(
            faces,
            _SHAPE_BOXES[shape],
            yaw=yaw,
            pitch=pitch,
            glow=glow_tile,
            glow_strength=glow_strength,
            scale=chosen,
        )
    use_ctm = ctm is not None
    cells = set(_CLUSTER) if use_ctm else {(0, 0, 0)}
    span = 2 if use_ctm else 1
    n = TILE
    last = n - 1
    horizontal = span * n - 1
    cap, left_name, right_name = block_view(yaw, pitch)
    shades = {cap: 1.0 if cap == "top" else 0.62, left_name: 0.92, right_name: 0.72}
    width = span * n * 2
    height = span * n + n
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    depth = np.full(canvas.shape[:2], -1, dtype=np.int16)
    below = int(pitch) % 2 == 1
    for pos in cells:
        for name in (right_name, left_name, cap):
            if _face_hidden(cells, pos, name):
                continue
            if use_ctm:
                tile = read_ctm(np.asarray(ctm), _connection_mask(cells, pos, name))
            else:
                tile = faces[name]
            painted = _with_glow(tile, glow_tile, glow_strength, shades[name])
            bx, _by, bz = pos
            for row in range(n):
                for col in range(n):
                    color = painted[row, col]
                    if int(color[3]) == 0:
                        continue
                    local_x, local_y, local_z = _face_xyz(name, col, row, last)
                    x = local_x + bx * n
                    z = local_z + bz * n
                    y = local_y
                    x, z = _turn_xz(x, z, yaw, horizontal)
                    sx = (x - z) + span * n
                    if below:
                        sy = (x + z) // 2 + y
                    else:
                        sy = (x + z) // 2 + (last - y)
                    if not (0 <= sy < canvas.shape[0] and 0 <= sx < canvas.shape[1]):
                        continue
                    near = (x + z) * 2 + (n if name == cap else 0)
                    if near < int(depth[sy, sx]):
                        continue
                    depth[sy, sx] = near
                    canvas[sy, sx] = color
    chosen = 2 if use_ctm else 4
    if scale is not None:
        chosen = max(1, int(scale))
    preview = np.repeat(np.repeat(canvas, chosen, axis=0), chosen, axis=1)
    return on_checker(preview, cell=max(4, chosen * 2))


def sprite_preview(image: np.ndarray, pixels: int = 96) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    if height < 1 or width < 1:
        array = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        height, width = TILE, TILE
    scale = max(1, min(8, pixels // max(height, width)))
    scaled = np.repeat(np.repeat(array, scale, axis=0), scale, axis=1)
    return on_checker(scaled, cell=max(4, scale))


def _crop_package(
    asset_id: str,
    image: np.ndarray,
    doc: Document,
) -> tuple[dict, dict[str, np.ndarray], dict[str, dict]]:
    """A crop file plus one plant sprite per growth stage.

    Seed and produce items are included when their file names don't collide
    with the crop file. Their icons are the first and last stages.
    """
    crop = doc.crop
    stages = int(crop.stages)
    seeds = crop.seeds or f"{asset_id}_seeds"
    produce = crop.produce or asset_id
    if seeds == produce:
        seeds = f"{asset_id}_seeds" if produce != f"{asset_id}_seeds" else f"{asset_id}_seed"
    files: dict[str, np.ndarray] = {}
    for index in range(stages):
        files[f"{asset_id}_{index}"] = read_stage(image, index)
    if seeds not in files:
        files[seeds] = read_stage(image, 0)
    if produce not in files:
        files[produce] = read_stage(image, stages - 1)
    payload = {
        "id": asset_id,
        "stages": stages,
        "min_time": _json_number(crop.min_time),
        "max_time": _json_number(crop.max_time),
        "needs_water": bool(crop.needs_water),
        "water_radius": int(crop.water_radius),
        "water_vertical": int(crop.water_vertical),
        "seeds": seeds,
        "produce": produce,
        "food": int(crop.food),
        "ground": crop.ground.split(),
        "seed_tab": crop.seed_tab,
        "produce_tab": crop.produce_tab,
    }
    extra: dict[str, dict] = {}
    taken = {f"{asset_id}.json"}
    seed_name = f"{seeds}.json"
    if seed_name in taken:
        seed_name = f"{seeds}.item.json"
    taken.add(seed_name)
    extra[seed_name] = {
            "id": seeds,
            "lang": f"item.{seeds}",
            "tab": crop.seed_tab,
            "model": "flat",
            "texture": seeds,
            "block": f"{asset_id}_0",
            "max_stack": 64,
            "left": "none",
            "right": "place",
            "food": 0,
        }
    produce_name = f"{produce}.json"
    if produce_name in taken:
        produce_name = f"{produce}.item.json"
    extra[produce_name] = {
            "id": produce,
            "lang": f"item.{produce}",
            "tab": crop.produce_tab,
            "model": "flat",
            "texture": produce,
            "max_stack": 64,
            "left": "none",
            "right": "eat",
            "food": int(crop.food),
        }
    return payload, files, extra


def _json_number(value: float) -> int | float:
    number = float(value)
    if number == int(number):
        return int(number)
    return number


def preview_for(
    image: np.ndarray,
    kind: str,
    *,
    yaw: int = 0,
    pitch: int = 0,
    ctm: np.ndarray | None = None,
    glow: np.ndarray | None = None,
    glow_color: str | None = None,
    glow_strength: float = 0.0,
    scale: int | None = None,
    shape: str = "cube",
    stage: int = 0,
) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if kind == "crop":
        chosen = 4 if scale is None else max(1, int(scale))
        return _cross_preview(
            read_stage(array, stage),
            yaw=yaw,
            pitch=pitch,
            glow=None,
            glow_strength=0.0,
            scale=chosen,
        )
    if kind == "block":
        return block_preview(
            array,
            yaw=yaw,
            pitch=pitch,
            ctm=ctm,
            glow=glow,
            glow_color=glow_color,
            glow_strength=glow_strength,
            scale=scale,
            shape=shape,
        )
    if kind == "item":
        return sprite_preview(array)
    return sprite_preview(array)


def on_checker(image: np.ndarray, cell: int = 8) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    light = ((xx // cell) + (yy // cell)) % 2 == 0
    background = np.empty((height, width, 3), dtype=np.float32)
    background[..., 0] = np.where(light, 58, 36)
    background[..., 1] = np.where(light, 56, 34)
    background[..., 2] = np.where(light, 52, 32)
    src = image.astype(np.float32)
    alpha = (src[..., 3:4] / 255.0)
    rgb = background * (1.0 - alpha) + src[..., :3] * alpha
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def export_definition(
    doc: Document,
    library: ScriptLibrary,
    path: Path,
    asset_id: str | None = None,
    variant_index: int = 1,
) -> ExportedDefinition:
    """Write a zip with a block or item definition and its textures.

    The zip is a file the user asked for. Nothing is written into a
    block-game checkout.
    """
    kind = doc.kind
    if kind not in ("block", "item", "crop"):
        raise PublishError("Only a block, an item, or a crop exports a definition.")
    asset_id = sanitize_id(asset_id or doc.name)
    if not ASSET_ID.match(asset_id):
        raise PublishError(f"'{asset_id}' is not a block-game id.")
    path = Path(path)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    count = stack_variant_count(doc, library)
    variant_index = max(1, min(int(variant_index), count))
    rendered: list[np.ndarray] = []
    errors: list[str] = []
    for index in range(1, count + 1):
        result = render_variant(doc, index, library)
        if index == 1:
            errors.extend(result.errors)
        rendered.append(np.array(result.image, dtype=np.uint8))
    extra_json: dict[str, dict] = {}
    if kind == "block":
        payload, files = _block_package(asset_id, rendered, doc)
    elif kind == "crop":
        payload, files, extra_json = _crop_package(asset_id, rendered[variant_index - 1], doc)
    else:
        payload, files = _item_package(
            asset_id, rendered[variant_index - 1], variant_index, count, doc.traits.tab
        )
    names = _write_definition_zip(path, f"{asset_id}.json", payload, files, extra_json)
    return ExportedDefinition(kind, asset_id, path, names, errors)


def _block_package(
    asset_id: str,
    rendered: list[np.ndarray],
    doc: Document,
) -> tuple[dict, dict[str, np.ndarray]]:
    maps: list[dict[str, str]] = []
    files: dict[str, np.ndarray] = {}
    count = len(rendered)
    for index, image in enumerate(rendered, start=1):
        tiles = {name: read_face(image, name) for name in BLOCK_FACE_ORIGIN}
        mapping, part = _block_textures(asset_id, tiles, index, count)
        maps.append(mapping)
        files.update(part)
    traits = doc.traits
    model_name, block_type = _SHAPE_EXPORT.get(traits.shape, ("cube", ""))
    block = {
        "id": asset_id,
        "lang": f"block.{asset_id}",
        "tab": traits.tab,
        "model": model_name,
        "hardness": traits.hardness,
        "tool": traits.tool,
        "mining_level": int(traits.mining_level),
        "sound": traits.sound,
        "textures": maps[0],
    }
    if block_type:
        block["type"] = block_type
    if traits.shape == "vertical_slab" and traits.full_block:
        block["full_block"] = traits.full_block
    if traits.shape == "fence":
        block["properties"] = {"fence": True}
    if not traits.solid:
        block["solid"] = False
    if traits.gravity:
        block["gravity"] = True
    if traits.light:
        block["light"] = int(traits.light)
    if traits.render != "solid":
        block["render"] = traits.render
    if doc.ctm:
        for mapping in maps:
            for index in range(16):
                tile = read_ctm(doc.ctm_pixels, index)
                mapping[f"ctm_{index}"] = _store_tile(files, f"{asset_id}_ctm_{index}", tile)
        block["ctm"] = True
    if doc.fluor.enabled and doc.fluor.strength > 0:
        fluor = {
            "start": int(doc.fluor.start),
            "full": int(doc.fluor.full),
            "strength": round(float(doc.fluor.strength), 4),
        }
        if doc.fluor.mode == "color":
            fluor["color"] = doc.fluor.color
        else:
            fluor["texture"] = _store_tile(files, f"{asset_id}_glow", np.array(doc.fluor.pixels, copy=True))
        block["fluorescence"] = fluor
    if traits.shape == "cross":
        for mapping in maps:
            mapping["cross"] = (
                mapping.get("south")
                or mapping.get("side")
                or mapping.get("north")
                or mapping.get("top")
            )
        block["plant"] = True
        block["fluid_breakable"] = True
        block["render"] = "translucent" if traits.render == "translucent" else "cutout"
        block["solid"] = bool(traits.solid)
    if traits.shape in ("fence", "wall"):
        for mapping in maps:
            mapping["texture"] = (
                mapping.get("side")
                or mapping.get("south")
                or mapping.get("north")
                or mapping.get("top")
            )
    if count > 1:
        block["palettes"] = maps
    return block, files


def _store_tile(files: dict[str, np.ndarray], preferred: str, tile: np.ndarray) -> str:
    for name, pixels in files.items():
        if np.array_equal(pixels, tile):
            return name
    files[preferred] = tile
    return preferred


def _block_textures(
    asset_id: str,
    tiles: dict[str, np.ndarray],
    index: int,
    count: int,
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    suffix = "" if count == 1 else f"_{index}"
    sides = ("north", "south", "east", "west")
    sides_same = all(np.array_equal(tiles[sides[0]], tiles[name]) for name in sides[1:])
    all_same = (
        sides_same
        and np.array_equal(tiles["top"], tiles["bottom"])
        and np.array_equal(tiles["top"], tiles[sides[0]])
    )
    files: dict[str, np.ndarray] = {}
    if all_same:
        name = f"{asset_id}{suffix}"
        files[name] = tiles["top"]
        return {"top": name, "side": name, "bottom": name}, files

    mapping: dict[str, str] = {}
    if sides_same:
        side_name = f"{asset_id}_side{suffix}"
        files[side_name] = tiles["north"]
        mapping["side"] = side_name
    else:
        for face in sides:
            name = f"{asset_id}_{face}{suffix}"
            files[name] = tiles[face]
            mapping[face] = name
    for face in ("top", "bottom"):
        shared = next((name for name, pixels in files.items() if np.array_equal(pixels, tiles[face])), None)
        if shared is not None:
            mapping[face] = shared
            continue
        name = f"{asset_id}_{face}{suffix}"
        files[name] = tiles[face]
        mapping[face] = name
    return mapping, files


def _item_package(
    asset_id: str,
    image: np.ndarray,
    variant_index: int,
    count: int,
    tab: str = "materials",
) -> tuple[dict, dict[str, np.ndarray]]:
    texture = asset_id if count == 1 else f"{asset_id}_var_{variant_index}"
    item = {
        "id": asset_id,
        "lang": f"item.{asset_id}",
        "tab": tab or "materials",
        "model": "flat",
        "texture": texture,
        "max_stack": 64,
        "left": "none",
        "right": "none",
        "durability": 0,
        "mining_level": 0,
        "mining_speed": 1,
        "attack": 1,
        "food": 0,
    }
    return item, {texture: np.asarray(image, dtype=np.uint8)}


def _write_definition_zip(
    path: Path,
    json_name: str,
    payload: dict,
    files: dict[str, np.ndarray],
    extra_json: dict[str, dict] | None = None,
) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [json_name, *[f"{name}.png" for name in files]]
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(json_name, json.dumps(payload, indent=2) + "\n")
        for name, body in (extra_json or {}).items():
            if name == json_name:
                continue
            archive.writestr(name, json.dumps(body, indent=2) + "\n")
            names.append(name)
        for name, pixels in files.items():
            buffer = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8), mode="RGBA").save(buffer, format="PNG")
            archive.writestr(f"{name}.png", buffer.getvalue())
    os.replace(tmp, path)
    return names


def _fit_tile(image: np.ndarray, side: int) -> np.ndarray:
    tile = _square(image)
    if tile.shape[0] == side:
        return tile
    fitted = np.zeros((side, side, 4), dtype=np.uint8)
    span = min(side, tile.shape[0])
    fitted[:span, :span] = tile[:span, :span]
    return fitted


def _square(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 4:
        raise PublishError("A preview needs an RGBA image.")
    side = int(array.shape[0])
    if array.shape[1] != side:
        side = min(array.shape[0], array.shape[1])
        array = array[:side, :side]
    if side < 1:
        raise PublishError("A preview needs at least one pixel.")
    return np.ascontiguousarray(array)


def _shade(image: np.ndarray, factor: float) -> np.ndarray:
    out = np.array(image, copy=True)
    rgb = out[..., :3].astype(np.float32) * float(factor)
    out[..., :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return out
