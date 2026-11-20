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

from .model import (
    BLOCK_FACE_ORIGIN,
    TILE,
    expand_block_sheet,
    resolve_cube_faces,
    WOOD_MEMBERS,
    WOOD_PARTS,
    BlockTraits,
    Document,
    read_ctm,
    read_face,
    read_part,
    read_stage,
    write_face,
)
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


def _yaw_xyz(points: np.ndarray, yaw: int, span: float) -> np.ndarray:
    """Turn world points clockwise about Y, one quarter turn at a time."""
    out = np.array(points, dtype=np.float32, copy=True)
    for _ in range(int(yaw) % 4):
        x = out[:, 0].copy()
        out[:, 0] = out[:, 2]
        out[:, 2] = span - x
    return out


def _project_xyz(
    points: np.ndarray,
    pitch: int,
    scale: float,
    span: float,
    margin: float,
    height: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Isometric screen position and a depth that grows toward the camera."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    sx = (x - z) * scale + span * scale + margin
    if int(pitch) % 2:
        sy = ((x + z) * 0.5 + y) * scale + margin
    else:
        sy = ((x + z) * 0.5 + (height - y)) * scale + margin
    return np.stack((sx, sy), axis=1).astype(np.float32), (x + z).astype(np.float32)


def _canvas_size(span: float, scale: float, margin: float, height: float = 16.0) -> tuple[int, int]:
    width = int(math.ceil(span * 2.0 * scale + margin * 2.0))
    tall = int(math.ceil((span + height) * scale + margin * 2.0))
    return max(1, width), max(1, tall)


def _uv_at(face: str, x: float, y: float, z: float, bounds: tuple[int, ...], uv: tuple[int, int, int, int]) -> tuple[float, float]:
    """Texture coordinate of a point on a box face. Row 0 of the texture is the top."""
    x0, y0, z0, x1, y1, z1 = (float(value) for value in bounds)
    u0, v0, u1, v1 = (float(value) for value in uv)
    span_x = max(1e-6, x1 - x0)
    span_y = max(1e-6, y1 - y0)
    span_z = max(1e-6, z1 - z0)
    if face == "south":
        across = (x - x0) / span_x
        down = 1.0 - (y - y0) / span_y
    elif face == "north":
        across = (x1 - x) / span_x
        down = 1.0 - (y - y0) / span_y
    elif face == "east":
        across = (z1 - z) / span_z
        down = 1.0 - (y - y0) / span_y
    elif face == "west":
        across = (z - z0) / span_z
        down = 1.0 - (y - y0) / span_y
    elif face == "top":
        across = (x - x0) / span_x
        down = (z - z0) / span_z
    else:
        across = (x - x0) / span_x
        down = (z1 - z) / span_z
    return u0 + across * (u1 - u0), v0 + down * (v1 - v0)


def _face_corners(face: str, a0: float, b0: float, a1: float, b1: float, plane: float) -> list[tuple[float, float, float]]:
    """Texture-order corners of a rectangle on a face: top-left, top-right, bottom-right, bottom-left."""
    if face == "south":
        return [(a0, b1, plane), (a1, b1, plane), (a1, b0, plane), (a0, b0, plane)]
    if face == "north":
        return [(a1, b1, plane), (a0, b1, plane), (a0, b0, plane), (a1, b0, plane)]
    if face == "east":
        return [(plane, b1, a1), (plane, b1, a0), (plane, b0, a0), (plane, b0, a1)]
    if face == "west":
        return [(plane, b1, a0), (plane, b1, a1), (plane, b0, a1), (plane, b0, a0)]
    if face == "top":
        return [(a0, plane, b0), (a1, plane, b0), (a1, plane, b1), (a0, plane, b1)]
    return [(a0, plane, b1), (a1, plane, b1), (a1, plane, b0), (a0, plane, b0)]


def _face_plane(face: str, bounds: tuple[int, ...]) -> float:
    x0, y0, z0, x1, y1, z1 = bounds
    if face == "south":
        return float(z1)
    if face == "north":
        return float(z0)
    if face == "east":
        return float(x1)
    if face == "west":
        return float(x0)
    if face == "top":
        return float(y1)
    return float(y0)


def _full_rect(face: str, origin: tuple[int, int, int]) -> tuple[int, int, int, int]:
    ox, oy, oz = origin
    if face in ("south", "north"):
        return (ox, oy, ox + TILE, oy + TILE)
    if face in ("east", "west"):
        return (oz, oy, oz + TILE, oy + TILE)
    return (ox, oz, ox + TILE, oz + TILE)


def _merge_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Greedy solid rectangles. Each cell covers one world unit."""
    remaining = set(cells)
    rects: list[tuple[int, int, int, int]] = []
    while remaining:
        a, b = min(remaining)
        width = 1
        while (a + width, b) in remaining:
            width += 1
        height = 1
        while all((a + i, b + height) in remaining for i in range(width)):
            height += 1
        for i in range(width):
            for j in range(height):
                remaining.remove((a + i, b + j))
        rects.append((a, b, a + width, b + height))
    return rects


def _exposed_rects(face: str, bounds: tuple[int, ...], occupied: set[tuple[int, int, int]]) -> list[tuple[int, int, int, int]]:
    step = _FACE_STEP[face]
    cells: list[tuple[int, int]] = []
    for x, y, z in _face_cells(face, bounds):
        outward = (x + step[0], y + step[1], z + step[2])
        if outward in occupied:
            continue
        if face in ("south", "north"):
            cells.append((x, y))
        elif face in ("east", "west"):
            cells.append((z, y))
        else:
            cells.append((x, z))
    return _merge_cells(cells)


def _rect_quad(
    face: str,
    bounds: tuple[int, ...],
    rect: tuple[int, int, int, int],
    uv: tuple[int, int, int, int],
    texture: np.ndarray,
    shade: float,
    bias: float,
    glow: np.ndarray | None,
    glow_strength: float,
) -> tuple:
    a0, b0, a1, b1 = rect
    corners = _face_corners(face, float(a0), float(b0), float(a1), float(b1), _face_plane(face, bounds))
    tex_uv = [_uv_at(face, x, y, z, bounds, uv) for x, y, z in corners]
    return (corners, tex_uv, texture, shade, bias, glow, glow_strength)


def _raster_screen(quads: list[tuple], width: int, height: int) -> np.ndarray:
    """Draw textured quads. Two samples per pixel, then averaged, so edges stay smooth.

    Texture lookup stays on texel boundaries, so the art remains the block's
    pixels. Only the silhouette is blended.
    """
    samples = 2
    wide, tall = width * samples, height * samples
    premul = np.zeros((tall, wide, 4), dtype=np.float32)
    depth_buf = np.full((tall, wide), -1e9, dtype=np.float32)
    for screen, depth, uvs, texture, shade, bias, glow, glow_strength in quads:
        projected = np.asarray(screen, dtype=np.float32) * samples
        center = projected.mean(axis=0)
        projected = center + (projected - center) * 1.02
        depths = np.asarray(depth, dtype=np.float32) + float(bias)
        coords = np.asarray(uvs, dtype=np.float32)
        _fill_tri_aa(
            premul, depth_buf, projected[0], projected[1], projected[2],
            depths[0], depths[1], depths[2], coords[0], coords[1], coords[2],
            texture, shade, glow, glow_strength,
        )
        _fill_tri_aa(
            premul, depth_buf, projected[0], projected[2], projected[3],
            depths[0], depths[2], depths[3], coords[0], coords[2], coords[3],
            texture, shade, glow, glow_strength,
        )
    folded = premul.reshape(height, samples, width, samples, 4).mean(axis=(1, 3))
    alpha = folded[:, :, 3:4]
    rgb = np.zeros_like(folded[:, :, :3])
    mask = alpha[:, :, 0] > 0.25
    rgb[mask] = folded[:, :, :3][mask] / np.maximum(alpha[:, :, 0][mask] / 255.0, 1e-6)[:, None]
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[:, :, :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(np.round(alpha[:, :, 0]), 0, 255).astype(np.uint8)
    return out


def _fill_tri_aa(
    premul: np.ndarray,
    depth_buf: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    da: float,
    db: float,
    dc: float,
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    uv_c: np.ndarray,
    texture: np.ndarray,
    shade: float,
    glow: np.ndarray | None,
    glow_strength: float,
) -> None:
    height, width = premul.shape[:2]
    points = np.stack((a, b, c))
    min_x = max(int(np.floor(points[:, 0].min())), 0)
    max_x = min(int(np.ceil(points[:, 0].max())), width - 1)
    min_y = max(int(np.floor(points[:, 1].min())), 0)
    max_y = min(int(np.ceil(points[:, 1].max())), height - 1)
    if min_x > max_x or min_y > max_y:
        return
    xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    v0x, v0y = float(c[0] - a[0]), float(c[1] - a[1])
    v1x, v1y = float(b[0] - a[0]), float(b[1] - a[1])
    v2x = grid_x - float(a[0])
    v2y = grid_y - float(a[1])
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot11 = v1x * v1x + v1y * v1y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-8:
        return
    inv = 1.0 / denom
    dot02 = v0x * v2x + v0y * v2y
    dot12 = v1x * v2x + v1y * v2y
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    mask = (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1.0 + 1e-4)
    if not np.any(mask):
        return
    d = float(da) + (float(dc) - float(da)) * u + (float(db) - float(da)) * v
    patch_depth = depth_buf[min_y : max_y + 1, min_x : max_x + 1]
    closer = mask & (d >= patch_depth - 1e-4)
    if not np.any(closer):
        return
    tex_h, tex_w = texture.shape[:2]
    uu = float(uv_a[0]) + (float(uv_c[0]) - float(uv_a[0])) * u + (float(uv_b[0]) - float(uv_a[0])) * v
    vv = float(uv_a[1]) + (float(uv_c[1]) - float(uv_a[1])) * u + (float(uv_b[1]) - float(uv_a[1])) * v
    tx = np.clip(np.floor(uu).astype(np.int32), 0, tex_w - 1)
    ty = np.clip(np.floor(vv).astype(np.int32), 0, tex_h - 1)
    samples = texture[ty, tx].astype(np.float32)
    if glow is not None and glow_strength > 0:
        gh, gw = glow.shape[:2]
        gx = np.clip(np.floor(uu).astype(np.int32), 0, gw - 1)
        gy = np.clip(np.floor(vv).astype(np.int32), 0, gh - 1)
        add = glow[gy, gx].astype(np.float32)
        samples[:, :, :3] = samples[:, :, :3] * shade + add[:, :, :3] * (add[:, :, 3:4] / 255.0) * float(glow_strength)
        samples[:, :, 3] = np.maximum(samples[:, :, 3], add[:, :, 3])
    else:
        samples[:, :, :3] *= shade
    write = closer & (samples[:, :, 3] > 0)
    if not np.any(write):
        return
    alpha = samples[:, :, 3:4] / 255.0
    color = np.empty_like(samples)
    color[:, :, :3] = samples[:, :, :3] * alpha[:, :, 0:1]
    color[:, :, 3] = samples[:, :, 3]
    patch = premul[min_y : max_y + 1, min_x : max_x + 1]
    patch[write] = color[write]
    patch_depth[write] = d[write]


def _render_world(quads: list[tuple], *, yaw: int, pitch: int, scale: int, span: float) -> np.ndarray:
    margin = 2.0
    pixels = max(1, int(scale))
    width, height = _canvas_size(span, pixels, margin)
    screen_quads = []
    for corners, uvs, texture, shade, bias, glow, glow_strength in quads:
        turned = _yaw_xyz(np.asarray(corners, dtype=np.float32), yaw, span)
        screen, depth = _project_xyz(turned, pitch, pixels, span, margin)
        screen_quads.append((screen, depth, uvs, texture, shade, bias, glow, glow_strength))
    image = _raster_screen(screen_quads, width, height)
    return on_checker(image, cell=max(4, pixels * 2))


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
    box = (0, 0, 0, TILE, TILE, TILE, {name: (0, 0, TILE, TILE) for name in tiles})
    return _shape_preview(
        tiles,
        (box,),
        yaw=yaw,
        pitch=pitch,
        glow=None,
        glow_strength=0.0,
        scale=max(1, int(scale)),
    )


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
    """Draw one block made of the model's boxes, from the same camera as a cube.

    Each visible face is one quad (or a few, where a neighbor covers part of
    it), so the outline is the block's edge instead of a staircase of texels.
    """
    cap, left_name, right_name = block_view(yaw, pitch)
    shades = {cap: 1.0 if cap == "top" else 0.62, left_name: 0.92, right_name: 0.72}
    occupied: set[tuple[int, int, int]] = set()
    for box in boxes:
        x0, y0, z0, x1, y1, z1, _uvs = box
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    occupied.add((x, y, z))
    quads = []
    for box in boxes:
        bounds = box[:6]
        uvs = box[6]
        for name in (right_name, left_name, cap):
            if name not in uvs:
                continue
            shade = shades[name]
            bias = 8.0 if name == cap else 0.0
            for rect in _exposed_rects(name, bounds, occupied):
                quads.append(
                    _rect_quad(name, bounds, rect, uvs[name], faces[name], shade, bias, glow, glow_strength)
                )
    return _render_world(quads, yaw=yaw, pitch=pitch, scale=max(1, int(scale)), span=float(TILE))


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
    pixels = max(1, int(scale))
    angle = int(yaw) * math.pi / 2.0 + math.radians(22)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    below = int(pitch) % 2 == 1

    def project(x: float, y: float, z: float) -> tuple[float, float, float]:
        rx = (x - 8.0) * cos_a - (z - 8.0) * sin_a
        rz = (x - 8.0) * sin_a + (z - 8.0) * cos_a
        sx = (rx - rz) * pixels + 24.0 * pixels
        if below:
            sy = ((rx + rz) * 0.5 + y) * pixels + 8.0 * pixels
        else:
            sy = ((rx + rz) * 0.5 + (16.0 - y)) * pixels + 8.0 * pixels
        return sx, sy, rx + rz

    planes = (
        (((0.0, 16.0, 0.0), (16.0, 16.0, 16.0), (16.0, 0.0, 16.0), (0.0, 0.0, 0.0)), 0.82),
        (((0.0, 16.0, 16.0), (16.0, 16.0, 0.0), (16.0, 0.0, 0.0), (0.0, 0.0, 16.0)), 1.0),
    )
    uvs = ((0.0, 0.0), (16.0, 0.0), (16.0, 16.0), (0.0, 16.0))
    quads = []
    for index, (corners, shade) in enumerate(planes):
        screen = []
        depth = []
        for x, y, z in corners:
            sx, sy, near = project(x, y, z)
            screen.append((sx, sy))
            depth.append(near + index * 0.01)
        quads.append((screen, depth, uvs, tile, shade, 0.0, glow, glow_strength))
    image = _raster_screen(quads, 48 * pixels, 48 * pixels)
    return on_checker(image, cell=max(4, pixels * 2))


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
    faces: str = "six",
) -> np.ndarray:
    """The block as it sits in the world.

    A cube with connected textures draws four blocks that share edges, and
    each face uses the tile for those neighbors. Slabs, stairs, fences, and
    walls draw that model's boxes. Fluorescence is added at full ultraviolet,
    scaled by strength. A flat glow color fills every face.
    """
    array = expand_block_sheet(np.asarray(image, dtype=np.uint8), faces)
    cube_faces = {name: read_face(array, name) for name in BLOCK_FACE_ORIGIN}
    if glow is None and glow_color and glow_strength > 0:
        glow = _color_tile(glow_color)
    glow_tile = None if glow is None or glow_strength <= 0 else _fit_tile(np.asarray(glow, dtype=np.uint8), TILE)
    if shape == "cross":
        chosen = 6 if scale is None else max(1, int(scale))
        return _cross_preview(
            _plant_tile(cube_faces),
            yaw=yaw,
            pitch=pitch,
            glow=glow_tile,
            glow_strength=glow_strength,
            scale=chosen,
        )
    if shape in _SHAPE_BOXES:
        chosen = 6 if scale is None else max(1, int(scale))
        return _shape_preview(
            cube_faces,
            _SHAPE_BOXES[shape],
            yaw=yaw,
            pitch=pitch,
            glow=glow_tile,
            glow_strength=glow_strength,
            scale=chosen,
        )
    use_ctm = ctm is not None
    cells = set(_CLUSTER) if use_ctm else {(0, 0, 0)}
    span_blocks = 2 if use_ctm else 1
    chosen = (3 if use_ctm else 6) if scale is None else max(1, int(scale))
    cap, left_name, right_name = block_view(yaw, pitch)
    shades = {cap: 1.0 if cap == "top" else 0.62, left_name: 0.92, right_name: 0.72}
    full_uv = (0, 0, TILE, TILE)
    quads = []
    for pos in cells:
        origin = (pos[0] * TILE, 0, pos[2] * TILE)
        bounds = (origin[0], origin[1], origin[2], origin[0] + TILE, origin[1] + TILE, origin[2] + TILE)
        for name in (right_name, left_name, cap):
            if _face_hidden(cells, pos, name):
                continue
            tile = read_ctm(np.asarray(ctm), _connection_mask(cells, pos, name)) if use_ctm else cube_faces[name]
            quads.append(
                _rect_quad(
                    name,
                    bounds,
                    _full_rect(name, origin),
                    full_uv,
                    tile,
                    shades[name],
                    8.0 if name == cap else 0.0,
                    glow_tile,
                    glow_strength,
                )
            )
    return _render_world(quads, yaw=yaw, pitch=pitch, scale=chosen, span=float(TILE * span_blocks))


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
        "loot_table": f"crops/{asset_id}_ripe",
        "ground": crop.ground.split(),
        "seed_tab": crop.seed_tab,
        "produce_tab": crop.produce_tab,
    }
    extra: dict[str, dict] = {
        f"loot/crops/{asset_id}_ripe.json": _loot_table(_ripe_entries(crop, produce, seeds)),
        f"loot/crops/{asset_id}_young.json": _loot_table(
            [{"type": "item", "name": seeds, "count": 1}]
        ),
    }
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
            "right": "none",
            "food": 0,
        }
    return payload, files, extra


def _ripe_entries(crop, produce: str, seeds: str) -> list[dict]:
    """Loot rows for a ripe plant. A blank item follows the produce, then the seeds."""
    entries: list[dict] = []
    for index, drop in enumerate(crop.loot):
        item = drop.item or (produce if index == 0 else seeds if index == 1 else "")
        if not item:
            continue
        entry: dict = {"type": "item", "name": item}
        if int(drop.minimum) == int(drop.maximum):
            entry["count"] = int(drop.minimum)
        else:
            entry["count"] = [int(drop.minimum), int(drop.maximum)]
        if float(drop.chance) < 1.0:
            entry["chance"] = float(drop.chance)
        entries.append(entry)
    return entries


def _loot_table(entries: list[dict]) -> dict:
    return {"type": "block", "pools": [{"rolls": 1, "entries": entries}]}


def _opening_package(
    kind: str,
    asset_id: str,
    image: np.ndarray,
    doc: Document,
) -> tuple[dict, dict[str, np.ndarray], dict[str, dict]]:
    """A door or trapdoor block, its behavior file, the placeable item, and loot.

    A door paints lower, upper, and edge. A trapdoor paints the panel and the edge.
    The item icon is the panel, or the upper half stacked on the lower half.
    """
    traits = doc.traits
    door = kind == "door"
    if door:
        lower = read_part(image, 0)
        upper = read_part(image, 1)
        edge = read_part(image, 2)
        tiles = (("lower", lower), ("upper", upper), ("edge", edge))
        icon = _half_stack(upper, lower)
    else:
        panel = read_part(image, 0)
        edge = read_part(image, 1)
        tiles = (("texture", panel), ("edge", edge))
        icon = panel
    files: dict[str, np.ndarray] = {}
    textures = {key: _store_tile(files, f"{asset_id}_{key}", tile) for key, tile in tiles}
    icon_name = _store_tile(files, asset_id, icon)
    folder = "doors" if door else "trapdoors"
    block = {
        "id": asset_id,
        "lang": f"block.{asset_id}",
        "tab": traits.tab or "building",
        "model": "door_lower" if door else "trapdoor",
        "type": "door" if door else "trapdoor",
        "render": traits.render if traits.render != "solid" else "cutout",
        "solid": bool(traits.solid),
        "hardness": traits.hardness,
        "tool": traits.tool,
        "mining_level": int(traits.mining_level),
        "sound": traits.sound,
        "loot_table": f"blocks/{asset_id}",
        "properties": (
            {"door": True, "half": "lower", "facing": "south", "open": False}
            if door
            else {"trapdoor": True, "half": "bottom", "facing": "north", "open": False}
        ),
        "textures": textures,
    }
    behavior = {
        "id": asset_id,
        "sound": traits.sound,
        "hardness": traits.hardness,
        "tool": traits.tool,
        "openable_by_hand": True,
        "item": asset_id,
        "loot_table": f"blocks/{asset_id}",
        "textures": textures,
    }
    item = {
        "id": asset_id,
        "lang": f"item.{asset_id}",
        "tab": traits.tab or "building",
        "model": "flat",
        "texture": icon_name,
        "block": asset_id,
        "max_stack": 64,
        "left": "break",
        "right": "place",
        "durability": 0,
    }
    loot = _loot_table([{"type": "item", "name": asset_id, "count": 1}])
    extra = {
        f"{folder}/{asset_id}.json": behavior,
        f"items/{asset_id}.json": item,
        f"loot/blocks/{asset_id}.json": loot,
    }
    return block, files, extra


def _half_stack(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Squeeze the upper and lower door tiles into one 16×16 item icon."""
    icon = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    icon[:8] = upper[::2]
    icon[8:] = lower[::2]
    return icon


def opening_preview(
    image: np.ndarray,
    kind: str,
    *,
    yaw: int = 0,
    pitch: int = 0,
    scale: int | None = None,
) -> np.ndarray:
    """A closed door or trapdoor from the same camera as a cube."""
    if kind == "door":
        face = _half_stack(read_part(image, 1), read_part(image, 0))
        edge = read_part(image, 2)
        faces = {name: edge for name in BLOCK_FACE_ORIGIN}
        faces["south"] = face
        faces["north"] = face
        boxes = (
            (0, 0, 13, 16, 16, 16, {
                "south": _FULL,
                "north": _FULL,
                "east": (0, 0, 3, 16),
                "west": (13, 0, 16, 16),
                "top": (0, 0, 16, 3),
                "bottom": (0, 0, 16, 3),
            }),
        )
    else:
        panel = read_part(image, 0)
        edge = read_part(image, 1)
        faces = {name: edge for name in BLOCK_FACE_ORIGIN}
        faces["top"] = panel
        faces["bottom"] = panel
        boxes = (
            (0, 0, 0, 16, 3, 16, {
                "north": (0, 0, 16, 3),
                "south": (0, 0, 16, 3),
                "east": (0, 0, 16, 3),
                "west": (0, 0, 16, 3),
                "top": _FULL,
                "bottom": _FULL,
            }),
        )
    chosen = 6 if scale is None else max(1, int(scale))
    return _shape_preview(
        faces,
        boxes,
        yaw=yaw,
        pitch=pitch,
        glow=None,
        glow_strength=0.0,
        scale=chosen,
    )


def _json_number(value: float) -> int | float:
    number = float(value)
    if number == int(number):
        return int(number)
    return number


def wood_member_id(prefix: str, member: str) -> str:
    """Block or item id for one piece of a wood set."""
    if member.startswith("vertical_planks"):
        rest = member.removeprefix("vertical_")
        return f"vertical_{prefix}_{rest}"
    return f"{prefix}_{member}"


def _wood_tiles(image: np.ndarray) -> dict[str, np.ndarray]:
    return {key: read_part(image, index) for index, (key, _label, _caption) in enumerate(WOOD_PARTS)}


def _uniform_faces(tile: np.ndarray) -> dict[str, np.ndarray]:
    return {name: tile for name in BLOCK_FACE_ORIGIN}


def _vertical_planks(tile: np.ndarray) -> np.ndarray:
    """Turn planks the way block-game turns pine planks into vertical planks."""
    return np.ascontiguousarray(np.rot90(np.asarray(tile, dtype=np.uint8), 1))


def _wood_faces(tiles: dict[str, np.ndarray], member: str) -> dict[str, np.ndarray]:
    if member == "log":
        faces = _uniform_faces(tiles["log_side"])
        faces["top"] = tiles["log_top"]
        faces["bottom"] = tiles["log_top"]
        return faces
    if member == "wood":
        return _uniform_faces(tiles["log_side"])
    if member.startswith("vertical_planks"):
        return _uniform_faces(_vertical_planks(tiles["planks"]))
    if member == "crafting_table":
        faces = _uniform_faces(tiles["table_side"])
        faces["top"] = tiles["table_top"]
        faces["bottom"] = tiles["log_top"]
        faces["north"] = tiles["table_front"]
        faces["south"] = tiles["table_front"]
        return faces
    return _uniform_faces(tiles["planks"])


def _faces_sheet(faces: dict[str, np.ndarray]) -> np.ndarray:
    sheet = np.zeros((TILE * 3, TILE * 4, 4), dtype=np.uint8)
    for name, tile in faces.items():
        write_face(sheet, name, tile)
    return sheet


def _part_strip(parts: tuple[np.ndarray, ...]) -> np.ndarray:
    image = np.zeros((TILE, TILE * len(parts), 4), dtype=np.uint8)
    for index, tile in enumerate(parts):
        image[:, index * TILE : (index + 1) * TILE] = tile
    return image


_WOOD_SHAPES = {
    "log": "cube",
    "wood": "cube",
    "planks": "cube",
    "planks_slab": "slab",
    "planks_vertical_slab": "vertical_slab",
    "planks_stairs": "stairs",
    "vertical_planks": "cube",
    "vertical_planks_slab": "slab",
    "vertical_planks_vertical_slab": "vertical_slab",
    "vertical_planks_stairs": "stairs",
    "crafting_table": "cube",
}


def wood_preview(
    image: np.ndarray,
    member: str = "log",
    *,
    yaw: int = 0,
    pitch: int = 0,
    scale: int | None = None,
) -> np.ndarray:
    """One wood-set piece, from the same camera as a single block."""
    tiles = _wood_tiles(image)
    key = member if member in _WOOD_SHAPES or member in ("stick", "door", "trapdoor") else "log"
    if key == "stick":
        return sprite_preview(tiles["stick"])
    if key == "door":
        strip = _part_strip((tiles["door_lower"], tiles["door_upper"], tiles["door_edge"]))
        return opening_preview(strip, "door", yaw=yaw, pitch=pitch, scale=scale)
    if key == "trapdoor":
        strip = _part_strip((tiles["trapdoor"], tiles["door_edge"]))
        return opening_preview(strip, "trapdoor", yaw=yaw, pitch=pitch, scale=scale)
    return block_preview(
        _faces_sheet(_wood_faces(tiles, key)),
        yaw=yaw,
        pitch=pitch,
        scale=scale,
        shape=_WOOD_SHAPES[key],
    )


def _wood_block(
    asset_id: str,
    shape: str,
    textures: dict[str, str],
    hardness: float,
    *,
    tab: str | list[str] = "building",
    rotation: str = "",
    station: str = "",
    full_block: str = "",
) -> dict:
    model_name, block_type = _SHAPE_EXPORT[shape]
    block = {
        "id": asset_id,
        "lang": f"block.{asset_id}",
        "tab": tab,
        "model": model_name,
        "hardness": _json_number(hardness),
        "tool": "axe",
        "mining_level": 0,
        "sound": "wood",
        "loot_table": f"blocks/{asset_id}",
        "textures": textures,
    }
    if block_type:
        block["type"] = block_type
    if full_block:
        block["full_block"] = full_block
    if rotation:
        block["properties"] = {"rotation": rotation}
    if station:
        block["station"] = station
    return block


def _place_item(asset_id: str, tab: str | list[str]) -> dict:
    return {
        "id": asset_id,
        "lang": f"item.{asset_id}",
        "tab": tab,
        "model": "block",
        "block": asset_id,
        "max_stack": 64,
        "left": "break",
        "right": "place",
        "durability": 0,
        "mining_level": 0,
        "mining_speed": 1,
        "attack": 1,
        "food": 0,
    }


def _absorb_package(
    files: dict[str, np.ndarray],
    incoming: dict[str, np.ndarray],
    block: dict,
    extra: dict[str, dict],
) -> None:
    """Copy a door or trapdoor package in, sharing any texture already stored."""
    mapping = {name: _store_tile(files, name, tile) for name, tile in incoming.items()}

    def remap(body: dict) -> None:
        if isinstance(body.get("textures"), dict):
            body["textures"] = {key: mapping.get(value, value) for key, value in body["textures"].items()}
        if body.get("texture") in mapping:
            body["texture"] = mapping[body["texture"]]

    remap(block)
    for body in extra.values():
        remap(body)


def _wood_package(
    prefix: str,
    image: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray], dict[str, dict]]:
    """Every block and item a wood set exports, plus the textures they share."""
    tiles = _wood_tiles(image)
    files: dict[str, np.ndarray] = {}

    def keep(name: str, tile: np.ndarray) -> str:
        return _store_tile(files, name, np.ascontiguousarray(tile, dtype=np.uint8))

    log_top = keep(f"{prefix}_log_top", tiles["log_top"])
    log_side = keep(f"{prefix}_log_side", tiles["log_side"])
    planks = keep(f"{prefix}_planks", tiles["planks"])
    vertical = keep(f"vertical_{prefix}_planks", _vertical_planks(tiles["planks"]))
    stick = keep(f"{prefix}_stick", tiles["stick"])
    table_top = keep(f"{prefix}_crafting_table_top", tiles["table_top"])
    table_front = keep(f"{prefix}_crafting_table_front", tiles["table_front"])
    table_side = keep(f"{prefix}_crafting_table_side", tiles["table_side"])
    both = ["building", "natural"]
    pieces: list[tuple[str, dict]] = [
        (wood_member_id(prefix, "log"), _wood_block(
            wood_member_id(prefix, "log"), "cube",
            {"top": log_top, "side": log_side, "bottom": log_top},
            2.0, tab=both, rotation="axis",
        )),
        (wood_member_id(prefix, "wood"), _wood_block(
            wood_member_id(prefix, "wood"), "cube",
            {"top": log_side, "side": log_side, "bottom": log_side},
            2.0, tab=both, rotation="axis",
        )),
        (wood_member_id(prefix, "planks"), _wood_block(
            wood_member_id(prefix, "planks"), "cube",
            {"top": planks, "side": planks, "bottom": planks},
            2.0,
        )),
        (wood_member_id(prefix, "planks_slab"), _wood_block(
            wood_member_id(prefix, "planks_slab"), "slab",
            {"top": planks, "side": planks, "bottom": planks},
            1.0,
        )),
        (wood_member_id(prefix, "planks_vertical_slab"), _wood_block(
            wood_member_id(prefix, "planks_vertical_slab"), "vertical_slab",
            {"top": planks, "side": planks, "bottom": planks},
            1.0, full_block=wood_member_id(prefix, "planks"),
        )),
        (wood_member_id(prefix, "planks_stairs"), _wood_block(
            wood_member_id(prefix, "planks_stairs"), "stairs",
            {"top": planks, "side": planks, "bottom": planks},
            2.0,
        )),
        (wood_member_id(prefix, "vertical_planks"), _wood_block(
            wood_member_id(prefix, "vertical_planks"), "cube",
            {"top": vertical, "side": vertical, "bottom": vertical},
            2.0,
        )),
        (wood_member_id(prefix, "vertical_planks_slab"), _wood_block(
            wood_member_id(prefix, "vertical_planks_slab"), "slab",
            {"top": vertical, "side": vertical, "bottom": vertical},
            2.0,
        )),
        (wood_member_id(prefix, "vertical_planks_vertical_slab"), _wood_block(
            wood_member_id(prefix, "vertical_planks_vertical_slab"), "vertical_slab",
            {"top": vertical, "side": vertical, "bottom": vertical},
            2.0, full_block=wood_member_id(prefix, "vertical_planks"),
        )),
        (wood_member_id(prefix, "vertical_planks_stairs"), _wood_block(
            wood_member_id(prefix, "vertical_planks_stairs"), "stairs",
            {"top": vertical, "side": vertical, "bottom": vertical},
            2.0,
        )),
        (wood_member_id(prefix, "crafting_table"), _wood_block(
            wood_member_id(prefix, "crafting_table"), "cube",
            {
                "up": table_top,
                "down": log_top,
                "north": table_front,
                "south": table_front,
                "east": table_side,
                "west": table_side,
            },
            2.5, station="crafting_table",
        )),
    ]
    extra: dict[str, dict] = {}
    for asset_id, block in pieces:
        tab = block["tab"]
        extra[f"blocks/{asset_id}.json"] = block
        extra[f"items/{asset_id}.json"] = _place_item(asset_id, tab)
        extra[f"loot/blocks/{asset_id}.json"] = _loot_table(
            [{"type": "item", "name": asset_id, "count": 1}]
        )
    stick_id = wood_member_id(prefix, "stick")
    extra[f"items/{stick_id}.json"] = {
        "id": stick_id,
        "lang": f"item.{stick_id}",
        "tab": "materials",
        "model": "flat",
        "texture": stick,
        "max_stack": 64,
        "left": "none",
        "right": "none",
        "durability": 0,
        "mining_level": 0,
        "mining_speed": 1,
        "attack": 1,
        "food": 0,
    }
    door_traits = BlockTraits(tab="building", hardness=3.0, tool="axe", sound="wood", render="cutout", solid=True)
    door_doc = Document(TILE, TILE, name=prefix, kind="door", traits=door_traits)
    door_id = wood_member_id(prefix, "door")
    trap_id = wood_member_id(prefix, "trapdoor")
    door_block, door_files, door_extra = _opening_package(
        "door",
        door_id,
        _part_strip((tiles["door_lower"], tiles["door_upper"], tiles["door_edge"])),
        door_doc,
    )
    trap_block, trap_files, trap_extra = _opening_package(
        "trapdoor",
        trap_id,
        _part_strip((tiles["trapdoor"], tiles["door_edge"])),
        door_doc,
    )
    _absorb_package(files, door_files, door_block, door_extra)
    _absorb_package(files, trap_files, trap_block, trap_extra)
    extra[f"blocks/{door_id}.json"] = door_block
    extra[f"blocks/{trap_id}.json"] = trap_block
    extra.update(door_extra)
    extra.update(trap_extra)
    manifest = {
        "set": "wood",
        "id": prefix,
        "members": [wood_member_id(prefix, key) for key, _label in WOOD_MEMBERS],
    }
    return manifest, files, extra


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
    member: str = "",
    faces: str = "six",
) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if kind == "set":
        return wood_preview(array, member or "log", yaw=yaw, pitch=pitch, scale=scale)
    if kind == "crop":
        chosen = 6 if scale is None else max(1, int(scale))
        return _cross_preview(
            read_stage(array, stage),
            yaw=yaw,
            pitch=pitch,
            glow=None,
            glow_strength=0.0,
            scale=chosen,
        )
    if kind in ("door", "trapdoor"):
        return opening_preview(array, kind, yaw=yaw, pitch=pitch, scale=scale)
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
            faces=faces,
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
    if kind not in ("block", "item", "crop", "door", "trapdoor", "set"):
        raise PublishError(
            "Only a block, an item, a crop, a door, a trapdoor, or a set exports a definition."
        )
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
    elif kind in ("door", "trapdoor"):
        payload, files, extra_json = _opening_package(kind, asset_id, rendered[variant_index - 1], doc)
    elif kind == "set":
        payload, files, extra_json = _wood_package(asset_id, rendered[variant_index - 1])
        files = {f"textures/{name}": pixels for name, pixels in files.items()}
        names = _write_definition_zip(path, f"sets/{asset_id}.json", payload, files, extra_json)
        return ExportedDefinition(kind, asset_id, path, names, errors)
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
        tiles = resolve_cube_faces(image, doc.traits.faces)
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
    properties: dict = {}
    if traits.shape == "fence":
        properties["fence"] = True
    if traits.rotation in ("axis", "horizontal"):
        properties["rotation"] = traits.rotation
    if properties:
        block["properties"] = properties
    if traits.station:
        block["station"] = traits.station
        if traits.recipe_types:
            block["recipe_types"] = traits.recipe_types.split()
        if traits.station == "furnace" and float(traits.cook_speed) != 1.0:
            block["cook_speed"] = _json_number(traits.cook_speed)
    if not traits.solid:
        block["solid"] = False
    if traits.gravity:
        block["gravity"] = True
    if any(traits.light):
        red, green, blue, ultraviolet, infrared = traits.light
        block["light"] = {
            "r": int(red),
            "g": int(green),
            "b": int(blue),
            "uv": int(ultraviolet),
            "ir": int(infrared),
        }
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
