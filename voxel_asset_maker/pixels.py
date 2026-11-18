"""Pixel drawing operations on RGBA uint8 arrays."""

from __future__ import annotations

import numpy as np

from .model import Color

Rect = tuple[int, int, int, int]


def point_in_rect(x: int, y: int, rect: Rect) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def normalize_rect(x0: int, y0: int, x1: int, y1: int) -> Rect:
    xa, xb = sorted((int(x0), int(x1)))
    ya, yb = sorted((int(y0), int(y1)))
    return (xa, ya, xb - xa + 1, yb - ya + 1)


def blit(
    dest: np.ndarray,
    src: np.ndarray,
    x: int,
    y: int,
    selection: Rect | None = None,
) -> None:
    """Copy src onto dest with its top-left at (x, y). Clip to the canvas and selection."""
    dest_h, dest_w = dest.shape[:2]
    src_h, src_w = src.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(dest_w, x + src_w)
    y1 = min(dest_h, y + src_h)
    if x0 >= x1 or y0 >= y1:
        return
    patch = src[y0 - y : y1 - y, x0 - x : x1 - x]
    if selection is None:
        dest[y0:y1, x0:x1] = patch
        return
    sel = _selection_mask(y0, y1, x0, x1, selection)
    region = dest[y0:y1, x0:x1]
    region[sel] = patch[sel]


def extract(pixels: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Copy a rectangle, padding with transparency where it leaves the canvas."""
    out = np.zeros((h, w, 4), dtype=np.uint8)
    dest_h, dest_w = pixels.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(dest_w, x + w)
    y1 = min(dest_h, y + h)
    if x0 >= x1 or y0 >= y1:
        return out
    out[y0 - y : y1 - y, x0 - x : x1 - x] = pixels[y0:y1, x0:x1]
    return out


def clear_region(pixels: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    blank = np.zeros((max(h, 0), max(w, 0), 4), dtype=np.uint8)
    if h > 0 and w > 0:
        blit(pixels, blank, x, y)


def stamp_brush(
    pixels: np.ndarray,
    x: int,
    y: int,
    color: Color,
    size: int,
    selection: Rect | None = None,
) -> None:
    size = max(1, int(size))
    half = size // 2
    brush = np.empty((size, size, 4), dtype=np.uint8)
    brush[:, :] = color
    blit(pixels, brush, x - half, y - half, selection)


def bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def stroke_line(
    pixels: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Color,
    size: int,
    selection: Rect | None = None,
) -> None:
    for x, y in bresenham(x0, y0, x1, y1):
        stamp_brush(pixels, x, y, color, size, selection)


def stroke_rect(
    pixels: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Color,
    size: int,
    filled: bool,
    selection: Rect | None = None,
) -> None:
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    if filled:
        block = np.empty((yb - ya + 1, xb - xa + 1, 4), dtype=np.uint8)
        block[:, :] = color
        blit(pixels, block, xa, ya, selection)
        return
    stroke_line(pixels, xa, ya, xb, ya, color, size, selection)
    stroke_line(pixels, xa, yb, xb, yb, color, size, selection)
    stroke_line(pixels, xa, ya, xa, yb, color, size, selection)
    stroke_line(pixels, xb, ya, xb, yb, color, size, selection)


def flood_fill(
    pixels: np.ndarray,
    x: int,
    y: int,
    color: Color,
    selection: Rect | None = None,
) -> bool:
    """4-connected fill of the exact RGBA color under (x, y)."""
    height, width = pixels.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return False
    if selection is not None and not point_in_rect(x, y, selection):
        return False
    target = pixels[y, x].copy()
    replacement = np.array(color, dtype=np.uint8)
    if np.array_equal(target, replacement):
        return False
    match = np.all(pixels == target, axis=-1)
    if selection is not None:
        sx, sy, sw, sh = selection
        allowed = np.zeros((height, width), dtype=bool)
        y0, y1 = max(0, sy), min(height, sy + sh)
        x0, x1 = max(0, sx), min(width, sx + sw)
        allowed[y0:y1, x0:x1] = True
        match &= allowed
    visited = np.zeros((height, width), dtype=bool)
    stack = [(x, y)]
    changed = False
    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            continue
        if visited[cy, cx] or not match[cy, cx]:
            continue
        left = cx
        while left > 0 and match[cy, left - 1] and not visited[cy, left - 1]:
            left -= 1
        right = cx
        while right + 1 < width and match[cy, right + 1] and not visited[cy, right + 1]:
            right += 1
        for i in range(left, right + 1):
            visited[cy, i] = True
            pixels[cy, i] = replacement
            changed = True
            if cy > 0 and match[cy - 1, i] and not visited[cy - 1, i]:
                stack.append((i, cy - 1))
            if cy + 1 < height and match[cy + 1, i] and not visited[cy + 1, i]:
                stack.append((i, cy + 1))
    return changed


def flip_horizontal(pixels: np.ndarray, selection: Rect | None) -> None:
    if selection is None:
        pixels[:] = np.ascontiguousarray(pixels[:, ::-1, :])
        return
    x, y, w, h = selection
    patch = np.ascontiguousarray(extract(pixels, x, y, w, h)[:, ::-1, :])
    clear_region(pixels, x, y, w, h)
    blit(pixels, patch, x, y)


def flip_vertical(pixels: np.ndarray, selection: Rect | None) -> None:
    if selection is None:
        pixels[:] = np.ascontiguousarray(pixels[::-1, :, :])
        return
    x, y, w, h = selection
    patch = np.ascontiguousarray(extract(pixels, x, y, w, h)[::-1, :, :])
    clear_region(pixels, x, y, w, h)
    blit(pixels, patch, x, y)


def rotate_cw(pixels: np.ndarray, selection: Rect | None) -> Rect | None:
    """Rotate 90 degrees clockwise. Returns the new selection rect when there is one."""
    if selection is None:
        rotated = np.ascontiguousarray(np.rot90(pixels, k=-1))
        pixels[:] = 0
        blit(pixels, rotated, 0, 0)
        return None
    x, y, w, h = selection
    patch = extract(pixels, x, y, w, h)
    rotated = np.ascontiguousarray(np.rot90(patch, k=-1))
    clear_region(pixels, x, y, w, h)
    blit(pixels, rotated, x, y)
    return (x, y, h, w)


def _selection_mask(y0: int, y1: int, x0: int, x1: int, selection: Rect) -> np.ndarray:
    sx, sy, sw, sh = selection
    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    return (ys >= sy) & (ys < sy + sh) & (xs >= sx) & (xs < sx + sw)
