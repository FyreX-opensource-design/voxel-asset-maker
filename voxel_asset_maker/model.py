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


@dataclass
class ScriptBinding:
    script_id: str = "none"
    params: dict[str, Any] = field(default_factory=dict)
    script_path: str = ""

    def copy(self) -> ScriptBinding:
        return ScriptBinding(self.script_id, deepcopy(self.params), self.script_path)


@dataclass
class Layer:
    name: str
    pixels: np.ndarray
    id: str = field(default_factory=new_id)
    visible: bool = True
    opacity: float = 1.0
    blend: str = "normal"
    offset: tuple[int, int] = (0, 0)
    script: ScriptBinding = field(default_factory=ScriptBinding)

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
            script=self.script.copy(),
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
            script=self.script.copy(),
        )


@dataclass
class Document:
    width: int
    height: int
    name: str = "texture"
    layers: list[Layer] = field(default_factory=list)
    active_index: int = 0
    palette: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
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

    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active_index]
