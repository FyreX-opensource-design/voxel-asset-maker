"""Darken texels that sit against transparency.

Useful on item and foliage layers. A script in user_scripts/ or
~/.voxel-asset-maker/scripts/ with the same shape shows up in the layer
script list.

Variant ZIP files use the largest Count in the stack. Give a script a
count param (or a variant_count function) and read context.variant_index
when each export should differ.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

NAME = "Edge shade"
BLURB = "Darken pixels that touch transparency."

PARAMS = [
    {
        "key": "strength",
        "label": "Strength",
        "type": "float",
        "default": 0.35,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "help": "How far edge pixels move toward black.",
    },
    {
        "key": "passes",
        "label": "Thickness",
        "type": "int",
        "default": 1,
        "min": 1,
        "max": 4,
        "step": 1,
        "help": "How many pixels deep the shade reaches.",
    },
]


def apply(image, params, context):
    del context
    arr = np.array(image.convert("RGBA"), dtype=np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3].copy()
    strength = float(params.get("strength", 0.35))
    passes = max(1, int(params.get("passes", 1)))
    opaque = alpha > 0
    for _ in range(passes):
        transparent = alpha <= 0
        neighbor = np.zeros(transparent.shape, dtype=bool)
        neighbor[1:, :] |= transparent[:-1, :]
        neighbor[:-1, :] |= transparent[1:, :]
        neighbor[:, 1:] |= transparent[:, :-1]
        neighbor[:, :-1] |= transparent[:, 1:]
        edge = neighbor & (alpha > 0) & opaque
        rgb[edge] *= 1.0 - strength
        alpha = alpha.copy()
        alpha[edge] = 0
    out = np.array(image.convert("RGBA"), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")
