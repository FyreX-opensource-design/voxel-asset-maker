"""Example layer script that produces several brightness variants.

Drop other layer scripts next to this one, or in ~/.voxel-asset-maker/scripts/.
Game definitions go in the definitions folder beside those scripts.
Reload in the editor picks both up. Export Variant ZIP writes one
composited texture per count.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

NAME = "Brightness steps"
BLURB = "Each variant is a brightness step. Count sets the ZIP length when it's the largest."

PARAMS = [
    {
        "key": "count",
        "label": "Count",
        "type": "int",
        "default": 4,
        "min": 1,
        "max": 16,
        "step": 1,
    },
    {
        "key": "step",
        "label": "Step",
        "type": "float",
        "default": -0.12,
        "min": -1.0,
        "max": 1.0,
        "step": 0.01,
        "help": "Added for each variant after the first. Negative steps get darker.",
    },
]


def apply(image, params, context):
    factor = 1.0 + (context.variant_index - 1) * float(params.get("step", -0.12))
    arr = np.array(image.convert("RGBA"), dtype=np.float32)
    rgb = np.clip(arr[..., :3] * factor, 0, 255)
    out = np.array(image.convert("RGBA"), dtype=np.uint8)
    out[..., :3] = np.round(rgb).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")
