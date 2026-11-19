"""A small game definition. Copy this file into user_scripts/definitions/ to try it.

File → New → Lanterns → Lantern paints two tiles and exports a JSON file
plus the two textures. docs/definitions.md explains every part.
"""

from voxel_asset_maker.definitions import read_tile

ID = "lantern"
NAME = "Lanterns"
BLURB = "A hanging lantern with a front and a top."

ASSETS = [
    {
        "id": "lantern",
        "label": "Lantern",
        "blurb": "Front and top. Light is how bright it is when placed.",
        "width": 32,
        "height": 16,
        "tiles": [
            {"id": "front", "label": "front", "caption": "Front", "x": 0, "y": 0},
            {"id": "top", "label": "top", "caption": "Top", "x": 16, "y": 0},
        ],
        "fields": [
            {
                "key": "light",
                "label": "Light",
                "type": "int",
                "default": 12,
                "min": 0,
                "max": 15,
                "help": "Block light level.",
            },
            {
                "key": "hang",
                "label": "Hangs",
                "type": "bool",
                "default": True,
            },
        ],
    },
]


def preview(image, asset, fields, context):
    del fields, context
    return read_tile(image, asset, "front")


def export(image, asset, fields, context):
    front = read_tile(image, asset, "front")
    top = read_tile(image, asset, "top")
    asset_id = context.asset_id
    return {
        f"lanterns/{asset_id}.json": {
            "id": asset_id,
            "light": int(fields.get("light", 0)),
            "hang": bool(fields.get("hang", False)),
            "textures": {
                "front": f"{asset_id}_front",
                "top": f"{asset_id}_top",
            },
        },
        f"textures/{asset_id}_front.png": front,
        f"textures/{asset_id}_top.png": top,
    }
