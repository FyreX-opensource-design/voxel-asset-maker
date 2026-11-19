# Custom definitions

Block game ships with the app: blocks, items, crops, doors, trapdoors, and the wood set, under File → New → Block game. Another game is a Python file. The editor lists it beside Block game, opens a sheet of tiles for each asset, and calls your `export` function when the user chooses Export definition.

Copy `examples/lantern.py` into the definitions namespace, then choose Reload in the layer panel. That button rescans layer scripts and definitions together.

- `user_scripts/definitions/` next to the app
- `~/.voxel-asset-maker/scripts/definitions/`

Layer scripts stay in the parent folder (`user_scripts/*.py`). The editor does not treat those as definitions, and it does not treat a file in `definitions/` as a layer script. A file in the wrong folder is reported with the folder it belongs in.

Files whose names start with `_` are skipped. Each file needs its own id. A second file that reuses an id is reported and not loaded.

The file runs inside the app. It can read and write whatever the app can.

## The file

```python
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
            {"key": "light", "label": "Light", "type": "int", "default": 12, "min": 0, "max": 15},
        ],
    },
]


def preview(image, asset, fields, context):
    return read_tile(image, asset, "front")


def export(image, asset, fields, context):
    asset_id = context.asset_id
    return {
        f"lanterns/{asset_id}.json": {
            "id": asset_id,
            "light": int(fields["light"]),
            "textures": {"front": f"{asset_id}_front"},
        },
        f"textures/{asset_id}_front.png": read_tile(image, asset, "front"),
    }
```

`ID` is the token stored in the project file. Leave it out and the file name is used. Letters, digits, and underscores are kept, and everything else becomes an underscore. `NAME` is the File → New menu. `BLURB` is the menu tooltip.

`ASSETS` is required and needs at least one asset. Each asset becomes an item in that menu.

`export` is required. `preview` is optional.

## An asset

`id` becomes part of the project. `label` is the menu item, with an ellipsis added. `blurb` is the preview caption.

`width` and `height` are the canvas, in pixels. Both are multiples of 16, at least 16, and at most 4096. The user cannot resize the canvas afterward, so the tiles stay put.

`tiles` marks the cells painting stays inside. A stroke that starts on a tile cannot leave it, the same way a block face works. Each tile needs:

- `id`, a token
- `x`, `y`, the top-left corner
- `w` and `h`, the size, 16 when left out
- `label`, drawn on the canvas, the id when left out
- `caption`, the status-bar name, the label when left out

Every one of `x`, `y`, `w`, and `h` sits on the 16 pixel grid, and the rectangle has to fit on the sheet. Tiles should not overlap. The first one that contains a pixel wins.

Leave `tiles` out, or pass an empty list, when the whole sheet is one picture and strokes may cross it.

The sheet can grow with a field. Define `layout(asset, fields)` and return a dict with `width`, `height`, and `tiles`. The editor resizes the canvas when those fields change and keeps the pixels already painted in the top-left. `export` and `preview` then see that sheet on `asset`, not the size written in `ASSETS`. Without `layout`, the sheet stays at the asset's width, height, and tiles. File → New → Minecraft → Block uses this so Frames, Faces, and Resolution change the canvas. Each animation frame is as tall as the texture is wide, stacked under the frame above it.

Those tile ids are also the parts a layer script can be limited to. On the script form, check one or more tiles and that script runs on each of them and pastes the result back. Leave them all unchecked and the script runs on the whole sheet. `context.part` in the layer script is the tile id. See `docs/layer-scripts.md`.

`turn` is false unless you set it. True shows the Left, Right, and Underside buttons and passes `context.yaw` (0 to 3) and `context.pitch` (0 or 1) into preview. The built-in block preview uses those for the quarter turns. A flat icon can ignore them.

`fields` are the controls in the asset panel. They use the same types as layer-script parameters: `int`, `float`, `bool`, `choice`, `text`, and `color`. See `docs/layer-scripts.md` for what each one stores. The values are saved in the project and passed to `export` and `preview` as a dict. A missing or out-of-range number is replaced with the default, clamped to `min` and `max`.

## export

```python
def export(image, asset, fields, context):
    ...
    return {zip_path: contents}
```

`image` is a NumPy RGBA array of the flattened sheet for the variant the user is viewing. Layer scripts have already run. `asset` is a dict with the id, label, size, tiles, and the other keys you put on that asset. `fields` is the dict described above.

`context.asset_id` is the id from the asset panel, already sanitized to letters, digits, and underscores. `context.name` is the project name. `context.variant_index` starts at 1. `context.variant_count` is how many variants the layer scripts produce. `image` is the variant on the slider. `context.variants` is every rendered sheet, variant 1 first, so an export can write each one. Preview leaves `variants` empty. The Minecraft block uses them as blockstate variants, and writes a `.mcmeta` file beside a texture when that texture has more than one frame. The block registration is a KubeJS startup script in the same zip.

Return a dict. Each key is a path inside the zip, including the extension, with folders when you want them (`lanterns/oak.json`, `textures/oak_front.png`). Each value is one of:

- a dict or a list, written as JSON
- a NumPy RGBA array or a PIL image, written as a PNG
- `str`, written as UTF-8 text
- `bytes`, written unchanged

The dict has to contain at least one file. The editor puts those paths in the zip and nothing else. It does not modify a game checkout. An exception is shown as the export error and no zip is left behind.

`read_tile(image, asset, tile_id)` copies one tile out of the sheet. Pixels that fall outside the image stay transparent. It raises when the tile id is not on the asset.

## preview

```python
def preview(image, asset, fields, context):
    return read_tile(image, asset, "front")
```

Return a NumPy RGBA array or a PIL image. That picture is what the asset panel shows. Without `preview`, the panel shows the first tile, or the whole sheet when there are no tiles, scaled as a crisp sprite.

Yaw and pitch are on `context` when `turn` is true. They are 0 otherwise.

## The project file

A custom asset is a `.vam` project like any other texture. The project remembers the definition id, the asset id, the field values, and the layers. Opening it on another machine needs the same definition file in one of the folders above. Until that file is loaded, the sheet still opens and paints, and export explains that the definition is missing.
