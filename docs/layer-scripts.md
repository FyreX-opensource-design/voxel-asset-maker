# Layer scripts

A layer script changes one layer's pixels before that layer is flattened with the others. Noise, Palette noise, and Adjust color are built in. Anything else is a Python file the editor loads from disk.

Put the file directly in one of these folders, then choose Reload in the layer panel:

- `user_scripts/` next to the app
- `~/.voxel-asset-maker/scripts/`

`user_scripts/definitions/` is a different kind of file. A game definition goes there, and a layer script does not. The editor only reads `*.py` in the folder itself, not in other subfolders. A definition dropped next to a layer script is reported and not added to the script list.

Files whose names start with `_` are skipped. Two files with the same name keep the first one and report the second as an error. `user_scripts/brightness_steps.py` and `voxel_asset_maker/bundled_scripts/edge_shade.py` are working copies of this shape.

The script runs inside the app. It can read and write whatever the app can.

## The file

```python
NAME = "Brightness steps"
BLURB = "Each variant is a brightness step."

PARAMS = [
    {
        "key": "count",
        "label": "Count",
        "type": "int",
        "default": 4,
        "min": 1,
        "max": 16,
        "step": 1,
        "help": "How many variants this layer produces.",
    },
    {
        "key": "step",
        "label": "Step",
        "type": "float",
        "default": -0.12,
        "min": -1.0,
        "max": 1.0,
        "step": 0.01,
        "help": "Added for each variant after the first.",
    },
]


def apply(image, params, context):
    factor = 1.0 + (context.variant_index - 1) * float(params["step"])
    # image is a PIL RGBA image. Return one too.
    return image
```

`NAME` is the label in the script list. `BLURB` is the sentence under the list. Both can be left out. The file name is the label then, with underscores turned into spaces.

`apply` is required. It receives the image from the script above it, or the painted pixels when it is the first script. Return a PIL image. RGBA is the mode the rest of the stack expects. Any other mode is converted. Raising an exception keeps the layer's painted pixels and shows the message in the layer panel.

## Parameters

`PARAMS` is a list. Each entry needs a `key`. The editor builds a control from `type`:

| type | control | stored value |
| --- | --- | --- |
| `int` | whole number | int |
| `float` | decimal, the default when type is left out | float |
| `bool` | checkbox | True or False |
| `choice` | menu of `choices` | one of those strings |
| `text` | one line | string |
| `color` | color button | `#rrggbb` or `#rrggbbaa`, or empty |
| `colors` | swatches with add and remove | hex colors separated by spaces |

`label` is the text beside the control. `default` is the value a new script starts with. `min`, `max`, and `step` apply to numbers. `help` is the tooltip.

A parameter named `count` is special. The variant slider and Export Variant ZIP use the largest count on any script in the stack. The count is at least 1 and at most 256. A script without a count parameter contributes one variant.

If the count is not a single parameter, define a function instead:

```python
def variant_count(params):
    return int(params.get("shades", 1))
```

## The context

`context` tells the script which variant it is building.

- `variant_index` starts at 1 and counts this script's own series.
- `variant_count` is that series length.
- `export_index` and `export_count` are the composited ZIP being written.
- `width` and `height` are the image this call received. On a whole layer they are the layer size. On one part they are that part's size.
- `layer_name` is the layer.
- `part` is the part id when the script is limited to parts, and empty when it runs on the whole layer.

When this script's count is shorter than the export, `variant_index` cycles. A noise script with count 4, under an export of 8, sees indexes 1, 2, 3, 4, 1, 2, 3, 4. Read `variant_index` when each export should look different. The built-in Noise and Palette noise scripts add it to the seed so each variant is a new grain from the same settings.

Palette noise replaces each opaque pixel with one color from its set. A noise value chooses the color, and Difference is how far that color may drift lighter or darker. 0 keeps the exact color. Grain size blocks pixels into the same choice. Transparent pixels stay transparent while Keep transparency is checked. On a parted sheet, each checked part gets its own grain. The Colors control is the `colors` parameter type: a space-separated list such as `#6e655d #8a8178 #a3988e`.

## Where it runs

Scripts on a layer run from top to bottom. Each one sees the image the previous script returned. The layer's opacity and blend mode happen after the whole list, when the layer joins the stack.

## Parts

A block, crop, door, trapdoor, wood set, or custom definition is a sheet of named parts. The script form then shows a Parts row of checkboxes, in the same order as the tiles on the canvas. Check one or more and the script runs once for each of those rectangles: it receives only that tile, and the result is pasted back. The other tiles stay as they were. Every checked part sees the same variant index. `context.part` is that tile's id, and `context.width` and `context.height` are the tile.

Leave every box unchecked and the script runs once on the whole layer, which is how a script with no parts saved has always worked. A plain texture or an item has no parts, so the row stays hidden. The script list names the parts you checked. Saving the project stores those ids. A part id that is not on the sheet is skipped. If none of the chosen parts are on the sheet, the layer is left unchanged and the panel shows an error.

Result view shows the flattened texture for the variant on the slider. Layer pixels view shows the active layer before its scripts, which is the bitmap you paint. Export PNG writes the variant you are viewing. Export Variant ZIP writes one composited PNG per variant, plus `manifest.json`.

Add puts another script on the same layer. The list beside Add is the order they run. Up and Down move the selected script. Remove drops it. The painted pixels stay.

Saving a project stores the script id, the parameter values, the part ids when any are checked, and the file path when the script came from a file. Opening that project on another machine needs the same file in one of the script folders, or the layer shows a missing-script line and paints the original pixels.
