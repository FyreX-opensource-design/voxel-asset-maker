"""Layer scripts.

Built-ins call the texture_noise and texture_adjust tools. Any other .py file
with PARAMS and apply(image, params, context) can be attached to a layer too.
"""

from __future__ import annotations

import importlib.util
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from . import bootstrap  # noqa: F401
import texture_adjust
import texture_noise

from .model import format_hex_color, parse_hex_color

BUNDLED_DIR = Path(__file__).resolve().parent / "bundled_scripts"
MAX_VARIANTS = 256


class ProcessError(Exception):
    """A layer script failed. The layer falls back to its painted pixels."""


@dataclass
class ScriptContext:
    """Passed to custom scripts.

    variant_index / variant_count are this script's own series (1-based).
    export_index / export_count are the composited ZIP being written.
    When this script's count is shorter than the export, variant_index cycles.
    part is the tile id when the script is limited to parts, otherwise empty.
    width and height match the image passed in, which is that tile or the layer.
    """

    variant_index: int
    variant_count: int
    export_index: int
    export_count: int
    width: int
    height: int
    layer_name: str
    part: str = ""


@dataclass
class Param:
    key: str
    label: str
    type: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    help: str = ""


@dataclass
class Processor:
    id: str
    label: str
    params: list[Param] = field(default_factory=list)
    blurb: str = ""
    apply_fn: Callable[..., Image.Image] | None = None
    count_fn: Callable[[dict[str, Any]], int] | None = None
    path: Path | None = None
    _mtime: float | None = None
    _module: Any = None

    def variant_count(self, params: dict[str, Any]) -> int:
        self._reload_if_needed()
        values = resolve_params(self, params)
        if self.count_fn is not None:
            try:
                return clamp_count(self.count_fn(values))
            except Exception as exc:
                raise ProcessError(str(exc)) from exc
        if any(param.key == "count" for param in self.params):
            return clamp_count(values.get("count", 1))
        return 1

    def apply(self, image: Image.Image, params: dict[str, Any], context: ScriptContext) -> Any:
        self._reload_if_needed()
        if self.apply_fn is None:
            return image
        values = resolve_params(self, params)
        try:
            return self.apply_fn(image, values, context)
        except ProcessError:
            raise
        except Exception as exc:
            raise ProcessError(str(exc)) from exc

    def _reload_if_needed(self) -> None:
        if self.path is None:
            return
        try:
            mtime = self.path.stat().st_mtime
        except OSError as exc:
            raise ProcessError(f"Can't read {self.path.name}: {exc}") from exc
        if self._module is not None and self._mtime == mtime:
            return
        module = load_module(self.path)
        self._module = module
        self._mtime = mtime
        self.params = params_from_module(module)
        self.label = str(getattr(module, "NAME", self.path.stem))
        self.blurb = str(getattr(module, "BLURB", self.blurb))
        self.apply_fn = getattr(module, "apply")
        count = getattr(module, "variant_count", None)
        self.count_fn = count if callable(count) else None


def clamp_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(MAX_VARIANTS, count))


def resolve_params(proc: Processor, params: dict[str, Any]) -> dict[str, Any]:
    values = {param.key: param.default for param in proc.params}
    for key, value in params.items():
        if key in values:
            values[key] = value
    return values


def array_to_image(arr: np.ndarray) -> Image.Image:
    copied = np.array(arr, dtype=np.uint8, copy=True)
    return Image.fromarray(copied, mode="RGBA")


def image_to_array(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise ProcessError("Script has to return a PIL image.")
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ProcessError(f"Script returned an image with shape {getattr(arr, 'shape', None)}.")
    return np.array(arr, dtype=np.uint8, copy=True)


def _noise_apply(image: Image.Image, params: dict[str, Any], context: ScriptContext) -> Image.Image:
    seed = int(params.get("seed") or 0)
    rng = np.random.default_rng(seed + int(context.variant_index))
    mode = params.get("mode", "mono")
    mode = "mono" if mode in ("mono", "monochrome") else "color"
    noise_alpha = bool(params.get("noise_alpha", False))
    preserve_alpha = False if noise_alpha else bool(params.get("preserve_alpha", True))
    clamp = params.get("clamp_delta", 0)
    try:
        clamp_delta = float(clamp) if clamp not in (None, "", 0, 0.0) else None
    except (TypeError, ValueError):
        clamp_delta = None
    if clamp_delta is not None and clamp_delta <= 0:
        clamp_delta = None
    return texture_noise.apply_noise(
        img=image,
        sigma=float(params.get("sigma", 12.0)),
        mean=float(params.get("mean", 0.0)),
        mode=mode,
        dist_type=str(params.get("dist_type", "gaussian")),
        grain_size=max(1, int(params.get("grain_size", 1))),
        preserve_alpha=preserve_alpha,
        noise_alpha=noise_alpha,
        clamp_delta=clamp_delta,
        rng=rng,
    )


def palette_colors(value: Any) -> list[tuple[int, int, int, int]]:
    """Read a color set from a string, or from a list of strings.

    Separators are spaces, commas, and semicolons. A token that is not a
    hex color is skipped.
    """
    if isinstance(value, (list, tuple)):
        pieces = [str(item) for item in value]
    else:
        pieces = ["" if value is None else str(value)]
    colors: list[tuple[int, int, int, int]] = []
    for piece in pieces:
        for token in piece.replace(",", " ").replace(";", " ").split():
            try:
                colors.append(parse_hex_color(token))
            except ValueError:
                continue
    return colors


def format_palette(value: Any) -> str:
    return " ".join(format_hex_color(color) for color in palette_colors(value))


def _palette_apply(image: Image.Image, params: dict[str, Any], context: ScriptContext) -> Image.Image:
    colors = palette_colors(params.get("colors"))
    if not colors:
        raise ProcessError("Palette noise needs at least one color, like #6e655d #8a8178.")
    try:
        spread = int(float(params.get("spread", 10)))
    except (TypeError, ValueError):
        spread = 10
    spread = max(0, min(64, spread))
    try:
        grain = int(params.get("grain_size", 1))
    except (TypeError, ValueError):
        grain = 1
    grain = max(1, grain)
    try:
        seed = int(params.get("seed") or 0)
    except (TypeError, ValueError):
        seed = 0
    mixed = (seed + int(context.variant_index) * 1_000_003) & 0xFFFFFFFF
    part = str(getattr(context, "part", "") or "")
    if part:
        mixed = (mixed ^ zlib.crc32(part.encode("utf-8"))) & 0xFFFFFFFF
    rng = np.random.default_rng(mixed)
    source = np.array(image.convert("RGBA"), dtype=np.uint8)
    height, width = source.shape[:2]
    cells_y = max(1, math.ceil(height / grain))
    cells_x = max(1, math.ceil(width / grain))
    pick = rng.integers(0, len(colors), size=(cells_y, cells_x))
    shade = rng.integers(-spread, spread + 1, size=(cells_y, cells_x))
    if grain != 1:
        pick = np.repeat(np.repeat(pick, grain, axis=0), grain, axis=1)[:height, :width]
        shade = np.repeat(np.repeat(shade, grain, axis=0), grain, axis=1)[:height, :width]
    table = np.asarray(colors, dtype=np.int16)
    chosen = table[pick]
    rgb = np.clip(chosen[..., :3] + shade.astype(np.int16)[..., None], 0, 255).astype(np.uint8)
    out = np.array(source, copy=True)
    if bool(params.get("preserve_alpha", True)):
        mask = source[..., 3] > 0
        out[mask, :3] = rgb[mask]
    else:
        out[..., :3] = rgb
        out[..., 3] = chosen[..., 3].astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _hsb_apply(image: Image.Image, params: dict[str, Any], context: ScriptContext) -> Image.Image:
    own = clamp_count(params.get("count", 1))
    step = (context.variant_index - 1) if own > 1 else 0
    hue = float(params.get("hue_shift", 0.0)) + step * float(params.get("hue_step", 0.0))
    sat_delta = float(params.get("sat_delta", 0.0)) + step * float(params.get("sat_step", 0.0))
    bright_delta = float(params.get("brightness_delta", 0.0)) + step * float(
        params.get("brightness_step", 0.0)
    )
    set_hue = float(params.get("set_hue", 0.0)) if params.get("use_set_hue") else None
    tint = str(params.get("tint") or "").strip() or None
    strength = float(params.get("tint_strength", 0.0))
    return texture_adjust.adjust_hsb(
        img=image,
        hue_shift=hue,
        set_hue=set_hue,
        saturation=float(params.get("saturation", 1.0)),
        saturation_delta=sat_delta,
        brightness=float(params.get("brightness", 1.0)),
        brightness_delta=bright_delta,
        contrast=float(params.get("contrast", 1.0)),
        tint_color=tint,
        tint_strength=strength if tint else 0.0,
        preserve_alpha=bool(params.get("preserve_alpha", True)),
        alpha_mult=float(params.get("alpha_mult", 1.0)),
    )


def _built_in_processors() -> list[Processor]:
    noise = Processor(
        id="noise",
        label="Noise",
        blurb="Grain from texture_noise. Count writes one ZIP entry per variant.",
        apply_fn=_noise_apply,
        params=[
            Param(
                "count",
                "Count",
                "int",
                4,
                1,
                MAX_VARIANTS,
                1,
                help="How many variants this layer produces. The variant slider and ZIP export use the largest Count in the stack.",
            ),
            Param("seed", "Seed", "int", 1, 0, 2_147_483_647, 1, help="Same seed always rebuilds the same grain."),
            Param("sigma", "Intensity", "float", 12.0, 0.0, 128.0, 0.5),
            Param("mean", "Mean shift", "float", 0.0, -128.0, 128.0, 0.5, help="Positive brightens, negative darkens."),
            Param("mode", "Mode", "choice", "mono", choices=("mono", "color"), help="Mono keeps the hue. Color noises each channel."),
            Param("dist_type", "Distribution", "choice", "gaussian", choices=("gaussian", "uniform")),
            Param("grain_size", "Grain size", "int", 1, 1, 32, 1),
            Param("clamp_delta", "Clamp delta", "float", 0.0, 0.0, 255.0, 1.0, help="0 leaves outliers alone."),
            Param("preserve_alpha", "Keep transparency", "bool", True),
            Param("noise_alpha", "Noise alpha", "bool", False),
        ],
    )
    palette = Processor(
        id="palette",
        label="Palette noise",
        blurb="Noise picks one of your colors for each grain, then drifts it within the difference.",
        apply_fn=_palette_apply,
        params=[
            Param(
                "count",
                "Count",
                "int",
                1,
                1,
                MAX_VARIANTS,
                1,
                help="How many variants this layer produces. The variant slider and ZIP export use the largest Count in the stack.",
            ),
            Param("seed", "Seed", "int", 1, 0, 2_147_483_647, 1, help="Same seed always rebuilds the same pattern."),
            Param(
                "colors",
                "Colors",
                "colors",
                "#8a8178 #6e655d #a3988e #554e48",
                help="The colors noise can choose. Each grain gets one, then a small lighter or darker shift.",
            ),
            Param(
                "spread",
                "Difference",
                "int",
                10,
                0,
                64,
                1,
                help="How far the chosen color can drift lighter or darker. 0 stays on the exact color.",
            ),
            Param("grain_size", "Grain size", "int", 1, 1, 32, 1),
            Param("preserve_alpha", "Keep transparency", "bool", True),
        ],
    )
    hsb = Processor(
        id="hsb",
        label="Adjust color",
        blurb="Hue, saturation, brightness, contrast, and tint from texture_adjust.",
        apply_fn=_hsb_apply,
        params=[
            Param("count", "Count", "int", 1, 1, MAX_VARIANTS, 1),
            Param("hue_shift", "Hue shift", "float", 0.0, -360.0, 360.0, 1.0, help="Degrees. 120 turns red toward green."),
            Param("hue_step", "Hue step", "float", 0.0, -360.0, 360.0, 1.0, help="Added to the hue for each variant after the first."),
            Param("use_set_hue", "Override hue", "bool", False),
            Param("set_hue", "Hue", "float", 0.0, 0.0, 360.0, 1.0),
            Param("saturation", "Saturation", "float", 1.0, 0.0, 4.0, 0.05),
            Param("sat_delta", "Saturation add", "float", 0.0, -1.0, 1.0, 0.05),
            Param("sat_step", "Saturation step", "float", 0.0, -1.0, 1.0, 0.05),
            Param("brightness", "Brightness", "float", 1.0, 0.0, 4.0, 0.05),
            Param("brightness_delta", "Brightness add", "float", 0.0, -1.0, 1.0, 0.05),
            Param("brightness_step", "Brightness step", "float", 0.0, -1.0, 1.0, 0.05),
            Param("contrast", "Contrast", "float", 1.0, 0.0, 4.0, 0.05),
            Param("tint", "Tint", "color", ""),
            Param("tint_strength", "Tint strength", "float", 0.0, 0.0, 1.0, 0.05),
            Param("preserve_alpha", "Keep transparency", "bool", True),
            Param("alpha_mult", "Alpha multiply", "float", 1.0, 0.0, 1.0, 0.05),
        ],
    )
    none = Processor(
        id="none",
        label="None",
        blurb="Painted pixels, composited as they are.",
    )
    return [none, noise, palette, hsb]


class ScriptLibrary:
    """Built-in scripts plus anything discovered on disk."""

    def __init__(self) -> None:
        self.processors: dict[str, Processor] = {}
        self.errors: list[str] = []
        self.order: list[str] = []
        self.reload()

    def reload(self) -> None:
        found: dict[str, Processor] = {}
        order: list[str] = []
        errors: list[str] = []
        for proc in _built_in_processors():
            found[proc.id] = proc
            order.append(proc.id)
        seen_stems: dict[str, Path] = {}
        for folder in script_directories():
            for path in sorted(folder.glob("*.py")):
                if path.name.startswith("_"):
                    continue
                stem = path.stem
                if stem in seen_stems:
                    errors.append(f"{path.name} in {path.parent} is already loaded from {seen_stems[stem]}.")
                    continue
                try:
                    proc = processor_from_path(path)
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
                    continue
                seen_stems[stem] = path
                found[proc.id] = proc
                order.append(proc.id)
        # Keep file scripts the user loaded from outside the search folders.
        for script_id, proc in self.processors.items():
            if script_id in found or proc.path is None:
                continue
            if script_id.startswith("script:"):
                found[script_id] = proc
                order.append(script_id)
        self.processors = found
        self.order = order
        self.errors = errors

    def get(self, script_id: str) -> Processor:
        proc = self.processors.get(script_id)
        if proc is not None:
            return proc
        return self.processors["none"]

    def known(self, script_id: str) -> bool:
        return script_id in self.processors

    def file_scripts(self) -> list[Processor]:
        return [self.processors[script_id] for script_id in self.order if script_id.startswith("script:")]

    def load_file(self, path: Path) -> Processor:
        path = Path(path).resolve()
        proc = processor_from_path(path)
        self.processors[proc.id] = proc
        if proc.id not in self.order:
            self.order.append(proc.id)
        return proc


def script_directories() -> list[Path]:
    dirs: list[Path] = []
    candidates = [
        BUNDLED_DIR,
        Path(__file__).resolve().parent.parent / "user_scripts",
        Path.home() / ".voxel-asset-maker" / "scripts",
        Path.cwd() / "user_scripts",
    ]
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_dir():
            continue
        seen.add(resolved)
        dirs.append(path)
    return dirs


def processor_from_path(path: Path) -> Processor:
    module = load_module(path)
    if not hasattr(module, "apply"):
        if hasattr(module, "export") or hasattr(module, "ASSETS"):
            raise ProcessError(
                f"{path.name} is a definition. Put it in user_scripts/definitions/, not next to the layer scripts."
            )
        raise ProcessError(f"{path.name} needs an apply(image, params, context) function.")
    count = getattr(module, "variant_count", None)
    stem = path.stem
    return Processor(
        id=f"script:{stem}",
        label=str(getattr(module, "NAME", stem.replace("_", " "))),
        blurb=str(getattr(module, "BLURB", "")),
        params=params_from_module(module),
        apply_fn=module.apply,
        count_fn=count if callable(count) else None,
        path=path,
        _mtime=path.stat().st_mtime,
        _module=module,
    )


def load_module(path: Path) -> Any:
    module_name = f"vam_user_script_{path.stem}_{abs(hash(str(path.resolve()))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ProcessError(f"Can't load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_params(raw: Any) -> list[Param]:
    """Read a PARAMS or fields list into the widgets the editor already knows."""
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ProcessError("PARAMS has to be a list of objects with a key.")
    params: list[Param] = []
    for item in raw:
        if not isinstance(item, dict) or "key" not in item:
            raise ProcessError("Each PARAMS entry needs a key.")
        choices = item.get("choices")
        params.append(
            Param(
                key=str(item["key"]),
                label=str(item.get("label", item["key"])),
                type=str(item.get("type", "float")),
                default=item.get("default"),
                min=item.get("min"),
                max=item.get("max"),
                step=item.get("step"),
                choices=tuple(str(choice) for choice in choices) if choices else None,
                help=str(item.get("help", "")),
            )
        )
    return params


def params_from_module(module: Any) -> list[Param]:
    return parse_params(getattr(module, "PARAMS", None) or [])
