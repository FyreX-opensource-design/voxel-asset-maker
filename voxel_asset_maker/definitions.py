"""Game definitions other people can drop in beside the built-in block game.

A definition is a Python file with ASSETS and export(). Put it in
user_scripts/definitions/ (or ~/.voxel-asset-maker/scripts/definitions/).
Layer scripts stay in the parent folder. The editor lists each asset under
File → New, clips painting to its tiles, and calls export when the user
writes a definition zip. See docs/definitions.md.
"""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .model import TILE, Document, asset_token, validate_size
from .processors import Param, ProcessError, format_palette, parse_params
from .render import render_variant, stack_variant_count


class DefinitionError(Exception):
    """A custom definition can't be loaded or exported."""


@dataclass
class TileSpec:
    id: str
    label: str
    caption: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, x: int, y: int) -> bool:
        return self.x <= int(x) < self.x + self.w and self.y <= int(y) < self.y + self.h

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "caption": self.caption,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
        }


@dataclass
class AssetSpec:
    id: str
    label: str
    blurb: str
    width: int
    height: int
    tiles: tuple[TileSpec, ...]
    fields: tuple[Param, ...]
    turn: bool
    raw: dict[str, Any]

    def defaults(self) -> dict[str, Any]:
        return {param.key: param.default for param in self.fields}

    def public(self) -> dict[str, Any]:
        """The asset dict a definition's export and preview functions receive."""
        body = deepcopy(self.raw)
        body["id"] = self.id
        body["label"] = self.label
        body["width"] = self.width
        body["height"] = self.height
        body["tiles"] = [tile.as_dict() for tile in self.tiles]
        body["turn"] = self.turn
        return body


@dataclass
class DefinitionContext:
    """Passed to export() and preview().

    asset_id is the id typed in the asset panel. variant_index is the
    variant currently showing, starting at 1. variants holds every
    rendered sheet, variant 1 first. Export fills it. Preview leaves it empty.
    """

    asset_id: str
    name: str
    variant_index: int
    variant_count: int
    yaw: int = 0
    pitch: int = 0
    variants: tuple[np.ndarray, ...] = ()


@dataclass
class GameDefinition:
    id: str
    name: str
    blurb: str
    assets: tuple[AssetSpec, ...]
    path: Path | None = None
    export_fn: Callable[..., Any] | None = None
    preview_fn: Callable[..., Any] | None = None
    _mtime: float | None = None
    _module: Any = None

    def asset(self, asset_id: str) -> AssetSpec:
        for item in self.assets:
            if item.id == asset_id:
                return item
        raise DefinitionError(f"{self.name} has no asset '{asset_id}'.")

    def resolved_fields(self, asset: AssetSpec, stored: dict[str, Any] | None) -> dict[str, Any]:
        values = asset.defaults()
        incoming = stored if isinstance(stored, dict) else {}
        for key, value in incoming.items():
            param = next((item for item in asset.fields if item.key == key), None)
            if param is None:
                if _plain(value):
                    values[key] = value
                continue
            values[key] = coerce_field(param, value)
        for param in asset.fields:
            values[param.key] = coerce_field(param, values.get(param.key))
        return values

    def preview(
        self,
        image: np.ndarray,
        asset: AssetSpec,
        fields: dict[str, Any],
        context: DefinitionContext,
    ) -> np.ndarray:
        self._reload_if_needed()
        if self.preview_fn is None:
            sample = read_tile(image, asset.public(), asset.tiles[0].id) if asset.tiles else image
            from .blockgame import sprite_preview

            return sprite_preview(sample)
        try:
            produced = self.preview_fn(np.asarray(image), self.public_for(asset, fields), dict(fields), context)
        except DefinitionError:
            raise
        except Exception as exc:
            raise DefinitionError(str(exc)) from exc
        return _as_rgba(produced)

    def export(
        self,
        image: np.ndarray,
        asset: AssetSpec,
        fields: dict[str, Any],
        context: DefinitionContext,
    ) -> dict[str, Any]:
        self._reload_if_needed()
        if self.export_fn is None:
            raise DefinitionError(f"{self.name} has no export function.")
        try:
            produced = self.export_fn(np.asarray(image), self.public_for(asset, fields), dict(fields), context)
        except DefinitionError:
            raise
        except Exception as exc:
            raise DefinitionError(str(exc)) from exc
        if not isinstance(produced, dict) or not produced:
            raise DefinitionError(f"{self.name} export has to return a dict of zip paths.")
        return produced

    def _reload_if_needed(self) -> None:
        if self.path is None:
            return
        try:
            mtime = self.path.stat().st_mtime
        except OSError as exc:
            raise DefinitionError(f"Can't read {self.path.name}: {exc}") from exc
        if self._module is not None and self._mtime == mtime:
            return
        loaded = definition_from_path(self.path)
        self._module = loaded._module
        self._mtime = mtime
        self.name = loaded.name
        self.blurb = loaded.blurb
        self.assets = loaded.assets
        self.export_fn = loaded.export_fn
        self.preview_fn = loaded.preview_fn

    def sheet(self, asset: AssetSpec, fields: dict[str, Any]) -> SheetLayout:
        """The canvas for these field values.

        A definition may define layout(asset, fields). Without one, the
        asset's own width, height, and tiles are the sheet.
        """
        self._reload_if_needed()
        layout = None if self._module is None else getattr(self._module, "layout", None)
        if not callable(layout):
            return SheetLayout(asset.width, asset.height, asset.tiles)
        try:
            raw = layout(asset.public(), dict(fields))
        except DefinitionError:
            raise
        except Exception as exc:
            raise DefinitionError(str(exc)) from exc
        if not isinstance(raw, dict):
            raise DefinitionError(f"{self.name} layout has to return a dict.")
        try:
            width, height = validate_size(int(raw.get("width", asset.width)), int(raw.get("height", asset.height)))
        except (TypeError, ValueError) as exc:
            raise DefinitionError(f"{self.name} layout: {exc}") from exc
        tiles = tuple(_tile_from_dict(item, width, height) for item in raw.get("tiles") or [])
        return SheetLayout(width, height, tiles)

    def public_for(self, asset: AssetSpec, fields: dict[str, Any]) -> dict[str, Any]:
        """The asset dict export and preview see, including a layout() sheet."""
        laid = self.sheet(asset, fields)
        body = asset.public()
        body["width"] = laid.width
        body["height"] = laid.height
        body["tiles"] = [tile.as_dict() for tile in laid.tiles]
        return body


@dataclass(frozen=True)
class SheetLayout:
    width: int
    height: int
    tiles: tuple[TileSpec, ...]


@dataclass
class ExportedCustom:
    definition: str
    asset_id: str
    path: Path
    names: list[str]
    errors: list[str] = field(default_factory=list)


class DefinitionLibrary:
    """Definitions discovered on disk, plus any file loaded by hand."""

    def __init__(self, folders: list[Path] | None = None) -> None:
        self.folders = folders
        self.definitions: dict[str, GameDefinition] = {}
        self.order: list[str] = []
        self.errors: list[str] = []
        self.reload()

    def reload(self) -> None:
        found: dict[str, GameDefinition] = {}
        order: list[str] = []
        errors: list[str] = []
        seen: dict[str, Path] = {}
        for folder in self._folders():
            for path in sorted(folder.glob("*.py")):
                if path.name.startswith("_"):
                    continue
                try:
                    spec = definition_from_path(path)
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
                    continue
                if spec.id in seen:
                    errors.append(f"{path.name} uses id '{spec.id}', already loaded from {seen[spec.id]}.")
                    continue
                seen[spec.id] = path
                found[spec.id] = spec
                order.append(spec.id)
        for definition_id, spec in self.definitions.items():
            if definition_id in found or spec.path is None:
                continue
            found[definition_id] = spec
            order.append(definition_id)
        self.definitions = found
        self.order = order
        self.errors = errors

    def ordered(self) -> list[GameDefinition]:
        return [self.definitions[definition_id] for definition_id in self.order if definition_id in self.definitions]

    def get(self, definition_id: str) -> GameDefinition | None:
        return self.definitions.get(definition_id)

    def load_file(self, path: Path) -> GameDefinition:
        spec = definition_from_path(Path(path))
        self.definitions[spec.id] = spec
        if spec.id not in self.order:
            self.order.append(spec.id)
        return spec

    def tiles_for(self, doc: Document) -> tuple[TileSpec, ...] | None:
        if doc.kind != "custom":
            return None
        spec = self.get(doc.definition)
        if spec is None:
            return None
        try:
            asset = spec.asset(doc.asset)
            return spec.sheet(asset, spec.resolved_fields(asset, doc.fields)).tiles
        except DefinitionError:
            return None

    def _folders(self) -> list[Path]:
        if self.folders is not None:
            return [path for path in self.folders if path.is_dir()]
        return definition_directories()


def definition_directories() -> list[Path]:
    """The definitions namespace inside each folder that already holds layer scripts."""
    root = Path(__file__).resolve().parent.parent
    bundled = Path(__file__).resolve().parent / "bundled_definitions"
    candidates = [
        bundled,
        root / "user_scripts" / "definitions",
        Path.home() / ".voxel-asset-maker" / "scripts" / "definitions",
        Path.cwd() / "user_scripts" / "definitions",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_dir():
            continue
        seen.add(resolved)
        found.append(path)
    return found


def definition_from_path(path: Path) -> GameDefinition:
    module = _load_module(path)
    if not hasattr(module, "export"):
        if hasattr(module, "apply"):
            raise DefinitionError(
                f"{path.name} is a layer script. Put it in user_scripts/, not user_scripts/definitions/."
            )
        raise DefinitionError(f"{path.name} needs an export(image, asset, fields, context) function.")
    raw_assets = getattr(module, "ASSETS", None)
    if not isinstance(raw_assets, (list, tuple)) or not raw_assets:
        raise DefinitionError(f"{path.name} needs an ASSETS list with at least one asset.")
    stem = asset_token(getattr(module, "ID", "") or path.stem, "")
    if not stem:
        raise DefinitionError(f"{path.name} needs an id made of letters, digits, and underscores.")
    assets = tuple(_asset_from_dict(item) for item in raw_assets)
    preview = getattr(module, "preview", None)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    return GameDefinition(
        id=stem,
        name=str(getattr(module, "NAME", stem.replace("_", " "))),
        blurb=str(getattr(module, "BLURB", "")),
        assets=assets,
        path=path,
        export_fn=module.export,
        preview_fn=preview if callable(preview) else None,
        _mtime=mtime,
        _module=module,
    )


def export_custom(
    doc: Document,
    library: DefinitionLibrary,
    script_library: Any,
    path: Path,
    asset_id: str | None = None,
    variant_index: int = 1,
) -> ExportedCustom:
    """Write the zip a custom definition returned for this document."""
    if doc.kind != "custom":
        raise DefinitionError("This file is not a custom definition.")
    spec = library.get(doc.definition)
    if spec is None:
        raise DefinitionError(
            f"Definition '{doc.definition}' isn't loaded. Put its .py file in user_scripts/definitions/ and reload."
        )
    asset = spec.asset(doc.asset)
    from .blockgame import sanitize_id, ASSET_ID, PublishError

    chosen = sanitize_id(asset_id or doc.name)
    if not ASSET_ID.match(chosen):
        raise PublishError(f"'{chosen}' is not an id.")
    path = Path(path)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    count = stack_variant_count(doc, script_library)
    variant_index = max(1, min(int(variant_index), count))
    errors: list[str] = []
    images: list[np.ndarray] = []
    for index in range(1, count + 1):
        result = render_variant(doc, index, script_library, library)
        if index == 1:
            errors.extend(result.errors)
        images.append(np.array(result.image, dtype=np.uint8))
    if not images:
        raise DefinitionError("Nothing to export.")
    rendered = images[variant_index - 1]
    fields = spec.resolved_fields(asset, doc.fields)
    context = DefinitionContext(
        asset_id=chosen,
        name=doc.name,
        variant_index=variant_index,
        variant_count=count,
        variants=tuple(images),
    )
    produced = spec.export(rendered, asset, fields, context)
    names = _write_zip(path, produced)
    return ExportedCustom(spec.id, chosen, path, names, errors)


def read_tile(image: np.ndarray, asset: dict[str, Any], tile_id: str) -> np.ndarray:
    """Copy one tile out of a definition sheet. Missing pixels stay transparent."""
    tiles = asset.get("tiles") or []
    match = next((tile for tile in tiles if str(tile.get("id")) == str(tile_id)), None)
    if match is None:
        raise DefinitionError(f"This asset has no tile '{tile_id}'.")
    x = int(match.get("x", 0))
    y = int(match.get("y", 0))
    w = int(match.get("w", TILE))
    h = int(match.get("h", TILE))
    out = np.zeros((h, w, 4), dtype=np.uint8)
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 > x and y1 > y and x >= 0 and y >= 0:
        patch = array[y:y1, x:x1]
        out[: patch.shape[0], : patch.shape[1]] = patch
    return out


def coerce_field(param: Param, value: Any) -> Any:
    kind = param.type
    if kind == "bool":
        return bool(value)
    if kind == "int":
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = int(param.default or 0)
        if param.min is not None:
            number = max(int(param.min), number)
        if param.max is not None:
            number = min(int(param.max), number)
        return number
    if kind == "choice":
        text = "" if value is None else str(value)
        choices = param.choices or ()
        if text in choices:
            return text
        if param.default in choices:
            return param.default
        return choices[0] if choices else text
    if kind == "colors":
        text = format_palette(value if value not in (None, "") else param.default)
        if not text:
            text = format_palette(param.default)
        return text
    if kind in ("text", "color"):
        return "" if value is None else str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        try:
            number = float(param.default or 0)
        except (TypeError, ValueError):
            number = 0.0
    if param.min is not None:
        number = max(float(param.min), number)
    if param.max is not None:
        number = min(float(param.max), number)
    return number


def plain_fields(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if _plain(value)}


def _plain(value: Any) -> bool:
    return type(value) in (str, int, float, bool)


def _asset_from_dict(item: Any) -> AssetSpec:
    if not isinstance(item, dict):
        raise DefinitionError("Each asset has to be an object with an id, width, and height.")
    asset_id = asset_token(item.get("id", ""), "")
    if not asset_id:
        raise DefinitionError("Each asset needs an id.")
    try:
        width, height = validate_size(int(item.get("width", 0)), int(item.get("height", 0)))
    except (TypeError, ValueError) as exc:
        raise DefinitionError(f"{asset_id}: {exc}") from exc
    tiles = tuple(_tile_from_dict(entry, width, height) for entry in item.get("tiles") or [])
    try:
        fields = tuple(parse_params(item.get("fields") or []))
    except ProcessError as exc:
        raise DefinitionError(f"{asset_id}: {exc}") from exc
    return AssetSpec(
        id=asset_id,
        label=str(item.get("label") or asset_id),
        blurb=str(item.get("blurb") or ""),
        width=width,
        height=height,
        tiles=tiles,
        fields=fields,
        turn=bool(item.get("turn", False)),
        raw=dict(item),
    )


def _tile_from_dict(item: Any, width: int, height: int) -> TileSpec:
    if not isinstance(item, dict) or "id" not in item:
        raise DefinitionError("Each tile needs an id.")
    tile_id = asset_token(item.get("id", ""), "")
    if not tile_id:
        raise DefinitionError("Each tile needs an id.")
    try:
        x = int(item.get("x", 0))
        y = int(item.get("y", 0))
        w = int(item.get("w", TILE))
        h = int(item.get("h", TILE))
    except (TypeError, ValueError) as exc:
        raise DefinitionError(f"Tile {tile_id} has a size that isn't a number.") from exc
    if min(w, h) < TILE or any(value % TILE for value in (x, y, w, h)):
        raise DefinitionError(f"Tile {tile_id} has to sit on the 16 pixel grid.")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise DefinitionError(f"Tile {tile_id} doesn't fit on the {width}×{height} sheet.")
    label = str(item.get("label") or tile_id)
    return TileSpec(tile_id, label, str(item.get("caption") or label), x, y, w, h)


def _as_rgba(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        array = np.array(value.convert("RGBA"), dtype=np.uint8)
    else:
        array = np.asarray(value, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 4 or array.shape[0] < 1 or array.shape[1] < 1:
        raise DefinitionError("preview has to return an RGBA image.")
    return np.ascontiguousarray(array)


def _write_zip(path: Path, produced: dict[str, Any]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in produced.items():
            entry = str(name).lstrip("/")
            if not entry or entry.endswith("/"):
                raise DefinitionError(f"'{name}' is not a file path.")
            archive.writestr(entry, _file_bytes(entry, body))
            names.append(entry)
    tmp.replace(path)
    return names


def _file_bytes(name: str, body: Any) -> bytes:
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, Image.Image) or (isinstance(body, np.ndarray) and body.ndim == 3):
        array = _as_rgba(body)
        buffer = io.BytesIO()
        Image.fromarray(array, mode="RGBA").save(buffer, format="PNG")
        return buffer.getvalue()
    if isinstance(body, (dict, list)):
        return (json.dumps(body, indent=2) + "\n").encode("utf-8")
    raise DefinitionError(f"{name} has to be JSON, text, bytes, or an RGBA image.")


def _load_module(path: Path) -> Any:
    module_name = f"vam_definition_{path.stem}_{abs(hash(str(path.resolve()))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DefinitionError(f"Can't load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
