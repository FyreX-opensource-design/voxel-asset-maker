"""Undo history for a texture document."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Document, Layer


@dataclass
class Snapshot:
    generation: int
    width: int
    height: int
    active: int
    name: str
    palette: list[str]
    layers: list[Layer]


class History:
    """Snapshot stack. Arm a snapshot before an edit, then commit it."""

    def __init__(self, limit: int = 80) -> None:
        self.undo_stack: list[Snapshot] = []
        self.redo_stack: list[Snapshot] = []
        self.limit = limit
        self.generation = 0
        self.clean_generation = 0
        self._armed: Snapshot | None = None

    def capture(self, doc: Document) -> Snapshot:
        return Snapshot(
            generation=self.generation,
            width=doc.width,
            height=doc.height,
            active=doc.active_index,
            name=doc.name,
            palette=list(doc.palette),
            layers=[layer.clone_state() for layer in doc.layers],
        )

    def begin(self, doc: Document) -> None:
        if self._armed is None:
            self._armed = self.capture(doc)

    def abandon(self) -> None:
        self._armed = None

    def commit(self, doc: Document) -> bool:
        before = self._armed
        self._armed = None
        if before is None or _same(before, doc):
            return False
        self.undo_stack.append(before)
        if len(self.undo_stack) > self.limit:
            del self.undo_stack[: len(self.undo_stack) - self.limit]
        self.redo_stack.clear()
        self.generation += 1
        return True

    def mark_clean(self) -> None:
        self.clean_generation = self.generation

    def is_clean(self) -> bool:
        return self.generation == self.clean_generation and self._armed is None

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def undo(self, doc: Document) -> bool:
        if not self.undo_stack:
            return False
        self._armed = None
        self.redo_stack.append(self.capture(doc))
        snap = self.undo_stack.pop()
        _restore(doc, snap)
        self.generation = snap.generation
        return True

    def redo(self, doc: Document) -> bool:
        if not self.redo_stack:
            return False
        self._armed = None
        self.undo_stack.append(self.capture(doc))
        snap = self.redo_stack.pop()
        _restore(doc, snap)
        self.generation = snap.generation
        return True

    def reset(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.generation = 0
        self.clean_generation = 0
        self._armed = None


def _restore(doc: Document, snap: Snapshot) -> None:
    doc.width = snap.width
    doc.height = snap.height
    doc.active_index = snap.active
    doc.name = snap.name
    doc.palette = list(snap.palette)
    doc.layers = [layer.clone_state() for layer in snap.layers]
    if doc.layers:
        doc.active_index = max(0, min(doc.active_index, len(doc.layers) - 1))


def _same(snap: Snapshot, doc: Document) -> bool:
    if (
        snap.width != doc.width
        or snap.height != doc.height
        or snap.active != doc.active_index
        or snap.name != doc.name
        or snap.palette != doc.palette
        or len(snap.layers) != len(doc.layers)
    ):
        return False
    for left, right in zip(snap.layers, doc.layers):
        if (
            left.id != right.id
            or left.name != right.name
            or left.visible != right.visible
            or left.opacity != right.opacity
            or left.blend != right.blend
            or tuple(left.offset) != tuple(right.offset)
            or left.script.script_id != right.script.script_id
            or left.script.script_path != right.script.script_path
            or left.script.params != right.script.params
            or not np.array_equal(left.pixels, right.pixels)
        ):
            return False
    return True
