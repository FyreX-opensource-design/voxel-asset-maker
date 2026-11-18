"""Launch the texture editor.

    python -m voxel_asset_maker
    python -m voxel_asset_maker stone.vam
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .gui.theme import apply_theme
from .gui.window import MainWindow


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    app = QApplication(args)
    apply_theme(app)
    window = MainWindow()
    if len(args) > 1 and not args[1].startswith("-"):
        window._open_path(Path(args[1]))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
