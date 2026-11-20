"""Main window for the voxel texture editor."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QCloseEvent,
    QKeyEvent,
    QKeySequence,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..blockgame import block_view, export_definition, preview_for, sanitize_id, wood_member_id
from ..definitions import export_custom
from ..model import (
    BLOCK_FACE_ORIGIN,
    block_face_origins,
    BLOCK_RENDERS,
    BLOCK_ROTATIONS,
    BLOCK_SHAPES,
    BLOCK_SOUNDS,
    BLOCK_STATIONS,
    BLOCK_TABS,
    BLOCK_TOOLS,
    DOOR_PARTS,
    ROTATION_LABELS,
    SHAPE_LABELS,
    STATION_LABELS,
    TILE,
    TRAPDOOR_PARTS,
    WOOD_MEMBERS,
    WOOD_PARTS,
    BLEND_MODES,
    LootDrop,
    ctm_at,
    ctm_label,
    ctm_origin,
    ctm_short,
    face_on_block,
    part_at,
    format_hex_color,
    parse_hex_color,
)
from ..processors import Param, palette_colors
from ..render import render_variant
from ..session import Session
from .canvas import CanvasWidget
from .dialogs import NewDefinitionDialog, NewTextureDialog, ResizeDialog, ask_text
from .images import numpy_to_qimage, qimage_to_numpy
from .widgets import PaletteChip, Swatch, tool_icon

TOOLS = (
    ("pencil", "Pencil", "B"),
    ("eraser", "Eraser", "E"),
    ("fill", "Fill", "G"),
    ("picker", "Pick", "I"),
    ("line", "Line", "L"),
    ("rect", "Rect", "R"),
    ("select", "Select", "M"),
)

ABOUT = """\
Voxel Asset Maker paints block and item textures. Width and height stay multiples of 16 so they fit Minecraft, block-game, and Lunati slots. A 16×48 canvas is a three-frame strip; the amber grid marks every 16 pixels.

Layers stack from bottom to top. Scripts on a layer run top to bottom, and each one sees the image from the script above it, before the stack is flattened. Result view shows the flattened texture for the variant on the slider. Layer pixels view shows the active layer before its scripts, which is the bitmap you paint.

Noise is texture_noise. Palette noise picks one color from a set you define for each grain, then shifts that color a little lighter or darker. Adjust color is texture_adjust. Edge shade and Brightness steps ship as extra scripts. Drop another .py file in user_scripts/ next to the app, or in ~/.voxel-asset-maker/scripts/, then choose Reload. Add puts another script on the same layer. On a sheet split into parts, check the parts that script should change. Leave them all unchecked and it runs on the whole layer. Help → Layer scripts describes that file. A game other than block game is a definition file in user_scripts/definitions/, or in ~/.voxel-asset-maker/scripts/definitions/. Help → Custom definitions describes it, and examples/lantern.py is a complete one. Reload picks up both kinds of file. File → New → Minecraft → Block paints a cube for KubeJS. Frames stack down each texture, one width tall, and export writes the startup script, the PNGs, and an mcmeta file when there is more than one frame. Extra variants from a layer script become blockstate variants.

The variant slider and box under the toolbar scrub those variants in Result view. They stay on 1 until a script's Count is above 1. Noise starts at 4. Export PNG writes the variant you're viewing. Export Variant ZIP writes one composited PNG per variant, plus manifest.json. The file count is the highest Count on any script. A Noise script with Count 8 makes eight textures, and a shorter script repeats.

File → New → Texture starts a free canvas. Block, item, crop, door, and trapdoor files are under File → New → Block game → Assets, and that kind stays with the file. A wood set is under File → New → Block game → Sets. It paints log top, log side, a stick, planks, the crafting table's top, front, and side, the door's lower half, upper half, and edge, and the trapdoor panel. Export writes the log, the bark wood block, the stick, planks with both slabs and stairs, the vertical-plank versions of those (the planks texture turned a quarter turn), the crafting table, the door, and the trapdoor. The door edge is the trapdoor edge too. Block unfolds the cube into six faces — top, bottom, north, south, east, and west — and drawing stays on the face you started the stroke on. Faces can instead be one texture for every side, or a column of top, side, and bottom. Item is one icon. A crop is a row of plant stages, with its grow time, water, seeds, produce, the soil it roots in, and the loot a ripe plant drops. An unripe plant drops one seed. A block can be a cube, slab, vertical slab, stairs, fence, wall, or a plant. A plant is two crossed planes, and you paint that sprite on the south face. Rotation is none, axis (a log), or horizontal (a furnace faces you). A crafting table or a furnace can name the recipe types it accepts. A door is the lower half, the upper half, and the thin edge. A trapdoor is the panel and the edge. The tab and the sound can be any name the game loads. Hardness, tool, mining level, render mode, and the other block fields are part of a block file. Light is five channels — red, green, blue, ultraviolet, and infrared — each from 0 to 15. A block can also carry connected textures (sixteen tiles, one per neighbor mask) and fluorescence (a glow map or a flat color, with the UV levels where it fades in). Export definition writes a zip with the definition and its textures. It does not change a block-game checkout. The block preview starts looking from the southeast and turns in quarter steps, including a look at the underside.

Custom script shape:

    NAME = "My script"
    PARAMS = [{"key": "count", "label": "Count", "type": "int", "default": 4, "min": 1, "max": 64}]

    def apply(image, params, context):
        # image is a PIL RGBA image. context.variant_index starts at 1.
        return image

Shortcuts: B pencil, E eraser, G fill, I pick, L line, R rectangle, M select.
X swaps colors. [ and ] change the brush. Arrows nudge a selection; Shift jumps 16 pixels.
Ctrl+Z undo, Ctrl+Y or Ctrl+Shift+Z redo, Ctrl+S save.
Ctrl+E exports a PNG. Ctrl+Shift+E exports the variant ZIP.

Scripts you attach run inside this app and can read and write whatever the app can.
"""


class MainWindow(QMainWindow):
    def __init__(self, session: Session | None = None) -> None:
        super().__init__()
        self.session = session or Session()
        self._loading = False
        self._fitted = False
        self._fit_attempts = 0
        self._hover = (-1, -1)
        self.preview_yaw = 0
        self.preview_pitch = 0
        self._history_widgets: set[QWidget] = set()
        self._field_widgets: set[QWidget] = set()
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._preview_while_stroking)
        self._definition_actions: list[QAction] = []
        self._palette_chips: list[PaletteChip] = []
        self.settings = QSettings("voxel-asset-maker", "VoxelAssetMaker")

        self.canvas = CanvasWidget()
        self._build_actions()
        self._build_toolbar()
        self._build_panels()
        self._build_statusbar()
        self._wire()

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.sync_all()
        self.resize(1180, 760)
        self.setMinimumSize(960, 620)

    # --- construction -------------------------------------------------------

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        edit_menu = self.menuBar().addMenu("Edit")
        layer_menu = self.menuBar().addMenu("Layer")
        select_menu = self.menuBar().addMenu("Select")
        view_menu = self.menuBar().addMenu("View")
        help_menu = self.menuBar().addMenu("Help")

        new_menu = file_menu.addMenu("New")
        self.new_menu = new_menu
        self._add_action(new_menu, "Texture…", lambda: self.new_texture("texture"), QKeySequence.StandardKey.New)
        block_game = new_menu.addMenu("Block game")
        assets_menu = block_game.addMenu("Assets")
        self._add_action(assets_menu, "Block…", lambda: self.new_texture("block"))
        self._add_action(assets_menu, "Item…", lambda: self.new_texture("item"))
        self._add_action(assets_menu, "Crop…", lambda: self.new_texture("crop"))
        self._add_action(assets_menu, "Door…", lambda: self.new_texture("door"))
        self._add_action(assets_menu, "Trapdoor…", lambda: self.new_texture("trapdoor"))
        sets_menu = block_game.addMenu("Sets")
        self._add_action(sets_menu, "Wood…", lambda: self.new_texture("set"))
        self._rebuild_definition_menus()
        self._add_action(file_menu, "Open project…", self.open_project, QKeySequence.StandardKey.Open)
        self._add_action(file_menu, "Open image…", self.open_image)
        self._add_action(file_menu, "Save", self.save_project, QKeySequence.StandardKey.Save)
        self._add_action(file_menu, "Save as…", self.save_project_as, QKeySequence.StandardKey.SaveAs)
        file_menu.addSeparator()
        self._add_action(file_menu, "Import layer…", self.import_layer)
        self._add_action(file_menu, "Export PNG…", self.export_png, "Ctrl+E")
        self._add_action(file_menu, "Export variant ZIP…", self.export_zip, "Ctrl+Shift+E")
        self._add_action(file_menu, "Export definition…", self.export_asset_definition, "Ctrl+Shift+B")
        file_menu.addSeparator()
        self._add_action(file_menu, "Quit", self.close, QKeySequence.StandardKey.Quit)

        self._add_action(edit_menu, "Undo", self.undo, QKeySequence.StandardKey.Undo)
        self._add_action(edit_menu, "Redo", self.redo, QKeySequence.StandardKey.Redo)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Cut", self.cut, QKeySequence.StandardKey.Cut)
        self._add_action(edit_menu, "Copy", self.copy, QKeySequence.StandardKey.Copy)
        self._add_action(edit_menu, "Paste", self.paste, QKeySequence.StandardKey.Paste)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Flip horizontal", self._flip_horizontal)
        self._add_action(edit_menu, "Flip vertical", self._flip_vertical)
        self._add_action(edit_menu, "Rotate 90° clockwise", self._rotate)
        self._add_action(edit_menu, "Clear layer", self._clear_layer)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Resize canvas…", self.resize_canvas)
        self._add_action(edit_menu, "Rename texture…", self.rename_texture)

        self._add_action(layer_menu, "New layer", self._add_layer, "Ctrl+Shift+N")
        self._add_action(layer_menu, "Duplicate layer", self._duplicate_layer)
        self._add_action(layer_menu, "Delete layer", self._delete_layer)
        self._add_action(layer_menu, "Merge down", self._merge_down)
        self._add_action(layer_menu, "Move up", lambda: self._move_layer(1))
        self._add_action(layer_menu, "Move down", lambda: self._move_layer(-1))

        self._add_action(select_menu, "All", self._select_all)
        self._add_action(select_menu, "None", self._deselect)
        self._add_action(select_menu, "Delete selection", self._delete_selection)

        self.action_minor_grid = self._add_action(view_menu, "Pixel grid", self._toggle_minor_grid)
        self.action_minor_grid.setCheckable(True)
        self.action_minor_grid.setChecked(True)
        self.action_major_grid = self._add_action(view_menu, "16px block grid", self._toggle_major_grid)
        self.action_major_grid.setCheckable(True)
        self.action_major_grid.setChecked(True)
        view_menu.addSeparator()
        self._add_action(view_menu, "Fit", self.canvas.fit, "Ctrl+0")
        self._add_action(view_menu, "Zoom in", lambda: self.canvas.zoom_by(1), QKeySequence.StandardKey.ZoomIn)
        self._add_action(view_menu, "Zoom out", lambda: self.canvas.zoom_by(-1), QKeySequence.StandardKey.ZoomOut)

        self._add_action(help_menu, "How this editor works", self.show_about)
        self._add_action(help_menu, "Layer scripts", lambda: self.show_doc("layer-scripts.md", "Layer scripts"))
        self._add_action(help_menu, "Custom definitions", lambda: self.show_doc("definitions.md", "Custom definitions"))

    def _add_action(self, menu, label: str, slot, shortcut=None) -> QAction:
        action = QAction(label, self)
        if shortcut is not None:
            if isinstance(shortcut, str):
                action.setShortcut(QKeySequence(shortcut))
            else:
                action.setShortcut(shortcut)
        action.triggered.connect(lambda _checked=False, fn=slot: fn())
        menu.addAction(action)
        return action

    def _rebuild_definition_menus(self) -> None:
        for action in self._definition_actions:
            self.new_menu.removeAction(action)
        self._definition_actions = []
        for spec in self.session.definitions.ordered():
            menu = self.new_menu.addMenu(spec.name)
            if spec.blurb:
                menu.setToolTip(spec.blurb)
            self._definition_actions.append(menu.menuAction())
            for asset in spec.assets:
                self._add_action(
                    menu,
                    f"{asset.label}…",
                    lambda definition_id=spec.id, asset_id=asset.id: self.new_definition(definition_id, asset_id),
                )

    def _build_toolbar(self) -> None:
        bar = QToolBar()
        bar.setMovable(False)
        self.addToolBar(bar)
        self.view_result = QPushButton("Result")
        self.view_result.setCheckable(True)
        self.view_result.setChecked(True)
        self.view_result.setToolTip("Flattened texture with each layer's scripts applied.")
        self.view_layer = QPushButton("Layer pixels")
        self.view_layer.setCheckable(True)
        self.view_layer.setToolTip("The active layer before its scripts run. This is what you paint.")
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_group.addButton(self.view_result)
        self.view_group.addButton(self.view_layer)
        bar.addWidget(self.view_result)
        bar.addWidget(self.view_layer)
        bar.addSeparator()
        fit = QPushButton("Fit")
        fit.clicked.connect(self.canvas.fit)
        bar.addWidget(fit)
        self.zoom_label = QLabel("16×")
        bar.addWidget(self.zoom_label)

    def _build_panels(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self.canvas)
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([200, 720, 300])
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_variant_bar())
        layout.addWidget(splitter, 1)
        self.setCentralWidget(host)

    def _build_variant_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("variantBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        title = QLabel("Variant")
        self.variant_spin = QSpinBox()
        self.variant_spin.setRange(1, 1)
        self.variant_spin.setEnabled(False)
        self.variant_spin.setToolTip("Which variant Result view and Export PNG use.")
        self.variant_slider = QSlider(Qt.Orientation.Horizontal)
        self.variant_slider.setObjectName("variantSlider")
        self.variant_slider.setRange(1, 1)
        self.variant_slider.setEnabled(False)
        self.variant_slider.setMinimumWidth(160)
        self.variant_slider.setToolTip("Scrub variants. A script's Count sets how many there are.")
        self.variant_label = QLabel("of 1 — set Count above 1 on a script")
        self.variant_label.setObjectName("hint")
        layout.addWidget(title)
        layout.addWidget(self.variant_spin)
        layout.addWidget(self.variant_slider, 1)
        layout.addWidget(self.variant_label)
        return bar

    def _build_left(self) -> QWidget:
        inner = QWidget()
        inner.setObjectName("panel")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        tools = QGroupBox("Tools")
        grid = QGridLayout(tools)
        self.tool_buttons: dict[str, QPushButton] = {}
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for index, (tool, label, shortcut) in enumerate(TOOLS):
            button = QPushButton(tool_icon(tool), label)
            button.setCheckable(True)
            button.setToolTip(f"{label} ({shortcut})")
            button.setProperty("tool", True)
            self.tool_group.addButton(button)
            self.tool_buttons[tool] = button
            grid.addWidget(button, index // 2, index % 2)
            button.clicked.connect(lambda _checked=False, name=tool: self._set_tool(name))
        layout.addWidget(tools)

        options = QGroupBox("Brush")
        options_layout = QFormLayout(options)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 32)
        self.brush_size.setValue(1)
        self.brush_size.setSuffix(" px")
        self.filled_shapes = QCheckBox("Filled rectangle")
        options_layout.addRow("Size", self.brush_size)
        options_layout.addRow("", self.filled_shapes)
        layout.addWidget(options)

        color_box = QGroupBox("Color")
        color_layout = QVBoxLayout(color_box)
        chips = QWidget()
        chips.setFixedSize(68, 64)
        self.fg_swatch = Swatch(chips)
        self.fg_swatch.setGeometry(0, 0, 40, 40)
        self.fg_swatch.setToolTip("Front color. Left click paints it. Click the chip to edit.")
        self.bg_swatch = Swatch(chips)
        self.bg_swatch.setGeometry(28, 26, 34, 34)
        self.bg_swatch.setToolTip("Back color. Right click paints it. Click the chip to edit.")
        self.fg_swatch.raise_()
        color_layout.addWidget(chips)
        self.hex_field = QLineEdit()
        self.hex_field.setObjectName("hexField")
        self.hex_field.setPlaceholderText("#rrggbb")
        color_layout.addWidget(self.hex_field)
        swap = QPushButton("Swap (X)")
        swap.clicked.connect(self._swap_colors)
        color_layout.addWidget(swap)
        layout.addWidget(color_box)

        palette_box = QGroupBox("Palette")
        self.palette_layout = QGridLayout(palette_box)
        self.palette_layout.setSpacing(2)
        add = QPushButton("Save swatch")
        add.clicked.connect(lambda: self._add_swatch())
        layout.addWidget(palette_box)
        layout.addWidget(add)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(196)
        scroll.setMaximumWidth(240)
        return scroll

    def _build_right(self) -> QWidget:
        inner = QWidget()
        inner.setObjectName("panel")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layers = QGroupBox("Layers")
        layers_layout = QVBoxLayout(layers)
        self.layer_list = QListWidget()
        self.layer_list.setMinimumHeight(150)
        layers_layout.addWidget(self.layer_list)
        row = QHBoxLayout()
        for label, slot in (
            ("Add", self._add_layer),
            ("Copy", self._duplicate_layer),
            ("Del", self._delete_layer),
            ("Up", lambda: self._move_layer(1)),
            ("Down", lambda: self._move_layer(-1)),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            row.addWidget(button)
        layers_layout.addLayout(row)
        merge = QPushButton("Merge down")
        merge.clicked.connect(self._merge_down)
        layers_layout.addWidget(merge)
        layout.addWidget(layers)

        stack = QGroupBox("Stack")
        form = QFormLayout(stack)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_label = QLabel("100%")
        opacity_row = QWidget()
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.addWidget(self.opacity_slider, 1)
        opacity_layout.addWidget(self.opacity_label)
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(BLEND_MODES)
        self.blend_combo.setToolTip("How this layer combines with the layers under it. The bottom layer is the base.")
        self.offset_x = QSpinBox()
        self.offset_y = QSpinBox()
        for box in (self.offset_x, self.offset_y):
            box.setRange(-512, 512)
        offset_row = QWidget()
        offset_layout = QHBoxLayout(offset_row)
        offset_layout.setContentsMargins(0, 0, 0, 0)
        offset_layout.addWidget(self.offset_x)
        offset_layout.addWidget(self.offset_y)
        form.addRow("Opacity", opacity_row)
        form.addRow("Blend", self.blend_combo)
        form.addRow("Offset", offset_row)
        layout.addWidget(stack)

        script = QGroupBox("Layer scripts")
        script_layout = QVBoxLayout(script)
        self.script_list = QListWidget()
        self.script_list.setMinimumHeight(72)
        self.script_list.setMaximumHeight(110)
        self.script_list.setToolTip("Scripts on this layer, from first to last. Each one sees the previous result.")
        script_layout.addWidget(self.script_list)
        script_buttons = QHBoxLayout()
        self.add_script_button = QPushButton("Add")
        self.remove_script_button = QPushButton("Remove")
        self.script_up_button = QPushButton("Up")
        self.script_down_button = QPushButton("Down")
        self.add_script_button.setToolTip("Add another script after the ones already on this layer.")
        self.script_up_button.setToolTip("Run the selected script earlier.")
        self.script_down_button.setToolTip("Run the selected script later.")
        for button in (
            self.add_script_button,
            self.remove_script_button,
            self.script_up_button,
            self.script_down_button,
        ):
            script_buttons.addWidget(button)
        script_layout.addLayout(script_buttons)
        script_row = QHBoxLayout()
        self.script_combo = QComboBox()
        self.reload_scripts_button = QPushButton("Reload")
        self.reload_scripts_button.setToolTip(
            "Rescan user_scripts and ~/.voxel-asset-maker/scripts, including the definitions folders inside them."
        )
        script_row.addWidget(self.script_combo, 1)
        script_row.addWidget(self.reload_scripts_button)
        script_layout.addLayout(script_row)
        self.script_blurb = QLabel("")
        self.script_blurb.setObjectName("hint")
        self.script_blurb.setWordWrap(True)
        script_layout.addWidget(self.script_blurb)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.script_scroll = QScrollArea()
        self.script_scroll.setWidgetResizable(True)
        self.script_scroll.setMinimumHeight(120)
        self.script_scroll.setWidget(self.form_host)
        script_layout.addWidget(self.script_scroll, 1)
        self.error_label = QLabel("")
        self.error_label.setObjectName("error")
        self.error_label.setWordWrap(True)
        script_layout.addWidget(self.error_label)
        layout.addWidget(script, 1)
        layout.addWidget(self._build_block_game())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(400)
        return scroll

    def _build_block_game(self) -> QGroupBox:
        box = QGroupBox("Asset")
        layout = QVBoxLayout(box)
        kind_row = QHBoxLayout()
        self.asset_kind = QLabel("Texture")
        self.asset_kind.setToolTip(
            "Set when the file was created. Texture is File → New → Texture. "
            "Block, item, crop, door, and trapdoor are File → New → Block game → Assets. "
            "A wood set is File → New → Block game → Sets."
        )
        self.asset_id = QLineEdit()
        self.asset_id.setPlaceholderText("asset_id")
        self.asset_id.setToolTip("Id used in the exported definition.")
        kind_row.addWidget(self.asset_kind)
        kind_row.addWidget(self.asset_id, 1)
        layout.addLayout(kind_row)
        self.asset_preview = QLabel()
        self.asset_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.asset_preview.setMinimumHeight(128)
        self.asset_preview.setStyleSheet("background: #141418; border: 1px solid #34343a;")
        layout.addWidget(self.asset_preview)
        self.set_member = QComboBox()
        self.set_member.setToolTip("Which piece the preview shows. Export writes every piece in the set.")
        self.set_member.blockSignals(True)
        for key, label in WOOD_MEMBERS:
            self.set_member.addItem(label, key)
        self.set_member.blockSignals(False)
        self.set_member.currentIndexChanged.connect(self._set_member_changed)
        layout.addWidget(self.set_member)
        self.definition_host = QWidget()
        self.definition_form = QFormLayout(self.definition_host)
        self.definition_form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.definition_host)
        self.preview_turn = QWidget()
        turn_row = QHBoxLayout(self.preview_turn)
        turn_row.setContentsMargins(0, 0, 0, 0)
        self.preview_left = QPushButton("Left")
        self.preview_left.setToolTip("Turn the view one face to the left.")
        self.preview_right = QPushButton("Right")
        self.preview_right.setToolTip("Turn the view one face to the right.")
        self.preview_flip = QPushButton("Underside")
        self.preview_flip.setCheckable(True)
        self.preview_flip.setToolTip("Look at the bottom face. Each side hangs from the edge that meets the underside.")
        turn_row.addWidget(self.preview_left)
        turn_row.addWidget(self.preview_right)
        turn_row.addWidget(self.preview_flip)
        layout.addWidget(self.preview_turn)
        self.asset_caption = QLabel("")
        self.asset_caption.setObjectName("hint")
        self.asset_caption.setWordWrap(True)
        layout.addWidget(self.asset_caption)
        self.tab_host = QWidget()
        tab_form = QFormLayout(self.tab_host)
        tab_form.setContentsMargins(0, 0, 0, 0)
        self.tab_combo = QComboBox()
        self.tab_combo.setEditable(True)
        self.tab_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for key in BLOCK_TABS:
            self.tab_combo.addItem(key)
        self.tab_combo.setToolTip("Creative tab. Choose one, or type any tab id.")
        tab_form.addRow("Tab", self.tab_combo)
        layout.addWidget(self.tab_host)
        self.block_extra = QWidget()
        extra = QVBoxLayout(self.block_extra)
        extra.setContentsMargins(0, 0, 0, 0)
        fields = QFormLayout()
        fields.setContentsMargins(0, 0, 0, 0)
        self.shape_combo = QComboBox()
        for key in BLOCK_SHAPES:
            self.shape_combo.addItem(SHAPE_LABELS[key], key)
        self.shape_combo.setToolTip(
            "The block's model. Fences and walls preview the shape you hold. "
            "A plant is two crossed planes painted on the south face. "
            "Growth stages and soil are a crop, under File → New → Block game → Assets → Crop. "
            "Doors and trapdoors are their own files in that same menu."
        )
        self.rotation_combo = QComboBox()
        for key in BLOCK_ROTATIONS:
            self.rotation_combo.addItem(ROTATION_LABELS[key], key)
        self.rotation_combo.setToolTip(
            "None stays as placed. Axis turns like a log onto the face you click. "
            "Horizontal faces you, the way a furnace does."
        )
        self.station_combo = QComboBox()
        for key in BLOCK_STATIONS:
            self.station_combo.addItem(STATION_LABELS[key], key)
        self.station_combo.setToolTip(
            "A crafting table opens crafting recipes. A furnace cooks. "
            "Leave this on none for an ordinary block."
        )
        self.recipe_edit = QLineEdit()
        self.recipe_edit.setPlaceholderText("recipe_type")
        self.recipe_edit.setToolTip(
            "Recipe types this station accepts. Separate ids with spaces. "
            "Leave blank to use the station's usual recipes."
        )
        self.cook_spin = QDoubleSpinBox()
        self.cook_spin.setRange(0.25, 16.0)
        self.cook_spin.setSingleStep(0.25)
        self.cook_spin.setDecimals(2)
        self.cook_spin.setToolTip("How fast a furnace cooks, compared with an ordinary furnace.")
        self.hardness_spin = QDoubleSpinBox()
        self.hardness_spin.setRange(0.0, 100.0)
        self.hardness_spin.setSingleStep(0.5)
        self.hardness_spin.setDecimals(2)
        self.tool_combo = QComboBox()
        for key in BLOCK_TOOLS:
            self.tool_combo.addItem(key.capitalize(), key)
        self.mining_spin = QSpinBox()
        self.mining_spin.setRange(0, 4)
        self.mining_spin.setToolTip("0 is wood or by hand, 1 is stone, 2 is iron.")
        self.sound_combo = QComboBox()
        self.sound_combo.setEditable(True)
        self.sound_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for key in BLOCK_SOUNDS:
            self.sound_combo.addItem(key)
        self.sound_combo.setToolTip("Sound set name. Choose one, or type any set the game loads.")
        self.render_combo = QComboBox()
        for key in BLOCK_RENDERS:
            self.render_combo.addItem(key.capitalize(), key)
        self.render_combo.setToolTip("Cutout keeps transparent pixels. Translucent is for glass and water.")
        self.solid_check = QCheckBox("Solid")
        self.solid_check.setToolTip("The block stops movement.")
        self.gravity_check = QCheckBox("Gravity")
        self.gravity_check.setToolTip("The block falls when nothing is under it.")
        self.light_host = QWidget()
        light_grid = QGridLayout(self.light_host)
        light_grid.setContentsMargins(0, 0, 0, 0)
        light_grid.setHorizontalSpacing(8)
        light_grid.setVerticalSpacing(2)
        self.light_spins: dict[str, QSpinBox] = {}
        for index, (key, label, tip) in enumerate(
            (
                ("r", "R", "Red light, from 0 to 15."),
                ("g", "G", "Green light, from 0 to 15."),
                ("b", "B", "Blue light, from 0 to 15."),
                ("uv", "UV", "Ultraviolet light, from 0 to 15. Fluorescence reacts to this."),
                ("ir", "IR", "Infrared light, from 0 to 15."),
            )
        ):
            spin = QSpinBox()
            spin.setRange(0, 15)
            spin.setToolTip(tip)
            caption = QLabel(label)
            caption.setToolTip(tip)
            cell = QWidget()
            cell_row = QHBoxLayout(cell)
            cell_row.setContentsMargins(0, 0, 0, 0)
            cell_row.setSpacing(4)
            cell_row.addWidget(caption)
            cell_row.addWidget(spin)
            light_grid.addWidget(cell, index // 3, index % 3)
            self.light_spins[key] = spin
        self.full_block_edit = QLineEdit()
        self.full_block_edit.setPlaceholderText("full_block_id")
        self.full_block_edit.setToolTip("The full block two of these vertical slabs combine into.")
        flags = QWidget()
        flag_row = QHBoxLayout(flags)
        flag_row.setContentsMargins(0, 0, 0, 0)
        flag_row.addWidget(self.solid_check)
        flag_row.addWidget(self.gravity_check)
        fields.addRow("Shape", self.shape_combo)
        self.face_mode = QComboBox()
        self.face_mode.addItem("Six faces", "six")
        self.face_mode.addItem("One texture", "all")
        self.face_mode.addItem("Column", "column")
        self.face_mode.setToolTip(
            "Six faces is the unfolded cube. One texture is painted once and used on every face. "
            "Column is top, side, and bottom."
        )
        fields.addRow("Faces", self.face_mode)
        fields.addRow("Rotation", self.rotation_combo)
        fields.addRow("Station", self.station_combo)
        fields.addRow("Recipes", self.recipe_edit)
        fields.addRow("Cook speed", self.cook_spin)
        fields.addRow("Hardness", self.hardness_spin)
        fields.addRow("Tool", self.tool_combo)
        fields.addRow("Mining", self.mining_spin)
        fields.addRow("Sound", self.sound_combo)
        fields.addRow("Render", self.render_combo)
        fields.addRow("Light", self.light_host)
        fields.addRow(flags)
        fields.addRow("Full block", self.full_block_edit)
        extra.addLayout(fields)
        self._block_fields = fields
        surface_row = QHBoxLayout()
        self.surface_group = QButtonGroup(self)
        self.surface_group.setExclusive(True)
        self.surface_buttons: dict[str, QPushButton] = {}
        for key, label, tip in (
            ("faces", "Faces", "Paint the block textures."),
            ("ctm", "Connections", "Paint the sixteen connected-texture tiles. U up, D down, L left, R right."),
            ("glow", "Glow", "Paint the fluorescence map."),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(tip)
            self.surface_group.addButton(button)
            self.surface_buttons[key] = button
            surface_row.addWidget(button)
        extra.addLayout(surface_row)
        self.ctm_check = QCheckBox("Connected textures")
        self.ctm_check.setToolTip(
            "Export ctm_0 through ctm_15. Each tile is the face when those edges touch the same block."
        )
        extra.addWidget(self.ctm_check)
        self.fluor_check = QCheckBox("Fluorescence")
        self.fluor_check.setToolTip("The block glows under ultraviolet light.")
        extra.addWidget(self.fluor_check)
        self.fluor_options = QWidget()
        fluor_form = QFormLayout(self.fluor_options)
        fluor_form.setContentsMargins(0, 0, 0, 0)
        self.fluor_mode = QComboBox()
        self.fluor_mode.addItem("Glow map", "texture")
        self.fluor_mode.addItem("Flat color", "color")
        self.fluor_color = QPushButton("#9dff57")
        self.fluor_start = QSpinBox()
        self.fluor_start.setRange(0, 15)
        self.fluor_full = QSpinBox()
        self.fluor_full.setRange(0, 15)
        self.fluor_strength = QDoubleSpinBox()
        self.fluor_strength.setRange(0.0, 4.0)
        self.fluor_strength.setSingleStep(0.25)
        self.fluor_strength.setDecimals(2)
        fluor_form.addRow("Glow", self.fluor_mode)
        fluor_form.addRow("Color", self.fluor_color)
        fluor_form.addRow("Start", self.fluor_start)
        fluor_form.addRow("Full", self.fluor_full)
        fluor_form.addRow("Strength", self.fluor_strength)
        extra.addWidget(self.fluor_options)
        layout.addWidget(self.block_extra)
        self.crop_extra = QWidget()
        crop_form = QFormLayout(self.crop_extra)
        crop_form.setContentsMargins(0, 0, 0, 0)
        self.crop_stages = QSpinBox()
        self.crop_stages.setRange(2, 16)
        self.crop_stages.setToolTip("How many growth sprites. Each one is a 16×16 plant.")
        self.crop_view = QSpinBox()
        self.crop_view.setRange(1, 16)
        self.crop_view.setToolTip("Which stage the preview shows. Painting a stage selects it too.")
        self.crop_min = QDoubleSpinBox()
        self.crop_min.setRange(1.0, 10000.0)
        self.crop_min.setDecimals(1)
        self.crop_min.setSingleStep(10.0)
        self.crop_max = QDoubleSpinBox()
        self.crop_max.setRange(1.0, 10000.0)
        self.crop_max.setDecimals(1)
        self.crop_max.setSingleStep(10.0)
        self.crop_water = QCheckBox("Needs water")
        self.crop_radius = QSpinBox()
        self.crop_radius.setRange(0, 16)
        self.crop_vertical = QSpinBox()
        self.crop_vertical.setRange(0, 8)
        self.crop_seeds = QLineEdit()
        self.crop_seeds.setPlaceholderText("crop_seeds")
        self.crop_produce = QLineEdit()
        self.crop_produce.setPlaceholderText("same as the crop id")
        self.crop_loot_host = QWidget()
        self.crop_loot_rows = QVBoxLayout(self.crop_loot_host)
        self.crop_loot_rows.setContentsMargins(0, 0, 0, 0)
        self.crop_loot_rows.setSpacing(4)
        self.loot_rows: list[QWidget] = []
        self.crop_loot_add = QPushButton("Add drop")
        self.crop_loot_add.setToolTip("Add another item a ripe plant can drop.")
        loot_box = QWidget()
        loot_layout = QVBoxLayout(loot_box)
        loot_layout.setContentsMargins(0, 0, 0, 0)
        loot_layout.addWidget(self.crop_loot_host)
        loot_layout.addWidget(self.crop_loot_add)
        loot_box.setToolTip(
            "What a ripe plant drops. The two counts are the least and the most. "
            "Leave an item blank to use the produce, then the seeds. An unripe plant drops one seed."
        )
        self.crop_ground = QLineEdit()
        self.crop_ground.setPlaceholderText("farmland")
        self.crop_ground.setToolTip("Blocks this crop can root in. Separate ids with spaces or commas.")
        self.crop_seed_tab = QComboBox()
        self.crop_seed_tab.setEditable(True)
        self.crop_seed_tab.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.crop_produce_tab = QComboBox()
        self.crop_produce_tab.setEditable(True)
        self.crop_produce_tab.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for combo in (self.crop_seed_tab, self.crop_produce_tab):
            for key in BLOCK_TABS:
                combo.addItem(key)
            combo.setToolTip("Creative tab. Choose one, or type any tab id.")
        crop_form.addRow("Stages", self.crop_stages)
        crop_form.addRow("Preview", self.crop_view)
        crop_form.addRow("Min time", self.crop_min)
        crop_form.addRow("Max time", self.crop_max)
        crop_form.addRow(self.crop_water)
        crop_form.addRow("Water radius", self.crop_radius)
        crop_form.addRow("Water vertical", self.crop_vertical)
        crop_form.addRow("Seeds", self.crop_seeds)
        crop_form.addRow("Seed tab", self.crop_seed_tab)
        crop_form.addRow("Produce", self.crop_produce)
        crop_form.addRow("Produce tab", self.crop_produce_tab)
        crop_form.addRow("Loot", loot_box)
        crop_form.addRow("Soil", self.crop_ground)
        layout.addWidget(self.crop_extra)
        self.export_definition_button = QPushButton("Export definition…")
        self.export_definition_button.clicked.connect(self.export_asset_definition)
        layout.addWidget(self.export_definition_button)
        self._asset_id_auto = ""
        return box

    def _build_statusbar(self) -> None:
        status = QStatusBar()
        self.status_left = QLabel("")
        self.status_right = QLabel("")
        status.addWidget(self.status_left, 1)
        status.addPermanentWidget(self.status_right)
        self.setStatusBar(status)

    def _wire(self) -> None:
        self.canvas.pressed.connect(self._pointer_down)
        self.canvas.moved.connect(self._pointer_move)
        self.canvas.released.connect(self._pointer_up)
        self.canvas.hovered.connect(self._hovered)
        self.canvas.zoomChanged.connect(lambda zoom: self.zoom_label.setText(f"{zoom}×"))
        self.view_result.clicked.connect(lambda: self._set_view("composite"))
        self.view_layer.clicked.connect(lambda: self._set_view("layer"))
        self.variant_slider.valueChanged.connect(self._set_variant)
        self.variant_spin.valueChanged.connect(self._set_variant)
        self.brush_size.valueChanged.connect(self._set_brush)
        self.filled_shapes.toggled.connect(self._set_filled)
        self.fg_swatch.clicked.connect(lambda _button: self._edit_color(True))
        self.bg_swatch.clicked.connect(lambda _button: self._edit_color(False))
        self.hex_field.editingFinished.connect(self._apply_hex)
        self.layer_list.currentRowChanged.connect(self._layer_row_changed)
        self.layer_list.itemChanged.connect(self._layer_check_changed)
        self.layer_list.itemDoubleClicked.connect(self._rename_layer)
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        self.opacity_slider.sliderPressed.connect(self.session.begin_edit)
        self.opacity_slider.sliderReleased.connect(self._slider_released)
        self.blend_combo.currentTextChanged.connect(self._blend_changed)
        self.offset_x.valueChanged.connect(self._offset_changed)
        self.offset_y.valueChanged.connect(self._offset_changed)
        self.script_list.currentRowChanged.connect(self._script_row_changed)
        self.add_script_button.clicked.connect(self._add_script)
        self.remove_script_button.clicked.connect(self._remove_script)
        self.script_up_button.clicked.connect(lambda: self._move_script(-1))
        self.script_down_button.clicked.connect(lambda: self._move_script(1))
        self.script_combo.currentIndexChanged.connect(self._script_changed)
        self.reload_scripts_button.clicked.connect(self._reload_scripts)
        self.preview_left.clicked.connect(lambda: self._turn_preview(-1))
        self.preview_right.clicked.connect(lambda: self._turn_preview(1))
        self.preview_flip.toggled.connect(self._flip_preview)
        self.asset_id.textEdited.connect(self._asset_id_edited)
        for key, button in self.surface_buttons.items():
            button.clicked.connect(lambda _checked=False, name=key: self._surface_clicked(name))
        self.ctm_check.toggled.connect(self._ctm_toggled)
        self.fluor_check.toggled.connect(self._fluor_toggled)
        self.fluor_mode.currentIndexChanged.connect(self._fluor_mode_changed)
        self.fluor_color.clicked.connect(self._pick_fluor_color)
        self.fluor_start.valueChanged.connect(self._fluor_range_changed)
        self.fluor_full.valueChanged.connect(self._fluor_range_changed)
        self.fluor_strength.valueChanged.connect(self._fluor_strength_changed)
        self.shape_combo.currentIndexChanged.connect(self._shape_changed)
        self.face_mode.currentIndexChanged.connect(self._face_mode_changed)
        self.rotation_combo.currentIndexChanged.connect(self._block_trait_changed)
        self.station_combo.currentIndexChanged.connect(self._station_changed)
        self.recipe_edit.textEdited.connect(self._block_trait_changed)
        self.recipe_edit.editingFinished.connect(self._recipe_finished)
        self.cook_spin.valueChanged.connect(self._block_trait_changed)
        self.tool_combo.currentIndexChanged.connect(self._block_trait_changed)
        self.sound_combo.currentTextChanged.connect(self._block_trait_changed)
        self.sound_combo.lineEdit().editingFinished.connect(self._sound_finished)
        self.tab_combo.currentTextChanged.connect(self._tab_edited)
        self.tab_combo.lineEdit().editingFinished.connect(self._tab_finished)
        self.crop_stages.valueChanged.connect(self._crop_stages_changed)
        self.crop_view.valueChanged.connect(self._crop_view_changed)
        self.crop_min.valueChanged.connect(self._crop_changed)
        self.crop_max.valueChanged.connect(self._crop_changed)
        self.crop_water.toggled.connect(self._crop_changed)
        self.crop_radius.valueChanged.connect(self._crop_changed)
        self.crop_vertical.valueChanged.connect(self._crop_changed)
        self.crop_loot_add.clicked.connect(self._add_loot_row)
        self.crop_seeds.textEdited.connect(self._crop_changed)
        self.crop_produce.textEdited.connect(self._crop_changed)
        self.crop_ground.textEdited.connect(self._crop_changed)
        self.crop_seed_tab.currentTextChanged.connect(self._crop_changed)
        self.crop_produce_tab.currentTextChanged.connect(self._crop_changed)
        for edit in (self.crop_seeds, self.crop_produce, self.crop_ground):
            edit.editingFinished.connect(self._crop_finished)
        self.crop_seed_tab.lineEdit().editingFinished.connect(self._crop_finished)
        self.crop_produce_tab.lineEdit().editingFinished.connect(self._crop_finished)
        self.render_combo.currentIndexChanged.connect(self._block_trait_changed)
        self.hardness_spin.valueChanged.connect(self._block_trait_changed)
        self.mining_spin.valueChanged.connect(self._block_trait_changed)
        for spin in self.light_spins.values():
            spin.valueChanged.connect(self._block_trait_changed)
        self.solid_check.toggled.connect(self._block_trait_changed)
        self.gravity_check.toggled.connect(self._block_trait_changed)
        self.full_block_edit.textEdited.connect(self._full_block_edited)
        self.full_block_edit.editingFinished.connect(self._full_block_finished)
        for widget in (
            self.opacity_slider,
            self.blend_combo,
            self.offset_x,
            self.offset_y,
            self.ctm_check,
            self.fluor_check,
            self.fluor_mode,
            self.fluor_start,
            self.fluor_full,
            self.fluor_strength,
            self.shape_combo,
            self.face_mode,
            self.rotation_combo,
            self.station_combo,
            self.recipe_edit,
            self.cook_spin,
            self.tool_combo,
            self.sound_combo,
            self.render_combo,
            self.hardness_spin,
            self.mining_spin,
            *self.light_spins.values(),
            self.solid_check,
            self.gravity_check,
            self.full_block_edit,
            self.tab_combo,
            self.crop_stages,
            self.crop_min,
            self.crop_max,
            self.crop_water,
            self.crop_radius,
            self.crop_vertical,
            self.crop_seeds,
            self.crop_produce,
            self.crop_loot_add,
            self.crop_ground,
            self.crop_seed_tab,
            self.crop_produce_tab,
            *self.surface_buttons.values(),
        ):
            self._watch_history(widget)
        self.setAcceptDrops(True)

    # --- sync ---------------------------------------------------------------

    def sync_all(self) -> None:
        self._fill_script_combo()
        self.rebuild_layers()
        self.sync_layer_controls()
        self.sync_colors()
        self.rebuild_palette()
        self._sync_tool_buttons()
        self.view_result.setChecked(self.session.view_mode == "composite")
        self.view_layer.setChecked(self.session.view_mode == "layer")
        self.brush_size.blockSignals(True)
        self.brush_size.setValue(self.session.brush_size)
        self.brush_size.blockSignals(False)
        self.filled_shapes.blockSignals(True)
        self.filled_shapes.setChecked(self.session.filled_shapes)
        self.filled_shapes.blockSignals(False)
        self._sync_kind()
        self._sync_definition_fields()
        self._sync_block_extra()
        self._sync_crop()
        self._sync_asset_id()
        self.refresh_canvas()
        self.sync_title()

    def sync_title(self) -> None:
        name = self.session.path.name if self.session.path else self.session.doc.name
        star = "" if self.session.history.is_clean() else "*"
        self.setWindowTitle(f"{name}{star} — Voxel Asset Maker")

    def sync_colors(self) -> None:
        self._loading = True
        self.fg_swatch.set_color(self.session.fg)
        self.bg_swatch.set_color(self.session.bg)
        self.hex_field.setText(format_hex_color(self.session.fg))
        self._loading = False

    def rebuild_palette(self) -> None:
        while self.palette_layout.count():
            item = self.palette_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._palette_chips.clear()
        for index, hex_color in enumerate(self.session.doc.palette):
            chip = PaletteChip(hex_color)
            chip.picked.connect(self._pick_palette)
            self.palette_layout.addWidget(chip, index // 8, index % 8)
            self._palette_chips.append(chip)

    def rebuild_layers(self) -> None:
        self._loading = True
        self.layer_list.clear()
        for layer in reversed(self.session.doc.layers):
            item = QListWidgetItem(layer.name)
            item.setData(Qt.ItemDataRole.UserRole, layer.id)
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            self.layer_list.addItem(item)
        row = len(self.session.doc.layers) - 1 - self.session.doc.active_index
        self.layer_list.setCurrentRow(row)
        self._loading = False

    def sync_layer_controls(self) -> None:
        layer = self.session.doc.active_layer
        self._loading = True
        self.opacity_slider.setValue(int(round(layer.opacity * 100)))
        self.opacity_label.setText(f"{int(round(layer.opacity * 100))}%")
        blend_index = self.blend_combo.findText(layer.blend)
        if blend_index >= 0:
            self.blend_combo.setCurrentIndex(blend_index)
        blocked = self.session.doc.kind in ("block", "crop", "set", "custom")
        self.offset_x.setEnabled(not blocked)
        self.offset_y.setEnabled(not blocked)
        self.offset_x.setValue(0 if blocked else int(layer.offset[0]))
        self.offset_y.setValue(0 if blocked else int(layer.offset[1]))
        self._loading = False
        self.rebuild_script_list()
        self.rebuild_script_form()

    def rebuild_script_list(self) -> None:
        self._loading = True
        self.script_list.clear()
        layer = self.session.doc.active_layer
        for index, binding in enumerate(layer.scripts):
            if self.session.library.known(binding.script_id):
                label = self.session.library.get(binding.script_id).label
            else:
                label = binding.script_id
            if binding.parts:
                names = {part.id: part.label for part in self.session.sheet_parts()}
                shown = ", ".join(names.get(part_id, part_id) for part_id in binding.parts)
                label = f"{label} · {shown}"
            self.script_list.addItem(QListWidgetItem(f"{index + 1}. {label}"))
        if layer.scripts:
            row = max(0, min(self.session.script_index, len(layer.scripts) - 1))
            self.session.script_index = row
            self.script_list.setCurrentRow(row)
        self._loading = False

    def rebuild_script_form(self) -> None:
        self._loading = True
        self._history_widgets.clear()
        for widget in (self.opacity_slider, self.blend_combo, self.offset_x, self.offset_y):
            self._history_widgets.add(widget)
        while self.form.rowCount():
            self.form.removeRow(0)
        binding = self.session.current_binding()
        script_id = binding.script_id if binding is not None else "none"
        proc = self.session.library.get(script_id)
        if binding is None:
            self.script_blurb.setText("Add a script to run it on this layer. Scripts run top to bottom.")
            self._select_script("none")
            self._loading = False
            return
        if script_id != "none" and not self.session.library.known(script_id):
            missing = QLabel(f"Missing script {script_id}.")
            missing.setObjectName("error")
            missing.setWordWrap(True)
            self.form.addRow(missing)
        self.script_blurb.setText(proc.blurb)
        self._select_script(script_id)
        params = {param.key: param.default for param in proc.params}
        params.update({key: value for key, value in binding.params.items() if key in params})
        for param in proc.params:
            widget = self._param_widget(param, params.get(param.key))
            label = QLabel(param.label)
            if param.help:
                label.setToolTip(param.help)
                widget.setToolTip(param.help)
            self.form.addRow(label, widget)
        if any(param.key == "seed" for param in proc.params):
            button = QPushButton("Randomize seed")
            button.clicked.connect(self._randomize_seed)
            self.form.addRow("", button)
        parts = self.session.sheet_parts() if script_id != "none" else ()
        if parts:
            host = QWidget()
            grid = QGridLayout(host)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(2)
            chosen = set(binding.parts)
            for index, part in enumerate(parts):
                box = QCheckBox(part.label)
                box.setObjectName(f"part-{part.id}")
                box.setToolTip(part.id)
                box.setChecked(part.id in chosen)
                box.toggled.connect(lambda _checked, part_id=part.id: self._toggle_script_part(part_id))
                self._watch_history(box)
                grid.addWidget(box, index // 3, index % 3)
            caption = QLabel("Parts")
            caption.setToolTip("Checked parts get this script. Leave them all unchecked and it runs on the whole layer.")
            self.form.addRow(caption, host)
        self._loading = False

    def refresh_canvas(self, *, live: bool = True) -> None:
        self.canvas.set_pixels(self.session.display_array())
        self.canvas.set_selection(self.session.selection, self.session.selection_offset())
        self.canvas.show_minor_grid = self.session.show_minor_grid
        self.canvas.show_major_grid = self.session.show_major_grid
        if self.session.doc.kind == "block" and self.session.surface == "ctm":
            self.canvas.face_boxes = tuple(
                (ctm_short(index), *ctm_origin(index), TILE, TILE) for index in range(16)
            )
        elif self.session.doc.kind == "crop" and self.session.surface != "glow":
            self.canvas.face_boxes = tuple(
                (str(index + 1), index * TILE, 0, TILE, TILE) for index in range(self.session.doc.crop.stages)
            )
        elif self.session.doc.kind == "door":
            self.canvas.face_boxes = tuple(
                (name, index * TILE, 0, TILE, TILE) for index, name in enumerate(DOOR_PARTS)
            )
        elif self.session.doc.kind == "trapdoor":
            self.canvas.face_boxes = tuple(
                (name, index * TILE, 0, TILE, TILE) for index, name in enumerate(TRAPDOOR_PARTS)
            )
        elif self.session.doc.kind == "set":
            self.canvas.face_boxes = tuple(
                (label, index * TILE, 0, TILE, TILE) for index, (_key, label, _caption) in enumerate(WOOD_PARTS)
            )
        elif self.session.doc.kind == "custom":
            tiles = self.session.definitions.tiles_for(self.session.doc) or ()
            self.canvas.face_boxes = tuple(
                (tile.label, tile.x, tile.y, tile.w, tile.h) for tile in tiles
            )
        elif self.session.doc.kind == "block" and self.session.surface != "glow":
            mode = self.session.doc.traits.faces
            origins = block_face_origins(mode)
            if self.session.doc.traits.shape == "cross":
                if mode == "all":
                    self.canvas.face_boxes = (("plant", 0, 0, TILE, TILE),)
                elif mode == "column":
                    self.canvas.face_boxes = (("plant", TILE, 0, TILE, TILE),)
                else:
                    origin_x, origin_y = BLOCK_FACE_ORIGIN["south"]
                    self.canvas.face_boxes = (("plant", origin_x, origin_y, TILE, TILE),)
            else:
                self.canvas.face_boxes = tuple(
                    (name, origin_x, origin_y, TILE, TILE) for name, (origin_x, origin_y) in origins.items()
                )
        else:
            self.canvas.face_boxes = ()
        if live:
            self.update_variant_controls()
            self.update_errors()
            self._preview_timer.stop()
            self.refresh_preview()
        elif not self._preview_timer.isActive():
            # A block preview is slow enough that one per touch sample leaves the stroke behind the finger.
            self._preview_timer.start(80)
        self.update_status()

    def update_variant_controls(self) -> None:
        count = max(1, self.session.variant_count())
        index = min(self.session.variant_index, count)
        for widget in (self.variant_slider, self.variant_spin):
            widget.blockSignals(True)
            widget.setRange(1, count)
            widget.setValue(index)
            widget.setEnabled(count > 1)
            widget.blockSignals(False)
        sources = self.session.variant_labels()
        if count <= 1:
            text = "of 1 — set Count above 1 on a script"
        else:
            text = f"of {count}"
            if sources:
                text += "  (" + ", ".join(f"{name} ×{number}" for name, number in sources) + ")"
        self.variant_label.setText(text)

    def update_errors(self) -> None:
        lines = list(self.session.script_errors)
        lines.extend(self.session.library.errors)
        lines.extend(self.session.definitions.errors)
        self.error_label.setText("\n".join(lines))
        self.error_label.setVisible(bool(lines))

    def update_status(self) -> None:
        doc = self.session.doc
        x, y = self._hover
        if x < 0:
            position = "—"
            sampled = ""
        else:
            position = f"x {x}  y {y}"
            image = self.session.display_array()
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                sampled = "  " + format_hex_color(tuple(int(channel) for channel in image[y, x]))
            else:
                sampled = ""
        face = ""
        if doc.kind == "block" and self.session.surface == "ctm" and x >= 0:
            index = ctm_at(x, y)
            face = f"    {ctm_label(index)}" if index is not None else "    outside tiles"
        elif doc.kind == "block" and self.session.surface == "glow" and x >= 0:
            face = "    glow"
        elif doc.kind in ("door", "trapdoor", "set") and x >= 0:
            if doc.kind == "set":
                index = part_at(x, y, len(WOOD_PARTS))
                face = f"    {WOOD_PARTS[index][2]}" if index is not None else "    outside tiles"
            else:
                names = DOOR_PARTS if doc.kind == "door" else TRAPDOOR_PARTS
                index = part_at(x, y, len(names))
                face = f"    {names[index]}" if index is not None else "    outside tiles"
        elif doc.kind == "custom" and x >= 0:
            tiles = self.session.definitions.tiles_for(doc) or ()
            match = next((tile for tile in tiles if tile.contains(x, y)), None)
            if tiles:
                face = f"    {match.caption}" if match is not None else "    outside tiles"
        elif doc.kind == "crop" and x >= 0:
            index = x // TILE
            face = f"    stage {index + 1}" if 0 <= index < doc.crop.stages and 0 <= y < TILE else "    outside stages"
        elif doc.kind == "block" and x >= 0:
            mode = doc.traits.faces
            name = face_on_block(x, y, mode)
            plant = doc.traits.shape == "cross"
            if plant and ((mode == "six" and name == "south") or (mode == "column" and name == "side") or mode == "all"):
                name = "plant" if name else None
            elif plant and name:
                name = None
            face = f"    {name}" if name else "    outside faces"
        if self.session.surface == "ctm":
            size = "64×64"
        elif self.session.surface == "glow":
            size = "16×16"
        else:
            size = f"{doc.width}×{doc.height}"
        self.status_left.setText(
            f"{size}    {self.canvas.zoom}×    {position}{sampled}    {self.session.tool}{face}"
        )
        self.status_right.setText(self.session.message)

    # --- painting -----------------------------------------------------------

    def _pointer_down(self, x: int, y: int, button: int) -> None:
        focused = QApplication.focusWidget()
        if focused is not None and focused is not self.canvas:
            focused.clearFocus()
        self.session.pointer_down(x, y, button)
        if self.session.tool == "picker":
            self.sync_colors()
        self.refresh_canvas(live=not self.session.painting)
        self.sync_title()

    def _pointer_move(self, x: int, y: int) -> None:
        if self.session.pointer_move(x, y):
            self.refresh_canvas(live=False)

    def _pointer_up(self, x: int, y: int) -> None:
        self._preview_timer.stop()
        self.session.pointer_up(x, y)
        self.refresh_canvas()
        self.sync_title()

    def _preview_while_stroking(self) -> None:
        if not self.session.painting:
            return
        self.refresh_preview()
        if self.session.painting and not self._preview_timer.isActive():
            self._preview_timer.start(80)

    def _hovered(self, x: int, y: int) -> None:
        self._hover = (x, y)
        self.update_status()

    def _set_tool(self, name: str) -> None:
        self.session.tool = name
        self._sync_tool_buttons()
        self.update_status()

    def _sync_tool_buttons(self) -> None:
        button = self.tool_buttons.get(self.session.tool)
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def _set_view(self, mode: str) -> None:
        self.session.view_mode = mode
        self.session.invalidate()
        self.refresh_canvas()

    def _set_variant(self, value: int) -> None:
        if self._loading:
            return
        if self.session.view_mode != "composite":
            self.session.view_mode = "composite"
            self.view_result.blockSignals(True)
            self.view_layer.blockSignals(True)
            self.view_result.setChecked(True)
            self.view_layer.setChecked(False)
            self.view_result.blockSignals(False)
            self.view_layer.blockSignals(False)
        self.session.set_variant(int(value))
        self.refresh_canvas()

    def _set_brush(self, value: int) -> None:
        if self._loading:
            return
        self.session.brush_size = int(value)

    def _set_filled(self, checked: bool) -> None:
        if self._loading:
            return
        self.session.filled_shapes = bool(checked)

    # --- color --------------------------------------------------------------

    def _edit_color(self, front: bool) -> None:
        current = self.session.fg if front else self.session.bg
        chosen = QColorDialog.getColor(
            QColor(*current),
            self,
            "Front color" if front else "Back color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not chosen.isValid():
            return
        color = (chosen.red(), chosen.green(), chosen.blue(), chosen.alpha())
        if front:
            self.session.fg = color
        else:
            self.session.bg = color
        self.sync_colors()

    def _apply_hex(self) -> None:
        if self._loading:
            return
        try:
            self.session.fg = parse_hex_color(self.hex_field.text())
        except ValueError:
            pass
        self.sync_colors()

    def _swap_colors(self) -> None:
        self.session.swap_colors()
        self.sync_colors()

    def _pick_palette(self, hex_color: str, button: int) -> None:
        try:
            color = parse_hex_color(hex_color)
        except ValueError:
            return
        if button == 2:
            self.session.bg = color
        else:
            self.session.fg = color
        self.sync_colors()

    def _add_swatch(self) -> None:
        self.session.add_swatch()
        self.rebuild_palette()
        self.sync_title()

    # --- layers -------------------------------------------------------------

    def _layer_row_changed(self, row: int) -> None:
        if self._loading or row < 0:
            return
        index = len(self.session.doc.layers) - 1 - row
        self.session.set_active(index)
        self.sync_layer_controls()
        self.refresh_canvas()

    def _layer_check_changed(self, item: QListWidgetItem) -> None:
        if self._loading:
            return
        self.session.begin_edit()
        self.session.set_layer_visible(item.data(Qt.ItemDataRole.UserRole), item.checkState() == Qt.CheckState.Checked)
        self.session.end_edit()
        self.refresh_canvas()
        self.sync_title()

    def _rename_layer(self, item: QListWidgetItem) -> None:
        name = ask_text(self, "Rename layer", "Layer name", item.text())
        if not name:
            return
        self.session.begin_edit()
        self.session.doc.active_layer.name = name
        self.session.end_edit()
        self.rebuild_layers()
        self.sync_title()

    def _add_layer(self) -> None:
        self.session.add_layer()
        self._after_structure()

    def _duplicate_layer(self) -> None:
        self.session.duplicate_layer()
        self._after_structure()

    def _delete_layer(self) -> None:
        self.session.delete_layer()
        self._after_structure()

    def _move_layer(self, delta: int) -> None:
        self.session.move_layer(delta)
        self._after_structure()

    def _merge_down(self) -> None:
        layer = self.session.doc.active_layer
        if any(binding.script_id not in ("", "none") for binding in layer.scripts):
            answer = QMessageBox.question(
                self,
                "Merge layer",
                "Merge blends the painted pixels onto the layer below. "
                "The scripts on this layer are dropped and stay unbaked.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        if self.session.merge_down():
            self._after_structure()

    def _after_structure(self) -> None:
        self.rebuild_layers()
        self.sync_layer_controls()
        self.refresh_canvas()
        self.sync_title()
        self.update_status()

    def _slider_released(self) -> None:
        self.session.end_edit()
        self.sync_title()

    def _flip_horizontal(self) -> None:
        self.session.flip_horizontal()
        self.refresh_canvas()
        self.sync_title()
        self.update_status()

    def _flip_vertical(self) -> None:
        self.session.flip_vertical()
        self.refresh_canvas()
        self.sync_title()
        self.update_status()

    def _opacity_changed(self, value: int) -> None:
        if self._loading:
            return
        self.session.set_opacity(value / 100.0)
        self.opacity_label.setText(f"{value}%")
        self.refresh_canvas()
        self.sync_title()

    def _blend_changed(self, mode: str) -> None:
        if self._loading or not mode:
            return
        self.session.set_blend(mode)
        self.refresh_canvas()
        self.sync_title()

    def _offset_changed(self, _value: int) -> None:
        if self._loading:
            return
        self.session.set_offset(self.offset_x.value(), self.offset_y.value())
        self.refresh_canvas()
        self.sync_title()

    # --- scripts ------------------------------------------------------------

    def _current_script_id(self) -> str:
        binding = self.session.current_binding()
        return binding.script_id if binding is not None else "none"

    def _fill_script_combo(self) -> None:
        current = self._current_script_id()
        self.script_combo.blockSignals(True)
        self.script_combo.clear()
        separated = False
        for script_id in self.session.library.order:
            proc = self.session.library.processors[script_id]
            if script_id.startswith("script:") and not separated:
                self.script_combo.insertSeparator(self.script_combo.count())
                separated = True
            self.script_combo.addItem(proc.label, script_id)
        if current not in self.session.library.processors:
            self.script_combo.insertItem(0, f"Missing ({current})", current)
        self.script_combo.insertSeparator(self.script_combo.count())
        self.script_combo.addItem("Load script file…", "__load__")
        self.script_combo.blockSignals(False)
        self._select_script(current)

    def _select_script(self, script_id: str) -> None:
        self.script_combo.blockSignals(True)
        index = self.script_combo.findData(script_id)
        if index < 0:
            index = 0
        self.script_combo.setCurrentIndex(index)
        self.script_combo.blockSignals(False)

    def _script_changed(self, index: int) -> None:
        if self._loading or index < 0:
            return
        script_id = self.script_combo.itemData(index)
        if script_id == "__load__":
            self._load_script_file()
            return
        self.session.begin_edit()
        self.session.set_script_id(str(script_id))
        self.session.end_edit()
        self.rebuild_script_list()
        self.rebuild_script_form()
        self.refresh_canvas()
        self.sync_title()

    def _script_row_changed(self, row: int) -> None:
        if self._loading or row < 0:
            return
        self.session.script_index = row
        self.rebuild_script_form()

    def _add_script(self) -> None:
        self.session.begin_edit()
        self.session.add_script("none")
        self.session.end_edit()
        self.sync_layer_controls()
        self.refresh_canvas()
        self.sync_title()

    def _remove_script(self) -> None:
        self.session.begin_edit()
        self.session.remove_script()
        self.session.end_edit()
        self.sync_layer_controls()
        self.refresh_canvas()
        self.sync_title()

    def _move_script(self, delta: int) -> None:
        self.session.begin_edit()
        self.session.move_script(delta)
        self.session.end_edit()
        self.sync_layer_controls()
        self.refresh_canvas()
        self.sync_title()

    def _load_script_file(self) -> None:
        start = self._last_dir()
        path, _selected = QFileDialog.getOpenFileName(self, "Load layer script", start, "Python (*.py)")
        current = self._current_script_id()
        if not path:
            self._select_script(current)
            return
        try:
            proc = self.session.library.load_file(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Script didn't load", str(exc))
            self._select_script(current)
            return
        self._remember_dir(path)
        self.session.begin_edit()
        self.session.set_script_id(proc.id, str(Path(path).resolve()))
        self.session.end_edit()
        self._fill_script_combo()
        self.sync_layer_controls()
        self.refresh_canvas()
        self.sync_title()

    def _reload_scripts(self) -> None:
        self.session.reload_scripts()
        self._fill_script_combo()
        self._rebuild_definition_menus()
        self.sync_layer_controls()
        self.refresh_canvas()
        self.session.message = "Reloaded scripts and definitions."
        self.update_status()

    def _param_widget(self, param: Param, value) -> QWidget:
        if param.type == "bool":
            box = QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(lambda checked, key=param.key: self._set_param(key, bool(checked)))
            self._watch_history(box)
            return box
        if param.type == "choice":
            box = QComboBox()
            for choice in param.choices or ():
                box.addItem(choice)
            box.setCurrentText(str(value if value is not None else ""))
            box.currentTextChanged.connect(lambda text, key=param.key: self._set_param(key, text))
            self._watch_history(box)
            return box
        if param.type == "int":
            box = QSpinBox()
            box.setRange(_as_int(param.min, -1_000_000), _as_int(param.max, 1_000_000))
            box.setSingleStep(_as_int(param.step, 1) or 1)
            box.setValue(_as_int(value, _as_int(param.default, 0)))
            box.valueChanged.connect(lambda number, key=param.key: self._set_param(key, int(number)))
            self._watch_history(box)
            return box
        if param.type == "color":
            return self._color_param(param, "" if value is None else str(value))
        if param.type == "colors":
            return self._colors_param(param, value)
        if param.type == "text":
            box = QLineEdit("" if value is None else str(value))
            box.editingFinished.connect(lambda key=param.key, edit=box: self._set_param(key, edit.text()))
            self._watch_history(box)
            return box
        box = QDoubleSpinBox()
        box.setDecimals(3)
        box.setRange(float(param.min if param.min is not None else -1_000_000), float(param.max if param.max is not None else 1_000_000))
        box.setSingleStep(float(param.step if param.step is not None else 0.1))
        try:
            box.setValue(float(value if value is not None else param.default or 0))
        except (TypeError, ValueError):
            box.setValue(float(param.default or 0))
        box.valueChanged.connect(lambda number, key=param.key: self._set_param(key, float(number)))
        self._watch_history(box)
        return box

    def _color_param(self, param: Param, value: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton(value or "None")
        clear = QPushButton("Clear")

        def choose() -> None:
            initial = QColor(255, 255, 255, 255)
            current = button.text()
            if current and current != "None":
                try:
                    initial = QColor(*parse_hex_color(current))
                except ValueError:
                    pass
            chosen = QColorDialog.getColor(initial, self, param.label, QColorDialog.ColorDialogOption.ShowAlphaChannel)
            if not chosen.isValid():
                return
            hex_color = format_hex_color((chosen.red(), chosen.green(), chosen.blue(), chosen.alpha()))
            self.session.begin_edit()
            self.session.set_script_param(param.key, hex_color)
            self.session.end_edit()
            button.setText(hex_color)
            self._after_param()

        def clear_color() -> None:
            self.session.begin_edit()
            self.session.set_script_param(param.key, "")
            self.session.end_edit()
            button.setText("None")
            self._after_param()

        button.clicked.connect(choose)
        clear.clicked.connect(clear_color)
        layout.addWidget(button, 1)
        layout.addWidget(clear)
        return row

    def _colors_param(self, param: Param, value, *, definition: bool = False) -> QWidget:
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        parsed = palette_colors(value if value not in (None, "") else param.default)
        if not parsed:
            parsed = palette_colors(param.default)
        host_colors = [format_hex_color(color) for color in parsed]

        def commit() -> None:
            text = " ".join(host_colors)
            self.session.begin_edit()
            if definition:
                self.session.set_definition_field(param.key, text)
            else:
                self.session.set_script_param(param.key, text)
            self.session.end_edit()
            if definition:
                self.refresh_preview()
                self.sync_title()
            else:
                self._after_param()

        def rebuild() -> None:
            while grid.count():
                item = grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            columns = 4
            for index, color in enumerate(host_colors):
                swatch = QPushButton()
                swatch.setFixedSize(28, 22)
                swatch.setToolTip(f"{color}\nClick to change.")
                swatch.setStyleSheet(f"background: {color}; border: 1px solid #2a2a2e;")
                swatch.clicked.connect(lambda _checked=False, i=index: edit(i))
                remove = QPushButton("×")
                remove.setFixedSize(22, 22)
                remove.setToolTip(f"Remove {color}")
                remove.setEnabled(len(host_colors) > 1)
                remove.clicked.connect(lambda _checked=False, i=index: drop(i))
                pair = QWidget()
                pair_layout = QHBoxLayout(pair)
                pair_layout.setContentsMargins(0, 0, 0, 0)
                pair_layout.setSpacing(2)
                pair_layout.addWidget(swatch)
                pair_layout.addWidget(remove)
                row, col = divmod(index, columns)
                grid.addWidget(pair, row, col)
            add = QPushButton("Add color")
            add.setToolTip("Add another color the noise can choose.")
            add.setEnabled(len(host_colors) < 24)
            add.clicked.connect(add_color)
            grid.addWidget(add, (len(host_colors) // columns) + 1, 0, 1, columns)

        def edit(index: int) -> None:
            try:
                initial = QColor(*parse_hex_color(host_colors[index]))
            except ValueError:
                initial = QColor(255, 255, 255, 255)
            chosen = QColorDialog.getColor(
                initial,
                self,
                param.label,
                QColorDialog.ColorDialogOption.ShowAlphaChannel,
            )
            if not chosen.isValid():
                return
            host_colors[index] = format_hex_color(
                (chosen.red(), chosen.green(), chosen.blue(), chosen.alpha())
            )
            commit()
            QTimer.singleShot(0, rebuild)

        def drop(index: int) -> None:
            if len(host_colors) <= 1:
                return
            del host_colors[index]
            commit()
            QTimer.singleShot(0, rebuild)

        def add_color() -> None:
            if len(host_colors) >= 24:
                return
            host_colors.append(host_colors[-1] if host_colors else "#808080")
            commit()
            QTimer.singleShot(0, rebuild)

        rebuild()
        return host

    def _toggle_script_part(self, part_id: str) -> None:
        if self._loading:
            return
        binding = self.session.current_binding()
        if binding is None:
            return
        chosen = [part for part in binding.parts if part != part_id]
        if part_id not in binding.parts:
            chosen.append(part_id)
        self.session.set_script_parts(chosen)
        self.rebuild_script_list()
        self.refresh_canvas()
        self.sync_title()

    def _set_param(self, key: str, value) -> None:
        if self._loading:
            return
        self.session.set_script_param(key, value)
        self._after_param()

    def _after_param(self) -> None:
        self.refresh_canvas()
        self.sync_title()

    def _randomize_seed(self) -> None:
        self.session.randomize_seed()
        self.rebuild_script_form()
        self.refresh_canvas()
        self.sync_title()

    def _watch_history(self, widget: QWidget) -> None:
        widget.installEventFilter(self)
        self._history_widgets.add(widget)

    # --- file ---------------------------------------------------------------

    def new_texture(self, kind: str = "texture") -> None:
        if not self._confirm_discard():
            return
        dialog = NewTextureDialog(self, kind=kind)
        if dialog.exec() != NewTextureDialog.DialogCode.Accepted or dialog.result_size is None:
            return
        width, height = dialog.result_size
        self._reset_preview_angle()
        self.session.new_document(
            width,
            height,
            dialog.result_name,
            dialog.result_fill,
            dialog.result_kind,
            stages=dialog.result_stages,
        )
        self.sync_all()
        self.canvas.fit()

    def new_definition(self, definition_id: str, asset_id: str) -> None:
        spec = self.session.definitions.get(definition_id)
        if spec is None:
            QMessageBox.warning(self, "New asset", f"Definition '{definition_id}' isn't loaded.")
            return
        try:
            asset = spec.asset(asset_id)
        except Exception as exc:
            QMessageBox.warning(self, "New asset", str(exc))
            return
        if not self._confirm_discard():
            return
        dialog = NewDefinitionDialog(spec.name, asset, self)
        if dialog.exec() != NewDefinitionDialog.DialogCode.Accepted or not dialog.result_name:
            return
        self._reset_preview_angle()
        try:
            self.session.new_document(
                asset.width,
                asset.height,
                dialog.result_name,
                None,
                "custom",
                definition=definition_id,
                asset=asset_id,
            )
        except Exception as exc:
            QMessageBox.warning(self, "New asset", str(exc))
            return
        self.sync_all()
        self.canvas.fit()

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _selected = QFileDialog.getOpenFileName(
            self, "Open project", self._last_dir(), "Voxel texture (*.vam)"
        )
        if not path:
            return
        self._open_path(Path(path))

    def open_image(self) -> None:
        if not self._confirm_discard():
            return
        path, _selected = QFileDialog.getOpenFileName(
            self, "Open image", self._last_dir(), "Images (*.png *.webp *.bmp *.gif)"
        )
        if not path:
            return
        self._open_path(Path(path))

    def import_layer(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "Import layer", self._last_dir(), "Images (*.png *.webp *.bmp *.gif)"
        )
        if not path:
            return
        try:
            self.session.import_layer(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        self._remember_dir(path)
        self._after_structure()

    def save_project(self) -> bool:
        if self.session.path is None:
            return self.save_project_as()
        try:
            self.session.save()
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False
        self.sync_title()
        self.update_status()
        return True

    def save_project_as(self) -> bool:
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save project",
            str(Path(self._last_dir()) / f"{self.session.doc.name}.vam"),
            "Voxel texture (*.vam)",
        )
        if not path:
            return False
        self._remember_dir(path)
        try:
            self.session.save(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False
        self.sync_title()
        self.update_status()
        return True

    def export_png(self) -> None:
        count = self.session.variant_count()
        if count > 1:
            default = f"{self.session.doc.name}_{self.session.variant_index}.png"
        else:
            default = f"{self.session.doc.name}.png"
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export PNG", str(Path(self._last_dir()) / default), "PNG (*.png)"
        )
        if not path:
            return
        self._remember_dir(path)
        try:
            errors = self.session.export_png(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._report_export(errors)

    def export_zip(self) -> None:
        default = f"{self.session.doc.name}_variants.zip"
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export variant ZIP", str(Path(self._last_dir()) / default), "Zip archive (*.zip)"
        )
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        self._remember_dir(path)
        try:
            _names, errors = self.session.export_zip(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._report_export(errors)

    def resize_canvas(self) -> None:
        if self.session.doc.kind == "block":
            self.session.message = (
                "Each block tile stays 16×16. Faces in the asset panel switches between "
                "six faces, one texture, and a column."
            )
            self.update_status()
            return
        if self.session.doc.kind == "crop":
            self.session.message = "A crop grows by its stage count. Change Stages in the Asset panel."
            self.update_status()
            return
        if self.session.doc.kind == "set":
            self.session.message = "A wood set keeps one tile for each texture. The pieces share those tiles."
            self.update_status()
            return
        if self.session.doc.kind == "custom":
            self.session.message = "This sheet size comes from the definition. The tiles stay where that file put them."
            self.update_status()
            return
        dialog = ResizeDialog(self.session.doc, self)
        if dialog.exec() != ResizeDialog.DialogCode.Accepted or dialog.result_size is None:
            return
        width, height = dialog.result_size
        self.session.resize(width, height, dialog.result_anchor)
        self.sync_all()
        self.canvas.fit()

    def rename_texture(self) -> None:
        name = ask_text(self, "Rename texture", "Texture name", self.session.doc.name)
        if not name:
            return
        self.session.set_name(name)
        self._sync_asset_id()
        self.refresh_preview()
        self.sync_title()

    def show_about(self) -> None:
        QMessageBox.information(self, "Voxel Asset Maker", ABOUT)

    def show_doc(self, filename: str, title: str) -> None:
        path = Path(__file__).resolve().parents[2] / "docs" / filename
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, title, str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(680, 560)
        dialog.exec()

    def undo(self) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, QLineEdit):
            focused.undo()
            return
        if isinstance(focused, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.session.undo():
            self.sync_all()

    def redo(self) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, QLineEdit):
            focused.redo()
            return
        if isinstance(focused, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.session.redo():
            self.sync_all()

    def copy(self) -> None:
        copied = self.session.copy_selection()
        QApplication.clipboard().setImage(numpy_to_qimage(copied))

    def cut(self) -> None:
        copied = self.session.cut_selection()
        if copied is not None:
            QApplication.clipboard().setImage(numpy_to_qimage(copied))
            self.refresh_canvas()
            self.sync_title()

    def paste(self) -> None:
        image = qimage_to_numpy(QApplication.clipboard().image())
        if image is None:
            self.session.paste_pixels(None)
        else:
            self.session.paste_pixels(image)
        self.refresh_canvas()
        self.sync_title()

    def _rotate(self) -> None:
        self.session.rotate_cw()
        self.refresh_canvas()
        self.sync_title()
        self.update_status()

    def _clear_layer(self) -> None:
        self.session.clear_layer()
        self.refresh_canvas()
        self.sync_title()

    def _select_all(self) -> None:
        self.session.select_all()
        self.refresh_canvas()

    def _deselect(self) -> None:
        self.session.deselect()
        self.refresh_canvas()

    def _delete_selection(self) -> None:
        self.session.delete_selection()
        self.refresh_canvas()
        self.sync_title()

    def _toggle_minor_grid(self) -> None:
        self.session.show_minor_grid = self.action_minor_grid.isChecked()
        self.refresh_canvas()

    def _toggle_major_grid(self) -> None:
        self.session.show_major_grid = self.action_major_grid.isChecked()
        self.refresh_canvas()

    # --- events -------------------------------------------------------------

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if not isinstance(watched, QWidget):
            return super().eventFilter(watched, event)
        watched_field = watched in self._history_widgets or watched in self._field_widgets
        if event.type() == QEvent.Type.FocusIn and watched_field and not self._loading:
            self.session.begin_edit()
        elif event.type() == QEvent.Type.FocusOut and watched_field and not self._loading:
            if self.session.end_edit():
                self.sync_title()
        elif event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if self._handle_plain_key(event):
                return True
        return super().eventFilter(watched, event)

    def _handle_plain_key(self, event: QKeyEvent) -> bool:
        if (
            QApplication.activeModalWidget() is not None
            or QApplication.activePopupWidget() is not None
            or self._typing()
        ):
            return False
        modifiers = event.modifiers()
        ctrl = bool(
            modifiers
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
        )
        key = event.key()
        arrows = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }
        if event.isAutoRepeat() and key not in arrows:
            return False
        if ctrl and key == Qt.Key.Key_A and not (
            modifiers & (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
        ):
            self._select_all()
            return True
        if ctrl:
            return False
        tools = {
            Qt.Key.Key_B: "pencil",
            Qt.Key.Key_E: "eraser",
            Qt.Key.Key_G: "fill",
            Qt.Key.Key_I: "picker",
            Qt.Key.Key_L: "line",
            Qt.Key.Key_R: "rect",
            Qt.Key.Key_M: "select",
        }
        if key in tools and not (modifiers & Qt.KeyboardModifier.ShiftModifier):
            self._set_tool(tools[key])
            return True
        if key == Qt.Key.Key_X:
            self._swap_colors()
            return True
        if key == Qt.Key.Key_BracketLeft:
            self.brush_size.setValue(max(1, self.brush_size.value() - 1))
            return True
        if key == Qt.Key.Key_BracketRight:
            self.brush_size.setValue(min(32, self.brush_size.value() + 1))
            return True
        if key in arrows and self.session.selection is not None:
            step = 16 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
            dx, dy = arrows[key]
            self.session.nudge(dx * step, dy * step)
            self.refresh_canvas()
            self.sync_title()
            return True
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.session.selection is not None:
            self._delete_selection()
            return True
        if key == Qt.Key.Key_Escape:
            self.session.deselect()
            self.refresh_canvas()
            return True
        return False

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if not self._handle_plain_key(event):
            super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if not paths:
            return
        path = paths[0]
        suffix = path.suffix.lower()
        if suffix == ".vam":
            if self._confirm_discard():
                self._open_path(path)
        elif suffix in {".png", ".webp", ".bmp", ".gif"}:
            try:
                self.session.import_layer(path)
            except Exception as exc:
                QMessageBox.warning(self, "Import failed", str(exc))
                return
            self._after_structure()
        elif suffix == ".py":
            try:
                proc = self.session.library.load_file(path)
            except Exception as exc:
                QMessageBox.warning(self, "Script didn't load", str(exc))
                return
            self.session.begin_edit()
            self.session.set_script_id(proc.id, str(path.resolve()))
            self.session.end_edit()
            self._fill_script_combo()
            self.sync_layer_controls()
            self.refresh_canvas()
            self.sync_title()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._initial_fit)

    def _initial_fit(self) -> None:
        if self._fitted:
            return
        if self.canvas.width() < 20 or self.canvas.height() < 20:
            self._fit_attempts += 1
            if self._fit_attempts < 8:
                QTimer.singleShot(30, self._initial_fit)
            return
        self._fitted = True
        self.canvas.fit()

    # --- helpers ------------------------------------------------------------

    def _confirm_discard(self) -> bool:
        if self.session.history.is_clean():
            return True
        name = self.session.path.name if self.session.path else self.session.doc.name
        choice = QMessageBox.question(
            self,
            "Unsaved changes",
            f"Save changes to {name}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            return self.save_project()
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        return True

    def _open_path(self, path: Path) -> None:
        try:
            if path.suffix.lower() == ".vam":
                self.session.open_project(path)
            else:
                self.session.open_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", str(exc))
            return
        self._remember_dir(str(path))
        self._reset_preview_angle()
        self.sync_all()
        self.canvas.fit()

    def _sync_asset_id(self) -> None:
        auto = sanitize_id(self.session.doc.name)
        current = self.asset_id.text().strip()
        if current == "" or current == self._asset_id_auto:
            self.asset_id.blockSignals(True)
            self.asset_id.setText(auto)
            self.asset_id.blockSignals(False)
        self._asset_id_auto = auto
        if self.session.doc.kind == "crop":
            self._refresh_loot_placeholders()

    def _asset_id_edited(self, _text: str) -> None:
        self._asset_id_auto = ""

    def _set_member_changed(self, _index: int) -> None:
        if self._loading:
            return
        self.refresh_preview()

    def _member_key(self) -> str:
        key = self.set_member.currentData()
        return str(key or "log")

    def _custom_asset(self):
        doc = self.session.doc
        if doc.kind != "custom":
            return None
        spec = self.session.definitions.get(doc.definition)
        if spec is None:
            return None
        try:
            return spec, spec.asset(doc.asset)
        except Exception:
            return None

    def _sync_definition_fields(self) -> None:
        for widget in list(self._field_widgets):
            self._field_widgets.discard(widget)
        while self.definition_form.rowCount():
            self.definition_form.removeRow(0)
        found = self._custom_asset()
        self.definition_host.setVisible(found is not None and bool(found[1].fields))
        if found is None:
            return
        spec, asset = found
        values = spec.resolved_fields(asset, self.session.doc.fields)
        self._loading = True
        for param in asset.fields:
            widget = self._field_widget(param, values.get(param.key))
            label = QLabel(param.label)
            if param.help:
                label.setToolTip(param.help)
                widget.setToolTip(param.help)
            self.definition_form.addRow(label, widget)
        self._loading = False

    def _field_widget(self, param, value) -> QWidget:
        if param.type == "bool":
            box = QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(lambda checked, key=param.key: self._set_field(key, bool(checked)))
            self._watch_field(box)
            return box
        if param.type == "choice":
            box = QComboBox()
            for choice in param.choices or ():
                box.addItem(choice)
            box.setCurrentText(str(value if value is not None else ""))
            box.currentTextChanged.connect(lambda text, key=param.key: self._set_field(key, text))
            self._watch_field(box)
            return box
        if param.type == "int":
            box = QSpinBox()
            box.setRange(_as_int(param.min, -1_000_000), _as_int(param.max, 1_000_000))
            box.setSingleStep(_as_int(param.step, 1) or 1)
            box.setValue(_as_int(value, _as_int(param.default, 0)))
            box.valueChanged.connect(lambda number, key=param.key: self._set_field(key, int(number)))
            self._watch_field(box)
            return box
        if param.type == "text":
            box = QLineEdit("" if value is None else str(value))
            box.editingFinished.connect(lambda key=param.key, edit=box: self._set_field(key, edit.text()))
            self._watch_field(box)
            return box
        if param.type == "colors":
            return self._colors_param(param, value, definition=True)
        if param.type == "color":
            button = QPushButton(str(value or "None"))

            def choose(key=param.key, target=button) -> None:
                initial = QColor(255, 255, 255)
                if target.text() and target.text() != "None":
                    try:
                        initial = QColor(*parse_hex_color(target.text()))
                    except ValueError:
                        pass
                chosen = QColorDialog.getColor(initial, self, param.label)
                if not chosen.isValid():
                    return
                text = f"#{chosen.red():02x}{chosen.green():02x}{chosen.blue():02x}"
                self.session.begin_edit()
                self.session.set_definition_field(key, text)
                self.session.end_edit()
                target.setText(text)
                self.refresh_preview()
                self.sync_title()

            button.clicked.connect(choose)
            return button
        box = QDoubleSpinBox()
        box.setDecimals(3)
        box.setRange(float(param.min if param.min is not None else -1_000_000), float(param.max if param.max is not None else 1_000_000))
        box.setSingleStep(float(param.step if param.step is not None else 0.1))
        try:
            box.setValue(float(value if value is not None else param.default or 0))
        except (TypeError, ValueError):
            box.setValue(0.0)
        box.valueChanged.connect(lambda number, key=param.key: self._set_field(key, float(number)))
        self._watch_field(box)
        return box

    def _watch_field(self, widget: QWidget) -> None:
        widget.installEventFilter(self)
        self._field_widgets.add(widget)

    def _set_field(self, key: str, value) -> None:
        if self._loading:
            return
        before = (self.session.doc.width, self.session.doc.height)
        self.session.set_definition_field(key, value)
        self.refresh_canvas()
        if (self.session.doc.width, self.session.doc.height) != before:
            self.canvas.fit()
        self.sync_title()

    def _sync_kind(self) -> None:
        kind = self.session.doc.kind
        custom = self._custom_asset()
        if kind == "set" and self.session.doc.family == "wood":
            self.asset_kind.setText("Wood set")
        elif custom is not None:
            self.asset_kind.setText(custom[0].name)
        else:
            self.asset_kind.setText(kind.replace("_", " ").capitalize())
        opening = kind in ("door", "trapdoor")
        turns = kind in ("block", "crop", "door", "trapdoor") or (
            kind == "set" and self._member_key() != "stick"
        ) or (custom is not None and custom[1].turn)
        self.preview_turn.setVisible(turns)
        self.set_member.setVisible(kind == "set")
        self.tab_host.setVisible(kind in ("block", "item", "door", "trapdoor"))
        self.block_extra.setVisible(kind == "block" or opening)
        self.crop_extra.setVisible(kind == "crop")
        self.export_definition_button.setEnabled(
            kind in ("block", "item", "crop", "door", "trapdoor", "set", "custom")
        )
        if kind in ("block", "item", "door", "trapdoor"):
            self._loading = True
            self.tab_combo.setEditText(self.session.doc.traits.tab)
            self._loading = False

    def _sync_block_extra(self) -> None:
        doc = self.session.doc
        block = doc.kind == "block"
        opening = doc.kind in ("door", "trapdoor")
        self.block_extra.setVisible(block or opening)
        if not block and not opening:
            return
        fluor = doc.fluor
        self._loading = True
        self.ctm_check.setChecked(doc.ctm)
        self.fluor_check.setChecked(fluor.enabled)
        mode_index = self.fluor_mode.findData(fluor.mode)
        if mode_index >= 0:
            self.fluor_mode.setCurrentIndex(mode_index)
        self.fluor_start.setValue(int(fluor.start))
        self.fluor_full.setValue(int(fluor.full))
        self.fluor_strength.setValue(float(fluor.strength))
        self.fluor_color.setText(fluor.color)
        self.fluor_color.setStyleSheet(f"background: {fluor.color}; color: #111;")
        self.fluor_options.setVisible(fluor.enabled)
        show_color = fluor.enabled and fluor.mode == "color"
        self.fluor_color.setVisible(show_color)
        color_label = self.fluor_options.layout().labelForField(self.fluor_color)
        if color_label is not None:
            color_label.setVisible(show_color)
        surface = self.session.surface if self.session.surface in self.surface_buttons else "faces"
        self.surface_buttons[surface].setChecked(True)
        traits = doc.traits
        self._select_data(self.shape_combo, traits.shape)
        self._select_data(self.face_mode, traits.faces)
        self._select_data(self.rotation_combo, traits.rotation)
        self._select_data(self.station_combo, traits.station)
        self.recipe_edit.setText(traits.recipe_types)
        self.cook_spin.setValue(float(traits.cook_speed))
        self._select_data(self.tool_combo, traits.tool)
        self.sound_combo.setEditText(traits.sound)
        self._select_data(self.render_combo, traits.render)
        self.hardness_spin.setValue(float(traits.hardness))
        self.mining_spin.setValue(int(traits.mining_level))
        for spin, level in zip(self.light_spins.values(), traits.light, strict=True):
            spin.setValue(int(level))
        self.solid_check.setChecked(bool(traits.solid))
        self.gravity_check.setChecked(bool(traits.gravity))
        self.full_block_edit.setText(traits.full_block)
        show_full = block and traits.shape == "vertical_slab"
        self._show_field(self._block_fields, self.shape_combo, block)
        self._show_field(self._block_fields, self.face_mode, block)
        self._show_field(self._block_fields, self.rotation_combo, block)
        self._show_field(self._block_fields, self.station_combo, block)
        self._show_field(self._block_fields, self.recipe_edit, block and bool(traits.station))
        self._show_field(self._block_fields, self.cook_spin, block and traits.station == "furnace")
        self._show_field(self._block_fields, self.light_host, block)
        self._show_field(self._block_fields, self.full_block_edit, show_full)
        self.gravity_check.setVisible(block)
        for button in self.surface_buttons.values():
            button.setVisible(block)
        self.ctm_check.setVisible(block)
        self.fluor_check.setVisible(block)
        self.fluor_options.setVisible(block and fluor.enabled)
        self._loading = False

    def _show_field(self, form: QFormLayout, widget: QWidget, visible: bool) -> None:
        widget.setVisible(visible)
        label = form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)

    def _select_data(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _surface_clicked(self, surface: str) -> None:
        if self._loading:
            return
        doc = self.session.doc
        if surface == "ctm" and not doc.ctm:
            self.session.set_ctm(True)
        elif surface == "glow" and not doc.fluor.enabled:
            doc.fluor.mode = "texture"
            self.session.set_fluor_enabled(True)
        elif surface == "glow" and doc.fluor.mode != "texture":
            self.session.set_fluor_mode("texture")
        else:
            self.session.set_surface(surface)
        self.sync_all()
        self.canvas.fit()

    def _ctm_toggled(self, checked: bool) -> None:
        if self._loading:
            return
        self.session.set_ctm(checked)
        self.sync_all()
        self.canvas.fit()

    def _fluor_toggled(self, checked: bool) -> None:
        if self._loading:
            return
        self.session.set_fluor_enabled(checked)
        self.sync_all()
        self.canvas.fit()

    def _fluor_mode_changed(self, _index: int) -> None:
        if self._loading:
            return
        self.session.set_fluor_mode(str(self.fluor_mode.currentData() or "texture"))
        self.sync_all()
        self.canvas.fit()

    def _fluor_range_changed(self, _value: int) -> None:
        if self._loading:
            return
        self.session.set_fluor_range(self.fluor_start.value(), self.fluor_full.value())
        self._sync_block_extra()

    def _fluor_strength_changed(self, value: float) -> None:
        if self._loading:
            return
        self.session.set_fluor_strength(value)

    def _shape_changed(self, _index: int) -> None:
        if self._loading:
            return
        shape = str(self.shape_combo.currentData() or "cube")
        changes: dict[str, object] = {"shape": shape}
        if shape == "cross":
            changes.update(render="cutout", solid=False, tool="none", hardness=0.0, tab="natural")
        self.session.set_block_trait(**changes)
        self._sync_block_extra()
        self.refresh_canvas()

    def _face_mode_changed(self, _index: int) -> None:
        if self._loading:
            return
        before = (self.session.doc.width, self.session.doc.height)
        self.session.set_block_trait(faces=str(self.face_mode.currentData() or "six"))
        self.refresh_canvas()
        if (self.session.doc.width, self.session.doc.height) != before:
            self.canvas.fit()
        self.sync_title()

    def _block_trait_changed(self, _value: object = None) -> None:
        if self._loading:
            return
        self.session.set_block_trait(
            hardness=float(self.hardness_spin.value()),
            tool=str(self.tool_combo.currentData() or "pickaxe"),
            mining_level=int(self.mining_spin.value()),
            sound=self.sound_combo.currentText(),
            render=str(self.render_combo.currentData() or "solid"),
            solid=self.solid_check.isChecked(),
            gravity=self.gravity_check.isChecked(),
            light=tuple(self.light_spins[key].value() for key in ("r", "g", "b", "uv", "ir")),
            rotation=str(self.rotation_combo.currentData() or ""),
            station=str(self.station_combo.currentData() or ""),
            recipe_types=self.recipe_edit.text(),
            cook_speed=float(self.cook_spin.value()),
        )
        self.refresh_preview()

    def _station_changed(self, _index: int) -> None:
        if self._loading:
            return
        station = str(self.station_combo.currentData() or "")
        changes: dict[str, object] = {"station": station}
        if station == "furnace" and not self.rotation_combo.currentData():
            changes["rotation"] = "horizontal"
        if not station:
            changes["recipe_types"] = ""
            changes["cook_speed"] = 1.0
        self.session.set_block_trait(**changes)
        self._sync_block_extra()
        self.refresh_preview()

    def _recipe_finished(self) -> None:
        if self._loading:
            return
        self._block_trait_changed()
        self._loading = True
        self.recipe_edit.setText(self.session.doc.traits.recipe_types)
        self._loading = False

    def _tab_edited(self, text: str) -> None:
        if self._loading:
            return
        self.session.set_block_trait(tab=text)

    def _tab_finished(self) -> None:
        if self._loading:
            return
        self.session.set_block_trait(tab=self.tab_combo.currentText())
        self._loading = True
        self.tab_combo.setEditText(self.session.doc.traits.tab)
        self._loading = False

    def _sync_crop(self) -> None:
        crop = self.session.doc.crop
        show = self.session.doc.kind == "crop"
        self.crop_extra.setVisible(show)
        if not show:
            return
        self._loading = True
        self.crop_stages.setValue(int(crop.stages))
        self.crop_view.setRange(1, int(crop.stages))
        self.crop_view.setValue(self.session.stage_index + 1)
        self.crop_min.setValue(float(crop.min_time))
        self.crop_max.setValue(float(crop.max_time))
        self.crop_water.setChecked(bool(crop.needs_water))
        self.crop_radius.setValue(int(crop.water_radius))
        self.crop_vertical.setValue(int(crop.water_vertical))
        self.crop_seeds.setText(crop.seeds)
        self.crop_produce.setText(crop.produce)
        self.crop_ground.setText(crop.ground)
        self._rebuild_loot()
        self.crop_seed_tab.setEditText(crop.seed_tab)
        self.crop_produce_tab.setEditText(crop.produce_tab)
        self._loading = False

    def _crop_stages_changed(self, value: int) -> None:
        if self._loading:
            return
        self.session.set_crop(stages=int(value))
        self._sync_crop()
        self.refresh_canvas()
        self.canvas.fit()

    def _crop_view_changed(self, value: int) -> None:
        if self._loading:
            return
        self.session.stage_index = max(0, int(value) - 1)
        self.refresh_preview()

    def _crop_changed(self, _value: object = None) -> None:
        if self._loading:
            return
        self.session.set_crop(
            min_time=float(self.crop_min.value()),
            max_time=float(self.crop_max.value()),
            needs_water=self.crop_water.isChecked(),
            water_radius=int(self.crop_radius.value()),
            water_vertical=int(self.crop_vertical.value()),
            seeds=self.crop_seeds.text(),
            produce=self.crop_produce.text(),
            loot=self._read_loot(),
            ground=self.crop_ground.text(),
            seed_tab=self.crop_seed_tab.currentText(),
            produce_tab=self.crop_produce_tab.currentText(),
        )
        self._refresh_loot_placeholders()
        self.refresh_preview()

    def _loot_placeholders(self) -> tuple[str, str]:
        produce = self.crop_produce.text().strip() or self.asset_id.text().strip() or "produce"
        seeds = self.crop_seeds.text().strip() or f"{produce}_seeds"
        return produce, seeds

    def _refresh_loot_placeholders(self) -> None:
        produce, seeds = self._loot_placeholders()
        for index, row in enumerate(self.loot_rows):
            hint = produce if index == 0 else seeds if index == 1 else "item_id"
            row.item.setPlaceholderText(hint)

    def _rebuild_loot(self) -> None:
        for row in self.loot_rows:
            for widget in row.watched:
                self._history_widgets.discard(widget)
            row.setParent(None)
            row.deleteLater()
        self.loot_rows = []
        produce, seeds = self._loot_placeholders()
        for index, drop in enumerate(self.session.doc.crop.loot):
            hint = produce if index == 0 else seeds if index == 1 else "item_id"
            row = self._make_loot_row(index, drop, hint)
            self.crop_loot_rows.addWidget(row)
            self.loot_rows.append(row)

    def _make_loot_row(self, index: int, drop: LootDrop, hint: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        item = QLineEdit(drop.item)
        item.setPlaceholderText(hint)
        item.setToolTip("Item id. Leave the first row blank to drop the produce, and the second to drop the seeds.")
        minimum = QSpinBox()
        minimum.setRange(1, 64)
        minimum.setValue(int(drop.minimum))
        minimum.setToolTip("Fewest items in this drop.")
        maximum = QSpinBox()
        maximum.setRange(1, 64)
        maximum.setValue(int(drop.maximum))
        maximum.setToolTip("Most items in this drop.")
        chance = QSpinBox()
        chance.setRange(0, 100)
        chance.setSuffix("%")
        chance.setValue(int(round(float(drop.chance) * 100)))
        chance.setToolTip("Chance this drop is rolled. 100% always drops.")
        remove = QPushButton("×")
        remove.setFixedWidth(28)
        remove.setToolTip("Remove this drop.")
        for widget in (item, minimum, maximum, chance):
            widget.setMinimumWidth(0)
        minimum.setMaximumWidth(54)
        maximum.setMaximumWidth(54)
        chance.setMaximumWidth(68)
        layout.addWidget(item, 1)
        layout.addWidget(minimum)
        layout.addWidget(maximum)
        layout.addWidget(chance)
        layout.addWidget(remove)
        row.item = item
        row.minimum = minimum
        row.maximum = maximum
        row.chance = chance
        row.watched = (item, minimum, maximum, chance, remove)
        for widget in row.watched:
            self._watch_history(widget)
        item.textEdited.connect(self._crop_changed)
        item.editingFinished.connect(self._crop_finished)
        minimum.valueChanged.connect(self._crop_changed)
        maximum.valueChanged.connect(self._crop_changed)
        chance.valueChanged.connect(self._crop_changed)
        remove.clicked.connect(lambda _checked=False, row_index=index: self._remove_loot_row(row_index))
        return row

    def _read_loot(self) -> list[LootDrop]:
        drops: list[LootDrop] = []
        for row in self.loot_rows:
            drops.append(
                LootDrop(
                    row.item.text(),
                    int(row.minimum.value()),
                    int(row.maximum.value()),
                    int(row.chance.value()) / 100,
                )
            )
        return drops

    def _add_loot_row(self) -> None:
        if self._loading:
            return
        drops = self._read_loot()
        drops.append(LootDrop())
        self.session.set_crop(loot=drops)
        self._loading = True
        self._rebuild_loot()
        self._loading = False

    def _remove_loot_row(self, index: int) -> None:
        if self._loading:
            return
        drops = self._read_loot()
        if 0 <= index < len(drops):
            del drops[index]
        self.session.set_crop(loot=tuple(drops))
        self._loading = True
        self._rebuild_loot()
        self._loading = False

    def _crop_finished(self) -> None:
        if self._loading:
            return
        self._crop_changed()
        self._sync_crop()

    def _sound_finished(self) -> None:
        if self._loading:
            return
        self.session.set_block_trait(sound=self.sound_combo.currentText())
        self._loading = True
        self.sound_combo.setEditText(self.session.doc.traits.sound)
        self._loading = False

    def _full_block_edited(self, text: str) -> None:
        if self._loading:
            return
        self.session.set_block_trait(full_block=text)

    def _full_block_finished(self) -> None:
        if self._loading:
            return
        self.session.set_block_trait(full_block=self.full_block_edit.text())
        self._loading = True
        self.full_block_edit.setText(self.session.doc.traits.full_block)
        self._loading = False

    def _pick_fluor_color(self) -> None:
        current = parse_hex_color(self.session.doc.fluor.color)
        chosen = QColorDialog.getColor(QColor(*current), self, "Fluorescence")
        if not chosen.isValid():
            return
        self.session.begin_edit()
        self.session.set_fluor_color(f"#{chosen.red():02x}{chosen.green():02x}{chosen.blue():02x}")
        self.session.end_edit()
        self._sync_block_extra()
        self.sync_title()

    def _turn_preview(self, step: int) -> None:
        self.preview_yaw = (self.preview_yaw + step) % 4
        self.refresh_preview()

    def _flip_preview(self, checked: bool) -> None:
        self.preview_pitch = 1 if checked else 0
        self.refresh_preview()

    def _reset_preview_angle(self) -> None:
        self.preview_yaw = 0
        self.preview_pitch = 0
        self.preview_flip.blockSignals(True)
        self.preview_flip.setChecked(False)
        self.preview_flip.blockSignals(False)

    def refresh_preview(self) -> None:
        kind = self.session.doc.kind
        member = self._member_key()
        if kind not in ("block", "item", "crop", "door", "trapdoor", "set", "custom"):
            self.asset_preview.clear()
            self.asset_preview.setText("Texture")
            self.asset_caption.setText(
                "Texture. Start a block, an item, a crop, a door, or a trapdoor from "
                "File → New → Block game → Assets, a wood set from Sets, or another game from its own menu."
            )
            return
        try:
            if kind == "crop":
                rendered = render_variant(self.session.doc, self.session.variant_index, self.session.library, self.session.definitions)
                preview = preview_for(
                    rendered.image,
                    kind,
                    yaw=self.preview_yaw,
                    pitch=self.preview_pitch,
                    stage=self.session.stage_index,
                )
            elif kind == "custom":
                rendered = render_variant(self.session.doc, self.session.variant_index, self.session.library, self.session.definitions)
                spec = self.session.definitions.get(self.session.doc.definition)
                if spec is None:
                    raise ValueError(
                        f"Definition '{self.session.doc.definition}' isn't loaded. "
                        "Put its file in user_scripts/definitions/ and choose Reload."
                    )
                asset = spec.asset(self.session.doc.asset)
                from ..definitions import DefinitionContext

                fields = spec.resolved_fields(asset, self.session.doc.fields)
                preview = spec.preview(
                    rendered.image,
                    asset,
                    fields,
                    DefinitionContext(
                        asset_id=sanitize_id(self.asset_id.text()),
                        name=self.session.doc.name,
                        variant_index=self.session.variant_index,
                        variant_count=self.session.variant_count(),
                        yaw=self.preview_yaw,
                        pitch=self.preview_pitch,
                    ),
                )
            elif kind == "set":
                rendered = render_variant(self.session.doc, self.session.variant_index, self.session.library, self.session.definitions)
                preview = preview_for(
                    rendered.image,
                    kind,
                    yaw=self.preview_yaw,
                    pitch=self.preview_pitch,
                    member=member,
                )
            elif kind in ("block", "door", "trapdoor"):
                rendered = render_variant(self.session.doc, self.session.variant_index, self.session.library, self.session.definitions)
                fluor = self.session.doc.fluor
                glowing = fluor.enabled and fluor.strength > 0
                preview = preview_for(
                    rendered.image,
                    kind,
                    yaw=self.preview_yaw,
                    pitch=self.preview_pitch,
                    ctm=self.session.doc.ctm_pixels if self.session.doc.ctm else None,
                    glow=fluor.pixels if glowing and fluor.mode == "texture" else None,
                    glow_color=fluor.color if glowing and fluor.mode == "color" else None,
                    glow_strength=fluor.strength if glowing else 0.0,
                    shape=self.session.doc.traits.shape,
                    faces=self.session.doc.traits.faces,
                )
            elif self.session.view_mode == "composite":
                image = self.session.display_array()
                preview = preview_for(image, kind, yaw=self.preview_yaw, pitch=self.preview_pitch)
            else:
                image = render_variant(
                    self.session.doc,
                    self.session.variant_index,
                    self.session.library,
                    self.session.definitions,
                ).image
                preview = preview_for(image, kind, yaw=self.preview_yaw, pitch=self.preview_pitch)
        except Exception as exc:
            self.asset_preview.setText(str(exc))
            self.asset_caption.setText("")
            return
        pixmap = QPixmap.fromImage(numpy_to_qimage(preview))
        limit = max(96, self.asset_preview.width() - 12)
        if pixmap.width() > limit:
            smooth = kind in ("block", "crop", "door", "trapdoor") or (kind == "set" and member != "stick") or (
                kind == "custom" and bool(self._custom_asset() and self._custom_asset()[1].turn)
            )
            mode = (
                Qt.TransformationMode.SmoothTransformation
                if smooth
                else Qt.TransformationMode.FastTransformation
            )
            pixmap = pixmap.scaledToWidth(limit, mode)
        self.asset_preview.setPixmap(pixmap)
        asset_id = sanitize_id(self.asset_id.text())
        count = self.session.variant_count()
        if kind == "crop":
            crop = self.session.doc.crop
            stage = self.session.stage_index + 1
            if self.crop_view.value() != stage:
                self._loading = True
                self.crop_view.setValue(stage)
                self._loading = False
            soil = crop.ground.replace(" ", ", ")
            seeds = crop.seeds or f"{asset_id}_seeds"
            produce = crop.produce or asset_id
            caption = (
                f"Crop {asset_id}. Stage {stage} of {crop.stages}, drawn as a plant. "
                f"Roots in {soil}. Seeds {seeds} on the {crop.seed_tab} tab, "
                f"produce {produce} on the {crop.produce_tab} tab."
            )
        elif kind == "door":
            caption = (
                f"Door {asset_id}. Lower, upper, and the thin edge. "
                "Export writes the block, the door file, the item, and loot that drops the door."
            )
        elif kind == "trapdoor":
            caption = (
                f"Trapdoor {asset_id}. The panel and the thin edge. "
                "Export writes the block, the trapdoor file, the item, and loot that drops it."
            )
        elif kind == "block":
            cap, left, right = block_view(self.preview_yaw, self.preview_pitch)
            palettes = f" Export includes {count} palettes." if count > 1 else ""
            where = "the underside" if cap == "bottom" else "the top"
            shape = self.session.doc.traits.shape
            label = SHAPE_LABELS.get(shape, "Cube").lower()
            extras = ""
            if shape == "fence":
                extras += " The preview is the fence you hold, with posts and rails."
            elif shape == "wall":
                extras += " The preview is the wall you hold, with a post and a run."
            elif shape == "cross":
                extras += " The south face is the plant, drawn as two crossed planes."
            if self.session.doc.ctm and shape == "cube":
                extras += " Four blocks share edges, so each face shows the connected tile."
            elif self.session.doc.ctm:
                extras += " Connected textures are in the export. The preview shows the shape."
            if self.session.doc.fluor.enabled:
                fluor = self.session.doc.fluor
                extras += f" Fluorescence is added at full UV, from {fluor.start} to {fluor.full}."
            traits = self.session.doc.traits
            red, green, blue, ultraviolet, infrared = traits.light
            if any(traits.light):
                extras += (
                    f" Light is red {red}, green {green}, blue {blue}, "
                    f"ultraviolet {ultraviolet}, infrared {infrared}."
                )
            if traits.rotation == "axis":
                extras += " It rotates like a log."
            elif traits.rotation == "horizontal":
                extras += " It faces the way you place it."
            if traits.station == "furnace":
                recipes = traits.recipe_types.replace(" ", ", ") or "smelting"
                extras += f" Furnace recipes: {recipes}."
            elif traits.station == "crafting_table":
                recipes = traits.recipe_types.replace(" ", ", ") or "the usual crafting recipes"
                extras += f" Crafting recipes: {recipes}."
            caption = (
                f"Block {asset_id}, a {label}.{palettes} "
                f"Looking at {where}, {left} on the left and {right} on the right.{extras}"
            )
        elif kind == "custom":
            found = self._custom_asset()
            if found is None:
                caption = (
                    f"Custom file '{self.session.doc.definition}'. "
                    "That definition isn't loaded. Put its .py file in user_scripts/definitions/ and choose Reload."
                )
            else:
                spec, asset = found
                caption = f"{spec.name} {asset.label}, id {asset_id}. {asset.blurb}".strip()
        elif kind == "set":
            piece = wood_member_id(asset_id, member)
            label = self.set_member.currentText()
            self.preview_turn.setVisible(member != "stick")
            caption = (
                f"Wood set {asset_id}. {label} is {piece}. "
                "Export writes the log, wood, stick, planks, both slabs, stairs, "
                "the vertical plank versions, the crafting table, the door, and the trapdoor."
            )
        else:
            which = f" Variant {self.session.variant_index} of {count}." if count > 1 else ""
            caption = f"Item {asset_id}.{which} Export writes this icon and an item definition."
        self.asset_caption.setText(caption)

    def export_asset_definition(self) -> None:
        kind = self.session.doc.kind
        if kind not in ("block", "item", "crop", "door", "trapdoor", "set", "custom"):
            QMessageBox.information(
                self,
                "Export definition",
                "This file is a texture. Start a block-game asset, a wood set, or a custom definition to export one.",
            )
            return
        asset_id = sanitize_id(self.asset_id.text())
        default = f"{asset_id}.zip"
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Export definition",
            str(Path(self._last_dir()) / default),
            "Zip archive (*.zip)",
        )
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        self._remember_dir(path)
        try:
            if kind == "custom":
                exported = export_custom(
                    self.session.doc,
                    self.session.definitions,
                    self.session.library,
                    Path(path),
                    asset_id,
                    self.session.variant_index,
                )
            else:
                exported = export_definition(
                    self.session.doc,
                    self.session.library,
                    Path(path),
                    asset_id,
                    self.session.variant_index,
                )
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.session.message = f"Exported {exported.asset_id} ({len(exported.names)} files)"
        self.update_status()
        if exported.errors:
            QMessageBox.warning(self, "Export finished with script errors", "\n".join(exported.errors))

    def _report_export(self, errors: list[str]) -> None:
        self.update_status()
        self.update_errors()
        if errors:
            QMessageBox.warning(self, "Export finished with script errors", "\n".join(errors))

    def _typing(self) -> bool:
        focused = QApplication.focusWidget()
        return isinstance(focused, (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox))

    def _last_dir(self) -> str:
        stored = self.settings.value("lastDir", "")
        if isinstance(stored, str) and stored:
            return stored
        return str(Path.cwd())

    def _remember_dir(self, path: str) -> None:
        folder = str(Path(path).resolve().parent)
        self.settings.setValue("lastDir", folder)


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
