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
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
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
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..model import BLEND_MODES, format_hex_color, parse_hex_color
from ..processors import Param
from ..session import Session
from .canvas import CanvasWidget
from .dialogs import NewTextureDialog, ResizeDialog, ask_text
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

Layers stack from bottom to top. A script on a layer runs on that layer's pixels before the stack is flattened. Result view shows the flattened texture for the variant on the slider. Layer pixels view shows the active layer before its script, which is the bitmap you paint.

Noise is texture_noise. Adjust color is texture_adjust. Edge shade and Brightness steps ship as extra scripts. Drop another .py file in user_scripts/ next to the app, or in ~/.voxel-asset-maker/scripts/, then choose Reload scripts.

The variant slider and box under the toolbar scrub those variants in Result view. They stay on 1 until a layer's Count is above 1. Noise starts at 4. Export PNG writes the variant you're viewing. Export Variant ZIP writes one composited PNG per variant, plus manifest.json. The file count is the highest Count on any layer. A Noise layer with Count 8 makes eight textures, each with its own grain, and the other layers composite on top of every one. A script with a shorter count repeats.

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
        self._history_widgets: set[QWidget] = set()
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

        self._add_action(file_menu, "New…", self.new_texture, QKeySequence.StandardKey.New)
        self._add_action(file_menu, "Open project…", self.open_project, QKeySequence.StandardKey.Open)
        self._add_action(file_menu, "Open image…", self.open_image)
        self._add_action(file_menu, "Save", self.save_project, QKeySequence.StandardKey.Save)
        self._add_action(file_menu, "Save as…", self.save_project_as, QKeySequence.StandardKey.SaveAs)
        file_menu.addSeparator()
        self._add_action(file_menu, "Import layer…", self.import_layer)
        self._add_action(file_menu, "Export PNG…", self.export_png, "Ctrl+E")
        self._add_action(file_menu, "Export variant ZIP…", self.export_zip, "Ctrl+Shift+E")
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

    def _build_toolbar(self) -> None:
        bar = QToolBar()
        bar.setMovable(False)
        self.addToolBar(bar)
        self.view_result = QPushButton("Result")
        self.view_result.setCheckable(True)
        self.view_result.setChecked(True)
        self.view_result.setToolTip("Flattened texture with each layer's script applied.")
        self.view_layer = QPushButton("Layer pixels")
        self.view_layer.setCheckable(True)
        self.view_layer.setToolTip("The active layer before its script runs. This is what you paint.")
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
        self.variant_slider.setToolTip("Scrub variants. A layer script's Count sets how many there are.")
        self.variant_label = QLabel("of 1 — set Count above 1 on the layer script")
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

        script = QGroupBox("Layer script")
        script_layout = QVBoxLayout(script)
        script_row = QHBoxLayout()
        self.script_combo = QComboBox()
        self.reload_scripts_button = QPushButton("Reload")
        self.reload_scripts_button.setToolTip("Rescan user_scripts and ~/.voxel-asset-maker/scripts.")
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
        self.script_scroll.setWidget(self.form_host)
        script_layout.addWidget(self.script_scroll, 1)
        self.error_label = QLabel("")
        self.error_label.setObjectName("error")
        self.error_label.setWordWrap(True)
        script_layout.addWidget(self.error_label)
        layout.addWidget(script, 1)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(inner)
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(380)
        return panel

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
        self.script_combo.currentIndexChanged.connect(self._script_changed)
        self.reload_scripts_button.clicked.connect(self._reload_scripts)
        for widget in (self.opacity_slider, self.blend_combo, self.offset_x, self.offset_y):
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
        self.offset_x.setValue(int(layer.offset[0]))
        self.offset_y.setValue(int(layer.offset[1]))
        self._select_script(layer.script.script_id)
        self._loading = False
        self.rebuild_script_form()

    def rebuild_script_form(self) -> None:
        self._loading = True
        self._history_widgets.clear()
        for widget in (self.opacity_slider, self.blend_combo, self.offset_x, self.offset_y):
            self._history_widgets.add(widget)
        while self.form.rowCount():
            self.form.removeRow(0)
        layer = self.session.doc.active_layer
        proc = self.session.library.get(layer.script.script_id)
        if layer.script.script_id != "none" and not self.session.library.known(layer.script.script_id):
            missing = QLabel(f"Missing script {layer.script.script_id}.")
            missing.setObjectName("error")
            missing.setWordWrap(True)
            self.form.addRow(missing)
        self.script_blurb.setText(proc.blurb)
        params = {param.key: param.default for param in proc.params}
        params.update({key: value for key, value in layer.script.params.items() if key in params})
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
        self._loading = False

    def refresh_canvas(self) -> None:
        self.canvas.set_pixels(self.session.display_array())
        self.canvas.set_selection(self.session.selection, self.session.selection_offset())
        self.canvas.show_minor_grid = self.session.show_minor_grid
        self.canvas.show_major_grid = self.session.show_major_grid
        self.update_variant_controls()
        self.update_errors()
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
            text = "of 1 — set Count above 1 on the layer script"
        else:
            text = f"of {count}"
            if sources:
                text += "  (" + ", ".join(f"{name} ×{number}" for name, number in sources) + ")"
        self.variant_label.setText(text)

    def update_errors(self) -> None:
        lines = list(self.session.script_errors)
        lines.extend(self.session.library.errors)
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
        self.status_left.setText(
            f"{doc.width}×{doc.height}    {self.canvas.zoom}×    {position}{sampled}    {self.session.tool}"
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
        self.refresh_canvas()
        self.sync_title()

    def _pointer_move(self, x: int, y: int) -> None:
        if self.session.pointer_move(x, y):
            self.refresh_canvas()

    def _pointer_up(self, x: int, y: int) -> None:
        self.session.pointer_up(x, y)
        self.refresh_canvas()
        self.sync_title()

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
        if layer.script.script_id != "none":
            answer = QMessageBox.question(
                self,
                "Merge layer",
                "Merge blends the painted pixels onto the layer below. "
                "The script on this layer is dropped and stays unbaked.",
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

    def _fill_script_combo(self) -> None:
        current = self.session.doc.active_layer.script.script_id
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
        self.rebuild_script_form()
        self.refresh_canvas()
        self.sync_title()

    def _load_script_file(self) -> None:
        start = self._last_dir()
        path, _selected = QFileDialog.getOpenFileName(self, "Load layer script", start, "Python (*.py)")
        current = self.session.doc.active_layer.script.script_id
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
        self.sync_layer_controls()
        self.refresh_canvas()
        self.session.message = "Reloaded scripts."
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

    def new_texture(self) -> None:
        if not self._confirm_discard():
            return
        dialog = NewTextureDialog(self)
        if dialog.exec() != NewTextureDialog.DialogCode.Accepted or dialog.result_size is None:
            return
        width, height = dialog.result_size
        self.session.new_document(width, height, dialog.result_name, dialog.result_fill)
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
        self.sync_title()

    def show_about(self) -> None:
        QMessageBox.information(self, "Voxel Asset Maker", ABOUT)

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
        if event.type() == QEvent.Type.FocusIn and watched in self._history_widgets and not self._loading:
            self.session.begin_edit()
        elif event.type() == QEvent.Type.FocusOut and watched in self._history_widgets and not self._loading:
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
        self.sync_all()
        self.canvas.fit()

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
