import numpy as np
import pyqtgraph as qtg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import time
from matplotlib.axes._base import _process_plot_format
from matplotlib.colors import is_color_like, to_rgba
from matplotlib.markers import MarkerStyle


class _AxisLockedViewBox(qtg.ViewBox):
    """ViewBox with modifier-key-constrained scroll zoom.
    Ctrl+scroll  -> x-axis only
    Shift+scroll -> y-axis only
    plain scroll -> both axes (default)
    """

    def wheelEvent(self, ev, axis=None):
        mods = ev.modifiers()
        if mods & QtCore.Qt.ControlModifier:
            axis = 0
        elif mods & QtCore.Qt.ShiftModifier:
            axis = 1
        super().wheelEvent(ev, axis=axis)


class _PlotterKeyEventFilter(QtCore.QObject):
    """Global key filter for plotter-window keyboard navigation."""

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and OnlinePlotter.handle_keyboard_event(event):
            return True
        return super().eventFilter(obj, event)


class _SequenceSeriesItem:
    def __init__(self, series, name="", units=""):
        self._series = series
        self.name = name
        self.units = units

    @property
    def v(self):
        if len(self._series) == 0:
            return float("nan")
        return self._series[-1]


class _FillLegendSample(qtg.ItemSample):
    """The default legend sample expects line options absent on FillBetweenItem."""

    def paint(self, painter, *args):
        painter.setPen(qtg.mkPen(None))
        brush = self.item.brush() if self.item.isVisible() else qtg.mkBrush('#777777')
        painter.setBrush(brush)
        painter.drawRect(QtCore.QRectF(2, 5, 16, 10))


class OnlinePlotter:
    app = None
    main_window = None
    tab_widget = None
    follow_button = None
    tooltip_button = None
    instances = []
    font_targets = []
    tab_shortcuts = []
    key_event_filter = None
    KEYBOARD_ZOOM_IN_FACTOR = 1.15
    KEYBOARD_ZOOM_OUT_FACTOR = 1.0 / KEYBOARD_ZOOM_IN_FACTOR
    KEYBOARD_PAN_PIXELS = 80
    n_plotters = 0
    font_size_pt = 10
    min_font_size_pt = 6
    max_font_size_pt = 30
    UI_EVENT_PROCESS_INTERVAL_SEC = 0.02
    UI_EVENT_PROCESS_BUDGET_MS = 3
    _last_ui_event_process = 0.0
    # Fixed durations in seconds; a plotting year is 365 days.
    _SECONDS_PER_TIME_UNIT = {
        'nsec': 1e-9,
        'msec': 1e-3,
        'sec': 1.0,
        'min': 60.0,
        'hr': 3600.0,
        'days': 86400.0,
        'weeks': 604800.0,
        'years': 31536000.0,
    }

    @classmethod
    def build_window(cls, plotter_size=(0.9, 0.9)):
        if cls.app is None:
            cls.app = qtg.mkQApp()

        if cls.main_window is not None:
            return cls.main_window

        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle("ᴇꜱᴄʟᴀʙ ☀")
        tab_widget = QtWidgets.QTabWidget()

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(6, 6, 6, 0)
        controls_layout.setSpacing(6)

        font_up_button = QtWidgets.QPushButton("🗚")
        font_down_button = QtWidgets.QPushButton("🗛")
        font_up_button.setFixedSize(30, 30)
        font_down_button.setFixedSize(30, 30)
        font_up_button.setToolTip("Increase font size")
        font_down_button.setToolTip("Decrease font size")
        controls_layout.addWidget(font_up_button)
        controls_layout.addWidget(font_down_button)
        follow_button = QtWidgets.QPushButton("Follow")
        follow_button.setCheckable(True)
        follow_button.setChecked(True)
        follow_button.setFixedSize(80, 30)
        follow_button.setToolTip("Auto-follow the latest data in the plot (data plotter only)")
        controls_layout.addWidget(follow_button)
        tooltip_button = QtWidgets.QPushButton("Tooltip")
        tooltip_button.setCheckable(True)
        tooltip_button.setChecked(False)
        tooltip_button.setFixedSize(70, 30)
        tooltip_button.setToolTip("Show tooltip with values at the cursor position (data plotter only)")
        controls_layout.addWidget(tooltip_button)

        fit_all_button = QtWidgets.QPushButton("Fit All")
        fit_all_button.setFixedSize(70, 30)
        fit_all_button.setToolTip("Zoom to fit the entire data history (data plotter only)")
        controls_layout.addWidget(fit_all_button)
        
        save_png_button = QtWidgets.QPushButton("Save Image")
        save_png_button.setFixedSize(80, 30)
        save_png_button.setToolTip("Save the current plot as a PNG image")
        controls_layout.addWidget(save_png_button)
        
        controls_layout.addStretch()

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(controls_widget)
        container_layout.addWidget(tab_widget)
        main_window.setCentralWidget(container)

        font_down_button.clicked.connect(lambda: cls.adjust_font_size(-1))
        font_up_button.clicked.connect(lambda: cls.adjust_font_size(1))
        follow_button.clicked.connect(lambda checked: cls.set_auto_follow(checked))
        tooltip_button.clicked.connect(lambda checked: cls.set_plot_tooltip(checked))
        fit_all_button.clicked.connect(lambda: cls.fit_current_tab_to_data())
        save_png_button.clicked.connect(lambda: cls.save_current_tab_png())
        next_tab_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Tab"), main_window)
        prev_tab_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Tab"), main_window)
        next_tab_shortcut.setContext(QtCore.Qt.WindowShortcut)
        prev_tab_shortcut.setContext(QtCore.Qt.WindowShortcut)
        next_tab_shortcut.activated.connect(lambda: cls.cycle_tabs(forward=True))
        prev_tab_shortcut.activated.connect(lambda: cls.cycle_tabs(forward=False))

        all_shortcuts = [next_tab_shortcut, prev_tab_shortcut]
        for shortcut in all_shortcuts:
            shortcut.setContext(QtCore.Qt.WindowShortcut)

        cls.tab_shortcuts = all_shortcuts
        cls.follow_button = follow_button
        cls.tooltip_button = tooltip_button

        if cls.key_event_filter is None:
            cls.key_event_filter = _PlotterKeyEventFilter()
            cls.app.installEventFilter(cls.key_event_filter)

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            screen_rect = screen.availableGeometry()
            if plotter_size is not None:
                if plotter_size[0] <= 1.0 and plotter_size[1] <= 1.0:
                    main_window.resize(int(screen_rect.width() * plotter_size[0]), int(screen_rect.height() * plotter_size[1]))
                else:
                    main_window.resize(plotter_size[0], plotter_size[1])
            frame = main_window.frameGeometry()
            frame.moveCenter(screen_rect.center())
            main_window.move(frame.topLeft())
        elif plotter_size is not None:
            if plotter_size[0] <= 1.0 and plotter_size[1] <= 1.0:
                main_window.resize(1000, 600)
            else:
                main_window.resize(plotter_size[0], plotter_size[1])

        main_window.show()
        cls.main_window = main_window
        cls.tab_widget = tab_widget
        return main_window

    @classmethod
    def cycle_tabs(cls, forward=True):
        if cls.tab_widget is None:
            return

        count = cls.tab_widget.count()
        if count <= 1:
            return

        current = cls.tab_widget.currentIndex()
        if current < 0:
            current = 0

        step = 1 if forward else -1
        cls.tab_widget.setCurrentIndex((current + step) % count)

    @classmethod
    def save_current_tab_png(cls, dpi=300):
        widget = cls._current_tab_widget()
        if widget is None:
            return
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(cls.main_window, "Save Image", "", "PNG Files (*.png)")
        if not path:
            return
            
        if not path.lower().endswith('.png'):
            path += '.png'
            
        screen_dpi = getattr(cls.main_window, 'logicalDpiX', lambda: 96.0)()
        scale_factor = dpi / screen_dpi

        orig_size = widget.size()
        scaled_width = int(orig_size.width() * scale_factor)
        scaled_height = int(orig_size.height() * scale_factor)
        scaled_size = QtCore.QSize(scaled_width, scaled_height)
        
        image = QtGui.QImage(scaled_size, QtGui.QImage.Format_ARGB32)
        image.fill(QtGui.QColor("white"))
        
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        if hasattr(widget, 'scene') and widget.scene() is not None:
            target_rect = QtCore.QRectF(0, 0, scaled_size.width(), scaled_size.height())
            source_rect = QtCore.QRectF(0, 0, orig_size.width(), orig_size.height())
            
            # Temporarily disable ItemIgnoresTransformations for all items
            # (such as pyqtgraph's LegendItem) so they scale up correctly
            flags_to_restore = []
            export_items = []
            
            for item in widget.scene().items():
                if int(item.flags()) & QtWidgets.QGraphicsItem.ItemIgnoresTransformations:
                    item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, False)
                    flags_to_restore.append(item)
                
                # Notify pyqtgraph items so cosmetic pens scale up natively
                if hasattr(item, 'setExportMode'):
                    item.setExportMode(True, {'resolutionScale': scale_factor, 'antialias': True})
                    export_items.append(item)
                    
            widget.scene().render(painter, target_rect, source_rect)
            
            # Restore the flags and export mode
            for item in flags_to_restore:
                item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
            for item in export_items:
                item.setExportMode(False)
        else:
            painter.scale(scale_factor, scale_factor)
            widget.render(painter)
            
        painter.end()
        
        # Set DPI metadata for the image file
        dpm = int(dpi * 100 / 2.54)
        image.setDotsPerMeterX(dpm)
        image.setDotsPerMeterY(dpm)
        
        image.save(path, "PNG")

    @classmethod
    def handle_keyboard_event(cls, event):
        """Handle Ctrl+zoom/pan keys for network tabs only."""
        if cls.main_window is None or cls.tab_widget is None:
            return False
        if not cls.main_window.isActiveWindow():
            return False

        mods = event.modifiers()
        if not (mods & QtCore.Qt.ControlModifier):
            return False
        if mods & QtCore.Qt.AltModifier or mods & QtCore.Qt.MetaModifier:
            return False

        key = event.key()
        if key in (QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab):
            return False

        if key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            return cls.zoom_current_network_tab(zoom_in=False)
        if key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            return cls.zoom_current_network_tab(zoom_in=True)
        if key == QtCore.Qt.Key_Up:
            return cls.pan_current_network_tab("up")
        if key == QtCore.Qt.Key_Down:
            return cls.pan_current_network_tab("down")
        if key == QtCore.Qt.Key_Left:
            return cls.pan_current_network_tab("left")
        if key == QtCore.Qt.Key_Right:
            return cls.pan_current_network_tab("right")

        return False

    @classmethod
    def _current_tab_widget(cls):
        if cls.tab_widget is None:
            return None
        return cls.tab_widget.currentWidget()

    @classmethod
    def _current_network_view(cls):
        widget = cls._current_tab_widget()
        if isinstance(widget, QtWidgets.QGraphicsView):
            return widget
        return None

    @classmethod
    def fit_current_tab_to_data(cls):
        widget = cls._current_tab_widget()
        if widget is None:
            return
        for plotter in cls.instances:
            if plotter.win is widget:
                plotter._fit_to_full_data()
                break

    @classmethod
    def zoom_current_network_tab(cls, zoom_in=True):
        view = cls._current_network_view()
        if view is None:
            return False

        factor = cls.KEYBOARD_ZOOM_IN_FACTOR if zoom_in else cls.KEYBOARD_ZOOM_OUT_FACTOR
        current_scale = view.transform().m11()
        if current_scale <= 0.0:
            current_scale = 1.0
        min_scale = getattr(view, "MIN_SCALE", 0.02)
        max_scale = getattr(view, "MAX_SCALE", 50.0)
        target_scale = max(min_scale, min(max_scale, current_scale * factor))
        scale_factor = target_scale / current_scale
        view.scale(scale_factor, scale_factor)
        callback = getattr(view, "transform_changed_callback", None)
        if callback is not None:
            callback()
        return True

    @classmethod
    def pan_current_network_tab(cls, direction):
        view = cls._current_network_view()
        if view is None:
            return False

        step = int(cls.KEYBOARD_PAN_PIXELS)
        hbar = view.horizontalScrollBar()
        vbar = view.verticalScrollBar()
        if direction == "left":
            hbar.setValue(hbar.value() - step)
        elif direction == "right":
            hbar.setValue(hbar.value() + step)
        elif direction == "up":
            vbar.setValue(vbar.value() - step)
        elif direction == "down":
            vbar.setValue(vbar.value() + step)
        else:
            return False

        return True

    """
    y1 | [a, b, c, ...] component input or output values to be plotted on axis 1
    y2 | [d, e, f, ...] component input or output values to be plotted on axis 2
    """

    def __init__(self, y1, y2, y1lim, y2lim, y1label, y2label, nmax_points, update_every, plotter_size=(0.9, 0.9), tab_title=None, show_live=True, time_units='sec', *, font_size=None, font_color='#c0c0c0', bg_color=None, line_weights=2, line_styles=None, line_colors=None, grid=True, grid_horizontal=True, grid_vertical=True, array_config=None):
        """Plot time supplied in seconds using the selected display units.

        time_units must be nsec, msec, sec (default), min, hr, day, week,
        or year. A year is a fixed duration of 365 days. Each plotter has
        its own units, used for live and deferred rendering alike.
        """
        assert isinstance(y1, type([]))
        assert isinstance(y2, type([])) or y2 is None

        if not isinstance(time_units, str) or time_units not in self._SECONDS_PER_TIME_UNIT:
            valid_units = ', '.join(self._SECONDS_PER_TIME_UNIT)
            raise ValueError(f"Unsupported time_units {time_units!r}; expected one of: {valid_units}")
        self.time_units = time_units
        self._seconds_per_time_unit = self._SECONDS_PER_TIME_UNIT[time_units]
        self.show_live = show_live
        self.current_step = -1
        self._fill_idx = 0
        self._auto_follow = True
        self.y1label = y1label
        self.y2label = y2label
        self.nmax_points = nmax_points
        self.update_every = update_every
        self.y1_items = [self._normalize_item(item, self.y1label, idx) for idx, item in enumerate(y1)]
        self.y2_items = None if y2 is None else [self._normalize_item(item, self.y2label, idx) for idx, item in enumerate(y2)]
        self._y1lim = y1lim
        self._y2lim = y2lim
        self._plotter_size = plotter_size
        self._tab_title = tab_title
        self.grid = grid
        self.grid_horizontal = grid_horizontal
        self.grid_vertical = grid_vertical
        if font_size is not None and (not np.isfinite(font_size) or font_size <= 0):
            raise ValueError('font_size must be a positive finite number')
        self._initial_font_size = font_size
        self._font_size_baseline = OnlinePlotter.font_size_pt
        self.font_color = self._color(font_color)
        self.bg_color = None if bg_color is None else self._color(bg_color)
        nseries = len(y1) + (len(y2) if y2 is not None else 0)
        configs = array_config if isinstance(array_config, list) else [array_config] * nseries
        if len(configs) != nseries:
            raise ValueError(f'array_config must contain exactly {nseries} values (y1 then y2)')
        self.array_config = [self._array_options(config) for config in configs]
        self._array_artists = {}
        self.line_weights = self._series_options(line_weights, nseries, 'line_weights')
        self.line_styles = self._series_options(line_styles, nseries, 'line_styles')
        self.line_colors = self._series_options(line_colors, nseries, 'line_colors')
        # Validate before creating windows, including for deferred plotters.
        self._line_options = [self._make_line_options(i) for i in range(nseries)]
        self._timestep = 1.0

        # Pre-allocated numpy buffers, doubled on overflow; the initial size is
        # only a hint since timesteps (and so step count) can vary.
        self.preallocate(n_steps=1024, timestep=self._timestep)

        if show_live:
            self._build_widgets()
        else:
            self.y1_lines = []
            self.y2_lines = []

    @staticmethod
    def _array_options(config):
        defaults = dict(mode='fill_between', percentiles=(5, 95), indices=None,
                        show_mean=True, fill_alpha=0.2, spaghetti_alpha=0.15,
                        spaghetti_width=1, mean_width=None)
        if config is None:
            return defaults
        if not isinstance(config, dict):
            raise TypeError('array_config entries must be dicts or None')
        unknown = config.keys() - defaults.keys()
        if unknown:
            raise ValueError(f'Unknown array_config options: {sorted(unknown)}')
        defaults.update(config)
        mode = defaults['mode']
        if mode not in ('fill_between', 'spaghetti', 'indices'):
            raise ValueError('array_config mode must be fill_between, spaghetti, or indices')
        if mode == 'indices' and 'show_mean' not in config:
            defaults['show_mean'] = False
        percentiles = np.asarray(defaults['percentiles'], dtype=float)
        if percentiles.shape != (2,) or not np.all(np.isfinite(percentiles)) or not 0 <= percentiles[0] < percentiles[1] <= 100:
            raise ValueError('array_config percentiles must be two increasing numbers in [0, 100]')
        defaults['percentiles'] = tuple(percentiles)
        indices = defaults['indices']
        if indices is not None:
            if isinstance(indices, (int, np.integer)) and not isinstance(indices, bool):
                indices = [indices]
            if not isinstance(indices, (list, tuple, np.ndarray)) or not len(indices) or any(
                not isinstance(i, (int, np.integer)) or isinstance(i, bool) or i < 0 for i in indices
            ):
                raise ValueError('array_config indices must be a nonempty sequence of nonnegative integers')
            defaults['indices'] = list(dict.fromkeys(indices))
        if mode == 'indices' and indices is None:
            raise ValueError('array_config indices mode requires indices')
        if not isinstance(defaults['show_mean'], bool):
            raise ValueError('array_config show_mean must be bool')
        for key in ('fill_alpha', 'spaghetti_alpha'):
            if not np.isfinite(defaults[key]) or not 0 <= defaults[key] <= 1:
                raise ValueError(f'array_config {key} must be in [0, 1]')
        for key in ('spaghetti_width', 'mean_width'):
            value = defaults[key]
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f'array_config {key} must be finite and nonnegative')
        if defaults['spaghetti_width'] is None:
            raise ValueError('array_config spaghetti_width must be a number')
        return defaults

    @staticmethod
    def _color(value):
        if isinstance(value, QtGui.QColor):
            return QtGui.QColor(value)
        return QtGui.QColor.fromRgbF(*to_rgba(value))

    def _series_options(self, value, count, name):
        # Tuples represent scalar RGB(A) colors; lists represent series options.
        if isinstance(value, (list, np.ndarray)):
            value = list(value)
            nested = any(isinstance(item, (list, np.ndarray)) and
                         not (name == 'line_colors' and is_color_like(item))
                         for item in value)
            if nested:
                counts = (len(self.y1_items), len(self.y2_items or []))
                if len(value) != 2 or any(
                    not isinstance(group, (list, np.ndarray)) or len(group) != size
                    for group, size in zip(value, counts)
                ):
                    raise ValueError(f'{name} axis groups must contain {counts[0]} y1 '
                                     f'and {counts[1]} y2 values')
                value = [item for group in value for item in group]
            elif len(value) == 1:
                return value * count
            if len(value) != count:
                raise ValueError(f'{name} must contain exactly {count} values (y1 then y2)')
            return value
        return [value] * count

    def _make_line_options(self, index):
        palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                   '#984ea3', '#999999', '#e41a1c', '#dede00', '#00ffe5', '#5d0ab6']
        width = self.line_weights[index]
        if width is None:
            width = 2
        if not np.isfinite(width) or width < 0:
            raise ValueError('line_weights must be finite nonnegative numbers')
        fmt = self.line_styles[index]
        if fmt is None:
            style, marker, fmt_color = ('-' if index < len(self.y1_items) else '--'), None, None
        else:
            # Reuse Matplotlib's format grammar, including marker/color combinations.
            style, marker, fmt_color = _process_plot_format(fmt)
        styles = {'-': QtCore.Qt.SolidLine, '--': QtCore.Qt.DashLine,
                  ':': QtCore.Qt.DotLine, '-.': QtCore.Qt.DashDotLine}
        color = self.line_colors[index]
        color = self._color(color if color is not None else
                            fmt_color if fmt_color is not None else palette[index % len(palette)])
        pen = None if style in (None, 'None', '', ' ') else qtg.mkPen(color, width=float(width), style=styles[style])
        symbol = None
        if marker not in (None, 'None', '', ' '):
            mpl_marker = MarkerStyle(marker)
            path = mpl_marker.get_path().transformed(mpl_marker.get_transform())
            symbol = QtGui.QPainterPath()
            for polygon in path.to_polygons(closed_only=False):
                symbol.addPolygon(QtGui.QPolygonF([QtCore.QPointF(float(x), float(-y)) for x, y in polygon]))
        return dict(pen=pen, symbol=symbol, symbolPen=qtg.mkPen(color), symbolBrush=qtg.mkBrush(color), symbolSize=7)

    @staticmethod
    def _normalize_item(item, axis_label, index):
        """
        If a component input/output is passed, return it directly. 
        If a list/tuple/array is passed, wrap it in a _SequenceSeriesItem 
        with a default name. If a single value is passed, wrap it in a 
        _SequenceSeriesItem with a default name.
        """
        if hasattr(item, "v"):  #item is a component input/output with a .v property
            return item
        if isinstance(item, (list, tuple, np.ndarray)):  #item is array-like
            fallback_name = axis_label if axis_label else f"Series {index + 1}"
            if index > 0 and axis_label:
                fallback_name = f"{axis_label} {index + 1}"
            return _SequenceSeriesItem(item, name=fallback_name)
        #item is a single value
        fallback_name = axis_label if axis_label else f"Series {index + 1}"
        return _SequenceSeriesItem([item], name=fallback_name)

    def _build_widgets(self):
        """Create all Qt plot widgets and register this plotter in the shared window."""
        OnlinePlotter.build_window(plotter_size=self._plotter_size)
        self.app = OnlinePlotter.app

        self.win = qtg.GraphicsLayoutWidget()
        if self.bg_color is not None:
            self.win.setBackground(self.bg_color)
        OnlinePlotter.n_plotters += 1
        tab_label = self._tab_title if self._tab_title not in [None, ""] else self.y1label if self.y1label not in [None, ""] else f"Plot {OnlinePlotter.n_plotters}"
        OnlinePlotter.tab_widget.addTab(self.win, tab_label)

        self.ax1 = self.win.addPlot(viewBox=_AxisLockedViewBox())
        self.ax1.setLabel("bottom", f"Time ({self.time_units})")
        self.ax1.setLabel("left", self.y1label)
        self.legend_y1 = self.ax1.addLegend(offset=(10, 10))
        self.legend_y1.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 40)))
        self.legend_y2 = None
        self.ax1.showGrid(x=self.grid and self.grid_vertical,
                          y=self.grid and self.grid_horizontal, alpha=0.3)
        # pyqtgraph's built-in autoscale button only fits the currently-loaded
        # window of data, not the full history; use the "Fit All" button instead.
        self.ax1.hideButtons()
        self._tooltip_vline = qtg.InfiniteLine(
            angle=90,
            movable=False,
            pen=qtg.mkPen(QtGui.QColor(220, 40, 40, 220), width=1.5),
        )
        self._tooltip_vline.setZValue(1000)
        self.ax1.addItem(self._tooltip_vline, ignoreBounds=True)
        self._tooltip_vline.hide()

        if self._y1lim is not None:
            self.ax1.setYRange(*self._y1lim)

        self.y1_lines = []
        c = 0
        for i in range(len(self.y1_items)):
            line = self.ax1.plot([], [], name=self.y1_items[i].name, **self._line_options[c])
            self.y1_lines.append(line)
            c += 1

        self.y2_lines = []
        if self.y2_items is not None:
            self.ax2 = _AxisLockedViewBox()
            self.ax1.showAxis("right")
            self.ax1.scene().addItem(self.ax2)
            self.ax1.getAxis("right").linkToView(self.ax2)
            self.ax2.setXLink(self.ax1)
            self.ax1.getAxis("right").setLabel(self.y2label, color="#c0c0c0")
            self.legend_y2 = qtg.LegendItem(offset=(-10, 10))
            self.legend_y2.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 40)))
            self.legend_y2.setParentItem(self.ax1.vb)

            if self._y2lim is not None:
                self.ax2.setYRange(*self._y2lim)

            def updateViews():
                self.ax2.setGeometry(self.ax1.vb.sceneBoundingRect())
                self.ax2.linkedViewChanged(self.ax1.vb, self.ax2.XAxis)

            updateViews()
            self.ax1.vb.sigResized.connect(updateViews)

            for i in range(len(self.y2_items)):
                line = qtg.PlotDataItem([], [], name=self.y2_items[i].name, **self._line_options[c])
                self.ax2.addItem(line)
                self.legend_y2.addItem(line, self.y2_items[i].name)
                self.y2_lines.append(line)
                c += 1

        self.win.nextRow()
        self.ax_conv = self.win.addPlot()
        self.ax_conv.setMaximumHeight(18)
        self.ax_conv.setMinimumHeight(18)
        for _axis in ("left", "bottom", "right", "top"):
            self.ax_conv.hideAxis(_axis)
        self.ax_conv.hideButtons()
        self.ax_conv.setMouseEnabled(x=False, y=False)
        self.ax_conv.setYRange(0, 1, padding=0)
        self.ax_conv.setXLink(self.ax1)
        self._conv_bar_item = None

        self.win.nextRow()
        self.ax_iter = self.win.addPlot()
        self.ax_iter.setMaximumHeight(18)
        self.ax_iter.setMinimumHeight(18)
        for _axis in ("left", "bottom", "right", "top"):
            self.ax_iter.hideAxis(_axis)
        self.ax_iter.hideButtons()
        self.ax_iter.setMouseEnabled(x=False, y=False)
        self.ax_iter.setYRange(0, 1, padding=0)
        self.ax_iter.setXLink(self.ax1)
        self._iter_bar_item = None

        if self not in OnlinePlotter.instances:
            OnlinePlotter.instances.append(self)
        self.apply_font_size()
        self.ax1.vb.sigRangeChangedManually.connect(self._on_range_changed_manually)
        self._setup_strip_tooltip()

    def _finalize(self):
        """Build Qt widgets and render all buffered data. Called from wait_for_plots() for deferred plotters."""
        self._build_widgets()
        self.__refresh_plot()

    @classmethod
    def adjust_font_size(cls, delta):
        new_font_size = max(cls.min_font_size_pt, min(cls.max_font_size_pt, cls.font_size_pt + delta))
        if new_font_size == cls.font_size_pt:
            return

        cls.font_size_pt = new_font_size
        for plotter in cls.instances:
            plotter.apply_font_size()
        for target in cls.font_targets:
            target.apply_font_size()

    @classmethod
    def register_font_target(cls, target):
        if target not in cls.font_targets:
            cls.font_targets.append(target)

    @property
    def plot_font_size(self):
        if self._initial_font_size is None:
            return OnlinePlotter.font_size_pt
        return max(1, self._initial_font_size + OnlinePlotter.font_size_pt - self._font_size_baseline)

    def apply_font_size(self):
        size = self.plot_font_size
        tick_font = QtGui.QFont()
        tick_font.setPointSizeF(size)
        label_color = self.font_color.name(QtGui.QColor.HexArgb)
        label_style = {"font-size": f"{size}pt"}

        for name in ('left', 'bottom', 'right', 'top'):
            axis = self.ax1.getAxis(name)
            axis.setTextPen(self.font_color)
            axis.setPen(self.font_color)

        left_axis = self.ax1.getAxis("left")
        bottom_axis = self.ax1.getAxis("bottom")
        left_axis.setStyle(tickFont=tick_font)
        bottom_axis.setStyle(tickFont=tick_font)

        self.ax1.setLabel("left", self.y1label, color=label_color, **label_style)
        self.ax1.setLabel("bottom", f"Time ({self.time_units})", color=label_color, **label_style)

        if self.legend_y1 is not None:
            label_font = self.legend_y1.font()
            label_font.setPointSizeF(size)
            self.legend_y1.setFont(label_font)

            for _, label_item in self.legend_y1.items:
                label_item.setText(label_item.text, size=f"{size}pt", color=self.font_color)

        if self.y2_items is not None:
            right_axis = self.ax1.getAxis("right")
            right_axis.setStyle(tickFont=tick_font)
            right_axis.setLabel(self.y2label, color=label_color, **label_style)

            if self.legend_y2 is not None:
                for _, label_item in self.legend_y2.items:
                    label_item.setText(label_item.text, size=f"{size}pt", color=self.font_color)

    @staticmethod
    def _fraction_to_brush(f):
        if f <= 0.0:
            return QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        if f <= 0.5:
            t = f * 2.0
            r, g, b = 255, int(255 - t * (255 - 165)), 0
        else:
            t = (f - 0.5) * 2.0
            r, g, b = int(255 - t * (255 - 200)), int(165 * (1.0 - t)), 0
        return QtGui.QBrush(QtGui.QColor(r, g, b, 220))

    @staticmethod
    def _iter_fraction_to_brush(f):
        if f <= 0.0:
            return QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        alpha = int(f * 220)
        r = int(255 * (1.0 - f))
        g = int(255 * (1.0 - f))
        b = 220
        return QtGui.QBrush(QtGui.QColor(r, g, b, alpha))

    def preallocate(self, n_steps=None, timestep=1.0):
        self._fill_idx = 0
        self._timestep = timestep
        cap = max(1, int(n_steps) if n_steps is not None else 1024)
        self._capacity = cap
        self.x_data = np.empty(cap, dtype=float)
        self.y1_data = [np.empty(cap, dtype=float) for _ in self.y1_items]
        self.y2_data = [np.empty(cap, dtype=float) for _ in self.y2_items] if self.y2_items is not None else []
        self.conv_data = np.empty(cap, dtype=float)
        self.iter_data = np.empty(cap, dtype=float)
        self._step_timesteps = np.empty(cap, dtype=float)
        self._signal_shapes = [None] * (len(self.y1_items) + len(self.y2_items or []))

    def _grow_if_needed(self):
        """Double buffer capacity on overflow; correct regardless of timestep variability."""
        if self._fill_idx < self._capacity:
            return
        new_cap = self._capacity * 2

        def _grow(arr):
            new_arr = np.empty((new_cap,) + arr.shape[1:], dtype=arr.dtype)
            new_arr[: arr.shape[0]] = arr
            return new_arr

        self.x_data = _grow(self.x_data)
        self.y1_data = [_grow(a) for a in self.y1_data]
        if self.y2_items is not None:
            self.y2_data = [_grow(a) for a in self.y2_data]
        self.conv_data = _grow(self.conv_data)
        self.iter_data = _grow(self.iter_data)
        self._step_timesteps = _grow(self._step_timesteps)
        self._capacity = new_cap

    def log_step(self, time, conv_fraction=0.0, iter_fraction=0.0, timestep=None):
        """Buffer a step with time and timestep supplied in seconds."""
        values = []
        items = self.y1_items + (self.y2_items or [])
        for j, item in enumerate(items):
            value = np.asarray(item.v, dtype=float)
            if value.ndim == 2 and value.shape[0] == 1:
                value = value[0]
            if value.ndim > 1 or (value.ndim == 1 and value.size == 0):
                raise ValueError(f'{item.name}: plotted values must be scalar or nonempty arrays shaped (m,) or (1, m)')
            if self._signal_shapes[j] is not None and value.shape != self._signal_shapes[j]:
                raise ValueError(f'{item.name}: plotted value shape changed from {self._signal_shapes[j]} to {value.shape}')
            indices = self.array_config[j]['indices']
            if value.ndim == 1 and indices is not None and max(indices) >= value.size:
                raise ValueError(f'{item.name}: array_config index {max(indices)} is out of range for {value.size} members')
            values.append(value)
        self.current_step += 1
        if timestep is None:
            timestep = self._timestep

        self._grow_if_needed()
        idx = self._fill_idx

        self.x_data[idx] = time / self._seconds_per_time_unit
        self.conv_data[idx] = conv_fraction
        self.iter_data[idx] = iter_fraction
        self._step_timesteps[idx] = timestep / self._seconds_per_time_unit

        for j, value in enumerate(values):
            buffers = self.y1_data if j < len(self.y1_items) else self.y2_data
            k = j if j < len(self.y1_items) else j - len(self.y1_items)
            if self._signal_shapes[j] is None:
                buffers[k] = np.empty((self._capacity,) + value.shape, dtype=float)
                self._signal_shapes[j] = value.shape
            buffers[k][idx] = value

        self._fill_idx += 1

        if self.show_live and self.current_step % self.update_every == 0:
            self.__refresh_plot()

    def __refresh_plot(self):
        n = self._fill_idx
        if n == 0:
            return

        if self._auto_follow:
            x_view = self.x_data[:n]
            x_start = x_view[max(0, n - self.nmax_points)]
            x_end = x_view[n - 1]
            self.ax1.setXRange(x_start, x_end, padding=0.02)

        self._render_visible_window()

    def _render_visible_window(self):
        """Re-slice buffered data to whatever the current view range covers and push it to the plot items.

        Needed both after new steps are logged and after a manual pan/zoom, since the
        latter doesn't add new data but does change which slice of the buffers is visible.
        """
        n = self._fill_idx
        if n == 0:
            return
        x_view = self.x_data[:n]

        # Only render points that fall in (or near) the current view: rendering
        # cost must depend on what's on screen, not on total steps taken so far.
        view_x_start, view_x_end = self.ax1.viewRange()[0]
        start_idx = max(0, int(np.searchsorted(x_view, view_x_start, side="left")) - 1)
        end_idx = min(n, int(np.searchsorted(x_view, view_x_end, side="right")) + 1)

        x_render = x_view[start_idx:end_idx]

        for axis, (lines, buffers, items) in enumerate((
            (self.y1_lines, self.y1_data, self.y1_items),
            (self.y2_lines, self.y2_data, self.y2_items or []),
        )):
            for i, line in enumerate(lines):
                values = buffers[i][start_idx:end_idx]
                if values.ndim == 1:
                    line.setData(x_render, values)
                else:
                    index = i if axis == 0 else len(self.y1_items) + i
                    self._render_array(index, axis, line, items[i], x_render, values)

        if self._conv_bar_item is not None:
            self.ax_conv.removeItem(self._conv_bar_item)
            self._conv_bar_item = None
        conv_data = self.conv_data[start_idx:end_idx]
        nc_mask = conv_data > 0
        if np.any(nc_mask):
            nc_x = x_render[nc_mask]
            nc_widths = self._step_timesteps[start_idx:end_idx][nc_mask]
            brushes = [self._fraction_to_brush(f) for f in conv_data[nc_mask]]
            self._conv_bar_item = qtg.BarGraphItem(
                x0=nc_x - (nc_widths * 0.5),
                x1=nc_x + (nc_widths * 0.5),
                height=1,
                brushes=brushes,
                pen=qtg.mkPen(None),
            )
            self.ax_conv.addItem(self._conv_bar_item)

        if self._iter_bar_item is not None:
            self.ax_iter.removeItem(self._iter_bar_item)
            self._iter_bar_item = None
        iter_data = self.iter_data[start_idx:end_idx]
        iter_widths = self._step_timesteps[start_idx:end_idx]
        iter_brushes = [self._iter_fraction_to_brush(f) for f in iter_data]
        self._iter_bar_item = qtg.BarGraphItem(
            x0=x_render - (iter_widths * 0.5),
            x1=x_render + (iter_widths * 0.5),
            height=1,
            brushes=iter_brushes,
            pen=qtg.mkPen(None),
        )
        self.ax_iter.addItem(self._iter_bar_item)

    def _render_array(self, index, axis, mean_line, item, x, values):
        config = self.array_config[index]
        if index not in self._array_artists:
            target = self.ax1 if axis == 0 else self.ax2
            legend = self.legend_y1 if axis == 0 else self.legend_y2
            color = self._line_options[index]['symbolBrush'].color()
            members = config['indices']
            if members is None:
                members = list(range(values.shape[1])) if config['mode'] == 'spaghetti' else []
            member_lines = []
            for member in members:
                options = dict(self._line_options[index])
                if config['mode'] == 'spaghetti':
                    member_color = QtGui.QColor(color)
                    member_color.setAlphaF(config['spaghetti_alpha'])
                    options = dict(pen=qtg.mkPen(member_color, width=config['spaghetti_width']))
                line = qtg.PlotDataItem([], [], **options)
                target.addItem(line)
                if config['mode'] != 'spaghetti':
                    legend.addItem(line, f'{item.name}[{member}]')
                member_lines.append((member, line))
            bounds = None
            if config['mode'] == 'fill_between':
                # PlotDataItem skips updating its curve when pen=None, leaving
                # FillBetweenItem with an empty path. PlotCurveItem retains
                # boundary geometry even when its outline is not drawn.
                lower = qtg.PlotCurveItem(pen=None)
                upper = qtg.PlotCurveItem(pen=None)
                target.addItem(lower)
                target.addItem(upper)
                fill_color = QtGui.QColor(color)
                fill_color.setAlphaF(config['fill_alpha'])
                fill = qtg.FillBetweenItem(lower, upper, brush=qtg.mkBrush(fill_color))
                fill.setZValue(-10)
                target.addItem(fill)
                low, high = config['percentiles']
                legend.addItem(_FillLegendSample(fill), f'{item.name} ({low:g}–{high:g}%)')
                bounds = (lower, upper, fill)
            legend.removeItem(mean_line)
            if config['show_mean']:
                legend.addItem(mean_line, f'{item.name} mean')
                mean_line.setZValue(10)
                if config['mean_width'] is not None:
                    pen = qtg.mkPen(mean_line.opts['pen'])
                    pen.setWidthF(config['mean_width'])
                    mean_line.setPen(pen)
            mean_line.setVisible(config['show_mean'])
            self._array_artists[index] = (member_lines, bounds)
            self.apply_font_size()
        member_lines, bounds = self._array_artists[index]
        mean_line.setData(x, values.mean(axis=1))
        for member, line in member_lines:
            line.setData(x, values[:, member])
        if bounds is not None:
            lower, upper = np.percentile(values, config['percentiles'], axis=1)
            bounds[0].setData(x, lower)
            bounds[1].setData(x, upper)

    def _fit_to_full_data(self):
        """Zoom out to the entire logged history, bypassing pyqtgraph's own autoscale
        button which only fits to whatever slice is currently set on the plot items."""
        n = self._fill_idx
        if n == 0:
            return

        self._auto_follow = False
        if OnlinePlotter.follow_button is not None:
            OnlinePlotter.follow_button.setChecked(False)

        x_view = self.x_data[:n]
        self.ax1.setXRange(x_view[0], x_view[n - 1], padding=0.02)

        y1_values = [self._plotted_values(i, a[:n]) for i, a in enumerate(self.y1_data)]
        y1_min = min(np.nanmin(a) for a in y1_values)
        y1_max = max(np.nanmax(a) for a in y1_values)
        if self._y1lim is None and y1_min <= y1_max:
            self.ax1.setYRange(y1_min, y1_max, padding=0.05)

        if self.y2_items is not None and self._y2lim is None:
            y2_values = [self._plotted_values(len(self.y1_items) + i, a[:n])
                         for i, a in enumerate(self.y2_data)]
            y2_min = min(np.nanmin(a) for a in y2_values)
            y2_max = max(np.nanmax(a) for a in y2_values)
            if y2_min <= y2_max:
                self.ax2.setYRange(y2_min, y2_max, padding=0.05)

        self._render_visible_window()

    def _plotted_values(self, index, values):
        """Bounds for Fit All reflect the selected traces and statistics."""
        if values.ndim == 1:
            return values
        config = self.array_config[index]
        displayed = []
        if config['show_mean']:
            displayed.append(values.mean(axis=1).ravel())
        if config['mode'] == 'fill_between':
            displayed.append(np.percentile(values, config['percentiles'], axis=1).ravel())
        members = config['indices']
        if members is not None:
            displayed.append(values[:, members].ravel())
        elif config['mode'] == 'spaghetti':
            displayed.append(values.ravel())
        return np.concatenate(displayed)

    def _setup_strip_tooltip(self):
        self._strip_proxy = qtg.SignalProxy(self.ax1.scene().sigMouseMoved, rateLimit=20, slot=self._on_strip_mouse_move)
        self._tooltip_last_idx = -1
        self._plot_tooltip_enabled = False
        self._tip_label = QtWidgets.QLabel()
        self._tip_label.setWindowFlags(QtCore.Qt.ToolTip | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        tooltip_bg = self.win.backgroundBrush().color().name()
        self._tip_label.setStyleSheet(
            f"QLabel {{ background-color: {tooltip_bg}; color: {self.font_color.name()}; "
            "border: 1px solid #aaaaaa; padding: 4px 8px; }"
        )
        self._tip_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._tip_label.hide()

    def _on_strip_mouse_move(self, args):
        scene_pos = args[0]

        if self.x_data is None or self._fill_idx == 0:
            self._tip_label.hide()
            self._tooltip_vline.hide()
            self._tooltip_last_idx = -1
            return

        in_conv = self.ax_conv.vb.sceneBoundingRect().contains(scene_pos)
        in_iter = self.ax_iter.vb.sceneBoundingRect().contains(scene_pos)
        in_plot = self.ax1.vb.sceneBoundingRect().contains(scene_pos)
        in_y2 = self.y2_items is not None and hasattr(self, "ax2") and self.ax2.sceneBoundingRect().contains(scene_pos)

        if not in_conv and not in_iter and not in_plot and not in_y2:
            self._tip_label.hide()
            self._tooltip_vline.hide()
            self._tooltip_last_idx = -1
            return

        if (in_plot or in_y2) and not in_conv and not in_iter and not self._plot_tooltip_enabled:
            self._tip_label.hide()
            self._tooltip_vline.hide()
            self._tooltip_last_idx = -1
            return

        vb = self.ax_conv.vb if in_conv else (self.ax_iter.vb if in_iter else self.ax1.vb)
        x_val = vb.mapSceneToView(scene_pos).x()

        n = self._fill_idx
        x_slice = self.x_data[:n]
        idx = int(np.searchsorted(x_slice, x_val, side="left"))

        idx = max(0, min(idx, n - 1))
        if idx > 0 and abs(x_slice[idx - 1] - x_val) < abs(x_slice[idx] - x_val):
            idx -= 1

        t = x_slice[idx]
        if self._plot_tooltip_enabled:
            self._tooltip_vline.setPos(t)
            self._tooltip_vline.show()
        else:
            self._tooltip_vline.hide()

        cursor_pos = QtGui.QCursor.pos()
        self._tip_label.move(cursor_pos.x() + 16, cursor_pos.y() + 16)

        if idx == self._tooltip_last_idx and self._tip_label.isVisible():
            return
        self._tooltip_last_idx = idx

        if in_conv or in_iter:
            text = f"time | {t:.6g} {self.time_units}\nConv failures: {self.conv_data[idx]:.1%}\nIter fraction: {self.iter_data[idx]:.1%}"
        else:
            lines = [f"time | {t:.6g} {self.time_units}"]
            display_items = self.y1_items + (self.y2_items if self.y2_items is not None else [])
            ncmax = -1
            for item in display_items:
                ncmax = max(ncmax, len(item.name))
            for j, item in enumerate(display_items):
                unit_str = f" {item.units}" if item.units else ""
                # which axis data are we pointing at?
                axis_data = self.y1_data if j < len(self.y1_items) else self.y2_data
                jj = j if j < len(self.y1_items) else j - len(self.y1_items)
                axis_values = np.asarray(axis_data[jj], dtype=float)
                value = axis_values[idx]
                if np.ndim(value):
                    summary = f'mean {np.mean(value):.4g}; ' + np.array2string(value, precision=4, threshold=8)
                else:
                    summary = f'{value:.4g}'
                lines.append(f"{item.name:<{ncmax}s} | {summary}{unit_str}")
            text = "\n".join(lines)

        tooltip_font = QtGui.QFont('Consolas')
        tooltip_font.setPointSizeF(self.plot_font_size)
        self._tip_label.setFont(tooltip_font)
        self._tip_label.setText(text)
        self._tip_label.adjustSize()
        self._tip_label.show()

    def _on_range_changed_manually(self, viewRange):
        self._auto_follow = False
        if OnlinePlotter.follow_button is not None:
            OnlinePlotter.follow_button.setChecked(False)
        self._render_visible_window()

    def _enable_y_autorange(self):
        """Manual pan/zoom disables autorange per-axis; re-enable Y so Follow tracks
        both axes again, not just X (which __refresh_plot re-enables explicitly)."""
        if self._y1lim is None:
            self.ax1.vb.enableAutoRange(y=True)
        if self.y2_items is not None and self._y2lim is None:
            self.ax2.enableAutoRange(y=True)

    @classmethod
    def set_auto_follow(cls, value):
        for plotter in cls.instances:
            plotter._auto_follow = value
            if value:
                plotter._enable_y_autorange()
                plotter.__refresh_plot()
        if cls.follow_button is not None:
            cls.follow_button.setChecked(value)

    @classmethod
    def set_plot_tooltip(cls, value):
        for plotter in cls.instances:
            plotter._plot_tooltip_enabled = value
            if not value:
                plotter._tip_label.hide()
                plotter._tooltip_vline.hide()
                plotter._tooltip_last_idx = -1
        if cls.tooltip_button is not None:
            cls.tooltip_button.setChecked(value)

    @classmethod
    def process_ui_events(cls, force=False):
        """Process Qt events with throttling so UI activity cannot starve simulation."""
        if cls.app is None:
            return

        now = time.perf_counter()
        if not force and (now - cls._last_ui_event_process) < cls.UI_EVENT_PROCESS_INTERVAL_SEC:
            return

        try:
            cls.app.processEvents(QtCore.QEventLoop.AllEvents, cls.UI_EVENT_PROCESS_BUDGET_MS)
        except TypeError:
            # Compatibility fallback for bindings without max-time overload.
            cls.app.processEvents()

        cls._last_ui_event_process = now
