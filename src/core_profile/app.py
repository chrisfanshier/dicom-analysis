"""core-profile viewer: open a stitched coronal MPR, look at it coronally and axially, set the greyscale window,
and place the top and bottom of the core.

Step one of the profile tool; it has the same view controls as the stitch window (slice spin, [ ] and PageUp/Down,
wheel zoom, drag pan, histogram with black/white handles, draggable cyan/amber core-end guides with inset boxes)
and nothing else yet. No profile or segmentation is computed here.

Coordinates: both views are drawn in millimetres. Depth runs down the coronal view (row 0 = top of the image).
The axial view is made at the depth line by taking that row of every coronal slice; the stack is coarse (often 2 mm),
so the axial picture is interpolated across the stack for display only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy import ndimage as ndi

from core_stitch.load import MprSeries, load_series
from core_stitch.scale import auto_bounds, clamp_insets

from .export import write_profile_csv
from .mask import MIN_BLOB_PX, MIN_ROW_VOXELS, auto_clast_threshold, auto_thresholds, fit_axis_path, fit_outer_circle, mask_3d, mask_slice, row_profile

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

TOP_COLOR = "#22d3ee"  # same colours as the stitch window's guides
BOTTOM_COLOR = "#f59e0b"
DEPTH_COLOR = "#ffffff"
MASK_COLOR = "#22c55e"
CLAST_COLOR = "#e879f9"
CIRCLE_COLOR = "#facc15"
LINER_MM = 6.35  # a quarter-inch liner wall


def axial_image(volume: np.ndarray, row: int, dz_mm: float, dc_mm: float) -> np.ndarray:
    """Axial slice at ``row``: every coronal slice's row, stacked (stack axis down, columns across), interpolated
    across the stack to square pixels (display only)."""
    a = np.asarray(volume[:, int(row), :], np.float32)
    f = dz_mm / dc_mm
    return a if abs(f - 1) < 1e-3 else ndi.zoom(a, (f, 1.0), order=1, mode="nearest", grid_mode=True)


def histogram_range(vals: np.ndarray) -> tuple[float, float]:
    """The same range the stitch window uses: 0.5-99.5th percentile plus 5 % padding."""
    vmin, vmax = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))
    pad = 0.05 * max(vmax - vmin, 1.0)
    vmin, vmax = vmin - pad, vmax + pad
    return (vmin, vmax) if vmax > vmin else (vmin, vmin + 1.0)


def core_values(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Core+liner mask of one coronal slice (Otsu, biggest blob) and the HU values inside it.

    Unlike the stitch window's segmentation there is no "invert if the mask fills the picture" rule: the core is
    always brighter than the air around it, and a stitched export is cropped tight so the core can fill most of it.
    Falls back to every pixel if no core is found.
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    finite = np.isfinite(image)
    mask = finite & (image >= threshold_otsu(image[finite])) if finite.any() else np.zeros(image.shape, bool)
    lab = label(mask, connectivity=2)
    if lab.max() > 0:
        mask = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
    v = image[mask]
    if v.size == 0:
        mask = np.ones(image.shape, bool)
        v = image.ravel()
    return mask, v


class _Signals(QtCore.QObject):
    done = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)


class _Job(QtCore.QRunnable):
    """Run ``fn()`` on a pool thread and report ``(generation, result)``."""

    def __init__(self, gen: int, fn) -> None:
        super().__init__()
        self.gen, self.fn, self.sig = gen, fn, _Signals()

    def run(self) -> None:
        try:
            res = self.fn()
        except Exception:  # noqa: BLE001 - shown in the window
            import traceback

            res, err = None, traceback.format_exc()
        else:
            err = None
        try:
            (self.sig.done.emit(self.gen, res) if err is None else self.sig.failed.emit(self.gen, err))
        except RuntimeError:
            pass  # the window closed while this ran


class SliderSpin(QtWidgets.QWidget):
    """A slider and a spin box on one value."""

    changed = QtCore.Signal(float)

    def __init__(self, label: str, decimals: int = 1) -> None:
        super().__init__()
        self._busy = False
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel(label)
        self.label.setMinimumWidth(40)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(-1e6, 1e6)
        self.spin.setKeyboardTracking(False)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._lo, self._hi = 0.0, 1.0
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def set_range(self, lo: float, hi: float) -> None:
        self._lo, self._hi = lo, hi
        self.set_value(self.spin.value())

    def set_value(self, v: float) -> None:
        self._busy = True
        self.spin.setValue(v)
        t = (v - self._lo) / max(self._hi - self._lo, 1e-9)
        self.slider.setValue(int(round(np.clip(t, 0, 1) * 1000)))
        self._busy = False

    def value(self) -> float:
        return self.spin.value()

    def _from_slider(self, k: int) -> None:
        if self._busy:
            return
        v = self._lo + k / 1000 * (self._hi - self._lo)
        self._busy = True
        self.spin.setValue(v)
        self._busy = False
        self.changed.emit(v)

    def _from_spin(self, v: float) -> None:
        if self._busy:
            return
        self.set_value(v)
        self.changed.emit(v)


class Viewer(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("core-profile - viewer")
        self.resize(1400, 900)
        self.series: MprSeries | None = None
        self.frame = 0
        self.depth_mm = 0.0
        self.win = (0.0, 1.0)
        self.hist_range = (0.0, 1.0)
        self.top_cm = 0.0
        self.bottom_cm = 0.0
        self.auto_ends = (0.0, 0.0)
        self.bounds_source = "auto"
        self._hist_values = np.zeros(1)
        self._busy = False
        self.mask3d: np.ndarray | None = None  # sediment mask per coronal slice (None until computed)
        self.profile: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # mean, SD, count per row
        self._mask_gen = 0
        self._pool = QtCore.QThreadPool.globalInstance()
        self._jobs: list = []
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.recompute_mask)
        self._roi_ready = False
        self._liner_prev = LINER_MM
        self._path: tuple[np.ndarray, np.ndarray] | None = None  # axis (cx, cy) per row when following the wobble
        self._off = (0.0, 0.0)  # how far the user has dragged the circle off that path
        self._build()
        self.statusBar().showMessage("Open a stitched coronal MPR folder.")

    # ------------------------------------------------------------------ geometry helpers
    @property
    def dz(self) -> float:
        return self.series.spacing_mm[0]

    @property
    def dr(self) -> float:
        return self.series.spacing_mm[1]

    @property
    def dc(self) -> float:
        return self.series.spacing_mm[2]

    @property
    def n_frames(self) -> int:
        return int(self.series.volume.shape[0])

    @property
    def extent_cm(self) -> float:
        return self.series.volume.shape[1] * self.dr / 10.0

    # ------------------------------------------------------------------ UI
    def _group(self, title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout]:
        g = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(g)
        return g, v

    @staticmethod
    def _hint(text: str) -> QtWidgets.QLabel:
        h = QtWidgets.QLabel(text)
        h.setStyleSheet("font-size: 10px")
        h.setWordWrap(True)
        return h

    def _build(self) -> None:
        side = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(side)

        g, v = self._group("Scan")
        self.open_btn = QtWidgets.QPushButton("Open MPR folder...")
        self.open_btn.clicked.connect(self.open_folder)
        self.info = QtWidgets.QLabel("-")
        self.info.setWordWrap(True)
        v.addWidget(self.open_btn)
        v.addWidget(self.info)
        col.addWidget(g)

        g, v = self._group("Slice / view")
        self.z_label = QtWidgets.QLabel("Coronal slice: -")
        row = QtWidgets.QHBoxLayout()
        self.z_spin = QtWidgets.QSpinBox()
        self.z_spin.setKeyboardTracking(False)
        self.z_spin.valueChanged.connect(self.set_frame)
        mid = QtWidgets.QPushButton("Mid")
        mid.clicked.connect(lambda: self.set_frame(self.n_frames // 2) if self.series else None)
        row.addWidget(self.z_spin)
        row.addWidget(mid)
        row.addStretch(1)
        v.addWidget(self.z_label)
        v.addLayout(row)
        v.addWidget(self._hint("[ ] / PageUp-PageDown = previous / next coronal slice"))
        zr = QtWidgets.QHBoxLayout()
        for text, fn in (("-", lambda: self.zoom_by(1 / 1.25)), ("+", lambda: self.zoom_by(1.25)), ("Fit", self.zoom_fit), ("100%", self.zoom_100)):
            b = QtWidgets.QPushButton(text)
            b.setMaximumWidth(48)
            b.clicked.connect(fn)
            zr.addWidget(b)
        zr.addStretch(1)
        v.addLayout(zr)
        v.addWidget(self._hint("Wheel = zoom at pointer; drag = pan; drag a cyan/amber guide to move it"))
        col.addWidget(g)

        g, v = self._group("Axial slice")
        self.depth_spin = QtWidgets.QDoubleSpinBox()
        self.depth_spin.setDecimals(2)
        self.depth_spin.setSuffix(" cm from top of image")
        self.depth_spin.setKeyboardTracking(False)
        self.depth_spin.valueChanged.connect(lambda cm: None if self._busy else self.set_depth_mm(cm * 10))
        v.addWidget(self.depth_spin)
        v.addWidget(self._hint("White dashed line on the coronal view = where the axial slice is cut. White dashed line on the axial view = the coronal slice shown (drag it to change slice)."))
        col.addWidget(g)

        g, v = self._group("Greyscale window (HU)")
        self.hist = pg.PlotWidget()
        self.hist.setFixedHeight(110)
        self.hist.setMenuEnabled(False)
        self.hist.setMouseEnabled(False, False)
        self.hist.hideAxis("left")
        self.hist_bars = pg.BarGraphItem(x=[0], height=[0], width=1, brush=(106, 159, 181))
        self.hist.addItem(self.hist_bars)
        self.region = pg.LinearRegionItem(values=(0, 1), brush=(255, 255, 255, 60))
        self.region.lines[0].setPen(pg.mkPen("#ff6666", width=2))
        self.region.lines[1].setPen(pg.mkPen("#66ff99", width=2))
        self.region.sigRegionChanged.connect(self._region_moved)
        self.hist.addItem(self.region)
        self.t_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#fb923c", width=2, style=QtCore.Qt.DashLine), hoverPen=pg.mkPen("#fb923c", width=4))
        self.t_line.sigPositionChanged.connect(self._threshold_dragged)
        self.hist.addItem(self.t_line)
        self.c_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(CLAST_COLOR, width=2, style=QtCore.Qt.DashLine), hoverPen=pg.mkPen(CLAST_COLOR, width=4))
        self.c_line.sigPositionChanged.connect(self._clast_dragged)
        self.c_line.setVisible(False)
        self.hist.addItem(self.c_line)
        v.addWidget(self.hist)
        self.black = SliderSpin("Black")
        self.white = SliderSpin("White")
        self.black.changed.connect(lambda _x: self._window_from_boxes())
        self.white.changed.connect(lambda _x: self._window_from_boxes())
        v.addWidget(self.black)
        v.addWidget(self.white)
        row = QtWidgets.QHBoxLayout()
        auto = QtWidgets.QPushButton("Auto 1-99%")
        auto.clicked.connect(lambda: self.auto_window())
        self.win_label = QtWidgets.QLabel("")
        row.addWidget(auto)
        row.addWidget(self.win_label, 1)
        v.addLayout(row)
        col.addWidget(g)

        g, v = self._group("Sediment mask")
        self.cyl_box = QtWidgets.QCheckBox("Cylinder (cuts away the liner)")
        self.cyl_box.toggled.connect(self._cyl_toggled)
        v.addWidget(self.cyl_box)
        row = QtWidgets.QHBoxLayout()
        self.liner_spin = QtWidgets.QDoubleSpinBox()
        self.liner_spin.setDecimals(2)
        self.liner_spin.setRange(0, 30)
        self.liner_spin.setSingleStep(0.25)
        self.liner_spin.setValue(LINER_MM)
        self.liner_spin.setSuffix(" mm")
        self.liner_spin.setKeyboardTracking(False)
        self.liner_spin.valueChanged.connect(self._liner_changed)
        fit = QtWidgets.QPushButton("Auto-fit")
        fit.clicked.connect(self._autofit_clicked)
        row.addWidget(QtWidgets.QLabel("Liner wall"))
        row.addWidget(self.liner_spin, 1)
        row.addWidget(fit)
        v.addLayout(row)
        self.follow_box = QtWidgets.QCheckBox("Follow the core's wobble (fit at many depths)")
        self.follow_box.toggled.connect(self._follow_toggled)
        v.addWidget(self.follow_box)
        self.cyl_label = QtWidgets.QLabel("")
        v.addWidget(self.cyl_label)
        v.addWidget(self._hint("Auto-fit finds the outer edge of the liner at the axial cut and draws a yellow circle inside it. Drag the circle to move it, drag its handle to resize it. The threshold below is left as it is: the mask is the threshold AND the circle."))
        row = QtWidgets.QHBoxLayout()
        self.t_spin = QtWidgets.QDoubleSpinBox()
        self.t_spin.setDecimals(0)
        self.t_spin.setRange(-5000, 20000)
        self.t_spin.setSuffix(" HU")
        self.t_spin.setKeyboardTracking(False)
        self.t_spin.valueChanged.connect(self._threshold_from_spin)
        auto_t = QtWidgets.QPushButton("Auto (Otsu)")
        auto_t.clicked.connect(self.auto_threshold)
        row.addWidget(QtWidgets.QLabel("Sediment at or above"))
        row.addWidget(self.t_spin, 1)
        row.addWidget(auto_t)
        v.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.margin_spin = QtWidgets.QDoubleSpinBox()
        self.margin_spin.setDecimals(1)
        self.margin_spin.setRange(0, 20)
        self.margin_spin.setSingleStep(0.5)
        self.margin_spin.setValue(1.0)
        self.margin_spin.setSuffix(" mm")
        self.margin_spin.setKeyboardTracking(False)
        self.margin_spin.valueChanged.connect(lambda _v: self._mask_params_changed())
        row.addWidget(QtWidgets.QLabel("Shrink from edges"))
        row.addWidget(self.margin_spin, 1)
        v.addLayout(row)
        self.void_box = QtWidgets.QCheckBox("Remove air / voids (below the orange line, inside the sediment)")
        self.void_box.toggled.connect(lambda _c: self._mask_params_changed())
        self.clast_box = QtWidgets.QCheckBox("Remove clasts / dense things (at or above the pink line)")
        self.clast_box.toggled.connect(self._clast_toggled)
        row_c = QtWidgets.QHBoxLayout()
        self.c_spin = QtWidgets.QDoubleSpinBox()
        self.c_spin.setDecimals(0)
        self.c_spin.setRange(-5000, 20000)
        self.c_spin.setSuffix(" HU")
        self.c_spin.setKeyboardTracking(False)
        self.c_spin.valueChanged.connect(self._clast_from_spin)
        auto_c = QtWidgets.QPushButton("Auto")
        auto_c.clicked.connect(self.auto_clast)
        row_c.addWidget(QtWidgets.QLabel("Dense at or above"))
        row_c.addWidget(self.c_spin, 1)
        row_c.addWidget(auto_c)
        self.show_mask = QtWidgets.QCheckBox("Show mask (green)")
        self.show_mask.setChecked(True)
        self.show_mask.toggled.connect(self._mask_visible)
        v.addWidget(self.void_box)
        v.addWidget(self.clast_box)
        v.addLayout(row_c)
        v.addWidget(self.show_mask)
        self.mask_label = QtWidgets.QLabel("mask: -")
        v.addWidget(self.mask_label)
        v.addWidget(self._hint("Drag the orange line on the histogram, or type a value. Per coronal slice the sediment is the biggest blob at or above it, so whole-round and split cores work alike. If the liner reads brighter than the sediment this one number cannot separate them. Remove air / voids and Remove clasts cut those voxels out of the mask (and the profile); both use one global level, so they work best on cores without a strong density trend."))
        col.addWidget(g)

        g, v = self._group("Core ends")
        self.show_guides = QtWidgets.QCheckBox("Show core-end guides")
        self.show_guides.setChecked(True)
        self.show_guides.toggled.connect(self._guides_visible)
        v.addWidget(self.show_guides)
        self.top_edit = QtWidgets.QDoubleSpinBox()
        self.bot_edit = QtWidgets.QDoubleSpinBox()
        for e in (self.top_edit, self.bot_edit):
            e.setDecimals(2)
            e.setRange(0, 1e4)
            e.setSingleStep(0.1)
            e.setSuffix(" cm")
            e.setKeyboardTracking(False)
        self.top_edit.valueChanged.connect(lambda _v: self._ends_from_boxes())
        self.bot_edit.valueChanged.connect(lambda _v: self._ends_from_boxes())
        for text, color, edit, which in (("Top inset", TOP_COLOR, self.top_edit, "top"), ("Bottom inset", BOTTOM_COLOR, self.bot_edit, "bottom")):
            row = QtWidgets.QHBoxLayout()
            sw = QtWidgets.QLabel("■")
            sw.setStyleSheet(f"color: {color}")
            go = QtWidgets.QPushButton("Go to")
            go.clicked.connect(lambda _c=False, w=which: self.goto_end(w))
            row.addWidget(sw)
            row.addWidget(QtWidgets.QLabel(text))
            row.addWidget(edit, 1)
            row.addWidget(go)
            v.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("Reset to automatic")
        reset.clicked.connect(self.reset_ends)
        row.addWidget(reset)
        row.addStretch(1)
        v.addLayout(row)
        self.len_label = QtWidgets.QLabel("Core length: -")
        v.addWidget(self.len_label)
        v.addWidget(self._hint("Insets are measured in from the top and bottom of the image, as in the stitch window."))
        col.addWidget(g)
        g, v = self._group("Profile")
        save = QtWidgets.QPushButton("Save profile (CSV)...")
        save.clicked.connect(lambda: self.save_profile())
        v.addWidget(save)
        self.save_label = QtWidgets.QLabel("")
        self.save_label.setWordWrap(True)
        v.addWidget(self.save_label)
        v.addWidget(self._hint("One line per row between the core-end guides, with a commented header holding the input file and every setting above."))
        col.addWidget(g)
        col.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(420)

        # coronal view
        self.cor = pg.PlotWidget()
        self.cor_vb = self.cor.getPlotItem().getViewBox()
        self.cor_vb.invertY(True)
        self.cor_vb.setAspectLocked(True)
        self.cor_vb.setMouseMode(pg.ViewBox.PanMode)
        self.cor.setLabel("left", "depth (mm)")
        self.cor.setLabel("bottom", "across (mm)")
        self.cor_img = pg.ImageItem()
        self.cor.addItem(self.cor_img)
        lut = np.array([[0, 0, 0, 0], [34, 197, 94, 90]], np.uint8)
        self.cor_mask = pg.ImageItem()
        self.cor_mask.setLookupTable(lut)
        self.cor_mask.setLevels((0, 1))
        self.cor.addItem(self.cor_mask)
        self.cor_title = QtWidgets.QLabel("coronal")
        self.top_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen(TOP_COLOR, width=2), hoverPen=pg.mkPen(TOP_COLOR, width=4))
        self.bot_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen(BOTTOM_COLOR, width=2), hoverPen=pg.mkPen(BOTTOM_COLOR, width=4))
        dash = pg.mkPen(DEPTH_COLOR, width=1, style=QtCore.Qt.DashLine)
        self.depth_line = pg.InfiniteLine(angle=0, movable=True, pen=dash, hoverPen=pg.mkPen(DEPTH_COLOR, width=3))
        for ln in (self.top_line, self.bot_line, self.depth_line):
            self.cor.addItem(ln)
        self.top_line.sigPositionChanged.connect(self._guide_dragged)
        self.bot_line.sigPositionChanged.connect(self._guide_dragged)
        self.depth_line.sigPositionChanged.connect(self._depth_dragged)

        # axial view
        self.ax = pg.PlotWidget()
        self.ax_vb = self.ax.getPlotItem().getViewBox()
        self.ax_vb.invertY(True)
        self.ax_vb.setAspectLocked(True)
        self.ax_vb.setMouseMode(pg.ViewBox.PanMode)
        self.ax.hideAxis("left")
        self.ax.setLabel("bottom", "across (mm)")
        self.ax_img = pg.ImageItem()
        self.ax.addItem(self.ax_img)
        self.ax_mask = pg.ImageItem()
        self.ax_mask.setLookupTable(lut)
        self.ax_mask.setLevels((0, 1))
        self.ax.addItem(self.ax_mask)
        self.ax_title = QtWidgets.QLabel("axial")
        self.roi = pg.CircleROI((0, 0), radius=10, pen=pg.mkPen(CIRCLE_COLOR, width=2), movable=True)
        self.roi.setZValue(20)
        self.roi.setVisible(False)
        self.ax.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self._roi_moved)
        self.frame_line = pg.InfiniteLine(angle=0, movable=True, pen=dash, hoverPen=pg.mkPen(DEPTH_COLOR, width=3))
        self.ax.addItem(self.frame_line)
        self.frame_line.sigPositionChanged.connect(self._frame_dragged)

        self.prof_w = pg.PlotWidget()
        self.prof_vb = self.prof_w.getPlotItem().getViewBox()
        self.prof_vb.invertY(True)
        self.prof_w.getPlotItem().setYLink(self.cor.getPlotItem())
        self.prof_w.hideAxis("left")
        self.prof_w.setLabel("bottom", "mean HU in mask")
        self.prof_w.showGrid(x=True, y=True, alpha=0.3)
        self.prof_curve = self.prof_w.plot(pen=pg.mkPen("w", width=1.5))
        self.prof_depth = pg.InfiniteLine(angle=0, pen=dash)
        self.prof_w.addItem(self.prof_depth)
        self.prof_title = QtWidgets.QLabel("profile")
        prof_box = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(prof_box)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.prof_title)
        pl.addWidget(self.prof_w, 1)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.cor_title)
        ll.addWidget(self.cor, 1)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.ax_title)
        rl.addWidget(self.ax, 1)

        sp = QtWidgets.QSplitter()
        sp.addWidget(scroll)
        sp.addWidget(left)
        sp.addWidget(prof_box)
        sp.addWidget(right)
        sp.setStretchFactor(1, 3)
        sp.setStretchFactor(2, 2)
        sp.setStretchFactor(3, 2)
        sp.setSizes([380, 300, 260, 460])
        self.setCentralWidget(sp)

        for key, delta in (("[", -1), ("]", 1), ("PgUp", -1), ("PgDown", 1)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(lambda d=delta: self.set_frame(self.frame + d) if self.series else None)

    # ------------------------------------------------------------------ loading
    def open_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder with the stitched coronal MPR DICOM series")
        if not d:
            return
        self.statusBar().showMessage(f"reading {d} ...")
        QtWidgets.QApplication.processEvents()
        try:
            self.load_series_obj(load_series(d))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Open", str(e))

    def load_series_obj(self, s: MprSeries, sync: bool = False) -> None:
        self.series = s
        nz, nr, nc = s.volume.shape
        self.info.setText(
            f"{s.path.name}\n{nz} coronal slices of {nr} x {nc} px\n"
            f"spacing: slice {self.dz:.2f} mm, pixel {self.dr:.3f} x {self.dc:.3f} mm\nlength {nr * self.dr / 10:.1f} cm"
        )
        self._busy = True
        self.z_spin.setRange(0, nz - 1)
        self.depth_spin.setRange(0, self.extent_cm)
        self._busy = False
        self.bounds_source = "auto"
        self.frame = -1
        self.mask3d = None
        self.profile = None
        self.t_air, self.t_split = auto_thresholds(s.volume)
        self._busy = True
        self.cyl_box.setChecked(False)
        self._busy = False
        self._roi_ready = False
        self._path, self._off = None, (0.0, 0.0)
        self._busy = True
        self.follow_box.setChecked(False)
        self._busy = False
        self.roi.setVisible(False)
        self.cyl_label.setText("")
        self._set_threshold(self.t_split, recompute=False)
        self._busy = True
        self.clast_box.setChecked(False)
        self.void_box.setChecked(False)
        self._busy = False
        self.c_line.setVisible(False)
        self._set_clast(auto_clast_threshold(s.volume, self.t_split), recompute=False)
        self.set_frame(nz // 2, first=True)
        self.zoom_fit()
        self.recompute_mask(sync=sync)
        self.statusBar().showMessage("loaded")

    # ------------------------------------------------------------------ slices
    def set_frame(self, k: int, first: bool = False) -> None:
        if self.series is None or self._busy:
            return
        k = int(np.clip(k, 0, self.n_frames - 1))
        if k == self.frame and not first:
            return
        self.frame = k
        self._busy = True
        self.z_spin.setValue(k)
        self.frame_line.setValue((k + 0.5) * self.dz)
        self._busy = False
        self.z_label.setText(f"Coronal slice: {k} of {self.n_frames - 1}")
        img = self.series.volume[k]
        if first:
            mask, vals = core_values(img)
            self._hist_values = vals
            rows = np.where(mask.any(1))[0]
            r0, r1 = (int(rows[0]), int(rows[-1])) if rows.size else (0, img.shape[0] - 1)
            top, bot = auto_bounds(mask, (0, img.shape[0], 0, img.shape[1]), self.dr)
            self.auto_ends = (top, bot)
            self.top_cm, self.bottom_cm = clamp_insets(top, bot, self.extent_cm)
            self.set_depth_mm((r0 + r1) / 2 * self.dr, refresh=False)
            self.auto_window(redraw=False)
            self._draw_histogram()
            self._ends_to_view()
        self.cor_img.setImage(img, levels=self.win)
        self.cor_img.setRect(QtCore.QRectF(0, 0, img.shape[1] * self.dc, img.shape[0] * self.dr))
        self.cor_title.setText(f"Coronal slice {k}")
        self._refresh_cor_mask()
        self._refresh_axial()

    def set_depth_mm(self, mm: float, refresh: bool = True) -> None:
        if self.series is None:
            return
        mm = float(np.clip(mm, 0, (self.series.volume.shape[1] - 1) * self.dr))
        self.depth_mm = mm
        was, self._busy = self._busy, True
        self.depth_line.setValue(mm)
        self.prof_depth.setValue(mm)
        self.depth_spin.setValue(mm / 10)
        self._busy = was
        if refresh:
            self._refresh_axial()

    def _row(self) -> int:
        return int(np.clip(round(self.depth_mm / self.dr), 0, self.series.volume.shape[1] - 1))

    def _roi_to_depth(self) -> None:
        """When following the wobble the circle sits where the axis is at the depth shown."""
        if self._path is None or not self._roi_ready:
            return
        i = self._row()
        r = float(self.roi.size()[0]) / 2
        was, self._busy = self._busy, True
        self.roi.setPos((float(self._path[0][i]) + self._off[0] - r, float(self._path[1][i]) + self._off[1] - r))
        self._busy = was

    def _refresh_axial(self) -> None:
        if self.series is None:
            return
        self._roi_to_depth()
        row = int(round(self.depth_mm / self.dr))
        a = axial_image(self.series.volume, row, self.dz, self.dc)
        self.ax_img.setImage(a, levels=self.win)
        self.ax_img.setRect(QtCore.QRectF(0, 0, a.shape[1] * self.dc, a.shape[0] * self.dc))
        was, self._busy = self._busy, True
        self.frame_line.setValue((self.frame + 0.5) * self.dz)
        self._busy = was
        self.ax_title.setText(f"Axial slice at {self.depth_mm / 10:.2f} cm (row {row})")
        if self.mask3d is not None:
            f = self.dz / self.dc
            am = self.mask3d[:, row, :].astype(np.float32)
            am = am if abs(f - 1) < 1e-3 else ndi.zoom(am, (f, 1.0), order=1, mode="nearest", grid_mode=True)
            self.ax_mask.setImage(am > 0.5, levels=(0, 1))
            self.ax_mask.setRect(QtCore.QRectF(0, 0, am.shape[1] * self.dc, am.shape[0] * self.dc))
            self.ax_mask.setVisible(self.show_mask.isChecked())
        else:
            self.ax_mask.setVisible(False)

    def _depth_dragged(self) -> None:
        if not self._busy:
            self.set_depth_mm(self.depth_line.value())

    def _frame_dragged(self) -> None:
        if not self._busy and self.series is not None:
            self.set_frame(int(np.floor(self.frame_line.value() / self.dz)))

    # ------------------------------------------------------------------ greyscale window
    def auto_window(self, redraw: bool = True) -> None:
        if self.series is None:
            return
        v = self._hist_values
        self.win = (float(np.percentile(v, 1)), float(np.percentile(v, 99)))
        self.hist_range = histogram_range(v)
        self._window_to_widgets()
        if redraw:
            self._draw_histogram()
            self._apply_window()

    def _window_to_widgets(self) -> None:
        lo, hi = self.hist_range
        self._busy = True
        for w in (self.black, self.white):
            w.set_range(lo, hi)
        self.black.set_value(self.win[0])
        self.white.set_value(self.win[1])
        self.region.setRegion(self.win)
        self.win_label.setText(f"[{self.win[0]:.0f}, {self.win[1]:.0f}] HU")
        self._busy = False

    def _hist_view(self) -> None:
        """Histogram x range: the greyscale range, widened so the orange threshold line is always on screen."""
        lo, hi = self.hist_range
        pad = 0.03 * (hi - lo)
        ts = [self.t_sed] + ([self.t_clast] if self.clast_box.isChecked() else [])
        self.hist.setXRange(min([lo] + [t - pad for t in ts]), max([hi] + [t + pad for t in ts]), padding=0)

    def _draw_histogram(self) -> None:
        lo, hi = self.hist_range
        counts, edges = np.histogram(self._hist_values, bins=128, range=(lo, hi))
        centres = 0.5 * (edges[:-1] + edges[1:])
        self.hist_bars.setOpts(x=centres, height=counts / max(counts.max(), 1), width=(hi - lo) / 128)
        self._hist_view()
        self.hist.setYRange(0, 1.05, padding=0)
        self.region.setBounds((lo, hi))

    def _region_moved(self) -> None:
        if self._busy:
            return
        lo, hi = self.region.getRegion()
        self.win = (float(lo), float(max(hi, lo + 1)))
        self._busy = True
        self.black.set_value(self.win[0])
        self.white.set_value(self.win[1])
        self._busy = False
        self.win_label.setText(f"[{self.win[0]:.0f}, {self.win[1]:.0f}] HU")
        self._apply_window()

    def _window_from_boxes(self) -> None:
        if self._busy:
            return
        lo, hi = self.black.value(), self.white.value()
        if hi < lo:
            lo, hi = hi, lo
        self.win = (lo, max(hi, lo + 1))
        self._busy = True
        self.region.setRegion(self.win)
        self._busy = False
        self.win_label.setText(f"[{self.win[0]:.0f}, {self.win[1]:.0f}] HU")
        self._apply_window()

    def _apply_window(self) -> None:
        self.cor_img.setLevels(self.win)
        self.ax_img.setLevels(self.win)

    # ------------------------------------------------------------------ core ends
    def _ends_to_view(self) -> None:
        """Put the guide lines, the boxes and the length text at the current insets."""
        was, self._busy = self._busy, True
        self.top_line.setValue(self.top_cm * 10)
        self.bot_line.setValue((self.extent_cm - self.bottom_cm) * 10)
        self.top_edit.setMaximum(self.extent_cm)
        self.bot_edit.setMaximum(self.extent_cm)
        self.top_edit.setValue(self.top_cm)
        self.bot_edit.setValue(self.bottom_cm)
        self._busy = was
        length = max(self.extent_cm - self.top_cm - self.bottom_cm, 0)
        self.len_label.setText(f"Core length: {length:.2f} cm ({self.bounds_source})")

    def _ends_from_boxes(self) -> None:
        if self._busy or self.series is None:
            return
        self.top_cm, self.bottom_cm = clamp_insets(self.top_edit.value(), self.bot_edit.value(), self.extent_cm)
        self.bounds_source = "manual"
        self._ends_to_view()

    def _guide_dragged(self) -> None:
        if self._busy or self.series is None:
            return
        top = self.top_line.value() / 10
        bot = self.extent_cm - self.bot_line.value() / 10
        self.top_cm, self.bottom_cm = clamp_insets(top, bot, self.extent_cm)
        self.bounds_source = "manual"
        self._ends_to_view()

    def reset_ends(self) -> None:
        if self.series is None:
            return
        self.top_cm, self.bottom_cm = clamp_insets(*self.auto_ends, self.extent_cm)
        self.bounds_source = "auto"
        self._ends_to_view()

    def _guides_visible(self, on: bool) -> None:
        self.top_line.setVisible(on)
        self.bot_line.setVisible(on)

    def goto_end(self, which: str) -> None:
        """Centre the coronal view on one end of the core, zoomed in enough to place the guide."""
        if self.series is None:
            return
        y = self.top_cm * 10 if which == "top" else (self.extent_cm - self.bottom_cm) * 10
        w = self.series.volume.shape[2] * self.dc
        self.cor_vb.setRange(xRange=(0, w), yRange=(y - 0.35 * w, y + 0.35 * w), padding=0)
        self.set_depth_mm(y)

    # ------------------------------------------------------------------ sediment mask and profile
    def _set_threshold(self, t: float, recompute: bool = True) -> None:
        was, self._busy = self._busy, True
        self.t_sed = float(t)
        self.t_spin.setValue(self.t_sed)
        self.t_line.setValue(self.t_sed)
        self._busy = was
        self._hist_view()
        if recompute:
            self._mask_params_changed()

    def _threshold_from_spin(self, t: float) -> None:
        if not self._busy:
            self._set_threshold(t)

    def _threshold_dragged(self) -> None:
        if not self._busy:
            self._set_threshold(self.t_line.value())

    def auto_threshold(self) -> None:
        if self.series is not None:
            self.t_air, self.t_split = auto_thresholds(self.series.volume)
            self._set_threshold(self.t_split)

    def _set_clast(self, t: float, recompute: bool = True) -> None:
        was, self._busy = self._busy, True
        self.t_clast = float(t)
        self.c_spin.setValue(self.t_clast)
        self.c_line.setValue(self.t_clast)
        self._busy = was
        self._hist_view()
        if recompute and self.clast_box.isChecked():
            self._mask_params_changed()

    def _clast_from_spin(self, t: float) -> None:
        if not self._busy:
            self._set_clast(t)

    def _clast_dragged(self) -> None:
        if not self._busy:
            self._set_clast(self.c_line.value())

    def auto_clast(self) -> None:
        if self.series is not None:
            self._set_clast(auto_clast_threshold(self.series.volume, self.t_sed))

    def _clast_toggled(self, on: bool) -> None:
        self.c_line.setVisible(on)
        self._hist_view()
        self._mask_params_changed()

    def _mask_params_changed(self) -> None:
        """Quick: redraw this slice's mask now; the whole stack and the profile follow after a short pause."""
        if self.series is None or self._busy:
            return
        self.mask3d = None
        self.mask_label.setText("mask: recomputing ...")
        self._refresh_cor_mask()
        self._timer.start()

    def _mask_args(self) -> tuple:
        """(sediment threshold, margin mm, fill holes, circle, clast threshold or None)."""
        clast = self.t_clast if self.clast_box.isChecked() else None
        return self.t_sed, self.margin_spin.value(), not self.void_box.isChecked(), self._circle(), clast

    def _refresh_cor_mask(self) -> None:
        if self.series is None:
            return
        if self.mask3d is not None:
            m = self.mask3d[self.frame]
        else:
            t, margin, fill, circle, clast = self._mask_args()
            m = mask_slice(self.series.volume[self.frame], self.frame, t, margin, self.dz, self.dc, fill, circle, clast)
        self.cor_mask.setImage(m, levels=(0, 1))
        self.cor_mask.setRect(QtCore.QRectF(0, 0, m.shape[1] * self.dc, m.shape[0] * self.dr))
        self.cor_mask.setVisible(self.show_mask.isChecked())

    def _mask_visible(self, on: bool) -> None:
        self.cor_mask.setVisible(on)
        self.ax_mask.setVisible(on and self.mask3d is not None)

    def recompute_mask(self, sync: bool = False) -> None:
        """Mask for every slice and the mean-per-row profile over it (worker thread unless ``sync``)."""
        if self.series is None:
            return
        vol, dc = self.series.volume, self.dc
        t, margin, fill, circle, clast = self._mask_args()
        dz = self.dz
        self._mask_gen += 1
        gen = self._mask_gen

        def work():
            m = mask_3d(vol, t, margin, dc, fill, circle, dz, clast)
            return m, row_profile(vol, m)

        if sync:
            self._mask_done(gen, work())
            return
        job = _Job(gen, work)
        job.sig.done.connect(self._mask_done)
        job.sig.failed.connect(lambda g, tb: self.mask_label.setText("mask failed: " + tb.strip().splitlines()[-1]))
        self._jobs = [j for j in self._jobs if j is not job][-3:] + [job]
        self._pool.start(job)

    def _mask_done(self, gen: int, res) -> None:
        if gen != self._mask_gen:
            return  # a newer request is running
        self.mask3d, self.profile = res
        mean, sd, cnt = self.profile
        rows = (np.arange(len(mean)) + 0.5) * self.dr
        self.prof_curve.setData(mean, rows)
        n_ok = int(np.isfinite(mean).sum())
        self.prof_title.setText(f"Mean HU per depth ({n_ok} of {len(mean)} rows)")
        self.mask_label.setText(f"mask: {100 * self.mask3d.mean():.1f}% of the volume, {int(cnt.max())} voxels in the fullest row")
        self._refresh_cor_mask()
        self._refresh_axial()

    # ------------------------------------------------------------------ export
    def profile_meta(self) -> dict:
        """Everything that defines the profile: the input file and every choice, as nested plain values."""
        import datetime
        from importlib import metadata

        s = self.series
        try:
            version = "core-stitch " + metadata.version("core-stitch")
        except metadata.PackageNotFoundError:
            version = "core-stitch (unknown version)"
        circle = self._circle_any() if self._roi_ready else None
        on = self.cyl_box.isChecked() and self._roi_ready
        cyl: dict = {"on": on, "liner_wall_mm": round(self.liner_spin.value(), 3)}
        if on:
            cyl["radius_mm"] = round(circle[2], 3)
            cyl["follow_axis"] = self._path is not None
            if self._path is None:
                cyl["centre_across_mm"], cyl["centre_along_stack_mm"] = round(circle[0], 3), round(circle[1], 3)
            else:
                cyl["axis_offset_dragged_mm"] = [round(self._off[0], 3), round(self._off[1], 3)]
                cyl["axis_range_across_mm"] = round(float(np.ptp(self._path[0])), 3)
                cyl["axis_range_along_stack_mm"] = round(float(np.ptp(self._path[1])), 3)
            cyl["air_threshold_used_for_fit_hu"] = round(self.t_air, 2)
        return {
            "saved_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "software": "core-profile viewer, " + version,
            "input": {
                "folder": str(s.path),
                "series_description": s.series_description,
                "series_uid": s.series_uid,
                "n_slices": int(s.volume.shape[0]),
                "rows": int(s.volume.shape[1]),
                "cols": int(s.volume.shape[2]),
                "slice_spacing_mm": round(self.dz, 4),
                "row_spacing_mm": round(self.dr, 4),
                "col_spacing_mm": round(self.dc, 4),
                "units": "HU",
            },
            "core_ends": {
                "top_inset_cm": round(self.top_cm, 3),
                "bottom_inset_cm": round(self.bottom_cm, 3),
                "set_by": self.bounds_source,
                "core_length_cm": round(self.extent_cm - self.top_cm - self.bottom_cm, 3),
                "image_length_cm": round(self.extent_cm, 3),
            },
            "mask": {
                "sediment_threshold_hu": round(self.t_sed, 2),
                "rule": "per coronal slice: pixels at or above the threshold" + (" inside the cylinder; patches of at least %d pixels" % MIN_BLOB_PX if on else "; biggest blob only"),
                "shrink_from_edges_mm": round(self.margin_spin.value(), 3),
                "remove_air_voids": self.void_box.isChecked(),
                "remove_clasts": self.clast_box.isChecked(),
                "clast_threshold_hu": round(self.t_clast, 2) if self.clast_box.isChecked() else None,
                "cylinder": cyl,
            },
            "profile": {
                "statistic": "mean and SD of masked voxels per row, over all coronal slices",
                "min_voxels_per_row": MIN_ROW_VOXELS,
            },
        }

    def save_profile(self, path: str | Path | None = None) -> Path | None:
        """Write the profile CSV (header with input and settings, then one line per row between the core ends)."""
        if self.series is None or self.profile is None or self.mask3d is None:
            self.save_label.setText("Nothing to save yet: the mask is still being computed (wait for it, then try again).")
            return None
        if path is None:
            default = self.series.path.parent / f"{self.series.path.name}_profile.csv"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save profile", str(default), "CSV (*.csv)")
            if not path:
                return None
        mean, sd, cnt = self.profile
        axis = None
        if self._path is not None and self.cyl_box.isChecked():
            axis = (self._path[0] + self._off[0], self._path[1] + self._off[1])
        try:
            n = write_profile_csv(path, mean, sd, cnt, self.dr, self.top_cm * 10, (self.extent_cm - self.bottom_cm) * 10, self.profile_meta(), axis)
        except OSError as e:
            self.save_label.setText(f"Could not save: {e}")
            return None
        self.save_label.setText(f"Saved {n} rows to {path}")
        return Path(path)

    # ------------------------------------------------------------------ cylinder
    def _circle(self) -> tuple[float, float, float] | None:
        """(centre across, centre along the stack, radius) in mm of the yellow circle, or None when it is off."""
        if not self.cyl_box.isChecked() or not self._roi_ready:
            return None
        cx, cy, r = self._circle_any()
        if self._path is not None:  # one centre per row: the fitted axis plus however far the user moved the circle
            return self._path[0] + self._off[0], self._path[1] + self._off[1], r
        return cx, cy, r

    def _circle_any(self) -> tuple[float, float, float]:
        pos, size = self.roi.pos(), self.roi.size()
        r = float(size[0]) / 2
        return float(pos[0]) + r, float(pos[1]) + r, r

    def _circle_text(self, r: float) -> None:
        follow = ", following the axis" if self._path is not None else ""
        self.cyl_label.setText(f"circle: radius {r:.1f} mm{follow}; the liner wall ({self.liner_spin.value():.2f} mm) lies just outside it")

    def _set_circle(self, cx: float, cy: float, r: float) -> None:
        was, self._busy = self._busy, True
        self.roi.setPos((cx - r, cy - r))
        self.roi.setSize((2 * r, 2 * r))
        self._busy = was
        self._roi_ready = True
        self._circle_text(r)

    def auto_fit(self) -> bool:
        """Fit the circle to the outer edge of the liner at the current axial cut, then take off the liner wall."""
        if self.series is None:
            return False
        row = self._row()
        if self.follow_box.isChecked():
            return self._fit_path(row)
        res = fit_outer_circle(self.series.volume, row, self.t_air, self.dz, self.dc)
        if res is None:
            self.cyl_label.setText("Auto-fit: no circle found here - put the axial cut across the core")
            return False
        cx, cy, r, _n = res
        self._path, self._off = None, (0.0, 0.0)
        self._set_circle(cx, cy, max(r - self.liner_spin.value(), 1.0))
        return True

    def _fit_path(self, row: int) -> bool:
        """Fit the axis at many depths between the core-end guides; the circle then follows it."""
        nr = self.series.volume.shape[1]
        r0 = int(round(self.top_cm * 10 / self.dr))
        r1 = nr - 1 - int(round(self.bottom_cm * 10 / self.dr))
        if r1 - r0 < 10:
            r0, r1 = 0, nr - 1
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            res = fit_axis_path(self.series.volume, self.t_air, self.dz, self.dr, self.dc, r0, r1)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if res is None:
            self.cyl_label.setText("Follow: could not fit the core at enough depths")
            return False
        cx_r, cy_r, r_out, n = res
        self._path, self._off = (cx_r, cy_r), (0.0, 0.0)
        seg = slice(r0, r1 + 1)
        r_in = float(self.roi.size()[0]) / 2 if self._roi_ready else max(r_out - self.liner_spin.value(), 1.0)
        self._set_circle(float(cx_r[row]), float(cy_r[row]), r_in)
        self.cyl_label.setText(
            f"circle: radius {r_in:.1f} mm, following the axis (fitted at {n} depths: "
            f"{np.ptp(cx_r[seg]):.1f} mm across, {np.ptp(cy_r[seg]):.1f} mm through the stack)"
        )
        return True

    def _follow_toggled(self, on: bool) -> None:
        if self._busy or self.series is None:
            return
        if on:
            if not self.auto_fit():
                was, self._busy = self._busy, True
                self.follow_box.setChecked(False)
                self._busy = was
                return
            was, self._busy = self._busy, True
            self.cyl_box.setChecked(True)
            self._busy = was
            self._cyl_state()
        else:  # keep the circle where it is now, fixed for every depth
            self._path, self._off = None, (0.0, 0.0)
            self._circle_text(self._circle_any()[2])
            self._mask_params_changed()

    def _autofit_clicked(self) -> None:
        if self.series is None or not self.auto_fit():
            return
        was, self._busy = self._busy, True
        self.cyl_box.setChecked(True)
        self._busy = was
        self._cyl_state()

    def _cyl_toggled(self, on: bool) -> None:
        if self._busy or self.series is None:
            return
        if on and not self._roi_ready and not self.auto_fit():
            nc = self.series.volume.shape[2]
            width, length = nc * self.dc, (self.n_frames - 1) * self.dz
            self._set_circle(width / 2, length / 2, 0.35 * min(width, length))
        self._cyl_state()

    def _cyl_state(self) -> None:
        """Show or hide the circle and redo the mask. The threshold is left alone: the cylinder is an extra clip."""
        self.roi.setVisible(self.cyl_box.isChecked() and self._roi_ready)
        self._mask_params_changed()

    def _liner_changed(self, v: float) -> None:
        old, self._liner_prev = self._liner_prev, float(v)
        if self.series is None or self._busy or not self._roi_ready:
            return
        cx, cy, r = self._circle_any()
        self._set_circle(cx, cy, max(r - (float(v) - old), 1.0))
        self._mask_params_changed()

    def _roi_moved(self) -> None:
        if self._busy:
            return
        cx, cy, r = self._circle_any()
        if self._path is not None:
            i = self._row()
            self._off = (cx - float(self._path[0][i]), cy - float(self._path[1][i]))
        self._circle_text(r)
        self._mask_params_changed()

    # ------------------------------------------------------------------ zoom
    def zoom_by(self, f: float) -> None:
        self.cor_vb.scaleBy((1 / f, 1 / f))

    def zoom_fit(self) -> None:
        if self.series is None:
            return
        nr, nc = self.series.volume.shape[1:]
        self.cor_vb.setRange(xRange=(0, nc * self.dc), yRange=(0, nr * self.dr), padding=0.02)
        na = max(1, int(round((self.n_frames - 1) * self.dz / self.dc)))
        self.ax_vb.setRange(xRange=(0, nc * self.dc), yRange=(0, na * self.dc), padding=0.05)

    def zoom_100(self) -> None:
        """One screen pixel per image pixel, centred on the current view."""
        if self.series is None:
            return
        (x0, x1), (y0, y1) = self.cor_vb.viewRange()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = self.cor.width() * self.dc, self.cor.height() * self.dr
        self.cor_vb.setRange(xRange=(cx - w / 2, cx + w / 2), yRange=(cy - h / 2, cy + h / 2), padding=0)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer()
    w.show()
    if len(sys.argv) > 1:
        w.load_series_obj(load_series(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
