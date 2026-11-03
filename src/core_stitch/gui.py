"""Coron TOP/BTM stitch GUI."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from core_stitch.analyze import LINER_PRESETS
from core_stitch.export import export_accept
from core_stitch.overlap import FUSE_MODES
from core_stitch.pipeline import (
    StitchState,
    build_sediment_volume_mask,
    coarse_align,
    detect_sediment_roi,
    ensure_downcore_profile,
    ensure_posed,
    invalidate_downcore_profile,
    masked_hu_values,
    preview_arrays,
    preview_xz_arrays,
    rebuild_sediment_mask_2d,
    recompute_overlap,
    refine_join,
    set_auto_window,
    set_flip,
    set_nudge,
    set_roll,
    set_view_z,
    window_limits,
)


class StitchGui(tk.Tk):
    def __init__(
        self,
        path_a: Path | None = None,
        path_b: Path | None = None,
    ) -> None:
        super().__init__()
        self.title("core-stitch — Coron TOP/BTM")
        self.geometry("1400x900")
        self.minsize(1000, 640)

        self.path_a: Path | None = path_a
        self.path_b: Path | None = path_b
        self.state: StitchState | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._full_im: Image.Image | None = None
        self._refresh_job: str | None = None
        self._nudge_job: str | None = None
        self._roll_job: str | None = None

        self.status = tk.StringVar(
            value="Select TOP and BTM series folders, then Run coarse align."
        )
        self.path_a_var = tk.StringVar(value=str(path_a) if path_a else "")
        self.path_b_var = tk.StringVar(value=str(path_b) if path_b else "")
        self.metrics = tk.StringVar(value="—")
        self.nudge_label = tk.StringVar(value="B nudge X/Y: 0.0 / 0.0 mm")
        self.nudge_range_label = tk.StringVar(value="Slider limits ±80 mm")
        self.zoom_label = tk.StringVar(value="Zoom: 100%")

        self.flip_var = tk.BooleanVar(value=False)
        self.roll_var = tk.DoubleVar(value=0.0)
        self.align_var = tk.BooleanVar(value=True)
        self.fuse_mode = tk.StringVar(value="max")
        self.match_hu_var = tk.BooleanVar(value=False)
        self.opacity_a = tk.DoubleVar(value=0.75)
        self.opacity_b = tk.DoubleVar(value=0.75)
        self.nudge_x = tk.DoubleVar(value=0.0)
        self.nudge_y = tk.DoubleVar(value=0.0)
        self.nudge_range = tk.DoubleVar(value=80.0)
        self.zoom_var = tk.DoubleVar(value=1.0)  # multiplier on fit-to-window
        self.z_var = tk.IntVar(value=0)
        self.z_label = tk.StringVar(value="Slice: —")
        self.crop_core_var = tk.BooleanVar(value=True)
        self.export_jpeg = tk.BooleanVar(value=True)
        self.export_tiff = tk.BooleanVar(value=True)
        self.export_dicom = tk.BooleanVar(value=True)
        self.export_npy = tk.BooleanVar(value=True)
        self.export_json = tk.BooleanVar(value=True)
        self.export_sediment_mask = tk.BooleanVar(value=False)
        self.export_sediment_roi = tk.BooleanVar(value=False)
        self.show_sediment = tk.BooleanVar(value=True)
        self.show_downcore = tk.BooleanVar(value=False)
        self.exclude_low = tk.BooleanVar(value=True)
        self.exclude_high = tk.BooleanVar(value=True)
        self.sed_erode = tk.DoubleVar(value=0.0)
        self.sed_nudge_x = tk.DoubleVar(value=0.0)
        self.sed_nudge_z = tk.DoubleVar(value=0.0)
        self.thr_lo = tk.DoubleVar(value=-200.0)
        self.thr_hi = tk.DoubleVar(value=800.0)
        self.sed_label = tk.StringVar(value="Sediment ROI: not detected")
        self.filter_label = tk.StringVar(value="HU filter: not run")
        self.liner_preset = tk.StringVar(value="Auto (no prior)")
        self.liner_od = tk.DoubleVar(value=70.0)
        self.liner_wall = tk.DoubleVar(value=3.0)
        self._thr_job: str | None = None
        self._ignore_thr_slide = False
        self.xz_y_frac = tk.DoubleVar(value=0.5)
        self.xz_label = tk.StringVar(value="XZ: idle (builds after Z stack)")
        self.win_lo = tk.DoubleVar(value=0.0)
        self.win_hi = tk.DoubleVar(value=1.0)
        self.win_label = tk.StringVar(value="Window: —")
        self._z_job: str | None = None
        self._win_job: str | None = None
        self._sed_job: str | None = None
        self._xz_job: str | None = None
        self._downcore_job: str | None = None
        self._downcore_recompute_otsu: bool = False
        self._z_load_job: str | None = None
        self._hist_range: tuple[float, float] = (0.0, 1.0)
        self._hist_counts: np.ndarray | None = None
        self._hist_edges: np.ndarray | None = None
        self._photo_xz: ImageTk.PhotoImage | None = None

        self._pan_start: tuple[int, int] | None = None
        self._build()
        self._bind_keys()
        if path_a and path_b:
            self.after(200, self.run_coarse)

    def _build(self) -> None:
        self._hist_drag_which: str | None = None

        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # ----- Left control rail -----
        left = ttk.Frame(root, padding=(8, 8, 4, 4), width=360)
        left.grid(row=0, column=0, sticky="nsw")
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="core-stitch", font=("", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        nb = ttk.Notebook(left)
        nb.grid(row=1, column=0, sticky="nsew")
        tab_align = ttk.Frame(nb, padding=8)
        tab_display = ttk.Frame(nb, padding=8)
        tab_analyze = ttk.Frame(nb, padding=8)
        tab_export = ttk.Frame(nb, padding=8)
        nb.add(tab_align, text="Align")
        nb.add(tab_display, text="Display")
        nb.add(tab_analyze, text="Analyze")
        nb.add(tab_export, text="Export")
        self._notebook = nb

        self._build_align_tab(tab_align)
        self._build_display_tab(tab_display)
        self._build_analyze_tab(tab_analyze)
        self._build_export_tab(tab_export)

        ttk.Label(left, textvariable=self.status, wraplength=340).grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )

        # ----- Right image pane (coronal + compact XZ) -----
        canvas_frame = ttk.Frame(root, padding=(4, 8, 8, 8))
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.rowconfigure(2, weight=0)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, background="#111", highlightthickness=0)
        self.vbar = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.hbar = ttk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        self.canvas.configure(
            xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")

        self.xz_frame = ttk.LabelFrame(
            canvas_frame, text="XZ cross-section (sediment cylinder)", padding=4
        )
        self.xz_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.xz_frame.columnconfigure(0, weight=1)
        ttk.Label(self.xz_frame, textvariable=self.xz_label, font=("", 8)).grid(
            row=0, column=0, sticky="w"
        )
        self.xz_canvas = tk.Canvas(
            self.xz_frame, background="#111", highlightthickness=0, height=160
        )
        self.xz_canvas.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        yrow = ttk.Frame(self.xz_frame)
        yrow.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(yrow, text="Y cut", font=("", 8)).pack(side=tk.LEFT)
        ttk.Scale(
            yrow,
            from_=0.05,
            to=0.95,
            variable=self.xz_y_frac,
            command=self._on_xz_y_slide,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(yrow, text="← top · btm →", font=("", 8)).pack(side=tk.LEFT)

        self.canvas.bind("<Configure>", lambda _e: self._schedule_refresh())
        self.xz_canvas.bind("<Configure>", lambda _e: self._schedule_xz_refresh())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_mousewheel_linux(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._on_mousewheel_linux(e, -1))
        self.canvas.bind("<ButtonPress-2>", self._pan_press)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<ButtonPress-1>", self._alt_pan_press)
        self.canvas.bind("<B1-Motion>", self._alt_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._alt_pan_release)

    def _build_align_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        ttk.Button(parent, text="Open TOP (A)…", command=self.browse_a).pack(
            fill=tk.X, pady=(0, 4)
        )
        ttk.Button(parent, text="Open BTM (B)…", command=self.browse_b).pack(
            fill=tk.X, pady=(0, 4)
        )
        ttk.Label(parent, textvariable=self.path_a_var, wraplength=320).pack(
            anchor=tk.W
        )
        ttk.Label(parent, textvariable=self.path_b_var, wraplength=320).pack(
            anchor=tk.W, pady=(0, 8)
        )

        ttk.Button(parent, text="Run coarse align", command=self.run_coarse).pack(
            fill=tk.X, pady=(0, 4)
        )
        ttk.Button(parent, text="Refine join", command=self.run_refine).pack(
            fill=tk.X, pady=(0, 8)
        )

        ttk.Checkbutton(
            parent,
            text="Flip B (end-for-end)",
            variable=self.flip_var,
            command=self._on_flip,
        ).pack(anchor=tk.W)

        roll = ttk.Frame(parent)
        roll.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(roll, text="Roll °").pack(side=tk.LEFT)
        ttk.Scale(
            roll,
            from_=-20,
            to=20,
            variable=self.roll_var,
            command=self._on_roll_slide,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(roll, text="0", width=3, command=self._roll_zero).pack(side=tk.LEFT)
        ttk.Button(parent, text="Re-run overlap", command=self._rerun_overlap).pack(
            fill=tk.X, pady=(6, 8)
        )

        nudge = ttk.LabelFrame(parent, text="B nudge (mm)", padding=6)
        nudge.pack(fill=tk.X)
        ttk.Label(nudge, textvariable=self.nudge_label).pack(anchor=tk.W)
        nr = ttk.Frame(nudge)
        nr.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(nr, text="X").pack(side=tk.LEFT)
        self.nudge_x_scale = ttk.Scale(
            nr,
            from_=-80,
            to=80,
            variable=self.nudge_x,
            command=self._on_nudge_slide,
        )
        self.nudge_x_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(nr, text="Y").pack(side=tk.LEFT)
        self.nudge_y_scale = ttk.Scale(
            nr,
            from_=-80,
            to=80,
            variable=self.nudge_y,
            command=self._on_nudge_slide,
        )
        self.nudge_y_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(nudge, text="Reset nudge", command=self._reset_nudge).pack(
            anchor=tk.W, pady=(4, 0)
        )
        nrr = ttk.Frame(nudge)
        nrr.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(nrr, textvariable=self.nudge_range_label).pack(side=tk.LEFT)
        ttk.Button(nrr, text="Widen", command=self._widen_nudge_range).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Label(parent, text="Arrows = 1 mm (Shift = 5)", font=("", 8)).pack(
            anchor=tk.W, pady=(4, 0)
        )

        ttk.Label(parent, textvariable=self.metrics, wraplength=320).pack(
            anchor=tk.W, pady=(10, 0)
        )

    def _build_display_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            parent,
            text="Alignment colors (A=magenta, B=green)",
            variable=self.align_var,
            command=self._schedule_refresh,
        ).pack(anchor=tk.W)

        op = ttk.Frame(parent)
        op.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(op, text="Opacity A").pack(anchor=tk.W)
        ttk.Scale(
            op,
            from_=0.0,
            to=1.0,
            variable=self.opacity_a,
            command=lambda _v: self._schedule_refresh(),
        ).pack(fill=tk.X)
        ttk.Label(op, text="Opacity B").pack(anchor=tk.W, pady=(4, 0))
        ttk.Scale(
            op,
            from_=0.0,
            to=1.0,
            variable=self.opacity_b,
            command=lambda _v: self._schedule_refresh(),
        ).pack(fill=tk.X)

        fuse = ttk.LabelFrame(parent, text="Overlap fuse", padding=6)
        fuse.pack(fill=tk.X, pady=(10, 0))
        self.fuse_combo = ttk.Combobox(
            fuse,
            textvariable=self.fuse_mode,
            values=list(FUSE_MODES),
            width=12,
            state="readonly",
        )
        self.fuse_combo.pack(fill=tk.X)
        self.fuse_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_fuse_mode())
        ttk.Checkbutton(
            fuse,
            text="Match HU in overlap",
            variable=self.match_hu_var,
            command=self._on_fuse_mode,
        ).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(
            fuse,
            text="max / mean / blend / feather / min / a / b",
            font=("", 8),
        ).pack(anchor=tk.W)

        sl = ttk.LabelFrame(parent, text="Slice / view", padding=6)
        sl.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(sl, textvariable=self.z_label).pack(anchor=tk.W)
        zr = ttk.Frame(sl)
        zr.pack(fill=tk.X, pady=(4, 0))
        self.z_spin = ttk.Spinbox(
            zr,
            from_=0,
            to=0,
            textvariable=self.z_var,
            width=6,
            command=self._on_z_spin,
        )
        self.z_spin.pack(side=tk.LEFT)
        self.z_spin.bind("<Return>", lambda _e: self._on_z_spin())
        self.z_spin.bind("<FocusOut>", lambda _e: self._on_z_spin())
        ttk.Button(zr, text="Mid", width=4, command=self._z_mid).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Checkbutton(
            sl,
            text="Crop to core",
            variable=self.crop_core_var,
            command=self._schedule_refresh,
        ).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(sl, text="[ ] / PageUp·Down = prev/next", font=("", 8)).pack(
            anchor=tk.W
        )

        zm = ttk.LabelFrame(parent, text="Zoom", padding=6)
        zm.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(zm, textvariable=self.zoom_label).pack(anchor=tk.W)
        zb = ttk.Frame(zm)
        zb.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(zb, text="−", width=3, command=lambda: self._bump_zoom(1 / 1.25)).pack(
            side=tk.LEFT
        )
        ttk.Button(zb, text="+", width=3, command=lambda: self._bump_zoom(1.25)).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(zb, text="Fit", command=self._zoom_fit).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(zb, text="100%", command=self._zoom_100).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(zm, text="Wheel=zoom · Mid/Alt-drag=pan", font=("", 8)).pack(
            anchor=tk.W, pady=(4, 0)
        )

        win = ttk.LabelFrame(parent, text="Greyscale window (HU)", padding=6)
        win.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.hist_canvas = tk.Canvas(
            win, height=100, background="#1a1a1a", highlightthickness=0
        )
        self.hist_canvas.pack(fill=tk.X)
        self.hist_canvas.bind("<Configure>", lambda _e: self._draw_histogram())
        self.hist_canvas.bind("<Button-1>", self._hist_click)
        self.hist_canvas.bind("<B1-Motion>", self._hist_drag)
        self.hist_canvas.bind(
            "<ButtonRelease-1>", lambda _e: setattr(self, "_hist_drag_which", None)
        )
        ttk.Label(win, text="Black").pack(anchor=tk.W, pady=(4, 0))
        self.win_lo_scale = ttk.Scale(
            win, from_=0, to=1, variable=self.win_lo, command=self._on_window_slide
        )
        self.win_lo_scale.pack(fill=tk.X)
        ttk.Label(win, text="White").pack(anchor=tk.W, pady=(4, 0))
        self.win_hi_scale = ttk.Scale(
            win, from_=0, to=1, variable=self.win_hi, command=self._on_window_slide
        )
        self.win_hi_scale.pack(fill=tk.X)
        ttk.Button(win, text="Auto 1–99%", command=self._auto_window).pack(
            anchor=tk.W, pady=(6, 0)
        )
        ttk.Label(win, textvariable=self.win_label).pack(anchor=tk.W)

    def _build_analyze_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text="Whole-core sediment ROI: outside-in liner/void peel (optional).",
            wraplength=320,
        ).pack(anchor=tk.W)

        liner = ttk.LabelFrame(parent, text="Liner prior (optional)", padding=6)
        liner.pack(fill=tk.X, pady=(10, 0))
        self.liner_combo = ttk.Combobox(
            liner,
            textvariable=self.liner_preset,
            values=list(LINER_PRESETS.keys()),
            state="readonly",
            width=28,
        )
        self.liner_combo.pack(fill=tk.X)
        self.liner_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_liner_preset())
        lr = ttk.Frame(liner)
        lr.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(lr, text="OD mm").pack(side=tk.LEFT)
        ttk.Spinbox(
            lr, from_=20, to=200, increment=0.5, textvariable=self.liner_od, width=7
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(lr, text="Wall mm").pack(side=tk.LEFT)
        ttk.Spinbox(
            lr, from_=0.5, to=20, increment=0.1, textvariable=self.liner_wall, width=6
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(
            liner,
            text="Guides outer search; leave Auto if unknown.",
            font=("", 8),
        ).pack(anchor=tk.W, pady=(4, 0))

        ttk.Button(
            parent, text="Auto sediment (outside→in)", command=self._run_sediment
        ).pack(fill=tk.X, pady=(10, 4))
        ttk.Checkbutton(
            parent,
            text="Show sediment overlay",
            variable=self.show_sediment,
            command=self._on_show_sediment,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            parent,
            text="Show downcore mean HU plot",
            variable=self.show_downcore,
            command=self._on_show_downcore,
        ).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="Plot sits to the right of the core; Y matches core top→bottom. "
            "First compute can take a while (runs after the image paints).",
            font=("", 8),
            wraplength=320,
        ).pack(anchor=tk.W)
        ttk.Label(parent, textvariable=self.sed_label, wraplength=320).pack(
            anchor=tk.W, pady=(6, 0)
        )

        filt = ttk.LabelFrame(parent, text="Void / clast HU filter", padding=6)
        filt.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(
            filt,
            text="Exclude lows (voids)",
            variable=self.exclude_low,
            command=self._on_filter_toggle,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            filt,
            text="Exclude highs (clasts)",
            variable=self.exclude_high,
            command=self._on_filter_toggle,
        ).pack(anchor=tk.W)
        ttk.Button(
            filt, text="Auto thresholds (Otsu)", command=self._run_otsu
        ).pack(fill=tk.X, pady=(6, 4))
        ttk.Label(filt, textvariable=self.filter_label, wraplength=300, font=("", 8)).pack(
            anchor=tk.W
        )
        ttk.Label(filt, text="Thr lo (HU)", font=("", 8)).pack(anchor=tk.W, pady=(4, 0))
        ttk.Scale(
            filt,
            from_=-800.0,
            to=1500.0,
            variable=self.thr_lo,
            command=self._on_thr_slide,
        ).pack(fill=tk.X)
        ttk.Label(filt, text="Thr hi (HU)", font=("", 8)).pack(anchor=tk.W, pady=(4, 0))
        ttk.Scale(
            filt,
            from_=-800.0,
            to=2500.0,
            variable=self.thr_hi,
            command=self._on_thr_slide,
        ).pack(fill=tk.X)
        ttk.Label(
            filt,
            text="Green=included · cyan=void · orange=clast. "
            "Faint plot line = raw cylinder mean.",
            font=("", 8),
            wraplength=300,
        ).pack(anchor=tk.W, pady=(4, 0))

        er = ttk.LabelFrame(parent, text="Erode inward (mm)", padding=6)
        er.pack(fill=tk.X, pady=(10, 0))
        ttk.Scale(
            er,
            from_=0.0,
            to=50.0,
            variable=self.sed_erode,
            command=self._on_sediment_adjust,
        ).pack(fill=tk.X)
        ttk.Label(
            er, text="Shrink after auto detect (0–50 mm)", font=("", 8)
        ).pack(anchor=tk.W)

        nd = ttk.LabelFrame(parent, text="Center nudge (mm)", padding=6)
        nd.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(nd, text="ΔX (lateral)").pack(anchor=tk.W)
        ttk.Scale(
            nd,
            from_=-10.0,
            to=10.0,
            variable=self.sed_nudge_x,
            command=self._on_sediment_adjust,
        ).pack(fill=tk.X)
        ttk.Label(nd, text="ΔZ (through stack)").pack(anchor=tk.W, pady=(4, 0))
        ttk.Scale(
            nd,
            from_=-10.0,
            to=10.0,
            variable=self.sed_nudge_z,
            command=self._on_sediment_adjust,
        ).pack(fill=tk.X)
        ttk.Button(nd, text="Reset nudge", command=self._reset_sediment_nudge).pack(
            anchor=tk.W, pady=(6, 0)
        )

        ttk.Button(
            parent,
            text="Apply to all Z (build 3D mask)",
            command=self._sediment_all_z,
        ).pack(fill=tk.X, pady=(14, 0))
        ttk.Label(
            parent,
            text="Export can run with or without Analyze. "
            "3D mask file is written only after Apply to all Z.",
            wraplength=320,
            font=("", 8),
        ).pack(anchor=tk.W, pady=(8, 0))

    def _build_export_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text="Stitch exports work anytime after Align. "
            "Sediment extras need Analyze.",
            wraplength=320,
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(parent, text="Stitch products", font=("", 9, "bold")).pack(anchor=tk.W)
        ttk.Checkbutton(
            parent, text="JPEG (windowed 8-bit)", variable=self.export_jpeg
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            parent,
            text="TIFF (8-bit LZW + 16-bit HU)",
            variable=self.export_tiff,
        ).pack(anchor=tk.W, pady=(4, 0))
        ttk.Checkbutton(parent, text="DICOM MPR stack", variable=self.export_dicom).pack(
            anchor=tk.W, pady=(4, 0)
        )
        ttk.Checkbutton(
            parent, text="NumPy (2D fused + 3D MPR)", variable=self.export_npy
        ).pack(anchor=tk.W, pady=(4, 0))
        ttk.Checkbutton(parent, text="JSON sidecar", variable=self.export_json).pack(
            anchor=tk.W, pady=(4, 0)
        )

        ttk.Label(parent, text="Sediment (Analyze)", font=("", 9, "bold")).pack(
            anchor=tk.W, pady=(12, 0)
        )
        ttk.Checkbutton(
            parent,
            text="Sediment mask (TIFF + NPY)",
            variable=self.export_sediment_mask,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            parent,
            text="Sediment ROI image (clipped 2D)",
            variable=self.export_sediment_roi,
        ).pack(anchor=tk.W, pady=(4, 0))

        ttk.Button(parent, text="Export selected…", command=self.accept_export).pack(
            fill=tk.X, pady=(16, 0)
        )
        ttk.Label(
            parent,
            text="Crop-to-core from Display applies to JPEG/TIFF/2D NumPy.",
            wraplength=320,
            font=("", 8),
        ).pack(anchor=tk.W, pady=(8, 0))

    def _bind_keys(self) -> None:
        self.bind("<Left>", lambda _e: self._nudge_keys(-1, 0))
        self.bind("<Right>", lambda _e: self._nudge_keys(1, 0))
        self.bind("<Up>", lambda _e: self._nudge_keys(0, -1))
        self.bind("<Down>", lambda _e: self._nudge_keys(0, 1))
        self.bind("<Shift-Left>", lambda _e: self._nudge_keys(-5, 0))
        self.bind("<Shift-Right>", lambda _e: self._nudge_keys(5, 0))
        self.bind("<Shift-Up>", lambda _e: self._nudge_keys(0, -5))
        self.bind("<Shift-Down>", lambda _e: self._nudge_keys(0, 5))
        self.bind("<Control-plus>", lambda _e: self._bump_zoom(1.25))
        self.bind("<Control-equal>", lambda _e: self._bump_zoom(1.25))
        self.bind("<Control-minus>", lambda _e: self._bump_zoom(1 / 1.25))
        self.bind("<Control-0>", lambda _e: self._zoom_fit())
        self.bind("<Prior>", lambda _e: self._bump_z(-1))  # PageUp
        self.bind("<Next>", lambda _e: self._bump_z(1))  # PageDown
        self.bind("<bracketleft>", lambda _e: self._bump_z(-1))
        self.bind("<bracketright>", lambda _e: self._bump_z(1))

    def browse_a(self) -> None:
        p = filedialog.askdirectory(title="Select TOP (A) series folder")
        if p:
            self.path_a = Path(p)
            self.path_a_var.set(p)

    def browse_b(self) -> None:
        p = filedialog.askdirectory(title="Select BTM (B) series folder")
        if p:
            self.path_b = Path(p)
            self.path_b_var.set(p)

    def run_coarse(self) -> None:
        if not self.path_a or not self.path_b:
            messagebox.showinfo("Need folders", "Select both TOP and BTM series folders.")
            return
        self.status.set("Running coarse align…")
        self.update_idletasks()
        try:
            self.state = coarse_align(self.path_a, self.path_b)
        except Exception as exc:
            messagebox.showerror("Align failed", str(exc))
            self.status.set(f"Failed: {exc}")
            return
        self.flip_var.set(False)
        self.roll_var.set(0.0)
        self.fuse_mode.set(self.state.fuse_mode)
        self.match_hu_var.set(self.state.match_hu)
        self.sed_erode.set(0.0)
        self.sed_nudge_x.set(0.0)
        self.sed_nudge_z.set(0.0)
        self.export_sediment_mask.set(False)
        self.export_sediment_roi.set(False)
        self._sync_sediment_label()
        self._sync_nudge_vars()
        # Widen nudge range from core length
        extent = max(
            self.state.fixed.geom.length_mm,
            self.state.mobile_base.geom.length_mm,
            80.0,
        )
        self.nudge_range.set(min(500.0, max(80.0, 0.35 * extent)))
        self._apply_nudge_range()
        self._update_metrics()
        for w in self.state.warnings:
            self.status.set(f"Note: {w}")
        if self.state.low_confidence:
            messagebox.showwarning(
                "Low confidence overlap",
                "Overlap peak is ambiguous or weak.\n"
                "Use Flip, opacity, and X/Y nudge to QC, then Accept.",
            )
        self.zoom_var.set(1.0)
        self._auto_window()
        # Paint mid-slice immediately — do not block on full Z stack / XZ / profile
        self.status.set("Coarse align done — showing mid-slice…")
        self._schedule_refresh()
        self.z_label.set("Slice: loading stack…")
        if self._z_load_job is not None:
            self.after_cancel(self._z_load_job)
        self._z_load_job = self.after(50, self._load_z_stack_deferred)

    def _load_z_stack_deferred(self) -> None:
        """Build posed volumes off the critical path so the first image can paint."""
        self._z_load_job = None
        if self.state is None:
            return
        self.status.set("Loading Z stack for slice browser…")
        self.update_idletasks()
        try:
            self._init_z_browser()
        except Exception as exc:
            self.z_label.set("Slice: mid only")
            self.status.set(f"Coarse done (Z browser unavailable: {exc})")
            return
        self.status.set(
            "Z stack ready — set greyscale window, cycle Slice, then Export."
        )
        self._schedule_refresh()
        self._schedule_xz_refresh(delay_ms=300)

    def _init_z_browser(self) -> None:
        if self.state is None:
            return
        posed = ensure_posed(self.state)
        nz = posed.n_slices
        mid = nz // 2
        self.z_spin.configure(from_=0, to=max(0, nz - 1))
        self.z_var.set(mid)
        set_view_z(self.state, mid)
        self.z_label.set(f"Slice: {mid} / {nz - 1}")

    def _on_z_spin(self) -> None:
        if self._z_job is not None:
            self.after_cancel(self._z_job)
        self._z_job = self.after(80, self._apply_z)

    def _apply_z(self) -> None:
        self._z_job = None
        if self.state is None:
            return
        try:
            z = int(self.z_var.get())
        except (TypeError, ValueError, tk.TclError):
            return
        self.status.set(f"Showing slice {z}…")
        self.update_idletasks()
        try:
            if self.state.posed is None:
                ensure_posed(self.state)
            set_view_z(self.state, z)
            nz = self.state.posed.n_slices if self.state.posed else z
            self.z_label.set(f"Slice: {self.state.view_z} / {nz - 1}")
        except Exception as exc:
            messagebox.showerror("Slice", str(exc))
            return
        self._schedule_refresh()
        self._schedule_xz_refresh(delay_ms=50)

    def _z_mid(self) -> None:
        if self.state is None:
            return
        posed = ensure_posed(self.state)
        self.z_var.set(posed.n_slices // 2)
        self._apply_z()

    def _bump_z(self, delta: int) -> None:
        if self.state is None or self.state.posed is None:
            return
        z = int(np.clip(int(self.z_var.get()) + delta, 0, self.state.posed.n_slices - 1))
        self.z_var.set(z)
        self._apply_z()

    def _sync_nudge_vars(self) -> None:
        if self.state is None:
            return
        self.nudge_x.set(self.state.nudge_x_mm)
        self.nudge_y.set(self.state.nudge_y_mm)
        self.nudge_label.set(
            f"B nudge X/Y: {self.state.nudge_x_mm:.2f} / {self.state.nudge_y_mm:.2f} mm"
        )

    def _apply_nudge_range(self) -> None:
        """Update X/Y slider end-stops only (does not move B by itself)."""
        try:
            r = abs(float(self.nudge_range.get()))
        except (TypeError, ValueError):
            r = 80.0
        r = max(20.0, min(r, 500.0))
        self.nudge_range.set(r)
        self.nudge_range_label.set(f"Slider limits ±{r:.0f} mm")
        self.nudge_x_scale.configure(from_=-r, to=r)
        self.nudge_y_scale.configure(from_=-r, to=r)
        self.nudge_x.set(float(np.clip(self.nudge_x.get(), -r, r)))
        self.nudge_y.set(float(np.clip(self.nudge_y.get(), -r, r)))

    def _widen_nudge_range(self) -> None:
        """Double slider travel if you hit the end while nudging."""
        try:
            r = abs(float(self.nudge_range.get()))
        except (TypeError, ValueError):
            r = 80.0
        self.nudge_range.set(min(500.0, max(40.0, r * 2.0)))
        self._apply_nudge_range()

    def _on_flip(self) -> None:
        if self.state is None:
            return
        set_flip(self.state, self.flip_var.get())
        self._sync_nudge_vars()
        self._update_metrics()
        self.status.set("Flip applied — overlap re-searched (nudge reset).")
        self._schedule_refresh()

    def _on_roll_slide(self, _value: str | None = None) -> None:
        if self._roll_job is not None:
            self.after_cancel(self._roll_job)
        self._roll_job = self.after(200, self._apply_roll)

    def _apply_roll(self) -> None:
        self._roll_job = None
        if self.state is None:
            return
        set_roll(self.state, float(self.roll_var.get()))
        self._sync_nudge_vars()
        self._update_metrics()
        self._schedule_refresh()

    def _roll_zero(self) -> None:
        self.roll_var.set(0.0)
        self._apply_roll()

    def _rerun_overlap(self) -> None:
        if self.state is None:
            return
        recompute_overlap(self.state, reset_nudge=True)
        self._sync_nudge_vars()
        self._update_metrics()
        self.status.set("Overlap re-searched (nudge reset).")
        self._schedule_refresh()

    def _on_fuse_mode(self) -> None:
        if self.state is None:
            return
        self.state.fuse_mode = str(self.fuse_mode.get() or "max")
        self.state.match_hu = bool(self.match_hu_var.get())
        # Greyscale preview uses fuse mode; alignment colors ignore it
        if not self.align_var.get():
            self._schedule_refresh()
        else:
            self.status.set(
                f"Overlap mode={self.state.fuse_mode} (turn off Alignment colors to preview fuse)"
            )

    def _on_nudge_slide(self, _value: str | None = None) -> None:
        if self._nudge_job is not None:
            self.after_cancel(self._nudge_job)
        self._nudge_job = self.after(30, self._apply_nudge)

    def _apply_nudge(self) -> None:
        self._nudge_job = None
        if self.state is None:
            return
        set_nudge(
            self.state,
            x_mm=float(self.nudge_x.get()),
            y_mm=float(self.nudge_y.get()),
        )
        self._sync_nudge_vars()
        self._update_metrics()
        self._schedule_refresh()

    def _reset_nudge(self) -> None:
        self.nudge_x.set(0.0)
        self.nudge_y.set(0.0)
        self._apply_nudge()

    def _nudge_keys(self, dx: float, dy: float) -> None:
        if self.state is None:
            return
        r = float(self.nudge_range.get())
        self.nudge_x.set(float(np.clip(float(self.nudge_x.get()) + dx, -r, r)))
        self.nudge_y.set(float(np.clip(float(self.nudge_y.get()) + dy, -r, r)))
        self._apply_nudge()

    def run_refine(self) -> None:
        if self.state is None:
            messagebox.showinfo("Refine join", "Run coarse align first.")
            return
        self.status.set("Refine join — 3D overlap search (may take a bit)…")
        self.update_idletasks()
        try:
            result = refine_join(self.state)
        except Exception as exc:
            messagebox.showerror("Refine failed", str(exc))
            self.status.set(f"Refine failed: {exc}")
            return
        self._sync_nudge_vars()
        self._update_metrics()
        self.status.set(
            f"Refine join: Δ=({result.dx_mm:.2f},{result.dy_mm:.2f},{result.dz_mm:.2f}) mm  "
            f"NCC {result.score_before:.3f}→{result.score:.3f}  [{result.method}]"
        )
        self._schedule_refresh()

    def _update_metrics(self) -> None:
        if self.state is None or self.state.overlap is None:
            self.metrics.set("—")
            return
        ov = self.state.overlap
        conf = "LOW CONFIDENCE" if ov.low_confidence else "ok"
        ref = self.state.refine
        ref_s = ""
        if ref is not None:
            ref_s = (
                f"  refine Δ=({ref.dx_mm:.2f},{ref.dy_mm:.2f},{ref.dz_mm:.2f}) mm "
                f"NCC={ref.score:.3f}"
            )
        self.metrics.set(
            f"auto overlap={ov.offset_mm:.1f} mm  NCC={ov.peak_score:.3f}  "
            f"ratio={ov.peak_ratio:.3f}  [{conf}]  "
            f"flip={self.state.flip}  roll={self.state.roll_deg:.1f}°  "
            f"nudge=({self.state.nudge_x_mm:.1f},{self.state.nudge_y_mm:.1f},"
            f"{self.state.nudge_z_mm:.1f}) mm  "
            f"fuse={self.state.fuse_mode}  "
            f"grid={self.state.spacing_mm:.4f} mm/px{ref_s}"
        )

    def _sync_window_from_state(self) -> None:
        if self.state is None:
            return
        lo, hi = window_limits(self.state)
        vals = masked_hu_values(self.state)
        if vals.size == 0:
            vmin, vmax = lo, hi
        else:
            vmin = float(np.percentile(vals, 0.5))
            vmax = float(np.percentile(vals, 99.5))
            pad = 0.05 * max(vmax - vmin, 1.0)
            vmin, vmax = vmin - pad, vmax + pad
        if vmax <= vmin:
            vmax = vmin + 1.0
        self._hist_range = (vmin, vmax)
        self.win_lo_scale.configure(from_=vmin, to=vmax)
        self.win_hi_scale.configure(from_=vmin, to=vmax)
        self.win_lo.set(lo)
        self.win_hi.set(hi)
        self.state.window_lo = lo
        self.state.window_hi = hi
        self.win_label.set(f"Window: [{lo:.1f}, {hi:.1f}] HU")
        counts, edges = np.histogram(vals, bins=128, range=(vmin, vmax))
        self._hist_counts = counts.astype(np.float64)
        self._hist_edges = edges
        self._draw_histogram()

    def _auto_window(self) -> None:
        if self.state is None:
            return
        set_auto_window(self.state)
        self._sync_window_from_state()
        if not self.align_var.get():
            self._schedule_refresh()

    def _on_window_slide(self, _value: str | None = None) -> None:
        if self._win_job is not None:
            self.after_cancel(self._win_job)
        self._win_job = self.after(40, self._apply_window)

    def _apply_window(self) -> None:
        self._win_job = None
        if self.state is None:
            return
        lo = float(self.win_lo.get())
        hi = float(self.win_hi.get())
        if hi < lo:
            lo, hi = hi, lo
            self.win_lo.set(lo)
            self.win_hi.set(hi)
        self.state.window_lo = lo
        self.state.window_hi = hi
        self.win_label.set(f"Window: [{lo:.1f}, {hi:.1f}] HU")
        self._draw_histogram()
        if not self.align_var.get():
            self._schedule_refresh()

    def _hu_to_x(self, hu: float, width: int) -> float:
        vmin, vmax = self._hist_range
        if vmax <= vmin:
            return 0.0
        return (hu - vmin) / (vmax - vmin) * max(width - 1, 1)

    def _x_to_hu(self, x: float, width: int) -> float:
        vmin, vmax = self._hist_range
        t = np.clip(x / max(width - 1, 1), 0.0, 1.0)
        return float(vmin + t * (vmax - vmin))

    def _draw_histogram(self) -> None:
        c = self.hist_canvas
        c.delete("all")
        w = max(c.winfo_width(), 2)
        h = max(c.winfo_height(), 2)
        if self._hist_counts is None or self._hist_counts.size == 0:
            c.create_text(w // 2, h // 2, text="Histogram after coarse align", fill="#888")
            return
        counts = self._hist_counts
        peak = float(counts.max()) if counts.max() > 0 else 1.0
        n = len(counts)
        bar_w = w / n
        for i, v in enumerate(counts):
            bh = (v / peak) * (h - 4)
            x0 = i * bar_w
            c.create_rectangle(
                x0, h - bh, x0 + bar_w + 0.5, h, fill="#6a9fb5", outline="", tags="bar"
            )
        lo = float(self.win_lo.get())
        hi = float(self.win_hi.get())
        x_lo = self._hu_to_x(lo, w)
        x_hi = self._hu_to_x(hi, w)
        c.create_rectangle(x_lo, 0, x_hi, h, fill="#ffffff", stipple="gray50", outline="")
        c.create_line(x_lo, 0, x_lo, h, fill="#ff6666", width=2, tags="lo")
        c.create_line(x_hi, 0, x_hi, h, fill="#66ff99", width=2, tags="hi")
        c.create_text(6, 10, anchor=tk.W, fill="#ccc", text=f"{self._hist_range[0]:.0f}")
        c.create_text(w - 6, 10, anchor=tk.E, fill="#ccc", text=f"{self._hist_range[1]:.0f}")

    def _hist_click(self, event: tk.Event) -> None:
        if self.state is None:
            return
        w = max(self.hist_canvas.winfo_width(), 2)
        x_lo = self._hu_to_x(float(self.win_lo.get()), w)
        x_hi = self._hu_to_x(float(self.win_hi.get()), w)
        if abs(event.x - x_lo) <= abs(event.x - x_hi):
            self._hist_drag_which = "lo"
        else:
            self._hist_drag_which = "hi"
        self._hist_drag(event)

    def _hist_drag(self, event: tk.Event) -> None:
        if self.state is None or self._hist_drag_which is None:
            return
        w = max(self.hist_canvas.winfo_width(), 2)
        hu = self._x_to_hu(float(event.x), w)
        if self._hist_drag_which == "lo":
            self.win_lo.set(min(hu, float(self.win_hi.get()) - 1.0))
        else:
            self.win_hi.set(max(hu, float(self.win_lo.get()) + 1.0))
        self._apply_window()

    def _sync_sediment_label(self) -> None:
        if self.state is None or self.state.sediment is None:
            self.sed_label.set("Sediment ROI: not detected")
            self.filter_label.set("HU filter: not run")
            return
        s = self.state.sediment
        liner = f"{s.liner_hu:.0f}" if s.liner_hu is not None else "?"
        void = f"{s.void_hu:.0f}" if s.void_hu is not None else "?"
        z3 = " · 3D ready" if s.mask_3d is not None else ""
        self.sed_label.set(
            f"R={s.effective_radius_mm:.1f} mm (auto {s.radius_mm:.1f} − erode "
            f"{s.erode_mm:.1f}) · liner≈{liner} void≈{void} HU · {s.note}{z3}"
        )
        self._sync_filter_label()

    def _sync_filter_label(self) -> None:
        if self.state is None or self.state.sediment is None:
            self.filter_label.set("HU filter: not run")
            return
        s = self.state.sediment
        if not s.has_thresholds:
            self.filter_label.set("HU filter: run Auto thresholds (Otsu)")
            return
        excl = []
        if s.exclude_low:
            excl.append("lows")
        if s.exclude_high:
            excl.append("highs")
        excl_s = "+".join(excl) if excl else "none"
        fr = ""
        if s.frac_low is not None and s.frac_high is not None:
            fr = f" · void {100 * s.frac_low:.0f}% clast {100 * s.frac_high:.0f}%"
        self.filter_label.set(
            f"lo={s.thr_lo_hu:.0f} · hi={s.thr_hi_hu:.0f} HU ({s.filter_method}) "
            f"· exclude {excl_s}{fr}"
        )

    def _sync_thr_vars_from_state(self) -> None:
        if self.state is None or self.state.sediment is None:
            return
        s = self.state.sediment
        if not s.has_thresholds:
            return
        self._ignore_thr_slide = True
        try:
            self.thr_lo.set(float(s.thr_lo_hu))  # type: ignore[arg-type]
            self.thr_hi.set(float(s.thr_hi_hu))  # type: ignore[arg-type]
        finally:
            self._ignore_thr_slide = False

    def _on_liner_preset(self) -> None:
        key = str(self.liner_preset.get())
        od, wall = LINER_PRESETS.get(key, (None, None))
        if od is not None:
            self.liner_od.set(float(od))
        if wall is not None:
            self.liner_wall.set(float(wall))

    def _liner_prior_args(self) -> tuple[float | None, float | None]:
        key = str(self.liner_preset.get())
        if key.startswith("Auto"):
            return None, None
        try:
            return float(self.liner_od.get()), float(self.liner_wall.get())
        except (TypeError, ValueError, tk.TclError):
            return None, None

    def _run_sediment(self) -> None:
        if self.state is None:
            messagebox.showinfo("Analyze", "Run coarse align first.")
            return
        od, wall = self._liner_prior_args()
        self.status.set("Detecting sediment ROI (outside→in)…")
        self.update_idletasks()
        try:
            sed = detect_sediment_roi(
                self.state, liner_od_mm=od, liner_wall_mm=wall
            )
        except Exception as exc:
            messagebox.showerror("Analyze failed", str(exc))
            self.status.set(f"Analyze failed: {exc}")
            return
        self.sed_erode.set(sed.erode_mm)
        self.sed_nudge_x.set(sed.nudge_x_mm)
        self.sed_nudge_z.set(sed.nudge_z_mm)
        self.exclude_low.set(sed.exclude_low)
        self.exclude_high.set(sed.exclude_high)
        self.state.show_sediment = bool(self.show_sediment.get())
        self.state.show_downcore_plot = bool(self.show_downcore.get())
        invalidate_downcore_profile(self.state)
        self._sync_sediment_label()
        self.status.set(
            f"Sediment ROI: R={sed.effective_radius_mm:.1f} mm — tweak erode/nudge as needed."
        )
        self._schedule_refresh()
        self._schedule_xz_refresh(delay_ms=200)
        if self._want_analysis_compute():
            self._schedule_downcore_compute()

    def _want_analysis_compute(self) -> bool:
        if self.state is None or self.state.sediment is None:
            return False
        return bool(
            self.show_downcore.get()
            or self.exclude_low.get()
            or self.exclude_high.get()
        )

    def _on_show_sediment(self) -> None:
        if self.state is not None:
            self.state.show_sediment = bool(self.show_sediment.get())
        self._schedule_refresh()
        self._schedule_xz_refresh(delay_ms=100)

    def _on_show_downcore(self) -> None:
        if self.state is not None:
            self.state.show_downcore_plot = bool(self.show_downcore.get())
        self._schedule_refresh()
        if bool(self.show_downcore.get()):
            self._schedule_downcore_compute()

    def _on_filter_toggle(self) -> None:
        if self.state is None or self.state.sediment is None:
            return
        self.state.sediment.exclude_low = bool(self.exclude_low.get())
        self.state.sediment.exclude_high = bool(self.exclude_high.get())
        invalidate_downcore_profile(self.state)
        self._sync_filter_label()
        self._schedule_refresh()
        if self._want_analysis_compute():
            self._schedule_downcore_compute(delay_ms=100)

    def _run_otsu(self) -> None:
        if self.state is None or self.state.sediment is None:
            messagebox.showinfo("Analyze", "Run Auto sediment first.")
            return
        self.state.sediment.exclude_low = bool(self.exclude_low.get())
        self.state.sediment.exclude_high = bool(self.exclude_high.get())
        self.status.set("Running multi-Otsu on cylinder HU…")
        self._schedule_downcore_compute(delay_ms=50, recompute_otsu=True)

    def _on_thr_slide(self, _value: str | None = None) -> None:
        if self._ignore_thr_slide:
            return
        if self._thr_job is not None:
            self.after_cancel(self._thr_job)
        self._thr_job = self.after(120, self._apply_thr_manual)

    def _apply_thr_manual(self) -> None:
        self._thr_job = None
        if self.state is None or self.state.sediment is None:
            return
        lo = float(self.thr_lo.get())
        hi = float(self.thr_hi.get())
        if hi < lo:
            lo, hi = hi, lo
            self._ignore_thr_slide = True
            try:
                self.thr_lo.set(lo)
                self.thr_hi.set(hi)
            finally:
                self._ignore_thr_slide = False
        sed = self.state.sediment
        sed.thr_lo_hu = lo
        sed.thr_hi_hu = hi
        sed.filter_method = "manual"
        sed.exclude_low = bool(self.exclude_low.get())
        sed.exclude_high = bool(self.exclude_high.get())
        invalidate_downcore_profile(self.state)
        rebuild_sediment_mask_2d(self.state)
        self._sync_filter_label()
        self._schedule_refresh()
        if self._want_analysis_compute():
            self._schedule_downcore_compute(delay_ms=80)

    def _on_sediment_adjust(self, _value: str | None = None) -> None:
        if self._sed_job is not None:
            self.after_cancel(self._sed_job)
        self._sed_job = self.after(50, self._apply_sediment_adjust)

    def _apply_sediment_adjust(self) -> None:
        self._sed_job = None
        if self.state is None or self.state.sediment is None:
            return
        self.state.sediment.erode_mm = float(self.sed_erode.get())
        self.state.sediment.nudge_x_mm = float(self.sed_nudge_x.get())
        self.state.sediment.nudge_z_mm = float(self.sed_nudge_z.get())
        self.state.sediment.mask_3d = None  # invalidate volume mask
        # Geometry change → clear Otsu until recomputed
        self.state.sediment.thr_lo_hu = None
        self.state.sediment.thr_hi_hu = None
        self.state.sediment.filter_method = "none"
        invalidate_downcore_profile(self.state)
        rebuild_sediment_mask_2d(self.state)
        self._sync_sediment_label()
        self._schedule_refresh()
        self._schedule_xz_refresh(delay_ms=150)
        if self._want_analysis_compute():
            self._schedule_downcore_compute(delay_ms=200)

    def _reset_sediment_nudge(self) -> None:
        self.sed_nudge_x.set(0.0)
        self.sed_nudge_z.set(0.0)
        self._apply_sediment_adjust()

    def _sediment_all_z(self) -> None:
        if self.state is None or self.state.sediment is None:
            messagebox.showinfo("Analyze", "Run Auto sediment first.")
            return
        self.status.set("Building 3D sediment mask (all Z)…")
        self.update_idletasks()
        try:
            m3 = build_sediment_volume_mask(self.state)
        except Exception as exc:
            messagebox.showerror("3D mask failed", str(exc))
            self.status.set(f"3D mask failed: {exc}")
            return
        self._sync_sediment_label()
        invalidate_downcore_profile(self.state)
        self.status.set(
            f"3D sediment mask ready: {tuple(m3.shape)} · "
            f"{int(m3.sum())} voxels — enable mask export on Export tab."
        )
        self.export_sediment_mask.set(True)
        self._schedule_refresh()

    def accept_export(self) -> None:
        if self.state is None:
            messagebox.showinfo("Export", "Run coarse align first.")
            return
        jpeg = bool(self.export_jpeg.get())
        tiff = bool(self.export_tiff.get())
        dicom = bool(self.export_dicom.get())
        npy = bool(self.export_npy.get())
        js = bool(self.export_json.get())
        sed_m = bool(self.export_sediment_mask.get())
        sed_r = bool(self.export_sediment_roi.get())
        if not any((jpeg, tiff, dicom, npy, js, sed_m, sed_r)):
            messagebox.showinfo("Export", "Select at least one export type.")
            return
        if (sed_m or sed_r) and (
            self.state.sediment is None or self.state.sediment.mask_2d is None
        ):
            messagebox.showinfo(
                "Export",
                "Sediment extras need Analyze → Auto sediment first.",
            )
            return
        if self.state.low_confidence:
            if not messagebox.askyesno(
                "Low confidence",
                "Overlap is marked low confidence.\nExport anyway?",
            ):
                return
        out = filedialog.askdirectory(title="Export folder")
        if not out:
            return
        self.status.set("Exporting selected products…")
        self.update_idletasks()
        try:
            paths = export_accept(
                self.state,
                out,
                stem="stitch",
                jpeg=jpeg,
                tiff=tiff,
                dicom=dicom,
                npy=npy,
                json_sidecar=js,
                sediment_mask=sed_m,
                sediment_roi_image=sed_r,
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            self.status.set(f"Export failed: {exc}")
            return
        lines = [str(p) for p in paths.values()]
        self.status.set(f"Exported {len(paths)} item(s) → {out}")
        messagebox.showinfo("Exported", "Wrote:\n" + "\n".join(lines))

    # ----- zoom / pan -----
    def _bump_zoom(self, factor: float) -> None:
        z = float(np.clip(float(self.zoom_var.get()) * factor, 0.25, 16.0))
        self.zoom_var.set(z)
        self.zoom_label.set(f"Zoom: {int(round(z * 100))}%")
        self._schedule_refresh()

    def _zoom_fit(self) -> None:
        self.zoom_var.set(1.0)
        self.zoom_label.set("Zoom: 100%")
        self._schedule_refresh()

    def _zoom_100(self) -> None:
        """1:1 image pixels (may exceed window)."""
        if self._full_im is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        fit = min(cw / self._full_im.width, ch / self._full_im.height, 1.0)
        if fit <= 1e-6:
            fit = 1.0
        # zoom multiplier such that fit * zoom = 1.0 (native pixels)
        self.zoom_var.set(float(np.clip(1.0 / fit, 0.25, 16.0)))
        self.zoom_label.set(f"Zoom: {int(round(float(self.zoom_var.get()) * 100))}%")
        self._schedule_refresh()

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.delta > 0:
            self._bump_zoom(1.15)
        else:
            self._bump_zoom(1 / 1.15)

    def _on_mousewheel_linux(self, event: tk.Event, direction: int) -> None:
        self._bump_zoom(1.15 if direction > 0 else 1 / 1.15)

    def _pan_press(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _pan_move(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _alt_pan_press(self, event: tk.Event) -> None:
        # Pan with Alt+LMB so normal click isn't stolen forever
        if event.state & 0x0008:  # Alt
            self._pan_start = (event.x, event.y)
            self.canvas.scan_mark(event.x, event.y)
        else:
            self._pan_start = None

    def _alt_pan_move(self, event: tk.Event) -> None:
        if self._pan_start is not None:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _alt_pan_release(self, _event: tk.Event) -> None:
        self._pan_start = None

    def _schedule_refresh(self) -> None:
        if self._refresh_job is not None:
            self.after_cancel(self._refresh_job)
        self._refresh_job = self.after(40, self.refresh)

    def _schedule_xz_refresh(self, delay_ms: int = 80) -> None:
        if self._xz_job is not None:
            self.after_cancel(self._xz_job)
        self._xz_job = self.after(delay_ms, self._refresh_xz)

    def _schedule_downcore_compute(
        self, delay_ms: int = 100, *, recompute_otsu: bool = False
    ) -> None:
        if self._downcore_job is not None:
            self.after_cancel(self._downcore_job)
        self._downcore_recompute_otsu = recompute_otsu
        self._downcore_job = self.after(delay_ms, self._compute_downcore)

    def _compute_downcore(self) -> None:
        """Heavy stack average / Otsu — runs after the coronal image has painted."""
        self._downcore_job = None
        recompute_otsu = bool(getattr(self, "_downcore_recompute_otsu", False))
        self._downcore_recompute_otsu = False
        if self.state is None or self.state.sediment is None:
            return
        if not recompute_otsu and not self._want_analysis_compute():
            return
        if self.state.posed is None:
            self.status.set("Analysis waiting for Z stack…")
            self._schedule_downcore_compute(
                delay_ms=400, recompute_otsu=recompute_otsu
            )
            return
        self.state.sediment.exclude_low = bool(self.exclude_low.get())
        self.state.sediment.exclude_high = bool(self.exclude_high.get())
        self.state.show_downcore_plot = bool(self.show_downcore.get())
        msg = (
            "Running multi-Otsu + downcore mean HU…"
            if recompute_otsu
            else "Computing downcore mean HU (all Z)…"
        )
        self.status.set(msg)
        self.update_idletasks()
        try:
            prof = ensure_downcore_profile(
                self.state, recompute_otsu=recompute_otsu
            )
        except Exception as exc:
            self.status.set(f"Downcore HU failed: {exc}")
            return
        self._sync_thr_vars_from_state()
        self._sync_filter_label()
        n_ok = int(np.isfinite(prof.mean_hu).sum())
        extra = ""
        if prof.frac_excluded is not None and prof.frac_excluded > 0.005:
            extra = f" · excluded {100 * prof.frac_excluded:.0f}%"
        if prof.thr_lo_hu is not None:
            extra += f" · thr {prof.thr_lo_hu:.0f}–{prof.thr_hi_hu:.0f}"
        self.status.set(f"Downcore mean HU ready ({n_ok} rows){extra}")
        self._schedule_refresh()

    def _on_xz_y_slide(self, _value: str | None = None) -> None:
        self._schedule_xz_refresh(delay_ms=80)

    def _refresh_xz(self) -> None:
        self._xz_job = None
        if self.state is None:
            self.xz_label.set("XZ: run coarse align first")
            self.xz_canvas.delete("all")
            return
        if self.state.posed is None:
            self.xz_label.set("XZ: waiting for Z stack…")
            return
        # Wait until the panel has a real width (first layout can be ~1 px)
        cw = max(self.xz_canvas.winfo_width(), self.xz_frame.winfo_width() - 16)
        ch = max(self.xz_canvas.winfo_height(), 120)
        if cw < 40:
            self.xz_label.set("XZ: waiting for layout…")
            self._schedule_xz_refresh(delay_ms=150)
            return
        # Fast path: redraw from cache only
        cache_ready = self.state._xz_hu is not None
        if not cache_ready:
            self.xz_label.set("XZ: building… (first time can take a few seconds)")
            self.update_idletasks()
        try:
            if self.state.view_z is None:
                self.state.view_z = self.state.posed.n_slices // 2
            arr = preview_xz_arrays(
                self.state,
                y_frac=float(self.xz_y_frac.get()),
                overlay_sediment=bool(self.show_sediment.get()),
                crop_core=True,
                pad_mm=3.0,
            )
        except Exception as exc:
            self.xz_label.set(f"XZ: {exc}")
            return
        if arr is None:
            self.xz_label.set("XZ: unavailable (posed stack failed)")
            self.xz_canvas.delete("all")
            return
        full = Image.fromarray(arr, mode="RGB")
        fit = min(cw / full.width, ch / full.height)
        disp_w = max(1, int(round(full.width * fit)))
        disp_h = max(1, int(round(full.height * fit)))
        im = full.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self._photo_xz = ImageTk.PhotoImage(im)
        self.xz_canvas.delete("all")
        x0 = max(0, (cw - disp_w) // 2)
        y0 = max(0, (ch - disp_h) // 2)
        self.xz_canvas.create_image(x0, y0, image=self._photo_xz, anchor=tk.NW)
        z = self.state.view_z
        sed = self.state.sediment
        if sed is not None:
            self.xz_label.set(
                f"XZ · Y={float(self.xz_y_frac.get()):.0%} · "
                f"R={sed.effective_radius_mm:.1f} mm · "
                f"yellow=coronal Z{z} · green=cylinder"
            )
        else:
            self.xz_label.set(
                f"XZ · Y={float(self.xz_y_frac.get()):.0%} · "
                f"yellow=coronal Z{z} · Auto sediment for circle"
            )

    def refresh(self) -> None:
        self._refresh_job = None
        if self.state is None:
            return
        self.state.show_downcore_plot = bool(self.show_downcore.get())
        try:
            arr = preview_arrays(
                self.state,
                align_mode=self.align_var.get(),
                opacity_a=float(self.opacity_a.get()),
                opacity_b=float(self.opacity_b.get()),
                crop_core=bool(self.crop_core_var.get()),
                pad_mm=3.0,
                downcore_plot=bool(self.show_downcore.get()),
            )
        except Exception as exc:
            self.status.set(f"Preview failed: {exc}")
            return
        if arr.ndim == 2:
            full = Image.fromarray(arr, mode="L")
        else:
            full = Image.fromarray(arr, mode="RGB")
        self._full_im = full

        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        fit = min(cw / full.width, ch / full.height, 1.0)
        zoom = float(self.zoom_var.get())
        scale = fit * zoom
        disp_w = max(1, int(round(full.width * scale)))
        disp_h = max(1, int(round(full.height * scale)))
        im = full.resize((disp_w, disp_h), Image.Resampling.BILINEAR)

        self._photo = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW, tags=("img",))
        self.canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
        self.zoom_label.set(f"Zoom: {int(round(zoom * 100))}%")
        # Keep histogram markers in sync when slice/fuse changes
        if self._hist_counts is not None:
            try:
                vals = masked_hu_values(self.state)
                vmin, vmax = self._hist_range
                counts, edges = np.histogram(vals, bins=128, range=(vmin, vmax))
                self._hist_counts = counts.astype(np.float64)
                self._hist_edges = edges
                self._draw_histogram()
            except Exception:
                pass
        # Kick deferred analysis if needed and cache empty
        if (
            self.state.sediment is not None
            and self.state._downcore_profile is None
            and self._want_analysis_compute()
        ):
            self._schedule_downcore_compute(delay_ms=50)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="core-stitch")
    p.add_argument("top", nargs="?", help="TOP series folder")
    p.add_argument("btm", nargs="?", help="BTM series folder")
    args = p.parse_args(argv)
    app = StitchGui(
        path_a=Path(args.top) if args.top else None,
        path_b=Path(args.btm) if args.btm else None,
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
