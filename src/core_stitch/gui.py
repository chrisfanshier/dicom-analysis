"""Coron TOP/BTM stitch GUI."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from core_stitch.export import _crop_box as _export_crop_box
from core_stitch.export import export_accept
from core_stitch.overlap import FUSE_MODES
from core_stitch.scale import auto_bounds
from core_stitch.header_state import HeaderState, export_header
from core_stitch.single import SingleState, export_single, prepare_single
from core_stitch.scale_gui import CoreGuides
from core_stitch.worklist_gui import WorklistPanel
from core_stitch.pipeline import (
    StitchState,
    coarse_align,
    ensure_posed,
    masked_hu_values,
    preview_arrays,
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
        # Series picked from the catalog worklist (None = largest group in the folder)
        self.uid_a: str | None = None
        self.uid_b: str | None = None
        self.current_job = None  # worklist Job being stitched, if any
        self.worklist: WorklistPanel | None = None
        self.state: StitchState | None = None
        # Any single-image mode: a scan with only one view (SingleState) or the scanner-geometry
        # stitch (HeaderState). Both answer the same questions (frame, window, crop, preview).
        self.single: SingleState | HeaderState | None = None
        self.mode_note = tk.StringVar(value="")
        self.use_scanner_geometry = tk.BooleanVar(value=True)
        self.labels_swapped = False  # True while TOP and BOTTOM are exchanged (the Swap button toggles it)
        self.swap_text = tk.StringVar(value="Using TOP and BOTTOM as labelled in the catalog.")
        self.upside_flag = False  # True while the picture is turned 180 degrees (the Rotate button toggles it)
        self.turn_a = False  # the series labelled TOP in the catalog was scanned upside down: turn it around in place
        self.turn_b = False  # the same for the series labelled BTM
        self.turn_text = tk.StringVar(value="Neither half turned.")
        self.upside_text = tk.StringVar(value="Picture as scanned.")
        self.straighten = tk.BooleanVar(value=False)  # rotate so the core is exactly vertical (interpolates pixels)
        self.hdr_along = tk.DoubleVar(value=0.0)
        self.hdr_across = tk.DoubleVar(value=0.0)
        self.hdr_depth = tk.DoubleVar(value=0.0)
        self.hdr_whole_pixel = tk.BooleanVar(value=True)
        self.hdr_info = tk.StringVar(value="")
        self.hdr_check = tk.StringVar(value="")
        self.hdr_quality = tk.StringVar(value="")
        self.hdr_xsec_label = tk.StringVar(value="")
        self.hdr_xsec_frac = tk.DoubleVar(value=0.5)
        self._xsec_photo = None
        self._xsec_job: str | None = None
        self.hdr_refine_note = tk.StringVar(value="")
        self.hdr_refine_result = tk.StringVar(value="")
        self._hdr_job: str | None = None
        self._scale = 1.0  # canvas px per image px
        self._fit = 1.0
        self._guide_offset = 0
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
        self.win_lo = tk.DoubleVar(value=0.0)
        self.win_hi = tk.DoubleVar(value=1.0)
        self.win_label = tk.StringVar(value="Window: —")
        self._z_job: str | None = None
        self._win_job: str | None = None
        self._z_load_job: str | None = None
        self._hist_range: tuple[float, float] = (0.0, 1.0)
        self._hist_counts: np.ndarray | None = None
        self._hist_edges: np.ndarray | None = None

        # The Export tab is built before the preview canvas, so the guides get theirs later.
        self.guides = CoreGuides(None)  # type: ignore[arg-type]
        self.show_guides = tk.BooleanVar(value=True)
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
        tab_worklist = ttk.Frame(nb, padding=6)
        tab_align = ttk.Frame(nb, padding=8)
        tab_display = ttk.Frame(nb, padding=8)
        tab_export = ttk.Frame(nb, padding=8)
        nb.add(tab_worklist, text="Worklist")
        nb.add(tab_align, text="Align")
        nb.add(tab_display, text="Display")
        nb.add(tab_export, text="Export")
        self._notebook = nb
        self._left = left
        self._tab_align = tab_align
        self._tab_display = tab_display
        self.worklist = WorklistPanel(tab_worklist, self)
        self.worklist.pack(fill=tk.BOTH, expand=True)
        tab_worklist.columnconfigure(0, weight=1)
        tab_worklist.rowconfigure(0, weight=1)
        nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._on_tab_changed()

        self._build_align_tab(self._scrollable(tab_align))
        self._build_display_tab(tab_display)
        self._build_export_tab(tab_export)

        ttk.Label(left, textvariable=self.status, wraplength=340).grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )

        # ----- Right image pane -----
        canvas_frame = ttk.Frame(root, padding=(4, 8, 8, 8))
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, background="#111", highlightthickness=0)
        self.guides.canvas = self.canvas
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

        self.canvas.bind("<Configure>", lambda _e: self._schedule_refresh())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_mousewheel_linux(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._on_mousewheel_linux(e, -1))
        self.canvas.bind("<ButtonPress-2>", self._pan_press)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        # same pointer behaviour as the multi-view tool: a left press near a guide moves it,
        # a left press anywhere else pans
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Motion>", self.guides.motion)

    def _scrollable(self, tab: ttk.Frame) -> ttk.Frame:
        """A vertically scrolling area inside ``tab``: the Align controls are taller than a small window."""
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        cv = tk.Canvas(tab, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=cv.yview)
        cv.configure(yscrollcommand=bar.set)
        cv.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        inner = ttk.Frame(cv)
        win = cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))

        def wheel(e: tk.Event) -> None:
            if inner.winfo_reqheight() > cv.winfo_height():
                cv.yview_scroll(-1 if e.delta > 0 else 1, "units")

        def over(_e: tk.Event) -> None:
            cv.bind_all("<MouseWheel>", wheel)

        def away(e: tk.Event) -> None:
            under = tab.winfo_containing(e.x_root, e.y_root)
            if under is not None and str(under).startswith(str(tab)):
                return  # still over the tab (moved onto a control)
            cv.unbind_all("<MouseWheel>")

        tab.bind("<Enter>", over)
        tab.bind("<Leave>", away)
        return inner

    def _build_align_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent, textvariable=self.mode_note, wraplength=320, foreground="#b45309", font=("", 9, "bold")
        ).pack(anchor=tk.W, pady=(0, 6))

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

        ttk.Checkbutton(
            parent,
            text="Use scanner geometry for worklist pairs (SimpleITK)",
            variable=self.use_scanner_geometry,
        ).pack(anchor=tk.W, pady=(0, 4))
        ttk.Button(parent, text="Load pair with scanner geometry", command=self.load_header_pair).pack(
            fill=tk.X, pady=(0, 4)
        )
        ttk.Button(parent, text="Swap TOP and BOTTOM (re-order the halves)", command=self.swap_top_bottom).pack(fill=tk.X)
        ttk.Label(parent, textvariable=self.swap_text, wraplength=320, font=("", 8)).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(
            parent,
            text="Use when the two series are labelled backwards but each is the right way up. The half labelled TOP "
            "goes below the other and the join is found from the images (about a minute); the headers cannot place "
            "them in that order. Click again to go back to the scanner positions. Recorded in the sidecar (top_bottom_swapped).",
            wraplength=320, font=("", 8),
        ).pack(anchor=tk.W, pady=(0, 6))
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 0))
        ttk.Button(row, text="Turn TOP half 180\u00b0", command=lambda: self.turn_half("top")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Turn BOTTOM half 180\u00b0", command=lambda: self.turn_half("bottom")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(parent, textvariable=self.turn_text, wraplength=320, font=("", 8)).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(
            parent,
            text="Use when only one half was scanned upside down. That half is turned around in place (lossless) and "
            "the join is then found from the images (about a minute). Click again to undo. Recorded in the sidecar (turned_180).",
            wraplength=320, font=("", 8),
        ).pack(anchor=tk.W, pady=(0, 6))
        ttk.Button(parent, text="Rotate 180 degrees (core is upside down)", command=self.toggle_upside_down).pack(fill=tk.X)
        ttk.Label(parent, textvariable=self.upside_text, wraplength=320, font=("", 8)).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(
            parent,
            text="Use when the core's top is at the bottom of the picture. Turns the whole picture around "
            "(top becomes bottom, left becomes right); no pixel value changes. Click again to undo. "
            "Recorded in the sidecar (rotate_180).",
            wraplength=320, font=("", 8),
        ).pack(anchor=tk.W, pady=(0, 6))
        ttk.Checkbutton(
            parent,
            text="Straighten the core (rotate the picture)",
            variable=self.straighten,
            command=self._on_straighten,
        ).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="Makes the core exactly vertical. This interpolates the pixels, so the values change slightly; "
            "off keeps them as scanned. The measured tilt is recorded either way (sidecar: straighten). "
            "For stitched pairs it is applied when you export.",
            wraplength=320, font=("", 8),
        ).pack(anchor=tk.W, pady=(0, 8))
        self._old_align = ttk.Frame(parent)
        self._hdr_align = ttk.Frame(parent)
        self._build_old_align(self._old_align)
        self._build_header_align(self._hdr_align)
        self._set_align_mode("none")  # nothing loaded yet: show no alignment controls (the older ones appear only if used)

    def _set_align_mode(self, mode: str) -> None:
        """Show the controls for the current mode: older mask method, or scanner geometry."""
        self._old_align.pack_forget()
        self._hdr_align.pack_forget()
        if mode == "old":
            self._old_align.pack(fill=tk.X)
        elif mode == "header":
            self._hdr_align.pack(fill=tk.X)

    def _build_header_align(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        self._hdr_check_label = ttk.Label(parent, textvariable=self.hdr_check, wraplength=320, font=("", 9, "bold"))
        self._hdr_check_label.pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(parent, textvariable=self.hdr_info, wraplength=320, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))

        box = ttk.LabelFrame(parent, text="Shift bottom half from scanner position (mm)", padding=6)
        box.pack(fill=tk.X)
        rows = (
            ("Along the core (+ = down)", self.hdr_along),
            ("Across the core", self.hdr_across),
            ("Through the stack", self.hdr_depth),
        )
        for label, var in rows:
            row = ttk.Frame(box)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=24).pack(side=tk.LEFT)
            sb = ttk.Spinbox(row, from_=-1000, to=1000, increment=0.5, width=8, textvariable=var, command=self._hdr_apply_soon)
            sb.pack(side=tk.LEFT)
            sb.bind("<Return>", lambda _e: self._hdr_apply_soon())
            sb.bind("<FocusOut>", lambda _e: self._hdr_apply_soon())
        ttk.Label(
            box, text="Arrow keys: 1 mm (Shift = 5). Left/Right = across, Up/Down = along.", font=("", 8), wraplength=320
        ).pack(anchor=tk.W, pady=(4, 0))
        ttk.Button(box, text="Reset to scanner position", command=self._hdr_reset).pack(anchor=tk.W, pady=(4, 0))

        ttk.Checkbutton(
            parent,
            text="Whole-pixel placement when both halves share a pixel size",
            variable=self.hdr_whole_pixel,
            command=self._hdr_apply_soon,
        ).pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(
            parent,
            text="On: every voxel stays as scanned (placement good to half a pixel). Off: sub-pixel "
            "placement, which interpolates the bottom half.",
            wraplength=320, font=("", 8),
        ).pack(anchor=tk.W)
        rj = ttk.Frame(parent)
        rj.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(rj, text="Refine join", command=self._hdr_refine).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rj, text="Undo refine", command=self._hdr_undo_refine).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(parent, textvariable=self.hdr_refine_note, wraplength=320, font=("", 8)).pack(anchor=tk.W)
        ttk.Label(parent, textvariable=self.hdr_refine_result, wraplength=320, font=("", 9, "bold")).pack(anchor=tk.W, pady=(2, 0))
        ttk.Button(parent, text="Re-align from image (mask + profile)", command=self._hdr_realign).pack(
            fill=tk.X, pady=(10, 0)
        )
        ttk.Label(
            parent,
            text="Use this only if the scanner position is way off. It runs the older mask + profile "
            "alignment (about a minute) and starts from its answer.",
            wraplength=320, font=("", 8),
        ).pack(anchor=tk.W)
        self._build_header_checks(parent)

    def _build_header_checks(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Check the join", padding=6)
        box.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(
            box,
            text="Cross-section across the core at a row inside the overlap (top magenta, bottom green). "
            "When the join is right the two outlines coincide; a depth error shows them displaced.",
            wraplength=320, font=("", 8),
        ).pack(anchor=tk.W)
        self._xsec_canvas = tk.Canvas(box, width=320, height=210, bg="#0b0b0b", highlightthickness=0)
        self._xsec_canvas.pack(pady=(4, 2))
        ttk.Label(box, textvariable=self.hdr_xsec_label, wraplength=320, font=("", 8)).pack(anchor=tk.W)
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Row in overlap").pack(side=tk.LEFT)
        ttk.Scale(row, from_=0.0, to=1.0, variable=self.hdr_xsec_frac, command=lambda _v: self._hdr_xsec_soon()).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )
        ttk.Button(box, text="Check join quality", command=self._hdr_check_quality).pack(fill=tk.X, pady=(6, 0))
        ttk.Label(box, textvariable=self.hdr_quality, wraplength=320, font=("", 9)).pack(anchor=tk.W, pady=(4, 0))

    def _build_old_align(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Button(parent, text="Run coarse align (older mask method)", command=self.run_coarse).pack(
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
        ttk.Label(zm, text="Wheel = zoom at pointer · drag = pan · drag a cyan/amber guide to move it", font=("", 8)).pack(
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

    def _build_export_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text="Exports work anytime after Align.",
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

        ttk.Label(parent, text="Core ends (scale)", font=("", 9, "bold")).pack(
            anchor=tk.W, pady=(12, 0)
        )
        ttk.Checkbutton(
            parent, text="Show draggable guides on preview", variable=self.show_guides,
            command=self._on_show_guides,
        ).pack(anchor=tk.W)
        self.guides.controls(parent, wraplength=320).pack(fill=tk.X, pady=(2, 0))
        ttk.Button(parent, text="Reset guides to auto", command=self._reset_guides).pack(
            anchor=tk.W, pady=(2, 0)
        )
        ttk.Label(
            parent,
            text="Guides are not drawn into the images. Their positions go in the JSON "
            "sidecar (core_scale).",
            wraplength=320, font=("", 8),
        ).pack(anchor=tk.W, pady=(2, 0))

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

    def _on_tab_changed(self, _event=None) -> None:
        """The Worklist tree needs more room than the other tabs."""
        try:
            text = self._notebook.tab(self._notebook.select(), "text")
        except tk.TclError:
            return
        self._left.configure(width=560 if text == "Worklist" else 360)

    def browse_a(self) -> None:
        p = filedialog.askdirectory(title="Select TOP (A) series folder")
        if p:
            self.path_a = Path(p)
            self.path_a_var.set(p)
            self.uid_a = None
            self.current_job = None

    def browse_b(self) -> None:
        p = filedialog.askdirectory(title="Select BTM (B) series folder")
        if p:
            self.path_b = Path(p)
            self.path_b_var.set(p)
            self.uid_b = None
            self.current_job = None

    def load_pair_job(self, job) -> None:
        """Load a worklist TOP/BTM pair into the stitch workflow (Align tab)."""
        self.current_job = job
        self.path_a, self.path_b = job.a.path, job.b.path
        self.uid_a, self.uid_b = job.a.series_uid, job.b.series_uid
        self.path_a_var.set(str(job.a.path))
        self.path_b_var.set(str(job.b.path))
        self._notebook.select(self._tab_align)
        if self.use_scanner_geometry.get():
            self.load_header_pair(job)
            return
        self.run_coarse()
        if self.state is None:
            self.current_job = None

    # ----- scanner-geometry (SimpleITK) stitch -----

    def _set_upside(self, on: bool) -> None:
        self.upside_flag = bool(on)
        self.upside_text.set(
            "TURNED 180 degrees: the whole picture is rotated around (lossless)." if self.upside_flag else "Picture as scanned."
        )

    def toggle_upside_down(self) -> None:
        """Turn the whole picture 180 degrees, or back.  Instant and lossless."""
        st = self.single
        if st is None or self.state is not None:
            messagebox.showinfo("Rotate 180 degrees", "Load a pair (scanner geometry) or a single scan first.")
            return
        on = not self.upside_flag
        try:
            st.set_upside_down(on)
        except Exception as exc:
            messagebox.showerror("Rotate 180 degrees", str(exc))
            return
        self._set_upside(on)
        self.guides.reset()  # the end guides are measured on the picture as shown, so they start again
        if isinstance(st, HeaderState):
            self._hdr_sync()
        self._sync_single_controls()
        self._sync_window_from_state()
        self._schedule_refresh()
        self.status.set(
            "Picture turned 180 degrees: the core's top end is now at the top of the picture. Place the end guides, then Export."
            if on else "Picture back as scanned."
        )

    def _set_swapped(self, swapped: bool) -> None:
        self.labels_swapped = bool(swapped)
        self.swap_text.set(
            "SWAPPED: the series labelled TOP in the catalog now sits BELOW the other (each half stays upright), "
            "placed by image matching."
            if self.labels_swapped else "Using TOP and BOTTOM as labelled in the catalog."
        )

    def swap_top_bottom(self) -> None:
        """Reload the current pair with TOP and BOTTOM exchanged (and back again on the next click)."""
        if not self.path_a or not self.path_b:
            messagebox.showinfo("Swap TOP and BOTTOM", "Choose a pair from the Worklist first.")
            return
        self._set_swapped(not self.labels_swapped)
        if not self._reload_and_place("Swapped" if self.labels_swapped else "Swapped back"):
            self._set_swapped(not self.labels_swapped)  # the load failed: leave the label as it was

    def _set_turned(self, a: bool, b: bool) -> None:
        self.turn_a, self.turn_b = bool(a), bool(b)
        parts = [n for n, f in (("the series labelled TOP in the catalog", self.turn_a), ("the series labelled BTM", self.turn_b)) if f]
        self.turn_text.set(("TURNED 180 degrees in place: " + " and ".join(parts) + ".") if parts else "Neither half turned.")

    def turn_half(self, role: str) -> None:
        """Turn the half now in the TOP or BOTTOM role 180 degrees in place, or back, and re-place the pair by the images."""
        if not self.path_a or not self.path_b:
            messagebox.showinfo("Turn a half", "Choose a pair from the Worklist first.")
            return
        # the flags follow the series, so find which catalog series sits in that role now
        is_a = (role == "top") != self.labels_swapped
        before = (self.turn_a, self.turn_b)
        if is_a:
            self._set_turned(not self.turn_a, self.turn_b)
        else:
            self._set_turned(self.turn_a, not self.turn_b)
        turning = (self.turn_a if is_a else self.turn_b)
        if not self._reload_and_place(f"The {role} half is " + ("turned 180 degrees" if turning else "back as scanned")):
            self._set_turned(*before)

    def _reload_and_place(self, what: str) -> bool:
        """Reload the current pair with the swap / turn settings, and place it: from the headers if the pair is as
        scanned, otherwise by matching the images (the headers no longer describe a swapped or turned half)."""
        before = self.single
        self.load_header_pair()
        if self.single is before:
            return False  # the load failed
        st = self.single
        names = f"TOP is '{st.top.series.series_description.strip()}', BOTTOM is '{st.bottom.series.series_description.strip()}'."
        if self.labels_swapped or self.turn_a or self.turn_b:
            self.status.set(f"{what}. Placing the halves by image matching (about a minute)...")
            self.update_idletasks()
            # a swapped pair has its order fixed: the new bottom half goes below the new top half
            ok = self._hdr_realign(require_below=self.labels_swapped)
            self.status.set((f"{what}. " + names + " Placed by image matching; check the join.") if ok
                            else f"{what}, but the image matching failed; the halves are still at their scanner positions.")
            corr = self.single.join_correlation() if ok else float("nan")
            if ok and corr == corr and corr < 0.8:
                messagebox.showwarning(
                    "Check the join",
                    f"In this arrangement the two halves match poorly (join correlation {corr:.2f}).\n\n"
                    "If the series are not really swapped or upside down, click the same button again to undo it.",
                )
        else:
            self.status.set(f"{what}. " + names + " The halves are at their scanner positions.")
        return True

    def _on_straighten(self) -> None:
        on = bool(self.straighten.get())
        st = self.single
        if st is None or self.state is not None:
            return
        if isinstance(st, HeaderState):
            st.straighten = on  # a stitched pair is rotated once, when exporting
            self._hdr_sync()  # the panel text states the tilt and whether it will be straightened
            self.status.set(
                "Straighten ON: the stitch will be rotated upright when you export (the preview itself is not rotated)." if on
                else "Straighten OFF: the stitch will be exported as scanned."
            )
            return
        try:
            st.set_straighten(on)
        except Exception as exc:
            messagebox.showerror("Straighten", str(exc))
            return
        self.guides.reset()  # the rotated picture has a different size, so the end guides start again
        self._sync_single_controls()
        self._sync_window_from_state()
        self._schedule_refresh()
        note = f"Core tilt {st.half.tilt_deg:+.2f} deg: " + (
            "STRAIGHTENED (the slice was rotated; pixels interpolated)." if on else "not straightened (pixels exactly as scanned)."
        )
        self.mode_note.set(note)
        self.status.set(note)

    def _hdr_active(self) -> bool:
        return isinstance(self.single, HeaderState) and self.state is None

    def load_header_pair(self, job=None) -> None:
        """Show TOP + BTM where the scanner put them (DICOM header geometry)."""
        if job is not None:
            self.current_job = job
            self.path_a, self.path_b = job.a.path, job.b.path
            self.uid_a, self.uid_b = job.a.series_uid, job.b.series_uid
            self.path_a_var.set(str(job.a.path))
            self.path_b_var.set(str(job.b.path))
            self._set_swapped(False)  # a newly chosen pair starts with the catalog's labels
            self._set_upside(False)
            self._set_turned(False, False)
        if not self.path_a or not self.path_b:
            messagebox.showinfo("Need folders", "Select both TOP and BTM series folders.")
            return
        self.status.set("Loading both halves and placing them from the scanner geometry...")
        self.update_idletasks()
        placement = "whole_pixel" if self.hdr_whole_pixel.get() else "subpixel"
        try:
            st = HeaderState.load(
                self.path_a, self.uid_a, self.path_b, self.uid_b, placement=placement, swapped=self.labels_swapped,
                top_turned=self.turn_a, bottom_turned=self.turn_b,
            )
            st.straighten = bool(self.straighten.get())
            st.set_upside_down(self.upside_flag)
        except Exception as exc:
            messagebox.showwarning("Scanner geometry not usable for this pair", str(exc))
            self.status.set(f"Load failed: {exc}")
            if job is not None:
                self.current_job = None
            return
        self.state = None
        self.single = st
        self.mode_note.set("")
        self.guides.reset()
        self.zoom_var.set(1.0)
        self._set_align_mode("header")
        self._hdr_sync()
        self._sync_single_controls()
        self._sync_window_from_state()
        self._notebook.select(self._tab_align)
        self.status.set("Placed from the scanner geometry. Check the join, shift if needed, then Export.")
        self._schedule_refresh()

    def _hdr_sync(self) -> None:
        """Copy the state's shift and text into the controls."""
        st = self.single
        if not isinstance(st, HeaderState):
            return
        self.hdr_along.set(round(st.move_mm[0], 3))
        self.hdr_across.set(round(st.move_mm[1], 3))
        self.hdr_depth.set(round(st.move_mm[2], 3))
        self.hdr_info.set(st.summary())
        self.hdr_refine_note.set(st.refine_note())
        self.hdr_quality.set("")  # a new placement: any earlier join-quality result no longer applies
        self._hdr_xsec_soon()
        text, ok = st.check_text()
        self.hdr_check.set(text)
        self._hdr_check_label.configure(foreground="#15803d" if ok else "#b45309")

    def _hdr_apply_soon(self) -> None:
        if self._hdr_job is not None:
            self.after_cancel(self._hdr_job)
        self._hdr_job = self.after(250, self._hdr_apply)

    def _hdr_apply(self) -> None:
        self._hdr_job = None
        st = self.single
        if not self._hdr_active():
            return
        try:
            a, b, d = float(self.hdr_along.get()), float(self.hdr_across.get()), float(self.hdr_depth.get())
        except (TypeError, ValueError, tk.TclError):
            return
        placement = "whole_pixel" if self.hdr_whole_pixel.get() else "subpixel"
        if placement == st.placement and all(abs(x - y) < 1e-6 for x, y in zip((a, b, d), st.move_mm)):
            return  # nothing changed (e.g. focus left a box without an edit)
        self.status.set("Re-placing the bottom half...")
        self.update_idletasks()
        self.hdr_refine_result.set("")
        try:
            st.placement = placement
            moved = any(abs(x) > 1e-9 for x in (a, b, d))
            st.set_move(a, b, d, source=None if moved else "scanner header")
        except Exception as exc:
            messagebox.showerror("Shift", str(exc))
            self.status.set(f"Shift failed: {exc}")
            return
        self._hdr_sync()
        self._sync_single_controls()
        self.status.set("Updated.")
        self._schedule_refresh()

    def _hdr_reset(self) -> None:
        if not self._hdr_active():
            return
        self.hdr_refine_result.set("")
        self.single.set_move(0.0, 0.0, 0.0, source="scanner header")
        self._hdr_sync()
        self._sync_single_controls()
        self._schedule_refresh()

    def _hdr_refine(self) -> None:
        if not self._hdr_active():
            messagebox.showinfo("Refine join", "Load a pair first.")
            return
        self.status.set("Refining the join (registration over the overlap, several seconds)...")
        self.update_idletasks()
        try:
            out = self.single.refine()
        except Exception as exc:
            messagebox.showerror("Refine failed", str(exc))
            self.status.set(f"Refine failed: {exc}")
            return
        self._hdr_sync()
        self._sync_single_controls()
        if out.applied:
            self.hdr_refine_result.set(
                f"Refined: bottom half moved {out.along_mm:+.2f} mm along, {out.across_mm:+.2f} mm across; "
                f"join correlation as placed {out.score_before:.4f} -> {out.score_after:.4f}."
            )
            self.status.set("Refined. Check the join, then Export.")
        else:
            self.hdr_refine_result.set(out.note + (f" (correlation {out.score_before:.4f}.)" if out.score_before == out.score_before else ""))
            self.status.set("Refine made no change.")
        self._schedule_refresh()

    def _hdr_xsec_soon(self) -> None:
        if self._xsec_job is not None:
            self.after_cancel(self._xsec_job)
        self._xsec_job = self.after(200, self._hdr_xsec_refresh)

    def _hdr_xsec_refresh(self) -> None:
        self._xsec_job = None
        if not self._hdr_active():
            return
        c = self._xsec_canvas
        c.delete("all")
        try:
            rgb, info = self.single.cross_section(float(self.hdr_xsec_frac.get()))
        except Exception as exc:
            c.create_text(160, 105, text=str(exc), fill="#bbb", width=290)
            self.hdr_xsec_label.set("")
            return
        h, w = rgb.shape[:2]
        # one slice is 2 mm thick but a pixel is under 1 mm: stretch depth so the picture is to scale
        tall = max(int(round(h * info["slice_step_over_pixel"])), 1)
        scale = min(316.0 / w, 206.0 / tall)
        img = Image.fromarray(rgb).resize((max(int(w * scale), 1), max(int(tall * scale), 1)), Image.NEAREST)
        self._xsec_photo = ImageTk.PhotoImage(img)
        c.create_image(160, 105, image=self._xsec_photo)
        ov = info["outline_overlap"]
        self.hdr_xsec_label.set(
            f"Row {info['row']} of the stitched picture. The two outlines overlap by {ov * 100:.0f}% (100% = coincide)."
            if ov == ov else f"Row {info['row']}."
        )

    def _hdr_check_quality(self) -> None:
        if not self._hdr_active():
            messagebox.showinfo("Join quality", "Load a pair first.")
            return
        self.status.set("Checking the join over the whole overlap...")
        self.update_idletasks()
        try:
            q = self.single.join_quality()
        except Exception as exc:
            messagebox.showerror("Join quality", str(exc))
            self.status.set(f"Join check failed: {exc}")
            return
        if not q.get("evaluated"):
            self.hdr_quality.set("Could not evaluate: " + q.get("reason", "unknown"))
        elif q["consistent"]:
            b = q["best_shift"]
            self.hdr_quality.set(
                f"Join looks consistent: correlation {q['correlation']:.4f} over {q['voxels']:,} voxels. "
                f"No nearby position is better (best shifts: depth {b['depth_slices']:+d} slices, "
                f"along {b['along_px']:+d} px, across {b['across_px']:+d} px)."
            )
        else:
            self.hdr_quality.set("WARNING: " + "; ".join(q["flags"]) + ".")
        self.status.set("Join check done.")

    def _hdr_undo_refine(self) -> None:
        if not self._hdr_active() or not self.single.undo_refine():
            self.status.set("Nothing to undo: no refine since the last change.")
            return
        self.hdr_refine_result.set("Refine undone: back to the position before it.")
        self._hdr_sync()
        self._sync_single_controls()
        self.status.set("Refine undone.")
        self._schedule_refresh()

    def _hdr_realign(self, require_below: bool = False) -> bool:
        if not self._hdr_active():
            messagebox.showinfo("Re-align", "Load a pair first.")
            return False
        self.status.set("Re-aligning from the images (mask + profile)... this takes about a minute.")
        self.update_idletasks()
        try:
            self.single.realign_from_image(require_below=require_below)
        except Exception as exc:
            messagebox.showerror("Re-align failed", str(exc))
            self.status.set(f"Re-align failed: {exc}")
            return False
        self._hdr_sync()
        self._sync_single_controls()
        self.status.set("Placed from the image alignment. Check the join, then Export.")
        self._schedule_refresh()
        return True

    def load_single_job(self, job) -> None:
        """A scan with only one view: same window, no join. Window, ends, then Export."""
        self.current_job = job
        self.status.set("Loading single series...")
        self.update_idletasks()
        try:
            st = prepare_single(job.a.path, job.a.series_uid, straighten=bool(self.straighten.get()))
            self._set_upside(False)
            self._single_tilt_note = f"Core tilt {st.half.tilt_deg:+.2f} deg: " + (
                "STRAIGHTENED (pixels interpolated)." if st.straighten else "not straightened (pixels exactly as scanned)."
            )
        except Exception as exc:
            messagebox.showerror("Single scan", str(exc))
            self.status.set(f"Single load failed: {exc}")
            self.current_job = None
            return
        self.state = None
        self.single = st
        self.path_a_var.set(str(job.a.path))
        self.path_b_var.set("")
        self.mode_note.set(
            "Only 1 view found in this scan, so there is nothing to stitch. "
            "Set the window, place the core ends, then Export. " + getattr(self, "_single_tilt_note", "")
        )
        self.guides.reset()
        self.zoom_var.set(1.0)
        self._set_align_mode("none")
        self._sync_single_controls()
        self._sync_window_from_state()
        self._notebook.select(self._tab_display)
        self.status.set("Only 1 view found - nothing to stitch. Set the window, place the core ends, then Export.")
        self._schedule_refresh()

    def _sync_single_controls(self) -> None:
        if self.single is None:
            return
        n = self.single.n_frames
        self.z_spin.configure(from_=0, to=max(0, n - 1))
        self.z_var.set(self.single.frame)
        self.z_label.set(f"Frame: {self.single.frame} / {n - 1}")

    def run_coarse(self) -> None:
        if not self.path_a or not self.path_b:
            messagebox.showinfo("Need folders", "Select both TOP and BTM series folders.")
            return
        self.guides.reset()
        self.status.set("Running coarse align…")
        self.update_idletasks()
        try:
            self.state = coarse_align(self.path_a, self.path_b, self.uid_a, self.uid_b)
        except Exception as exc:
            messagebox.showerror("Align failed", str(exc))
            self.status.set(f"Failed: {exc}")
            return
        self.single = None
        self.mode_note.set("")
        self._set_align_mode("old")
        self.flip_var.set(False)
        self.roll_var.set(0.0)
        self.fuse_mode.set(self.state.fuse_mode)
        self.match_hu_var.set(self.state.match_hu)
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
        if self.single is not None and self.state is None:
            try:
                z = int(self.z_var.get())
            except (TypeError, ValueError, tk.TclError):
                return
            self.single.set_frame(z)
            self._sync_single_controls()
            self._schedule_refresh()
            return
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

    def _z_mid(self) -> None:
        if self.single is not None and self.state is None:
            self.z_var.set(self.single.n_frames // 2)
            self._apply_z()
            return
        if self.state is None:
            return
        posed = ensure_posed(self.state)
        self.z_var.set(posed.n_slices // 2)
        self._apply_z()

    def _bump_z(self, delta: int) -> None:
        if self.single is not None and self.state is None:
            self.z_var.set(int(np.clip(int(self.z_var.get()) + delta, 0, self.single.n_frames - 1)))
            self._apply_z()
            return
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
        if self._hdr_active():
            self.status.set("Overlap fuse is fixed in scanner-geometry mode: the average, with edge rows trimmed.")
            return
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
        if self._hdr_active():
            self.hdr_across.set(round(float(self.hdr_across.get()) + dx, 3))
            self.hdr_along.set(round(float(self.hdr_along.get()) + dy, 3))
            self._hdr_apply_soon()
            return
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

    def _has_image(self) -> bool:
        return self.state is not None or self.single is not None

    def _win_obj(self):
        """The object that holds window_lo / window_hi (stitch state or single scan)."""
        return self.state if self.state is not None else self.single

    def _hu_values(self) -> np.ndarray:
        if self.state is None and self.single is not None:
            return self.single.hu_values()
        return masked_hu_values(self.state)

    def _sync_window_from_state(self) -> None:
        if not self._has_image():
            return
        if self.state is None:
            lo, hi = self.single.window_limits()
        else:
            lo, hi = window_limits(self.state)
        vals = self._hu_values()
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
        self._win_obj().window_lo = lo
        self._win_obj().window_hi = hi
        self.win_label.set(f"Window: [{lo:.1f}, {hi:.1f}] HU")
        counts, edges = np.histogram(vals, bins=128, range=(vmin, vmax))
        self._hist_counts = counts.astype(np.float64)
        self._hist_edges = edges
        self._draw_histogram()

    def _auto_window(self) -> None:
        if not self._has_image():
            return
        if self.state is None:
            self.single.auto_window()
        else:
            set_auto_window(self.state)
        self._sync_window_from_state()
        if self.state is None or not self.align_var.get():
            self._schedule_refresh()

    def _on_window_slide(self, _value: str | None = None) -> None:
        if self._win_job is not None:
            self.after_cancel(self._win_job)
        self._win_job = self.after(40, self._apply_window)

    def _apply_window(self) -> None:
        self._win_job = None
        if not self._has_image():
            return
        lo = float(self.win_lo.get())
        hi = float(self.win_hi.get())
        if hi < lo:
            lo, hi = hi, lo
            self.win_lo.set(lo)
            self.win_hi.set(hi)
        self._win_obj().window_lo = lo
        self._win_obj().window_hi = hi
        self.win_label.set(f"Window: [{lo:.1f}, {hi:.1f}] HU")
        self._draw_histogram()
        if self.state is None or not self.align_var.get():
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
        if not self._has_image():
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
        if not self._has_image() or self._hist_drag_which is None:
            return
        w = max(self.hist_canvas.winfo_width(), 2)
        hu = self._x_to_hu(float(event.x), w)
        if self._hist_drag_which == "lo":
            self.win_lo.set(min(hu, float(self.win_hi.get()) - 1.0))
        else:
            self.win_hi.set(max(hu, float(self.win_lo.get()) + 1.0))
        self._apply_window()

    def _on_show_guides(self) -> None:
        self.guides.enabled = bool(self.show_guides.get())
        self._schedule_refresh()

    def _reset_guides(self) -> None:
        self.guides.reset()
        self._schedule_refresh()

    def _expected_length_cm(self) -> float | None:
        job = self.current_job
        if job is None or self.worklist is None or self.worklist.catalog is None:
            return None
        g = self.worklist.catalog.groups.get(job.stitch_group)
        return g.expected_length_cm if g else None

    def accept_export(self) -> None:
        hdr = self._hdr_active()
        single = self.single is not None and self.state is None and not hdr
        if self.state is None and not single and not hdr:
            messagebox.showinfo("Export", "Run coarse align first.")
            return
        jpeg = bool(self.export_jpeg.get())
        tiff = bool(self.export_tiff.get())
        js = bool(self.export_json.get())
        # a single view has no DICOM stack or NumPy fusion to write
        dicom = False if single else bool(self.export_dicom.get())
        npy = False if (single or hdr) else bool(self.export_npy.get())
        if not any((jpeg, tiff, dicom, npy, js)):
            messagebox.showinfo("Export", "Select at least one export type.")
            return
        if not single and not hdr and self.state.low_confidence:
            if not messagebox.askyesno(
                "Low confidence",
                "Overlap is marked low confidence." + chr(10) + "Export anyway?",
            ):
                return
        job = self.current_job if self.worklist is not None else None
        target = None
        if job is not None:
            target = self.worklist.export_target(job)
            out = str(target.out_dir)
        else:
            out = filedialog.askdirectory(title="Export folder")
            if not out:
                return
        scale: dict = {}
        self.status.set("Exporting selected products…")
        self.update_idletasks()
        try:
            if hdr:
                paths = export_header(
                    self.single,
                    out,
                    stem=target.stem if target else "stitch",
                    extra_meta=target.extra_meta if target else None,
                    jpeg=jpeg,
                    tiff=tiff,
                    dicom=dicom,
                    json_sidecar=js,
                    core_bounds=self.guides.bounds(),
                    scale_out=scale,
                )
            elif single:
                paths = export_single(
                    self.single,
                    out,
                    stem=target.stem if target else "single",
                    extra_meta=target.extra_meta if target else None,
                    jpeg=jpeg,
                    tiff=tiff,
                    json_sidecar=js,
                    core_bounds=self.guides.bounds(),
                    scale_out=scale,
                )
            else:
                paths = export_accept(
                    self.state,
                    out,
                    stem=target.stem if target else "stitch",
                    extra_meta=target.extra_meta if target else None,
                    jpeg=jpeg,
                    tiff=tiff,
                    dicom=dicom,
                    npy=npy,
                    json_sidecar=js,
                    core_bounds=self.guides.bounds(),
                    scale_out=scale,
                )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            self.status.set(f"Export failed: {exc}")
            return
        lines = [str(p) for p in paths.values()]
        self.status.set(f"Exported {len(paths)} item(s) → {out}")
        messagebox.showinfo("Exported", "Wrote:" + chr(10) + chr(10).join(lines))
        if job is not None and target is not None:
            self.worklist.record_export(job, target, paths, scale, parent=self)

    # ----- zoom / pan (same behaviour as the multi-view tool) -----
    # The preview image is cached; wheel zoom and pan only re-scale it, they never rebuild it.

    DISPLAY_PIXEL_BUDGET = 40_000_000

    def _bump_zoom(self, factor: float) -> None:
        """Buttons and keys: zoom about the middle of the view."""
        self._zoom_at(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, factor)

    def _zoom_at(self, x: int, y: int, factor: float) -> None:
        """Zoom so the image point under pixel (x, y) stays under the pointer."""
        if self._full_im is None:
            return
        old = float(self.zoom_var.get())
        new = float(np.clip(old * factor, 0.25, 16.0))
        if abs(new - old) < 1e-9:
            return
        s = max(self._scale, 1e-9)
        ix, iy = self.canvas.canvasx(x) / s, self.canvas.canvasy(y) / s
        self.zoom_var.set(new)
        self._layout(anchor=(ix, iy, x, y))

    def _zoom_fit(self) -> None:
        self.zoom_var.set(1.0)
        self._layout(reset_scroll=True)

    def _zoom_100(self) -> None:
        """1:1 image pixels (may exceed the window)."""
        if self._full_im is None:
            return
        self.zoom_var.set(float(np.clip(1.0 / max(self._fit, 1e-6), 0.25, 16.0)))
        self._layout(reset_scroll=True)

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, 1.15 if event.delta > 0 else 1 / 1.15)

    def _on_mousewheel_linux(self, event: tk.Event, direction: int) -> None:
        self._zoom_at(event.x, event.y, 1.15 if direction > 0 else 1 / 1.15)

    def _pan_press(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _pan_move(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _canvas_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        if self.guides.press(event):
            return  # grabbed a guide
        self.canvas.scan_mark(event.x, event.y)

    def _canvas_drag(self, event: tk.Event) -> None:
        if self.guides.drag(event):
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _canvas_release(self, event: tk.Event) -> None:
        self.guides.release(event)

    def _schedule_refresh(self) -> None:
        if self._refresh_job is not None:
            self.after_cancel(self._refresh_job)
        self._refresh_job = self.after(40, self.refresh)

    def _layout(
        self,
        anchor: tuple[float, float, int, int] | None = None,
        reset_scroll: bool = False,
    ) -> None:
        """Scale the cached preview to the current zoom and redraw the guides."""
        full = self._full_im
        if full is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        fit = min(cw / full.width, ch / full.height, 1.0)
        zoom = float(self.zoom_var.get())
        scale = fit * zoom
        disp_w = max(1, int(round(full.width * scale)))
        disp_h = max(1, int(round(full.height * scale)))
        if disp_w * disp_h > self.DISPLAY_PIXEL_BUDGET:
            scale *= (self.DISPLAY_PIXEL_BUDGET / (disp_w * disp_h)) ** 0.5
            self.zoom_var.set(scale / fit)
            disp_w = max(1, int(round(full.width * scale)))
            disp_h = max(1, int(round(full.height * scale)))
            self.status.set("Zoom capped for display size.")
        self._fit, self._scale = fit, scale
        im = full.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW, tags=("img",))
        self.canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
        self.guides.draw(0, 0, scale, disp_w, self._guide_offset)
        self.zoom_label.set(f"Zoom: {int(round(scale * 100))}% of image pixels")
        if reset_scroll:
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
        elif anchor is not None:
            ix, iy, x, y = anchor
            self.canvas.xview_moveto(max(0.0, (ix * scale - x) / disp_w))
            self.canvas.yview_moveto(max(0.0, (iy * scale - y) / disp_h))

    def refresh(self) -> None:
        """Rebuild the preview image (and guide frame) from the current state."""
        self._refresh_job = None
        single = self.state is None and self.single is not None
        if self.state is None and not single:
            return
        try:
            if single:
                if isinstance(self.single, HeaderState) and self.align_var.get():
                    arr = self.single.align_preview(float(self.opacity_a.get()), float(self.opacity_b.get()))
                else:
                    arr = self.single.gray_preview()
                box = self.single.crop_box(3.0)
                mask = self.single.core_mask()
                spacing = self.single.spacing_mm
                offset = 0
            else:
                arr = preview_arrays(
                    self.state,
                    align_mode=self.align_var.get(),
                    opacity_a=float(self.opacity_a.get()),
                    opacity_b=float(self.opacity_b.get()),
                    crop_core=bool(self.crop_core_var.get()),
                    pad_mm=3.0,
                )
                box = _export_crop_box(self.state)
                mask = self.state.mask_a | self.state.mask_b
                spacing = self.state.spacing_mm
                offset = 0 if self.crop_core_var.get() else box[0]
        except Exception as exc:
            self.status.set(f"Preview failed: {exc}")
            return
        full = Image.fromarray(arr)
        self._full_im = full
        try:
            auto = auto_bounds(mask, box, spacing)
            self.guides.set_frame(box[1] - box[0], spacing, auto, self._expected_length_cm())
            self._guide_offset = offset
        except Exception as exc:  # guides are advisory; never break the preview
            self.status.set(f"Guides unavailable: {exc}")
        self._layout()
        # Keep histogram markers in sync when slice/fuse changes
        if self._hist_counts is not None:
            try:
                vals = self._hu_values()
                vmin, vmax = self._hist_range
                counts, edges = np.histogram(vals, bins=128, range=(vmin, vmax))
                self._hist_counts = counts.astype(np.float64)
                self._hist_edges = edges
                self._draw_histogram()
            except Exception:
                pass


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
