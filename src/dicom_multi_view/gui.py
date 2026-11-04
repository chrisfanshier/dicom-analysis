"""DICOM multi-view GUI for sediment core sections (no stitching / export)."""

from __future__ import annotations

import argparse
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from dicom_multi_view.render import (
    BASE_MAX_HEIGHT_PX,
    ComposeLayout,
    compose_section_rgb,
    native_isotropic_mm,
    sample_section_image_roi,
    shared_mm_per_px,
    title_strip,
)
from dicom_multi_view.section import (
    LoadedSection,
    auto_window_hu,
    load_section,
    set_plane,
    suggest_labels,
    sync_continuous_tops,
)

# Opt-in high quality: native HU sample of the selected section's visible ROI.
# Overview path stays default (fast pan/zoom). Debounce when High is on.
DETAIL_DEBOUNCE_MS = 120
DETAIL_MAX_PATCH_PX = 6_000_000
DISPLAY_PIXEL_BUDGET = 40_000_000


@dataclass
class SectionSheetLayout:
    """Where a section's image sits on the pre-zoom base sheet."""

    index: int
    img_x0: int
    img_y0: int
    layout: ComposeLayout


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SectionMetaDialog(tk.Toplevel):
    """Core name, section, and optional top depth (cm) on add / edit."""

    def __init__(
        self,
        master: tk.Tk,
        path: Path,
        core: str,
        section: str,
        top_depth_cm: float = 0.0,
        *,
        title: str = "Section labels",
        allow_empty_top: bool = True,
    ) -> None:
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.result: tuple[str, str, float, bool] | None = None
        self.transient(master)
        self.grab_set()

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Folder:", font=("", 9, "bold")).grid(row=0, column=0, sticky="nw")
        ttk.Label(frm, text=str(path), wraplength=440).grid(
            row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 8)
        )

        ttk.Label(frm, text="Core name").grid(row=1, column=0, sticky="w", pady=4)
        self.core_var = tk.StringVar(value=core)
        core_e = ttk.Entry(frm, textvariable=self.core_var, width=44)
        core_e.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)

        ttk.Label(frm, text="Section").grid(row=2, column=0, sticky="w", pady=4)
        self.sec_var = tk.StringVar(value=section)
        ttk.Entry(frm, textvariable=self.sec_var, width=44).grid(
            row=2, column=1, sticky="ew", padx=(8, 0), pady=4
        )

        ttk.Label(frm, text="Top depth (cm)").grid(row=3, column=0, sticky="w", pady=4)
        top_txt = "" if (allow_empty_top and top_depth_cm == 0.0) else f"{top_depth_cm:g}"
        self.top_var = tk.StringVar(value=top_txt)
        ttk.Entry(frm, textvariable=self.top_var, width=16).grid(
            row=3, column=1, sticky="w", padx=(8, 0), pady=4
        )
        ttk.Label(
            frm,
            text="Leave blank (or 0) to start at 0 cm. Enter a value to pin a manual top depth.",
            wraplength=440,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 4))

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text="OK", command=self._ok).pack(side=tk.RIGHT)

        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self._cancel())
        core_e.focus_set()
        core_e.selection_range(0, tk.END)
        self.update_idletasks()
        self.geometry(f"+{master.winfo_rootx() + 80}+{master.winfo_rooty() + 80}")
        self.wait_window(self)

    def _ok(self) -> None:
        raw = self.top_var.get().strip()
        if raw == "":
            top = 0.0
            manual = False
        else:
            try:
                top = float(raw)
            except ValueError:
                messagebox.showerror("Top depth", "Enter a number in cm, or leave blank.", parent=self)
                return
            manual = True
        self.result = (
            self.core_var.get().strip(),
            self.sec_var.get().strip(),
            top,
            manual,
        )
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class MultiViewGui(tk.Tk):
    def __init__(self, initial_paths: list[Path] | None = None) -> None:
        super().__init__()
        self.title("dicom-multi-view — core sections")
        self.geometry("1280x860")
        self.minsize(960, 600)

        self.sections: list[LoadedSection] = []
        self._photo: ImageTk.PhotoImage | None = None
        self._detail_photos: list[ImageTk.PhotoImage] = []
        self._base_sheet: Image.Image | None = None
        self._sheet_layouts: list[SectionSheetLayout] = []
        self._overview_mm_per_px: float = 1.0
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 8.0
        self._rebuild_job: str | None = None
        self._plane_job: str | None = None
        self._win_job: str | None = None
        self._marker_job: str | None = None
        self._detail_job: str | None = None
        self._ignore_plane = False
        self._ignore_top = False
        self._ignore_insets = False
        self._ignore_window = False
        self._hist_range = (0.0, 1.0)
        self._hist_counts: np.ndarray | None = None
        self._hist_drag_which: str | None = None
        self._marker_drag: tuple[int, str] | None = None  # (section_index, "top"|"bottom")

        self.status = tk.StringVar(
            value="Add folder… · drag cyan/amber core guides · shared zoom/pan · scale in cm"
        )
        # Opt-in: native ROI for the selected section only (never auto on zoom).
        self.high_quality = tk.BooleanVar(value=False)
        self.layout_mode = tk.StringVar(value="side")  # side | stack
        # continuous: tops from sum of core lengths (manual tops respected)
        # per: each section uses its own top (default 0)
        self.scale_mode = tk.StringVar(value="continuous")
        self.plane_label = tk.StringVar(value="Cut / frame: —")
        self.plane_help = tk.StringVar(
            value="Coronal MPR: parallel stack frames · Axial: coronal cut through diameter"
        )
        self.zoom_label = tk.StringVar(value="Zoom: 100% (shared)")
        self.top_depth_var = tk.StringVar(value="0")
        self.top_manual_var = tk.BooleanVar(value=False)
        self.core_top_inset_var = tk.StringVar(value="0")
        self.core_bottom_inset_var = tk.StringVar(value="0")
        self.core_length_label = tk.StringVar(value="Core length: —")
        # Greyscale window: "section" = selected only; "all" = shared HU window
        self.window_scope = tk.StringVar(value="section")
        self.win_lo = tk.DoubleVar(value=0.0)
        self.win_hi = tk.DoubleVar(value=1.0)
        self.win_label = tk.StringVar(value="Window: —")

        self._build()
        if initial_paths:
            self.after(150, lambda: self._load_paths(initial_paths, prompt=True))

    def _build(self) -> None:
        # Horizontal splitter: left controls | image view
        self.paned = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=6,
            bd=0,
            opaqueresize=True,
        )
        self.paned.pack(fill=tk.BOTH, expand=True)

        left_outer = ttk.Frame(self.paned)
        self.paned.add(left_outer, minsize=320, width=420, stretch="never")

        # Scrollable left control column (vertical scrollbar when content is tall)
        self._left_canvas = tk.Canvas(
            left_outer, highlightthickness=0, borderwidth=0, width=420
        )
        left_vsb = ttk.Scrollbar(
            left_outer, orient=tk.VERTICAL, command=self._left_canvas.yview
        )
        self._left_canvas.configure(yscrollcommand=left_vsb.set)
        left_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left = ttk.Frame(self._left_canvas, padding=(8, 8, 8, 8))
        self._left_window = self._left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.columnconfigure(0, weight=1)
        self._wrap_labels: list[ttk.Label] = []

        def _sync_left_scrollregion(_event: tk.Event | None = None) -> None:
            self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all"))

        def _sync_left_width(event: tk.Event) -> None:
            self._left_canvas.itemconfigure(self._left_window, width=event.width)
            wrap = max(200, event.width - 28)
            for lbl in self._wrap_labels:
                lbl.configure(wraplength=wrap)

        left.bind("<Configure>", _sync_left_scrollregion)
        self._left_canvas.bind("<Configure>", _sync_left_width)

        def _left_wheel_enter(_event: tk.Event) -> None:
            self.bind_all("<MouseWheel>", self._on_left_mousewheel)
            self.bind_all("<Button-4>", self._on_left_button4)
            self.bind_all("<Button-5>", self._on_left_button5)

        def _left_wheel_leave(_event: tk.Event) -> None:
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")

        left_outer.bind("<Enter>", _left_wheel_enter)
        left_outer.bind("<Leave>", _left_wheel_leave)

        ttk.Label(left, text="dicom-multi-view", font=("", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        subtitle = ttk.Label(
            left,
            text="Sediment core sections — view only (no stitch / export)",
            wraplength=380,
        )
        subtitle.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._wrap_labels.append(subtitle)

        actions = ttk.Frame(left)
        actions.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(actions, text="Add folder…", command=self.add_folder).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(actions, text="Edit…", command=self.edit_selected).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(actions, text="Remove", command=self.remove_selected).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(actions, text="↑", width=3, command=lambda: self.move_selected(-1)).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        ttk.Button(actions, text="↓", width=3, command=lambda: self.move_selected(1)).pack(
            side=tk.LEFT
        )

        list_fr = ttk.LabelFrame(left, text="Sections (order)", padding=4)
        list_fr.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        list_fr.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(list_fr, exportselection=False, height=8)
        self.listbox.grid(row=0, column=0, sticky="ew")
        sb = ttk.Scrollbar(list_fr, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.listbox.bind("<Double-Button-1>", lambda _e: self.edit_selected())

        depth_fr = ttk.LabelFrame(left, text="Depth scale (cm)", padding=6)
        depth_fr.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        ttk.Radiobutton(
            depth_fr,
            text="Continuous — tops = sum of core lengths in list order",
            variable=self.scale_mode,
            value="continuous",
            command=self._on_scale_mode,
        ).pack(anchor="w")
        ttk.Radiobutton(
            depth_fr,
            text="Per-section — each uses its own top (often 0)",
            variable=self.scale_mode,
            value="per",
            command=self._on_scale_mode,
        ).pack(anchor="w")
        ttk.Button(
            depth_fr, text="Recalc continuous tops", command=self._recalc_continuous
        ).pack(anchor="w", pady=(4, 2))

        top_row = ttk.Frame(depth_fr)
        top_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(top_row, text="Selected top (cm)").pack(side=tk.LEFT)
        top_e = ttk.Entry(top_row, textvariable=self.top_depth_var, width=8)
        top_e.pack(side=tk.LEFT, padx=6)
        top_e.bind("<Return>", lambda _e: self._apply_top_from_entry())
        top_e.bind("<FocusOut>", lambda _e: self._apply_top_from_entry())
        ttk.Checkbutton(
            depth_fr,
            text="Pin top (manual; skip auto continuous)",
            variable=self.top_manual_var,
            command=self._apply_top_manual_flag,
        ).pack(anchor="w")

        ttk.Separator(depth_fr, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        core_help = ttk.Label(
            depth_fr,
            text=(
                "Core bounds — cyan (top) / amber (bottom) guides. "
                "Insets from image ends; scale ticks & continuous chaining use the span between them."
            ),
            wraplength=380,
        )
        core_help.pack(anchor="w")
        self._wrap_labels.append(core_help)

        inset_top_row = ttk.Frame(depth_fr)
        inset_top_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(inset_top_row, text="Top inset (cm)").pack(side=tk.LEFT)
        top_in_e = ttk.Entry(inset_top_row, textvariable=self.core_top_inset_var, width=8)
        top_in_e.pack(side=tk.LEFT, padx=6)
        top_in_e.bind("<Return>", lambda _e: self._apply_insets_from_entry())
        top_in_e.bind("<FocusOut>", lambda _e: self._apply_insets_from_entry())

        inset_bot_row = ttk.Frame(depth_fr)
        inset_bot_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(inset_bot_row, text="Bottom inset (cm)").pack(side=tk.LEFT)
        bot_in_e = ttk.Entry(inset_bot_row, textvariable=self.core_bottom_inset_var, width=8)
        bot_in_e.pack(side=tk.LEFT, padx=6)
        bot_in_e.bind("<Return>", lambda _e: self._apply_insets_from_entry())
        bot_in_e.bind("<FocusOut>", lambda _e: self._apply_insets_from_entry())

        ttk.Label(depth_fr, textvariable=self.core_length_label).pack(anchor="w", pady=(4, 0))
        ttk.Button(
            depth_fr, text="Reset insets to image ends", command=self._reset_core_insets
        ).pack(anchor="w", pady=(2, 0))

        layout_fr = ttk.LabelFrame(left, text="Layout", padding=6)
        layout_fr.grid(row=5, column=0, sticky="ew", pady=(0, 6))
        ttk.Radiobutton(
            layout_fr, text="Side by side", variable=self.layout_mode, value="side", command=self.rebuild
        ).pack(anchor="w")
        ttk.Radiobutton(
            layout_fr,
            text="Stacked vertically",
            variable=self.layout_mode,
            value="stack",
            command=self.rebuild,
        ).pack(anchor="w")

        plane_fr = ttk.LabelFrame(left, text="Cut / frame (selected)", padding=6)
        plane_fr.grid(row=6, column=0, sticky="ew", pady=(0, 6))
        plane_lbl = ttk.Label(plane_fr, textvariable=self.plane_label, wraplength=380)
        plane_lbl.pack(anchor="w")
        self._wrap_labels.append(plane_lbl)
        self.plane_scale = ttk.Scale(
            plane_fr, from_=0, to=1, orient=tk.HORIZONTAL, command=self._on_plane_slide
        )
        self.plane_scale.pack(fill=tk.X, pady=4)
        ttk.Button(plane_fr, text="Mid cut", command=self._plane_mid).pack(anchor="w")
        plane_help_lbl = ttk.Label(
            plane_fr, textvariable=self.plane_help, wraplength=380
        )
        plane_help_lbl.pack(anchor="w", pady=(4, 0))
        self._wrap_labels.append(plane_help_lbl)

        win_fr = ttk.LabelFrame(left, text="Greyscale window (HU)", padding=6)
        win_fr.grid(row=7, column=0, sticky="ew", pady=(0, 6))
        scope = ttk.Frame(win_fr)
        scope.pack(fill=tk.X)
        ttk.Radiobutton(
            scope,
            text="This section",
            variable=self.window_scope,
            value="section",
            command=self._on_window_scope,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            scope,
            text="All sections",
            variable=self.window_scope,
            value="all",
            command=self._on_window_scope,
        ).pack(side=tk.LEFT, padx=(8, 0))
        self.hist_canvas = tk.Canvas(
            win_fr, height=72, background="#1a1a1a", highlightthickness=0
        )
        self.hist_canvas.pack(fill=tk.X, pady=(4, 0))
        self.hist_canvas.bind("<Configure>", lambda _e: self._draw_histogram())
        self.hist_canvas.bind("<Button-1>", self._hist_click)
        self.hist_canvas.bind("<B1-Motion>", self._hist_drag)
        self.hist_canvas.bind(
            "<ButtonRelease-1>", lambda _e: setattr(self, "_hist_drag_which", None)
        )
        ttk.Label(win_fr, text="Black").pack(anchor="w", pady=(4, 0))
        self.win_lo_scale = ttk.Scale(
            win_fr, from_=0, to=1, variable=self.win_lo, command=self._on_window_slide
        )
        self.win_lo_scale.pack(fill=tk.X)
        ttk.Label(win_fr, text="White").pack(anchor="w", pady=(2, 0))
        self.win_hi_scale = ttk.Scale(
            win_fr, from_=0, to=1, variable=self.win_hi, command=self._on_window_slide
        )
        self.win_hi_scale.pack(fill=tk.X)
        wb = ttk.Frame(win_fr)
        wb.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(wb, text="Auto 1–99%", command=self._auto_window).pack(side=tk.LEFT)
        win_lbl = ttk.Label(win_fr, textvariable=self.win_label, wraplength=380)
        win_lbl.pack(anchor="w")
        self._wrap_labels.append(win_lbl)

        view_fr = ttk.LabelFrame(left, text="View (shared)", padding=6)
        view_fr.grid(row=8, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(view_fr, textvariable=self.zoom_label).pack(anchor="w")
        vr = ttk.Frame(view_fr)
        vr.pack(fill=tk.X, pady=4)
        ttk.Button(vr, text="Fit width", command=self.fit_width).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(vr, text="Reset", command=self.reset_view).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(vr, text="Zoom +", command=lambda: self._zoom_by(1.25)).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        ttk.Button(vr, text="Zoom −", command=lambda: self._zoom_by(1 / 1.25)).pack(side=tk.LEFT)
        ttk.Checkbutton(
            view_fr,
            text="High quality (selected section, visible region)",
            variable=self.high_quality,
            command=self._on_high_quality_toggle,
        ).pack(anchor="w", pady=(6, 0))
        view_help = ttk.Label(
            view_fr,
            text=(
                "Wheel zooms toward cursor · drag pans · shared physical cm scale + zoom. "
                "Default is lightweight overview; enable High quality for a sharper "
                "native sample of the selected section’s visible area (debounced)."
            ),
            wraplength=380,
        )
        view_help.pack(anchor="w", pady=(4, 0))
        self._wrap_labels.append(view_help)

        status_lbl = ttk.Label(left, textvariable=self.status, wraplength=380)
        status_lbl.grid(row=9, column=0, sticky="ew")
        self._wrap_labels.append(status_lbl)

        right = ttk.Frame(self.paned, padding=(4, 8, 8, 4))
        self.paned.add(right, minsize=400, stretch="always")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, background="#1a1a1e", highlightthickness=0)
        self.hbar = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.15))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.15))
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-2>", lambda _e: self._schedule_detail_refresh())
        # Scrollbars: update view; refresh High-quality ROI only when that mode is on.
        self.hbar.configure(command=self._on_xview)
        self.vbar.configure(command=self._on_yview)
        self.canvas.configure(
            xscrollcommand=self._on_xscroll_set,
            yscrollcommand=self._on_yscroll_set,
        )

        # Place sash after geometry is realized (~420 px left panel)
        self.after_idle(lambda: self._set_left_sash(420))

    def _set_left_sash(self, width: int) -> None:
        try:
            self.paned.sash_place(0, width, 1)
        except tk.TclError:
            pass

    def _on_left_mousewheel(self, event: tk.Event) -> str | None:
        """Scroll the left control pane while the pointer is over it."""
        delta = getattr(event, "delta", 0)
        if delta:
            self._left_canvas.yview_scroll(int(-delta / 120), "units")
        return "break"

    def _on_left_button4(self, _event: tk.Event) -> str | None:
        self._left_canvas.yview_scroll(-1, "units")
        return "break"

    def _on_left_button5(self, _event: tk.Event) -> str | None:
        self._left_canvas.yview_scroll(1, "units")
        return "break"

    # ----- load / labels -----

    def add_folder(self) -> None:
        path = filedialog.askdirectory(title="Select DICOM series folder")
        if path:
            self._load_paths([Path(path)], prompt=True)

    def edit_selected(self) -> None:
        i = self._selected_index()
        if i is None:
            messagebox.showinfo("Edit", "Select a section first.")
            return
        sec = self.sections[i]
        dlg = SectionMetaDialog(
            self,
            sec.path,
            sec.core_name,
            sec.section_name,
            sec.top_depth_cm,
            title="Edit section",
            allow_empty_top=False,
        )
        if dlg.result is None:
            return
        core, section, top, manual = dlg.result
        sec.core_name = core
        sec.section_name = section
        sec.label = sec.display_label
        sec.top_depth_cm = top
        sec.manual_top = manual or top != 0.0
        if self.scale_mode.get() == "continuous":
            sync_continuous_tops(self.sections)
        self._rebuild_list()
        self.listbox.selection_set(i)
        self._on_select()
        self.rebuild()

    def _load_paths(self, paths: list[Path], *, prompt: bool) -> None:
        for path in paths:
            path = Path(path)
            if not path.is_dir():
                messagebox.showerror("Not a folder", str(path))
                continue
            core: str
            section: str
            top = 0.0
            manual = False
            if prompt:
                sug_c, sug_s = suggest_labels(path)
                dlg = SectionMetaDialog(self, path, sug_c, sug_s, 0.0)
                if dlg.result is None:
                    self.status.set(f"Skipped {path.name}")
                    continue
                core, section, top, manual = dlg.result
            else:
                core, section = suggest_labels(path)

            self.status.set(f"Loading {section or path.name}… (compressed may be slow)")
            self.update_idletasks()
            try:
                sec = load_section(
                    path,
                    core_name=core,
                    section_name=section,
                    top_depth_cm=top,
                    manual_top=manual,
                )
            except Exception as exc:
                messagebox.showerror("Load failed", f"{path}\n\n{exc}")
                self.status.set(f"Failed: {path.name}")
                continue
            self.sections.append(sec)
            if self.scale_mode.get() == "continuous":
                sync_continuous_tops(self.sections)
            note = "; ".join(sec.notes) if sec.notes else "ok"
            self.status.set(
                f"Loaded {sec.display_label}: {sec.badge_geometry}, "
                f"{sec.badge_compression} — {note}"
            )
        self._rebuild_list()
        if self.sections and not self.listbox.curselection():
            self.listbox.selection_set(0)
            self._on_select()
        self.rebuild(fit_after=True)

    def _rebuild_list(self) -> None:
        self.listbox.delete(0, tk.END)
        for sec in self.sections:
            pin = "*" if sec.manual_top else ""
            self.listbox.insert(
                tk.END,
                f"{sec.display_label}  @{sec.top_depth_cm:g}cm{pin}  "
                f"[{sec.badge_compression[:4]}] [{sec.badge_geometry}]",
            )

    def _selected_index(self) -> int | None:
        sel = self.listbox.curselection()
        return int(sel[0]) if sel else None

    def remove_selected(self) -> None:
        i = self._selected_index()
        if i is None:
            return
        self.sections.pop(i)
        if self.scale_mode.get() == "continuous":
            sync_continuous_tops(self.sections)
        self._rebuild_list()
        self.rebuild()
        self.status.set("Removed section.")

    def move_selected(self, delta: int) -> None:
        i = self._selected_index()
        if i is None:
            return
        j = i + delta
        if j < 0 or j >= len(self.sections):
            return
        self.sections[i], self.sections[j] = self.sections[j], self.sections[i]
        if self.scale_mode.get() == "continuous":
            sync_continuous_tops(self.sections)
        self._rebuild_list()
        self.listbox.selection_set(j)
        self._on_select()
        self.rebuild()

    def _on_scale_mode(self) -> None:
        if self.scale_mode.get() == "continuous":
            sync_continuous_tops(self.sections)
            self._rebuild_list()
        self.rebuild()

    def _recalc_continuous(self) -> None:
        for sec in self.sections:
            sec.manual_top = False
        sync_continuous_tops(self.sections)
        self.scale_mode.set("continuous")
        self._rebuild_list()
        self._on_select()
        self.rebuild()
        self.status.set("Continuous tops recalculated from core lengths (cm).")

    def _sync_inset_fields(self, sec: LoadedSection) -> None:
        self._ignore_insets = True
        self.core_top_inset_var.set(f"{sec.core_top_inset_cm:g}")
        self.core_bottom_inset_var.set(f"{sec.core_bottom_inset_cm:g}")
        self.core_length_label.set(
            f"Core length: {sec.core_length_cm:.2f} cm  "
            f"(image {sec.depth_extent_cm:.2f} cm)"
        )
        self._ignore_insets = False

    def _apply_insets_from_entry(self) -> None:
        if self._ignore_insets:
            return
        i = self._selected_index()
        if i is None:
            return
        try:
            top_in = float(self.core_top_inset_var.get().strip() or "0")
            bot_in = float(self.core_bottom_inset_var.get().strip() or "0")
        except ValueError:
            return
        sec = self.sections[i]
        sec.core_top_inset_cm = max(0.0, top_in)
        sec.core_bottom_inset_cm = max(0.0, bot_in)
        sec.clamp_core_insets()
        self._sync_inset_fields(sec)
        if self.scale_mode.get() == "continuous":
            sync_continuous_tops(self.sections)
            self._rebuild_list()
            self.listbox.selection_set(i)
            self._ignore_top = True
            self.top_depth_var.set(f"{sec.top_depth_cm:g}")
            self._ignore_top = False
        self.rebuild()

    def _reset_core_insets(self) -> None:
        i = self._selected_index()
        if i is None:
            return
        sec = self.sections[i]
        sec.core_top_inset_cm = 0.0
        sec.core_bottom_inset_cm = 0.0
        self._sync_inset_fields(sec)
        if self.scale_mode.get() == "continuous":
            sync_continuous_tops(self.sections)
            self._rebuild_list()
            self.listbox.selection_set(i)
        self.rebuild()
        self.status.set("Core insets reset to image ends (0 cm).")

    def _apply_top_from_entry(self) -> None:
        if self._ignore_top:
            return
        i = self._selected_index()
        if i is None:
            return
        raw = self.top_depth_var.get().strip()
        try:
            top = float(raw) if raw else 0.0
        except ValueError:
            return
        sec = self.sections[i]
        sec.top_depth_cm = top
        sec.manual_top = True
        self.top_manual_var.set(True)
        if self.scale_mode.get() == "continuous":
            sync_continuous_tops(self.sections)
        self._rebuild_list()
        self.listbox.selection_set(i)
        self.rebuild()

    def _apply_top_manual_flag(self) -> None:
        i = self._selected_index()
        if i is None:
            return
        self.sections[i].manual_top = bool(self.top_manual_var.get())
        if self.scale_mode.get() == "continuous":
            sync_continuous_tops(self.sections)
        self._rebuild_list()
        self.listbox.selection_set(i)
        self._on_select()
        self.rebuild()

    def _on_select(self, _event=None) -> None:
        i = self._selected_index()
        if i is None or not (0 <= i < len(self.sections)):
            self.plane_label.set("Cut / frame: —")
            self.win_label.set("Window: —")
            self.core_length_label.set("Core length: —")
            return
        sec = self.sections[i]
        self._ignore_top = True
        self.top_depth_var.set(f"{sec.top_depth_cm:g}")
        self.top_manual_var.set(sec.manual_top)
        self._ignore_top = False
        self._sync_inset_fields(sec)

        n = max(sec.plane_count - 1, 0)
        self._ignore_plane = True
        self.plane_scale.configure(from_=0, to=max(n, 1))
        self.plane_scale.set(sec.plane_index)
        self._ignore_plane = False
        if sec.view_mode == "axial_reformat":
            kind = "coronal cut Y"
        elif sec.view_mode == "coronal_mpr":
            kind = "MPR frame"
        else:
            kind = "frame"
        self.plane_label.set(
            f"{kind} {sec.plane_index} / {max(sec.plane_count - 1, 0)}  "
            f"({sec.plane_count} planes)"
        )
        self.plane_help.set(sec.slider_help)
        self._sync_window_controls()
        # Selected section drives High-quality ROI.
        if self.high_quality.get():
            self._schedule_detail_refresh(immediate=True)

    def _on_plane_slide(self, _val=None) -> None:
        if self._ignore_plane:
            return
        if self._plane_job is not None:
            self.after_cancel(self._plane_job)
        self._plane_job = self.after(80, self._apply_plane)

    def _apply_plane(self) -> None:
        self._plane_job = None
        i = self._selected_index()
        if i is None:
            return
        idx = int(round(float(self.plane_scale.get())))
        sec = self.sections[i]
        if idx == sec.plane_index:
            return
        self.sections[i] = set_plane(sec, idx)
        self._on_select()
        self._rebuild_list()
        self.listbox.selection_set(i)
        self.rebuild()

    def _plane_mid(self) -> None:
        i = self._selected_index()
        if i is None:
            return
        mid = self.sections[i].plane_count // 2
        self._ignore_plane = True
        self.plane_scale.set(mid)
        self._ignore_plane = False
        self._apply_plane()

    # ----- greyscale window / histogram -----

    def _hu_values_for_hist(self) -> np.ndarray:
        if self.window_scope.get() == "all" and self.sections:
            chunks = [s.image_hu.ravel() for s in self.sections]
            return np.concatenate(chunks)
        i = self._selected_index()
        if i is None or not self.sections:
            return np.asarray([], dtype=np.float32)
        return self.sections[i].image_hu.ravel()

    def _current_window_targets(self) -> list[LoadedSection]:
        if self.window_scope.get() == "all":
            return list(self.sections)
        i = self._selected_index()
        if i is None:
            return []
        return [self.sections[i]]

    def _sync_window_controls(self) -> None:
        targets = self._current_window_targets()
        vals = self._hu_values_for_hist()
        if not targets:
            self.win_label.set("Window: —")
            self._hist_counts = None
            self._draw_histogram()
            return
        if self.window_scope.get() == "all":
            los = {round(float(s.win_lo), 3) for s in targets if s.win_lo is not None}
            his = {round(float(s.win_hi), 3) for s in targets if s.win_hi is not None}
            if len(los) == 1 and len(his) == 1:
                lo, hi = float(targets[0].win_lo), float(targets[0].win_hi)
            else:
                lo, hi = auto_window_hu(vals) if vals.size else (0.0, 1.0)
                for s in targets:
                    s.win_lo, s.win_hi = lo, hi
        else:
            sec = targets[0]
            if sec.win_lo is None or sec.win_hi is None:
                sec.win_lo, sec.win_hi = auto_window_hu(sec.image_hu)
            lo, hi = float(sec.win_lo), float(sec.win_hi)

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
        self._ignore_window = True
        self.win_lo_scale.configure(from_=vmin, to=vmax)
        self.win_hi_scale.configure(from_=vmin, to=vmax)
        self.win_lo.set(lo)
        self.win_hi.set(hi)
        self._ignore_window = False
        self.win_label.set(f"Window: [{lo:.1f}, {hi:.1f}] HU")
        if vals.size:
            counts, _edges = np.histogram(vals, bins=128, range=(vmin, vmax))
            self._hist_counts = counts.astype(np.float64)
        else:
            self._hist_counts = None
        self._draw_histogram()

    def _on_window_scope(self) -> None:
        if self.window_scope.get() == "all" and self.sections:
            # Push selected window to all when switching into shared mode.
            i = self._selected_index()
            if i is not None:
                lo = self.sections[i].win_lo
                hi = self.sections[i].win_hi
                if lo is not None and hi is not None:
                    for s in self.sections:
                        s.win_lo, s.win_hi = float(lo), float(hi)
        self._sync_window_controls()
        self.rebuild()

    def _auto_window(self) -> None:
        targets = self._current_window_targets()
        if not targets:
            return
        if self.window_scope.get() == "all":
            vals = self._hu_values_for_hist()
            lo, hi = auto_window_hu(vals) if vals.size else (0.0, 1.0)
            for s in targets:
                s.win_lo, s.win_hi = lo, hi
        else:
            sec = targets[0]
            sec.win_lo, sec.win_hi = auto_window_hu(sec.image_hu)
        self._sync_window_controls()
        self.rebuild()

    def _on_window_slide(self, _value=None) -> None:
        if self._ignore_window:
            return
        if self._win_job is not None:
            self.after_cancel(self._win_job)
        self._win_job = self.after(40, self._apply_window)

    def _apply_window(self) -> None:
        self._win_job = None
        targets = self._current_window_targets()
        if not targets:
            return
        lo = float(self.win_lo.get())
        hi = float(self.win_hi.get())
        if hi < lo:
            lo, hi = hi, lo
            self._ignore_window = True
            self.win_lo.set(lo)
            self.win_hi.set(hi)
            self._ignore_window = False
        for s in targets:
            s.win_lo, s.win_hi = lo, hi
        self.win_label.set(f"Window: [{lo:.1f}, {hi:.1f}] HU")
        self._draw_histogram()
        self.rebuild()

    def _hu_to_x(self, hu: float, width: int) -> float:
        vmin, vmax = self._hist_range
        if vmax <= vmin:
            return 0.0
        return (hu - vmin) / (vmax - vmin) * max(width - 1, 1)

    def _x_to_hu(self, x: float, width: int) -> float:
        vmin, vmax = self._hist_range
        t = float(np.clip(x / max(width - 1, 1), 0.0, 1.0))
        return float(vmin + t * (vmax - vmin))

    def _draw_histogram(self) -> None:
        c = self.hist_canvas
        c.delete("all")
        w = max(c.winfo_width(), 2)
        h = max(c.winfo_height(), 2)
        if self._hist_counts is None or self._hist_counts.size == 0:
            c.create_text(w // 2, h // 2, text="Load a section", fill="#888")
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

    def _hist_click(self, event) -> None:
        if not self.sections:
            return
        w = max(self.hist_canvas.winfo_width(), 2)
        x_lo = self._hu_to_x(float(self.win_lo.get()), w)
        x_hi = self._hu_to_x(float(self.win_hi.get()), w)
        if abs(event.x - x_lo) <= abs(event.x - x_hi):
            self._hist_drag_which = "lo"
        else:
            self._hist_drag_which = "hi"
        self._hist_drag(event)

    def _hist_drag(self, event) -> None:
        if self._hist_drag_which is None or not self.sections:
            return
        w = max(self.hist_canvas.winfo_width(), 2)
        hu = self._x_to_hu(float(event.x), w)
        self._ignore_window = True
        if self._hist_drag_which == "lo":
            self.win_lo.set(min(hu, float(self.win_hi.get()) - 1.0))
        else:
            self.win_hi.set(max(hu, float(self.win_lo.get()) + 1.0))
        self._ignore_window = False
        self._apply_window()

    # ----- shared zoom / pan / core guides -----

    def _on_xview(self, *args) -> None:
        self.canvas.xview(*args)
        self._schedule_detail_refresh()

    def _on_yview(self, *args) -> None:
        self.canvas.yview(*args)
        self._schedule_detail_refresh()

    def _on_xscroll_set(self, first: str, last: str) -> None:
        self.hbar.set(first, last)

    def _on_yscroll_set(self, first: str, last: str) -> None:
        self.vbar.set(first, last)

    def _on_high_quality_toggle(self) -> None:
        if self.high_quality.get():
            self._schedule_detail_refresh(immediate=True)
            self.status.set("High quality on — selected section visible region (debounced).")
        else:
            self._clear_detail_overlays()
            self.zoom_label.set(f"Zoom: {int(round(self._zoom * 100))}% (shared)")
            self.status.set("Overview quality — fast pan/zoom.")

    def _pan_start(self, event) -> None:
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.focus_set()
        if self.high_quality.get():
            # Drop stale high-res tile while dragging; overview stays snappy.
            self.canvas.delete("detail")
            self._detail_photos.clear()
            if self._detail_job is not None:
                self.after_cancel(self._detail_job)
                self._detail_job = None

    def _pan_move(self, event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self._schedule_detail_refresh()

    def _marker_y_base(self, lay: SectionSheetLayout, which: str) -> float:
        sec = self.sections[lay.index]
        h = max(lay.layout.img_h, 1)
        extent = max(sec.depth_extent_cm, 1e-12)
        if which == "top":
            return lay.img_y0 + (sec.core_top_inset_cm / extent) * h
        return lay.img_y0 + h - 1 - (sec.core_bottom_inset_cm / extent) * h

    def _hit_core_marker(self, event) -> tuple[int, str] | None:
        """Return (section_index, 'top'|'bottom') if pointer is near a guide."""
        if self._base_sheet is None or not self._sheet_layouts:
            return None
        z = max(self._zoom, 1e-9)
        bx = self.canvas.canvasx(event.x) / z
        by = self.canvas.canvasy(event.y) / z
        thresh = max(5.0 / z, 0.5)
        best: tuple[float, int, str] | None = None
        for lay in self._sheet_layouts:
            if not (lay.img_x0 <= bx <= lay.img_x0 + lay.layout.img_w):
                continue
            for which in ("top", "bottom"):
                dy = abs(by - self._marker_y_base(lay, which))
                if dy <= thresh and (best is None or dy < best[0]):
                    best = (dy, lay.index, which)
        if best is None:
            return None
        return best[1], best[2]

    def _canvas_press(self, event) -> None:
        hit = self._hit_core_marker(event)
        if hit is not None:
            self._marker_drag = hit
            idx, which = hit
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self._on_select()
            self.canvas.focus_set()
            if self.high_quality.get():
                self.canvas.delete("detail")
                self._detail_photos.clear()
            self.status.set(
                f"Dragging core {'top' if which == 'top' else 'bottom'} guide…"
            )
            return
        self._marker_drag = None
        self._pan_start(event)

    def _canvas_drag(self, event) -> None:
        if self._marker_drag is None:
            self._pan_move(event)
            return
        idx, which = self._marker_drag
        if not (0 <= idx < len(self.sections)) or not self._sheet_layouts:
            return
        lay = next((L for L in self._sheet_layouts if L.index == idx), None)
        if lay is None:
            return
        z = max(self._zoom, 1e-9)
        by = self.canvas.canvasy(event.y) / z
        h = max(lay.layout.img_h, 1)
        extent = max(self.sections[idx].depth_extent_cm, 1e-12)
        frac = (by - lay.img_y0) / h
        frac = float(np.clip(frac, 0.0, 1.0))
        sec = self.sections[idx]
        if which == "top":
            # Leave at least a tiny core; don't cross bottom marker.
            max_top = max(0.0, extent - sec.core_bottom_inset_cm - 0.01)
            sec.core_top_inset_cm = float(np.clip(frac * extent, 0.0, max_top))
        else:
            from_bottom = (1.0 - frac) * extent
            max_bot = max(0.0, extent - sec.core_top_inset_cm - 0.01)
            sec.core_bottom_inset_cm = float(np.clip(from_bottom, 0.0, max_bot))
        sec.clamp_core_insets()
        self._sync_inset_fields(sec)
        if self._marker_job is not None:
            self.after_cancel(self._marker_job)
        self._marker_job = self.after(40, self._commit_marker_drag)

    def _commit_marker_drag(self) -> None:
        self._marker_job = None
        if self.scale_mode.get() == "continuous":
            sync_continuous_tops(self.sections)
            i = self._selected_index()
            self._rebuild_list()
            if i is not None:
                self.listbox.selection_set(i)
                self._ignore_top = True
                self.top_depth_var.set(f"{self.sections[i].top_depth_cm:g}")
                self._ignore_top = False
        self.rebuild()

    def _canvas_release(self, _event) -> None:
        if self._marker_drag is not None:
            self._marker_drag = None
            if self._marker_job is not None:
                self.after_cancel(self._marker_job)
                self._marker_job = None
            self._commit_marker_drag()
            self.status.set("Core bounds updated.")
            return
        # End of pan drag — refresh High-quality ROI if enabled.
        self._schedule_detail_refresh()

    def _on_mousewheel(self, event) -> None:
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self._zoom_at(event.x, event.y, factor)

    def _zoom_by(self, factor: float) -> None:
        self._zoom_at(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, factor)

    def _zoom_at(self, canvas_x: int, canvas_y: int, factor: float) -> None:
        if self._base_sheet is None:
            return
        old = self._zoom
        new = _clip(old * factor, self._min_zoom, self._max_zoom)
        if abs(new - old) < 1e-9:
            return
        ix = self.canvas.canvasx(canvas_x) / max(old, 1e-9)
        iy = self.canvas.canvasy(canvas_y) / max(old, 1e-9)
        self._zoom = new
        self._apply_zoom(anchor_ix=ix, anchor_iy=iy, canvas_x=canvas_x, canvas_y=canvas_y)

    def fit_width(self) -> None:
        if self._base_sheet is None:
            return
        cw = max(self.canvas.winfo_width() - 8, 1)
        self._zoom = _clip(cw / self._base_sheet.width, self._min_zoom, self._max_zoom)
        self._apply_zoom(reset_scroll=True)

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._apply_zoom(reset_scroll=True)

    def _apply_zoom(
        self,
        *,
        anchor_ix: float | None = None,
        anchor_iy: float | None = None,
        canvas_x: int | None = None,
        canvas_y: int | None = None,
        reset_scroll: bool = False,
    ) -> None:
        if self._base_sheet is None:
            return
        z = self._zoom
        w = max(1, int(round(self._base_sheet.width * z)))
        h = max(1, int(round(self._base_sheet.height * z)))
        if w * h > DISPLAY_PIXEL_BUDGET:
            z = max(
                self._min_zoom,
                (DISPLAY_PIXEL_BUDGET / (self._base_sheet.width * self._base_sheet.height))
                ** 0.5,
            )
            self._zoom = z
            w = max(1, int(round(self._base_sheet.width * z)))
            h = max(1, int(round(self._base_sheet.height * z)))
            self.status.set("Zoom capped for display size — use Fit width.")

        self._clear_detail_overlays()
        disp = self._base_sheet.resize((w, h), Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags=("base",))
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self.zoom_label.set(f"Zoom: {int(round(z * 100))}% (shared)")

        if reset_scroll:
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
        elif anchor_ix is not None and canvas_x is not None:
            left = anchor_ix * z - canvas_x
            top = (anchor_iy or 0.0) * z - (canvas_y or 0)
            if w > 0:
                self.canvas.xview_moveto(max(0.0, left / w))
            if h > 0:
                self.canvas.yview_moveto(max(0.0, top / h))

        # High quality is opt-in only — never auto on zoom.
        self._schedule_detail_refresh()

    def _clear_detail_overlays(self) -> None:
        if self._detail_job is not None:
            self.after_cancel(self._detail_job)
            self._detail_job = None
        self.canvas.delete("detail")
        self._detail_photos.clear()

    def _schedule_detail_refresh(self, *, immediate: bool = False) -> None:
        """No-op unless High quality is on (keeps overview pan/zoom snappy)."""
        if not self.high_quality.get():
            return
        if self._detail_job is not None:
            self.after_cancel(self._detail_job)
            self._detail_job = None
        if immediate:
            self._refresh_detail_overlays()
            return
        self._detail_job = self.after(DETAIL_DEBOUNCE_MS, self._refresh_detail_overlays)

    def _visible_base_rect(self) -> tuple[float, float, float, float] | None:
        """Visible canvas area in pre-zoom base-sheet coordinates."""
        if self._base_sheet is None:
            return None
        z = max(self._zoom, 1e-9)
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        x0 = self.canvas.canvasx(0) / z
        y0 = self.canvas.canvasy(0) / z
        x1 = self.canvas.canvasx(cw) / z
        y1 = self.canvas.canvasy(ch) / z
        return (
            max(0.0, float(x0)),
            max(0.0, float(y0)),
            min(float(self._base_sheet.width), float(x1)),
            min(float(self._base_sheet.height), float(y1)),
        )

    def _refresh_detail_overlays(self) -> None:
        """Native-sample the selected section's visible ROI (High quality mode)."""
        self._detail_job = None
        self.canvas.delete("detail")
        self._detail_photos.clear()
        if not self.high_quality.get() or self._base_sheet is None:
            return
        idx = self._selected_index()
        if idx is None:
            return
        lay = next((L for L in self._sheet_layouts if L.index == idx), None)
        if lay is None or not (0 <= idx < len(self.sections)):
            return
        sec = self.sections[idx]
        z = max(self._zoom, 1e-9)
        ov = self._overview_mm_per_px
        native = native_isotropic_mm(sec)
        # Skip if overview is already as sharp as native at this zoom.
        if ov / z <= native * 1.02:
            self.zoom_label.set(f"Zoom: {int(round(self._zoom * 100))}% (shared)")
            return

        vis = self._visible_base_rect()
        if vis is None:
            return
        vx0, vy0, vx1, vy1 = vis
        ix0 = float(lay.img_x0)
        iy0 = float(lay.img_y0)
        ix1 = ix0 + float(lay.layout.img_w)
        iy1 = iy0 + float(lay.layout.img_h)
        rx0 = max(vx0, ix0)
        ry0 = max(vy0, iy0)
        rx1 = min(vx1, ix1)
        ry1 = min(vy1, iy1)
        if rx1 - rx0 < 2 or ry1 - ry0 < 2:
            return

        out_w = max(1, int(round((rx1 - rx0) * z)))
        out_h = max(1, int(round((ry1 - ry0) * z)))
        if out_w * out_h > DETAIL_MAX_PATCH_PX:
            scale = (DETAIL_MAX_PATCH_PX / (out_w * out_h)) ** 0.5
            out_w = max(1, int(out_w * scale))
            out_h = max(1, int(out_h * scale))

        try:
            patch = sample_section_image_roi(
                sec,
                x0=rx0 - ix0,
                y0=ry0 - iy0,
                x1=rx1 - ix0,
                y1=ry1 - iy0,
                overview_mm_per_px=ov,
                out_w=out_w,
                out_h=out_h,
                win_lo=sec.win_lo,
                win_hi=sec.win_hi,
                draw_guides=True,
            )
        except Exception:
            return

        screen_w = max(1, int(round((rx1 - rx0) * z)))
        screen_h = max(1, int(round((ry1 - ry0) * z)))
        if patch.size != (screen_w, screen_h):
            patch = patch.resize((screen_w, screen_h), Image.Resampling.BILINEAR)

        photo = ImageTk.PhotoImage(patch)
        self._detail_photos.append(photo)
        self.canvas.create_image(
            rx0 * z,
            ry0 * z,
            anchor="nw",
            image=photo,
            tags=("detail",),
        )
        self.zoom_label.set(
            f"Zoom: {int(round(self._zoom * 100))}% (shared) · high quality"
        )

    def rebuild(self, *, fit_after: bool = False) -> None:
        if self._rebuild_job is not None:
            self.after_cancel(self._rebuild_job)

        def _do() -> None:
            self._rebuild_job = None
            self._compose_base()
            if fit_after and self._base_sheet is not None:
                self.fit_width()
            else:
                self._apply_zoom()

        self._rebuild_job = self.after(30, _do)

    def _compose_base(self) -> None:
        self._clear_detail_overlays()
        self.canvas.delete("all")
        self._photo = None
        self._base_sheet = None
        self._sheet_layouts = []
        self._overview_mm_per_px = 1.0
        if not self.sections:
            self.canvas.create_text(
                24,
                24,
                anchor="nw",
                fill="#888",
                text="No sections loaded.\nClick Add folder… to browse for a DICOM series.",
            )
            return

        # Shared physical scale: same mm/px for every section (then shared zoom).
        mm_px = shared_mm_per_px(
            self.sections,
            max_height_px=BASE_MAX_HEIGHT_PX,
            max_width_px=900,
        )
        self._overview_mm_per_px = float(mm_px)
        tiles: list[Image.Image] = []
        body_layouts: list[ComposeLayout] = []
        title_hs: list[int] = []
        for sec in self.sections:
            body, body_lay = compose_section_rgb(
                sec,
                top_depth_cm=sec.top_depth_cm,
                mm_per_px=mm_px,
            )
            title = (
                f"{sec.display_label}  ·  core {sec.top_depth_cm:g}–"
                f"{sec.top_depth_cm + sec.core_length_cm:.1f} cm  ·  "
                f"{sec.badge_geometry}  ·  {sec.badge_compression}"
            )
            head = title_strip(title, body.width)
            stacked = Image.new("RGB", (body.width, head.height + body.height), (20, 20, 22))
            stacked.paste(head, (0, 0))
            stacked.paste(body, (0, head.height))
            tiles.append(stacked)
            body_layouts.append(body_lay)
            title_hs.append(head.height)

        gap = 12
        layout = self.layout_mode.get()
        sheet_layouts: list[SectionSheetLayout] = []
        if layout == "side":
            total_w = sum(t.width for t in tiles) + gap * max(len(tiles) - 1, 0)
            total_h = max(t.height for t in tiles)
            sheet = Image.new("RGB", (total_w, total_h), (26, 26, 30))
            x = 0
            for i, t in enumerate(tiles):
                sheet.paste(t, (x, 0))
                bl = body_layouts[i]
                sheet_layouts.append(
                    SectionSheetLayout(
                        index=i,
                        img_x0=x,
                        img_y0=title_hs[i],
                        layout=ComposeLayout(
                            img_w=bl.img_w,
                            img_h=bl.img_h,
                            bar_width=bl.bar_width,
                            title_h=title_hs[i],
                        ),
                    )
                )
                x += t.width + gap
        else:
            total_w = max(t.width for t in tiles)
            total_h = sum(t.height for t in tiles) + gap * max(len(tiles) - 1, 0)
            sheet = Image.new("RGB", (total_w, total_h), (26, 26, 30))
            y = 0
            for i, t in enumerate(tiles):
                sheet.paste(t, (0, y))
                bl = body_layouts[i]
                sheet_layouts.append(
                    SectionSheetLayout(
                        index=i,
                        img_x0=0,
                        img_y0=y + title_hs[i],
                        layout=ComposeLayout(
                            img_w=bl.img_w,
                            img_h=bl.img_h,
                            bar_width=bl.bar_width,
                            title_h=title_hs[i],
                        ),
                    )
                )
                y += t.height + gap
        self._base_sheet = sheet
        self._sheet_layouts = sheet_layouts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DICOM multi-view for sediment core sections")
    p.add_argument(
        "folders",
        nargs="*",
        type=Path,
        help="Optional DICOM series folders to open on launch",
    )
    args = p.parse_args(argv)
    app = MultiViewGui(initial_paths=list(args.folders) if args.folders else None)
    app.mainloop()


if __name__ == "__main__":
    main()
