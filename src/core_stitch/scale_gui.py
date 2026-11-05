"""Draggable core-end guides for a Tk canvas (cyan = top, amber = bottom).

The guides only decorate the preview; they are never burned into the exported images.
Their position is saved in the JSON sidecar as the ``core_scale`` block (see ``scale.py``).

For careful placement the owner window zooms the canvas; the guides also come with exact
entry boxes (inset in cm from each image end) and "go to" buttons that scroll to each end.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from core_stitch.scale import clamp_insets, length_check

TOP_COLOR = "#22d3ee"
BOTTOM_COLOR = "#f59e0b"
TAG = "coreguide"


class CoreGuides:
    def __init__(self, canvas: tk.Canvas | None, on_change: Callable[[], None] | None = None) -> None:
        self.canvas = canvas  # may be attached later (the control tab is built first)
        self.on_change = on_change
        self.enabled = True
        self.source = "auto"  # "auto" until the user drags or types, then "manual"
        self.top_cm = 0.0
        self.bottom_cm = 0.0
        self.expected_cm: float | None = None
        self.mm_per_px = 1.0
        self.rows = 0  # rows of the exported image
        self.text = tk.StringVar(value="Core length: -")
        self.top_var = tk.StringVar(value="0.00")
        self.bottom_var = tk.StringVar(value="0.00")
        self._row_origin = 0.0  # canvas y of export row 0
        self._x_range = (0.0, 0.0)  # canvas x extent of the image
        self._scale = 1.0
        self._drag: str | None = None

    # ----- model -----------------------------------------------------------

    @property
    def extent_cm(self) -> float:
        return self.rows * self.mm_per_px / 10.0

    @property
    def length_cm(self) -> float:
        return max(self.extent_cm - self.top_cm - self.bottom_cm, 0.0)

    def reset(self) -> None:
        """Back to the automatic estimate (recomputed by the next ``set_frame``)."""
        self.source = "auto"

    def bounds(self) -> tuple[float, float, str]:
        return self.top_cm, self.bottom_cm, self.source

    def set_frame(
        self, rows: int, mm_per_px: float, auto: tuple[float, float], expected_cm: float | None
    ) -> None:
        self.rows, self.mm_per_px, self.expected_cm = int(rows), float(mm_per_px), expected_cm
        if self.source == "auto":
            self.top_cm, self.bottom_cm = clamp_insets(auto[0], auto[1], self.extent_cm)
        else:
            self.top_cm, self.bottom_cm = clamp_insets(self.top_cm, self.bottom_cm, self.extent_cm)
        self._update_text()

    def _update_text(self) -> None:
        txt = f"Core length: {self.length_cm:.2f} cm ({self.source})"
        if self.expected_cm:
            chk = length_check(self.length_cm, self.expected_cm)
            txt += f"\nList: {self.expected_cm:.1f} cm, difference {chk['length_difference_cm']:+.2f} cm"
        else:
            txt += "\nNo master-list length for this scan."
        self.text.set(txt)
        self.top_var.set(f"{self.top_cm:.2f}")
        self.bottom_var.set(f"{self.bottom_cm:.2f}")

    # ----- controls (entry boxes and go-to buttons) ---------------------------

    def controls(self, parent: tk.Misc, wraplength: int = 300) -> ttk.Frame:
        """Readout, exact insets and go-to buttons; the owner packs/grids the returned frame."""
        fr = ttk.Frame(parent)
        ttk.Label(fr, textvariable=self.text, wraplength=wraplength, font=("", 8)).pack(anchor=tk.W)
        for label, var, which, color in (
            ("Top inset (cm)", self.top_var, "top", TOP_COLOR),
            ("Bottom inset (cm)", self.bottom_var, "bottom", BOTTOM_COLOR),
        ):
            row = ttk.Frame(fr)
            row.pack(fill=tk.X, pady=(3, 0))
            tk.Label(row, text="■", fg=color).pack(side=tk.LEFT)
            ttk.Label(row, text=label, font=("", 8)).pack(side=tk.LEFT, padx=(2, 4))
            ent = ttk.Entry(row, textvariable=var, width=7)
            ent.pack(side=tk.LEFT)
            ent.bind("<Return>", lambda _e: self.apply_entries())
            ent.bind("<FocusOut>", lambda _e: self.apply_entries())
            ttk.Button(row, text="Go to", width=6, command=lambda w=which: self.goto(w)).pack(
                side=tk.LEFT, padx=(4, 0)
            )
        return fr

    def apply_entries(self) -> None:
        try:
            t = float(self.top_var.get().strip() or "0")
            b = float(self.bottom_var.get().strip() or "0")
        except ValueError:
            self._update_text()
            return
        if abs(t - self.top_cm) < 0.005 and abs(b - self.bottom_cm) < 0.005:
            return
        self.top_cm, self.bottom_cm = clamp_insets(t, b, self.extent_cm)
        self.source = "manual"
        self._update_text()
        self._redraw_lines()
        if self.on_change:
            self.on_change()

    def goto(self, which: str) -> None:
        """Scroll the canvas so the chosen guide is in the middle of the view."""
        c = self.canvas
        if c is None or self.rows <= 0:
            return
        px_per_cm = 10.0 / self.mm_per_px
        row = self.top_cm * px_per_cm if which == "top" else self.rows - self.bottom_cm * px_per_cm
        y = self._row_origin + row * self._scale
        region = c.cget("scrollregion")
        if not region:
            return
        total_h = float(str(region).split()[3])
        if total_h <= 0:
            return
        c.yview_moveto(max(0.0, (y - c.winfo_height() / 2.0) / total_h))

    # ----- drawing and dragging ---------------------------------------------

    def draw(self, x0: float, y0: float, scale: float, width_disp: float, row_offset_px: float = 0) -> None:
        """Draw over an image whose top-left is at (x0, y0) on the canvas.

        ``scale`` is canvas px per image px; ``row_offset_px`` is how many rows the
        previewed image starts above the exported frame (0 when the preview is cropped).
        """
        c = self.canvas
        if c is None:
            return
        c.delete(TAG)
        if not self.enabled or self.rows <= 0:
            return
        self._scale = max(scale, 1e-9)
        self._row_origin = y0 - row_offset_px * self._scale
        px_per_cm = 10.0 / self.mm_per_px
        self._x_range = (x0, x0 + width_disp)
        for which, color, row in (
            ("top", TOP_COLOR, self.top_cm * px_per_cm),
            ("bottom", BOTTOM_COLOR, self.rows - self.bottom_cm * px_per_cm),
        ):
            y = self._row_origin + row * self._scale
            c.create_line(x0, y, x0 + width_disp, y, fill=color, width=2, tags=(TAG,))
            c.create_text(
                x0 + 4, y - 8 if which == "top" else y + 8, text=which, fill=color, anchor="w", tags=(TAG,),
                font=("", 8),
            )

    # Pointer handling is done by the owner window's canvas bindings, the way the multi-view
    # tool does it: a left press near a guide grabs it, a left press anywhere else pans.

    def _guide_y(self, which: str) -> float:
        px_per_cm = 10.0 / self.mm_per_px
        row = self.top_cm * px_per_cm if which == "top" else self.rows - self.bottom_cm * px_per_cm
        return self._row_origin + row * self._scale

    def hit(self, event, thresh_px: float = 5.0) -> str | None:
        """'top' / 'bottom' if the pointer is within ``thresh_px`` of a guide, else None."""
        c = self.canvas
        if c is None or not self.enabled or self.rows <= 0:
            return None
        cx, cy = c.canvasx(event.x), c.canvasy(event.y)
        x0, x1 = self._x_range
        if not (x0 <= cx <= x1):
            return None
        best: tuple[float, str] | None = None
        for which in ("top", "bottom"):
            d = abs(cy - self._guide_y(which))
            if d <= thresh_px and (best is None or d < best[0]):
                best = (d, which)
        return best[1] if best else None

    def press(self, event) -> bool:
        """Start dragging a guide if one is under the pointer. False means: pan instead."""
        which = self.hit(event)
        self._drag = which
        return which is not None

    def motion(self, event) -> None:
        """Hover feedback: a resize cursor over a guide."""
        if self.canvas is not None and self._drag is None:
            self.canvas.configure(cursor="sb_v_double_arrow" if self.hit(event) else "")

    def drag(self, event) -> bool:
        if self._drag is None:
            return False
        c = self.canvas
        if c is not None:
            # keep going past the window edge: scroll the view while the pointer is outside it
            h = c.winfo_height()
            if event.y < 0:
                c.yview_scroll(-1, "units")
            elif event.y > h:
                c.yview_scroll(1, "units")
        self._move(event)
        return True

    def release(self, event) -> bool:
        if self._drag is None:
            return False
        self._release(event)
        return True

    def _move(self, event) -> None:
        if self._drag is None or self.rows <= 0 or self.canvas is None:
            return
        row = (self.canvas.canvasy(event.y) - self._row_origin) / self._scale
        cm = max(0.0, min(row, self.rows)) * self.mm_per_px / 10.0
        if self._drag == "top":
            self.top_cm, self.bottom_cm = clamp_insets(cm, self.bottom_cm, self.extent_cm)
        else:
            self.top_cm, self.bottom_cm = clamp_insets(self.top_cm, self.extent_cm - cm, self.extent_cm)
        self.source = "manual"
        self._update_text()
        self._redraw_lines()

    def _redraw_lines(self) -> None:
        # cheap redraw: keep the image, redraw only the guide items
        if self.canvas is None:
            return
        bbox = self.canvas.bbox("img")
        if bbox:
            self.draw(bbox[0], bbox[1], self._scale, bbox[2] - bbox[0],
                      (bbox[1] - self._row_origin) / self._scale)

    def _release(self, _event) -> None:
        self._drag = None
        if self.on_change:
            self.on_change()
