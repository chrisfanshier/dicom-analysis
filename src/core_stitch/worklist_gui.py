"""Worklist tab: browse the release catalog (cruise -> core -> section), choose what to
load into the stitch workflow, and track what has been exported.

The panel owns the catalog view and the progress store; the main window (``StitchGui``)
supplies ``load_pair_job`` and shows the stitch workflow. Single (unstitched) sections
open a small dialog that exports the same JPEG/TIFF products without a join.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from core_stitch.catalog import Catalog, GroupRow, SeriesRow, find_release_root, ytools_config_dir
from core_stitch.progress import ProgressStore
from core_stitch.worklist import (
    Job,
    build_jobs,
    build_tree,
    custom_job,
    export_base_dir,
    job_tag,
    make_pair_job,
    make_single_job,
    section_label,
)

MAX_GROUPS_SHOWN = 60
STATUS_FILTERS = (
    "all",
    "to do",
    "exported",
    "problem / skipped",
    "length compared",
    "length not compared",
)


def length_text(check: str, diff: float | None) -> str:
    """Jobs-table text: the difference in cm (measured minus list), '-' if none, or blank."""
    if check == "compared":
        return "" if diff is None else f"{diff:+.1f}"
    if check == "no_reference":
        return "-"
    return ""


def length_message(scale: dict) -> tuple[str, str]:
    """(level, text) for the completion note. Information only: no threshold is applied."""
    cm = scale.get("core_length_cm")
    head = f"Core length measured: {cm:.1f} cm (endpoints: {scale.get('bounds_source', 'auto')})."
    if scale.get("length_check") == "compared":
        return "info", (
            f"{head}\nMaster list: {scale['expected_length_cm']:.1f} cm.\n"
            f"Difference: {scale['length_difference_cm']:+.1f} cm "
            f"({scale['length_difference_pct']:+.1f}%).\n\n"
            "Recorded in the worklist."
        )
    return "info", f"{head}\nNo master-list length to compare against for this scan."


@dataclass
class ExportTarget:
    out_dir: Path
    stem: str
    version: int
    extra_meta: dict


class WorklistPanel(ttk.Frame):
    def __init__(self, parent: tk.Misc, app) -> None:
        super().__init__(parent)
        self.app = app
        self.catalog: Catalog | None = None
        self.store: ProgressStore | None = None
        self.jobs: dict[str, list[Job]] = {}
        self.node_groups: dict[str, list[GroupRow]] = {}
        self.group_node: dict[str, str] = {}
        self.section_ngroups: dict[tuple[str, str, str], int] = {}
        self._struct: dict = {}
        self._series_rows: list[tuple[GroupRow, SeriesRow]] = []
        self._shown_jobs: list[Job] = []
        self.release_var = tk.StringVar(value="No release loaded")
        self.filter_text = tk.StringVar()
        self.filter_status = tk.StringVar(value=STATUS_FILTERS[0])
        self.note_var = tk.StringVar()
        self.info_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self._build()
        self.after(150, self.open_default_release)

    # ----- UI -------------------------------------------------------------

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=3)
        self.rowconfigure(5, weight=2)
        self.rowconfigure(7, weight=2)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, textvariable=self.release_var, font=("", 8), wraplength=420).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(top, text="Release…", width=9, command=self.choose_release).grid(row=0, column=1)
        ttk.Button(top, text="Reload", width=7, command=self.reload).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(top, text="Re-check exports", command=self.recheck_exports).grid(row=0, column=3, padx=(4, 0))

        flt = ttk.Frame(self)
        flt.grid(row=1, column=0, sticky="ew", pady=(4, 2))
        flt.columnconfigure(1, weight=1)
        ttk.Label(flt, text="Find").grid(row=0, column=0)
        ent = ttk.Entry(flt, textvariable=self.filter_text)
        ent.grid(row=0, column=1, sticky="ew", padx=4)
        ent.bind("<Return>", lambda _e: self.populate())
        cb = ttk.Combobox(
            flt, textvariable=self.filter_status, values=STATUS_FILTERS, width=16, state="readonly"
        )
        cb.grid(row=0, column=2)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.populate())
        ttk.Button(flt, text="Go", width=4, command=self.populate).grid(row=0, column=3, padx=(4, 0))

        tf = ttk.Frame(self)
        tf.grid(row=2, column=0, sticky="nsew")
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(tf, columns=("done",), selectmode="browse", height=10)
        self.tree.heading("#0", text="Cruise / core / section")
        self.tree.heading("done", text="done")
        self.tree.column("#0", width=330, stretch=True)
        self.tree.column("done", width=60, anchor="e", stretch=False)
        sb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.tag_configure("done", foreground="#1a7f37")
        self.tree.tag_configure("part", foreground="#b45309")
        self.tree.tag_configure("prob", foreground="#b42318")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        ttk.Label(self, textvariable=self.summary_var, font=("", 8)).grid(row=3, column=0, sticky="w")

        ttk.Label(self, text="Jobs (suggested pairs / singles)", font=("", 9, "bold")).grid(
            row=4, column=0, sticky="w", pady=(6, 0)
        )
        jf = ttk.Frame(self)
        jf.grid(row=5, column=0, sticky="nsew")
        jf.columnconfigure(0, weight=1)
        jf.rowconfigure(0, weight=1)
        self.jobs_tree = ttk.Treeview(jf, columns=("status", "length"), selectmode="browse", height=5)
        self.jobs_tree.heading("#0", text="job")
        self.jobs_tree.heading("status", text="status")
        self.jobs_tree.heading("length", text="length vs list (cm)")
        self.jobs_tree.column("length", width=110, anchor="e", stretch=False)
        self.jobs_tree.column("#0", width=380, stretch=True)
        self.jobs_tree.column("status", width=70, stretch=False)
        jsb = ttk.Scrollbar(jf, orient=tk.VERTICAL, command=self.jobs_tree.yview)
        self.jobs_tree.configure(yscrollcommand=jsb.set)
        self.jobs_tree.grid(row=0, column=0, sticky="nsew")
        jsb.grid(row=0, column=1, sticky="ns")
        self.jobs_tree.bind("<<TreeviewSelect>>", self._on_job_select)
        self.jobs_tree.bind("<Double-1>", lambda _e: self.load_selected_job())

        ttk.Label(
            self,
            text="All series in the selection (judge by description / plane / size; "
            "pick 1 = single, 2 = TOP then BTM)",
            font=("", 8),
            wraplength=430,
        ).grid(row=6, column=0, sticky="w", pady=(6, 0))
        sf = ttk.Frame(self)
        sf.grid(row=7, column=0, sticky="nsew")
        sf.columnconfigure(0, weight=1)
        sf.rowconfigure(0, weight=1)
        cols = ("role", "plane", "slices", "step", "px", "size", "comp", "desc")
        self.series_tree = ttk.Treeview(sf, columns=cols, show="headings", selectmode="extended", height=5)
        for c, w in zip(cols, (50, 70, 45, 40, 50, 70, 40, 200)):
            self.series_tree.heading(c, text=c)
            self.series_tree.column(c, width=w, stretch=(c == "desc"))
        ssb = ttk.Scrollbar(sf, orient=tk.VERTICAL, command=self.series_tree.yview)
        self.series_tree.configure(yscrollcommand=ssb.set)
        self.series_tree.grid(row=0, column=0, sticky="nsew")
        ssb.grid(row=0, column=1, sticky="ns")

        ttk.Label(self, textvariable=self.info_var, font=("", 8), wraplength=430).grid(
            row=8, column=0, sticky="w", pady=(4, 0)
        )

        btn = ttk.Frame(self)
        btn.grid(row=9, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(btn, text="Load selected job", command=self.load_selected_job).pack(side=tk.LEFT)
        ttk.Button(btn, text="Load picked series", command=self.load_picked_series).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(btn, text="Next to do", command=self.next_todo).pack(side=tk.LEFT, padx=(4, 0))

        mark = ttk.Frame(self)
        mark.grid(row=10, column=0, sticky="ew", pady=(4, 0))
        mark.columnconfigure(0, weight=1)
        ttk.Entry(mark, textvariable=self.note_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(mark, text="Skip", width=6, command=lambda: self.mark("skipped")).grid(
            row=0, column=1, padx=(4, 0)
        )
        ttk.Button(mark, text="Problem", width=8, command=lambda: self.mark("problem")).grid(
            row=0, column=2, padx=(4, 0)
        )
        ttk.Button(mark, text="Reset", width=6, command=lambda: self.mark("todo")).grid(
            row=0, column=3, padx=(4, 0)
        )

    # ----- release / catalog ---------------------------------------------

    def open_default_release(self) -> None:
        root = find_release_root()
        if root is None:
            self.release_var.set("Release folder not found — use Release… (set CORE_STITCH_RELEASE)")
            return
        self.open_release(root)

    def choose_release(self) -> None:
        p = filedialog.askdirectory(title="Release folder (contains catalog/catalog.sqlite)")
        if p:
            self.open_release(Path(p))

    def destroy(self) -> None:
        if self.store is not None:  # release the progress file when the window closes
            try:
                self.store.close()
            except Exception:
                pass
            self.store = None
        super().destroy()

    def reload(self) -> None:
        if self.catalog is not None:
            self.open_release(self.catalog.release_root)

    def open_release(self, root: Path) -> None:
        try:
            cat = Catalog(root)
            store = ProgressStore(cat.progress_path)
        except Exception as exc:  # shown to the user, never silent
            self.release_var.set(f"Cannot open release {root}: {exc}")
            messagebox.showerror("Worklist", f"Cannot open release:\n{root}\n\n{exc}")
            return
        if self.store is not None:
            self.store.close()
        self.catalog, self.store = cat, store
        self.jobs = {k: build_jobs(g) for k, g in cat.groups.items()}
        self._restore_custom_jobs()
        groups = list(cat.groups.values())
        self._struct = build_tree(groups)
        self.section_ngroups = {}
        for g in groups:
            if g.section_key:
                k = (g.cruise_folder, g.core_name, g.section_key)
                self.section_ngroups[k] = self.section_ngroups.get(k, 0) + 1
        n_jobs = sum(len(v) for v in self.jobs.values())
        self.release_var.set(
            f"{cat.release_root} — {len(groups)} scan groups, {n_jobs} jobs"
        )
        self.populate()
        self.recheck_exports(quiet=True)

    def recheck_exports(self, quiet: bool = False) -> None:
        """Compare the progress file with the export folders: a deleted export sets its job back to 'to do'.

        Runs when a release is opened (silent unless something changed) and from the Re-check exports button.
        """
        if self.catalog is None or self.store is None:
            return
        if not self.catalog.release_root.is_dir():  # the release drive is not there: change nothing
            if not quiet:
                messagebox.showwarning("Re-check exports", f"Release folder not found:\n{self.catalog.release_root}")
            return
        try:
            res = self.store.reconcile(self.catalog.export_root)
        except Exception as exc:  # shown to the user, never silent
            messagebox.showerror("Re-check exports", f"Could not check the exports:\n{exc}")
            return
        if res.changed:
            self.populate()
            self.write_csv()
        self.app.status.set("Exports re-checked: " + res.text())
        if res.changed or res.unrecorded_dirs or not quiet:
            messagebox.showinfo("Re-check exports", res.text(), parent=self.app)

    def _restore_custom_jobs(self) -> None:
        """Re-create hand-made jobs saved in the progress store (matched by series UID)."""
        assert self.catalog is not None and self.store is not None
        known = {j.key for js in self.jobs.values() for j in js}
        for r in self.store.con.execute("SELECT * FROM job"):
            if r["job_key"] in known:
                continue
            g = self.catalog.groups.get(r["stitch_group"])
            if g is None:
                continue
            by_uid = {s.series_uid: s for s in g.series}
            a = by_uid.get(r["uid_a"])
            b = by_uid.get(r["uid_b"]) if r["uid_b"] else None
            if a is None or (r["uid_b"] and b is None):
                continue
            job = make_pair_job(g, a, b, "manual") if b else make_single_job(g, a, "manual")
            if job.key == r["job_key"]:
                self.jobs[g.stitch_group].append(job)

    # ----- tree -------------------------------------------------------------

    def _group_keys(self, g: GroupRow) -> list[str]:
        return [j.key for j in self.jobs.get(g.stitch_group, [])]

    def _visible(self, g: GroupRow, st: dict[str, str], needle: str) -> bool:
        if needle:
            hay = " ".join((g.cruise_folder, g.core_name, g.section_key, g.group_dir)).lower()
            if needle not in hay:
                return False
        mode = self.filter_status.get()
        statuses = [st.get(k, "todo") for k in self._group_keys(g)]
        if mode == "to do":
            return any(s in ("todo", "loaded") for s in statuses) or not statuses
        if mode == "exported":
            return "exported" in statuses
        if mode == "problem / skipped":
            return any(s in ("problem", "skipped") for s in statuses)
        if mode in ("length compared", "length not compared"):
            want = "compared" if mode == "length compared" else "no_reference"
            lc = self.store.all_length_checks() if self.store else {}
            return any(lc.get(k) == want for k in self._group_keys(g))
        return True

    def populate(self) -> None:
        if self.store is None:
            return
        self.tree.delete(*self.tree.get_children())
        self.node_groups.clear()
        self.group_node.clear()
        st = self.store.all_statuses()
        needle = self.filter_text.get().strip().lower()
        for cruise, cores in self._struct.items():
            c_iid = f"c|{cruise}"
            c_groups: list[GroupRow] = []
            for core_label, core in cores.items():
                k_iid = f"k|{cruise}|{core_label}"
                k_groups: list[GroupRow] = []
                for sec_label, sec in core.sections.items():
                    gs = [g for g in sec.groups if self._visible(g, st, needle)]
                    if not gs:
                        continue
                    if not k_groups:
                        if not c_groups:
                            self.tree.insert("", "end", iid=c_iid, text=cruise, open=bool(needle))
                        self.tree.insert(c_iid, "end", iid=k_iid, text=core_label, open=bool(needle))
                    s_iid = f"s|{cruise}|{core_label}|{sec_label}"
                    self.tree.insert(k_iid, "end", iid=s_iid, text=sec_label)
                    self.node_groups[s_iid] = gs
                    if len(gs) > 1:
                        for g in gs:
                            g_iid = f"g|{g.stitch_group}"
                            self.tree.insert(s_iid, "end", iid=g_iid, text=g.group_dir)
                            self.node_groups[g_iid] = [g]
                            self.group_node[g.stitch_group] = g_iid
                    else:
                        self.group_node[gs[0].stitch_group] = s_iid
                    k_groups.extend(gs)
                if k_groups:
                    self.node_groups[k_iid] = k_groups
                    c_groups.extend(k_groups)
            if c_groups:
                self.node_groups[c_iid] = c_groups
        self._update_counts()

    def _update_counts(self) -> None:
        if self.store is None:
            return
        st = self.store.all_statuses()
        total_all = done_all = 0
        for iid, groups in self.node_groups.items():
            keys = [k for g in groups for k in self._group_keys(g)]
            done = sum(1 for k in keys if st.get(k) == "exported")
            prob = sum(1 for k in keys if st.get(k) == "problem")
            self.tree.set(iid, "done", f"{done}/{len(keys)}")
            tag = ()
            if prob:
                tag = ("prob",)
            elif keys and done == len(keys):
                tag = ("done",)
            elif done:
                tag = ("part",)
            self.tree.item(iid, tags=tag)
        for js in self.jobs.values():
            for j in js:
                total_all += 1
                done_all += st.get(j.key) == "exported"
        self.summary_var.set(f"Exported {done_all} of {total_all} jobs in this release")

    def _on_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        self._show_groups(self.node_groups.get(sel[0], []))

    def _show_groups(self, groups: list[GroupRow]) -> None:
        self.jobs_tree.delete(*self.jobs_tree.get_children())
        self.series_tree.delete(*self.series_tree.get_children())
        self._shown_jobs, self._series_rows = [], []
        self.info_var.set("")
        if not groups:
            return
        if len(groups) > MAX_GROUPS_SHOWN:
            self.info_var.set(f"{len(groups)} scan groups — select a core or section to see its jobs.")
            return
        st = self.store.all_statuses() if self.store else {}
        multi = len(groups) > 1
        for g in groups:
            prefix = f"[{'/'.join(g.group_dir.split('/')[-2:])}] " if multi else ""
            for j in self.jobs.get(g.stitch_group, []):
                self._shown_jobs.append(j)
                self.jobs_tree.insert(
                    "", "end", iid=j.key, text=prefix + j.label,
                    values=(st.get(j.key, "todo"), length_text(*self.store.length_of(j.key)) if self.store else ""),
                )
            for s in g.series:
                i = len(self._series_rows)
                self._series_rows.append((g, s))
                step = "" if s.slice_step_mm is None else f"{s.slice_step_mm:g}"
                self.series_tree.insert(
                    "",
                    "end",
                    iid=str(i),
                    values=(
                        s.role,
                        s.plane,
                        s.n_slices,
                        step,
                        s.spacing_text,
                        s.size_text,
                        "J2K" if s.compressed == "yes" else "raw",
                        f"{s.description}  [{s.patient_name}]" if s.patient_name else s.description,
                    ),
                )
        if len(groups) == 1:
            g = groups[0]
            bits = [g.layout, g.flags, g.part_check and f"part check: {g.part_check}"]
            self.info_var.set("  |  ".join(b for b in bits if b))
        if self._shown_jobs:
            self.jobs_tree.selection_set(self._shown_jobs[0].key)

    def _on_job_select(self, _event=None) -> None:
        job = self._selected_job()
        if job is not None and self.store is not None:
            self.note_var.set(self.store.note_of(job.key))

    def _selected_job(self) -> Job | None:
        sel = self.jobs_tree.selection()
        if not sel:
            return None
        return next((j for j in self._shown_jobs if j.key == sel[0]), None)

    # ----- actions ---------------------------------------------------------

    def load_selected_job(self) -> None:
        job = self._selected_job()
        if job is None:
            messagebox.showinfo("Worklist", "Select a job first (or pick series below).")
            return
        self.load_job(job)

    def load_picked_series(self) -> None:
        picked = [self._series_rows[int(i)] for i in self.series_tree.selection()]
        if not picked:
            messagebox.showinfo("Worklist", "Pick one series (single) or two (TOP then BTM).")
            return
        if len({g.stitch_group for g, _ in picked}) != 1:
            messagebox.showinfo("Worklist", "Pick series from one scan group only.")
            return
        g = picked[0][0]
        try:
            job = custom_job(g, [s for _, s in picked])
        except ValueError as exc:
            messagebox.showinfo("Worklist", str(exc))
            return
        if job.key not in {j.key for j in self.jobs[g.stitch_group]}:
            self.jobs[g.stitch_group].append(job)
        self.load_job(job)

    def load_job(self, job: Job) -> None:
        assert self.store is not None
        self.store.ensure(job)
        if self.store.status_of(job.key) == "exported":
            if not messagebox.askyesno(
                "Already exported",
                "This job already has an export.\nReopen it? A new export is saved as a new version.",
            ):
                return
        else:
            self.store.set_status(job, "loaded")
        self._refresh_after_change(job)
        if job.is_pair:
            self.app.load_pair_job(job)
        else:
            self.app.load_single_job(job)

    def mark(self, status: str) -> None:
        job = self._selected_job()
        if job is None or self.store is None:
            return
        self.store.set_status(job, status, note=self.note_var.get().strip())
        self._refresh_after_change(job)

    def _refresh_after_change(self, job: Job) -> None:
        self._update_counts()
        if self.jobs_tree.exists(job.key) and self.store is not None:
            self.jobs_tree.set(job.key, "status", self.store.status_of(job.key))
            self.jobs_tree.set(job.key, "length", length_text(*self.store.length_of(job.key)))
        self.write_csv()

    def write_csv(self) -> None:
        cfg = ytools_config_dir()
        if cfg is None or self.store is None:
            return
        try:
            self.store.write_status_csv(
                cfg / "ct_stitch_status.csv",
                {k: [j.key for j in js] for k, js in self.jobs.items()},
            )
        except OSError as exc:
            self.app.status.set(f"Could not update ct_stitch_status.csv: {exc}")

    def next_todo(self) -> None:
        if self.store is None:
            return
        st = self.store.all_statuses()
        order = [
            (g, j)
            for iid in self.tree.get_children("")  # cruise order
            for g in self.node_groups.get(iid, [])
            for j in self.jobs.get(g.stitch_group, [])
        ]
        if not order:
            return
        cur = self._selected_job()
        start = 0
        if cur is not None:
            start = next((i for i, (_, j) in enumerate(order) if j.key == cur.key), -1) + 1
        for g, j in order[start:] + order[:start]:
            if st.get(j.key, "todo") == "todo":
                node = self.group_node.get(g.stitch_group)
                if node:
                    self.tree.selection_set(node)
                    self.tree.see(node)
                    self.update_idletasks()
                    if self.jobs_tree.exists(j.key):
                        self.jobs_tree.selection_set(j.key)
                return
        messagebox.showinfo("Worklist", "No jobs left to do in the current view.")

    # ----- export hooks (used by StitchGui.accept_export and SingleDialog) ------

    def export_target(self, job: Job) -> ExportTarget:
        assert self.catalog is not None and self.store is not None
        g = self.catalog.groups[job.stitch_group]
        siblings = self.jobs[job.stitch_group]
        n_sec = self.section_ngroups.get((g.cruise_folder, g.core_name, g.section_key), 1) if g.section_key else 1
        base = export_base_dir(self.catalog.export_root, g, job, siblings, n_sec)
        version = self.store.next_version(job.key)
        while (base / f"v{version:03d}").exists():  # never overwrite an existing folder
            version += 1
        extra = {
            "job_kind": job.kind,
            "cruise_folder": g.cruise_folder,
            "core_name": g.core_name,
            "section_key": g.section_key,
            "export_version": version,
            "catalog_section_length_cm": g.expected_length_cm,
            "series": {
                "a": {"uid": job.a.series_uid, "description": job.a.description,
                      "plane": job.a.plane},
                "b": None if job.b is None else {
                    "uid": job.b.series_uid, "description": job.b.description,
                    "plane": job.b.plane},
            },
        }
        return ExportTarget(
            out_dir=base / f"v{version:03d}",
            stem=f"{section_label(g)}_{job_tag(job, siblings)}",
            version=version,
            extra_meta=extra,
        )

    def record_export(
        self, job: Job, target: ExportTarget, paths: dict, scale: dict | None = None,
        parent: tk.Misc | None = None,
    ) -> None:
        """Log the export and tell the user how the measured length compares to the list."""
        assert self.store is not None
        self.store.record_export(job, target.version, target.out_dir, paths, scale)
        self._refresh_after_change(job)
        if scale and scale.get("length_check"):
            level, text = length_message(scale)
            messagebox.showinfo("Core length", text, parent=parent or self.app)
