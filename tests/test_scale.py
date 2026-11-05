"""Core scale block, length check, and the worklist record of it."""

from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core_stitch.catalog import Catalog
from core_stitch.progress import ProgressStore
from core_stitch.scale import clamp_insets, core_scale_block, length_check
from core_stitch.single import export_single, prepare_single
from core_stitch.worklist import build_jobs
from core_stitch.worklist_gui import length_message, length_text
from test_worklist import make_catalog, write_series


class ScaleBlockTests(unittest.TestCase):
    def test_geometry(self) -> None:
        b = core_scale_block((1000, 200), 0.5, 2.0, 3.0, source="manual", expected_length_cm=45.0)
        self.assertAlmostEqual(b["image_depth_extent_cm"], 50.0)
        self.assertAlmostEqual(b["core_length_cm"], 45.0)
        top, bot = b["endpoints_px"]["top"], b["endpoints_px"]["bottom"]
        self.assertAlmostEqual(top[0], 40.0)
        self.assertAlmostEqual(bot[0], 940.0)
        # distance between the endpoints times the pixel size is exactly the core length
        self.assertAlmostEqual((bot[0] - top[0]) * 0.5 / 10.0, b["core_length_cm"])
        self.assertEqual(b["length_check"], "compared")
        self.assertAlmostEqual(b["length_difference_cm"], 0.0)
        self.assertEqual(b["bounds_source"], "manual")

    def test_clamp(self) -> None:
        t, b = clamp_insets(30.0, 30.0, 40.0)
        self.assertLessEqual(t + b, 40.0)
        self.assertEqual(clamp_insets(-5.0, -1.0, 40.0), (0.0, 0.0))

    def test_length_check(self) -> None:
        # no threshold: any difference is recorded as numbers, never flagged
        self.assertEqual(length_check(100.0, 100.5)["length_check"], "compared")
        w = length_check(110.0, 100.0)
        self.assertEqual(w["length_check"], "compared")
        self.assertNotIn("length_check_note", w)
        self.assertAlmostEqual(w["length_difference_cm"], 10.0)
        self.assertAlmostEqual(w["length_difference_pct"], 10.0)
        self.assertEqual(length_check(100.0, None)["length_check"], "no_reference")
        self.assertEqual(length_check(100.0, 0)["length_check"], "no_reference")

    def test_messages(self) -> None:
        blk = core_scale_block((1000, 200), 0.5, 0, 0, source="auto", expected_length_cm=40.0)
        lvl, txt = length_message(blk)
        self.assertEqual(lvl, "info")
        self.assertIn("+10.0 cm", txt)
        self.assertEqual(length_text("compared", 10.0), "+10.0")
        self.assertEqual(length_text("no_reference", None), "-")
        self.assertEqual(length_text("", None), "")


class ExportScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "one"
        write_series(self.dir, "9.1", n=6)
        self.st = prepare_single(self.dir, "9.1")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _export(self, name, **kw):
        info: dict = {}
        paths = export_single(self.st, Path(self.tmp.name) / name, stem="s", scale_out=info, **kw)
        meta = json.loads(paths["json"].read_text(encoding="utf-8"))
        return paths, meta, info

    def test_sidecar_has_scale_block(self) -> None:
        _paths, meta, info = self._export("a")
        cs = meta["core_scale"]
        self.assertEqual(cs, info)
        self.assertEqual(cs["bounds_source"], "auto")
        self.assertEqual(cs["image_rows"], meta["export_shape_px"][0])
        self.assertEqual(cs["length_check"], "no_reference")
        self.assertAlmostEqual(cs["mm_per_px"], self.st.spacing_mm)

    def test_manual_bounds_and_difference(self) -> None:
        _p, meta, _i = self._export(
            "b", core_bounds=(0.5, 0.0, "manual"), extra_meta={"catalog_section_length_cm": 500.0}
        )
        cs = meta["core_scale"]
        self.assertEqual(cs["bounds_source"], "manual")
        self.assertAlmostEqual(cs["core_top_inset_cm"], 0.5)
        self.assertEqual(cs["length_check"], "compared")
        self.assertEqual(cs["expected_length_cm"], 500.0)
        self.assertLess(cs["length_difference_cm"], -400)

    def test_clean_images_carry_resolution(self) -> None:
        paths, _meta, _ = self._export("c")
        with Image.open(paths["tiff"]) as im:
            dpi = im.info.get("dpi")
            self.assertIsNotNone(dpi)
            self.assertAlmostEqual(25.4 / dpi[0], self.st.spacing_mm, places=3)
            self.assertEqual(im.mode, "L")  # plain greyscale, nothing burned in


class ProgressLengthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "release"
        make_catalog(self.root)
        self.cat = Catalog(self.root)
        self.store = ProgressStore(self.cat.progress_path)
        self.jobs = build_jobs(self.cat.groups["CR1|ctscan/A-1"])

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_recorded_and_filterable(self) -> None:
        j = self.jobs[0]
        warn = core_scale_block((1000, 200), 0.5, 0, 0, source="auto", expected_length_cm=40.0)
        self.store.record_export(j, 1, self.tmp.name, {"jpeg": "x.jpg"}, warn)
        chk, diff = self.store.length_of(j.key)
        self.assertEqual(chk, "compared")
        self.assertAlmostEqual(diff, 10.0)
        self.assertEqual(self.store.all_length_checks()[j.key], "compared")
        self.assertAlmostEqual(self.store.exports_of(j.key)[0]["length_diff_cm"], 10.0)
        # a later export with a good length replaces the job's current result; history stays
        ok = core_scale_block((1000, 200), 0.5, 0, 0, source="manual", expected_length_cm=50.0)
        self.store.record_export(j, 2, self.tmp.name, {}, ok)
        self.assertAlmostEqual(self.store.length_of(j.key)[1], 0.0)
        self.assertEqual([r["length_diff_cm"] for r in self.store.exports_of(j.key)], [10.0, 0.0])

    def test_csv_rollup_has_length_columns(self) -> None:
        j = self.jobs[0]
        warn = core_scale_block((1000, 200), 0.5, 0, 0, source="auto", expected_length_cm=40.0)
        self.store.record_export(j, 1, self.tmp.name, {}, warn)
        csv_path = Path(self.tmp.name) / "status.csv"
        self.store.write_status_csv(csv_path, {j.stitch_group: [x.key for x in self.jobs]})
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        self.assertEqual(rows[0]["length_check"], "compared")
        self.assertEqual(rows[0]["length_diff_cm"], "10.00")

    def test_old_progress_file_is_migrated(self) -> None:
        old = Path(self.tmp.name) / "old.sqlite"
        con = sqlite3.connect(old)
        con.executescript(
            "CREATE TABLE job (job_key TEXT PRIMARY KEY, stitch_group TEXT NOT NULL, cruise_folder TEXT,"
            " kind TEXT, plane TEXT, uid_a TEXT, uid_b TEXT, label TEXT, status TEXT NOT NULL DEFAULT 'todo',"
            " note TEXT DEFAULT '', who TEXT DEFAULT '', updated TEXT);"
            "CREATE TABLE export_run (id INTEGER PRIMARY KEY AUTOINCREMENT, job_key TEXT NOT NULL,"
            " version INTEGER NOT NULL, out_dir TEXT NOT NULL, files_json TEXT, who TEXT,"
            " exported_at TEXT NOT NULL);"
        )
        con.commit()
        con.close()
        s = ProgressStore(old)
        s.record_export(self.jobs[0], 1, self.tmp.name, {}, None)
        self.assertEqual(s.length_of(self.jobs[0].key)[0], "")
        s.close()

    def test_catalog_reads_master_list_length(self) -> None:
        con = sqlite3.connect(self.root / "catalog" / "catalog.sqlite")
        con.execute(
            "CREATE TABLE meta_section_units (cruise_folder TEXT, section_key TEXT, section_length_cm REAL)"
        )
        con.execute("INSERT INTO meta_section_units VALUES ('CR1','CR1-1GC-1',151.0)")
        con.commit()
        con.close()
        cat = Catalog(self.root)
        self.assertEqual(cat.groups["CR1|ctscan/A-1"].expected_length_cm, 151.0)
        self.assertIsNone(cat.groups["CR1|ctscan/B-2"].expected_length_cm)


class SingleViewInMainWindowTests(unittest.TestCase):
    """A scan with one view opens in the main window (no second window) and behaves like
    the multi-view tool: left-drag pans, the wheel zooms at the pointer, guides can be dragged."""

    def setUp(self) -> None:
        import os
        import tkinter as tk

        self.tmp = tempfile.TemporaryDirectory()
        self.release = Path(self.tmp.name) / "release"
        make_catalog(self.release)
        self.dir = Path(self.tmp.name) / "one"
        write_series(self.dir, "9.1", n=6, shape=(400, 80))
        self._env = {k: os.environ.get(k) for k in ("CORE_STITCH_RELEASE", "CORE_STITCH_YTOOLS_CONFIG")}
        os.environ["CORE_STITCH_RELEASE"] = str(self.release)
        os.environ["CORE_STITCH_YTOOLS_CONFIG"] = str(Path(self.tmp.name) / "cfg")
        (Path(self.tmp.name) / "cfg").mkdir()
        from core_stitch.gui import StitchGui

        try:
            self.app = StitchGui()
        except tk.TclError:
            self.skipTest("no display")
        self.app.geometry("1200x800")
        self.app.update()

    def tearDown(self) -> None:
        import os

        try:
            self.app.destroy()
        except Exception:
            pass
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def _job(self):
        import dataclasses

        job = build_jobs(self.app.worklist.catalog.groups["CR1|ctscan/B-2"])[0]
        return dataclasses.replace(job, a=dataclasses.replace(job.a, path=self.dir, series_uid="9.1"))

    class _Ev:
        def __init__(self, x, y):
            self.x, self.y = x, y

    def test_single_view_flow(self) -> None:
        from unittest import mock

        app = self.app
        app.load_single_job(self._job())
        app.refresh()
        app.update()
        self.assertIsNotNone(app.single)
        self.assertIsNone(app.state)
        self.assertIn("Only 1 view", app.mode_note.get())
        self.assertEqual([c for c in app.winfo_children() if c.winfo_class() == "Toplevel"], [])
        self.assertGreaterEqual(len(app.canvas.find_withtag("coreguide")), 4)

        # wheel zoom at the pointer
        fit_scale = app._scale
        app._zoom_at(50, 50, 4.0)
        app.update()
        self.assertGreater(app._scale, fit_scale * 3.5)

        # left-drag on empty image area pans the view
        app.canvas.xview_moveto(0.0)
        app.canvas.yview_moveto(0.2)
        app.update()
        y_before = app.canvas.yview()[0]
        e = self._Ev(10, 30)
        self.assertIsNone(app.guides.hit(e))  # nowhere near a guide, so this press pans
        app._canvas_press(e)
        self.assertIsNone(app.guides._drag)
        app._canvas_drag(self._Ev(10, 110))
        app.update()
        self.assertNotEqual(app.canvas.yview()[0], y_before)
        app._canvas_release(e)

        # a guide can be dragged right across the view, to the end of the image, but not past the other
        app._zoom_fit()
        app.update()
        g = app.guides
        wy_top = int(g._guide_y("top") - app.canvas.canvasy(0))
        wx = int(g._x_range[0] + 5 - app.canvas.canvasx(0))
        press = self._Ev(wx, wy_top)
        self.assertEqual(g.hit(press), "top")
        app._canvas_press(press)
        app._canvas_drag(self._Ev(wx, app.canvas.winfo_height() - 2))
        app._canvas_release(self._Ev(wx, app.canvas.winfo_height() - 2))
        self.assertEqual(g.source, "manual")
        self.assertLessEqual(g.top_cm + g.bottom_cm, g.extent_cm)
        self.assertGreater(g.top_cm, 0.0)

        # export: image products only, with the scale block in the sidecar
        with mock.patch("tkinter.messagebox.showinfo"), mock.patch("tkinter.messagebox.showwarning"):
            app.accept_export()
        outs = list((self.release / "ct_stitched").rglob("*.json"))
        self.assertEqual(len(outs), 1)
        meta = json.loads(outs[0].read_text(encoding="utf-8"))
        self.assertEqual(meta["core_scale"]["bounds_source"], "manual")
        self.assertTrue(meta["unstitched"])
        self.assertFalse(list((self.release / "ct_stitched").rglob("dicom_mpr")))


if __name__ == "__main__":
    unittest.main()
