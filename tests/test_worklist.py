"""Tests for the catalog worklist: jobs, tree, progress store, single export, uid loading.

Uses a tiny synthetic catalog and synthetic DICOM series; never touches real data.
"""

from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from core_stitch.catalog import Catalog, refuse_source_drive
from core_stitch.load import load_series
from core_stitch.progress import ProgressStore
from core_stitch.single import export_single, prepare_single
from core_stitch.worklist import (
    UNMATCHED,
    build_jobs,
    build_tree,
    custom_job,
    export_base_dir,
    section_label,
)

STITCH_COLS = (
    "stitch_group cruise_folder collection group_dir section_key tier core_name "
    "stitch_layout stitch_flags part_check"
).split()
SERIES_COLS = (
    "cruise_folder rel_path series_uid n_slices series_number series_description plane "
    "pixel_spacing_mm rows columns compressed transfer_syntax patient_name stitch_group "
    "stitch_role stitch_role_basis slice_step_median_mm"
).split()


def make_catalog(root: Path) -> None:
    (root / "catalog").mkdir(parents=True)
    con = sqlite3.connect(root / "catalog" / "catalog.sqlite")
    con.execute(f"CREATE TABLE ctprobe_stitch ({','.join(c + ' TEXT' for c in STITCH_COLS)})")
    con.execute(f"CREATE TABLE ctprobe_series ({','.join(c + ' TEXT' for c in SERIES_COLS)})")

    def grp(g, cruise, gdir, sec, tier, core, layout):
        con.execute(
            "INSERT INTO ctprobe_stitch VALUES (?,?,?,?,?,?,?,?,?,?)",
            (g, cruise, "Holdings", gdir, sec, tier, core, layout, "", ""),
        )

    def ser(cruise, rel, uid, desc, plane, role, g, n=50):
        con.execute(
            "INSERT INTO ctprobe_series VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cruise, rel, uid, n, 1, desc, plane, "0.5\\0.5", "512", "512", "no",
             "Explicit VR Little Endian", "pt", g, role, "description", "2.0"),
        )

    # one-folder layout, coronal + sagittal pairs
    g1 = "CR1|ctscan/A-1"
    grp(g1, "CR1", "ctscan/A-1", "CR1-1GC-1", 1, "CR1-1GC", "parts_in_one_folder")
    ser("CR1", "ctscan/A-1", "1.1", "Bone Cor TOP", "coronal", "top", g1)
    ser("CR1", "ctscan/A-1", "1.2", "Bone Cor BTM", "coronal", "bottom", g1)
    ser("CR1", "ctscan/A-1", "1.3", "Bone Sag TOP", "sagittal", "top", g1)
    ser("CR1", "ctscan/A-1", "1.4", "Bone Sag BTM", "sagittal", "bottom", g1)
    # single scan
    g2 = "CR1|ctscan/B-2"
    grp(g2, "CR1", "ctscan/B-2", "CR1-2GC-1", 1, "CR1-2GC", "single_scan")
    ser("CR1", "ctscan/B-2/SE1", "2.1", "Bone Coronal", "coronal", "single", g2)
    # tier 2 (core only) and tier 3 (unmatched)
    g3 = "CR1|ctscan/C"
    grp(g3, "CR1", "ctscan/C", "", 2, "CR1-3GC", "one_part_only")
    ser("CR1", "ctscan/C/SE1", "3.1", "Cor TOP", "coronal", "top", g3)
    g4 = "CR2|ctscan/D"
    grp(g4, "CR2", "ctscan/D", "", 3, "", "several_series_no_part")
    ser("CR2", "ctscan/D/SE1", "4.1", "x", "coronal", "other", g4)
    ser("CR2", "ctscan/D/SE2", "4.2", "y", "coronal", "other", g4)
    # a re-scan sharing a section key
    g5 = "CR1|ctscan/A-1-redo"
    grp(g5, "CR1", "ctscan/A-1-redo", "CR1-1GC-1", 1, "CR1-1GC", "single_scan")
    ser("CR1", "ctscan/A-1-redo/SE1", "5.1", "Bone Coronal", "coronal", "single", g5)
    con.commit()
    con.close()


def write_series(folder: Path, uid: str, n: int = 6, shape=(160, 80), desc="Cor TOP") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    study = generate_uid()
    for i in range(n):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"{uid}_{i}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = uid
        ds.SeriesDescription = desc
        ds.Modality = "CT"
        ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 0, -1]
        ds.ImagePositionPatient = [0, float(i * 2), 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.Rows, ds.Columns = shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = 1
        ds.RescaleIntercept = -1024
        arr = np.zeros(shape, dtype=np.int16)  # -1024 HU air
        arr[10 : shape[0] - 10, 25:55] = 1200  # long bright core strip
        ds.PixelData = arr.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(folder / f"{uid}_{i}.dcm")


class CatalogJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "release"
        make_catalog(self.root)
        self.cat = Catalog(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_loads_groups_and_series_paths(self) -> None:
        self.assertEqual(len(self.cat.groups), 5)
        g = self.cat.groups["CR1|ctscan/A-1"]
        self.assertEqual(len(g.series), 4)
        self.assertEqual(
            g.series[0].path, self.root / "snapshots" / "ct" / "ct_snapshot" / "CR1" / "ctscan" / "A-1"
        )

    def test_pairs_per_plane_and_no_default(self) -> None:
        jobs = build_jobs(self.cat.groups["CR1|ctscan/A-1"])
        self.assertEqual([j.kind for j in jobs], ["pair", "pair"])
        self.assertEqual({j.plane for j in jobs}, {"coronal", "sagittal"})
        self.assertEqual(len({j.key for j in jobs}), 2)

    def test_single_unpaired_and_candidates(self) -> None:
        self.assertEqual(build_jobs(self.cat.groups["CR1|ctscan/B-2"])[0].kind, "single")
        j = build_jobs(self.cat.groups["CR1|ctscan/C"])
        self.assertEqual(len(j), 1)
        self.assertIn("unpaired", j[0].note)
        cand = build_jobs(self.cat.groups["CR2|ctscan/D"])
        self.assertEqual(len(cand), 2)
        self.assertTrue(all(c.kind == "single" for c in cand))

    def test_job_key_stable_across_rebuild(self) -> None:
        k1 = [j.key for j in build_jobs(self.cat.groups["CR1|ctscan/A-1"])]
        k2 = [j.key for j in build_jobs(Catalog(self.root).groups["CR1|ctscan/A-1"])]
        self.assertEqual(k1, k2)

    def test_custom_job(self) -> None:
        g = self.cat.groups["CR1|ctscan/A-1"]
        self.assertEqual(custom_job(g, [g.series[0]]).kind, "single")
        pair = custom_job(g, [g.series[0], g.series[3]])
        self.assertTrue(pair.is_pair)
        self.assertEqual(pair.plane, "coronal+sagittal")
        with self.assertRaises(ValueError):
            custom_job(g, g.series[:3])

    def test_tree_levels(self) -> None:
        tree = build_tree(list(self.cat.groups.values()))
        self.assertEqual(set(tree), {"CR1", "CR2"})
        self.assertIn(UNMATCHED, tree["CR2"])
        cr1 = tree["CR1"]
        self.assertIn("CR1-1GC", cr1)
        secs = cr1["CR1-3GC"].sections
        self.assertTrue(next(iter(secs)).startswith("(section unknown)"))
        # the re-scan shares the section node with the first scan
        self.assertEqual(len(cr1["CR1-1GC"].sections["CR1-1GC-1"].groups), 2)

    def test_export_dirs_do_not_collide(self) -> None:
        g = self.cat.groups["CR1|ctscan/A-1"]
        jobs = build_jobs(g)
        d0 = export_base_dir(self.cat.export_root, g, jobs[0], jobs, 2)
        d1 = export_base_dir(self.cat.export_root, g, jobs[1], jobs, 2)
        self.assertNotEqual(d0, d1)
        redo = self.cat.groups["CR1|ctscan/A-1-redo"]
        rj = build_jobs(redo)
        self.assertNotEqual(d0.parent, export_base_dir(self.cat.export_root, redo, rj[0], rj, 2).parent)
        long = self.cat.groups["CR2|ctscan/D"]
        long_dir = "ctscan/" + "very_long_folder_name/" * 8 + "ST000000"
        self.assertLessEqual(len(section_label(type(long)(**{**long.__dict__, "group_dir": long_dir}))), 48)
        for d in (d0, d1):
            self.assertTrue(str(d).startswith(str(self.cat.export_root)))
            self.assertNotIn("snapshots", str(d))

    def test_source_drive_refused(self) -> None:
        with self.assertRaises(ValueError):
            refuse_source_drive("Y:/Collection/Holdings")


class ProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "release"
        make_catalog(self.root)
        self.cat = Catalog(self.root)
        self.store = ProgressStore(self.cat.progress_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_status_versions_and_rollup(self) -> None:
        g = self.cat.groups["CR1|ctscan/A-1"]
        jobs = build_jobs(g)
        keys = [j.key for j in jobs]
        self.assertEqual(self.store.status_of(keys[0]), "todo")
        self.assertEqual(self.store.group_status(keys), "todo")
        self.store.set_status(jobs[0], "loaded")
        self.assertEqual(self.store.group_status(keys), "in_progress")
        self.assertEqual(self.store.next_version(keys[0]), 1)
        self.store.record_export(jobs[0], 1, self.root / "x", {"jpeg": "a.jpg"})
        self.assertEqual(self.store.status_of(keys[0]), "exported")
        self.assertEqual(self.store.next_version(keys[0]), 2)
        self.assertEqual(self.store.group_status(keys), "partial")
        self.store.set_status(jobs[1], "skipped", note="sagittal is a scout")
        self.assertEqual(self.store.group_status(keys), "exported")
        # reopening keeps history
        self.store.record_export(jobs[0], 2, self.root / "y", {"jpeg": "b.jpg"})
        self.assertEqual(len(self.store.exports_of(keys[0])), 2)

    def test_csv_merge_keeps_other_rows(self) -> None:
        csv_path = Path(self.tmp.name) / "ct_stitch_status.csv"
        csv_path.write_text(
            "stitch_group,status,note,who,date\nOTHER|x,done,by hand,chris,2026-01-01\n",
            encoding="utf-8",
        )
        g = self.cat.groups["CR1|ctscan/B-2"]
        j = build_jobs(g)[0]
        untouched = self.cat.groups["CR2|ctscan/D"]
        group_jobs = {
            g.stitch_group: [j.key],
            untouched.stitch_group: [x.key for x in build_jobs(untouched)],
        }
        self.store.write_status_csv(csv_path, group_jobs)
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        self.assertEqual([r["stitch_group"] for r in rows], ["OTHER|x"])  # nothing touched yet
        self.store.record_export(j, 1, self.root / "o", {})
        self.store.write_status_csv(csv_path, group_jobs)
        rows = {r["stitch_group"]: r for r in csv.DictReader(csv_path.open(encoding="utf-8"))}
        self.assertEqual(rows["OTHER|x"]["status"], "done")
        self.assertEqual(rows[g.stitch_group]["status"], "exported")
        self.assertNotIn(untouched.stitch_group, rows)

    def _export(self, job, version: int) -> Path:
        out = self.cat.export_root / "CR1" / f"job_{abs(hash(job.key)) % 100000}" / f"v{version:03d}"
        out.mkdir(parents=True)
        (out / "x.json").write_text("{}", encoding="utf-8")
        self.store.record_export(job, version, out, {"json": out / "x.json"})
        return out

    def test_reconcile_resets_job_when_export_folder_deleted(self) -> None:
        import shutil

        jobs = build_jobs(self.cat.groups["CR1|ctscan/A-1"])
        out0 = self._export(jobs[0], 1)
        self._export(jobs[1], 1)
        res = self.store.reconcile(self.cat.export_root)
        self.assertFalse(res.changed)
        self.assertEqual((res.checked, res.unrecorded_dirs), (2, 0))
        shutil.rmtree(out0)
        res = self.store.reconcile(self.cat.export_root)
        self.assertEqual(res.reset_jobs, [jobs[0].key])
        self.assertEqual(self.store.status_of(jobs[0].key), "todo")
        self.assertEqual(self.store.status_of(jobs[1].key), "exported")
        self.assertIn("not found", self.store.note_of(jobs[0].key))
        self.assertEqual(len(self.store.exports_of(jobs[0].key)), 1)  # history kept
        self.assertEqual(self.store.next_version(jobs[0].key), 2)  # versions keep counting
        self.assertFalse(self.store.reconcile(self.cat.export_root).changed)  # nothing more to do
        # the folder comes back: the record is un-flagged (the status is left for the operator)
        out0.mkdir(parents=True)
        (out0 / "x.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.store.reconcile(self.cat.export_root).restored, 1)

    def test_reconcile_keeps_job_with_an_earlier_version_on_disk(self) -> None:
        import shutil

        j = build_jobs(self.cat.groups["CR1|ctscan/B-2"])[0]
        self._export(j, 1)
        out2 = self._export(j, 2)
        shutil.rmtree(out2)
        res = self.store.reconcile(self.cat.export_root)
        self.assertEqual((res.missing, res.reset_jobs), (1, []))
        self.assertEqual(self.store.status_of(j.key), "exported")

    def test_reconcile_follows_a_moved_release_and_counts_unrecorded_folders(self) -> None:
        j = build_jobs(self.cat.groups["CR1|ctscan/B-2"])[0]
        out = self._export(j, 1)
        # the same export was recorded under another drive letter / computer
        self.store.con.execute("UPDATE export_run SET out_dir=?, files_json=?",
                               ("Z:\\old_release\\ct_stitched\\CR1\\" + out.parent.name + "\\v001",
                                '{"json": "Z:\\\\old_release\\\\ct_stitched\\\\CR1\\\\' + out.parent.name + '\\\\v001\\\\x.json"}'))
        self.store.con.commit()
        self.assertFalse(self.store.reconcile(self.cat.export_root).changed)
        self.assertEqual(self.store.status_of(j.key), "exported")
        (self.cat.export_root / "CR1" / "stray" / "v001").mkdir(parents=True)
        self.assertEqual(self.store.reconcile(self.cat.export_root).unrecorded_dirs, 1)

    def test_csv_shows_a_reset_job_as_todo(self) -> None:
        import shutil

        csv_path = Path(self.tmp.name) / "ct_stitch_status.csv"
        g = self.cat.groups["CR1|ctscan/B-2"]
        j = build_jobs(g)[0]
        out = self._export(j, 1)
        self.store.write_status_csv(csv_path, {g.stitch_group: [j.key]})
        shutil.rmtree(out)
        self.store.reconcile(self.cat.export_root)
        self.store.write_status_csv(csv_path, {g.stitch_group: [j.key]})
        rows = {r["stitch_group"]: r for r in csv.DictReader(csv_path.open(encoding="utf-8"))}
        self.assertEqual(rows[g.stitch_group]["status"], "todo")

    def test_progress_survives_catalog_rebuild(self) -> None:
        j = build_jobs(self.cat.groups["CR1|ctscan/B-2"])[0]
        self.store.record_export(j, 1, self.root / "o", {})
        self.store.close()
        (self.root / "catalog" / "catalog.sqlite").unlink()
        make_catalog_again = Path(self.tmp.name) / "tmp_release"
        make_catalog(make_catalog_again)
        (self.root / "catalog" / "catalog.sqlite").write_bytes(
            (make_catalog_again / "catalog" / "catalog.sqlite").read_bytes()
        )
        store = ProgressStore(Catalog(self.root).progress_path)
        self.assertEqual(store.status_of(j.key), "exported")
        store.close()
        self.store = ProgressStore(self.cat.progress_path)


class DicomTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "one_folder"
        write_series(self.dir, "9.1", n=6, desc="Cor TOP")
        write_series(self.dir, "9.2", n=4, desc="Cor BTM")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_load_by_uid_separates_parts_in_one_folder(self) -> None:
        a = load_series(self.dir, "9.1")
        b = load_series(self.dir, "9.2")
        self.assertEqual((a.volume.shape[0], a.series_description), (6, "Cor TOP"))
        self.assertEqual((b.volume.shape[0], b.series_description), (4, "Cor BTM"))
        with self.assertRaises(FileNotFoundError):
            load_series(self.dir, "0.0")

    def test_load_by_abbreviated_catalog_uid(self) -> None:
        # the catalog keeps only the last 12 characters of the UID
        folder = Path(self.tmp.name) / "short"
        write_series(folder, "1.2.392.200036.9116.2.6.1.16.1613467799.1650933191.951896", n=3)
        write_series(folder, "1.2.392.200036.9116.2.6.1.16.1613467799.1650933192.111111", n=5)
        self.assertEqual(load_series(folder, "33191.951896").volume.shape[0], 3)
        self.assertEqual(load_series(folder, "33192.111111").volume.shape[0], 5)

    def test_single_export_products(self) -> None:
        st = prepare_single(self.dir, "9.1")
        out = Path(self.tmp.name) / "out" / "v001"
        paths = export_single(st, out, stem="s", extra_meta={"job_key": "k"})
        self.assertEqual(set(paths), {"jpeg", "tiff", "tiff_hu", "json"})
        for p in paths.values():
            self.assertTrue(p.is_file() and p.stat().st_size > 0)
        import json

        meta = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertTrue(meta["unstitched"])
        self.assertEqual(meta["job_key"], "k")
        self.assertEqual(meta["scan"]["series_uid"], "9.1")
        self.assertEqual(meta["version"], "0.2.0")


if __name__ == "__main__":
    unittest.main()
