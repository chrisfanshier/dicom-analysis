"""Stitch progress store: a small writable SQLite beside the (rebuilt) catalog.

The catalog is regenerated every release, so progress lives outside it and is
joined on ``stitch_group`` / series UIDs. A roll-up per group is mirrored to the
git-tracked ``config/ct_stitch_status.csv`` so the catalog build can show it.
"""

from __future__ import annotations

import csv
import getpass
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core_stitch.catalog import refuse_source_drive
from core_stitch.worklist import Job

STATUSES = ("todo", "loaded", "exported", "skipped", "problem")
CSV_COLUMNS = ["stitch_group", "status", "note", "who", "date", "length_check", "length_diff_cm"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS job (
    job_key TEXT PRIMARY KEY,
    stitch_group TEXT NOT NULL,
    cruise_folder TEXT,
    kind TEXT,
    plane TEXT,
    uid_a TEXT,
    uid_b TEXT,
    label TEXT,
    status TEXT NOT NULL DEFAULT 'todo',
    note TEXT DEFAULT '',
    who TEXT DEFAULT '',
    updated TEXT,
    length_check TEXT DEFAULT '',
    measured_cm REAL,
    expected_cm REAL,
    length_diff_cm REAL
);
CREATE INDEX IF NOT EXISTS job_group ON job(stitch_group);
CREATE TABLE IF NOT EXISTS export_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    out_dir TEXT NOT NULL,
    files_json TEXT,
    who TEXT,
    exported_at TEXT NOT NULL,
    length_check TEXT DEFAULT '',
    measured_cm REAL,
    expected_cm REAL,
    length_diff_cm REAL
);
CREATE INDEX IF NOT EXISTS export_job ON export_run(job_key);
"""


@dataclass
class ReconcileResult:
    """What ``ProgressStore.reconcile`` found when it compared the progress file with the export folders."""

    checked: int = 0  # export records looked at
    missing: int = 0  # records whose export folder (or all its files) is gone
    restored: int = 0  # records that had been flagged missing but are back on disk
    reset_jobs: list[str] = field(default_factory=list)  # job keys set back to 'todo'
    unrecorded_dirs: int = 0  # vNNN folders under the export root that no record points at

    @property
    def changed(self) -> bool:
        return bool(self.missing or self.restored or self.reset_jobs)

    def text(self) -> str:
        if not self.checked and not self.unrecorded_dirs:
            return "No exports recorded yet."
        parts = []
        if self.reset_jobs:
            parts.append(f"{len(self.reset_jobs)} job(s) set back to 'to do' because their export folder is gone")
        elif self.missing:
            parts.append(f"{self.missing} older export version(s) are gone, but those jobs still have a later one")
        if self.restored:
            parts.append(f"{self.restored} export(s) found on disk again")
        if self.unrecorded_dirs:
            parts.append(f"{self.unrecorded_dirs} export folder(s) on disk are not in the progress file (left alone)")
        if not parts:
            return f"All {self.checked} recorded export(s) are on disk."
        return "; ".join(parts) + "."


_VERSION_DIR = re.compile(r"^v\d{3,}$")


def _resolve_export_path(p: str, export_root: Path) -> Path:
    """``p`` if it exists, else re-rooted under ``export_root`` from its ``ct_stitched`` folder onwards."""
    path = Path(p)
    if path.exists():
        return path
    parts = path.parts
    if "ct_stitched" in parts:
        i = len(parts) - 1 - parts[::-1].index("ct_stitched")
        return export_root.joinpath(*parts[i + 1:])
    return path


def current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USERNAME", "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProgressStore:
    def __init__(self, path: Path | str) -> None:
        self.path = refuse_source_drive(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA)
        self._migrate()
        self.con.commit()

    def _migrate(self) -> None:
        """Add the length-check columns to a progress file made before they existed."""
        for table in ("job", "export_run"):
            have = {r["name"] for r in self.con.execute(f"PRAGMA table_info({table})")}
            for col, typ in (
                ("length_check", "TEXT DEFAULT ''"),
                ("measured_cm", "REAL"),
                ("expected_cm", "REAL"),
                ("length_diff_cm", "REAL"),
            ):
                if col not in have:
                    self.con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        have = {r["name"] for r in self.con.execute("PRAGMA table_info(export_run)")}
        if "missing" not in have:  # 1 = the export folder was not found at the last check (the record is kept)
            self.con.execute("ALTER TABLE export_run ADD COLUMN missing INTEGER DEFAULT 0")

    def close(self) -> None:
        self.con.close()

    # ----- jobs -----------------------------------------------------------

    def ensure(self, job: Job) -> None:
        self.con.execute(
            "INSERT OR IGNORE INTO job (job_key, stitch_group, cruise_folder, kind, plane,"
            " uid_a, uid_b, label, status, updated) VALUES (?,?,?,?,?,?,?,?, 'todo', ?)",
            (
                job.key,
                job.stitch_group,
                job.cruise_folder,
                job.kind,
                job.plane,
                job.a.series_uid,
                job.b.series_uid if job.b else "",
                job.label,
                _now(),
            ),
        )
        self.con.commit()

    def set_status(self, job: Job, status: str, note: str | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"Unknown status {status!r}")
        self.ensure(job)
        if note is None:
            self.con.execute(
                "UPDATE job SET status=?, who=?, updated=? WHERE job_key=?",
                (status, current_user(), _now(), job.key),
            )
        else:
            self.con.execute(
                "UPDATE job SET status=?, note=?, who=?, updated=? WHERE job_key=?",
                (status, note, current_user(), _now(), job.key),
            )
        self.con.commit()

    def status_of(self, job_key: str) -> str:
        r = self.con.execute("SELECT status FROM job WHERE job_key=?", (job_key,)).fetchone()
        return r["status"] if r else "todo"

    def note_of(self, job_key: str) -> str:
        r = self.con.execute("SELECT note FROM job WHERE job_key=?", (job_key,)).fetchone()
        return (r["note"] or "") if r else ""

    def length_of(self, job_key: str) -> tuple[str, float | None]:
        """(length_check, difference in cm) from the job's latest export."""
        r = self.con.execute(
            "SELECT length_check, length_diff_cm FROM job WHERE job_key=?", (job_key,)
        ).fetchone()
        return (r["length_check"] or "", r["length_diff_cm"]) if r else ("", None)

    def all_length_checks(self) -> dict[str, str]:
        return {
            r["job_key"]: r["length_check"] or ""
            for r in self.con.execute("SELECT job_key, length_check FROM job")
        }

    def all_statuses(self) -> dict[str, str]:
        return {r["job_key"]: r["status"] for r in self.con.execute("SELECT job_key, status FROM job")}

    # ----- exports --------------------------------------------------------

    def next_version(self, job_key: str) -> int:
        r = self.con.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM export_run WHERE job_key=?", (job_key,)
        ).fetchone()
        return int(r["v"]) + 1

    def exports_of(self, job_key: str) -> list[sqlite3.Row]:
        return list(
            self.con.execute(
                "SELECT * FROM export_run WHERE job_key=? ORDER BY version", (job_key,)
            )
        )

    def record_export(
        self,
        job: Job,
        version: int,
        out_dir: Path | str,
        files: dict[str, object],
        scale: dict | None = None,
    ) -> None:
        """Log an export. ``scale`` is the sidecar ``core_scale`` block (length check)."""
        self.ensure(job)
        sc = scale or {}
        ln = (
            str(sc.get("length_check") or ""),
            sc.get("core_length_cm"),
            sc.get("expected_length_cm"),
            sc.get("length_difference_cm"),
        )
        self.con.execute(
            "INSERT INTO export_run (job_key, version, out_dir, files_json, who, exported_at,"
            " length_check, measured_cm, expected_cm, length_diff_cm) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                job.key,
                int(version),
                str(out_dir),
                json.dumps({k: str(v) for k, v in files.items()}),
                current_user(),
                _now(),
                *ln,
            ),
        )
        # The job row always shows the length result of its latest export.
        self.con.execute(
            "UPDATE job SET status='exported', who=?, updated=?, length_check=?, measured_cm=?,"
            " expected_cm=?, length_diff_cm=? WHERE job_key=?",
            (current_user(), _now(), *ln, job.key),
        )
        self.con.commit()

    # ----- check the records against the disk ---------------------------------

    @staticmethod
    def _export_present(out_dir: str, files_json: str | None, export_root: Path) -> bool:
        """True if the export folder exists and still holds at least one of its recorded files.

        A recorded path that no longer exists (release moved to another drive or computer) is re-rooted under
        ``export_root`` from its ``ct_stitched`` folder name onwards.
        """
        def resolve(p: str) -> Path:
            return _resolve_export_path(p, export_root)

        d = resolve(out_dir)
        if not d.is_dir():
            return False
        try:
            files = [v for v in json.loads(files_json or "{}").values() if v]
        except (ValueError, AttributeError):
            files = []
        if not files:
            return any(d.iterdir())  # nothing recorded: a non-empty folder counts
        for f in files:
            fp = Path(f)
            if fp.is_absolute() or fp.parent != Path("."):
                if resolve(f).exists():
                    return True
            elif (d / f).exists():
                return True
        return False

    def reconcile(self, export_root: Path | str) -> ReconcileResult:
        """Compare the export records with the export folders on disk.

        A record whose folder (or every recorded file) is gone is flagged ``missing`` but kept, so the history and
        version numbers stay. A job marked 'exported' with no export left on disk goes back to 'todo' (with a note);
        a job that still has an earlier version on disk stays 'exported'. Folders on disk with no record are only
        counted, never touched. Nothing on disk is changed.
        """
        root = Path(export_root)
        res = ReconcileResult()
        runs = list(self.con.execute("SELECT id, job_key, out_dir, files_json, missing FROM export_run"))
        res.checked = len(runs)
        present_by_job: dict[str, int] = {}
        known_dirs: set[str] = set()
        for r in runs:
            ok = self._export_present(r["out_dir"], r["files_json"], root)
            known_dirs.add(os.path.normcase(os.path.normpath(_resolve_export_path(r["out_dir"], root))))
            if ok:
                present_by_job[r["job_key"]] = present_by_job.get(r["job_key"], 0) + 1
            if ok and r["missing"]:
                res.restored += 1
            elif not ok and not r["missing"]:
                res.missing += 1
            self.con.execute("UPDATE export_run SET missing=? WHERE id=?", (0 if ok else 1, r["id"]))
        now = _now()
        for r in self.con.execute("SELECT job_key, status, note FROM job WHERE status='exported'").fetchall():
            if present_by_job.get(r["job_key"]):
                continue
            note = (r["note"] or "").strip()
            added = f"export folder not found on {now[:10]}; set back to to do"
            self.con.execute(
                "UPDATE job SET status='todo', note=?, updated=?, length_check='', measured_cm=NULL,"
                " expected_cm=NULL, length_diff_cm=NULL WHERE job_key=?",
                ((note + "; " if note else "") + added, now, r["job_key"]),
            )
            res.reset_jobs.append(r["job_key"])
        self.con.commit()
        if root.is_dir():
            for dirpath, dirnames, _files in os.walk(root):
                for d in list(dirnames):
                    if _VERSION_DIR.match(d):
                        dirnames.remove(d)  # an export folder: do not look inside
                        full = os.path.normcase(os.path.normpath(os.path.join(dirpath, d)))
                        if full not in known_dirs:
                            res.unrecorded_dirs += 1
        return res

    # ----- group roll-up and CSV mirror -------------------------------------

    def group_status(self, job_keys: list[str]) -> str:
        """Roll up job statuses: todo / in_progress / partial / exported / skipped / problem."""
        if not job_keys:
            return "todo"
        st = [self.status_of(k) for k in job_keys]
        if "problem" in st:
            return "problem"
        n_exp = st.count("exported")
        if n_exp == len(st):
            return "exported"
        if n_exp and all(s in ("exported", "skipped") for s in st):
            return "exported"
        if n_exp:
            return "partial"
        if all(s == "skipped" for s in st):
            return "skipped"
        if any(s == "loaded" for s in st):
            return "in_progress"
        return "todo"

    def write_status_csv(self, csv_path: Path | str, group_jobs: dict[str, list[str]]) -> int:
        """Merge the roll-up of managed groups into ``ct_stitch_status.csv``.

        Rows for other groups are kept as they are; managed groups still at ``todo``
        with no history are not written. Returns the number of rows written.
        """
        csv_path = refuse_source_drive(csv_path)
        existing: dict[str, dict[str, str]] = {}
        if csv_path.is_file():
            with csv_path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("stitch_group"):
                        existing[row["stitch_group"]] = {c: row.get(c, "") or "" for c in CSV_COLUMNS}
        for group, keys in group_jobs.items():
            status = self.group_status(keys)
            touched = any(self.con.execute(
                "SELECT 1 FROM job WHERE job_key=? AND (status<>'todo' OR note<>'')", (k,)
            ).fetchone() for k in keys)
            if status == "todo" and not touched and group not in existing:
                continue  # (a group already in the file is still written, so a reset shows up there too)
            notes, last = [], ""
            for k in keys:
                r = self.con.execute("SELECT note, updated, who FROM job WHERE job_key=?", (k,)).fetchone()
                if r:
                    if r["note"]:
                        notes.append(r["note"])
                    last = max(last, r["updated"] or "")
            # group roll-up: the largest difference (by size) among its compared jobs
            worst, diff = "", ""
            for k in keys:
                chk, d = self.length_of(k)
                if chk == "compared" and d is not None and (
                    worst != "compared" or abs(d) > abs(float(diff))
                ):
                    worst, diff = chk, f"{d:.2f}"
                elif chk == "no_reference" and not worst:
                    worst = chk
            old = existing.get(group, {})
            existing[group] = {
                "stitch_group": group,
                "status": status,
                "note": "; ".join(notes) or old.get("note", ""),
                "who": current_user(),
                "date": last[:10],
                "length_check": worst or old.get("length_check", ""),
                "length_diff_cm": diff if worst else old.get("length_diff_cm", ""),
            }
        tmp = csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for g in sorted(existing):
                w.writerow(existing[g])
        os.replace(tmp, csv_path)
        return len(existing)
