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
import sqlite3
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
            if status == "todo" and not touched:
                continue
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
