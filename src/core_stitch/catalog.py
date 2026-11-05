"""Read-only access to the y_drive_tools release catalog (CT scan groups and series).

The catalog (``<release>/catalog/catalog.sqlite``) is treated as canon. It is
opened read-only; progress is kept elsewhere (see ``progress.py``). Nothing here
touches the source drive: all paths are local copies under the release folder.
"""

from __future__ import annotations

import os
import sqlite3
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ENV_RELEASE = "CORE_STITCH_RELEASE"
ENV_CONFIG_DIR = "CORE_STITCH_YTOOLS_CONFIG"

_YTOOLS_CONFIG = (
    Path.home()
    / "Documents"
    / "y_drive_tools"
    / "osu-mgr-ydrive-handoff"
    / "osu-mgr-ydrive"
    / "config"
)


def refuse_source_drive(path: Path | str) -> Path:
    """The Y: drive is read-only source material: never use it here."""
    p = Path(path)
    if str(p).upper().startswith("Y:"):
        raise ValueError(f"Refusing to use the source drive: {p}")
    return p


def ytools_config_dir() -> Path | None:
    env = os.environ.get(ENV_CONFIG_DIR)
    d = Path(env) if env else _YTOOLS_CONFIG
    return d if d.is_dir() else None


def find_release_root() -> Path | None:
    """Release folder from $CORE_STITCH_RELEASE, else y_drive_tools config/release.toml."""
    env = os.environ.get(ENV_RELEASE)
    if env:
        return refuse_source_drive(env)
    cfg = ytools_config_dir()
    if cfg is None:
        return None
    toml = cfg / "release.toml"
    if not toml.is_file():
        return None
    try:
        with toml.open("rb") as fh:
            root = tomllib.load(fh).get("release_root")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return refuse_source_drive(root) if root else None


@dataclass(frozen=True)
class SeriesRow:
    """One DICOM series (a row of ``ctprobe_series``)."""

    cruise_folder: str
    rel_path: str
    series_uid: str
    n_slices: int
    description: str
    plane: str
    pixel_spacing: str
    rows: str
    columns: str
    compressed: str
    patient_name: str
    role: str  # top | bottom | single | other
    role_basis: str
    slice_step_mm: float | None
    path: Path  # local snapshot folder holding the series

    @property
    def size_text(self) -> str:
        return f"{self.rows}x{self.columns}" if self.rows else ""

    @property
    def spacing_text(self) -> str:
        return self.pixel_spacing.split("\\")[0] if self.pixel_spacing else ""


@dataclass
class GroupRow:
    """One scan group (a row of ``ctprobe_stitch``) with its series."""

    stitch_group: str
    cruise_folder: str
    collection: str
    group_dir: str
    section_key: str
    core_name: str
    tier: int
    layout: str
    flags: str
    part_check: str
    series: list[SeriesRow] = field(default_factory=list)
    # Section length (cm) from the master list (meta_section_units); None when unknown
    expected_length_cm: float | None = None


def _s(v) -> str:
    return "" if v is None else str(v)


def _i(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Catalog:
    """In-memory view of the CT groups/series of one release."""

    def __init__(self, release_root: Path | str) -> None:
        self.release_root = refuse_source_drive(release_root)
        self.catalog_path = self.release_root / "catalog" / "catalog.sqlite"
        self.snapshot_root = self.release_root / "snapshots" / "ct" / "ct_snapshot"
        self.groups: dict[str, GroupRow] = {}
        self.load()

    @property
    def progress_path(self) -> Path:
        return self.release_root / "catalog" / "stitch_progress.sqlite"

    @property
    def export_root(self) -> Path:
        return self.release_root / "ct_stitched"

    def series_dir(self, cruise_folder: str, rel_path: str) -> Path:
        return self.snapshot_root.joinpath(cruise_folder, *rel_path.split("/"))

    def load(self) -> None:
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")
        uri = "file:" + self.catalog_path.as_posix() + "?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        con.row_factory = sqlite3.Row
        try:
            groups: dict[str, GroupRow] = {}
            for r in con.execute("SELECT * FROM ctprobe_stitch"):
                g = GroupRow(
                    stitch_group=_s(r["stitch_group"]),
                    cruise_folder=_s(r["cruise_folder"]),
                    collection=_s(r["collection"]),
                    group_dir=_s(r["group_dir"]),
                    section_key=_s(r["section_key"]),
                    core_name=_s(r["core_name"]),
                    tier=_i(r["tier"]),
                    layout=_s(r["stitch_layout"]),
                    flags=_s(r["stitch_flags"]),
                    part_check=_s(r["part_check"]),
                )
                groups[g.stitch_group] = g
            lengths: dict[tuple[str, str], float] = {}
            try:
                for r in con.execute(
                    "SELECT cruise_folder, section_key, section_length_cm FROM meta_section_units"
                    " WHERE section_length_cm IS NOT NULL"
                ):
                    v = _f(r["section_length_cm"])
                    if v and v > 0:
                        lengths[(_s(r["cruise_folder"]), _s(r["section_key"]))] = v
            except sqlite3.Error:
                pass  # older catalogs without the master list: no length check
            for g in groups.values():
                if g.section_key:
                    g.expected_length_cm = lengths.get((g.cruise_folder, g.section_key))
            for r in con.execute(
                "SELECT * FROM ctprobe_series WHERE stitch_group IS NOT NULL "
                "AND stitch_group <> '' ORDER BY cruise_folder, rel_path, series_number"
            ):
                g = groups.get(_s(r["stitch_group"]))
                if g is None:
                    continue
                g.series.append(
                    SeriesRow(
                        cruise_folder=_s(r["cruise_folder"]),
                        rel_path=_s(r["rel_path"]),
                        series_uid=_s(r["series_uid"]),
                        n_slices=_i(r["n_slices"]),
                        description=_s(r["series_description"]),
                        plane=_s(r["plane"]),
                        pixel_spacing=_s(r["pixel_spacing_mm"]),
                        rows=_s(r["rows"]),
                        columns=_s(r["columns"]),
                        compressed=_s(r["compressed"]),
                        patient_name=_s(r["patient_name"]),
                        role=_s(r["stitch_role"]) or "other",
                        role_basis=_s(r["stitch_role_basis"]),
                        slice_step_mm=_f(r["slice_step_median_mm"]),
                        path=self.series_dir(_s(r["cruise_folder"]), _s(r["rel_path"])),
                    )
                )
            self.groups = groups
        finally:
            con.close()

    def cruises(self) -> list[str]:
        return sorted({g.cruise_folder for g in self.groups.values()})
