"""Turn catalog scan groups into stitch *jobs* and a cruise -> core -> section tree."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core_stitch.catalog import GroupRow, SeriesRow

UNMATCHED = "(unmatched)"


@dataclass(frozen=True)
class Job:
    """One stitchable unit: a TOP/BTM pair, or a single series that needs no stitch."""

    key: str
    stitch_group: str
    cruise_folder: str
    kind: str  # "pair" | "single"
    plane: str
    a: SeriesRow  # TOP (fixed) or the single series
    b: SeriesRow | None  # BTM (mobile); None for single
    label: str
    note: str = ""

    @property
    def is_pair(self) -> bool:
        return self.kind == "pair" and self.b is not None


def job_key(stitch_group: str, plane: str, uid_a: str, uid_b: str = "") -> str:
    return "|".join((stitch_group, plane or "?", uid_a, uid_b))


def _short(text: str, n: int = 38) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def make_pair_job(group: GroupRow, a: SeriesRow, b: SeriesRow, note: str = "") -> Job:
    plane = a.plane if a.plane == b.plane else f"{a.plane}+{b.plane}"
    label = f"PAIR {plane}: {_short(a.description)} + {_short(b.description)}"
    return Job(
        key=job_key(group.stitch_group, plane, a.series_uid, b.series_uid),
        stitch_group=group.stitch_group,
        cruise_folder=group.cruise_folder,
        kind="pair",
        plane=plane,
        a=a,
        b=b,
        label=label,
        note=note,
    )


def make_single_job(group: GroupRow, s: SeriesRow, note: str = "") -> Job:
    label = f"SINGLE {s.plane}: {_short(s.description)} ({s.n_slices} sl)"
    if note:
        label += f" [{note}]"
    return Job(
        key=job_key(group.stitch_group, s.plane, s.series_uid),
        stitch_group=group.stitch_group,
        cruise_folder=group.cruise_folder,
        kind="single",
        plane=s.plane,
        a=s,
        b=None,
        label=label,
        note=note,
    )


def build_jobs(group: GroupRow) -> list[Job]:
    """Suggested jobs for a group (the user can still pick any series by hand).

    * every TOP x BTM pair of the same plane -> pair job (all planes, no default);
    * a series with role ``single`` -> single job;
    * an unpaired TOP/BTM (one part only) -> single job, labelled;
    * groups with no pair and no single (several series, no part named) offer each
      series as a single-job candidate so nothing is lost.
    """
    tops = [s for s in group.series if s.role == "top"]
    bottoms = [s for s in group.series if s.role == "bottom"]
    singles = [s for s in group.series if s.role == "single"]
    others = [s for s in group.series if s.role not in ("top", "bottom", "single")]

    jobs: list[Job] = []
    paired_uids: set[str] = set()
    for t in tops:
        for b in bottoms:
            if t.plane == b.plane and t.series_uid != b.series_uid:
                jobs.append(make_pair_job(group, t, b))
                paired_uids.update((t.series_uid, b.series_uid))
    for s in singles:
        jobs.append(make_single_job(group, s))
    for s in tops + bottoms:
        if s.series_uid not in paired_uids:
            jobs.append(make_single_job(group, s, note=f"unpaired {s.role}"))
    if not jobs:
        for s in others:
            jobs.append(make_single_job(group, s, note="candidate"))
    # De-duplicate keys (identical series listed twice) while keeping order.
    seen: set[str] = set()
    out: list[Job] = []
    for j in jobs:
        if j.key not in seen:
            seen.add(j.key)
            out.append(j)
    return out


def custom_job(group: GroupRow, picked: list[SeriesRow]) -> Job:
    """Job from the user's own series choice: 1 series = single, 2 = pair (first is TOP)."""
    if len(picked) == 1:
        return make_single_job(group, picked[0], note="manual")
    if len(picked) == 2:
        return make_pair_job(group, picked[0], picked[1], note="manual")
    raise ValueError("Pick one series (single) or two (TOP then BTM).")


# ----- tree -----------------------------------------------------------------


@dataclass
class SectionNode:
    label: str
    groups: list[GroupRow]


@dataclass
class CoreNode:
    label: str
    sections: dict[str, SectionNode]


def build_tree(groups: list[GroupRow]) -> dict[str, dict[str, CoreNode]]:
    """cruise -> core -> section -> groups.

    * tier 1: core + section known;
    * tier 2: core known, section not: section node labelled "(section unknown) <folder>";
    * tier 3: neither: core node "(unmatched)", section node labelled by the folder.
    """
    tree: dict[str, dict[str, CoreNode]] = {}
    for g in sorted(groups, key=lambda x: (x.cruise_folder, x.core_name, x.section_key, x.group_dir)):
        cores = tree.setdefault(g.cruise_folder, {})
        core_label = g.core_name or UNMATCHED
        core = cores.setdefault(core_label, CoreNode(core_label, {}))
        if g.section_key:
            sec_label = g.section_key
        elif g.core_name:
            sec_label = f"(section unknown) {g.group_dir}"
        else:
            sec_label = g.group_dir
        sec = core.sections.setdefault(sec_label, SectionNode(sec_label, []))
        sec.groups.append(g)
    return tree


# ----- export naming ---------------------------------------------------------

_BAD = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str, limit: int = 80) -> str:
    """Filesystem-safe name; long folder paths keep their tail (the distinctive end)."""
    s = _BAD.sub("_", text.replace("/", "__")).strip("_.")
    return (s or "x")[-limit:].strip("_.") or "x"


def section_label(group: GroupRow) -> str:
    # 48 chars keeps export paths well under the Windows 260-character limit
    return slug(group.section_key or group.group_dir, 48)


def job_tag(job: Job, siblings: list[Job]) -> str:
    """Folder tag for a job; adds UID tails only when the tag would collide."""
    base = f"{slug(job.plane)}_{job.kind}"
    same = [j for j in siblings if f"{slug(j.plane)}_{j.kind}" == base]
    if len(same) <= 1:
        return base
    tail = job.a.series_uid[-6:]
    if job.b is not None:
        tail += "-" + job.b.series_uid[-6:]
    return f"{base}_{slug(tail)}"


def export_base_dir(
    export_root: Path,
    group: GroupRow,
    job: Job,
    siblings: list[Job],
    n_groups_in_section: int = 1,
) -> Path:
    """<export_root>/<cruise>/<core>/<section>/<tag>; version folders live below it.

    When several scan groups share one section (re-scans), the scan folder name is
    added to the section folder so their exports never mix.
    """
    sec = section_label(group)
    if n_groups_in_section > 1:
        sec = f"{sec}__{slug(group.group_dir, 40)}"
    core = slug(group.core_name) if group.core_name else UNMATCHED.strip("()")
    # Unmatched/unknown-section groups share a section folder per scan folder, so two
    # scans of the same core never land in the same place.
    return (
        Path(export_root)
        / slug(group.cruise_folder)
        / core
        / sec
        / job_tag(job, siblings)
    )
