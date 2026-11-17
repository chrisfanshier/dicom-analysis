"""A profile recipe: every choice that defines one profile, as JSON, so a profile can be re-made and compared."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .classes import ClassParams, Cls
from .reduce import Region

SCHEMA = "core_profile_recipe/1"


@dataclass
class Recipe:
    region: Region = field(default_factory=Region)
    use: tuple[Cls, ...] = (Cls.MATRIX,)
    params: ClassParams = field(default_factory=ClassParams)
    core_thr: float = 0.8  # value separating core+liner from air
    zero_offset: float = 0.0  # 0 for attenuation-like units, 1000 for HU
    footprint_fwhm_mm: float | None = None  # smooth the profile to a detector footprint (e.g. gamma beam)
    note: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["use"] = [c.name for c in self.use]
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Recipe":
        d = json.loads(text)
        if d.pop("schema", None) != SCHEMA:
            raise ValueError("not a core_profile recipe")
        d["region"] = Region(**d["region"])
        d["use"] = tuple(Cls[n] for n in d["use"])
        d["params"] = ClassParams(**d["params"])
        return cls(**d)


# Starting values for the three unit-dependent settings. The numbers are starting points, not calibrated truths:
# HU: air -1000, water 0, so "core + liner" is anything above about -300 and "void" is air-like (below -200).
UNIT_PRESETS = {
    "attenuation (positive, Geotek-like)": dict(core_thr=0.8, zero_offset=0.0, void_below=0.5),
    "HU (medical CT)": dict(core_thr=-300.0, zero_offset=1000.0, void_below=-200.0),
}
