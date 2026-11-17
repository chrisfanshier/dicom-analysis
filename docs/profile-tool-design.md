# Down-core profile tool: design

Status: the engine (`src/core_profile/`) exists with synthetic-core tests. The PySide6 viewer (`core-profile`) is being rebuilt piece by piece from the stitch window's controls: so far it loads a stitched coronal MPR, windows it, sets the core ends, and builds a sediment mask (threshold, optional cylinder that can follow the axis) with a mean-HU-per-depth profile. Saving the profile, region options and MSCL co-plot are still to do.

## What it is for

Make a profile of CT value down a core section, in several ways, and compare the ways with each other and with
the MSCL gamma density. A colleague should be able to read a profile's *recipe* (one JSON file) and know exactly
what was averaged.

## Layers

1. **Engine** (`core_profile`, no GUI imports). Works on one *core-aligned axial volume* `(along, row, col)` plus
   its voxel size and slice spacing.
   - `geometry`: per-slice centre, outer radius, inner liner wall, sediment radius.
   - `levels`: expected matrix level per voxel = radial cupping shape x per-slice scalar, both taken as modes so
     clasts and voids cannot move them.
   - `classes`: matrix / void / clast / low-density / dense patch / wall contact, judged against that level.
   - `reduce`: region (whole disc, central disc, vertical band) x classes counted -> one value per slice;
     class fractions; smoothing to a detector footprint.
   - `sedct`: the profile part of the SedCT method, for comparison (MATLAB parity not verified).
   - `recipe`: every choice that defines a profile, as JSON.
2. **App** (PySide6 + pyqtgraph, separate from the stitch window; partly built, see Status): load a volume, show the profile with
   class overlays, co-plot MSCL, shift/stretch to align, save the recipe and the profile table.

Coronal-only inputs are not handled yet (they would be added as a chord-weighted 2D path in the same engine).

## Choices and why

- **Local level, not a global threshold.** Cupping makes one sediment read differently at the rim and centre, so a
  global threshold calls the rim "clast".
- **Modes, not means** for the level: a clast has to cover more than half of a disc (or of a ~12 cm stretch for the
  shape) to move it.
- **Void by an absolute floor.** Void is "air-like", not "lower than matrix", so it cannot overlap the low-density
  class.
- **Provenance.** The level and class steps are assembled from standard pieces (mode, median filter, MAD noise,
  hysteresis threshold, component size filter). They are not a published method and have only been validated on
  synthetic cores and by eye on real ones.

## Not yet decided or verified

- Which class cut-offs hold across scanners and cores (they are parameters in the recipe).
- Whether the matrix-only profile or a plane/band profile best tracks MSCL: this is what the comparison is for.
- Units: the engine takes any attenuation-like unit; `zero_offset` (0, or 1000 for HU) says where "zero attenuation" is.
