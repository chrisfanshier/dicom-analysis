# core-stitch

Geometry-first stitch GUI for Toshiba Aquilion **Coron TOP / Coron BTM** MPR DICOM folders.

Does **not** trust DICOM patient/FoR coordinates for coarse alignment. Masks core+liner vs air, aligns by geometry, then 1-DoF overlap along the core axis with Flip / Roll controls.

The separate `dicom_stuff` JPEG viewer is unrelated — keep using that for single-series mid-slice JPEGs.

## Install

```bash
cd dicom_stitch
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run

```bash
core-stitch
core-stitch "D:\path\SE1" "D:\path\SE2"
```

## Flow

Left rail tabs **Align** / **Display** / **Analyze** / **Export**; image fills the right pane.

1. **Align** — open TOP/BTM, coarse align, Flip/Roll/nudge, Refine join  
2. **Display** — alignment colors, slice browser, crop, zoom, overlap fuse, HU histogram window  
3. **Analyze** (optional) — auto sediment ROI, void/clast HU filter (Otsu), downcore mean HU plot, optional 3D mask  
4. **Export** — stitch products and/or sediment mask / ROI image (Analyze not required for stitch export)  


**B nudge:** X/Y move BTM; **Widen** only expands slider travel. Pan: middle-drag or Alt+drag. Arrows = 1 mm (Shift=5). Slice: `[` `]` or PageUp/Down.

See [docs/stitch-gui-memo.md](docs/stitch-gui-memo.md).

## Worklist tab (catalog-driven)

The **Worklist** tab reads the y_drive_tools release catalog (read-only) and lists
scan groups as cruise -> core -> section. Release folder: `$CORE_STITCH_RELEASE`, else
`release_root` from y_drive_tools `config/release.toml` (or use **Release…**).

- Each section shows its suggested **jobs**: every TOP x BTM pair of the same plane
  (coronal and sagittal pairs are separate jobs), plus single scans that need no
  stitching. Nothing is pre-selected: judge by description, plane, size and spacing,
  or pick any 1 or 2 series by hand (**Load picked series**).
- A pair loads into the Align tab (series chosen by UID, so TOP and BTM in one folder
  work). A scan with only one view opens in the same window (no second window) with a note
  that only 1 view was found: set the window, place the core ends, then Export
  (JPEG + 8-bit/16-bit HU TIFF + JSON, marked `unstitched`).
- Export writes to `<release>/ct_stitched/<cruise>/<core>/<section>/<plane>_<kind>/vNNN/`
  (a re-export of the same job is a new version).
- Progress (todo / loaded / exported / skipped / problem, notes, every export) lives in
  `<release>/catalog/stitch_progress.sqlite`, outside the rebuilt catalog, and a
  per-group roll-up is merged into y_drive_tools `config/ct_stitch_status.csv`.
- **Core ends and length check:** the preview shows draggable cyan (top) and amber (bottom)
  guides on the preview. Drag a guide to move it (it can't cross the other one), drag
  anywhere else to pan, wheel to zoom at the pointer, or type exact insets on the Export tab.
  They are never drawn into the images; their
  pixel endpoints, insets and core length go in the JSON sidecar as `core_scale`. After each
  export the measured length is compared with the master-list section length: an alert shows
  the difference (information only; no threshold is applied yet), and the numbers are kept
  in the worklist (`length vs list` column, "length compared" filter) and in
  `ct_stitch_status.csv`, so a cut-off can be chosen later.
  Field-by-field guide for users: [docs/sidecar-fields.md](docs/sidecar-fields.md).

Tests: `python -m unittest discover -s tests`.

## Scanner-geometry mode (experimental, SimpleITK)

On this branch a worklist pair first appears **where the scanner put it** (the DICOM header geometry), in the
Align tab. A banner says whether the scanner placement agrees with the images. If it does, check the join and
export. If not, shift the bottom half by hand (arrow keys or the three boxes), or press **Re-align from image**,
which runs the older core-mask + profile alignment and starts from its answer.

**Refine join** fine-tunes the bottom half's position by SimpleITK registration (a translation along and across the
core that maximises the correlation of the two halves over the overlap). It is applied only if it improves the
join, and the Align tab says what it costs in resampling for the loaded pair: nothing extra when the bottom/top half is
already being resampled to the finer pixel size, a rounding to the nearest pixel when both halves share a pixel size
and whole-pixel placement is on, and one interpolation when it is off.

The half with the finer pixels is kept exactly as scanned; the other half is resampled once onto its grid.
Tick or untick **Whole-pixel placement** to choose between keeping every voxel original (when both halves share a
pixel size) and sub-pixel placement. The older mask-based workflow is still there ("Use scanner geometry" box).
The sidecar for these exports is described in [docs/sidecar-fields.md](docs/sidecar-fields.md).

