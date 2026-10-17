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
3. **Analyze** (optional) — auto sediment ROI (liner/void peel), erode, X/Z nudge, optional 3D mask  
4. **Export** — stitch products and/or sediment mask / ROI image (Analyze not required for stitch export)  


**B nudge:** X/Y move BTM; **Widen** only expands slider travel. Pan: middle-drag or Alt+drag. Arrows = 1 mm (Shift=5). Slice: `[` `]` or PageUp/Down.

See [docs/stitch-gui-memo.md](docs/stitch-gui-memo.md).
