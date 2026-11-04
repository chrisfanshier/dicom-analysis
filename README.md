# dicom_stitch

Tools for sediment-core CT DICOM:

1. **dicom-multi-view** (preferred on branch `dicom-multi-view`) — load multiple series folders as core-section views with cm depth scale bars.
2. **core-stitch** — legacy Coron TOP/BTM stitch GUI (overlap join, Analyze ROI, Otsu, export). Not wired into multi-view.

## Install

```bash
cd dicom_stitch
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run multi-view (MVP)

```bash
dicom-multi-view
dicom-multi-view "D:\path\SE1" "D:\path\SE2"
python -m dicom_multi_view
```

### What it does

- **Add folder…** — browse for a DICOM series directory. On each add, enter **core name**, **section**, and optional **top depth (cm)** (blank/0 = start at 0; a number pins a manual top).
- Auto-detect **compressed/uncompressed** (Transfer Syntax) and **coronal MPR** vs **axial → coronal** reformat (detect uses the loaded primary series, not a stray scout).
- **Cut / frame slider** (selected section):
  - Coronal MPR: shows stack frames as-is (vertical core section); never re-reformats.
  - Axial brick: coronal reformat with section top at image top; slider moves the cut through the diameter (Y).
- **Shared physical scale:** all loaded sections use the same mm/px so an 80 cm section is shorter than a 150 cm one (not fatter). Scale bars: **1 cm minor ticks**, labels every **10 cm**.
- **Greyscale window (HU):** histogram + black/white sliders; apply to **this section** or **all sections**.
- **Depth (cm):** continuous mode sets tops from the sum of image lengths in list order (pinned manual tops kept); per-section mode uses each section’s own top (often 0). Edit top / pin on the left rail.
- **Layout:** side-by-side or stacked. **Reorder** with ↑/↓.
- **View:** shared wheel-zoom (toward cursor), drag-pan, Fit width / Reset. Scale bar zooms with the image so cm ticks stay physically true.
- Compressed series decode on read via pydicom/pylibjpeg (can be slow). Bulk decompress: `scripts/decompress_dicom_tree.py`.

### Out of scope (v1)

TOP/BTM stitching, sediment ROI Analyze, Otsu filters, downcore HU profiling, PNG/PDF export.

## Run stitch (legacy)

```bash
core-stitch
core-stitch "D:\path\SE1" "D:\path\SE2"
```

See [docs/stitch-gui-memo.md](docs/stitch-gui-memo.md).
