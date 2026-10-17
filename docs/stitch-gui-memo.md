# Memo: Geometry-first CT core stitch — Coron TOP/BTM MPR GUI (new project)

Paste this into a local coding agent starting a **new** sediment-core CT stitching app.

## Scope (read first)

| In scope | Out of scope |
|----------|----------------|
| Toshiba Aquilion **Coron TOP / Coron BTM** DICOM MPR series folders | Full helical/circular recon volumes as the primary path |
| Two overlapping halves of the **same** core → stitch → QC → export | TIFF-stack / SedCT-style datasets (separate pipeline; **no stitching**) |
| Geometry-from-mask coarse align + flip + axial overlap + optional fine tune | Relying on DICOM patient/FoR coords for coarse place |
| Core+liner vs air mask | PVC liner strip / sediment-only mask (later stage) |

**Do not** fold this into the existing mid-slice JPEG / drag-crop viewer unless explicitly asked. That app stays as-is for single-series JPEG export. This is a **new project**.

## Goal

Build a GUI that takes two Coron MPR series (TOP + BTM), produces an **initial automatic stitch** without trusting DICOM world coordinates, then lets the operator adjust **Flip** (and optionally **Roll**) before optional fine registration and accept/export.

Do **not** strip PVC before stitching; treat core+liner as one object vs air.

## What these inputs look like

Typical Aquilion Coron half folders:

- Short stack of coronal MPR frames (e.g. ~60 slices), large in-plane matrix
- `SeriesDescription` / comments like `Coron TOP` / `Coron BTM` (spellings vary)
- Same study / often same `FrameOfReferenceUID`, but **IPP/FOV boxes can be far apart or inconsistent** — useless for coarse 2D join
- TOP and BTM often differ in **Rows/Columns** and **PixelSpacing** → resample to a common mm grid before correlating
- Core appears as a long strip in-plane; stack axis is across the core (thin), not downcore

So: treat each series as a **2.5D MPR slab**, not a full isotropic CT brick. Do not over-build 6-DoF volume registration for v1.

## Pipeline to implement

```
DICOM series A (TOP), B (BTM)
  → load HU stacks (RescaleSlope/Intercept; compressed + uncompressed TS)
  → pick homogeneous matrix group if folder mixes sizes
  → resample both to common in-plane mm spacing
  → Otsu (core/liner vs air) per volume/slab
  → keep largest connected component → core mask
  → geometry: centroid, long-axis direction (in-plane; PCA on mask)
  → geometric align (translate centroids laterally; align long axes)
  → [operator] Flip  [optional] Roll
  → 1-DoF overlap search along core long axis (masked, intensity-normalized)
  → optional bounded intensity fine registration
  → QC preview → accept → export stitched product + sidecar metadata
```

### Step details

1. **Import** two DICOM series folders → HU arrays. Handle compressed + uncompressed transfer syntaxes. If a folder mixes matrix sizes (e.g. 512² + 1624²), keep the largest consistent series/shape group and warn.

2. **Resample** both halves to a shared in-plane mm grid (finer of the two spacings is fine) so overlap search is in physical units.

3. **Segment** each with Otsu → core/liner vs air. Keep the **largest connected component**. Flag if a second component is nearly as large.

4. **Geometry** from each mask: centroid, principal/long axis in the coronal plane, extent along that axis. (Full 3D inertia PCA is optional; in-plane is enough for these MPRs.)

5. **Coarse geometric alignment** (mobile → fixed):
   - Translate so centroids align **laterally** (perpendicular to core axis).
   - Rotate so long axes align in-plane.
   - Reduces the problem to **overlap along the core axis** plus discrete **Flip** (and rare **Roll**).

6. **Operator controls (required before fine tune)**  
   After geometric alignment, before/beside overlap search:
   - **Flip** — reverse polarity along the core axis (180° end-for-end). **Expected often**; PCA does not know which end is up. One-click toggle; **re-run overlap search** on toggle.
   - **Roll** — rotate about the core axis. **Secondary for v1** on round whole-core coronals (default 0). Expose a slider or ±N° for half-cores / feature match; debounce or button to re-run overlap. Thin MPR slabs make roll underconstrained — do not block v1 on perfect roll.

7. **Overlap search** along the aligned long axis only:
   - Cross-correlation of masked axial (along-core) intensity profiles / slice statistics, and/or constrained search on downsampled overlap patches.
   - Normalize intensities before correlating (recon/HU differences).
   - Search on **masked** voxels only; report offsets in **mm**.
   - If best/second peak ratio is weak, mark **low confidence** and force manual QA.

8. **Optional fine registration** only after geometry + flip/roll + overlap are sensible. Keep it **bounded** (small along-axis shift ± tiny roll) so it cannot undo coarse geometry on HU mismatch.

9. **QC** before export: fused mid-stack (and a few Z indices), overlap band highlight, peak score + peak ratio, flip/roll readout. Operator accept/reject.

10. **Export** on accept: stitched image or volume product as designed for the app, plus a **sidecar JSON** (series UIDs, paths, flip, roll, overlap mm, peak scores, spacing, timestamp) for batch reproducibility.

## Product rules

| Rule | Why |
|------|-----|
| New project; leave JPEG-only viewer alone | Separate job: stitch vs quick mid-slice JPEG |
| Coron TOP/BTM MPR folders only | TIFF stacks / other CT products use other pipelines |
| Geometry before intensity registration | FoR/IPP often place FOV boxes wrongly for 2D join |
| Mask = core+liner vs air | Liner HU can overlap sediment; stitch needs the cylinder |
| Flip is first-class | Axis polarity is ambiguous |
| Roll exists but is secondary on round coronals | Thin MPR; rarely needed for whole cores |
| Fail loudly on ambiguous overlap | Wrong auto-stitch is worse than manual at scale |
| Stitch before PVC strip | Same physical object including liner |
| Log deterministic params | Batch reproducibility |

## Half vs whole cores

- **Whole** round coronals: default roll = 0 after axis align; user may nudge.
- **Half** cores: if a flat face is detectable, use it to initialize/constrain roll; if not found, warn and fall back to whole-core behavior.

## Suggested UI flow

1. Select fixed (TOP) + mobile (BTM) series folders → **Run coarse align**  
2. Fused / multi-Z QC at current pose  
3. **Flip** toggle · **Roll** control (secondary)  
4. Run / re-run **overlap search** (auto on Flip if cheap)  
5. Optional **fine register**  
6. **Accept** → export + sidecar  

## Implementation notes (Python)

Reasonable stack: `pydicom`, `numpy`, `scipy`, `scikit-image`, `Pillow`/`tifffile` for export; PCA via `sklearn` or `np.linalg` on mask coords. Optional: `pylibjpeg` plugins for compressed DICOM.

Keep coarse align + overlap search **deterministic and logged**.

Dragonfly remains reference visualization only — stitch logic lives in this app.

## Out of scope for this memo / v1

- PVC liner removal / sediment-only mask  
- Downcore HU profile modes / SedCT plateau-trim  
- TIFF-stack datasets  
- Primary support for full helical volume stitching (different product)  
- Changing the existing JPEG coronal GUI  

## Acceptance check

- [ ] Two Coron MPR series coarse-align **without** trusting DICOM world coords  
- [ ] Different matrix/spacing halves resample to a common mm grid  
- [ ] Flip toggle changes polarity and updates preview / overlap  
- [ ] Roll control available (default 0 for whole cores) before fine tune  
- [ ] Overlap search is axis-constrained; ambiguous peaks surfaced to user  
- [ ] QC + accept gate before export  
- [ ] Sidecar metadata written on accept  
