# How the core stitches are made

This document has two parts. **Part 1** is a plain-language summary for anyone who uses the stitched
cores. **Part 2** is the complete technical description of the processing code (everything except the
window and button code), written so that someone with technical questions can follow every step, see which
software package does it, and see why each number was chosen. The meaning of each value in the JSON sidecar is in
[sidecar-fields.md](sidecar-fields.md).

---

# Part 1 - Plain-language summary

**What the data are.** Each core section is scanned in a medical CT scanner (a Toshiba Aquilion). Because a
section is longer than the scanner can image at once, it is scanned in two halves, a TOP half and a BOTTOM
half, one straight after the other. The core does not move between them; the scanner only pauses to cool.
The scanner then produces, for each half, a stack of flat "coronal" slices: images cut lengthwise through
the core, 2 mm apart, each stored as a standard DICOM file. Every pixel in these images is a CT number in
Hounsfield units (HU), a measure of how strongly that spot absorbs X-rays (air about -1000, water 0, dense
sediment or shell above 1000).

**What a stitch does.** It joins the two halves into one image of the whole section. The two halves
overlap by a few centimetres, so the join is made where the same material appears in both.

**Where the join is decided.** Every DICOM file records exactly where the scanner put its image in space.
So the first step is simply to put each half where the scanner says it is. Where the two halves land in that
placement gives the overlap. This needs no searching, and on checking it agrees with an image-based
alignment to within about a millimetre on the cores we compared it against. For a few cores (about 4 in 10 in
our survey of the 2026-11 release) the scanner's positions are not reliable; the program checks this when a
pair is loaded and warns, and an image-based alignment (finding the core in each image and matching their
brightness down the core) is available as a fall-back.

**What is done to the pixels.** Everything that can be left as scanned is left as scanned:

1. The half with the **finer pixels** is the reference and is copied into the result **unchanged**.
2. The other half has coarser pixels. To put both on one pixel grid it is **resampled once**, using linear
   interpolation, directly from its original pixels. Interpolation does not add detail; it only re-expresses
   the same information on finer pixels.
3. In the overlap the two halves are **averaged**, except at the very edge of each half's image (a few pixels),
   where only the half that has real data is used. No brightness (HU) adjustment is applied to either half.
4. Everything that is not core (air around the core) is set to real air.

**How good is the join?** It is measured, not assumed. On every export the program compares the two halves
in the overlap, voxel by voxel, in all 3 dimensions (a correlation, where 1 means identical). It also tries
nearby positions to see whether any would fit clearly better. The numbers are stored in the sidecar so a user
can see how well each core was joined. The operator also looks at every join (overlay colours and a
cross-section) before exporting.

**Optional fine-tuning.** A "refine join" step can slide the bottom half by a small fraction of a pixel to
maximise the match. On real cores the scanner position was already so good that this moved the half by only
0.01-0.13 mm.

**What you get.** A viewable JPEG and 8-bit TIFF, a full-precision TIFF whose pixel values are HU, a DICOM
stack of the stitched volume, and the JSON sidecar that records where the join is, how it was made, how well
the halves agree, and where the core ends in the picture (so a ruler can be drawn).

**What to keep in mind.** The finer-pixel half is exactly as scanned; the other half is smoothed to a finer grid,
which does not add information. Real resolution is set by the scanner, not by the output pixel size. The stitch is a
translation only: if the two halves had been rotated relative to each other (they should not have been), the
image-based fall-back or a manual check is needed.
If a section's two series are labelled backwards, a button swaps them (and swaps back) and the swap is recorded. A core that
leans slightly in the picture can optionally be rotated exactly upright; that interpolates the pixels, so it is off by
default, and the measured lean is recorded either way.

---

# Part 2 - Technical description

## 0. Overview of the pipeline

```
release catalog  ->  pick a TOP/BOTTOM series pair  ->  load DICOM -> HU volumes (pydicom)
   -> build physical-space images (SimpleITK)  -> segment the core (scikit-image)
   -> header check (profiles, NCC)  -> plan the stitch (reference, shift, output grid)
   -> render: resample once, fuse (SimpleITK + NumPy/SciPy)
   -> [optional] refine join (SimpleITK registration)   -> join-quality check (NumPy)
   -> export: JPEG / 8-bit TIFF (Pillow), HU TIFF (tifffile), DICOM (pydicom), JSON sidecar
```

Code lives in `src/core_stitch/`. The files that carry the method are:

| File | Role |
|---|---|
| `load.py` | Reads DICOM series into HU volumes. |
| `segment.py` | Otsu threshold + largest connected component to find the core. |
| `itk_stitch.py` | Physical-space model, placement planning, resampling, fusing, refine, join quality. |
| `header_check.py` | "Do the scanner positions agree with the images?" |
| `header_state.py` | Ties the above together for one pair; also writes the DICOM and sidecar. |
| `hu_tiff.py` | Writes the signed 16-bit HU TIFF. |
| `scale.py` | The `core_scale` block (pixel/centimetre conversion, length check). |
| `pipeline.py`, `geometry.py`, `overlap.py` | The image-based alignment (used only as an optional fall-back) and the single-scan preparation. |
| `catalog.py`, `progress.py`, `worklist.py` | Reading the release catalog and recording what has been exported. |

## 1. Software and why each package is used

| Package | Used for | Notes |
|---|---|---|
| **pydicom** | Reading DICOM files (headers and pixel data); writing the stitched DICOM stack. | Pixel data are decoded to NumPy arrays. |
| **pylibjpeg** (with `-libjpeg`, `-openjpeg`, `-rle`) | Decoding compressed DICOM pixel data. | The scanner stores the series as JPEG 2000 lossless (transfer syntax 1.2.840.10008.1.2.4.90); decoding is lossless, so the values are exactly what the scanner wrote. |
| **NumPy** | All array arithmetic (HU volumes, masks, correlations). | Volumes are `float32`, arranged (slice, row, column). |
| **SimpleITK** (ITK) | Physical-space images, resampling, translation registration. | The standard medical-imaging toolkit; version is recorded in the sidecar. Chosen because resampling directly from original voxels into a physical-space grid is exactly its job, and because it is widely used and cited. |
| **scikit-image** | Otsu threshold (`threshold_otsu`), connected components (`label`, `regionprops`); `resize` in the image-based fall-back and single scans. | |
| **SciPy** | `binary_erosion` (edge trimming); rotation/shift for the image-based fall-back and single scans. | |
| **Pillow** | JPEG and 8-bit TIFF writing. | JPEG quality 95, optimised Huffman tables; TIFF LZW (lossless). |
| **tifffile** | The signed 16-bit HU TIFF. | Pillow's TIFF writer is not reliable for signed 16-bit; tifffile writes it directly, with deflate (zlib) compression. |
| **sqlite3** (standard library) | Read-only access to the release catalog; a separate progress file. | The catalog is never written. |

(The graphical window uses Tk and is not described here.)

## 2. Input data and the release catalog

The inputs are DICOM series in a read-only snapshot of the 2026-11 release. A SQLite catalog
(`catalog.sqlite`) lists, for every group of scans, the cruise, core, section, and which series are the TOP
and BOTTOM halves (it stores only the last 12 characters of each Series Instance UID, e.g. `61922.877838`;
the loader matches by suffix). The program opens the catalog read-only and writes only to the export folder
(`ct_stitched\<cruise>\<core>\<section>\...\v001`, with a new version folder on each re-export) and to a separate progress
file.

Typical scan geometry of a pair (not a fixed rule; every number is read from the headers): coronal slices
2 mm apart, in-plane pixels 0.3-0.7 mm. The two halves of a pair are reformatted by the scanner with
**their own field of view**, so their pixel sizes usually differ (for example 0.388 mm and 0.616 mm).

## 3. Loading (`load.py`)

1. **Find files with pixels** in the series folder (`pydicom.dcmread`, with `force=True` for files that lack
   a standard preamble). If a folder holds more than one series (both halves can share a folder), the files
   are filtered by Series UID (suffix match). If more than one image-shape group remains, the largest group
   is used and a warning is raised.
2. **Sort slices** by the position along the slice normal: the dot product of `ImagePositionPatient` with the
   normal `row_direction x column_direction` (from `ImageOrientationPatient`). If positions are missing, by
   `InstanceNumber`.
3. **Convert to HU**: `pixel_array` as `float32`, then `value * RescaleSlope + RescaleIntercept` (only applied
   when not the identity). On the scanner's data slope = 1 and intercept = 0, so the stored values are
   already HU.
4. **Spacing**: row and column pitch from `PixelSpacing`; slice step is the **median of the absolute
   differences** of the slice positions along the normal (robust to one odd slice). Falls back to
   `SliceThickness`.
5. Stack into a (slice, row, column) `float32` array. No smoothing, no clipping.

## 4. Where the data sit in space: the SimpleITK model (`itk_stitch.load_half`)

Each half becomes a 3-D `SimpleITK.Image` in **physical space**, using three things from the DICOM headers:

* **Spacing** (column pitch, row pitch, slice step), in mm.
* **Origin** = `ImagePositionPatient` of the first sorted slice (the centre of its first voxel), in mm.
* **Direction** = the 3x3 matrix whose columns are: the row direction (direction in which the *column index*
  increases), the column direction (direction in which the *row index* increases, which is **down the core**), and
  the slice normal = row x column.

NumPy axes (slice, row, column) map to ITK index (k, j, i). DICOM patient coordinates and ITK physical
coordinates are both LPS (x towards patient left, y towards posterior, z towards head), so no coordinate
conversion is needed.

Two halves from the same scan session share the scanner's patient coordinate system, so once each is a
physical-space image they are **already where the scanner put them**: no search is needed to place them.

The header's orientation vectors are kept with each half (`row_dir`, `col_dir`, `normal`). For the usual
coronal pair the column direction points down the core. The program works in terms of three axes of the
reference half: **along** the core (the column direction, `col_dir`), **across** the core (`row_dir`) and
**through the stack** (`normal`).

## 5. Finding the core in an image (`segment.py`)

Used to decide where the core is (for the crop, the header check, and which pixels to keep).

1. Take the middle slice of the half.
2. **Otsu threshold** (`skimage.filters.threshold_otsu`) over all finite pixels. This separates the bright
   material (core, liner, end caps) from the air. Values at or above the threshold are "core".
3. If that mask covers more than half the image, the opposite side may have been chosen: the inverse is used
   when it is smaller.
4. **Largest connected component** with 8-connectivity (`skimage.measure.label(..., connectivity=2)`); everything else
   (noise, a second object) is dropped. A flag records whether a second component was at least 60% as large.
5. The Otsu threshold is stored per half and reused to mask *every* slice of the output (section 8).

Air level: `air_value` = the **5th percentile** of the half's middle slice (about -2048 on these scans). It
is the fill value outside the core and the "default pixel" for resampling.

This is a "core and liner versus air" mask; it is not a measurement of the sediment alone.

## 6. The header check (`header_check.py`)

When a pair is loaded, one middle slice of each half is used to ask whether the scanner's placement agrees
with what is in the images. It uses only the middle slice, so it takes well under a second.

* **Across**: the core centre column (mean column of the core mask) in each half, converted to a physical point on
  that half's first slice plane, then the difference of the two points along the top half's row direction. Doubtful if the two differ by more than **5 mm** (`ACROSS_TOL_MM`).
* **Depth**: the difference of those two points along the top half's normal (measured at the core, so a plane
  tilt does not look like a depth offset). Doubtful if more than **5 mm**
  (`DEPTH_TOL_MM`).
* **Along**: for each half, the mean HU inside the core mask at each row gives a 1-D brightness **profile** down
  the core, positioned in mm using the header. Both profiles are put on a 1 mm grid. In the overlap the
  headers imply (at least **10 mm**, `MIN_OVERLAP_MM`), the **normalised cross-correlation (NCC)** of the two
  profiles is computed at the header placement
  (this is `profile_correlation_at_header`). It is also computed with the second profile slid by every whole
  millimetre from -150 to +150 mm. If another position fits better by more than 0.1 in NCC and lies more
  than 10 mm away, the header placement is flagged.
* **Orientation**: if the two halves' direction vectors differ (beyond 1e-3) this is reported as a note, not as a
  reason to doubt the placement.
* Verdict: `headers_good` or `needs_image_stitch`, plus the reasons.

NCC here is the Pearson correlation: both series are mean-subtracted and divided by their standard
deviation, then averaged product. It is insensitive to an overall offset or scale in brightness. It needs at
least 15 valid samples.

Run on 1,485 plane-pairs of the release, about 59% were clean; failures depended on the cruise (some
cruises had poor header placement). Cores with mostly empty space or many gaps also give poor profile matches
even when the placement is right, because the profile has little to compare.

## 7. Planning the stitch (`plan_stitch`)

No pixels are resampled in this step; it only decides how.

0. **Labels / order.** If the operator uses the "Swap TOP and BOTTOM" button, the two series are exchanged: the series the
   catalog calls TOP becomes the bottom half and vice versa, each staying upright (nothing is flipped). The headers
   cannot place the pair in that order, so the new bottom half is placed below the new top half by the image-based
   alignment (section 11) with the order forced: the overlap search is restricted to positions where the second half
   starts below the first (`require_below`), instead of being free to choose either order. If the images then match
   poorly (join correlation below 0.8) the program warns that the series may not really be labelled backwards. Swapping
   back returns to the scanner (header) placement. The swap is recorded as `top_bottom_swapped`, and the catalog's own
   labels stay in `series.a`/`series.b`.
0b. **One half upside down.** If one half was scanned upside down relative to the other, the operator uses "Turn TOP half
   180 degrees" or "Turn BOTTOM half 180 degrees". That half's slices are turned 180 degrees about their own centre (rows and
   columns reversed; lossless; the header geometry is left as it was, so the content is turned in place), and the pair is
   placed by the image-based alignment (section 11), which is given the turned slices. It is recorded per half as
   `turned_180`. This is different from "Rotate 180 degrees", which turns the whole finished picture (section 12).
1. **Orientation.** Usually the two halves' row and column directions are identical. Sometimes the scanner cuts the
   two reformats a few degrees apart (about 5 degrees about the core's long axis on some DF80 sections; the angle is
   recorded as `header_check.orientation_difference_deg`). That is accepted: the headers give the true position and
   direction of every voxel, so the grid of the reference half is used and the other half is resampled into it, which
   includes interpolating between that half's neighbouring slices. On the DF80-31PC sections with a 5 degree
   difference the join correlation was 0.993 to 0.996 with a best shift of 0 on every axis, as good as untilted
   pairs. Planes more than **30 degrees** apart (`ORIENT_MAX_DEG`; for example coronal against sagittal) are not
   the same view of the core and are refused.
2. **Reference choice**: the half with the finer in-plane pixel pitch (the smaller of its column and row
   pitch). A tie goes to the top half. The other half is the "moving" half.
3. **The shift** `move_mm = (along, across, depth)` is the bottom half's offset from its header
   position, in mm. It is converted to a physical vector with the reference half's axes:
   `move = along * col_dir + across * row_dir + depth * normal`, sign-flipped if the bottom half is the
   reference (so the shift is always the bottom half's).
4. **Whole-pixel placement** (only if both halves share the same pixel grid, i.e. the same spacing and
   direction to 1e-3 mm): the bottom half's origin, after the shift, is rounded to the nearest whole voxel of
   the reference grid, so the moving half's voxels fall exactly on the reference lattice and can be copied with
   no interpolation. The distance moved is recorded (`snap_residual_mm`, at most half a voxel in each axis).
   With `subpixel`, or when the pixel sizes differ, no snap is made.
5. **Output grid**: the reference lattice, extended by whole voxels in every direction until it covers
   all 8 corners of both halves. A tolerance of 1e-3 voxel stops header rounding (about 1e-5 mm) from adding an
   extra row or slice. The output therefore has the reference half's pixel size and slice step; slice planes
   of the reference coincide with output slices.
6. **Width crop**: the output is cropped across the core to the core's width plus **10 mm**
   (`CROP_PAD_MM`), so it is not mostly air. Every row and slice is kept.

## 8. Rendering: resampling once and fusing (`render_plan`)

1. **Reference half** is placed with `SimpleITK.Resample` using nearest-neighbour interpolation and an
   identity transform. Because the output grid is the reference lattice, this is an exact copy: the
   reference voxels are bit-identical to the scan.
2. **Moving half** is resampled **once** with `Resample`: for each output voxel, the filter looks up the
   corresponding position in the original moving image (the transform is a translation of minus the shift), and
   interpolates the original voxels there. The interpolator is **linear** (trilinear: 8 neighbouring original voxels), or
   nearest-neighbour when the halves share a lattice and the shift was snapped. Pixels outside the moving
   image get the air value. The pixel-size change, the header placement, and any shift (including a fraction
   of a pixel) are all done in this **one** interpolation; no intermediate resampled copy is made.
3. **Footprint** of each half: a volume of ones resampled the same way (nearest neighbour) marks where
   that half has real data.
4. **Core mask** of each half, slice by slice: pixels at or above that half's Otsu threshold, largest
   8-connected component, restricted to the footprint. (A rule that flips the polarity when more than 55% of a
   slice is bright was removed from this path: with the output cropped to the core's width, a wide core can
   fill most of the picture, and the flip then selected the air.)
5. **Fuse**:
   * Pixels in only one half's core take that half's value.
   * Pixels in both cores and **at least 3 rows** (`trim_rows`) inside both halves' footprints: the **mean**,
     `0.5 * (reference + moving)`.
   * Pixels in both cores but within 3 rows of one half's image edge: only the other half is used. (Footprint
     edges are found with a `binary_erosion` along rows with a 7-row element.) Within 3 rows of both edges
     the reference is used.
   * Everything outside both cores is set to the reference half's air value.
   * **No HU matching or scaling is applied** to either half; the two share the scanner's own calibration.
   The mean treats the two halves equally; the edge rows are trimmed because a half's last row can be a
   partial-pixel average (a missing band at the image edge) that would otherwise pull the mean down.
6. The result is a `float32` volume in HU on the output grid, plus the separate reference and moving
   renderings, footprints and masks (used for checks and the alignment overlay).

Fraction of data changed: every voxel of the reference half is unchanged. The moving half's voxels are
interpolated unless it was snapped. In the overlap the output is the average of both.

## 9. Optional: Refine join (`refine_translation`)

A small intensity-based registration that slides the bottom half by up to a few millimetres to maximise
agreement in the overlap. It is optional and never changes depth. It is **SimpleITK's standard
`ImageRegistrationMethod`**, set up as follows (these particular settings were chosen by me and tuned on test
data; the method itself is the textbook rigid-registration recipe).

* **Orientation**: each half is expressed in the reference half's frame together with its own orientation relative to
  it (identity for the reference, a small rotation for a half cut in a plane a few degrees away), so the registration
  sees exactly the placement the stitch uses. (An earlier version ignored that rotation for tilted pairs: Refine then
  scored a good join near zero and could never find an improvement, while the join-quality check, which works on the
  rendered volume, found the better position. They now agree; this is covered by a regression test.)
* **Data**: a slab of **24 slices** around the middle of the stack, rendered with the current placement. The
  fixed image is the reference half, the moving image the other half (re-expressed in the reference's own
  frame, so a shift found is directly (across, along, depth) in mm, whatever the scan's tilt).
* **Region**: only rows where both halves have core data, plus **20 rows** of margin, with a mask eroded by
  **4 rows** from the ends (so the half edges do not dominate).
* **Transform**: translation, 3 parameters. **Depth is held at zero**
  (`SetOptimizerWeights([1, 1, 0])`): only across and along move.
* **Metric**: `SetMetricAsCorrelation` (normalised cross-correlation; insensitive to a brightness
  difference between halves), evaluated at **30%** of the pixels chosen at regular spacing (`REGULAR`).
* **Interpolator**: linear.
* **Optimiser**: `RegularStepGradientDescent`: learning rate 2.0, minimum step 1e-3, 120 iterations, step relaxation
  0.5 (step halves when the direction reverses), scales from the physical shift of the parameters.
* **Multi-resolution**: two levels: shrink by 2 with Gaussian smoothing (sigma 1 mm), then full resolution.
* **Scoring**: the registration's own metric is **not** reported. The join is scored explicitly, before and
  after, at full resolution, on the same pixels, restricted to pixels that lie inside the moving image at
  **both** positions (so air at an edge cannot make a shift look bad).
* **Acceptance**: applied only if the score improves by more than 1e-4 and the shift is no larger than
  **15 mm**. Otherwise nothing changes and the message says so.
* **Whole-pixel placement** (same-grid pairs only): the optimum is rounded to a whole pixel and re-scored; if
  rounding makes it no better (or the correlation falls by more than 0.002), the placement is left as it was
  and the user is told to switch to sub-pixel to use the fraction.
* The refine is folded into the placement shift, so the **final render still resamples the moving half once**.
  Undo restores the previous shift exactly (no resampling is involved).

Gradient descent follows the local slope of the correlation to the nearest good position; it can get stuck
in a local optimum, which is acceptable because the scanner position already starts it close.

## 10. The join-quality check (`join_quality`)

Run on every export. It does not move anything.

1. Take the overlap band of the rendered volume (all slices).
2. Use only voxels inside **both** halves' cores, and only the interior rows of the overlap (excluding
   `edge_rows` = 4 plus `max_px` = 3 rows at each end).
3. Compute the NCC of the reference and moving HU values over those voxels: `correlation`.
4. Recompute it with the moving half's voxels (and core mask) shifted by whole output voxels: -3...+3
   along the core, -3...+3 across the core, -3...+3 slices in depth, **one axis at a time**. The shift is a
   whole-voxel move of already-rendered data (edge voxels become invalid), not a new resampling.
5. **Flag** an axis when its best shift is at least 2 pixels (1 slice in depth) away from zero **and** it
   beats the current placement by more than a per-axis margin: depth 0.005, along 0.0001, across 0.001.

Why per-axis margins: sensitivity differs by axis. On real cores one slice of depth error cost about 0.05 in
correlation (the correlation fell to about 0.94 from 0.996), a pixel across the core about 0.005, and a pixel along the core only about
0.0002 (layering varies slowly down the core, so a slide along it changes little). A one-pixel best shift is
never flagged, because a placement that is correct to sub-pixel precision often has its best whole-pixel
shift at 0 or +-1. Validation by planting known errors in real pairs: 0.8 mm along or across and a one-slice depth
error were flagged; 0.4 mm was not; the scanner placement itself was not flagged.

The scan covers whole voxels and one axis at a time. It is a screening check, not a measurement of
accuracy; it does not test rotation.

## 11. Image-based alignment (optional fall-back)

Used only when the user asks (button), typically when the header check says the scanner placement is doubtful.
Implemented in `pipeline.coarse_align`, `geometry.py` and `overlap.py`:

1. Both halves are resampled in-plane to a common pixel size, the finer of the two (`skimage.transform.resize`,
   linear, anti-aliased).
2. Core mask per half (section 5); principal axis of the mask (eigenvector of its covariance) gives the core's
   tilt; each image is **rotated upright** so the core runs down the rows (linear interpolation).
3. The bottom half is shifted sideways so the two core centroids share a column.
4. A one-degree-of-freedom search slides the bottom half along the core to maximise the NCC of the two
   **masked row-profiles** (mean HU inside the core at each row), with a peak-to-second-best ratio as a
   confidence indicator.
5. (When used by the Swap button the search is restricted to positions where the second half starts below the first.)
6. The result is converted to a shift of the bottom half from its header position (along, across; depth 0)
   and handed to the same renderer, so the final export still resamples the moving half once from the original data.

Its resized, rotated and shifted copies are used only to *measure* the shift; they are not exported. The
measurement involves several interpolation steps, which is one reason the header placement is preferred where
the scanner position is trustworthy.

## 12. Exports (`header_state.export_header`, `hu_tiff.py`, `scale.py`)

Everything is rendered once from the final placement (every slice).

* **Crop box**: the bounding box of the core mask on the view slice plus **3 mm** padding (at least 2 px).
* **Greyscale window**: by default the 1st and 99th percentiles of the core HU values on the view slice; the user
  may change it. Greyscale = `clip((HU - lo) / (hi - lo) * 255, 0, 255)` as `uint8`.
* **JPEG**: window applied, quality 95, DPI tag from the pixel size (`25.4 / mm_per_px`).
* **8-bit TIFF**: the same pixels, LZW compression (lossless).
* **HU TIFF**: the cropped HU slice, rounded to the nearest whole HU, clipped to the int16 range, written as a
  **signed 16-bit** TIFF (deflate) by `tifffile`. Pixel value = HU.
* **DICOM stack**: the full stitched volume as CT slices, `int16`, `RescaleSlope = 1`, `RescaleIntercept = 0`
  (stored value = HU), `PixelRepresentation = 1` (signed). Orientation = the reference half's; position =
  the output grid's origin plus `k` slice steps along the normal; `PixelSpacing`, `SliceThickness` from the
  output grid. **Frame of Reference UID is reused from the source scan**, so the stack sits in the scanner's own patient
  space. Patient, study and orientation tags are copied from a template slice; a new Series UID is generated;
  `ImageType = DERIVED/SECONDARY/MPR/STITCH`; window centre/width are written from the window.
* **Rotate 180 degrees (optional, off by default)**: if the core is upside down (its top end at the bottom of the
  picture) every slice is turned 180 degrees in its own plane: rows and columns reversed, nothing else. It is lossless.
  The grid's row and column direction vectors are negated and its origin moved to the opposite corner, so the DICOM
  stack describes the same volume in the same place. It is done after any straightening and after the join-quality check;
  it is recorded in `rotate_180`.
* **Straighten (optional, off by default)**: the lean of the stitched core's long axis (principal axis of its mask on
  the view slice) is always measured and recorded. If the option is on, the finished stitched volume is
  resampled onto the same grid turned about the through-stack axis so the core runs exactly down the rows (linear
  interpolation for HU, nearest neighbour for the masks). This is a second resampling of the moving half and a first
  resampling of the reference half, so the reference voxels are no longer exactly as scanned; it is recorded in
  `straighten`. The DICOM orientation vectors follow the rotated grid, so every voxel keeps its true position in space.
  The join-quality numbers are always computed on the stitch before any rotation.
* **core_scale**: pixel-edge coordinates of the core ends, from the (auto or user-dragged) end guides, plus the
  length check against the master list (differences recorded; no pass/fail applied).
* **Sidecar**: every decision above (see `sidecar-fields.md`). No file paths.

## 12a. Single scans (nothing joined)

A section scanned as one piece is prepared by `single.py`:

1. Load the series (section 3) and take **one slice** (the middle one by default; the operator can change it).
2. Core mask on that slice (section 5).
3. In-plane resample to the finer of the scan's row/column pitch (a no-op when they are equal).
4. **Straighten (optional, off by default)**: the core's principal axis is found from the mask and its lean from the row
   direction is always measured and stored (`straighten.measured_tilt_deg`). Only if the option is on is the slice
   rotated so the core runs down the rows, with linear interpolation (`scipy.ndimage.rotate`, `reshape=True`,
   air-filled). Any non-zero angle interpolates every pixel (on a real single scan with a 0.1 degree lean, 42% of
   pixels changed value, by 34 HU on average inside the core), which is why it is off unless asked for.
5. **Rotate 180 degrees (optional, off by default)**: if the core is upside down the slice is turned 180 degrees (rows and
   columns reversed; lossless) after any straightening, and recorded in `rotate_180`.
6. Window, crop, JPEG, 8-bit TIFF, signed HU TIFF and sidecar as in section 12. There is no DICOM stack and no join.

## 13. Numerical choices at a glance

| Choice | Value | Why |
|---|---|---|
| Reference half | finer in-plane pixel (top wins a tie) | Keeps the sharper data exactly as scanned; only the coarser half is resampled. |
| Interpolation | linear (trilinear) | Standard, no overshoot, no extra detail claimed. B-spline is available in code but not used. |
| Resampling | once, from original voxels | No stacked interpolation; the shift and pixel-size change are one step. |
| Whole-pixel snap | same-grid pairs only; <= 0.5 voxel | Lets every voxel stay original when the grids already match. |
| Fuse | mean in overlap; 3 edge rows trimmed | Avoids partial-pixel edge rows; no HU matching. |
| Core mask | Otsu on middle slice + largest 8-connected component | Same recipe throughout; threshold per half. |
| Air fill | 5th percentile of the reference half's middle slice | Real air as the scanner records it (about -2048). |
| Output width | core width + 10 mm | Keeps the volume compact. |
| Header tolerances | across 5 mm, depth 5 mm, overlap >= 10 mm, orientation 1e-3 | See section 6. |
| Refine limit | 15 mm; depth fixed | A larger move means something else is wrong. |
| Join-quality search | +-3 voxels per axis, one at a time | Cheap; screening only. |
| Slice step | median of slice position differences (about 2 mm) | Robust to header rounding. |
| HU storage | int16 | Covers -32768..32767; values are rounded to whole HU. |

## 14. Limitations and things to know

* **Resolution.** The reference half is at scanner resolution. The other half is interpolated to the finer grid; it is
  smoother than a native finer scan would be. Slice thickness is 2 mm, much coarser than the in-plane pixels,
  so the through-stack direction is the weakest.
* **Translation only.** There is no rotation or scaling in the header method (apart from the optional straightening
  of the finished picture, which is a rotation about the through-stack axis for display convenience, not a correction of
  one half relative to the other). A roll between halves would show up
  as a mismatch toward the edges of the core (visible in the alignment overlay and the cross-section) and as a lower
  join correlation, but would not be corrected. Pairs cut in planes a few degrees apart are joined by their
  headers (see section 7); the interpolation between slices this needs is the weakest direction of the data,
  because slices are 2 mm thick.
* **Join quality is measured only where the halves overlap** and only through whole-voxel shifts; it does not
  prove absolute accuracy of the position.
* **Header trust.** About 4 in 10 pairs in the release have scanner positions that failed the check; some of
  those are cores with large gaps that make the check inconclusive.
* **Core mask.** It selects core plus liner, not only sediment. The end positions in `core_scale` can be placed
  by hand to mark the sediment.
* **HU values.** The overlap average is the only operation on values; no recalibration is done. Interpolation
  smooths sharp edges slightly (the moving half only).
* **Reproducibility.** The SimpleITK version is recorded; the series UIDs of both halves are recorded; the
  placement shifts and the refine result are recorded.

## 15. How it was tested

* **Synthetic phantom** (`tests/test_itk_stitch.py`): an analytic cylinder with known offsets (including
  half-pixel and header errors) scanned as two halves. Tests check that the reference half is bit-identical,
  that the stitch matches the truth within interpolation error, that a shift by `move_mm` lands where expected, that
  the fuse behaves as described, a bottom half cut in a plane turned 5 degrees about the core axis still matches the phantom's true values everywhere, and
  that planes 90 degrees apart are refused.
* **GUI/state tests** (`tests/test_header_gui.py`): one-slice rendering equals the same slice of the
  full render, shifts and reset, alignment colours, refine recovers planted 2.0 mm (along) and 1.5 mm (across) offsets
  and never worsens the score, undo restores exactly, join-quality flags planted depth and in-plane
  errors, exports record the quality block.
* **Real cores**: on pairs that had been stitched by hand, the header-placed result agreed to within about
  1 mm; refine moved the bottom half only 0.01-0.13 mm; join correlation 0.998-0.9997.
* **Survey** (`tools/survey_headers_fast.py`) of 1,485 plane-pairs of the release for header reliability.
* `tests/test_hu_tiff.py`: the HU TIFF round-trips as signed 16-bit with the pixel value equal to the HU.
