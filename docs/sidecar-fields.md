# The JSON sidecar, field by field

Every export writes a small text file next to the images: `<name>.json`. It records how the pictures
were made, where the two halves were joined, how good the join is, and where the core sits in the image.
This page explains every value in it, in the order they appear in the file.

You do not need to read the JSON by hand. Any program that reads JSON (Python `json`, R `jsonlite`,
MATLAB `jsondecode`, a browser) can pick values out by name. No file paths are stored; files are named
without folders.

**This page describes the sidecar (version 0.2.0)** written with every stitched export. A single scan that
needed no stitching has a shorter sidecar; its differences are listed at the end under
[Single scans](#single-scans-nothing-joined).
For *how* the stitch is made, see [methodology.md](methodology.md).

A real example is used throughout: section `HLY0501-6JC-7`, a coronal pair, exported 2026-11-06.

## Files in one export folder

| File | What it is |
|---|---|
| `<name>.jpg` | Greyscale picture of the core, cropped to the core. For looking at. Clean: no scale bar or labels. |
| `<name>.tif` | The same picture as an 8-bit TIFF. Only the greyscale *window* (see `window_lo`/`window_hi`) survives; values outside it are clipped and the rest is in 256 steps. For looking at, not measuring. |
| `<name>_hu.tif` | The real CT numbers: a **signed 16-bit TIFF whose pixel value is the Hounsfield unit** (HU), rounded to whole HU, clipped to the range -32768..32767. No offset to add or subtract. |
| `dicom_mpr/` | The stitched volume as a DICOM CT stack (signed 16-bit, rescale slope 1, intercept 0, so stored value = HU). Same patient coordinate system as the scanner's own series. |
| `<name>.json` | This sidecar. |

The JPEG and TIFFs carry the pixel size in their resolution tags, so most image programs show the correct
physical size.

---

## The sidecar, in file order

### Identity of the method

| Field | Example | Meaning |
|---|---|---|
| `version` | `"0.2.0"` | Version of this file's layout. A program reading the file should check it. |
| `engine` | `"simpleitk_header_geometry"` | How the join was made: halves placed from the scanner's DICOM geometry, resampled with SimpleITK. |
| `software.simpleitk` | `"2.5.7"` | Version of SimpleITK that did the resampling. Resampling results can differ slightly between versions, so it is recorded. |
| `stitched` | `true` | `true` when two halves were joined. (For a single scan the field is `unstitched: true` instead.) |
| `top_bottom_swapped` | `false` | `true` if the operator used the **Swap TOP and BOTTOM** button because the two series were labelled backwards but each was the right way up. The half the catalog calls TOP is then the lower half and the other becomes the upper half: the two are re-ordered one for one (each stays upright), placed below/above each other by matching the images (the headers cannot place them in that order), so `placement.source` says `image alignment (mask + profile)`. The `top` and `bottom` blocks below describe the corrected roles, while `series.a` / `series.b` further down keep the catalog's original labels, so the swap can always be traced. |

### `top` and `bottom` – the two scans that were joined

(After any swap: see `top_bottom_swapped`.)

| Field | Example | Meaning |
|---|---|---|
| `series_uid` | `"1.2.392...877838"` | The DICOM Series Instance UID of that half. It identifies the scan unambiguously. |
| `series_description` | `"Bone 2.0 Coron TOP.1"` | The scanner's own description of the series (reconstruction kernel, plane, which half). |
| `native_mm_per_px` | `0.615547` | Pixel size of that half *as scanned*, in mm (the in-plane pixel spacing). Top and bottom often differ, because the scanner reformats each half with its own field of view. |
| `slice_step_mm` | `2.0000037` | Distance between that half's slices (through the stack), in mm. Here 2 mm coronal slices. |
| `turned_180` | `false` | `true` if this half was scanned upside down relative to the other and was turned 180 degrees in place before joining (the **Turn TOP half / Turn BOTTOM half 180 degrees** buttons). Lossless: every slice has its rows and columns reversed, nothing else. Its header position then no longer describes its content, so the pair is placed by matching the images (`placement.source` says `image alignment (mask + profile)`). Not to be confused with `rotate_180`, which turns the whole finished picture. |

### `resolution` – what was done to the pixels

| Field | Example | Meaning |
|---|---|---|
| `output_mm_per_px` | `0.387743` | Pixel size of the stitched picture. It is the **finer** of the two halves' pixel sizes. |
| `reference_half` | `"bottom"` | The half with the finer pixels. Its voxels are copied into the output **unchanged** (bit for bit). |
| `interpolated_half` | `"top"` | The other half, which was resampled **once** to the reference pixel size. `null` if both halves already had the same pixel size. |
| `upsample_factor_of_interpolated_half` | `1.5875` | How much finer the interpolated half's pixels were made (1.59 = pixels 1.59 times smaller). `null` when `interpolated_half` is `null`. |
| `interpolation` | `"linear"` | How the resampling was done: `linear` interpolation, or `nearest (voxels on reference lattice)` when the halves share a pixel grid and the shift was snapped to a whole pixel (every voxel keeps its original value). |
| `note` | | Reminder that interpolated pixels are smooth estimates; they carry no extra detail. |

### `placement` – where the bottom half sits

The two halves are first placed exactly where the scanner recorded them (the DICOM header position and
orientation). `placement` says whether anything was changed from that.

| Field | Example | Meaning |
|---|---|---|
| `source` | `"scanner header"` | Where the position came from. `scanner header` = as recorded. `image alignment (mask + profile)` = you used *Re-align from image*. `... + refine` is added if *Refine join* was used. |
| `bottom_shift_mm.along_core` | `0.0` | Shift applied to the bottom half **from that starting position**, along the core, in mm (+ is down the core). |
| `bottom_shift_mm.across_core` | `0.0` | The same, across the core (sideways in the picture), in mm. |
| `bottom_shift_mm.through_stack` | `0.0` | The same, through the stack (depth, perpendicular to the coronal slices), in mm. |
| `pixel_placement` | `"whole_pixel"` | The choice made in the window. `whole_pixel`: when both halves have the same pixel grid, the bottom half is snapped to the nearest whole pixel so all voxels stay original. `subpixel`: the exact shift is kept and the bottom half interpolated. It changes nothing when the halves have different pixel sizes (the usual case). |
| `snapped_to_whole_pixel` | `false` | `true` only if a snap actually happened (same pixel grid and `whole_pixel`). |
| `snap_residual_mm` | `0.0` | How far the snap moved the half, in mm (at most half a pixel). 0 if no snap. |
| `refine` | `null` | `null` unless *Refine join* was used; then a block, below. |

**`placement.refine`** (present only if Refine join was used and accepted)

| Field | Meaning |
|---|---|
| `method` | What was done: SimpleITK translation registration, correlation metric, linear interpolation, 2 resolution levels. |
| `shift_mm.along_core`, `shift_mm.across_core` | How far the bottom half was moved, in mm. Refine never changes depth. |
| `optimum_correlation_before`, `optimum_correlation_after` | The optimiser's correlation over the overlap at the start and end of its search (on a slab of slices). Useful only for judging the search itself. |
| `join_correlation_as_placed_before`, `join_correlation_as_placed_after` | The correlation of the two halves over the overlap on the middle slice, measured on the images exactly as placed, before and after. These two are comparable with each other. |
| `iterations` | Optimiser iterations used. |

### `header_check` – do the scanner positions agree with the images?

Made once, when the pair was loaded, from one middle slice of each half. It is a sanity check on the
scanner's placement, not a measurement of the finished join.

| Field | Example | Meaning |
|---|---|---|
| `verdict` | `"headers_good"` | `headers_good`, or `needs_image_stitch` if any reason below applies. |
| `reasons` | `[]` | Plain-language reasons for a doubtful verdict (core centres far apart, stacks offset in depth, no overlap, profiles fit better at another position). Empty is good. |
| `across_diff_mm` | `-1.16` | Bottom core centre minus top core centre, across the core, in mm. More than 5 mm in either direction is doubtful. |
| `depth_diff_mm` | `-0.0` | Difference in where the two stacks' first slice planes pass through the core, through the stack, in mm (measured at the core, so halves cut in slightly different planes are comparable). More than 5 mm is doubtful. |
| `orientation_difference_deg` | `0.0` | Angle between the two halves' image planes, in degrees. 0 for the usual pair. Scanner reformats are sometimes cut a few degrees apart (about 5 degrees on some DF80 sections); the halves are then still joined by their headers, with the coarser half resampled into the other's orientation. |
| `notes` | `[]` | Information that is not a reason to doubt the placement, for example that the two halves were cut in planes a few degrees apart. |
| `profile_correlation_at_header` | `0.995` | Correlation of the two cores' brightness profiles (mean HU at each position down the core) in the overlap the headers imply. 1 = identical. This is a **pre-check at the scanner position**, one slice only, and does not change when you move things. Compare `join_quality.correlation`, which uses the whole overlap volume. |

### `join` – where the two halves meet

All row numbers refer to **rows of the exported image** (row 0 = top of the picture, counting down the core).

| Field | Example | Meaning |
|---|---|---|
| `overlap_rows_in_image` | `[2315, 2572]` | First and last row where **both** halves have data. `null` if they do not overlap in the picture. This is the single best answer to "where is the join". |
| `overlap_cm` | `10.0` | Length of the overlap, in cm. |
| `top_rows_in_image` | `[0, 2572]` | The rows the top half's scan covers in the picture. |
| `bottom_rows_in_image` | `[2315, 3938]` | The rows the bottom half's scan covers. |
| `fuse` | | How the two halves were combined in the overlap: the **average** of the two, except within 3 rows of either half's image edge, where only the other half is used. |
| `outside_core_fill_hu` | `-2048.0` | The HU value written everywhere that is not core (real air, as the scanner records it). |

### `warnings`

| Field | Meaning |
|---|---|
| `warnings` | Notes raised while loading or aligning, as a list of text. Empty is good. |

### Where the scan belongs (exports started from the Worklist)

| Field | Example | Meaning |
|---|---|---|
| `job_kind` | `"pair"` | `pair` (two halves stitched) or `single` (one scan, nothing joined). |
| `cruise_folder` | `"HLY0501"` | Cruise the scan belongs to. |
| `core_name` | `"HLY0501-6JC"` | Core. |
| `section_key` | `"HLY0501-6JC-7"` | Section of the core. |
| `export_version` | `1` | 1 for the first export of this job, 2 for the next, and so on. Earlier exports stay in their own folders. |
| `catalog_section_length_cm` | `150.5` | The section length on the master list. Used for the length check. `null` if not matched. |
| `series.a`, `series.b` | | The two scans used, as listed in the release catalog: `uid` (the last 12 characters of the Series UID), `description`, and `plane`. `a` is the top half and `b` the bottom half. |

### About this export

| Field | Example | Meaning |
|---|---|---|
| `exported_at` | `"2026-11-06T22:22:52+00:00"` | When the export was made, UTC. |
| `view_slice` | `17` | Which slice of the stitched stack (counting from 0) the JPEG and 8-bit TIFF were made from. |
| `n_slices` | `34` | Number of slices in the stitched stack. |
| `window_lo`, `window_hi` | `-304.0`, `1394.6` | The greyscale window, in HU: values at or below `window_lo` are black, at or above `window_hi` white, in the JPEG and 8-bit TIFF. By default the 1st and 99th percentile of the core's HU values on `view_slice`. The same numbers are also written as window centre and width in the DICOM. |

### `join_quality` – how well the two halves agree over the overlap

Computed on **every** export, from the overlap in all slices. It is a check, not a correction: nothing is
moved by it.

| Field | Example | Meaning |
|---|---|---|
| `evaluated` | `true` | `false` if there was too little overlap to evaluate (then `reason` says why and the other fields are missing). |
| `correlation` | `0.9964` | Correlation of the two halves' HU values over the overlap core, at the placement as exported. 1 = identical. On real, well-placed cores it has been about 0.99 to 0.9997. |
| `voxels` | `1592285` | How many voxels it was computed on (the interior of the overlap, inside both halves' cores, away from either half's edge rows). |
| `slices` | `34` | How many slices that includes. |
| `best_shift.depth_slices` | `0` | Of the whole-voxel shifts of the bottom half tried through the stack (-3 to +3 slices), the one with the highest correlation. 0 means the exported placement is already the best of those, which is what a good join gives. It is measured on the placement that was exported (after any shift or refine), so a non-zero value is a warning that the data prefer a different position; it is not a record of something that was undone. |
| `best_shift.along_px` | `0` | The same along the core (-3 to +3 pixels). |
| `best_shift.across_px` | `0` | The same across the core (-3 to +3 pixels). A best shift of 0 or +-1 pixel along or across is normal; only 2 or more pixels (1 or more slices in depth) is flagged. |
| `correlation_at_best_shift` | `{...}` | The correlation at each axis's best shift. Equal to `correlation` when the best shift is 0. |
| `scan` | `{...}` | Every correlation tried: for each axis, shift in voxels (as text) → correlation. Shows how sharply the peak falls off (depth is sharp; along the core the curve is flat). One axis at a time; the other two stay at the exported value. |
| `consistent` | `true` | `false` when some other position is clearly better (see `flags`). |
| `flags` | `[]` | Plain-language warnings, for example "the halves agree better if the bottom half is moved +2 px (+0.78 mm) across the core (correlation 0.953 -> 0.971)". An axis is flagged only when its best shift is 2 or more pixels away (1 or more slices in depth) **and** beats the exported placement by more than a per-axis margin (depth 0.005, along 0.0001, across 0.001). |
| `note` | | Reminder of what was computed: whole-voxel shifts of the output grid, one axis at a time. (In testing, planted errors of 0.8 mm or more along or across the core, and a one-slice depth error, were flagged; 0.4 mm was not.) |

### `rotate_180` – was the picture turned around?

Written on every export (stitched and single). Used when the core is upside down in the scan (its top end is at the bottom of
the picture). The whole picture is turned 180 degrees in its own plane: top becomes bottom and left becomes right. It is
lossless: rows and columns are only reversed, no pixel value changes, so it is separate from `straighten`.

| Field | Example | Meaning |
|---|---|---|
| `applied` | `false` | `true` if the picture was turned 180 degrees. |
| `note` | `null` | A reminder of what was done when `applied` is `true`. In a stitched export the DICOM orientation vectors are negated to match, so every voxel keeps its true position in space. |

### `straighten` – was the picture rotated?

Written on every export (stitched and single). Straightening is **off by default** because it interpolates
pixels. The tilt is measured and recorded either way.

| Field | Example | Meaning |
|---|---|---|
| `applied` | `false` | `true` if the picture was rotated so the core runs exactly down the rows. |
| `measured_tilt_deg` | `0.0995` | How far the core's long axis leaned from straight down the rows, in degrees, before any rotation. Positive = the core leans toward larger column numbers as it goes down the picture. Use it to correct a ruler if the picture was not straightened (a 0.1 degree lean moves the core about 1 mm sideways over 55 cm). |
| `rotation_applied_deg` | `0.0` | The rotation actually applied, in degrees (the negative of the tilt when `applied` is `true`, otherwise 0). |
| `interpolation` | `null` | `null` when nothing was rotated. Otherwise states that the values were interpolated: for a single scan the slice was resampled once; for a stitched pair the whole stitched volume (including the reference half) was resampled once more after stitching, so the reference half's voxels are no longer exactly as scanned. The DICOM stack's orientation vectors are rotated to match, so every voxel keeps its true position in space. |

### Cropping and shape of the exported pictures

| Field | Example | Meaning |
|---|---|---|
| `export_jpeg_pad_mm` | `3.0` | Margin left around the core when the pictures were cropped, in mm. |
| `export_crop_box_r0r1c0c1` | `[5, 3944, 18, 341]` | The part of the full working picture that was kept: first row, end row (exclusive), first column, end column (exclusive). |
| `export_shape_px` | `[3939, 323]` | Height and width of the exported pictures, in pixels. Equals `core_scale.image_rows` and `image_cols`. |

### `core_scale` – measuring on the picture

For drawing a ruler, or turning pixel positions into centimetres down the core. The picture's layout: the
core runs **down the rows**; row 0 is the **top** (shallow end). Positions are **pixel-edge coordinates**: row 0 is the
top edge of the first pixel row and row `image_rows` the bottom edge of the last, so
`(core_bottom_row_px - core_top_row_px) * mm_per_px` is exactly the core length.

| Field | Example | Meaning |
|---|---|---|
| `schema` | `"core_scale/1"` | Version of this block's layout. |
| `mm_per_px` | `0.387743` | Size of one pixel in mm (the same across and down). |
| `px_per_cm` | `25.79` | The same thing the other way round. |
| `depth_axis` | `"rows"` | Which way depth runs in the picture. Always `rows`. |
| `row_zero_is` | `"core_top"` | What the first row is. Always `core_top`. |
| `image_rows`, `image_cols` | `3939`, `323` | Height and width of the exported picture, in pixels. |
| `image_depth_extent_cm` | `152.732` | Full height of the picture in cm. Longer than the core because of the margin. |
| `core_top_inset_cm` | `0.9204` | How far in from the top edge of the picture the core starts, in cm. |
| `core_bottom_inset_cm` | `2.3949` | How far in from the bottom edge of the picture the core ends, in cm. |
| `core_length_cm` | `149.4166` | Length of the core between the two ends: picture extent minus both insets. |
| `core_top_row_px`, `core_bottom_row_px` | `23.74`, `3877.23` | Row of each end of the core, in pixels. |
| `core_center_col_px` | `161.5` | Column down the middle of the picture, where a ruler would be drawn. |
| `endpoints_px.top`, `endpoints_px.bottom` | `[23.74, 161.5]` | The two ends as `[row, column]` pairs. The easiest field to use for a ruler. |
| `bounds_source` | `"manual"` | `auto`: the program guessed the ends from the CT (generous: includes end caps and the liner, so the length is often a few cm too long). `manual`: a person dragged the guides onto the sediment ends. |

**Length check against the master list** (inside `core_scale`)

| Field | Example | Meaning |
|---|---|---|
| `expected_length_cm` | `150.5` | Section length from the master list. `null` if the scan is not matched to a section. |
| `length_difference_cm` | `-1.083` | Measured minus master list. Positive means the scan measures longer. |
| `length_difference_pct` | `-0.72` | The same as a percentage of the master-list length. |
| `length_check` | `"compared"` | `compared` (numbers recorded) or `no_reference` (nothing to compare with). **No pass/fail is applied**; the numbers are only recorded. A large difference can mean the guides are misplaced, the section label is wrong, or the scan covers only part of a section. |

### Files written

| Field | Example | Meaning |
|---|---|---|
| `export_jpeg` | `"..._coronal_pair.jpg"` | Name of the JPEG. |
| `export_tiff` | `"..._coronal_pair.tif"` | Name of the 8-bit TIFF. |
| `export_tiff_hu` | `"..._coronal_pair_hu.tif"` | Name of the HU TIFF. |
| `export_tiff_hu_offset` | `0.0` | To get real HU from `_hu.tif`: **HU = pixel value + this number.** Always 0: the TIFF is signed, so the pixel value *is* the HU. The field is kept so programs can apply the same formula without special cases. |
| `export_tiff_hu_spacing_mm` | `0.387743` | Pixel size of the HU TIFF in mm. |
| `export_dicom_dir` | `"dicom_mpr"` | Folder holding the DICOM stack. |
| `export_dicom_series_uid` | `"1.2.826.0.1.3680043..."` | Series Instance UID given to the stitched DICOM stack (new, generated for the export). |

---

## Example: building a ruler in Python

```python
import json
meta = json.load(open("section_pair.json"))
cs = meta["core_scale"]
(top_row, col), (bot_row, _) = cs["endpoints_px"]["top"], cs["endpoints_px"]["bottom"]

def row_to_cm(row):
    """Centimetres down the core from its top end, for a row in the exported image."""
    return (row - top_row) * cs["mm_per_px"] / 10.0

# ticks every 1 cm along the core
ticks = [(top_row + cm * cs["px_per_cm"], cm) for cm in range(int(cs["core_length_cm"]) + 1)]
```

To add real depth in the core, add the section's top depth from the master list to the value returned by
`row_to_cm`.

## Example: reading real HU from the HU TIFF

```python
import json, tifffile
meta = json.load(open("section_pair.json"))
hu = tifffile.imread("section_pair_hu.tif").astype("float32") + meta["export_tiff_hu_offset"]
```

---

## Single scans (nothing joined)

A section that needs no stitching still gets the JPEG, the 8-bit TIFF, the HU TIFF and a sidecar. Because
nothing is joined, the pictures are made from **one slice** (`frame`) of the scan, and no DICOM stack is written.
The sidecar has the worklist block (including `series.a`), the export fields, `core_scale`,
`window_lo`/`window_hi`, `exported_at`, the crop and shape fields and `warnings`, all as described above. The stitching
blocks (`top`, `bottom`, `resolution`, `placement`, `header_check`, `join`, `join_quality`) are absent, and these
fields are added:

| Field | Meaning |
|---|---|
| `unstitched` | `true`: nothing was joined. (Stitched exports have `stitched: true` instead.) |
| `scan` | The scan used: `series_uid`, `series_description`, `n_slices` (slices in the scan) and `spacing_mm_zrc` (spacing in mm as slice step, row pitch, column pitch). |
| `frame` | Which slice of the scan the pictures were made from (counting from 0). |
| `straighten` | Whether the picture was rotated so the core runs exactly top to bottom, and the measured tilt. Same block as for stitched exports (see `straighten` above). Off by default: the pixel values are then exactly as scanned. |
| `rotate_180` | Whether the picture was turned 180 degrees because the core was upside down. Same block as for stitched exports (see `rotate_180` above). Lossless. |
| `target_mm` | Pixel size of the exported picture, in mm (the scan's finer pixel pitch). |
