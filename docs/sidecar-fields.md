# What is in the JSON sidecar

Every export can write a small text file next to the images: `<name>.json`. It records how the
image was made and where the core sits in it. This page explains every field in plain language.

You do not need to read the JSON by hand. Any program that reads JSON (Python `json`, R
`jsonlite`, MATLAB `jsondecode`, a browser) can pick out the fields below by name.

**Files in one export folder**

| File | What it is |
|---|---|
| `<name>.jpg` | Greyscale picture of the core, cropped to the core. For looking at. |
| `<name>.tif` | The same picture as an 8-bit TIFF, no compression loss. |
| `<name>_hu.tif` | The real CT numbers (Hounsfield units, HU) as a 16-bit TIFF. See `export_tiff_hu_offset`. |
| `<name>.json` | This sidecar. |
| `*_fused.npy`, `*_mpr.npy`, `dicom_mpr/` | Optional extras for stitched cores (NumPy arrays, DICOM stack). |

**The pictures are clean.** No scale bar, guides or labels are drawn into the JPEG or TIFFs.
Everything needed to measure on them is in the sidecar (see "core_scale").
The JPEG and TIFFs also carry the pixel size in their resolution tags, so most image programs
show the correct physical size.

---

## Measuring on the image: `core_scale`

This block is in every sidecar, for stitched and single-scan exports alike.
Use it to draw a ruler, or to turn pixel positions into centimetres down the core.

**The picture's layout:** the core runs **down the rows** of the image. Row 0 is the **top** (shallow
end) of the core section. Positions are written in **pixel-edge coordinates**: row 0 is the top edge of the
first pixel row, and row `image_rows` is the bottom edge of the last one. Because of this,
`core_bottom_row_px - core_top_row_px` times `mm_per_px` is exactly the core length.

| Field | Plain meaning |
|---|---|
| `schema` | Version of this block's layout (`core_scale/1`). Lets future programs know how to read it. |
| `mm_per_px` | Size of one pixel in millimetres (the same across and down). |
| `px_per_cm` | The same thing the other way round: pixels in one centimetre. |
| `depth_axis` | Which way depth runs in the image. Always `rows` (top to bottom). |
| `row_zero_is` | What the first row is. Always `core_top`. |
| `image_rows`, `image_cols` | Height and width of the exported image in pixels. |
| `image_depth_extent_cm` | Full height of the image in cm. This is longer than the core because of a small margin. |
| `core_top_inset_cm` | How far in from the top edge of the image the core starts, in cm. |
| `core_bottom_inset_cm` | How far in from the bottom edge of the image the core ends, in cm. |
| `core_length_cm` | Length of the core between the two ends: image extent minus both insets. |
| `core_top_row_px` | Row number (in pixels) of the core's top end. |
| `core_bottom_row_px` | Row number (in pixels) of the core's bottom end. |
| `core_center_col_px` | Column that runs down the middle of the image, where a ruler would be drawn. |
| `endpoints_px` | The two ends as `[row, column]` pairs: `top` and `bottom`. The easiest field to use for a ruler. |
| `bounds_source` | `auto` means the program guessed the ends from the CT. `manual` means a person dragged the guides. |

The names `core_top_inset_cm`, `core_bottom_inset_cm` and `core_length_cm` are the same ones the
multi-view tool uses, so measurements agree between the tools.

**Heads up on `auto`:** the automatic ends are generous. They include the end caps and the liner
frame, so an automatic length is often a few centimetres longer than the sediment. Drag the guides
onto the sediment ends when you need the sediment length.

### Length check against the master list

When the scan is matched to a section in the master list, the sidecar compares the measured length to
the section length on record.

| Field | Plain meaning |
|---|---|
| `expected_length_cm` | Section length from the master list. Empty if the scan is not matched to a section. |
| `length_difference_cm` | Measured minus master list. Positive means the scan measures longer. |
| `length_difference_pct` | The same difference as a percentage of the master-list length. |
| `length_check` | `compared` (the numbers above were recorded) or `no_reference` (nothing to compare with). |

**No pass/fail is applied.** The difference is only recorded; what counts as too much will be
decided later from the recorded numbers. A large difference can mean the guides are in the wrong place,
the section label is wrong, or the scan covers only part of a section (or more than one).

The same numbers are stored in the worklist (the `length vs list` column shows the difference in cm)
and in `ct_stitch_status.csv` (`length_check`, `length_diff_cm`).

---

## Where the data came from (worklist exports)

Added when the export was started from the Worklist tab.

| Field | Plain meaning |
|---|---|
| `job_kind` | `pair` (two halves stitched) or `single` (one scan, no join). |
| `cruise_folder`, `core_name`, `section_key` | Where the scan belongs: cruise, core, section. |
| `export_version` | 1 for the first export of this job, 2 for the next, and so on. Older versions are kept. |
| `catalog_section_length_cm` | The master-list section length used for the length check. |
| `series` | The scan series used: `a` (and `b` for a pair) with series ID, description and plane. |

## About the scan and the export (all exports)

| Field | Plain meaning |
|---|---|
| `version` | Version of the sidecar layout. |
| `exported_at` | When the export was made (UTC time). |
| `unstitched` | `true` for a single scan: nothing was joined. Missing for stitched exports. |
| `target_mm` | Pixel size the picture was resampled to, in mm. |
| `warnings` | Notes the program raised while loading or aligning. Empty is good. |
| `export_shape_px` | Height and width of the exported picture in pixels. |
| `export_crop_box_r0r1c0c1` | The part of the full working image that was kept: first row, last row, first column, last column. |
| `export_jpeg_pad_mm`, `export_jpeg_cropped` | The margin left around the core when cropping, in mm, and whether cropping was on. |
| `window_lo`, `window_hi`, `export_window_lo`, `export_window_hi` | The greyscale window: HU values shown as black and white in the JPEG/8-bit TIFF. |
| `export_jpeg`, `export_tiff`, `export_tiff_hu`, `export_npy`, `export_mpr_npy` | Names of the files written (no folder paths). |
| `export_tiff_hu_offset` | To get real HU from `_hu.tif`: HU = pixel value + this number. |
| `export_tiff_hu_spacing_mm` | Pixel size of the HU TIFF in mm. |
| `export_dicom_dir`, `export_dicom_series_uid` | Name of the DICOM folder and ID of the DICOM stack, if written. |
| `export_mpr_shape_zyx`, `export_mpr_z_spacing_mm` | Size of the 3D stack and the gap between its slices in mm. |
| `export_view_z` | Which slice of the stack the 2D picture shows. |

### Single scans only

| Field | Plain meaning |
|---|---|
| `series` | The scan used: series ID, description, slice count, and spacing (mm) as slice, row, column. |
| `frame` | Which slice (frame) of the scan the picture was made from. |
| `upright_deg` | How far the picture was turned so the core runs top to bottom. |
| `window_lo`, `window_hi` | Greyscale window in HU. |

### Stitched cores only

| Field | Plain meaning |
|---|---|
| `fixed`, `mobile` | The two scans joined: the top half (`fixed`) and the bottom half (`mobile`): series ID and description. |
| `flip` | Whether the bottom half was flipped before joining. |
| `roll_deg` | Turn of the bottom half around the core's long axis, in degrees. |
| `nudge_x_mm`, `nudge_y_mm`, `nudge_z_mm` | Hand adjustments to the bottom half: across the image, down the core, and through the stack. |
| `fuse_mode` | How the two halves were blended where they overlap. |
| `match_hu` | Whether the bottom half's brightness was matched to the top half. |
| `view_z` | Slice shown while aligning. |
| `n_slices` | Slices in the joined stack. |
| `total_lag_px` | Where the bottom half starts: its first row sits this many pixels below the top half's first row. A larger number means a smaller overlap. This is the final position, including hand nudges and refine. |
| `total_col_offset_px` | The hand nudge across the core, in pixels. |
| `refine` | Optional fine adjustment: shifts in mm and pixels, the match score before and after, and the method. Empty if not used. |

---

### Stitches placed from the scanner geometry (sidecar version 0.2.0)

These exports come from the scanner-geometry (SimpleITK) mode. They have `"engine": "simpleitk_header_geometry"`
and `"stitched": true`. The two halves are put where the scanner recorded them (from the DICOM headers),
the half with the finer pixels is kept exactly as scanned, and the other half is resampled once onto its grid.
The older fields in the table above (`fixed`, `mobile`, `flip`, `roll_deg`, `nudge_*`, `total_lag_px`,
`refine`, ...) are not present in these files; the blocks below replace them.

| Field | Plain meaning |
|---|---|
| `engine`, `software` | The method, and the SimpleITK version that did the resampling. |
| `top`, `bottom` | The two scans joined: series ID, description, their original pixel size (`native_mm_per_px`) and slice step. |
| `resolution.output_mm_per_px` | Pixel size of the stitched picture, in mm. It is the finer of the two scans' pixel sizes. |
| `resolution.reference_half` | The half that was kept exactly as scanned (`top` or `bottom`). |
| `resolution.interpolated_half` | The half that was resampled to the finer pixel size, or `null` if both already matched. |
| `resolution.upsample_factor_of_interpolated_half` | How much finer the pixels were made (1.58 means 1.58 times). Interpolated pixels are smoothed estimates; they add no new detail. |
| `resolution.interpolation` | How the resampling was done (`linear`, or nearest-neighbour when every voxel kept its original value). |
| `placement.source` | Where the position came from: `scanner header` (as recorded), or `image alignment (mask + profile)` if you used Re-align from image. |
| `placement.bottom_shift_mm` | Any shift you applied to the bottom half from that position, in mm: `along_core` (+ is down), `across_core`, `through_stack`. All zero means the scanner position was used as is. |
| `placement.pixel_placement`, `snapped_to_whole_pixel`, `snap_residual_mm` | When both halves have the same pixel size, "whole_pixel" places the bottom half to the nearest whole pixel so every voxel stays original; the residual is how far that moved it (at most half a pixel). |
| `placement.refine` | Present only if Refine join was used (otherwise `null`): the method (SimpleITK translation registration, correlation metric), the shift it made along and across the core in mm, and the correlation of the two halves over the overlap before and after (`join_correlation_as_placed_*`, measured on the images as placed, so before and after are comparable). Refine never changes the depth shift. |
| `header_check` | A check made when the pair was loaded: do the scanner positions agree with the images? `verdict` is `headers_good` or `needs_image_stitch`, with the reasons, the sideways and depth differences in mm, and the correlation of the two cores' profiles at the scanner position. |
| `join.overlap_rows_in_image` | The rows of the exported picture where both halves overlap (first row, last row). |
| `join.overlap_cm` | Length of the overlap, in cm. |
| `join.top_rows_in_image`, `bottom_rows_in_image` | The rows of the picture that each half covers. |
| `join.fuse` | How the halves were combined in the overlap (the average, except right at each half's image edge, where only the other half is used). |
| `join.outside_core_fill_hu` | The HU value used outside the core (real air). |
| `join_quality` | A check made on every export, over the overlap in all slices. `correlation` is how well the two halves agree there (1 = identical). `scan` and `best_shift` try moving the bottom half by whole voxels, one axis at a time (up to 3 pixels along and across the core, 3 slices in depth): at a good join the best shift is 0 on every axis. `consistent` is false and `flags` explain it when some other position is clearly better (2 or more pixels away, or 1 or more slices). It is a check, not a refinement: nothing is moved. Depth errors show strongly here; errors along the core change the correlation only slightly, so they are flagged by where the peak is, not by how much the number drops. |
| `view_slice`, `n_slices` | Which slice the 2D picture shows, and how many slices the stack has. |

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

To add real depth in the core, add the section's top depth from the master list to the value
returned by `row_to_cm`.
