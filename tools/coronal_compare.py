"""Compare coronal MPRs rebuilt from the axial stack (Python SimpleITK) with the Vetmed Toshiba scanner's own coronals.

    python tools/coronal_compare.py --label NAME --axial DIR --coronal DIR [--coronal NAME=DIR ...]
                                    [--out OUTDIR] [--note "text"] [--plane-rule median|middle]
    python tools/coronal_compare.py --summarize OUTDIR

For each coronal series it rebuilds planes from the axial stack on the scanner's own pixel grid (orientation, origin and
pixel size from its DICOM header; a mean slab of the header's slice thickness), measures every plane that cuts the
core, and writes one PNG for one plane:
  Vetmed Toshiba | Python SimpleITK | difference (with an HU colour bar), one representative noise patch, one
  representative edge patch across the core and one along it, the difference histogram with the key numbers, and the
  down-core mean-HU profile.
Planes and patches are chosen by fixed rules, never by how good they look: the plane shown is the one whose spread of
the difference is the median among the planes through the main body of the core (``--plane-rule middle`` takes the middle
plane instead); the noise and edge patches are the ones closest to the median of their kind.  The figure checks itself
for overlapping text.  ``--summarize`` turns the JSON files of several runs into one summary page (with a per-plane chart)
and CSV.  Output defaults to ``tools/coronal_compare_output``, which git ignores.

Read-only on the DICOM; writes only into OUTDIR.  Needs matplotlib (``pip install -e .[figures]``).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from core_stitch import coronal_mpr as cm

TOSH = "Vetmed Toshiba"  # the scanner's own coronal (what is called "his" in the code)
PYSI = "Python SimpleITK"  # the coronal rebuilt from the axial stack (what is called "ours" in the code)


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# ---------------------------------------------------------------------------------------------- analysis


def choose_segment(his, core, rows, px, length_mm=190.0):
    """Start row of the displayed segment: the stretch of the core with the most fine detail (deterministic)."""
    g = np.hypot(ndi.sobel(his, axis=1), ndi.sobel(his, axis=0)) / 8.0
    per_row = (g * core).sum(axis=1)
    n = int(round(length_mm / px))
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    if hi - lo <= n:
        return lo, hi
    cs = np.concatenate([[0.0], np.cumsum(per_row)])
    best = max(range(lo, hi - n + 1), key=lambda s: cs[s + n] - cs[s])
    return best, best + n


def analyse(renderer, k, thickness, with_fit=True, with_sweep=True):
    cor = renderer.cor
    his = renderer.his_plane(k)
    ours, valid = renderer.render_plane(k, thickness=thickness)
    core, rows = cm.core_mask(his, valid, cor.px)
    if core.sum() < 5000:
        raise RuntimeError(f"plane {k}: too little core to compare ({int(core.sum())} pixels)")
    out = dict(k=k, his=his, ours=ours, valid=valid, core=core, rows=rows)
    out["stats"] = cm.difference_stats(ours, his, core)
    m_ = core & np.isfinite(ours)
    out["stats"]["mean_floor"] = float(np.mean((np.floor(ours) - his)[m_]))  # bias if ours were rounded down to whole numbers
    out["grad_ratio"] = cm.peak_gradient_ratio(ours, his, core)
    blocks = cm.flat_blocks(his, core, rows)
    out["blocks"] = blocks
    if blocks:
        out["sd_his"], out["sd_ours"] = cm.block_sd(his, blocks), cm.block_sd(ours, blocks)
    if with_fit:
        ap, _ = renderer.render_plane(k, thickness=thickness, dy=+0.25)
        am, _ = renderer.render_plane(k, thickness=thickness, dy=-0.25)
        out["fit"] = cm.fit_terms(his, ours, (ap - am) / 0.5, core)
    if with_sweep:
        out["sweep"] = cm.slab_sweep(renderer, k, his, core)
    return out


def choose_patches(res, px):
    """Noise block and edge window nearest the median (representative), preferably inside the displayed segment."""
    his, ours, core, rows = res["his"], res["ours"], res["core"], res["rows"]
    s0, s1 = res["segment"]
    # noise: the flat block whose SD is closest to the median block SD
    noise = None
    if res["blocks"]:
        sd, med = res["sd_his"], float(np.median(res["sd_his"]))
        inseg = [i for i, (r0, _c0) in enumerate(res["blocks"]) if s0 <= r0 <= s1 - cm.BLOCK]
        cand = inseg or list(range(len(sd)))
        i = min(cand, key=lambda j: abs(sd[j] - med))
        noise = dict(block=res["blocks"][i], sd_his=float(res["sd_his"][i]), sd_ours=float(res["sd_ours"][i]))
    # edge: a window of 20 consecutive profiles on the side wall whose width is closest to the median window width
    images = {"his": his, "ours": ours}
    best = None
    for side in ("left", "right"):
        prof, info = cm.edge_profiles(images, his, core, rows, side)
        if len(info["line"]) >= 40 and (best is None or len(info["line"]) > len(best[1]["line"])):
            best = (prof, info, side)
    edge = None
    if best is not None:
        prof, info, side = best
        n = len(info["line"])
        starts = list(range(0, n - 20 + 1, 10))
        widths = np.array([cm.edge_width(prof["his"][s:s + 20].mean(axis=0)) for s in starts])
        med = np.nanmedian(widths)
        lines = np.array(info["line"])
        inseg = [i for i, s in enumerate(starts) if s0 <= lines[s] and lines[s + 19] <= s1]
        cand = inseg or list(range(len(starts)))
        i = min(cand, key=lambda j: abs(widths[j] - med) if np.isfinite(widths[j]) else 1e9)
        s = starts[i]
        win = slice(s, s + 20)
        edge = dict(side=side, r0=int(lines[s]), r1=int(lines[s + 19]), col=int(np.median(np.array(info["edge"])[win])),
                    offs=info["offs"], his=prof["his"][win].mean(axis=0), ours=prof["ours"][win].mean(axis=0))
        edge["w_his"], edge["w_ours"] = cm.edge_width(edge["his"]), cm.edge_width(edge["ours"])
        edge["all_w_his"] = cm.edge_width(prof["his"].mean(axis=0))
        edge["all_w_ours"] = cm.edge_width(prof["ours"].mean(axis=0))
        edge["n_profiles"] = n
    # sharpness ALONG the core: the strongest edge that runs across the core, preferably inside the displayed segment
    along = None
    prof, info = cm.across_core_edge(images, his, core, (s0, s1))
    in_seg = prof is not None
    if prof is None:
        prof, info = cm.across_core_edge(images, his, core, (int(rows[0]) + 13, int(rows[-1]) - 13))
    if prof is not None:
        along = dict(offs=info["offs"], his=prof["his"].mean(axis=0), ours=prof["ours"].mean(axis=0), n=len(info["cols"]),
                     row=info["row"], c0=int(min(info["cols"])), c1=int(max(info["cols"])), in_segment=bool(in_seg))
        along["w_his"], along["w_ours"] = cm.edge_width(along["his"]), cm.edge_width(along["ours"])
    return noise, edge, along


# ---------------------------------------------------------------------------------------------- figure


def layout_problems(fig):
    """Text that overlaps other text or runs off the figure (rendered bounding boxes)."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    skip = set()  # tick labels outside the axis range are not drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for loc, lab in zip(axis.get_majorticklocs(), axis.get_majorticklabels()):
                if loc < lo - 1e-9 or loc > hi + 1e-9:
                    skip.add(lab)
    items = []
    for t in fig.findobj(mtext.Text):
        if t in skip or not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(rend)
        if bb.width >= 1 and bb.height >= 1:
            items.append((t, bb))
    W, H = fig.canvas.get_width_height()
    out = []
    for i, (t, bb) in enumerate(items):
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            out.append(("off the figure", t.get_text()[:50].replace("\n", " ")))
        for t2, bb2 in items[i + 1:]:
            if min(bb.x1, bb2.x1) - max(bb.x0, bb2.x0) > 2 and min(bb.y1, bb2.y1) - max(bb.y0, bb2.y0) > 2:
                out.append(("overlap", t.get_text()[:40].replace("\n", " "), t2.get_text()[:40].replace("\n", " ")))
    return out


def draw_figure(res, label, series_name, cor, stack, noise, edge, along, thickness, note, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    px = cor.px
    his, ours, valid, core, rows = res["his"], res["ours"], res["valid"], res["core"], res["rows"]
    s0, s1 = res["segment"]
    st, fit = res["stats"], res.get("fit", {})
    filled = ndi.binary_fill_holes(his > cm.CORE_HU)
    cols = np.where(filled.any(axis=0))[0]
    c0, c1 = max(int(cols[0]) - 8, 0), min(int(cols[-1]) + 9, his.shape[1])
    vmin, vmax = np.percentile(his[core], [1, 99.5])
    ext = lambda r0, r1, a0, a1: (a0 * px, a1 * px, r1 * px, r0 * px)  # x0, x1, bottom, top in mm
    sd_med = float(np.median(res["sd_his"])) if res["blocks"] else float("nan")
    diff = np.where(valid & filled, ours - his, np.nan)
    LIM = 20.0
    C_NOISE, C_WALL, C_ALONG = "#d62728", "#2ca02c", "#ff7f0e"

    fig = plt.figure(figsize=(17, 10.4), dpi=130)
    outer = fig.add_gridspec(3, 1, height_ratios=[0.26, 1.95, 1.4], hspace=0.30, left=0.05, right=0.985, top=0.975, bottom=0.05)
    # ---- title
    axt = fig.add_subplot(outer[0]); axt.axis("off")
    axt.text(0, 0.84, f"{label}  -  '{cor.description}':  {PYSI} coronal vs {TOSH} coronal", fontsize=15, fontweight="bold", va="center")
    axt.text(0, 0.50, f"plane {res['k'] + 1} of {cor.n_planes}: {res.get('rule', '')}", fontsize=10.5, va="center", color="#333333")
    axt.text(0, 0.24, f"{PYSI} = {thickness:g} mm mean slab rebuilt from the {stack.geom.thickness:g} mm axial "
                      f"({len(stack.geom.paths)} slices), on the {TOSH} pixel grid ({px:.3f} mm)",
             fontsize=10.5, va="center", color="#333333")
    if note:
        axt.text(0, 0.02, note, fontsize=10.5, va="center", color="#b02020", fontweight="bold")
    # ---- top row
    top = outer[1].subgridspec(1, 7, width_ratios=[0.42, 1, 1, 1, 0.06, 0.62, 2.0], wspace=0.12)
    axo = fig.add_subplot(top[0]); axh = fig.add_subplot(top[1]); axu = fig.add_subplot(top[2]); axd = fig.add_subplot(top[3])
    axc = fig.add_subplot(top[4])
    r_lo, r_hi = int(rows[0]), int(rows[-1]) + 1
    axo.imshow(his[r_lo:r_hi, c0:c1], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", extent=ext(r_lo, r_hi, c0, c1))
    axo.set_title("whole core", fontsize=10)
    axo.set_ylabel("distance along the core (mm)")
    axo.set_xticks([])
    axo.add_patch(Rectangle((c0 * px, s0 * px), (c1 - c0) * px, (s1 - s0) * px, fill=False, ec="#1f77b4", lw=1.8))
    for ax, img, ttl in ((axh, his, f"{TOSH}\n(scanner MPR)"), (axu, ours, f"{PYSI}\n(rebuilt from the axial)")):
        ax.imshow(img[s0:s1, c0:c1], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", extent=ext(s0, s1, c0, c1))
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    im = axd.imshow(diff[s0:s1, c0:c1], cmap="RdBu_r", vmin=-LIM, vmax=LIM, aspect="equal", extent=ext(s0, s1, c0, c1))
    axd.set_title(f"difference\n({PYSI} - {TOSH})", fontsize=9.5); axd.set_xticks([]); axd.set_yticks([])
    cb = fig.colorbar(im, cax=axc)
    cb.set_label("HU", labelpad=4)
    cb.set_ticks([-LIM, -10, 0, 10, LIM])
    axd.text(0.5, -0.012, "dark lines at the walls: liner / air edge,\noutside the core mask used for the numbers"
             + (f"\ntypical noise in the {TOSH} image:\nabout {sd_med:.0f} HU (1 SD)" if np.isfinite(sd_med) else ""),
             transform=axd.transAxes, ha="center", va="top", fontsize=8.5, color="#444444")
    axo.set_ylim(r_hi * px, r_lo * px)
    # patch boxes (main panels + a marker on the overview)
    if noise is not None:
        r0, cc0 = noise["block"]
        for ax in (axh, axu, axd):
            ax.add_patch(Rectangle((cc0 * px, r0 * px), cm.BLOCK * px, cm.BLOCK * px, fill=False, ec=C_NOISE, lw=1.6))
        axo.plot([cc0 * px], [r0 * px], marker="s", ms=4, mfc="none", mec=C_NOISE)
    if edge is not None:
        for ax in (axh, axu, axd):
            ax.add_patch(Rectangle(((edge["col"] - 12) * px, edge["r0"] * px), 24 * px, (edge["r1"] - edge["r0"]) * px, fill=False, ec=C_WALL, lw=1.6))
        axo.plot([edge["col"] * px], [edge["r0"] * px], marker="s", ms=4, mfc="none", mec=C_WALL)
    if along is not None:
        rect = lambda: Rectangle((along["c0"] * px, (along["row"] - 12) * px), (along["c1"] - along["c0"] + 1) * px, 24 * px, fill=False, ec=C_ALONG, lw=1.6)
        if along["in_segment"]:
            for ax in (axh, axu, axd):
                ax.add_patch(rect())
        axo.plot([(along["c0"] + along["c1"]) / 2 * px], [along["row"] * px], marker="s", ms=4, mfc="none", mec=C_ALONG)
    axo.text(0.5, -0.02, "blue: segment shown\nred: noise patch\ngreen: edge, across the core\norange: edge, along the core",
             transform=axo.transAxes, va="top", ha="center", fontsize=8.5)
    # right: histogram + a few numbers
    rg = top[6].subgridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.55)
    axhs = fig.add_subplot(rg[0])
    d = (ours - his)[core & np.isfinite(ours)]
    axhs.hist(d, bins=np.linspace(-40, 40, 161), log=True, color="#4c72b0")
    if np.isfinite(sd_med):
        for s_ in (-sd_med, sd_med):
            axhs.axvline(s_, color="#c44e52", ls="--", lw=1.4)
        axhs.text(sd_med * 1.04, 0.93, "typical noise\n(1 SD)", transform=axhs.get_xaxis_transform(), color="#c44e52", fontsize=8.5, va="top")
    from matplotlib.ticker import NullFormatter
    axhs.yaxis.set_minor_formatter(NullFormatter())  # no 6x10^3-style labels between the decades
    axhs.set_xlim(-40, 40); axhs.set_xlabel(f"{PYSI} - {TOSH} (HU), pixel by pixel, inside the core"); axhs.set_ylabel("pixels (log scale)")
    axhs.set_title("how far apart are the two values?")
    axn = fig.add_subplot(rg[1]); axn.axis("off")
    lines = [
        ("pixels compared", f"{st['n']:,}"),
        ("bias (mean difference)", f"{st['mean']:+.2f} HU"),
        ("bias if rounded down to whole numbers", f"{st['mean_floor']:+.2f} HU"),
        ("spread (SD of the difference)", f"{st['sd']:.2f} HU"),
        ("90 % of pixels within", f"{st['p5']:+.1f} .. {st['p95']:+.1f} HU"),
        ("pixels beyond +-10 HU", f"{100 * st['frac_gt10']:.2f} %"),
    ]
    for i, (a_, b_) in enumerate(lines):
        y = 0.98 - i * 0.165
        axn.text(0.0, y, a_, fontsize=10.5, va="top", transform=axn.transAxes)
        axn.text(1.0, y, b_, fontsize=10.5, va="top", ha="right", fontweight="bold", transform=axn.transAxes)

    # ---- bottom row
    bot = outer[2].subgridspec(1, 4, width_ratios=[1.0, 1.65, 1.65, 1.75], wspace=0.34)
    # noise patch
    g1 = bot[0].subgridspec(1, 2, wspace=0.12)
    if noise is not None:
        r0, cc0 = noise["block"]
        pr0, pr1 = max(r0 - 8, 0), min(r0 + cm.BLOCK + 8, his.shape[0])
        pc0, pc1 = max(cc0 - 8, 0), min(cc0 + cm.BLOCK + 8, his.shape[1])
        ctr = float(np.median(his[r0:r0 + cm.BLOCK, cc0:cc0 + cm.BLOCK]))
        w = 4 * noise["sd_his"]
        for j, (img, nm, sd) in enumerate(((his, TOSH, noise["sd_his"]), (ours, PYSI, noise["sd_ours"]))):
            ax = fig.add_subplot(g1[j])
            ax.imshow(img[pr0:pr1, pc0:pc1], cmap="gray", vmin=ctr - w, vmax=ctr + w, aspect="equal", interpolation="nearest",
                      extent=(pc0 * px, pc1 * px, pr1 * px, pr0 * px))
            ax.add_patch(Rectangle((cc0 * px, r0 * px), cm.BLOCK * px, cm.BLOCK * px, fill=False, ec=C_NOISE, lw=1.6))
            ax.set_title(f"{nm}\nSD {sd:.1f} HU", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ax.annotate(f"noise patch: flat sediment, {cm.BLOCK * px:.1f} mm\nsquare (red box); same grey window\n(+-4 SD) in both.  All {len(res['blocks'])} flat\n"
                    f"patches, median SD:\n{TOSH} {np.median(res['sd_his']):.1f} HU\n{PYSI} {np.median(res['sd_ours']):.1f} HU",
                    xy=(-1.0, -0.12), xycoords="axes fraction", fontsize=8.5, va="top", ha="left")
    else:
        ax = fig.add_subplot(g1[0]); ax.axis("off"); ax.text(0.0, 0.5, "no flat patch found")

    def edge_group(spec, ed, title, colour, crop, box, xlim, caption):
        g = spec.subgridspec(2, 3, width_ratios=[0.6, 0.6, 1.9], height_ratios=[1.0, 0.3], wspace=0.55, hspace=0.62)
        pr0_, pr1_, pc0_, pc1_ = crop
        for j, (img, nm) in enumerate(((his, "Toshiba"), (ours, "SimpleITK"))):
            ax_ = fig.add_subplot(g[0, j])
            ax_.imshow(img[pr0_:pr1_, pc0_:pc1_], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest",
                       extent=(pc0_ * px, pc1_ * px, pr1_ * px, pr0_ * px))
            ax_.add_patch(Rectangle((box[0] * px, box[1] * px), box[2] * px, box[3] * px, fill=False, ec=colour, lw=1.4))
            ax_.set_title(nm, fontsize=9); ax_.set_xticks([]); ax_.set_yticks([])
        ax_ = fig.add_subplot(g[0, 2])
        x = ed["offs"] * px
        ax_.plot(x, ed["his"], color="#222222", lw=2.2, label=f"{TOSH}  {ed['w_his'] * px:.2f} mm")
        ax_.plot(x, ed["ours"], color="#d95f02", lw=1.6, ls="--", label=f"{PYSI}  {ed['w_ours'] * px:.2f} mm")
        ax_.set_xlim(-xlim, xlim)
        top_ = max(1.15, 1.08 * float(np.nanmax(ed["his"])))
        ax_.set_ylim(min(-0.1, 1.05 * float(np.nanmin(ed["his"]))), top_ + 0.35)
        ax_.set_xlabel("distance across the edge (mm)")
        ax_.set_title(title, fontsize=9.5)
        ax_.legend(fontsize=8, loc="upper left", frameon=False)
        axc_ = fig.add_subplot(g[1, :]); axc_.axis("off")
        axc_.text(0.0, 1.0, caption, fontsize=8.5, va="top", ha="left", transform=axc_.transAxes)

    if edge is not None:
        edge_group(bot[1], edge, "sharpness across the core\n(side wall; rise 10-90 %)", C_WALL,
                   (edge["r0"] - 3, edge["r1"] + 4, edge["col"] - 14, edge["col"] + 15),
                   (edge["col"] - 12, edge["r0"], 24, edge["r1"] - edge["r0"]), 2.6,
                   f"green box: 20 profiles.  Whole wall ({edge['n_profiles']} profiles):\n"
                   f"{TOSH} {edge['all_w_his'] * px:.2f}  |  {PYSI} {edge['all_w_ours'] * px:.2f} mm")
    else:
        ax = fig.add_subplot(bot[1]); ax.axis("off"); ax.text(0.5, 0.5, "no side-wall edge found", ha="center", color="#777777")
    if along is not None:
        edge_group(bot[2], along, "sharpness along the core\n(edge of a void or gap; rise 10-90 %)", C_ALONG,
                   (along["row"] - 16, along["row"] + 17, max(along["c0"] - 4, (along["c0"] + along["c1"]) // 2 - 30),
                    min(along["c1"] + 5, (along["c0"] + along["c1"]) // 2 + 30)),  # a wide edge is shown over its middle 60 columns
                   (max(along["c0"], (along["c0"] + along["c1"]) // 2 - 28), along["row"] - 12,
                    min(along["c1"], (along["c0"] + along["c1"]) // 2 + 28) - max(along["c0"], (along["c0"] + along["c1"]) // 2 - 28) + 1, 24), 3.2,
                   (f"orange box: {along['n']} profiles across {(along['c1'] - along['c0'] + 1) * px:.0f} mm of the core"
                    + ("" if along["in_segment"] else "\n(this edge lies outside the segment shown above)")))
    else:
        ax = fig.add_subplot(bot[2]); ax.axis("off"); ax.text(0.5, 0.5, "no edge across the core\nin this plane", ha="center", color="#777777")
    # down-core profile
    g4 = bot[3].subgridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.12)
    axp = fig.add_subplot(g4[0]); axq = fig.add_subplot(g4[1], sharex=axp)
    ph, po = cm.downcore_profile(his, core), cm.downcore_profile(ours, core)
    y = np.arange(his.shape[0]) * px
    axp.plot(y, ph, color="#222222", lw=1.6, label=TOSH); axp.plot(y, po, color="#d95f02", lw=1.0, ls="--", label=PYSI)
    axp.set_ylabel("mean HU\nacross the core", fontsize=9)
    fin_p = ph[np.isfinite(ph)]
    axp.set_ylim(float(fin_p.min()) - 30, float(fin_p.max()) + 0.30 * float(fin_p.max() - fin_p.min()))
    axp.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper left")
    axp.set_title("down-core profile (mean of each image row, inside the core)", fontsize=9.5); plt.setp(axp.get_xticklabels(), visible=False)
    dp = po - ph
    axq.plot(y, dp, color="#4c72b0", lw=0.9); axq.axhline(0, color="#999999", lw=0.8)
    fin = dp[np.isfinite(dp)]
    lim = max(5.0, 1.1 * float(np.abs(fin).max())) if fin.size else 5.0
    axq.set_ylim(-lim, lim)
    axq.set_ylabel("SimpleITK -\nToshiba (HU)", fontsize=9); axq.set_xlabel("distance along the core (mm)")
    axq.set_xlim(rows[0] * px, rows[-1] * px)
    problems = layout_problems(fig)
    for pb in problems:
        print("   LAYOUT:", *(x.encode("ascii", "replace").decode() for x in pb))
    print(f"   layout check: {len(problems)} text problem(s)")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- driver


def run_example(args) -> None:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    geom = cm.read_axial_geometry(args.axial)
    series = []
    for spec in args.coronal:
        name, _, path = spec.partition("=") if "=" in spec else (Path(spec).name, "", spec)
        cor = cm.read_coronal(path)
        series.append((name, cor, cm.plane_window(cor, geom)))
    ymin = min(w[2] for _, _, w in series); ymax = max(w[3] for _, _, w in series)
    t0 = time.time()
    stack = cm.load_axial(geom, (ymin, ymax))
    print(f"[{args.label}] axial: {len(geom.paths)} slices at {geom.thickness:g} mm, px {geom.px_x:.3f}; rows {stack.rows}; "
          f"missing slices filled: {stack.filled_z or 'none'}  ({time.time() - t0:.0f} s)")
    for name, cor, (j_lo, j_hi, _y0, _y1) in series:
        t1 = time.time()
        thickness = args.thickness or cor.thickness or 2.0
        renderer = cm.PlaneRenderer(stack, cor, j_lo, j_hi)
        areas = [cm.core_area_estimate(renderer.his_plane(k), cor.px) for k in range(cor.n_planes)]
        # every plane that cuts the core: its spread of the difference (strip chart, and the median-plane rule)
        per = []
        amax = max(areas)
        for kk in range(cor.n_planes):
            if areas[kk] < 0.2 * amax:
                continue
            try:
                r = analyse(renderer, kk, thickness, with_fit=False, with_sweep=False)
            except RuntimeError:
                continue
            nr = float(np.median(r["sd_ours"] / r["sd_his"])) if r["blocks"] else None
            per.append(dict(k=int(kk + 1), sd=r["stats"]["sd"], mean=r["stats"]["mean"], corr=r["stats"]["corr"], noise_ratio=_f(nr),
                            noise_his=_f(np.median(r["sd_his"])) if r["blocks"] else None))
        pool = [p for p in per if areas[p["k"] - 1] >= 0.5 * amax]  # planes through the main body of the core, not its outer surface
        if args.plane_rule == "median" and pool:
            sds = np.array([p["sd"] for p in pool])
            k = pool[int(np.argmin(np.abs(sds - np.median(sds))))]["k"] - 1
            rule = f"the plane with the median spread among the {len(pool)} planes through the main body of the core"
        else:
            k = cm.pick_plane(areas)
            rule = "the middle plane among those that cut the core"
        print(f"[{args.label}/{name}] '{cor.description}': {cor.n_planes} planes, px {cor.px:.3f} (his/axial = {cor.px / geom.px_x:.2f}), "
              f"slab {thickness:g} mm; showing plane {k + 1} ({rule})")
        res = analyse(renderer, k, thickness)
        res["rule"] = rule
        res["segment"] = choose_segment(res["his"], res["core"], res["rows"], cor.px)
        noise, edge, along = choose_patches(res, cor.px)
        png = out / f"{args.label}__{name}.png"
        draw_figure(res, args.label, name, cor, stack, noise, edge, along, thickness, args.note, png)
        st = res["stats"]
        summary = dict(label=args.label, series=name, description=cor.description, note=args.note, plane=int(k + 1), n_planes=int(cor.n_planes),
                       px_his=cor.px, px_axial=geom.px_x, axial_thickness=geom.thickness, slab=thickness,
                       stats={a: _f(b) if isinstance(b, float) else b for a, b in st.items()},
                       fit={a: _f(b) for a, b in res["fit"].items()}, sweep={a: _f(b) for a, b in res["sweep"].items()},
                       grad_ratio=_f(res["grad_ratio"]), n_blocks=len(res["blocks"]),
                       noise_his=_f(np.median(res["sd_his"])) if res["blocks"] else None,
                       noise_ratio=_f(np.median(res["sd_ours"] / res["sd_his"])) if res["blocks"] else None,
                       edge_wall=None if edge is None else dict(side=edge["side"], w_his_mm=_f(edge["all_w_his"] * cor.px),
                                                                w_ours_mm=_f(edge["all_w_ours"] * cor.px), n=edge["n_profiles"]),
                       edge_along=None if along is None else dict(w_his_mm=_f(along["w_his"] * cor.px), w_ours_mm=_f(along["w_ours"] * cor.px), n=along["n"]),
                       png=png.name)
        summary["all_planes"] = per
        summary["plane_rule"] = rule
        (out / f"{args.label}__{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"   -> {png}   (bias {st['mean']:+.2f} HU, SD {st['sd']:.2f} HU, corr {st['corr']:.4f}, noise ratio {summary['noise_ratio'] or float('nan'):.3f}, "
              f"{time.time() - t1:.0f} s)")


def summarize(outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(outdir.glob("*__*.json"))]
    if not items:
        raise SystemExit(f"No *__*.json files in {outdir}")
    fmt = lambda v, f: "-" if v is None else format(v, f)
    rows = []
    for s in items:
        st, fit = s["stats"], s.get("fit", {})
        w = s.get("edge_wall"); a = s.get("edge_along")
        best = min(s["sweep"], key=s["sweep"].get) if s.get("sweep") else "-"
        rows.append([f"{s['label']} / {s['series']}", f"{s['plane']}/{s['n_planes']}", f"{s['px_his'] / s['px_axial']:.2f}x", f"{st['corr']:.4f}",
                     f"{st['mean']:+.2f}", f"{st.get('mean_floor', float('nan')):+.2f}", f"{st['sd']:.2f}", f"{st['p5']:+.1f}..{st['p95']:+.1f}", fmt(s.get("noise_ratio"), ".3f"),
                     "-" if not w else f"{w['w_ours_mm'] / w['w_his_mm']:.3f}", "-" if not a else f"{a['w_ours_mm'] / a['w_his_mm']:.2f}",
                     f"{fit.get('sx', float('nan')):+.2f}, {fit.get('sz', float('nan')):+.2f}", f"{fit.get('sy', float('nan')):+.3f}", best])
    head = ["example / series", "plane", "Toshiba px /\naxial px", "correlation", "bias\n(HU)", "bias, rounded\ndown (HU)", "SD of\ndiff (HU)", "90 % of\npixels (HU)", "noise ratio\nSimpleITK/Toshiba",
            "wall edge\nwidth ratio", "along-core\nedge ratio", "shift px\n(across, along)", "slab\noffset mm", "best\nslab"]
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh); wr.writerow([h.replace("\n", " ") for h in head]); wr.writerows(rows)
    allp = [s for s in items if s.get("all_planes")]
    fig = plt.figure(figsize=(17, 9.5), dpi=130)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.08, 0.45, 0.62], hspace=0.28, left=0.04, right=0.985, top=0.97, bottom=0.07)
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.text(0, 0.7, f"Summary: coronal MPR rebuilt from the axial stack ({PYSI}) vs the {TOSH} scanner's coronal", fontsize=15, fontweight="bold", va="center")
    ax.text(0, 0.05, "Differences exist. The reason is not established (the scanner software is not available to inspect). "
                     "These are their sizes, measured inside the core on the plane of median spread.", fontsize=11, va="center", color="#333333")
    ax = fig.add_subplot(gs[1]); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=head, loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 2.3)
    for (r_, c_), cell in tb.get_celld().items():
        if r_ == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#e8edf3")
        if c_ == 0:
            cell.set_text_props(ha="left"); cell._loc = "left"
    tb.auto_set_column_width(list(range(len(head))))
    axs = fig.add_subplot(gs[2])
    if allp:
        for i, s in enumerate(allp):
            pl = s["all_planes"]
            line, = axs.plot([p["k"] for p in pl], [p["sd"] for p in pl], marker="o", ms=3.5, lw=1.2,
                             label=f"{s['label']} / {s['series']}  ({TOSH} noise ~{np.nanmedian([p['noise_his'] or np.nan for p in pl]):.0f} HU)")
            shown = [p for p in pl if p["k"] == s["plane"]]
            if shown:  # a ring marks the plane drawn in that series' figure
                axs.plot([shown[0]["k"]], [shown[0]["sd"]], marker="o", ms=12, mfc="none", mec=line.get_color(), mew=1.8, ls="")
        axs.set_xlabel("plane number (2 mm apart, through the core)"); axs.set_ylabel(f"SD of ({PYSI} - {TOSH}) (HU)")
        axs.set_title("every plane that cuts the core: spread of the difference (ring = the plane shown in that figure)", fontsize=11)
        axs.legend(fontsize=9, frameon=False); axs.set_ylim(bottom=0); axs.grid(alpha=0.25)
    else:
        axs.axis("off"); axs.text(0.5, 0.5, "(run with --all-planes to add the per-plane chart)", ha="center", color="#777777")
    fig.savefig(outdir / "summary.png", dpi=130)
    plt.close(fig)
    print(f"wrote {outdir / 'summary.png'} and summary.csv  ({len(items)} series)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label"); ap.add_argument("--axial"); ap.add_argument("--coronal", action="append", default=[])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "coronal_compare_output"),
                    help="output folder (default: tools/coronal_compare_output, which is not tracked by git)")
    ap.add_argument("--note", default="", help="text added to the figure subtitle")
    ap.add_argument("--thickness", type=float, default=0.0, help="slab thickness in mm (default: the coronal's SliceThickness)")
    ap.add_argument("--all-planes", action="store_true", help="(kept for old command lines: every plane is always measured now)")
    ap.add_argument("--plane-rule", choices=("median", "middle"), default="median",
                    help="which plane the figure shows: the one with the median spread of the difference (default) or the middle one")
    ap.add_argument("--summarize", metavar="OUTDIR", help="build summary.png / summary.csv from the JSON files in OUTDIR")
    args = ap.parse_args()
    if args.summarize:
        summarize(Path(args.summarize)); return
    if not (args.label and args.axial and args.coronal):
        ap.error("--label, --axial and --coronal are required (or use --summarize)")
    run_example(args)


if __name__ == "__main__":
    main()
