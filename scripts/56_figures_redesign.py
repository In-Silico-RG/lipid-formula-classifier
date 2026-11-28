#!/usr/bin/env python
"""
56_figures_redesign.py — redesigned figures of the paper (Figs. 1-8).

Reads only files already in results/ (nothing is trained or recomputed). Every number drawn is either
read from a result file or derived from it by arithmetic stated in the code (sums, differences,
counts rebuilt from row percentages and entry numbers, Wilson intervals from a proportion and n).

One colour per lipid category in every figure (colour-blind-safe, Okabe-Ito based); the three
categories that exchange most errors (FA, PR, ST) share warm tones.

Usage:  python 56_figures_redesign.py            (all figures)
        python 56_figures_redesign.py 1 4 7      (selected figures)
Output: figures/v2_fig<k>_*.pdf / .png
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RES, FIG = BASE / "results", BASE / "figures"
SET, MODEL = "R2C", "RF-weighted"

INK, MUTED, GRID, LIGHT = "#000000", "#000000", "#ececec", "#c8c8c8"
# CARTO "Bold": intense, mutually harmonious hues
BLUE, CYAN, GREEN, YELLOW = "#3969AC", "#008695", "#11A579", "#F2B701"
FUCHSIA, ORANGE, RED, PURPLE = "#CF1C90", "#E68310", "#E73F74", "#7F3C8D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK, "text.color": INK,
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
                     "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left"})

# one colour per category, the same in every figure
CAT = ["Glycerophospholipids", "Sphingolipids", "Glycerolipids", "Saccharolipids", "Polyketides",
       "Fatty Acyls", "Prenol Lipids", "Sterol Lipids"]
CODE = dict(zip(CAT, ["GP", "SP", "GL", "SL", "PK", "FA", "PR", "ST"]))
COL = dict(zip(CAT, [BLUE, CYAN, GREEN, YELLOW, FUCHSIA, ORANGE, RED, PURPLE]))
NICE = {c: c.replace(" Lipids", " lipids").replace(" Acyls", " acyls") for c in CAT}
FINAL = BLUE
import matplotlib.patheffects as pe  # noqa: E402
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
LBOX = dict(boxstyle="round,pad=0.18,rounding_size=0.2", fc="white", ec="none", alpha=0.92)
EDGE = [pe.Stroke(linewidth=2.8, foreground=INK), pe.Normal()]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.pdf")


def wilson(p, n, z=1.96):
    """95 % Wilson interval for a proportion p (0-1) observed on n units, in %."""
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def tag(ax, k, x=-0.02, y=1.02):
    ax.text(x, y, f"({'abcdefgh'[k]})", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            color=INK, ha="right", va="bottom")


def models(task):
    a = pd.read_csv(RES / f"40_models_{SET}_{task}.csv")
    b = pd.read_csv(RES / f"43_networks_{SET}_{task}.csv")
    return pd.concat([a, b]).drop_duplicates("method", keep="last").set_index("method")


# ----------------------------------------------------------------------------------------- Fig. 1
def fig1():
    """(a) what leakage is worth: the five seeds of each split; (b) where every 100 entries go."""
    lk = pd.read_csv(RES / "20_entry_lookup.csv")
    mA, mC = models("A"), models("C")
    ceil = pd.read_csv(RES / "08_ceiling_extended.csv")
    lim = dict(zip(["Category", "Main class", "Subclass"], ceil.techo_estricto))
    st = pd.read_csv(RES / f"50_structure_{SET}_{MODEL}.csv").structure.mean()
    fig = plt.figure(figsize=(3.45, 4.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.6], hspace=0.62, left=0.29, right=0.84, top=0.95, bottom=0.17)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    # (a)
    rows = [("Split by entry", lk.by_entry_pct), ("Split by formula\n(used in this work)", lk.by_formula_pct)]
    for yv, (nm, v) in zip([1, 0], rows):
        a.scatter(v, np.full(len(v), yv), s=24, color=BLUE, alpha=0.9, ec="white", lw=0.6, zorder=3)
        a.plot([v.mean()] * 2, [yv - 0.25, yv + 0.25], color=INK, lw=1.6, zorder=4)
        a.text(v.mean(), yv + 0.30, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    maj = mA.at["majority", "accuracy"]
    a.axvline(maj, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    a.text(maj + 2, -0.55, f"always the most common category: {maj:.1f}", ha="left", va="center", fontsize=5.9, color=MUTED)
    e, f = lk.by_entry_pct.mean(), lk.by_formula_pct.mean()
    a.annotate("", xy=(f + 6, 0.62), xytext=(e - 3, 0.62),
               arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1, mutation_scale=9))
    a.text(66, 0.52, f"\u2212{e - f:.0f} points from leakage", ha="center", va="top",
           fontsize=6.4, color=INK, fontweight="bold")
    a.set_yticks([1, 0], [r[0] for r in rows], fontsize=7)
    a.set_ylim(-0.8, 1.7)
    a.set_xlim(0, 100)
    a.grid(axis="x", color=GRID)
    a.tick_params(axis="y", length=0, labelcolor=INK)
    a.set_xlabel("accuracy of a look-up table (%), five seeds")
    tag(a, 0, x=-0.05, y=1.02)
    # (b) stacked: correct / accepted by LIPID MAPS / genuine errors
    parts = []
    for lvl, m in (("Category", mA.loc[MODEL]), ("Main class", mC.loc[MODEL])):
        parts.append((lvl, m.accuracy, m.permissive_accuracy - m.accuracy, 100 - m.permissive_accuracy))
    cols = [BLUE, "#A9C0E4", RED]
    txtc = [INK, INK, INK]
    labs = ["correct", "\u201cerror\u201d that LIPID MAPS accepts\n(the formula is ambiguous)", "genuine error"]
    X0 = 75
    for yv, (lvl, c, acc, err) in zip([2, 1], parts):
        left = 0
        for w, col in zip((c, acc, err), cols):
            b.barh(yv, w, left=left, height=0.5, color=col, lw=0)
            left += w
        b.text((X0 + c) / 2, yv, f"{c:.2f} correct" if c - X0 > 14 else f"{c:.2f}", ha="center", va="center", color=INK, fontsize=6.8,
               fontweight="bold", bbox=LBOX)
        for dy_, val, col_ in ((0.12, acc, "#A9C0E4"), (-0.14, err, RED)):
            b.plot(100.9, yv + dy_, "s", ms=4.2, mfc=col_, mec="none", clip_on=False, zorder=7)
            b.text(101.6, yv + dy_, f"{val:.2f}", ha="left", va="center", fontsize=6.5, color=INK, fontweight="bold")
        L = lim[lvl]
        b.plot([L, L], [yv - 0.36, yv + 0.36], color=INK, lw=1.4, zorder=5)
        b.text(L, yv + 0.40, f"limit {L:.2f}", ha="center", va="bottom", fontsize=6.2, color=INK)
    b.plot([lim["Subclass"]] * 2, [-0.36, 0.36], color=INK, lw=1.4)
    b.text(lim["Subclass"], 0.40, f"limit {lim['Subclass']:.2f}", ha="center", va="bottom", fontsize=6.2, color=INK)
    b.text(lim["Subclass"] + 1, 0, "no model trained", ha="left", va="center", fontsize=6.2, color=MUTED, style="italic")
    b.plot(st, 2, marker="D", ms=5, color="white", mec=INK, mew=1.1, zorder=6, clip_on=False)
    b.annotate(f"structure seen\n{st:.2f}", (st, 2.25), xytext=(st, 2.95), ha="center", va="bottom", fontsize=6.0,
               color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    b.set_yticks([2, 1, 0], ["Category\n(8 labels)", "Main class\n(86 labels)", "Subclass\n(299 labels)"], fontsize=7)
    b.set_xlim(X0, 100)
    b.set_ylim(-0.6, 3.5)
    b.grid(axis="x", color=GRID)
    b.tick_params(axis="y", length=0, labelcolor=INK)
    b.set_xlabel("entries (%), final model, out-of-fold")
    h = [Patch(color=c, label=l) for c, l in zip(cols, labs)] + \
        [Line2D([], [], color=INK, lw=1.4, label="upper limit of the\nformula (strict)")]
    b.legend(handles=h, loc="upper center", bbox_to_anchor=(0.40, -0.20), ncol=2, fontsize=6.0,
             handlelength=1.2, columnspacing=0.8)
    tag(b, 1, x=-0.05, y=1.02)
    save(fig, "v2_fig1_leakage_and_limit")


# ----------------------------------------------------------------------------------------- Fig. 2
def fig2():
    """(a) the paired gain from the two carbon-number descriptors; (b) the selection rule as a table."""
    t = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("30_feature_selection_*.csv"))
                   if p.name != "30_feature_selection_all.csv"]).drop_duplicates(["model", "task", "set"], keep="last")
    t["seeds"] = t.macro_f1_per_seed.astype(str).str.split().apply(lambda v: np.array(v, float))
    t = t.set_index(["model", "task", "set"])
    MOD = [("rf", "Random forest"), ("xgb", "Boosted trees"), ("knn", "Nearest neighbours")]
    TASK = [("category", "Category"), ("main_class", "Main class")]
    fig = plt.figure(figsize=(7.0, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.30, left=0.235, right=0.985, top=0.86, bottom=0.17)
    for k, (task, tname) in enumerate(TASK):
        ax = fig.add_subplot(gs[0, k])
        y, labels = [], []
        yy = 0
        for m, mname in MOD:
            for frm, to in (("T10", "T12"), ("R2", "R2C")):
                d = t.loc[(m, task, to), "seeds"] - t.loc[(m, task, frm), "seeds"]
                col = FINAL if d.mean() > 0 else RED
                ax.plot([d.min(), d.max()], [yy, yy], color=col, lw=2.6, alpha=0.22, solid_capstyle="round")
                ax.scatter(d, np.full(3, yy), s=12, color=col, alpha=0.55, lw=0, zorder=3)
                ax.plot(d.mean(), yy, "o", ms=5.5, color=col, mec="white", mew=0.6, zorder=4)
                ax.text(d.mean(), yy + 0.36, f"{d.mean():+.1f}".replace("-", "\u2212"), ha="center", va="bottom", fontsize=6.2, color=INK)
                y.append(yy)
                labels.append(f"{mname}: {frm}→{to}" if k == 0 else "")
                yy -= 1
            yy -= 0.5
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y, labels, fontsize=6.8)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
        ax.set_ylim(min(y) - 0.7, 0.9)
        ax.grid(axis="x", color=GRID)
        ax.set_xlabel(f"change in macro F1 (points)\n{tname.lower()}")
        tag(ax, k, x=-0.02 if k else -0.02, y=1.06)
    # (c) heat table of the selection rule
    ax = fig.add_subplot(gs[0, 2])
    order = ["T10", "T12", "R1", "R2", "R2C", "R5"]
    nvar = {"R1": 13, "T10": 10, "T12": 12, "R2": 22, "R2C": 24, "R5": 45}
    colsk = [(m, task) for task, _ in TASK for m, _ in MOD]
    M = np.zeros((len(order), len(colsk)))
    inside = np.zeros_like(M, bool)
    for j, (m, task) in enumerate(colsk):
        r = t.xs((m, task), level=(0, 1)).reindex(order)
        best = r.macro_f1.idxmax()
        sd = r.macro_f1_sd[best]
        M[:, j] = (r.macro_f1[best] - r.macro_f1) / sd
        inside[:, j] = r.macro_f1 >= r.macro_f1[best] - sd
        for i, s in enumerate(order):
            pass
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("d", ["#7FA3D6", "#D3DFF1", "#ffffff"])
    ax.imshow(np.clip(M, 0, 6), cmap=cmap, vmin=0, vmax=6, aspect="auto")
    for j, (m, task) in enumerate(colsk):
        r = t.xs((m, task), level=(0, 1)).reindex(order)
        for i, s in enumerate(order):
            v = r.macro_f1[s]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.2,
                    color=INK, fontweight="bold" if inside[i, j] else "normal")
            if inside[i, j]:
                ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec=BLUE, lw=1.3))
    ax.set_xticks(range(6), ["RF", "XGB", "kNN"] * 2, fontsize=6.8)
    ax.set_yticks(range(len(order)), [f"{s} ({nvar[s]})" for s in order], fontsize=6.8)
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("R2C"):
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.tick_params(length=0, labelcolor=INK)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(2.5, color="white", lw=3)
    ax.text(1, -1.25, "Category", ha="center", fontsize=7, fontweight="bold", color=INK)
    ax.text(4, -1.25, "Main class", ha="center", fontsize=7, fontweight="bold", color=INK)
    ax.set_xlabel("macro F1 (%); framed: within one SD of the best list", fontsize=6.6)
    tag(ax, 2, x=-0.02, y=1.10)
    save(fig, "v2_fig2_carbon_descriptors_and_rule")


# ----------------------------------------------------------------------------------------- Fig. 3
FAM = {"trees": (BLUE, ["RF", "XGB"]), "networks": (FUCHSIA, ["MLP(R2C)", "CNN(R4)"]),
       "nearest neighbours": (GREEN, ["kNN-1", "kNN-5"])}
SHORT = {"RF": "RF", "RF-weighted": "RF weighted\n(final model)", "XGB": "XGB", "XGB-weighted": "XGB weighted",
         "MLP(R2C)": "MLP", "CNN(R4)": "CNN", "kNN-1": "kNN, k=1", "kNN-5": "kNN, k=5"}
FAMCOL = {m: c for c, ms in FAM.values() for m in ms}
FAMCOL.update({m + "-weighted": FAMCOL[m] for m in ("RF", "XGB", "MLP(R2C)", "CNN(R4)")})


def fig3():
    """Accuracy against macro F1: what class weighting trades (arrows); lower row magnifies the best models."""
    lim = {"A": 97.458, "C": 91.18}
    win = {"A": ((79, 98.5), (74, 96)), "B": ((88.5, 97.5), (76, 93)), "C": ((40, 92), (18, 62))}
    zoom = {"A": ((93.55, 95.35), (91.5, 93.55)), "B": ((96.1, 96.9), (90.25, 91.35)), "C": ((83.6, 85.7), (54.2, 58.4))}
    main_off = {"A": {"kNN-1": (5, 0, "left")}, "B": {"kNN-5": (5, 0, "left"), "kNN-1": (-5, 0, "right"), "CNN(R4)": (-5, 2, "right")},
                "C": {"MLP(R2C)": (-5, 0, "right")}}
    zoff = {"A": {"RF": (-6, 0, "right"), "RF-weighted": (-7, 4, "right"), "XGB": (6, 0, "left"),
                  "XGB-weighted": (6, 0, "left"), "MLP(R2C)": (6, 0, "left")},
            "B": {"RF": (-6, 5, "right"), "RF-weighted": (8, -6, "left"), "XGB": (-6, 0, "right"),
                  "XGB-weighted": (6, 0, "left")},
            "C": {"RF": (-6, 0, "right"), "RF-weighted": (7, 0, "left"), "XGB": (6, 0, "left")}}
    name = {"A": "Category", "B": "Lipid or not", "C": "Main class"}
    show = ["RF", "RF-weighted", "XGB", "XGB-weighted", "MLP(R2C)", "MLP(R2C)-weighted", "CNN(R4)",
            "CNN(R4)-weighted", "kNN-1", "kNN-5"]
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.42, wspace=0.28, left=0.08, right=0.98,
                          top=0.95, bottom=0.12)

    def dots(ax, s, big=7.5, small=5):
        for m in show:
            if m not in s.index:
                continue
            x, y = s.at[m, "accuracy"], s.at[m, "macro_f1"]
            col, w, fin = FAMCOL[m], m.endswith("-weighted"), m == MODEL
            ax.plot(x, y, "s" if w else "o", ms=(big if fin else small) * (0.92 if w else 1), mfc=col,
                    mec=INK if fin else "white", mew=1.2 if fin else 0.7, zorder=6 if fin else 4)
            if not w and m + "-weighted" in s.index:
                x2, y2 = s.at[m + "-weighted", "accuracy"], s.at[m + "-weighted", "macro_f1"]
                ax.annotate("", xy=(x2, y2), xytext=(x, y), arrowprops=dict(
                    arrowstyle="-|>", color=col, lw=0.9, alpha=0.55, mutation_scale=7, shrinkA=3, shrinkB=4), zorder=3)

    def label(ax, s, m, dx, dy, ha):
        fin = m == MODEL
        ax.annotate(SHORT[m], (s.at[m, "accuracy"], s.at[m, "macro_f1"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=6, color=INK, fontweight="bold" if fin else "normal")

    for k, task in enumerate("ABC"):
        s = models(task)
        ax = fig.add_subplot(gs[0, k])
        (zx, zy) = zoom[task]
        dots(ax, s)
        for m in show:
            if m.endswith("-weighted") or m not in s.index:
                continue
            x, y = s.at[m, "accuracy"], s.at[m, "macro_f1"]
            if zx[0] <= x <= zx[1] and zy[0] <= y <= zy[1]:
                continue
            label(ax, s, m, *main_off[task].get(m, (-5, 0, "right")))
        ax.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec="#555555", lw=0.7, ls=(0, (2, 1.5)), zorder=7))
        tx, ty, ha_, va_, ox, oy = {"A": (zx[0], zy[1], "right", "bottom", -2, 1),
                                    "B": (zx[0], zy[0], "right", "top", -2, -2),
                                    "C": (zx[1], zy[0], "left", "top", 2, -2)}[task]
        ax.annotate(f"({'def'[k]})", (tx, ty), xytext=(ox, oy), textcoords="offset points", ha=ha_,
                    va=va_, fontsize=6.5, color=INK, fontweight="bold")
        if task in lim:
            ax.axvline(lim[task], color=INK, lw=0.9, ls=(0, (3, 2)))
            ax.text(lim[task] - 0.3, win[task][1][0] + 0.02 * (win[task][1][1] - win[task][1][0]),
                    "upper limit of the formula", rotation=90, ha="right", va="bottom", fontsize=5.8, color=INK)
        ax.set_xlim(*win[task][0])
        ax.set_ylim(*win[task][1])
        ax.grid(color=GRID)
        ax.set_xlabel(f"accuracy (%), {name[task].lower()}")
        if k == 0:
            ax.set_ylabel("macro F1 (%)")
        tag(ax, k, y=1.0)
        # lower row: the magnified box
        z = fig.add_subplot(gs[1, k])
        dots(z, s, big=8, small=6)
        for m, (dx, dy, ha) in zoff[task].items():
            if m in s.index:
                x, y = s.at[m, "accuracy"], s.at[m, "macro_f1"]
                if zx[0] <= x <= zx[1] and zy[0] <= y <= zy[1]:
                    label(z, s, m, dx, dy, ha)
        z.set_xlim(*zx)
        z.set_ylim(*zy)
        z.grid(color=GRID)
        z.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            z.spines[sp].set_visible(True)
        for sp in z.spines.values():
            sp.set_color("#555555")
            sp.set_linewidth(0.7)
            sp.set_linestyle((0, (2, 1.5)))
        z.set_xlabel(f"accuracy (%), {name[task].lower()}")
        if k == 0:
            z.set_ylabel("macro F1 (%)")
        tag(z, 3 + k, y=1.0)
    h = [Line2D([], [], marker="o", ls="", ms=5, mfc=c, mec="white", label=f) for f, (c, _) in FAM.items()] + \
        [Line2D([], [], marker="s", ls="", ms=5, mfc="#9a9a9a", mec="white", label="with class weights (arrow from the circle)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.5, handletextpad=0.2,
               columnspacing=1.2)
    save(fig, "v2_fig3_accuracy_vs_macroF1")


# ----------------------------------------------------------------------------------------- Fig. 4
def ribbon(ax, y0l, y1l, y0r, y1r, xl, xr, color, alpha=0.55):
    xm = (xl + xr) / 2
    verts = [(xl, y0l), (xm, y0l), (xm, y0r), (xr, y0r), (xr, y1r), (xm, y1r), (xm, y1l), (xl, y1l), (xl, y0l)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4,
             MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", alpha=alpha))


def fig4():
    """Where the errors go: alluvial of the misassigned entries (seed 0), true -> predicted."""
    real = RES / "95_confusion_counts_category_seed0.csv"      # real counts (peer review), seed 0 out-of-fold
    if real.exists():
        full_cnt = pd.read_csv(real, index_col=0).loc[CAT, CAT].to_numpy().astype(int)
    else:                                                       # fallback: counts rebuilt from the percentages of Table S6
        c = pd.read_csv(RES / f"41_per_class_{SET}_{MODEL}_confusion.csv", index_col=0).loc[CAT, CAT]
        n = pd.read_csv(RES / f"41_per_class_{SET}_{MODEL}_categories.csv").set_index("category").n_mean.reindex(CAT)
        full_cnt = np.rint(c.to_numpy() / 100 * n.to_numpy()[:, None]).astype(int)
    cnt = full_cnt.copy()
    np.fill_diagonal(cnt, 0)
    tot = cnt.sum()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.6), width_ratios=[2.3, 1], layout="constrained")
    gap, XL, XR, W = 48, 0.0, 1.0, 0.06                  # V_15: more space between blocks
    out, inn = cnt.sum(1), cnt.sum(0)

    def stack(vals):
        pos, y = {}, 0
        for i, v in enumerate(vals):
            pos[i] = (y, y + v)
            y += v + gap
        return pos
    L, R = stack(out), stack(inn)
    lcur = {i: L[i][0] for i in L}
    rcur = {j: R[j][0] for j in R}
    for i in range(8):                     # ribbons, coloured by the true category
        for j in range(8):
            v = cnt[i, j]
            if v == 0:
                continue
            ribbon(ax, lcur[i], lcur[i] + v, rcur[j], rcur[j] + v, XL + W, XR - W, COL[CAT[i]],
                   alpha=0.55 if v >= 50 else 0.25)
            lcur[i] += v
            rcur[j] += v
    for i in range(8):
        for pos, x, ha, v in ((L, XL, "right", out[i]), (R, XR - W, "left", inn[i])):
            y0, y1 = pos[i]
            ax.add_patch(Rectangle((x, y0), W, max(y1 - y0, 2), color=COL[CAT[i]], lw=0))
            if v >= 1:                                   # V_15: every category labelled
                xt = x - 0.015 if ha == "right" else x + W + 0.015
                ax.text(xt, (y0 + y1) / 2, f"{CODE[CAT[i]]}  {v}" if ha == "right" else f"{v}  {CODE[CAT[i]]}",
                        ha=ha, va="center", fontsize=6.8, color=INK)
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(max(L[7][1], R[7][1]) + 30, -90)
    ax.axis("off")
    ax.text(XL + W / 2, -45, "true category", ha="center", fontsize=7, color=INK, fontweight="bold")
    ax.text(XR - W / 2, -45, "assigned category", ha="center", fontsize=7, color=INK, fontweight="bold")
    pr, prf, prs = int(cnt[6].sum()), int(cnt[6, 5]), int(cnt[6, 7])
    # V_15: the summary sentence is in the caption, not inside the plot
    tag(ax, 0, x=0.0, y=1.0)
    # (b) sensitivity and precision per category, same repetition
    diag = 100 * full_cnt.diagonal() / full_cnt.sum(1)        # sensitivity
    prec = 100 * full_cnt.diagonal() / full_cnt.sum(0)
    y = np.arange(8)[::-1]
    for yi, cat, s_, p_ in zip(y, CAT, diag, prec):
        bx.plot([s_, p_], [yi, yi], color=COL[cat], lw=2.6, alpha=0.22, solid_capstyle="round", zorder=1)
        bx.plot(s_, yi, "o", ms=5.5, color=COL[cat], mec="white", mew=0.6, zorder=3)
        bx.plot(p_, yi, "s", ms=5, mfc="white", mec=COL[cat], mew=1.5, zorder=3)
    bx.set_yticks(y, [CODE[c_] for c_ in CAT], fontsize=7.2)                # V_14: codes as in panel (a)
    bx.tick_params(axis="y", length=0, labelcolor=INK)
    bx.set_xlim(62, 102.5)
    bx.grid(axis="x", color=GRID)
    bx.set_xlabel("sensitivity or precision (%)")
    bx.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=MUTED, label="sensitivity"),
                       Line2D([], [], marker="s", ls="", ms=5, mfc="white", mec=MUTED, mew=1.2, label="precision")],
              loc="upper left", fontsize=6.3, handletextpad=0.2)
    tag(bx, 1, x=-0.02, y=1.0)
    save(fig, "v2_fig4_error_flow")


# ----------------------------------------------------------------------------------------- Fig. 5
def fig5():
    """Independent data: overall result with 95 % intervals, what each set contains, and per category."""
    t = pd.read_csv(RES / f"44_external_{SET}_{MODEL}.csv").set_index(["set", "task", "scenario"])
    p = pd.read_csv(RES / f"44_external_{SET}_{MODEL}_predictions.csv")
    p = p[p.level == "category"].copy()
    p["ok"] = [pr in tr.split(";") for pr, tr in zip(p.predicted, p.truth)]
    p["cat"] = p.truth.str.split(";").str[0]
    cvA = models("A").at[MODEL, "accuracy"]
    cvC = models("C").at[MODEL, "accuracy"]
    e = pd.read_csv(RES / f"51_error_analysis_{SET}_{MODEL}.csv").set_index("task").loc["B"]
    NL, LK = "no leakage", "leakage (all trained)"
    GREYD = "#8a8a8a"
    fig = plt.figure(figsize=(6.72, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.85], hspace=0.40, wspace=0.85,
                          left=0.18, right=0.95, top=0.94, bottom=0.14)

    def ci(ax, y, v, n, col=BLUE):
        lo, hi = wilson(v / 100, n)
        ax.plot([lo, hi], [y, y], color=col, lw=3.2, alpha=0.30, solid_capstyle="round", zorder=2)
        ax.plot(v, y, "o", ms=5.8, color=col, mec="white", mew=0.8, zorder=4)

    def clean(ax, xlim, xlabel):
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xlabel)

    # (a) overall
    a = fig.add_subplot(gs[0, 0])
    rows = [("GNPS, category", ("GNPS", "category"), cvA), ("MassBank, category", ("MassBank", "category"), cvA),
            ("NIST plasma, category", ("NIST", "category"), cvA), ("NIST plasma, main class", ("NIST", "main_class"), cvC)]
    for yv, (lab, key, cv) in zip(range(4)[::-1], rows):
        v, nn = t.at[(*key, NL), "score"], t.at[(*key, NL), "n"]
        a.plot([cv, cv], [yv - 0.22, yv + 0.22], color=INK, lw=1.0, zorder=3)
        if (*key, LK) in t.index:
            a.plot(t.at[(*key, LK), "score"], yv, "o", ms=5, mfc="white", mec=GREYD, mew=1.1, zorder=3)
        ci(a, yv, v, nn)
        lo = wilson(v / 100, nn)[0]
        anchor = cv if cv < lo < cv + 4 else lo          # number left of the interval, clear of the CV mark
        a.annotate(f"{v:.1f}", (anchor, yv), xytext=(-4, 0), textcoords="offset points", ha="right",
                   va="center", fontsize=6.4)
    a.set_yticks(range(4)[::-1], [r[0] for r in rows], fontsize=6.8)
    a.set_ylim(-0.5, 3.55)
    clean(a, (78, 101), "correct (%)")
    a.set_xticks([80, 85, 90, 95, 100])
    a.spines["bottom"].set_bounds(78, 100)
    tag(a, 0, x=-0.62, y=1.0)
    # (b) detection
    b = fig.add_subplot(gs[0, 1])
    D, R = "detection: lipids found", "detection: non-lipids rejected"
    rows = [("cross-validation", e.sensitivity, None), ("GNPS", *t.loc[("GNPS", D, NL), ["score", "n"]]),
            ("MassBank", *t.loc[("MassBank", D, NL), ["score", "n"]]), None,
            ("cross-validation", e.specificity, None), ("MassBank", *t.loc[("MassBank", R, NL), ["score", "n"]]),
            ("HMDB", *t.loc[("HMDB", R, NL), ["score", "n"]])]
    for yh, txt in ((7.0, "Lipid formulas recognised"), (3.0, "Non-lipid formulas rejected")):
        b.annotate(txt, (0, yh), xycoords=("axes fraction", "data"), xytext=(-4, 0), textcoords="offset points",
                   ha="right", va="center", fontsize=6.6, fontweight="bold", annotation_clip=False)
    ys, labs = [], []
    for yv, r in zip(range(len(rows))[::-1], rows):
        if r is None:
            continue
        nm, v, nn = r
        if nn is None:
            b.plot([v, v], [yv - 0.22, yv + 0.22], color=INK, lw=1.0, zorder=3)
            lo = v
        else:
            ci(b, yv, v, nn)
            lo = wilson(v / 100, nn)[0]
        b.annotate(f"{v:.1f}", (lo, yv), xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                   fontsize=6.4)
        ys.append(yv); labs.append(nm)
    b.set_yticks(ys, labs, fontsize=6.8)
    b.set_ylim(-0.6, 7.5)
    clean(b, (42, 101), "formulas (%)")
    b.set_xticks([50, 60, 70, 80, 90, 100])
    b.spines["bottom"].set_bounds(42, 100)
    tag(b, 1, x=-0.62, y=1.0)
    # shared legend for (a) and (b)
    fig.legend(handles=[Line2D([], [], marker="o", ls="-", lw=3, ms=5.5, color=BLUE, alpha=0.9, mec="white",
                               label="formulas withheld, 95 % CI"),
                        Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=GREYD, mew=1.1, label="formulas left in training"),
                        Line2D([], [], marker="|", ls="", ms=8, mew=1.0, color=INK, label="cross-validation")],
               loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, fontsize=6.5, handletextpad=0.4, columnspacing=1.6)
    # (c) composition: one row per category (same rows as d), one bar per set, colour tab = category
    c = fig.add_subplot(gs[1, 0])
    sets = ["GNPS", "MassBank", "NIST"]
    SN = {"GNPS": "GNPS", "MassBank": "MassBank", "NIST": "NIST plasma"}
    shade = {"GNPS": "#3a3a3a", "MassBank": "#8c8c88", "NIST": "#c8c8c8"}
    nset = {s_: int((p.set == s_).sum()) for s_ in sets}
    h = 0.26
    for yi, cat in zip(range(8)[::-1], CAT):
        for k, s_ in enumerate(sets):
            yy = yi + (1 - k) * h
            v = 100 * (p[p.set == s_].cat == cat).sum() / nset[s_]
            c.barh(yy, v, height=h * 0.92, color=shade[s_], lw=0)
            c.text(v + 0.8, yy, f"{v:.1f}" if v >= 0.05 else "\u2013", va="center", fontsize=5.4)
        c.add_patch(Rectangle((-2.6, yi - 1.5 * h), 1.6, 3 * h, color=COL[cat], clip_on=False, lw=0))
    c.set_yticks(range(8)[::-1], [CODE[c_] for c_ in CAT], fontsize=6.8)
    c.tick_params(axis="y", length=0, pad=2)
    c.spines["left"].set_visible(False)
    c.set_xlim(-3, 58)
    c.spines["bottom"].set_bounds(0, 58)
    c.set_ylim(-0.6, 7.6)
    c.grid(axis="x", color=GRID, lw=0.6)
    c.grid(axis="y", visible=False)
    c.set_xlabel("lipid formulas of each set (%)")
    c.legend(handles=[Patch(color=shade[s_], label=f"{SN[s_]} (n = {nset[s_]})") for s_ in sets],
             loc="lower right", bbox_to_anchor=(1.02, 0.0), ncol=1, fontsize=5.9, handlelength=1.0,
             handletextpad=0.3)
    tag(c, 2, x=-0.62, y=1.0)
    # (d) per category, seed 0, categories with at least 10 formulas in a set
    d = fig.add_subplot(gs[1, 1])
    off = {"GNPS": 0.29, "MassBank": 0.0, "NIST": -0.29}
    mk = {"GNPS": "o", "MassBank": "s", "NIST": "^"}
    cvf = RES / "91_external_composition_per_category.csv"       # peer review: the cross-validation value of each category
    cv = {}
    if cvf.exists():
        q91 = pd.read_csv(cvf)
        cv = {str(r.category).lower(): r.CV_sensitivity_formula for r in q91.itertuples()}
    for yi, cat in zip(range(8)[::-1], CAT):
        if cat.lower() in cv and any((p[(p.set == s_) & (p.cat == cat)].shape[0] >= 10) for s_ in sets):
            d.plot([cv[cat.lower()]] * 2, [yi - 0.44, yi + 0.44], color=INK, lw=1.1, zorder=1.5)
        if yi % 2 == 0:
            d.axhspan(yi - 0.5, yi + 0.5, color="#f4f4f4", lw=0, zorder=0)
    for s_ in sets:
        q = p[p.set == s_]
        for yi, cat in zip(range(8)[::-1], CAT):
            qc = q[q.cat == cat]
            if len(qc) < 10:
                continue
            v = 100 * qc.ok.mean()
            lo, hi = wilson(qc.ok.mean(), len(qc))
            yy_ = yi + off[s_]                                   # V_14: thin solid interval with end caps, larger marker
            d.plot([lo, hi], [yy_] * 2, color=COL[cat], lw=1.2, alpha=0.85, solid_capstyle="butt", zorder=2)
            d.plot([lo, lo], [yy_ - 0.07, yy_ + 0.07], color=COL[cat], lw=1.0, alpha=0.85, zorder=2)
            d.plot([hi, hi], [yy_ - 0.07, yy_ + 0.07], color=COL[cat], lw=1.0, alpha=0.85, zorder=2)
            d.plot(v, yy_, mk[s_], ms=5.6, color=COL[cat], mec="white", mew=0.7, zorder=3)
    d.set_yticks(range(8)[::-1], [CODE[c_] for c_ in CAT], fontsize=6.8)
    d.set_ylim(-0.6, 7.6)
    clean(d, (30, 103), "correct category (%), 95 % CI")
    d.set_xticks([40, 60, 80, 100])
    d.legend(handles=[Line2D([], [], marker=mk[s_], ls="", ms=4.8, color="#555555", mec="white",
                             label=s_ if s_ != "NIST" else "NIST plasma") for s_ in sets]
                    + ([Line2D([], [], marker="|", ls="", ms=7, color=INK, label="cross-validation")] if cv else []),
             loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.2, handletextpad=0.2,
             columnspacing=1.0)
    tag(d, 3, x=-0.62, y=1.0)
    fig.legend(handles=[Patch(fc=COL[c_], ec="none", label=f"{CODE[c_]} {NICE[c_].lower()}") for c_ in CAT],
               loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=4, fontsize=6.0, handlelength=1.0,
               handletextpad=0.35, columnspacing=1.2)
    save(fig, "v2_fig5_independent_data")


# ----------------------------------------------------------------------------------------- Fig. 6
def fig6():
    """(a, b) coverage minus target for each category, one threshold vs one per category; (c) what a user gets."""
    t = pd.read_csv(RES / f"45_conformal_{SET}_{MODEL}_per_label.csv")
    s = pd.read_csv(RES / f"45_conformal_{SET}_{MODEL}.csv")
    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.62, 1.15], wspace=0.12, left=0.07, right=0.95, top=0.85, bottom=0.17)
    targets = [90, 95, 99]
    shared = None
    for k, (var, title) in enumerate((("marginal", "One threshold for all"), ("mondrian", "One per category"))):
        ax = fig.add_subplot(gs[0, k], sharey=shared)
        shared = shared or ax
        q = t[(t.task == "A") & (t.variant == var)]
        for cat in CAT:
            r = q[q.label == cat].set_index("alpha").coverage
            dev = [r[round(1 - tg / 100, 2)] - tg for tg in targets]
            lw = 1.8 if cat == "Prenol Lipids" else 1.0
            ax.plot(targets, dev, "-o", color=COL[cat], lw=lw, ms=3.6, mec="white", mew=0.5)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xticks(targets, [f"{x} %" for x in targets])
        ax.set_xlim(88.5, 100.5)
        ax.grid(color=GRID)
        ax.set_xlabel(f"target coverage\n{title.lower()}")
        if k == 0:
            ax.set_ylabel("observed minus target (points)")
        else:
            ax.tick_params(labelleft=False)
        tag(ax, k, y=1.0)
    pr = t[(t.task == "A") & (t.variant == "marginal") & (t.label == "Prenol Lipids") & (np.isclose(t.alpha, 0.05))].coverage.iloc[0]
    shared.annotate(f"prenol lipids:\n{pr:.1f} % at a\n95 % target", (95, pr - 95), xytext=(90.4, -21),
                    fontsize=6, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    h = [Line2D([], [], color=COL[c], lw=2.0, label=NICE[c]) for c in CAT]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.30, 1.08), ncol=4, fontsize=6.2, handlelength=1.4,
               columnspacing=0.9)
    # (c) what a user receives, as released: category and task B per category, main class one threshold
    ax = fig.add_subplot(gs[0, 3])
    use = [("A", "mondrian", "Category"), ("C", "marginal", "Main class"), ("B", "mondrian", "Lipid or not")]
    y, labs = [], []
    yy = 0
    for task, var, nm in use:
        for tg in targets:
            r = s[(s.task == task) & (s.variant == var) & np.isclose(s.alpha, round(1 - tg / 100, 2))].iloc[0]
            empty, single = r.empty_sets, r.single_label_sets
            multi = max(100 - empty - single, 0)
            left = 0
            for w, col in zip((empty, single, multi), (RED, FINAL, ORANGE)):
                ax.barh(yy, w, left=left, height=0.66, color=col, lw=0.8, ec="white")
                left += w
            ax.text(101, yy, f"{r.mean_size:.2f}", va="center", fontsize=6, color=INK)
            y.append(yy); labs.append(f"{nm}, {tg} %")
            yy -= 1
        yy -= 0.6
    ax.set_yticks(y, labs, fontsize=6.4)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlim(0, 100)
    ax.text(101, 0.9, "mean\nsize", fontsize=5.8, color=MUTED, va="bottom")
    ax.set_xlabel("prediction sets (%)")
    ax.legend(handles=[Patch(fc=RED, ec="none", label="empty"), Patch(fc=BLUE, ec="none", label="one label"),
                       Patch(fc=ORANGE, ec="none", label="two or more")],
              loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=3, fontsize=6.2, handlelength=1, columnspacing=0.8)
    tag(ax, 2, x=-0.52, y=1.0)
    save(fig, "v2_fig6_prediction_sets")


# ----------------------------------------------------------------------------------------- Fig. 7
FEAT = {"P_COUNT": ("P", True), "N_COUNT": ("N", True), "C_COUNT": ("C", True), "O_COUNT": ("O", True),
        "H_COUNT": ("H", True), "S_COUNT": ("S", True), "DBE": ("DBE", False), "H_C_RATIO": ("H/C", False),
        "O_C_RATIO": ("O/C", False), "N_C_RATIO": ("N/C", False), "DBE_PER_C": ("DBE/C", False),
        "C_MOD5": ("C mod 5", True)}


def parse_tree(path):
    """sklearn export_text -> nested dict {'feat','thr','left','right'} or {'leaf': class}."""
    lines = [l.rstrip() for l in path.read_text().splitlines() if l.strip().startswith("|")]
    pat = re.compile(r"^((?:\|   )*)\|--- (.*)$")

    def build(i, depth):
        m = pat.match(lines[i])
        body = m.group(2)
        if body.startswith("class:"):
            return {"leaf": body.split(":", 1)[1].strip()}, i + 1
        feat, op, thr = re.match(r"(\S+)\s*(<=|>)\s*(\S+)", body).groups()
        left, j = build(i + 1, depth + 1)
        assert pat.match(lines[j]).group(2).startswith(f"{feat} >")
        right, k = build(j + 1, depth + 1)
        return {"feat": feat, "thr": float(thr), "left": left, "right": right}, k
    root, _ = build(0, 0)
    return root


def cond(feat, thr, left):
    nm, integer = FEAT.get(feat, (feat, False))
    if integer:
        v = int(np.floor(thr))
        if left:
            return f"{nm} = 0" if v == 0 else f"{nm} ≤ {v}"
        return f"{nm} ≥ {v + 1}"
    return f"{nm} ≤ {thr:g}" if left else f"{nm} > {thr:g}"


def fig7():
    tree = parse_tree(RES / "47_rules_category.txt")
    leaves = []

    def layout(node, depth):
        if "leaf" in node:
            node["x"], node["y"] = len(leaves), depth
            leaves.append(node)
            return node["x"]
        xl = layout(node["left"], depth + 1)
        xr = layout(node["right"], depth + 1)
        node["x"], node["y"] = (xl + xr) / 2, depth
        return node["x"]
    layout(tree, 0)
    depth = max(l["y"] for l in leaves)
    fig = plt.figure(figsize=(7.0, 4.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.22, left=0.01, right=0.99, top=0.95, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    dy = 1.0

    def draw(node):
        x, y = node["x"], -node["y"] * dy
        if "leaf" in node:
            cat = node["leaf"]
            ax.add_patch(FancyBboxPatch((x - 0.40, y - 0.26), 0.80, 0.52, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc=COL.get(cat, MUTED), ec="none", zorder=3))
            ax.text(x, y, CODE.get(cat, cat), ha="center", va="center", fontsize=7.2, fontweight="bold", zorder=4,
                    color=INK, bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.15", fc="white", ec="none"))
            return
        for side, child in (("left", node["left"]), ("right", node["right"])):
            cx, cy = child["x"], -child["y"] * dy
            ax.plot([x, cx, cx], [y - 0.12, y - 0.12, cy + (0.26 if "leaf" in child else 0.32)], color=LIGHT,
                    lw=1.1, zorder=1, solid_joinstyle="round")
            ax.text((x + cx) / 2, y - 0.16, "yes" if side == "left" else "no", ha="center", va="top", fontsize=5.2,
                    color=MUTED, zorder=2)
            draw(child)
        ax.text(x, y + 0.02, cond(node["feat"], node["thr"], True) + " ?", ha="center", va="bottom", fontsize=6.0,
                color=INK, zorder=5, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=LIGHT, lw=0.6))
    draw(tree)
    ax.set_xlim(-0.6, len(leaves) - 0.4)
    ax.set_ylim(-depth * dy - 0.45, 0.55)
    tag(ax, 0, x=0.0, y=1.0)
    h = [Patch(fc=COL[c], ec="none", label=f"{CODE[c]} {NICE[c].lower()}") for c in CAT]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, -0.10), ncol=4, fontsize=6, handlelength=1,
              columnspacing=0.8, handletextpad=0.4)
    # (b) accuracy against number of rules
    b = fig.add_subplot(gs[0, 1])
    t = pd.read_csv(RES / "47_rules_curve.csv")
    t = t[t.max_leaves != "unlimited"].copy()
    t["leaves"] = t.max_leaves.astype(int)
    final = models("A").at[MODEL, "accuracy"]
    style = {"readable": ("readable list", FINAL, "-", 1.8), "elements": ("atom counts", INK, "-", 1.0),
             "vK+DBE": ("H/C, O/C, DBE", INK, (0, (4, 2)), 1.0), "vK": ("H/C, O/C", INK, (0, (1, 1.5)), 1.0)}
    for key, (nm, col, ls, lw) in style.items():
        r = t[t.features == key].sort_values("leaves")
        b.plot(r.leaves, r.accuracy, color=col, lw=lw, ls=ls, label=nm)
    for v, nm in ((97.458, "upper limit"), (final, "final model")):
        b.axhline(v, color=MUTED, lw=0.7)
        b.text(4.2, v + 0.5, nm, fontsize=5.8, color=MUTED, va="bottom")
    r16 = t[(t.features == "readable") & (t.leaves == 16)].iloc[0]
    b.plot(16, r16.accuracy, "o", ms=6, color=FINAL, mec="white", mew=0.7, zorder=5)
    b.annotate(f"released\n{r16.accuracy:.1f} %", (16, r16.accuracy), xytext=(24, 76), fontsize=6, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    b.set_xscale("log", base=2)
    b.set_xticks([4, 16, 64], [4, 16, 64])
    b.set_xlim(3.6, 140)
    b.set_ylim(52, 101)
    b.grid(color=GRID)
    b.set_xlabel("leaves (rules)")
    b.set_ylabel("accuracy for the category (%)", fontsize=6.8)
    b.legend(loc="lower right", fontsize=5.6, handlelength=1.8)
    tag(b, 1, x=-0.05, y=1.0)
    save(fig, "v2_fig7_readable_rules")


# ----------------------------------------------------------------------------------------- Fig. 8
def fig8():
    t = pd.read_csv(RES / "48_novelty.csv")
    groups = [("held-out lipid formulas", "Lipids, held out", "lipids", FINAL),
              ("artificial N4a (perturbations)", "Small change to a lipid formula", "close to lipids", ORANGE),
              ("training non-lipids (N1-N3)", "Non-lipids used in training", "non-lipids", GREEN),
              ("HMDB test non-lipids", "Non-lipids, HMDB test", "non-lipids", GREEN),
              ("random valid CHNOPS, not in any table", "Random valid formulas", "artificial", FUCHSIA),
              ("artificial N4b (+21 Da)", "+21 Da shift", "artificial", FUCHSIA)]
    a95 = t[t.target == 0.95].set_index("set").flagged_pct
    a99 = t[t.target == 0.99].set_index("set").flagged_pct
    nn = t[t.target == 0.95].set_index("set").n
    fig, ax = plt.subplots(figsize=(3.45, 2.9), layout="constrained")
    ax.axvspan(0, 5, color="#E6EDF7", lw=0, zorder=0)
    ax.text(5.5, 5.55, "5 % of lipids flagged\nby construction", fontsize=5.8, color=INK, va="center")
    for yi, (k, nm, grp, col) in zip(range(6)[::-1], groups):
        ax.plot([a99[k], a95[k]], [yi, yi], color=col, lw=3.0, alpha=0.25, solid_capstyle="round", zorder=2)
        ax.plot(a99[k], yi, "o", ms=5.5, mfc="white", mec=col, mew=1.4, zorder=3)
        ax.plot(a95[k], yi, "o", ms=6, color=col, mec="white", mew=0.6, zorder=4)
        ax.text(a95[k] + 3, yi, f"{a95[k]:.1f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(6)[::-1], [g[1] + "\n(n = " + f"{int(nn[g[0]]):,}".replace(",", "\u2009") + ")" for g in groups], fontsize=6.3)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-0.6, 6.2)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("flagged as unlike any known lipid (%)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=MUTED, label="threshold accepting 95 % of lipids"),
                        Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=MUTED, mew=1.2, label="accepting 99 %")],
               loc="outside upper center", ncol=2, fontsize=6.2, handletextpad=0.1, columnspacing=0.8)
    save(fig, "v2_fig8_distance_check")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    funcs = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8}
    for k in ([int(a) for a in sys.argv[1:]] or funcs):
        print(f"Fig. {k}")
        funcs[k]()
