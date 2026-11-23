#!/usr/bin/env python
"""
53_figures_tables.py — Merged pipeline: figures and LaTeX tables of the merged paper.

Reads only results/ (no training). Every figure is skipped, with a message, when its input
does not exist yet, so the script can be run at any stage. Colours: the first four slots of the
validated reference categorical palette, always in the same order (one entity = one colour);
one axis per panel; thin bars; legend whenever there are two or more series.

Usage:  python 53_figures_tables.py --set=R2C --model=XGB
Output: figures/m*_*.pdf/.png and results/53_table_*.tex
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import merged_common as mc  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL = OPT.get("set", "R2C"), OPT.get("model", "XGB")
FIG = mc.BASE / "figures"
BLUE, GREY, LIGHT = "#3969AC", "#8c8c88", "#c8c8c8"   # same blue and greys as 56_figures_redesign.py (CARTO Bold)       # one colour for what the figure is about, greys for the rest
PAL = [BLUE, GREY, "#555553", LIGHT]
# CARTO "Bold" palette and category colours, identical to 56_figures_redesign.py (Figs. 1b, 4, 5, 7)
CYAN, GREEN, YELLOW = "#008695", "#11A579", "#F2B701"
FUCHSIA, ORANGE, RED, PURPLE = "#CF1C90", "#E68310", "#E73F74", "#7F3C8D"
CAT = ["Glycerophospholipids", "Sphingolipids", "Glycerolipids", "Saccharolipids", "Polyketides",
       "Fatty Acyls", "Prenol Lipids", "Sterol Lipids"]
COL = dict(zip(CAT, [BLUE, CYAN, GREEN, YELLOW, FUCHSIA, ORANGE, RED, PURPLE]))
INK, MUTED, GRID = "#000000", "#000000", "#ececec"   # all text black, as in the other figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
                     "pdf.fonttype": 42})
LIST_ORDER = ["R1", "T10", "T12", "R2", "R2C", "R5"]
MODEL_COLOR = {"rf": PAL[0], "xgb": PAL[1], "knn": PAL[2], "mlp": PAL[3]}
MODEL_NAME = {"rf": "random forest", "xgb": "boosted trees", "knn": "nearest neighbours", "mlp": "perceptron"}


def save(fig, name, log):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    log(f"  figure: figures/{name}.pdf")


def need(path, log):
    if not path.exists():
        log(f"  skipped (missing {path.name})")
        return False
    return True


def letter(ax, k, text="", x=0.0):
    """Panel letter above the top left corner of the panel (optionally followed by a short name)."""
    ax.text(x, 1.03, f"({'abcdef'[k]})" + (f" {text}" if text else ""), transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", color=INK)


def fig_leak_limit(log):
    """Fig. 1: (a) what leakage is worth; (b) the upper limit and how close the final model comes."""
    p20, pA, pC = mc.RES / "20_entry_lookup.csv", mc.RES / f"40_models_{SET}_A.csv", mc.RES / f"40_models_{SET}_C.csv"
    if not (need(p20, log) and need(pA, log) and need(pC, log)):
        return
    r = pd.read_csv(p20)
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.1, 3.1), layout="constrained", height_ratios=[0.6, 1.25])
    vals = [r.by_entry_pct.mean(), r.by_formula_pct.mean()]
    sds = [r.by_entry_pct.std(), r.by_formula_pct.std()]
    a.barh([1, 0], vals, xerr=sds, color=[ORANGE, BLUE], height=0.6, error_kw={"elinewidth": 0.8, "ecolor": INK})
    for yv, v, sd in zip([1, 0], vals, sds):
        a.text(v + sd + 2, yv, f"{v:.1f}", va="center", fontsize=7, color=INK)
    a.set_yticks([1, 0], ["Split by entry", "Split by formula"], fontsize=7)
    a.set_xlim(0, 100)
    a.set_ylim(-0.5, 1.5)
    a.grid(axis="x", color=GRID)
    a.grid(axis="y", visible=False)
    a.tick_params(axis="y", length=0, labelcolor=INK)
    a.set_xlabel("accuracy of a look-up table (%)")
    letter(a, 0, x=-0.38)
    mA = pd.read_csv(pA).set_index("method").loc[MODEL]
    mC = pd.read_csv(pC).set_index("method").loc[MODEL]
    ceil = pd.read_csv(mc.RES / "08_ceiling_extended.csv").techo_estricto.tolist()   # strict upper limits
    rows = [("Category", ceil[0], mA.accuracy, BLUE), ("Main class", ceil[1], mC.accuracy, CYAN),
            ("Subclass", ceil[2], None, GREEN)]
    from matplotlib.colors import to_rgba
    h = 0.36
    for yv, (_, lim, acc, col) in zip([2, 1, 0], rows):
        b.barh(yv + h / 2, lim, height=h, color=to_rgba(col, 0.35), lw=0)
        b.text(lim + 1, yv + h / 2, f"{lim:.2f}", va="center", fontsize=6.5, color=INK)
        if acc is None:
            b.text(1, yv - h / 2, "no model trained", va="center", fontsize=6.2, style="italic", color=INK)
            continue
        b.barh(yv - h / 2, acc, height=h, color=col, lw=0)
        b.text(acc + 1, yv - h / 2, f"{acc:.2f}", va="center", fontsize=6.5, color=INK)
    b.set_yticks([2, 1, 0], [x[0] for x in rows], fontsize=7)
    b.set_xlim(0, 100)
    b.set_ylim(-0.6, 2.6)
    b.grid(axis="x", color=GRID)
    b.grid(axis="y", visible=False)
    b.tick_params(axis="y", length=0, labelcolor=INK)
    b.set_xlabel("correct entries (%)")
    from matplotlib.patches import Patch
    hh = [Patch(color=to_rgba(GREY, 0.35), label="upper limit of the formula (light)"),
          Patch(color=GREY, label="final model (solid)")]
    b.legend(handles=hh, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=6, ncol=2, handlelength=1.2,
             columnspacing=1, handletextpad=0.4)
    letter(b, 1, x=-0.38)
    for ext in ("pdf", "png", "tiff"):
        fig.savefig(FIG / f"fig1_leakage_and_limit.{ext}", bbox_inches="tight",
                    **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}))
    plt.close(fig)
    log("  figure: figures/fig1_leakage_and_limit.pdf")


def fig_features(log):
    """Fig. 2: the rule that chooses the list, drawn. One panel per model and task; the band runs from
    the best list down to one standard deviation below it, so the lists inside the band meet the rule."""
    parts = sorted(mc.RES.glob("30_feature_selection_*.csv"))
    parts = [p for p in parts if p.name != "30_feature_selection_all.csv"]
    if not parts:
        return log("  feature selection: skipped (no results)")
    t = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["model", "task", "set"], keep="last")
    nvar = {"R1": 13, "T10": 10, "T12": 12, "R2": 22, "R2C": 24, "R5": 45}
    order = sorted(nvar, key=nvar.get)                       # from the shortest list to the longest
    models = [m for m in ("rf", "xgb", "knn") if m in set(t.model)]
    tasks = (("category", "Category"), ("main_class", "Main class"))
    fig, axes = plt.subplots(2, len(models), figsize=(7.0, 3.7), layout="constrained", squeeze=False)
    x = np.arange(len(order))
    for i, (task, tname) in enumerate(tasks):
        for j, m in enumerate(models):
            ax = axes[i][j]
            r = t[(t.task == task) & (t.model == m)].set_index("set").reindex(order)
            best = r.macro_f1.idxmax()
            lo, hi = r.macro_f1[best] - r.macro_f1_sd[best], r.macro_f1[best]
            ax.axhspan(lo, hi, color="#dededa", lw=0, zorder=0)
            for xi, (v, sd) in enumerate(zip(r.macro_f1, r.macro_f1_sd)):
                ax.errorbar(xi, v, yerr=sd, fmt="o", ms=5, color=BLUE if v >= lo else GREY, mec="white", mew=0.6,
                            elinewidth=1, capsize=0, zorder=3)
            span = r.macro_f1.max() - (r.macro_f1 - r.macro_f1_sd).min()
            ax.set_ylim((r.macro_f1 - r.macro_f1_sd).min() - 0.08 * span, r.macro_f1.max() + r.macro_f1_sd.max() + 0.10 * span)
            ax.set_xlim(-0.6, len(order) - 0.4)
            ax.set_xticks(x, [f"{n}\n{nvar[n]}" for n in order] if i == 1 else order, fontsize=7)
            ax.tick_params(axis="x", length=0, labelcolor=INK)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(visible=False)
            if j == 0:
                ax.set_ylabel(f"{tname}\nmacro F1 (%)")
            letter(ax, i * len(models) + j, MODEL_NAME[m])
    fig.supxlabel("list of variables (lower row: with its number of variables)", fontsize=8, color=INK)
    save(fig, "m1_feature_selection", log)


def fig_models(log):
    """Fig. 3. Macro F1 of every method in the three tasks: mean of three repetitions and one SD.
    Redesign (Nov 2026): rows grouped by kind of method in the order Section 3.5 discusses them, the final
    model as the only coloured element, and a fixed column with mean +/- SD so the variability between
    repetitions is readable even where the SD line is shorter than the dot. Per-repetition values are not
    stored in results/, so the three individual repetitions cannot be drawn."""
    parts = sorted(mc.RES.glob(f"40_models_{SET}*.csv")) + sorted(mc.RES.glob(f"43_networks_{SET}*.csv"))
    if not parts:
        return log("  models: skipped (no results)")
    t = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["task", "method"], keep="last")
    tasks = [x for x in ("A", "B", "C") if x in set(t.task)]
    names = {"A": "Category", "B": "Lipid or not", "C": "Main class"}
    groups = [("Trees", [("RF-weighted", "Random forest, weighted (final)"), ("RF", "Random forest"),
                         ("XGB", "Boosted trees"), ("XGB-weighted", "Boosted trees, weighted")]),
              ("Neural networks", [("MLP(R2C)", "Perceptron"), ("CNN(R4)", "Convolutional network")]),
              ("Nearest neighbours", [("kNN", "Nearest neighbours")]),
              ("Reference methods", [("chemical rules", "Chemical rules"),
                                     ("majority", "Most common label / look-up table")])]
    for task in tasks:                                 # the identity the shared row relies on
        s = t[t.task == task].set_index("method")
        if {"majority", "look-up table"} <= set(s.index):
            assert np.isclose(s.at["majority", "macro_f1"], s.at["look-up table", "macro_f1"]), task
    rows, heads, yy = [], [], 0.0
    for g, ms in groups:
        heads.append((yy, g))
        for m, lab in ms:
            rows.append((yy, m, lab, g))
            yy -= 1.0
        yy -= 0.3
    fs = 6.5
    from matplotlib.colors import to_rgba
    fig = plt.figure(figsize=(6.6, 2.75), layout="constrained")
    axes = [[]]
    for k in range(len(tasks)):
        axes[0].append(fig.add_subplot(1, len(tasks), k + 1, sharey=axes[0][0] if axes[0] else None))
    plotted = []

    def msd(v, sd):
        if sd < 0.005:                                 # deterministic methods: SD is zero by construction
            return f"{v:.1f}"
        return f"{v:.1f} \u00b1 " + ("<0.1" if sd < 0.05 else f"{sd:.1f}")
    for k, (ax, task) in enumerate(zip(axes[0], tasks)):
        s = t[t.task == task].set_index("method")
        knn = [m for m in ("kNN-1", "kNN-5") if m in s.index]
        if knn:
            s.loc["kNN"] = s.loc[max(knn, key=lambda m: s.at[m, "macro_f1"])]
        for yi, m, lab, g in rows:
            if m not in s.index:                       # chemical rules exist only for the category
                continue
            v, sd = s.at[m, "macro_f1"], s.at[m, "macro_f1_sd"]
            col = {"Trees": BLUE, "Neural networks": FUCHSIA, "Nearest neighbours": GREEN,
                   "Reference methods": GREY}[g]
            if m == "RF-weighted":                     # final model: light band behind the row, dot ringed in black
                ax.axhspan(yi - 0.45, yi + 0.45, color=to_rgba(BLUE, 0.10), lw=0, zorder=0)
            if sd >= 0.005:
                ax.plot([v - sd, v + sd], [yi, yi], color=col, lw=1.1, solid_capstyle="butt", zorder=2)
            ax.plot(v, yi, "o", ms=5.0 if m == "RF-weighted" else 4.4, color=col,
                    mec=INK if m == "RF-weighted" else "white", mew=0.8 if m == "RF-weighted" else 0.5, zorder=3,
                    clip_on=False)
            ax.annotate(f"{v:.1f}", (v + sd, yi), xytext=(4, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=fs - 0.3, color="#000000")
            plotted.append((task, m, v, sd))
        ax.set_xlim(0, 113)                            # room for the values beside the dots
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.spines["bottom"].set_bounds(0, 100)
        for xg in (25, 50, 75, 100):
            ax.axvline(xg, color=GRID, lw=0.5, zorder=0)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelcolor="#000000", labelsize=fs + 0.5, labelleft=(k == 0))
        ax.tick_params(axis="x", labelsize=fs, labelcolor="#000000")
        ax.set_xlabel("macro F1 (%)", fontsize=fs + 0.5, color="#000000")
        letter(ax, k)                                  # panel letters as in the other figures
    first = axes[0][0]
    first.set_yticks([r[0] for r in rows], [r[2] for r in rows])
    first.set_ylim(yy + 0.6, rows[0][0] + 0.6)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=4.4, color=c, mec="white", label=l) for l, c in
                        (("trees", BLUE), ("neural networks", FUCHSIA), ("nearest neighbours", GREEN),
                         ("reference methods", GREY))]
               + [Line2D([], [], marker="o", ls="", ms=5.0, color=BLUE, mec=INK, mew=0.8, label="final model")],
               loc="outside upper center", ncol=5, fontsize=6.3, handletextpad=0.1, columnspacing=1.2)
    pd.DataFrame(plotted, columns=["task", "method", "macro_f1", "macro_f1_sd"]).to_csv(
        FIG / "m2_models_plotted_values.csv", index=False)
    for ext in ("svg",):
        fig.savefig(FIG / f"m2_models.{ext}", bbox_inches="tight")
    save(fig, "m2_models", log)
    rows = []
    for task in tasks:
        for _, r in t[t.task == task].sort_values("macro_f1", ascending=False).iterrows():
            rows.append(f"{names[task]} & {r.method} & {r.accuracy:.2f} & {r.macro_f1:.2f} $\\pm$ {r.macro_f1_sd:.2f} & "
                        f"{r.balanced_accuracy:.2f} & {r.permissive_accuracy:.2f} \\\\")
    (mc.RES / f"53_table_models_{SET}.tex").write_text(
        "% generated by 53_figures_tables.py from results/40_models_* and 43_networks_*\n"
        "\\begin{tabular}{llrrrr}\n\\toprule\nTask & Method & Accuracy & Macro F1 & Bal. acc. & Permissive \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    log(f"  table: results/53_table_models_{SET}.tex")


def fig_confusion(log):
    """Which categories are confused with which (final model, seed 0, % of each true category)."""
    p = mc.RES / f"41_per_class_{SET}_{MODEL}_confusion.csv"
    if not need(p, log):
        return
    c = pd.read_csv(p, index_col=0)
    code = {"Glycerophospholipids": "GP", "Sphingolipids": "SP", "Glycerolipids": "GL", "Saccharolipids": "SL",
            "Polyketides": "PK", "Fatty Acyls": "FA", "Prenol Lipids": "PR", "Sterol Lipids": "ST"}
    order = list(code)                                   # the three categories that exchange errors come last
    c = c.loc[order, order]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", BLUE, "#163a63"])
    fig, ax = plt.subplots(figsize=(3.4, 3.1), layout="constrained")
    im = ax.imshow(c.to_numpy(), cmap=cmap, vmin=0, vmax=100, aspect="equal")
    n = len(order)
    for i in range(n):
        for j in range(n):
            v = c.iat[i, j]
            if v >= 0.45:
                ax.text(j, i, f"{v:.0f}" if v >= 99.95 else f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 70 else INK)
    ax.set_xticks(range(n), [code[k] for k in order], fontsize=7)
    ax.set_yticks(range(n), [f"{k.replace(' Lipids', ' lipids').replace(' Acyls', ' acyls')} ({code[k]})" for k in order], fontsize=7)
    # precision of each predicted category in this repetition: counts rebuilt from the row
    # percentages and the number of entries of each category (no new computation of the model)
    pn = mc.RES / f"41_per_class_{SET}_{MODEL}_categories.csv"
    if pn.exists():
        nent = pd.read_csv(pn).set_index("category").n_mean.reindex(order).to_numpy()
        cnt = c.to_numpy() / 100 * nent[:, None]
        prec = 100 * np.diag(cnt) / cnt.sum(axis=0)
        for j, v in enumerate(prec):
            ax.text(j, n + 0.15, f"{v:.1f}", ha="center", va="center", fontsize=6, color=INK)
        ax.text(-0.62, n + 0.15, "precision (%)", ha="right", va="center", fontsize=7, color=INK)
        ax.set_ylim(n + 0.6, -0.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("predicted category", labelpad=4)
    ax.set_ylabel("true category")
    ax.tick_params(length=0, labelcolor=INK)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=GRID, lw=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03, aspect=30)
    cb.set_label("entries of the true category (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)
    save(fig, "m7_confusion", log)


def fig_external(log):
    p, p51 = mc.RES / f"44_external_{SET}_{MODEL}.csv", mc.RES / f"51_error_analysis_{SET}_{MODEL}.csv"
    pA, pC = mc.RES / f"40_models_{SET}_A.csv", mc.RES / f"40_models_{SET}_C.csv"
    if not (need(p, log) and need(p51, log) and need(pA, log) and need(pC, log)):
        return
    t = pd.read_csv(p).set_index(["set", "task", "scenario"])
    cvA = pd.read_csv(pA).set_index("method").at[MODEL, "accuracy"]
    cvC = pd.read_csv(pC).set_index("method").at[MODEL, "accuracy"]
    e = pd.read_csv(p51).set_index("task").loc["B"]
    NL, LK = "no leakage", "leakage (all trained)"
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), layout="constrained")
    # (a) category / class: withheld (filled), left in training (open), cross-validation (tick)
    rows = [("GNPS, category", ("GNPS", "category"), cvA), ("MassBank, category", ("MassBank", "category"), cvA),
            ("NIST plasma, category", ("NIST", "category"), cvA), ("NIST plasma, main class", ("NIST", "main_class"), cvC)]
    for yv, (_, key, cv) in zip(range(len(rows))[::-1], rows):
        v, sd = t.at[(*key, NL), "score"], t.at[(*key, NL), "score_sd"]
        a.plot([cv, cv], [yv - 0.2, yv + 0.2], color=INK, lw=1.2, zorder=2)
        if (*key, LK) in t.index:
            lk = t.at[(*key, LK), "score"]
            a.plot([v, lk], [yv, yv], color=LIGHT, lw=1, zorder=1)
            a.plot(lk, yv, "o", ms=5.5, mfc="white", mec=BLUE, mew=1.2, zorder=3)
            a.text(lk + 0.8, yv, f"{lk:.1f}", va="center", fontsize=6.5, color=INK)
        a.errorbar(v, yv, xerr=sd, fmt="o", ms=5.5, color=BLUE, mec="white", mew=0.6, elinewidth=1, zorder=4)
        a.text(v, yv + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    a.set_yticks(range(len(rows))[::-1], [x[0] for x in rows], fontsize=7)
    a.set_xlim(72, 103)
    a.set_ylim(-0.5, len(rows) - 0.2)
    a.grid(axis="x", color=GRID)
    a.grid(axis="y", visible=False)
    a.tick_params(axis="y", length=0, labelcolor=INK)
    a.set_xlabel("correct (%)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=BLUE, label="formulas withheld"),
                      Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=BLUE, mew=1.2, label="left in training"),
                      Line2D([], [], marker="|", ls="", ms=8, mew=1.2, color=INK, label="cross-validation")],
             loc="outside upper left", ncol=3, fontsize=6.5, handletextpad=0.2, columnspacing=1.2)
    letter(a, 0, x=-0.5)
    # (b) detection
    D, R = "detection: lipids found", "detection: non-lipids rejected"
    rows = [("Lipids recognised: cross-validation", e.sensitivity, None, LIGHT),
            ("GNPS", t.at[("GNPS", D, NL), "score"], t.at[("GNPS", D, NL), "score_sd"], BLUE),
            ("MassBank", t.at[("MassBank", D, NL), "score"], t.at[("MassBank", D, NL), "score_sd"], BLUE),
            None,
            ("Non-lipids rejected: cross-validation", e.specificity, None, LIGHT),
            ("MassBank", t.at[("MassBank", R, NL), "score"], t.at[("MassBank", R, NL), "score_sd"], BLUE),
            ("HMDB", t.at[("HMDB", R, NL), "score"], t.at[("HMDB", R, NL), "score_sd"], BLUE)]
    ys, labs = [], []
    for yv, row in zip(range(len(rows))[::-1], rows):
        if row is None:
            continue
        nm, v, sd, col = row
        b.barh(yv, v, height=0.62, color=col, xerr=sd, error_kw={"elinewidth": 0.8, "ecolor": INK})
        b.text(v + 1.5, yv, f"{v:.1f}", va="center", fontsize=6.5, color=INK)
        ys.append(yv); labs.append(nm)
    b.set_yticks(ys, labs, fontsize=7)
    b.set_xlim(0, 108)
    b.set_ylim(-0.6, 6.6)
    b.set_xticks([0, 20, 40, 60, 80, 100])
    b.grid(axis="x", color=GRID)
    b.grid(axis="y", visible=False)
    b.tick_params(axis="y", length=0, labelcolor=INK)
    b.set_xlabel("formulas (%)")
    letter(b, 1, x=-0.72)
    save(fig, "m3_independent_data", log)
    t = pd.read_csv(p)
    rows = [f"{r.set} & {r.task} & {r.scenario} & {r.n} & {r.score:.2f} $\\pm$ {r.score_sd:.2f} \\\\" for _, r in t.iterrows()]
    (mc.RES / f"53_table_external_{SET}_{MODEL}.tex").write_text(
        "% generated by 53_figures_tables.py from results/44_external_*\n\\begin{tabular}{lllrr}\n\\toprule\n"
        "Set & Task & Scenario & $n$ & Score (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    log(f"  table: results/53_table_external_{SET}_{MODEL}.tex")


def fig_conformal(log):
    p = mc.RES / f"45_conformal_{SET}_{MODEL}_per_label.csv"
    if not need(p, log):
        return
    t = pd.read_csv(p)
    s = t[(t.task == "A") & (t.alpha == 0.05)]
    mg = s[s.variant == "marginal"].set_index("label").coverage.sort_values()
    md = s[s.variant == "mondrian"].set_index("label").coverage.reindex(mg.index)
    fig, ax = plt.subplots(figsize=(3.12, 2.7), layout="constrained")
    y = np.arange(len(mg))
    ax.axvline(95, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for yi, lab in zip(y, mg.index):
        col = COL[lab]
        lo, hi = sorted((mg[lab], md[lab]))
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, alpha=0.3, solid_capstyle="round", zorder=2)
        ax.plot(mg[lab], yi, "o", ms=5.2, mfc="white", mec=col, mew=1.2, zorder=3)
        ax.plot(md[lab], yi, "o", ms=5.2, color=col, mec="white", mew=0.5, zorder=4)
        ax.annotate(f"{lo:.1f}", (lo, yi), xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=6.2, color=INK)
        ax.annotate(f"{hi:.1f}", (hi, yi), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=6.2, color=INK)
    ax.set_yticks(y, [lab.replace(" Lipids", " lipids").replace(" Acyls", " acyls") for lab in mg.index], fontsize=7)
    ax.set_xlim(58, 108)
    ax.set_xticks([60, 70, 80, 90, 100])
    ax.spines["bottom"].set_bounds(58, 100)
    ax.set_ylim(-0.6, len(mg) - 0.4)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlabel("coverage of the 95 % sets (%)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=INK, mew=1.1, label="one threshold for all"),
                        Line2D([], [], marker="o", ls="", ms=5, color=INK, label="one threshold per category")],
               loc="outside upper center", ncol=2, fontsize=6.5, handletextpad=0.1, columnspacing=1)
    save(fig, "m4_prediction_sets", log)


def fig_rules(log):
    p, pA = mc.RES / "47_rules_curve.csv", mc.RES / f"40_models_{SET}_A.csv"
    if not (need(p, log) and need(pA, log)):
        return
    t = pd.read_csv(p)
    t = t[t.max_leaves != "unlimited"].copy()
    t["leaves"] = t.max_leaves.astype(int)
    final = pd.read_csv(pA).set_index("method").at[MODEL, "accuracy"]
    fig, ax = plt.subplots(figsize=(3.4, 2.8), layout="constrained")
    style = {"readable": ("readable list", BLUE, "-", 1.8), "elements": ("six atom counts", "#444442", "-", 1.1),
             "vK+DBE": ("H/C, O/C, rings and\ndouble bonds", "#444442", (0, (4, 2)), 1.1),
             "vK": ("H/C and O/C", "#444442", (0, (1, 1.5)), 1.1)}
    for key, (nm, col, ls, lw) in style.items():
        r = t[t.features == key].sort_values("leaves")
        ax.plot(r.leaves, r.accuracy, color=col, lw=lw, ls=ls)
        ax.text(140, r.accuracy.iloc[-1], nm, va="center", fontsize=6.5, color=col)
    for v, nm in ((97.46, "upper limit"), (final, "final model")):
        ax.axhline(v, color=MUTED, lw=0.7, xmax=0.72)
        ax.text(4, v + 0.4, nm, fontsize=6.3, color=MUTED, va="bottom")
    r16 = t[(t.features == "readable") & (t.leaves == 16)].iloc[0]
    ax.plot(16, r16.accuracy, "o", ms=5.5, color=BLUE, mec="white", mew=0.6, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128], [4, 8, 16, 32, 64, 128])
    ax.set_xlim(3.6, 560)
    ax.set_ylim(52, 101)
    ax.spines["bottom"].set_bounds(3.6, 128)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("leaves of the tree" + " " * 28)
    ax.set_ylabel("accuracy for the category (%)")
    save(fig, "m5_readable_rules", log)


def fig_novelty(log):
    """Fig. 8. Share of each set flagged by the distance check (mean distance to the five nearest LIPID MAPS
    formulas above the threshold) at the two thresholds. Redesign (Nov 2026): sets grouped by what they are,
    held-out lipid formulas shown as the calibration reference, both values in fixed columns, and the axis
    worded as 'flagged as unusual', because a flag means the label should not be trusted, not that the
    formula is not a lipid (Section 2.10)."""
    p = mc.RES / "48_novelty.csv"
    if not need(p, log):
        return
    t = pd.read_csv(p)
    a95 = t[np.isclose(t.target, 0.95)].set_index("set")
    a99 = t[np.isclose(t.target, 0.99)].set_index("set")
    groups = [("ref", [("held-out lipid formulas", "Lipids, held out")]),
              ("altered", [("artificial N4a (perturbations)", "Small change to a lipid"),
                           ("artificial N4b (+21 Da)", "Lipid shifted by 21 Da")]),
              ("non-lipids", [("training non-lipids (N1-N3)", "Non-lipids, training"),
                              ("HMDB test non-lipids", "Non-lipids, HMDB test")]),
              ("random", [("random valid CHNOPS, not in any table", "Random valid formulas")])]
    rows, heads, yy = [], [], 0.0
    for g, ks in groups:
        heads.append((yy, g))
        for k, lab in ks:
            rows.append((yy, k, lab, g))
            yy -= 1.0
        yy -= 0.3
    fs = 6.5
    fig = plt.figure(figsize=(3.16, 2.45), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    plotted = []
    for yi, k, lab, g in rows:
        v95, v99, n = a95.at[k, "flagged_pct"], a99.at[k, "flagged_pct"], int(a95.at[k, "n"])
        col = {"ref": BLUE, "altered": ORANGE, "non-lipids": GREEN, "random": FUCHSIA}[g]
        ax.plot([v99, v95], [yi, yi], color=col, lw=1.8, alpha=0.3, solid_capstyle="round", zorder=1)
        ax.plot(v99, yi, "o", ms=4.4, mfc="white", mec=col, mew=1.0, zorder=3, clip_on=False)
        ax.plot(v95, yi, "o", ms=4.4, color=col, mec="white", mew=0.5, zorder=4, clip_on=False)
        ax.annotate(f"{v95:.1f}", (v95, yi), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=fs - 0.3, color="#000000")
        ax.annotate(f"{v99:.1f}", (v99, yi), xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=fs - 0.3, color="#000000")
        plotted.append((k, n, v95, v99))

    ax.set_yticks([r[0] for r in rows], [r[2] for r in rows], fontsize=fs)   # y axis: the sets compared
    ax.tick_params(axis="y", length=2.5, color=INK, labelcolor=INK)
    ax.set_ylabel("set of formulas", fontsize=fs + 0.5, color=INK)
    ax.set_ylim(yy + 0.6, rows[0][0] + 0.6)
    ax.set_xlim(-20, 113)                              # room for the values on both sides of the dots
    ax.set_xticks([0, 25, 50, 75, 100])              # x and y axes meet at the corner
    for xg in (0, 25, 50, 75, 100):
        ax.axvline(xg, color=GRID, lw=0.5, zorder=0)
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=fs, labelcolor="#000000")
    ax.set_xlabel("formulas flagged as unusual (%)", fontsize=fs + 0.5, color="#000000")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=4.4, color=INK, mec=INK, mew=1.0,
                               label="threshold accepting 95 % of held-out lipid formulas"),
                        Line2D([], [], marker="o", ls="", ms=4.4, mfc="white", mec=INK, mew=1.0,
                               label="threshold accepting 99 % of held-out lipid formulas")],
               loc="outside upper center", ncol=1, fontsize=fs - 0.2, handletextpad=0.2, columnspacing=1.2,
               labelcolor="#000000")
    pd.DataFrame(plotted, columns=["set", "n", "flagged_pct_95", "flagged_pct_99"]).to_csv(
        FIG / "m6_distance_check_plotted_values.csv", index=False)
    fig.savefig(FIG / "m6_distance_check.svg", bbox_inches="tight")
    save(fig, "m6_distance_check", log)


def main():
    log = mc.logger(f"53_figures_tables_{SET}_{MODEL}")
    log(f"53 — FIGURES AND TABLES · {SET} · {MODEL}")
    FIG.mkdir(exist_ok=True)
    allf = (fig_leak_limit, fig_features, fig_models, fig_confusion, fig_external, fig_conformal, fig_rules, fig_novelty)
    only = OPT.get("only")                             # e.g. --only=fig_models,fig_novelty
    for f in allf:
        if only and f.__name__ not in only.split(","):
            continue
        log(f"- {f.__name__}")
        f(log)


if __name__ == "__main__":
    main()
