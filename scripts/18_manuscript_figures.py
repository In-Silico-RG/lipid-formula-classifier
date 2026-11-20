#!/usr/bin/env python
"""
18_manuscript_figures.py — Manuscript figures, at journal resolution (RSC: 600 dpi,
8.3 cm single column, 17.1 cm double column). All figures come from results/*.csv.

  Figure 1  (a) what leakage is worth: the lookup table under the two splits;
            (b) the upper limit by category: three levels × two criteria.
  Figure 2  concentration of the detection error: 3.95 % of formulas, 38.5 % of errors.
  Figure 3  coverage by category at 90 % guarantee: marginal → Mondrian, catalogue and MassBank.
  Figure 4  (session 33) all methods on the three tasks: macro F1 ± standard deviation across seeds.
  Figure 5  (session 33) formula versus structure, seed by seed, on ambiguous and non-ambiguous formulas.
  Figure 6  (session 33) sensitivity by category in catalogue, laboratory and MassBank.
  Figure 7  (session 33) artificial non-lipids: what each choice costs and what it gains.

Colours validated with the palette validator (blue #2a78d6, orange #eb6834; reference grey).
Direct labels on all marks (identity never depends on colour alone).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RES, FIG = BASE / "results", BASE / "figures"
FIG.mkdir(exist_ok=True)
AZUL, NARANJA, GRIS, TINTA, TINTA2 = "#2a78d6", "#eb6834", "#9a9995", "#0b0b0b", "#52514e"
AGUA = "#1baf7a"   # third categorical colour of the validated palette (fixed order: blue, orange, aqua)
CM = 1 / 2.54
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5,
                     "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": GRIS, "axes.linewidth": 0.6, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "axes.labelcolor": TINTA, "text.color": TINTA,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "legend.fontsize": 7})


def guardar(fig, nombre):
    fig.savefig(FIG / f"{nombre}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{nombre}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{nombre}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("  saved", nombre)


# ---------------------------------------------------------------- Figura 1
def figura1():
    # MEASURED accuracy of the lookup table, per entry, under the two splits
    # (20_entry_lookup.csv, session 29). Previously the leakage (89.94 %) was used as if it were accuracy.
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.3 * CM, 9.2 * CM), gridspec_kw={"hspace": 0.9})
    r20 = pd.read_csv(RES / "20_entry_lookup.csv")
    vals = [r20["by_entry_pct"].mean(), r20["by_formula_pct"].mean()]
    labs = ["Split by database entry\n(the usual protocol)", "Split by formula\n(no leakage)"]
    bars = a.barh(labs, vals, color=[NARANJA, AZUL], height=0.55)
    for r, v in zip(bars, vals):
        a.text(v + 1.5, r.get_y() + r.get_height() / 2, f"{v:.1f} %", va="center", fontsize=7.5, color=TINTA)
    a.set_xlim(0, 100); a.set_xlabel("Accuracy of an exact look-up table (%)")
    a.set_title("(a) What leakage is worth: the same look-up table under two splits", loc="left", fontsize=7.5)
    a.invert_yaxis()
    # (b) upper limit by level
    niveles = ["Category (8)", "Class (86)", "Subclass (299)"]
    estricto = [97.46, 91.18, 82.96]; permisivo = [100, 100, 100]
    x = np.arange(3)
    b.plot(x, permisivo, "-o", color=GRIS, lw=1.2, ms=4, label="permissive scoring (any accepted label)")
    b.plot(x, estricto, "-o", color=AZUL, lw=1.6, ms=5, label="strict scoring (the molecule's own label)")
    for xi, v, ha, dx in zip(x, estricto, ["left", "center", "right"], [0.06, 0, -0.06]):
        b.text(xi + dx, v - 1.4, f"{v:.2f} %", ha=ha, va="top", fontsize=7, color=TINTA)
    b.text(2.06, 100, "100 %", va="center", fontsize=7, color=TINTA2)
    b.set_xticks(x); b.set_xticklabels(niveles); b.set_ylim(76, 104); b.set_ylabel("Upper limit (%)")
    b.set_title("(b) The upper limit of formula-only classification, by level of detail", loc="left", fontsize=7.5)
    b.legend(loc="upper left", handlelength=1.6, bbox_to_anchor=(0.0, -0.22), ncol=1)
    guardar(fig, "fig1_leakage_and_limit")


# ---------------------------------------------------------------- Figura 2
def figura2():
    d = pd.read_csv(RES / "10_double_evaluation.csv")
    r = d[d.task.str.startswith("TASK B")].iloc[0]
    pct_amb = r["pct_ambiguous"]; rescat = r["rescued"] / r["strict_failures"] * 100
    fig, ax = plt.subplots(figsize=(8.3 * CM, 4.2 * CM))
    filas = ["Share of the test formulas", "Share of the detection errors"]
    amb = [pct_amb, rescat]; resto = [100 - pct_amb, 100 - rescat]
    ax.barh(filas, amb, color=NARANJA, height=0.5, label="formulas that lipids share with non-lipids")
    ax.barh(filas, resto, left=amb, color="#d9d8d3", height=0.5, label="all other formulas")
    for i, v in enumerate(amb):
        ax.text(v + 1, i, f"{v:.1f} %", va="center", fontsize=7.5, color=TINTA)
    ax.set_xlim(0, 100); ax.set_xlabel("Percent"); ax.invert_yaxis()
    ax.set_title(f"Where the detection errors are: {pct_amb:.1f} % of formulas, {rescat:.1f} % of errors", loc="left", fontsize=7.5)
    ax.legend(loc="lower right", ncol=1, handlelength=1.2, bbox_to_anchor=(1.0, -0.62))
    guardar(fig, "fig2_error_concentration")


# ---------------------------------------------------------------- Figura 3
def figura3():
    t = pd.read_csv(RES / "13_conformal.csv"); t = t[(t.task == "A") & (t.alpha == 0.10)]
    fams = ["Glycerophospholipids", "Sphingolipids", "Polyketides", "Fatty Acyls", "Glycerolipids",
            "Saccharolipids", "Sterol Lipids", "Prenol Lipids"]
    nombres = {f: f.replace("Fatty Acyls", "Fatty acyls").replace("Sterol Lipids", "Sterol lipids").replace("Prenol Lipids", "Prenol lipids") for f in fams}
    fig, axes = plt.subplots(1, 2, figsize=(17.1 * CM, 7.0 * CM), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, conjunto, titulo in zip(axes, ["interno", "MassBank (LIPID MAPS)"],
                                    ["(a) Catalogue (LIPID MAPS test formulas)", "(b) Measured formulas (MassBank), thresholds from the catalogue"]):
        m = t[(t.set == conjunto) & (t.variant == "marginal")].iloc[0]
        q = t[(t.set == conjunto) & (t.variant.str.startswith("MONDRIAN"))].iloc[0]
        y = np.arange(len(fams))[::-1]
        for yi, f in zip(y, fams):
            a, b = m[f"cov|{f}"], q[f"cov|{f}"]
            ax.plot([a, b], [yi, yi], color=GRIS, lw=1.2, zorder=1)
            ax.plot(a, yi, "o", color=NARANJA, ms=5, zorder=2); ax.plot(b, yi, "o", color=AZUL, ms=5, zorder=3)
            lo, hi = (a, NARANJA), (b, AZUL)
            if a > b: lo, hi = (b, AZUL), (a, NARANJA)
            ax.text(lo[0] - 2.5, yi, f"{lo[0]:.0f}", ha="right", va="center", fontsize=6.5, color=lo[1])
            ax.text(hi[0] + 2.5, yi, f"{hi[0]:.0f}", ha="left", va="center", fontsize=6.5, color=hi[1])
        ax.axvline(90, color=TINTA2, lw=0.7, ls="--"); ax.text(90, -0.85, "target 90 %", ha="center", va="top", fontsize=6.5, color=TINTA2)
        ax.set_ylim(-1.3, len(fams) - 0.4)
        ax.set_yticks(y); ax.set_yticklabels([nombres[f] for f in fams]); ax.set_xlim(0, 108)
        ax.set_xlabel("Category inside the prediction set (%)"); ax.set_title(titulo, loc="left", fontsize=7.5)
    axes[0].plot([], [], "o", color=NARANJA, label="marginal (one threshold for all categories)")
    axes[0].plot([], [], "o", color=AZUL, label="class-conditional (one threshold per category)")
    axes[0].legend(loc="lower left", handlelength=1.0, bbox_to_anchor=(0.0, -0.42), ncol=2)
    guardar(fig, "fig3_conformal_coverage_by_family")


# ---------------------------------------------------------------- reading logs with ± across seeds
import re
_FILA = re.compile(r"^\s+(R\d \S+|R1 exact)\s+(\S+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)% \(±([\d.]+)\)")


def leer_log(nombre):
    """{task: [(representation, model, accuracy, bal_acc, f1, sd_f1)]} from a results log."""
    out, tarea = {}, None
    for linea in (RES / nombre).read_text().splitlines():
        m = re.match(r"^TASK ([ABC])", linea)
        if m:
            tarea = m.group(1); out.setdefault(tarea, []); continue
        if linea.startswith("MEJOR"):
            tarea = None
        f = _FILA.match(linea)
        if f and tarea:
            out[tarea].append((f.group(1), f.group(2), *map(float, f.groups()[2:])))
    return out


def mejor(filas, rep, pref):
    c = [r for r in filas if r[0].startswith(rep) and r[1].startswith(pref)]
    return max(c, key=lambda r: r[4]) if c else None


# ---------------------------------------------------------------- Figura 4
def figura4():
    arb, red, r1 = leer_log("05_models.log"), leer_log("09_networks.log"), leer_log("21_r1_mitchell_exact.log")
    b = pd.read_csv(RES / "03_baselines.csv")
    base = lambda t, n: (b[b.task.str.startswith(f"TASK {t}") & b.baseline.str.startswith(n)].f1_macro.tolist() or [None])[0]
    titulos = {"A": "(a) Task A: which of 8 categories", "B": "(b) Task B: lipid or not", "C": "(c) Task C: which of 57 classes"}
    fig, axes = plt.subplots(1, 3, figsize=(17.1 * CM, 7.4 * CM), sharey=True, gridspec_kw={"wspace": 0.08})
    filas_nom = ["Always the most common answer", "Four chemical rules", "Nearest neighbour (k = 1)",
                 "Previous work's variables (R1), trees", "Random forest (R2)", "Boosted trees (R2)",
                 "Boosted trees on the isotopic pattern (R4)", "Dense neural network", "Convolutional network (R4)"]
    grupo = [GRIS, GRIS, GRIS, NARANJA, AZUL, AZUL, AZUL, AGUA, AGUA]
    y = np.arange(len(filas_nom))[::-1]
    for ax, t in zip(axes, "ABC"):
        vals = [(base(t, "1."), None), (base(t, "2."), None), (base(t, "4. k-NN (k=1)"), None)]
        for rep, pref, fuente in [("R1", "", r1), ("R2", "RF", arb), ("R2", "XGB", arb), ("R4", "XGB", arb), ("R", "MLP", red), ("R4", "CNN", red)]:
            r = mejor(fuente.get(t, []), rep, pref)
            vals.append((r[4], r[5]) if r else (None, None))
        for yi, (v, sd), c in zip(y, vals, grupo):
            if v is None:
                txt = "not applicable" if (t == "B" and yi == y[1]) else "not run"
                ax.text(2, yi, txt, va="center", fontsize=6, color=GRIS); continue
            if sd:
                ax.plot([v - sd, v + sd], [yi, yi], color=c, lw=1.2, zorder=2)
            ax.plot(v, yi, "o", color=c, ms=5, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
            ax.text(v + (sd or 0) + 2, yi, f"{v:.1f}", va="center", fontsize=6.3, color=TINTA)
        ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100]); ax.grid(axis="x", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
        ax.set_title(titulos[t], loc="left", fontsize=7.3); ax.set_xlabel("Macro F1 (%)")
    axes[0].set_yticks(y); axes[0].set_yticklabels(filas_nom)
    for c, lab in [(GRIS, "simple baseline methods"), (NARANJA, "previous work's variables"), (AZUL, "tree models, our descriptions"), (AGUA, "neural networks")]:
        axes[0].plot([], [], "o", color=c, label=lab)
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.9, -0.16), ncol=4, handlelength=0.8, columnspacing=1.2)
    guardar(fig, "fig4_all_methods")


# ---------------------------------------------------------------- Figura 5
def figura5():
    d = pd.read_csv(RES / "17_structure_oracle.csv")
    f, e = d[d.model == "formula_R2"].sort_values("seed"), d[d.model == "structure_oracle"].sort_values("seed")
    fig, axes = plt.subplots(1, 2, figsize=(12.0 * CM, 6.2 * CM), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, col, tit in zip(axes, ["acc_unambiguous", "acc_ambiguous"],
                            ["(a) Molecules whose formula\nhas one category", "(b) Molecules whose formula is\nshared by several categories"]):
        for (_, rf), (_, re_) in zip(f.iterrows(), e.iterrows()):
            ax.plot([0, 1], [rf[col], re_[col]], color="#cfcec9", lw=0.9, zorder=1)
        ax.plot(np.zeros(len(f)), f[col], "o", color=NARANJA, ms=5, zorder=2, markeredgecolor="white", markeredgewidth=0.6)
        ax.plot(np.ones(len(e)), e[col], "o", color=AZUL, ms=5, zorder=2, markeredgecolor="white", markeredgewidth=0.6)
        mf, me = f[col].mean(), e[col].mean()
        ax.plot([-0.09, 0.09], [mf, mf], color=TINTA, lw=1.4, zorder=4); ax.plot([0.91, 1.09], [me, me], color=TINTA, lw=1.4, zorder=4)
        ax.text(-0.12, mf, f"mean {mf:.1f}", ha="right", va="center", fontsize=6.5, color=TINTA)
        ax.text(1.12, me, f"mean {me:.1f}", ha="left", va="center", fontsize=6.5, color=TINTA)
        if col == "acc_ambiguous":
            ax.text(-0.12, f[col].min(), f"{f[col].min():.1f}", ha="right", va="center", fontsize=6, color=TINTA2)
            ax.text(-0.12, f[col].max(), f"{f[col].max():.1f}", ha="right", va="center", fontsize=6, color=TINTA2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["formula\nonly", "structure\n(Morgan)"]); ax.set_xlim(-0.8, 1.8)
        ax.set_title(tit, loc="left", fontsize=7.3)
    axes[0].set_ylim(40, 102); axes[0].set_ylabel("Accuracy per molecule (%)")
    fig.text(0.13, -0.06, "Each line joins the two models on one of the five test sets.", fontsize=6.5, color=TINTA2)
    guardar(fig, "fig5_formula_vs_structure")


# ---------------------------------------------------------------- Figura 6
def figura6():
    cat = pd.read_csv(RES / "14_phase5_A_category_per_class.csv").set_index("class")
    lab = pd.read_csv(RES / "11_external_validation.csv"); lab = lab[lab.scenario.str.startswith("NO LEAKAGE")].iloc[0]
    mb = pd.read_csv(RES / "12_massbank_validation.csv"); mb = mb[(mb.task == "A") & mb.scenario.str.startswith("NO LEAKAGE")].iloc[0]
    lin = [l for l in (RES / "11_external_validation.log").read_text().splitlines() if "external distribution:" in l][0]
    n_lab = {k.strip().rsplit(" ", 1)[0]: int(k.strip().rsplit(" ", 1)[1]) for k in lin.split(":", 1)[1].split(",")}
    n_mb = {"Fatty Acyls": 521, "Polyketides": 343, "Glycerophospholipids": 309, "Prenol Lipids": 197, "Sterol Lipids": 192,
            "Sphingolipids": 101, "Glycerolipids": 54, "Saccharolipids": 2}                                                  # 12_massbank_validation.log
    fams = ["Glycerophospholipids", "Sphingolipids", "Polyketides", "Fatty Acyls", "Glycerolipids", "Saccharolipids", "Sterol Lipids", "Prenol Lipids"]
    nom = lambda x: x.replace("Fatty Acyls", "Fatty acyls").replace("Sterol Lipids", "Sterol lipids").replace("Prenol Lipids", "Prenol lipids")
    fig, ax = plt.subplots(figsize=(12.0 * CM, 7.0 * CM))
    y = np.arange(len(fams))[::-1]
    for yi, fam in zip(y, fams):
        pts = [(cat.loc[fam, "sensitivity"], AZUL, int(cat.loc[fam, "n_formulas"]), 0.22)]
        if fam in n_lab: pts.append((float(lab[fam]), NARANJA, n_lab[fam], 0.0))
        pts.append((float(mb[fam]), AGUA, n_mb[fam], -0.22))
        for v, c, n, dy in pts:
            ms = 3.5 if n < 20 else 5
            ax.plot(v, yi + dy, "o", color=c, ms=ms, markeredgecolor="white", markeredgewidth=0.5,
                    alpha=0.55 if n < 20 else 1)
            ax.text(v - 1.5, yi + dy, f"{v:.0f}", ha="right", va="center", fontsize=5.8, color=TINTA2)
    ax.set_yticks(y); ax.set_yticklabels([nom(f) for f in fams]); ax.set_xlim(-5, 104)
    ax.set_xlabel("Members of the category that the model finds (%)"); ax.grid(axis="x", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
    for c, lab_ in [(AZUL, "catalogue test sets"), (NARANJA, "laboratory spectra (GNPS)"), (AGUA, "MassBank spectra")]:
        ax.plot([], [], "o", color=c, label=lab_)
    ax.plot([], [], "o", color=GRIS, ms=3.5, alpha=0.55, label="fewer than 20 formulas")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.15), ncol=4, handlelength=0.8, columnspacing=1.0)
    ax.set_title("Weighted random forest on R2; external formulas removed from training", loc="left", fontsize=7.3)
    guardar(fig, "fig6_family_sensitivity_three_sources")


# ---------------------------------------------------------------- Figura 7
def figura7():
    d = pd.read_csv(RES / "15_phase6_adversarial.csv").set_index("scenario")
    esc = [("N1-N3", "Three ordinary layers"), ("N1-N3+N4b", "+ shifted-mass formulas"),
           ("N1-N3+N4a_red+N4b", "+ shifted and perturbed,\nsame size (4,988 each)"), ("N1-N4", "+ shifted and all\n28,468 perturbed")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(17.1 * CM, 6.6 * CM), sharey=True, gridspec_kw={"wspace": 0.10})
    y = np.arange(len(esc))[::-1]
    for yi, (k, _) in zip(y, esc):
        r = d.loc[k]
        for v, c, dy in [(r.sens_internal, AZUL, 0.2), (r.sens_PNNL, NARANJA, 0.0), (r.sens_MassBank, AGUA, -0.2)]:
            a.plot(v, yi + dy, "o", color=c, ms=5, markeredgecolor="white", markeredgewidth=0.5)
            a.text(v + 1.5, yi + dy, f"{v:.1f}", va="center", fontsize=6, color=TINTA2)
        for v, c, dy in [(r.N4b_called_lipid, TINTA2, 0.12), (r.N4a_called_lipid, "#bdbcb6", -0.12)]:
            b.barh(yi + dy, v, height=0.22, color=c)
            b.text(v + 1.5, yi + dy, f"{v:.1f}", va="center", fontsize=6, color=TINTA2)
    a.set_yticks(y); a.set_yticklabels([n for _, n in esc]); a.set_xlim(40, 104)
    a.set_xlabel("Real lipids still recognised (%)"); a.set_title("(a) What each choice costs", loc="left", fontsize=7.3)
    b.set_xlim(0, 80); b.set_xlabel("Artificial formulas wrongly called lipid (%)"); b.set_title("(b) What each choice gains", loc="left", fontsize=7.3)
    for ax in (a, b):
        ax.grid(axis="x", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
    for c, l in [(AZUL, "catalogue"), (NARANJA, "laboratory"), (AGUA, "MassBank")]:
        a.plot([], [], "o", color=c, label=l)
    a.legend(loc="upper left", bbox_to_anchor=(0, -0.3), ncol=3, handlelength=0.8)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=TINTA2, label="shifted-mass (+21) formulas"), Patch(color="#bdbcb6", label="perturbed “almost lipids”")], loc="upper left", bbox_to_anchor=(0, -0.3), ncol=2, handlelength=1.0)
    guardar(fig, "fig7_artificial_nonlipids")


if __name__ == "__main__":
    figura1(); figura2(); figura3(); figura4(); figura5(); figura6(); figura7()
    print("listo:", sorted(p.name for p in FIG.glob("fig*_*.png")))
