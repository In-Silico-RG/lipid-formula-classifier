#!/usr/bin/env python
"""
19_supplementary.py — Generates the supplementary tables (results/supplementary_generated.tex). The submitted manuscript/supplementary.tex starts from this output and was then edited by hand (notes, wording).
from results/*.csv, so that no number is copied by hand.

  Table S1  hyperparameter search (06): Δ F1 and optimism per task/model.
  Table S2  per category / class (14): sensitivity, precision, F1, AUC-PR — tasks A, B and C.
  Table S3  cascade (14): flat vs hierarchical, error decomposition.
  Table S4  structural oracle (17).
  Figure S1 confusion matrices A and B (figures/14_confusion_*.png).
  Figure S2 physical checks of R4 (table from 04_representations.log).
"""
import re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RES, FIG, OUT = BASE / "results", BASE / "figures", BASE / "results" / "supplementary_generated.tex"


def esc(s):
    return str(s).replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")


def tabla(df, cols, cabeceras, fmt, caption, label, ancho="table", small=True, tam="small"):
    al = "l" + "r" * (len(cols) - 1)
    filas = []
    for _, r in df.iterrows():
        celdas = []
        for c, f in zip(cols, fmt):
            v = r[c]
            celdas.append(esc(v) if f is None else (f.format(v) if pd.notna(v) else "--"))
        filas.append(" & ".join(celdas) + " \\\\")
    return (f"\\begin{{{ancho}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n" + (f"\\{tam}\n" if small else "") +
            f"\\begin{{tabular}}{{{al}}}\n\\toprule\n" + " & ".join(cabeceras) + " \\\\\n\\midrule\n" +
            "\n".join(filas) + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{ancho}}}\n")


partes = []
# ---- S1 hyperparameters
h = pd.read_csv(RES / "06_hyperparameters.csv")
h["task"] = h["task"].astype(str).str.replace("A category", "A. Category").str.replace("B detection", "B. Lipid or not").str.replace("C class", "C. Class")
h["model"] = h["model"].astype(str).str.replace("XGB", "boosted trees").str.replace("RF", "random forest")
cols = [c for c in h.columns]
print("S1 columns:", cols)
usar = ["task", "model", "f1_macro", "f1_inner", "optimism", "sd_f1"]
partes.append(tabla(h, usar, ["Task", "Model", "F1 test (\\%)", "F1 in search (\\%)", "Optimism", "SD"],
                    [None, None, "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}"],
                    "Hyperparameter search (Optuna) with nested validation: a single inner 75/25 split of each training part, never touching the test part (not cross-validation). Columns: macro F1 on the untouched test part with the best settings found, macro F1 estimated during the search, the optimism of the search (score during the search minus score on the test part) and the standard deviation (SD) between the five seeds. Negative optimism means the search did not overestimate the result.",
                    "tab:s1", ancho="table*"))
# ---- S2 per class
for nombre, archivo, cap in (("A", "14_phase5_A_category_per_class.csv", "Task A, per category"),
                             ("B", "14_phase5_B_detection_per_class.csv", "Task B, per class"),
                             ("C", "14_phase5_C_class_per_class.csv", "Task C, per main class (57 classes with at least ten formulas)")):
    d = pd.read_csv(RES / archivo)
    d["class"] = d["class"].astype(str).str.replace(r"\s*\[[A-Z0-9]+\]$", "", regex=True).replace({"non_lipid": "not a lipid", "lipid": "lipid"})
    partes.append(tabla(d, ["class", "n_formulas", "sensitivity", "precision", "f1", "auc_pr"],
                        ["Class", "Formulas", "Sensitivity (\\%)", "Precision (\\%)", "F1 (\\%)", "AUC-PR"],
                        [None, "{:.0f}", "{:.1f}", "{:.1f}", "{:.1f}", "{:.3f}"],
                        f"{cap}: sensitivity, precision, F1 and area under the precision--recall curve, average of five seeds (formula-grouped split).",
                        f"tab:s2{nombre.lower()}", ancho="table*" if nombre == "C" else "table", tam="footnotesize" if nombre == "C" else "small"))
# ---- S3 cascade
c = pd.read_csv(RES / "14_phase5_cascade.csv").mean(numeric_only=True)
s3 = ("\\begin{table}[t]\n\\caption{Flat model (one model for 57 classes) against the two-step cascade (category first, then class within the predicted category); average of five seeds.}\n\\label{tab:s3}\n\\small\n"
      "\\begin{tabular}{lrr}\n\\toprule\n & Accuracy (\\%) & Macro F1 (\\%) \\\\\n\\midrule\n"
      f"Flat, 57 classes & {c['flat_acc']:.2f} & {c['flat_f1']:.2f} \\\\\nTwo-step cascade & {c['hier_acc']:.2f} & {c['hier_f1']:.2f} \\\\\n\\midrule\n"
      f"Cascade: category correct & {c['cat_acc']:.2f} & \\\\\nCascade: class correct given category correct & {c['class_given_cat_ok']:.2f} & \\\\\n"
      f"Cascade: error from wrong category & {c['hier_error_by_category']:.2f} & \\\\\nCascade: error within the right category & {c['hier_error_within_category']:.2f} & \\\\\n"
      f"Flat: implied category correct & {c['flat_cat_implicit_ok']:.2f} & \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
partes.append(s3)
# ---- S4 oracle
o = pd.read_csv(RES / "17_structure_oracle.csv").groupby("model").mean(numeric_only=True)
s4 = ("\\begin{table}[t]\n\\caption{The model that sees the structure (Morgan fingerprints, weighted random forest) against the formula-only model on the same test molecules; unit = molecule; formula-grouped split, five seeds.}\n\\label{tab:s4}\n\\small\n"
      "\\begin{tabular}{lrrrr}\n\\toprule\nModel & Accuracy & Macro F1 & Unambiguous & Ambiguous \\\\\n\\midrule\n")
for k, lab in (("structure_oracle", "Sees the structure"), ("formula_R2", "Formula only (R2)")):
    r = o.loc[k]; s4 += f"{lab} & {r['accuracy_per_molecule']:.2f} & {r['f1_macro']:.2f} & {r['acc_unambiguous']:.2f} & {r['acc_ambiguous']:.2f} \\\\\n"
s4 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
partes.append(s4)
# ---- Figure S1 confusion (redrawn in English from the CSVs)
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
for nombre, archivo, titulo in (("A", "14_phase5_A_category_confusion.csv", "Task A: categories"), ("B", "14_phase5_B_detection_confusion.csv", "Task B: lipid or not")):
    cm = pd.read_csv(RES / archivo, index_col=0)
    cm.index = [re.sub(r"\s*\[[A-Z0-9]+\]$", "", str(i)).replace("non_lipid", "not a lipid").replace("lipid", "lipid") for i in cm.index]; cm.columns = cm.index
    m = cm.values; n = m / np.maximum(m.sum(1, keepdims=True), 1); k = len(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2) if k > 2 else (2.6, 2.4))
    im = ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_yticks(range(k)); ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=6); ax.set_yticklabels(cm.index, fontsize=6)
    ax.set_xlabel("predicted", fontsize=7); ax.set_ylabel("true (majority label)", fontsize=7); ax.set_title(titulo, fontsize=8)
    for i in range(k):
        for j in range(k):
            if m[i, j]: ax.text(j, i, f"{n[i, j]*100:.0f}", ha="center", va="center", fontsize=5.5, color="white" if n[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of the row").ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(FIG / f"figS1_confusion_{nombre}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"figS1_confusion_{nombre}.png", dpi=600, bbox_inches="tight"); plt.close(fig)
partes.append("\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.52\\textwidth]{../figures/figS1_confusion_A.pdf}\\hfill\n\\includegraphics[width=0.40\\textwidth]{../figures/figS1_confusion_B.pdf}\n"
              "\\caption{Confusion matrices aggregated over five seeds: left, task A (categories; weighted random forest on R2); right, task B (lipid or not; boosted trees on R2). Each cell is the percentage of the row's true class.}\n\\label{fig:s1}\n\\end{figure*}\n")
# ---- S2 (figure → table) physical checks of R4
log = (RES / "04_representations.log").read_text(errors="replace")
checks = [("Monoisotopic mass vs LIPID MAPS (CHONPS formulas)", r"mean error\s*([\d.,e+-]+)\s*Da", "mean error {} Da"),
          ("M+1 height vs 1.07 \\% per carbon", r"median deviation\s*([\d.,]+)\s*%", "median deviation {} %"),
          ("Fractional mass, lipids minus non-lipids (300--600 Da)", r"difference\s*\+([\d.,]+)\s*Da", "+{} Da"),
          ("Median M+1 spacing deviation vs theoretical $^{13}$C (+0.00335 Da)", r"median spacing\s*(\+0[.,]003\d+)", "{} Da")]
filas = []
for nombre, pat, fmt in checks:
    m = re.search(pat, log)
    val = (fmt.format(m.group(1)) if (m and fmt) else (m.group(0) + " Da" if m else "see log"))
    filas.append(f"{nombre} & {esc(val)} \\\\")
partes.append("\\begin{table}[t]\n\\caption{Physical checks of the simulated isotopic pattern (R4) before use (from \\texttt{04\\_representations.log}).}\n\\label{tab:s5}\n\\small\n\\begin{tabular}{lr}\n\\toprule\nCheck & Result \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

doc = ("% Generated by scripts/19_supplementary.py — do not edit by hand.\n\\documentclass[11pt]{article}\n\\usepackage[a4paper,margin=2.2cm]{geometry}\n\\usepackage{booktabs,graphicx,amsmath}\n\\usepackage[T1]{fontenc}\\usepackage[utf8]{inputenc}\n\\renewcommand{\\thetable}{S\\arabic{table}}\\renewcommand{\\thefigure}{S\\arabic{figure}}\n"
       "\\title{Supplementary information\\\\\\large How much can a molecular formula tell us about a lipid?}\n\\author{Edwin R. Buelvas, Wilson Castro, Himer \\'Avila-George and Aldo F. Combariza}\n\\date{In Silico Group, Universidad de Sucre. Draft, September 2026}\n\\begin{document}\n\\maketitle\n"
       "Every table is generated from the result files of the public repository (\\texttt{results/*.csv}); the script that produces this document is \\texttt{scripts/19\\_supplementary.py}.\n\n" + "\n".join(partes) + "\\end{document}\n")
OUT.write_text(doc); print("written", OUT, len(doc), "chars")
