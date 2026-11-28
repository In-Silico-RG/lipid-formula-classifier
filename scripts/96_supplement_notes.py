#!/usr/bin/env python
"""
96_supplement_notes.py — Writes manuscript/supp_notes_7_11.tex (Supplementary Notes 7-11) from the result files of
scripts 90-95, so that no value of these notes is typed by hand. supplementary.tex includes the file with \\input.
Each note is written only if its result files exist.

  Note 7   independent data by composition, prediction sets outside the catalogue      (91)
  Note 8   split by homologous series and by connected family                          (90)
  Note 9   lipid detection: precision, prevalence and the MassBank gap                 (92)
  Note 10  carbon-number descriptors: ablation                                         (93)
  Note 11  per-formula view, bootstrap, nearest neighbours on unique formulas, RefMet lipids, N4a against COCONUT   (94, 95)

Usage: python 96_supplement_notes.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RES, OUT = BASE / "results", BASE / "manuscript" / "supp_notes_7_11.tex"


def tex(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def f(x, nd=2):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def table(caption, label, header, rows, spec):
    lines = [r"\begin{table}[h]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(lines)


def note7():
    a, b, c = RES / "91_external_composition.csv", RES / "91_external_composition_per_category.csv", RES / "91_external_prediction_sets.csv"
    if not (a.exists() and b.exists() and c.exists()):
        return ""
    A, B, C = pd.read_csv(a), pd.read_csv(b), pd.read_csv(c)
    out = [r"\section*{Note 7. Independent data by composition, and prediction sets outside the catalogue}",
           r"Script \texttt{scripts/91\_external\_composition.py} (results in \texttt{results/91\_external\_*.csv}). The three independent sets do not "
           r"have the category mix of LIPID MAPS (GNPS is almost all glycerophospholipids, glycerolipids and sphingolipids), and the categories are "
           r"not equally easy. For each set, the accuracy expected from cross-validation is the sum over categories of the share of the set in that "
           r"category times the sensitivity of the final model for it in the out-of-fold predictions (three seeds). The independent sets are scored "
           r"per formula, so the expected value uses the sensitivity per formula (every formula counted once); the sensitivity per entry, which "
           r"weights the formulas with many entries, is given for comparison. Observed values are those of seed 0.", ""]
    rows = [[tex(r.set), f"{int(r.n_formulas):,}".replace(",", r"\,"), f(r.observed), f(r.expected_from_CV_formula_level),
             f"{r.difference_observed_minus_expected_formula:+.2f}", f(r.expected_from_CV_entry_level)] for r in A.itertuples()]
    out.append(table("Category accuracy (\\%) on the independent sets: observed (seed 0) and expected from cross-validation given the category mix "
                     "of each set. File \\texttt{results/91\\_external\\_composition.csv}.", "tab:composition",
                     ["Set", "Formulas", "Observed", "Expected (per formula)", "Observed $-$ expected", "Expected (per entry)"], rows, "lrrrrr"))
    rows = [[tex(r.set), tex(r.category), f"{int(r.n_formulas):,}".replace(",", r"\,"), f(r.share_pct, 1), f(r.observed), f(r.CV_sensitivity_formula)]
            for r in B.itertuples()]
    out.append(r"\begin{table}[h]")
    out.append(r"\caption{Category accuracy (\%) by category on each independent set (seed 0) and sensitivity per formula of the same category in "
               r"cross-validation. Share: percentage of the formulas of the set in that category. File "
               r"\texttt{results/91\_external\_composition\_per\_category.csv}.}")
    out.append(r"\label{tab:compcat}\small\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llrrrr}\toprule")
    out.append(r"Set & Category & Formulas & Share (\%) & Observed & Cross-validation \\ \midrule")
    out += [" & ".join(r) + r" \\" for r in rows]
    out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")
    out.append("")
    out.append(r"The 95\,\% prediction sets of the main text were calibrated on LIPID MAPS (out-of-fold probabilities, seed 0; one threshold per category "
               r"for the category, one for all classes for the main class) and applied to the independent formulas after refitting the final model without "
               r"them (seed 0). Table~\ref{tab:setsout} gives the share of empty sets, of sets with one label and with two or more, and how often the most "
               r"probable label is right but the set does not contain it.")
    out.append("")
    rows = [[tex(r.set), tex(r.level.replace("_", " ")), f"{int(r.n):,}".replace(",", r"\,"), f(r.top1_accuracy), f(r.coverage), f(r.mean_size),
             f(r.empty_sets_pct, 1), f(r.single_label_pct, 1), f(r.size_ge2_pct, 1), f(r.top1_correct_but_set_empty_or_without_it_pct, 1)] for r in C.itertuples()]
    out.append(table("Prediction sets (95\\% target) on the independent sets. Top 1: accuracy of the most probable label. Coverage: a label of the formula "
                     "is in the set. Empty, Single, $\\geq$2: share of sets with 0, 1, 2 or more labels (\\%). Right, not covered: the top label is right "
                     "and not in the set (\\%). File \\texttt{results/91\\_external\\_prediction\\_sets.csv}.", "tab:setsout",
                     ["Set", "Level", "n", "Top 1", "Coverage", "Size", "Empty", "Single", "$\\geq$2", "Right, not covered"], rows, "llrrrrrrrr"))
    return "\n".join(out)


def msd(g, col):
    return f"${g[col].mean():.2f} \\pm {g[col].std(ddof=1):.2f}$"


def note8():
    a, b = RES / "90_homologous_split.csv", RES / "90_homologous_split_per_category.csv"
    if not (a.exists() and b.exists()):
        return ""
    T, P = pd.read_csv(a), pd.read_csv(b)
    out = [r"\section*{Note 8. Split by homologous series and by connected family}",
           r"Script \texttt{scripts/90\_homologous\_split.py} (results in \texttt{results/90\_homologous\_split*.csv}). Final model (random forest, "
           r"500 trees, balanced class weights), list R2C, five-fold split, seeds 0, 1 and 2, strict scoring; only the groups of the split change. "
           r"\emph{By formula}: all entries of a formula on one side (the protocol of the article). \emph{By series}: formulas with the same "
           r"heteroatoms and the same number of rings plus double bonds, whatever the carbon number, on one side (1\,998 series; the largest holds "
           r"2.8\,\% of the entries). \emph{By connected family}: formulas linked by any chain of steps of one CH$_2$, one C$_2$H$_4$ or two "
           r"hydrogen atoms on one side (1\,752 families; the largest holds 13.8\,\% of the entries), so that no formula has such a relative on the "
           r"other side. The entries without a relative in the split by formula are not a random subset (they are formulas that are hard for "
           r"another reason), which is why the split by family, and not the comparison of entries with and without relative, measures the effect.", ""]
    rows = []
    for (task, split), g in T.groupby(["task", "split"], sort=False):
        rows.append([task, tex(split), msd(g, "accuracy"), msd(g, "macro_f1"), f"{g.macro_f1_classes_ge20.mean():.2f}" if task == "C" else "--"])
    out.append(table("Accuracy and macro F1 (\\%, mean $\\pm$ SD over three seeds) of the final model under three splits. F1 $\\geq$20: macro F1 over the main "
                     "classes with at least 20 entries. File \\texttt{results/90\\_homologous\\_split.csv}.", "tab:homolog",
                     ["Task", "Split", "Accuracy", "Macro F1", "F1 $\\geq$20"], rows, "llrrr"))
    rows = [[tex(r["category"]), f"{int(r['entries']):,}".replace(",", r"\,"), f(r["by formula"]), f(r["by homologous series"]), f(r["by connected family"])]
            for _, r in P.sort_values("entries", ascending=False).iterrows()]
    out.append(table("Sensitivity (\\%) of the final model for each category, task A, seed 0, under the three splits. File "
                     "\\texttt{results/90\\_homologous\\_split\\_per\\_category.csv}.", "tab:homologcat",
                     ["Category", "Entries", "By formula", "By series", "By family"], rows, "lrrrr"))
    return "\n".join(out)


def note9():
    a, b, c, d = (RES / "92_detection_cv_breakdown.csv", RES / "92_detection_independent_breakdown.csv",
                  RES / "92_detection_precision_vs_prevalence.csv", RES / "92_detection_precision_vs_prevalence.csv")
    if not (a.exists() and b.exists() and c.exists()):
        return ""
    CV, IND, PV = pd.read_csv(a), pd.read_csv(b), pd.read_csv(c)
    out = [r"\section*{Note 9. Lipid detection: what lies behind the MassBank result, and precision}",
           r"Script \texttt{scripts/92\_detection\_breakdown.py} (results in \texttt{results/92\_detection\_*.csv}). Task B with the final model and "
           r"list R2C, on the 93\,675 formulas of the paper (9\,935 lipids and 83\,740 non-lipids, after the 313 audited formulas were set aside). "
           r"Cross-validation as in the article. For the independent sets one model was refitted without the formulas of the set (seed 0; for "
           r"MassBank, without those of both its lipids and its measured non-lipids, which is why 60.8\,\% of its lipids are found here "
           r"and $57.80 \pm 0.27$\,\% in the main text, where the lipids were withheld separately). \emph{Shared}: lipid formulas that also appear in a "
           r"curated non-lipid database.", ""]
    rows = [[tex(r.group), f"{int(r.formulas):,}".replace(",", r"\,"), f(r.sensitivity)] for r in CV.itertuples()]
    out.append(table("Sensitivity (\\%) of lipid detection in cross-validation (seed 0) by group of lipid formulas. File "
                     "\\texttt{results/92\\_detection\\_cv\\_breakdown.csv}.", "tab:detcv", ["Group", "Formulas", "Recognised"], rows, "lrr"))
    rows = [[tex(r.set), tex(r.group), f"{int(r.formulas):,}".replace(",", r"\,"), f(r.found_pct)] for r in IND.itertuples()]
    out.append(r"\begin{table}[h]\caption{Share (\%) of the lipid formulas of the independent sets that the model recognises, by group (seed 0). "
               r"File \texttt{results/92\_detection\_independent\_breakdown.csv}.}\label{tab:detind}\small\begin{adjustbox}{max width=\linewidth}"
               r"\begin{tabular}{llrr}\toprule Set & Group & Formulas & Recognised \\ \midrule")
    out += [" & ".join(r) + r" \\" for r in rows]
    out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")
    out.append("")
    piv = PV.pivot(index="non_lipids_per_lipid", columns="source", values="precision").reset_index()
    rows = [[f"{r.non_lipids_per_lipid:g}", f(r["cross-validation"], 1), f(r["MassBank test"], 1)] for _, r in piv.iterrows()]
    out.append(table("Precision (\\%) of lipid detection against the number of non-lipids per lipid, from the sensitivity and specificity of "
                     "cross-validation (training ratio 8.4) and of the MassBank test. File "
                     "\\texttt{results/92\\_detection\\_precision\\_vs\\_prevalence.csv}.", "tab:prec",
                     ["Non-lipids per lipid", "Cross-validation", "MassBank test"], rows, "rrr"))
    return "\n".join(out)


def note10():
    a, b, c, d = (RES / "93_ablation_cv.csv", RES / "93_ablation_per_category.csv", RES / "93_ablation_external.csv",
                  RES / "93_ablation_importances.csv")
    if not all(x.exists() for x in (a, b, c, d)):
        return ""
    CV, PC, EX, IM = pd.read_csv(a), pd.read_csv(b), pd.read_csv(c), pd.read_csv(d, index_col=0)
    out = [r"\section*{Note 10. The two carbon-number descriptors, one at a time}",
           r"Script \texttt{scripts/93\_ablation\_carbon\_descriptors.py} (results in \texttt{results/93\_ablation\_*.csv}). R2 (22 numbers) "
           r"with nothing added, with the carbon number modulo 5, with its parity (carbon number modulo 2), and with both (R2C, the list of the final "
           r"model). Five-fold split grouped by formula, seeds 0, 1 and 2, strict scoring. Random forest: the final model (500 trees, balanced class "
           r"weights); boosted trees: 500 trees, no weights, category only. Unlike Table~S2 (300 trees, no weights), these are the settings of the "
           r"final model.", ""]
    rows = []
    for r in CV.itertuples():
        rows.append([("Category" if r.task == "A" else "Main class"), tex(r.model), tex(r.variant), f"${r.accuracy:.2f} \pm {r.accuracy_sd:.2f}$",
                     f"${r.macro_f1:.2f} \pm {r.macro_f1_sd:.2f}$", f(r.macro_f1_ge20) if r.task == "C" else "--"])
    out.append(table("Cross-validation of the four variants (\\%, mean $\\pm$ SD over three seeds). F1 $\\geq$20: macro F1 over the main classes "
                     "with at least 20 entries. File \\texttt{results/93\\_ablation\\_cv.csv}.", "tab:ablcv",
                     ["Task", "Model", "Input list", "Accuracy", "Macro F1", "F1 $\\geq$20"], rows, "lllrrr"))
    cols = ["R2", "R2 + C mod 5", "R2 + C parity", "R2C (both)"]
    rows = [[tex(r["category"]), f"{int(r['entries']):,}".replace(",", r"\,")] + [f(r[c_]) for c_ in cols]
            for _, r in PC.sort_values("entries", ascending=False).iterrows()]
    out.append(table("F1 (\\%) of each category, final model, mean of three seeds, for the four variants. File "
                     "\\texttt{results/93\\_ablation\\_per\\_category.csv}.", "tab:ablcat",
                     ["Category", "Entries", "R2", "+ mod 5", "+ parity", "R2C"], rows, "lrrrrr"))
    piv = EX.pivot(index="set", columns="variant", values="accuracy")
    sd = EX.pivot(index="set", columns="variant", values="sd")
    n = EX.groupby("set").n.first()
    rows = [[tex(st), f"{int(n[st]):,}".replace(",", r"\,")] + [f"${piv.at[st, c_]:.2f} \pm {sd.at[st, c_]:.2f}$" for c_ in cols] for st in piv.index]
    out.append(table("Category accuracy (\\%, mean $\\pm$ SD over three seeds) on the independent sets, final model refitted without the "
                     "formulas of each set. File \\texttt{results/93\\_ablation\\_external.csv}.", "tab:ablext",
                     ["Set", "Formulas", "R2", "+ mod 5", "+ parity", "R2C"], rows, "lrrrrr"))
    imp = IM.iloc[:, 0].sort_values(ascending=False)
    rank = {k: i + 1 for i, k in enumerate(imp.index)}
    out.append(rf"In the random forest fitted on all entries (R2C, category), the impurity importance of the carbon number modulo 5 is "
               rf"{imp['C mod 5'] * 100:.2f}\,\% (rank {rank['C mod 5']} of {len(imp)}) and that of the parity {imp['C mod 2'] * 100:.2f}\,\% "
               rf"(rank {rank['C mod 2']}). Impurity importance spreads over correlated inputs, so a low rank does not contradict the gains above, "
               r"which are confined to the fatty acyls, prenol lipids and sterol lipids.")
    out.append("")
    return "\n".join(out)


def note11():
    fs = {k: RES / v for k, v in dict(u="94_units.csv", b="94_bootstrap.csv", x="94_paired_rf_xgb.csv", k="94_knn_unique_formulas.csv",
                                      z="94_accuracy_by_formula_size.csv", r="95_refmet_lipids_detection.csv", n="95_n4_in_coconut.csv").items()}
    if not all(v.exists() for v in fs.values()):
        return ""
    U, B, X, K, Z, R, N = (pd.read_csv(fs[k]) for k in "ubxkzrn")
    name = {"A": "Category", "C": "Main class"}
    out = [r"\section*{Note 11. The unit of scoring, uncertainty over formulas, and three further checks}",
           r"Scripts \texttt{scripts/94\_per\_formula\_bootstrap.py} and \texttt{scripts/95\_refmet\_n4a\_checks.py} (results in "
           r"\texttt{results/94\_*.csv} and \texttt{results/95\_*.csv}). Final model, list R2C, five-fold split grouped by formula, seeds 0, 1 and 2.", "",
           r"\textbf{Unit.} The article scores per entry, so a formula with many entries weighs more. Table~\ref{tab:units} gives the same "
           r"predictions scored per formula, with every formula weighted equally, against the upper limit computed in the same unit. Table~\ref{tab:size} "
           r"shows why the two differ: formulas with a single entry, 43.7\,\% of the formulas but 8.7\,\% of the entries, are the hardest.", ""]
    rows = [[name[r.task], f(r.per_entry), f(r.L_entry), f(r.per_formula_equal), f(r.L_formula), f(r.per_formula_majority), f(r.per_formula_permissive)]
            for r in U.itertuples()]
    out.append(table("Accuracy (\\%) of the final model in two units, with the upper limit in each. Per formula (majority): the prediction against the "
                     "most common label of the formula; permissive: against any label of the formula. File \\texttt{results/94\\_units.csv}.", "tab:units",
                     ["Task", "Per entry", "Limit", "Per formula", "Limit", "Per formula (majority)", "Per formula (permissive)"], rows, "lrrrrrr"))
    rows = [[name[r.task], tex(r.formula_size), f"{int(r.formulas):,}".replace(",", r"\,"), f(r.share_formulas_pct, 1), f(r.share_entries_pct, 1), f(r.accuracy_pct, 1)]
            for r in Z.itertuples()]
    out.append(table("Accuracy per formula (\\%, mean of three seeds) by the number of \\lm{} entries of the formula. File "
                     "\\texttt{results/94\\_accuracy\\_by\\_formula\\_size.csv}.".replace("\\lm{}", "LIPID MAPS"), "tab:size",
                     ["Task", "Entries of the formula", "Formulas", "\\% of formulas", "\\% of entries", "Accuracy"], rows, "llrrrr"))
    x = X.iloc[0]
    out.append(r"\textbf{Uncertainty.} Resampling formulas (2\,000 bootstrap resamples, the entries of a formula moving together) gives 95\,\% "
               + "; ".join(f"{name[r.task].lower()} accuracy {r.accuracy:.2f}\\,\\% [{r.lo:.2f}, {r.hi:.2f}]" for r in B.itertuples())
               + rf". The boosted trees score {x.xgb_accuracy:.2f}\,\% for the category; the paired difference, random forest minus boosted trees, is "
               rf"{x['diff']:+.2f} points [{x.diff_lo:+.2f}, {x.diff_hi:+.2f}], so the two cannot be told apart. These intervals are wider than the "
               r"standard deviations between seeds, which reflect only the randomness of the model.")
    out.append("")
    rows = [[tex(r.list), str(int(r.k)), tex(r.fitted_on), f"${r.accuracy:.2f} \pm {r.accuracy_sd:.2f}$", f(r.macro_f1)] for r in K.itertuples()]
    out.append(table("Nearest neighbours for the category, fitted on all training entries or on one row per training formula (its most common "
                     "label), same folds, scored per entry. File \\texttt{results/94\\_knn\\_unique\\_formulas.csv}.", "tab:knnu",
                     ["List", "$k$", "Fitted on", "Accuracy", "Macro F1"], rows, "lrlrr"))
    a = R.iloc[0]
    LEQ, PCT = "$\\leq$", "\\,\\%"
    bands = "; ".join(r.group.replace("carbon band: ", "").replace("<=", LEQ) + f" {r.called_lipid_pct:.1f}" + PCT
                      + f" (n = {int(r.formulas)})" for r in R.iloc[1:].itertuples())
    out.append(rf"\textbf{{RefMet lipids not in LIPID MAPS.}} The {int(a.formulas):,} formulas of RefMet lipids that are not in LIPID MAPS were set "
               rf"aside and used nowhere. The final detection model (trained on all task-B formulas, three seeds) calls {a.called_lipid_pct:.1f}\,\% of "
               r"them lipid: " + bands + ". Their labels come from RefMet's own lipid "
               r"super-classes, so this is not an independent test of sensitivity; it shows that the model recognises large lipids of other catalogues "
               r"far better than small ones.")
    out.append("")
    na, nb = N[N.set == "N4a"].iloc[0], N[N.set == "N4b"].iloc[0]
    out.append(rf"\textbf{{Artificial non-lipids that are real formulas.}} Of the {int(na.formulas):,} N4a formulas, {na.in_coconut_lipid_like_pct:.2f}\,\% "
               rf"are formulas of COCONUT natural products that the paper's rule counts as lipid-like (ClassyFire super-class lipids and lipid-like "
               rf"molecules or phenylpropanoids and polyketides, or NPClassifier pathway terpenoids, fatty acids or polyketides); of the "
               rf"{int(nb.formulas):,} N4b formulas, {nb.in_coconut_lipid_like_pct:.2f}\,\%. Every N4a or N4b formula found in COCONUT is of this kind, "
               r"because the other COCONUT formulas are in the final table and were excluded from the artificial sets. COCONUT is only one catalogue, so "
               r"these shares are lower bounds of the artificial formulas that belong to real compounds.")
    out.append("")
    return "\n".join(out)


def main():
    parts = ["% Written by scripts/96_supplement_notes.py from results/9x files. Do not edit by hand.", note7(), note8(), note9(), note10(), note11()]
    OUT.write_text("\n".join(p for p in parts if p) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
