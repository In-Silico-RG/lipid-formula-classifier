#!/usr/bin/env python
"""
90_homologous_split.py — Answer to the peer review (point 1): performance when the split keeps whole
homologous series on one side.

The split by formula (40_models.py) still lets formulas of the same series (differing by CH2, C2H4 or
H2) fall on both sides: 96.1 % of the test entries have such a relative in training, and accuracy is
94.67 % with a relative and 83.61 % without (52_relatives.py). Here the groups of the split are the
homologous series themselves: formulas with the same heteroatoms (O, N, P, S, others) and the same rings
plus double bonds, whatever the carbon number (review_common.series_key). Every other element of the
protocol is unchanged: final model (random forest, 500 trees, balanced weights), input list R2C,
StratifiedGroupKFold with 5 folds, seeds 0, 1, 2, tasks A (category) and C (main class), strict scoring.

Output: results/90_homologous_split.log + .csv (and results/90_series_groups.csv: series sizes)
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import merged_common as mc
import review_common as rc


def main():
    log = mc.logger("90_homologous_split")
    log("=" * 88)
    log("90 — SPLIT BY HOMOLOGOUS SERIES (peer-review point 1) · R2C · RF-weighted · 5 folds · seeds 0-2")
    log("=" * 88)
    d, acc, FE = rc.load_entries_features(("R2", "R2C"))
    X = FE["R2C"]
    forms = d.hill.to_numpy()
    uf = pd.Series(forms).drop_duplicates().tolist()
    keys = {f: rc.series_key(f) for f in uf}
    ids = {k: i for i, k in enumerate(sorted(set(keys.values()), key=str))}
    series = np.array([ids[keys[f]] for f in forms])
    family = rc.family_groups(forms)
    fsz = pd.Series(family).value_counts()
    log(f"connected families (formulas linked by chains of ±CH2, ±C2H4, ±H2): {len(fsz):,}; entries in the largest "
        f"{fsz.iloc[0]:,} ({100 * fsz.iloc[0] / len(d):.1f} %); with this grouping no formula has a one-step relative on the other side")
    sizes = pd.Series(series).value_counts()
    log(f"entries {len(d):,}; formulas {len(uf):,}; homologous series {len(ids):,}; entries in the largest series "
        f"{sizes.iloc[0]:,} ({100 * sizes.iloc[0] / len(d):.1f} %); series with a single formula "
        f"{sum(1 for k, v in pd.Series([ids[keys[f]] for f in uf]).value_counts().items() if v == 1):,}")
    pd.DataFrame({"series_id": sizes.index, "entries": sizes.values}).to_csv(mc.RES / "90_series_groups.csv", index=False)

    rows = []
    for task, col in (("A", "CATEGORY"), ("C", "MAIN_CLASS")):
        y = d[col].to_numpy()
        big = pd.Series(y).value_counts()
        big = big[big >= 20].index.to_numpy()
        for split, groups in (("by formula (as the paper)", forms), ("by homologous series", series),
                              ("by connected family (strict)", family)):
            for s in mc.SEEDS:
                t0 = time.time()
                tag = f"hs_{task}_" + ("formula" if split.startswith("by formula") else
                                       "series" if "series" in split else "family")
                pred, P, cl = rc.cached_oof(tag, rc.rf_weighted, X, y, groups, s)
                has = rc.share_with_relative(forms, groups, y, s) if s == 0 else None
                r = {"task": task, "split": split, "seed": s,
                     "accuracy": accuracy_score(y, pred) * 100,
                     "macro_f1": f1_score(y, pred, average="macro", zero_division=0) * 100,
                     "balanced_accuracy": balanced_accuracy_score(y, pred) * 100,
                     "macro_f1_classes_ge20": f1_score(y, pred, labels=big, average="macro", zero_division=0) * 100}
                if has is not None:
                    r["share_with_relative"] = has.mean() * 100
                    r["accuracy_with_relative"] = (pred == y)[has].mean() * 100 if has.any() else np.nan
                    r["accuracy_without_relative"] = (pred == y)[~has].mean() * 100 if (~has).any() else np.nan
                rows.append(r)
                extra = (f"; with relative {r['share_with_relative']:.1f} % of test entries, accuracy "
                         f"{r['accuracy_with_relative']:.2f} / {r['accuracy_without_relative']:.2f} % (with / without)"
                         if has is not None else "")
                log(f"  task {task} · split {split:26s} seed {s}: accuracy {r['accuracy']:6.2f}  macro F1 "
                    f"{r['macro_f1']:6.2f}  balanced acc. {r['balanced_accuracy']:6.2f}{extra}  [{time.time() - t0:.0f} s]")
            # per-category sensitivity, seed 0, both splits
    t = pd.DataFrame(rows)
    t.to_csv(mc.RES / "90_homologous_split.csv", index=False)
    log("\n  Mean over the three seeds (accuracy / macro F1):")
    for (task, split), g in t.groupby(["task", "split"], sort=False):
        log(f"    task {task} · {split:26s} accuracy {g.accuracy.mean():6.2f} ± {g.accuracy.std(ddof=1):.2f} · macro F1 "
            f"{g.macro_f1.mean():6.2f} ± {g.macro_f1.std(ddof=1):.2f} · F1 (classes ≥ 20 entries) "
            f"{g.macro_f1_classes_ge20.mean():6.2f}")
    # per-category sensitivity of task A under both splits (seed 0)
    y = d.CATEGORY.to_numpy()
    out = []
    for split, tag in (("by formula", "hs_A_formula"), ("by homologous series", "hs_A_series"),
                       ("by connected family", "hs_A_family")):
        pred = np.load(rc.CACHE / f"oof_{tag}_s0.npz", allow_pickle=True)["pred"]
        for c in np.unique(y):
            m = y == c
            out.append({"split": split, "category": c, "entries": int(m.sum()), "sensitivity": (pred[m] == c).mean() * 100})
    o = pd.DataFrame(out).pivot(index=["category", "entries"], columns="split", values="sensitivity").reset_index()
    o.to_csv(mc.RES / "90_homologous_split_per_category.csv", index=False)
    log("\n  Sensitivity per category, task A, seed 0 (by formula / by series / by connected family):")
    for _, r in o.sort_values("entries", ascending=False).iterrows():
        log(f"    {r['category']:28s} n = {int(r['entries']):6,}   {r['by formula']:6.2f} / {r['by homologous series']:6.2f} / "
            f"{r['by connected family']:6.2f}")
    log("\n  Saved: results/90_homologous_split.csv, 90_homologous_split_per_category.csv, 90_series_groups.csv")


if __name__ == "__main__":
    main()
