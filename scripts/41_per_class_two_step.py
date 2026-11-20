#!/usr/bin/env python
"""
41_per_class_two_step.py — Merged pipeline: results per category and per class, the confusion
between categories, and the two-step arrangement for the class (question iii; §2.6, §3.6).

Final model (MODEL) on the chosen input list (SET), merged protocol (entries; 5-fold grouped by
formula; seeds 0, 1, 2):
  - task A: sensitivity, precision, F1 and support per category (mean over seeds); area under
    the precision-recall curve per category (seed 0 probabilities); confusion table (seed 0,
    rows = true category, normalised per row);
  - task C: F1, sensitivity and support per main class (mean over seeds);
  - two-step arrangement modelled on the previous work: a first model chooses the category and
    a second model, trained only on the entries of that category, chooses the class within it
    (a category with a single class in the training part answers that class). Compared with
    the direct class model on the same splits.

Usage:  python 41_per_class_two_step.py --set=R2C --model=XGB [--jobs=24]
Output: results/41_per_class_<set>_<model>.log, _categories.csv, _classes.csv, _confusion.csv, _two_step.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings("ignore")
import merged_common as mc  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL = OPT["set"], OPT["model"]
JOBS = int(OPT.get("jobs", 8))


def make(seed, y):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    if MODEL.startswith("RF"):
        return RandomForestClassifier(n_estimators=500, n_jobs=JOBS, random_state=seed,
                                      class_weight="balanced" if MODEL == "RF-weighted" else None), {}
    return (xgb.XGBClassifier(n_estimators=500, tree_method="hist", n_jobs=JOBS, verbosity=0, random_state=seed),
            {"sample_weight": compute_sample_weight("balanced", y)} if MODEL == "XGB-weighted" else {})


def fit_pred(Xtr, ytr, Xte, seed, proba=False):
    classes, yi = np.unique(ytr, return_inverse=True)
    if len(classes) == 1:
        p = np.full(len(Xte), classes[0], dtype=object)
        return (p, np.ones((len(Xte), 1)), classes) if proba else p
    m, kw = make(seed, yi)
    m.fit(Xtr, yi, **kw)
    if proba:
        P = m.predict_proba(Xte)
        return classes[P.argmax(1)], P, classes
    return classes[m.predict(Xte).astype(int)]


def main():
    log = mc.logger(f"41_per_class_{SET}_{MODEL}")
    log(f"41 — PER CATEGORY, PER CLASS AND TWO-STEP · {SET} · {MODEL}")
    d, _ = mc.load_entries()
    X = mc.features(d.hill.tolist(), sets=[SET])[SET]
    yA, yC, g = d.CATEGORY.to_numpy(), d.MAIN_CLASS.to_numpy(), d.hill.to_numpy()
    cats, classes = np.unique(yA), np.unique(yC)

    per_cat, per_cls, two, conf = [], [], [], None
    for s in mc.SEEDS:
        pA = np.empty(len(yA), dtype=object)
        PA = np.zeros((len(yA), len(cats)))
        pC_direct = np.empty(len(yC), dtype=object)
        pC_two = np.empty(len(yC), dtype=object)
        for tr, te in mc.splits(yA, g, s):
            pa, P, seen = fit_pred(X[tr], yA[tr], X[te], s, proba=True)
            pA[te] = pa
            PA[np.ix_(te, np.searchsorted(cats, seen))] = P
            pC_direct[te] = fit_pred(X[tr], yC[tr], X[te], s)
            for c in np.unique(pa):                       # second step: class within the predicted category
                tt = te[pa == c]
                trc = tr[yA[tr] == c]
                pC_two[tt] = fit_pred(X[trc], yC[trc], X[tt], s) if len(trc) else pC_direct[tt]
        pr, rc, f1, sup = precision_recall_fscore_support(yA, pA.astype(str), labels=cats, zero_division=0)
        per_cat.append(pd.DataFrame({"category": cats, "precision": pr * 100, "sensitivity": rc * 100,
                                     "f1": f1 * 100, "n": sup, "seed": s}))
        pr, rc, f1, sup = precision_recall_fscore_support(yC, pC_direct.astype(str), labels=classes, zero_division=0)
        per_cls.append(pd.DataFrame({"main_class": classes, "precision": pr * 100, "sensitivity": rc * 100,
                                     "f1": f1 * 100, "n": sup, "seed": s}))
        for name, p in (("direct", pC_direct), ("two-step", pC_two)):
            sc = mc.scores(yC, p.astype(str))
            two.append({"seed": s, "arrangement": name, **sc})
        if s == 0:
            auc = [average_precision_score(yA == c, PA[:, i]) * 100 for i, c in enumerate(cats)]
            cm = confusion_matrix(yA, pA.astype(str), labels=cats).astype(float)
            conf = pd.DataFrame(cm / cm.sum(1, keepdims=True) * 100, index=cats, columns=cats)
    tc = pd.concat(per_cat).groupby("category").agg(["mean", "std"])
    tc.columns = [f"{a}_{b}" for a, b in tc.columns]
    tc["pr_auc_seed0"] = auc
    tc.to_csv(mc.RES / f"41_per_class_{SET}_{MODEL}_categories.csv")
    tk = pd.concat(per_cls).groupby("main_class").agg(["mean", "std"])
    tk.columns = [f"{a}_{b}" for a, b in tk.columns]
    tk.sort_values("n_mean", ascending=False).to_csv(mc.RES / f"41_per_class_{SET}_{MODEL}_classes.csv")
    conf.to_csv(mc.RES / f"41_per_class_{SET}_{MODEL}_confusion.csv")
    t2 = pd.DataFrame(two)
    t2.to_csv(mc.RES / f"41_per_class_{SET}_{MODEL}_two_step.csv", index=False)

    log("\n--- task A, per category (mean over seeds)")
    for c, r in tc.iterrows():
        log(f"  {c:22s} n {int(r.n_mean):6,}  sensitivity {r.sensitivity_mean:6.2f}  precision {r.precision_mean:6.2f}  "
            f"F1 {r.f1_mean:6.2f} ± {r.f1_std:.2f}  PR-AUC {r.pr_auc_seed0:6.2f}")
    log("\n--- confusion (seed 0, % of each true category; rows = true)")
    log(conf.round(1).to_string())
    log("\n--- task C: direct model vs two-step arrangement (mean ± SD over seeds)")
    for a, gr in t2.groupby("arrangement"):
        log(f"  {a:9s} acc {gr.accuracy.mean():6.2f} ± {gr.accuracy.std(ddof=1):.2f}   macro F1 "
            f"{gr.macro_f1.mean():6.2f} ± {gr.macro_f1.std(ddof=1):.2f}")
    log(f"\n  classes with F1 < 50 % (n ≥ 20): " + ", ".join(
        f"{i} ({r.f1_mean:.0f})" for i, r in tk[(tk.f1_mean < 50) & (tk.n_mean >= 20)].iterrows()))


if __name__ == "__main__":
    main()
