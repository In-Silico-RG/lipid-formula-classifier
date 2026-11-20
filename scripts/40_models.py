#!/usr/bin/env python
"""
40_models.py — Merged pipeline: baselines and tree models on tasks A, B and C (spec D2-D7).

Protocol (merged_common): tasks A (category) and C (main class) per LIPID MAPS entry; task B
(lipid vs non-lipid) per distinct formula; 5-fold StratifiedGroupKFold grouped by formula,
seeds 0, 1, 2. Scores on the pooled out-of-fold predictions of each seed; mean ± SD.

Baselines: majority label; five chemical rules (task A); look-up table (copies the label of
the formula from the training part, majority if absent: under a split by formula it never
finds the formula, which is the point); nearest neighbours (k = 1 and 5, standardized input).
Models: random forest and XGBoost, 500 trees (as Mitchell et al.), plain and weighted
("balanced" class weights), default settings otherwise (the search is script 42).

Task C is also scored over the classes with at least 20 entries (macro F1 restricted to them).
OOF class probabilities of RF and XGB (plain and weighted), seed 0, are saved for scripts 4x.

Usage:  python 40_models.py --set=R2C [--tasks=A,B,C] [--jobs=8] [--smoke]
Output: results/40_models_<set>.log + .csv; results/40_oof_<set>_<task>_<model>.npz
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
import merged_common as mc  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET = OPT["set"]
TASKS = OPT.get("tasks", "A,B,C").split(",")
TAG = SET + ("_" + "".join(TASKS) if "tasks" in OPT else "")   # one file per cluster job
JOBS = int(OPT.get("jobs", 8))
SMOKE = "--smoke" in sys.argv
N_TREES = 500
MIN_ENTRIES_C = 20


def rules(F):
    """Five hand-written rules (manuscript §2.5); inputs from R2 columns."""
    cols = list(mc.rep.R2_COLS)
    P, N, OC, DBE, C = (F[:, cols.index(k)] for k in ("P", "N", "O_C", "DBE", "C"))
    p = np.full(len(F), "Fatty Acyls", dtype=object)
    p[P >= 1] = "Glycerophospholipids"
    p[(P == 0) & (N >= 1)] = "Sphingolipids"
    p[(P == 0) & (N == 0) & (OC < 0.12) & (DBE >= 4) & (C <= 30)] = "Sterol Lipids"
    p[(P == 0) & (N == 0) & (OC >= 0.12) & (OC < 0.30) & (C >= 35)] = "Glycerolipids"
    return p


def model_factories():
    return {
        "RF": (lambda s: RandomForestClassifier(n_estimators=N_TREES, n_jobs=JOBS, random_state=s), None),
        "RF-weighted": (lambda s: RandomForestClassifier(n_estimators=N_TREES, n_jobs=JOBS, random_state=s,
                                                         class_weight="balanced"), None),
        "XGB": (lambda s: xgb.XGBClassifier(n_estimators=N_TREES, tree_method="hist", n_jobs=JOBS,
                                            verbosity=0, random_state=s), None),
        "XGB-weighted": (lambda s: xgb.XGBClassifier(n_estimators=N_TREES, tree_method="hist", n_jobs=JOBS,
                                                     verbosity=0, random_state=s),
                         lambda ytr: {"sample_weight": compute_sample_weight("balanced", ytr)}),
        "kNN-1": (lambda s: make_pipeline(StandardScaler(), KNeighborsClassifier(1, n_jobs=JOBS)), None),
        "kNN-5": (lambda s: make_pipeline(StandardScaler(), KNeighborsClassifier(5, n_jobs=JOBS)), None),
    }


def baseline_oof(kind, y, groups, formulas, R2, seed):
    pred = np.empty(len(y), dtype=y.dtype)
    for tr, te in mc.splits(y, groups, seed):
        maj = pd.Series(y[tr]).value_counts().index[0]
        if kind == "majority":
            pred[te] = maj
        elif kind == "look-up table":
            table = pd.Series(y[tr]).groupby(formulas[tr]).agg(lambda s: s.value_counts().index[0]).to_dict()
            pred[te] = [table.get(f, maj) for f in formulas[te]]
        elif kind == "chemical rules":
            pred[te] = rules(R2[te])
    return pred


def main():
    log = mc.logger(f"40_models_{TAG}")
    log("=" * 88)
    log(f"40 — BASELINES AND TREE MODELS, merged protocol · input list {SET}")
    log("=" * 88)
    d, acc = mc.load_entries()
    m, yB, accB = mc.load_detection()
    if SMOKE:
        d = d.sample(3000, random_state=0).reset_index(drop=True)
        keep = np.random.default_rng(0).random(len(m)) < 0.04
        m, yB = m[keep].reset_index(drop=True), yB[keep]
    log(f"entries {len(d):,} ({d.hill.nunique():,} formulas) · task B formulas {len(m):,} "
        f"({int(yB.sum()):,} lipids) · seeds {mc.SEEDS} · {mc.N_FOLDS}-fold grouped by formula")

    tasks = {}
    if "A" in TASKS or "C" in TASKS:
        FE = mc.features(d.hill.tolist(), sets=[SET, "R2"])
    if "A" in TASKS:
        tasks["A"] = dict(y=d.CATEGORY.to_numpy(), groups=d.hill.to_numpy(), X=FE[SET], R2=FE["R2"],
                          formulas=d.hill.to_numpy(), accepted=acc["category"], unit="entry")
    if "C" in TASKS:
        tasks["C"] = dict(y=d.MAIN_CLASS.to_numpy(), groups=d.hill.to_numpy(), X=FE[SET], R2=FE["R2"],
                          formulas=d.hill.to_numpy(), accepted=acc["main_class"], unit="entry")
    if "B" in TASKS:
        FB = mc.features(m.formula.tolist(), sets=[SET, "R2"])
        tasks["B"] = dict(y=yB, groups=m.formula.to_numpy(), X=FB[SET], R2=FB["R2"],
                          formulas=m.formula.to_numpy(), accepted=accB, unit="formula")

    rows = []
    for tname, T in tasks.items():
        y = T["y"]
        big = None
        if tname == "C":
            vc = pd.Series(y).value_counts()
            big = vc[vc >= MIN_ENTRIES_C].index.to_numpy()
        log(f"\n--- task {tname} ({len(np.unique(y))} labels, unit = {T['unit']}, n = {len(y):,})"
            + (f"; classes with ≥ {MIN_ENTRIES_C} entries: {len(big)}" if big is not None else ""))
        methods = ["majority", "look-up table"] + (["chemical rules"] if tname == "A" else []) \
            + list(model_factories())
        for meth in methods:
            t0, per_seed = time.time(), []
            for s in mc.SEEDS:
                if meth in ("majority", "look-up table", "chemical rules"):
                    pred = baseline_oof(meth, y, T["groups"], T["formulas"], T["R2"], s)
                    P = None
                else:
                    make, fit_kw = model_factories()[meth]
                    want = meth.startswith(("RF", "XGB")) and s == 0
                    res = mc.oof(make, T["X"], y, T["groups"], s, proba=want, fit_kw=fit_kw)
                    pred, P = res if want else (res, None)
                    if P is not None:
                        np.savez_compressed(mc.RES / f"40_oof_{SET}_{tname}_{meth}.npz", proba=P,
                                            classes=np.unique(y), y=y, formulas=T["formulas"])
                sc = mc.scores(y, pred, T["formulas"], T["accepted"])
                if big is not None:
                    sc["macro_f1_classes_ge20"] = f1_score(y, pred, labels=big, average="macro",
                                                           zero_division=0) * 100
                per_seed.append(sc)
            ms = mc.mean_sd(per_seed)
            row = {"set": SET, "task": tname, "method": meth}
            for k, (mu, sd) in ms.items():
                row[k], row[k + "_sd"] = mu, sd
            rows.append(row)
            extra = f"  F1(≥{MIN_ENTRIES_C}) {ms['macro_f1_classes_ge20'][0]:6.2f}" if big is not None else ""
            log(f"  {meth:16s} acc {ms['accuracy'][0]:6.2f} ± {ms['accuracy'][1]:.2f}   "
                f"macro F1 {ms['macro_f1'][0]:6.2f} ± {ms['macro_f1'][1]:.2f}   "
                f"bal. acc {ms['balanced_accuracy'][0]:6.2f}   permissive {ms['permissive_accuracy'][0]:6.2f}"
                f"{extra}   [{time.time() - t0:.0f} s]")
        pd.DataFrame(rows).to_csv(mc.RES / f"40_models_{TAG}.csv", index=False)
    log(f"\n  Saved: results/40_models_{TAG}.csv")


if __name__ == "__main__":
    main()
