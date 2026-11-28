#!/usr/bin/env python
"""
93_ablation_carbon_descriptors.py — Answer to the peer review (point 6): what do "carbon number modulo 5" and
"carbon number parity" add, each on its own, and do they help outside the catalogue?

Variants of the input list (R2 has 22 numbers; R2C adds the two descriptors, 24 numbers):
  R2 | R2 + C mod 5 | R2 + C parity (mod 2) | R2C (both)
(a) Cross-validation (5-fold split grouped by formula, seeds 0-2), tasks A (category) and C (main class), random
    forest (balanced weights, the final model); XGBoost (500 trees, hist, no weights) for task A.
    Reported: accuracy, macro F1 (and macro F1 over the classes with >= 20 entries for task C), per-category F1.
(b) Independent sets (GNPS, MassBank, NIST category; the final model refitted without the formulas of each set,
    seeds 0-2): accuracy per variant, on the formulas of results/44_external_R2C_RF-weighted_predictions.csv.
(c) Impurity importance of the two descriptors in the random forest fitted on all entries (R2C), and their rank.

Output: results/93_ablation_carbon_descriptors.log + 93_ablation_cv.csv, 93_ablation_external.csv, 93_ablation_per_category.csv
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

import merged_common as mc
import review_common as rc


def variants(R2):
    c = R2[:, 0]
    m5, m2 = (c % 5)[:, None], (c % 2)[:, None]
    return {"R2": R2, "R2 + C mod 5": np.hstack([R2, m5]), "R2 + C parity": np.hstack([R2, m2]),
            "R2C (both)": np.hstack([R2, m5, m2])}


def xgb_model(seed):
    return xgb.XGBClassifier(n_estimators=500, tree_method="hist", n_jobs=rc.JOBS, verbosity=0, random_state=seed)


def main():
    log = mc.logger("93_ablation_carbon_descriptors")
    log("=" * 88)
    log("93 — ABLATION OF THE CARBON-NUMBER DESCRIPTORS (peer-review point 6)")
    log("=" * 88)
    d, acc, FE = rc.load_entries_features(("R2", "R2C"))
    V = variants(FE["R2"])
    assert np.allclose(V["R2C (both)"], FE["R2C"])
    forms = d.hill.to_numpy()

    # ---------------------------------------------------------------- (a) cross-validation
    rows, percat = [], []
    log("(a) Cross-validation, mean ± SD over seeds 0-2")
    for task, col in (("A", "CATEGORY"), ("C", "MAIN_CLASS")):
        y = d[col].to_numpy()
        vc = pd.Series(y).value_counts()
        big = vc[vc >= 20].index.to_numpy()
        for model, make in (("RF-weighted", rc.rf_weighted),) + ((("XGB", xgb_model),) if task == "A" else ()):
            for vname, X in V.items():
                t0, per = time.time(), []
                preds = []
                for s in mc.SEEDS:
                    tag = f"abl_{task}_{model}_{vname.replace(' ', '').replace('+', '_')}"
                    pred, P, cl = rc.cached_oof(tag, make, X, y, forms, s)
                    preds.append(pred)
                    per.append({"accuracy": accuracy_score(y, pred) * 100,
                                "macro_f1": f1_score(y, pred, average="macro", zero_division=0) * 100,
                                "macro_f1_ge20": f1_score(y, pred, labels=big, average="macro", zero_division=0) * 100})
                t = pd.DataFrame(per)
                rows.append({"task": task, "model": model, "variant": vname, **{k: t[k].mean() for k in t},
                             **{k + "_sd": t[k].std(ddof=1) for k in t}})
                log(f"  task {task} {model:11s} {vname:16s} accuracy {t.accuracy.mean():6.2f} ± {t.accuracy.std(ddof=1):.2f}   macro F1 "
                    f"{t.macro_f1.mean():6.2f} ± {t.macro_f1.std(ddof=1):.2f}   F1(classes>=20) {t.macro_f1_ge20.mean():6.2f}   "
                    f"[{time.time() - t0:.0f} s]")
                if task == "A" and model == "RF-weighted":
                    for c in np.unique(y):
                        f1s = [f1_score(y == c, p == c, zero_division=0) * 100 for p in preds]
                        percat.append({"variant": vname, "category": c, "entries": int((y == c).sum()), "f1": np.mean(f1s)})
    pd.DataFrame(rows).to_csv(mc.RES / "93_ablation_cv.csv", index=False)
    t = pd.DataFrame(rows)
    log("\n  Gain over R2 (macro F1 points, mean of seeds):")
    for (task, model), g in t.groupby(["task", "model"]):
        base = g[g.variant == "R2"].macro_f1.iloc[0]
        log(f"    task {task} {model:11s} " + "   ".join(f"{r.variant}: {r.macro_f1 - base:+5.2f}" for r in g.itertuples() if r.variant != "R2"))
    pc = pd.DataFrame(percat).pivot(index=["category", "entries"], columns="variant", values="f1").reset_index()
    pc["gain_R2C_minus_R2"] = pc["R2C (both)"] - pc["R2"]
    pc.to_csv(mc.RES / "93_ablation_per_category.csv", index=False)
    log("\n  F1 per category, task A, RF-weighted (R2 → R2C):")
    for _, r in pc.sort_values("entries", ascending=False).iterrows():
        log(f"    {r['category']:26s} n = {int(r['entries']):6,}   {r['R2']:6.2f} → {r['R2C (both)']:6.2f}   ({r['gain_R2C_minus_R2']:+5.2f})")

    # ---------------------------------------------------------------- (b) independent sets
    pr = pd.read_csv(mc.RES / "44_external_R2C_RF-weighted_predictions.csv")
    yA = d.CATEGORY.to_numpy()
    out = []
    log("\n(b) Independent sets, category accuracy by variant (final model refitted without the set's formulas; seeds 0-2)")
    for setname in ("GNPS", "MassBank", "NIST"):
        g = pr[(pr.set == setname) & (pr.level == "category")]
        fs = g.formula.tolist()
        truth = [set(x.split(";")) for x in g.truth]
        keep = ~d.hill.isin(set(fs)).to_numpy()
        Xnew = mc.features(fs, sets=["R2"])["R2"]
        Vn = variants(Xnew)
        for vname in V:
            per = []
            for s in mc.SEEDS:
                classes, yc = np.unique(yA[keep], return_inverse=True)
                m = rc.rf_weighted(s)
                m.fit(V[vname][keep], yc)
                p = classes[m.predict(Vn[vname]).astype(int)]
                per.append(np.mean([pi in t for pi, t in zip(p, truth)]) * 100)
            out.append({"set": setname, "variant": vname, "n": len(fs), "accuracy": np.mean(per), "sd": np.std(per, ddof=1)})
            log(f"  {setname:9s} n = {len(fs):5,}  {vname:16s} {np.mean(per):6.2f} ± {np.std(per, ddof=1):.2f} %")
    pd.DataFrame(out).to_csv(mc.RES / "93_ablation_external.csv", index=False)

    # ---------------------------------------------------------------- (c) importances
    log("\n(c) Impurity importance in the random forest fitted on all entries (R2C), task A")
    cols = list(mc.rep.R2_COLS) + ["C mod 5", "C mod 2"]
    m = rc.rf_weighted(0)
    m.fit(FE["R2C"], yA)
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    rank = {k: i + 1 for i, k in enumerate(imp.index)}
    for k in ("C mod 5", "C mod 2"):
        log(f"    {k}: importance {imp[k] * 100:5.2f} %  (rank {rank[k]} of {len(cols)})")
    log("    top 8: " + ", ".join(f"{k} {v * 100:.1f} %" for k, v in imp.head(8).items()))
    imp.to_csv(mc.RES / "93_ablation_importances.csv", header=["importance"])
    log("\n  Saved: results/93_ablation_*.csv")


if __name__ == "__main__":
    main()
