#!/usr/bin/env python
"""
91_external_composition.py — Answer to the peer review (points 2 and 5): is the model "as accurate as in
cross-validation" on the independent sets once their composition is taken into account? And how often is
the 95 % prediction set empty on them?

(a) Composition-adjusted accuracy. The category mix of GNPS, MassBank and NIST differs from LIPID MAPS
    (GNPS is almost all glycerophospholipids, glycerolipids and sphingolipids). For each set, the accuracy
    expected from cross-validation is  sum_c share_c(set) x sensitivity_c(CV)  with the sensitivity of
    each category measured by the out-of-fold predictions of the final model (seeds 0-2), per formula
    (the external sets are scored per formula) and per entry. It is compared with the observed accuracy
    (results/44_external_R2C_RF-weighted_predictions.csv, seed 0). Per-category observed accuracy is given too.
(b) Prediction sets outside the catalogue: the 95 % thresholds are those of 44_external.py (calibrated on the
    out-of-fold probabilities of LIPID MAPS, seed 0; class-conditional for the category, marginal for the
    main class); the final model is refitted without the formulas of each set (seed 0) and the share of
    empty sets, the mean size and the coverage are computed, plus how often the top label is outside its own set.

Output: results/91_external_composition.log + .csv (a), 91_external_prediction_sets.csv (b)
"""
import numpy as np
import pandas as pd

import merged_common as mc
import review_common as rc

ALPHA, MIN_CAL = 0.05, 30


def thresholds(P, cl, y, mondrian):
    yi = np.searchsorted(cl, y)
    sc = 1 - P[np.arange(len(yi)), yi]

    def q(v):
        k = int(np.ceil((len(v) + 1) * (1 - ALPHA)))
        return np.sort(v)[min(k, len(v)) - 1]
    marg = q(sc)
    return np.array([q(sc[yi == c]) if mondrian and (yi == c).sum() >= MIN_CAL else marg for c in range(len(cl))])


def main():
    log = mc.logger("91_external_composition")
    log("=" * 88)
    log("91 — EXTERNAL SETS: composition-adjusted accuracy and empty prediction sets (peer-review points 2, 5)")
    log("=" * 88)
    d, acc, FE = rc.load_entries_features(("R2", "R2C"))
    X, forms = FE["R2C"], d.hill.to_numpy()
    yA, yC = d.CATEGORY.to_numpy(), d.MAIN_CLASS.to_numpy()

    # ---- cross-validation reference: out-of-fold predictions of the final model, 3 seeds
    sens_e, sens_f = [], []
    for s in mc.SEEDS:
        pred, P, cl = rc.cached_oof("hs_A_formula", rc.rf_weighted, X, yA, forms, s)
        e = pd.DataFrame({"c": yA, "ok": pred == yA, "f": forms})
        sens_e.append(e.groupby("c").ok.mean() * 100)
        fm = e.groupby("f").agg(c=("c", lambda v: v.value_counts().index[0]), ok=("ok", "mean"))   # formula level
        sens_f.append(fm.groupby("c").ok.mean() * 100)
    sens_e, sens_f = pd.concat(sens_e, axis=1).mean(1), pd.concat(sens_f, axis=1).mean(1)
    cv_all_e = np.mean([(rc.cached_oof("hs_A_formula", rc.rf_weighted, X, yA, forms, s)[0] == yA).mean() * 100
                        for s in mc.SEEDS])
    log(f"CV reference (task A, 3 seeds): entry accuracy {cv_all_e:.2f} %; per-category sensitivity (entries / formulas):")
    for c in sens_e.index:
        log(f"   {c:26s} {sens_e[c]:6.2f} / {sens_f[c]:6.2f}")

    # ---- (a) observed on the external sets
    pr = pd.read_csv(mc.RES / "44_external_R2C_RF-weighted_predictions.csv")
    rows, per_cat = [], []
    log("\n(a) Observed accuracy (seed 0) against the accuracy expected from cross-validation, given each set's composition")
    for setname in ("GNPS", "MassBank", "NIST"):
        g = pr[(pr.set == setname) & (pr.level == "category")].copy()
        g["t1"] = g.truth.str.split(";").str[0]
        g["ok"] = [p in t.split(";") for p, t in zip(g.predicted, g.truth)]
        share = g.t1.value_counts(normalize=True)
        exp_f = float(sum(share[c] * sens_f.get(c, np.nan) for c in share.index))
        exp_e = float(sum(share[c] * sens_e.get(c, np.nan) for c in share.index))
        obs = g.ok.mean() * 100
        rows.append({"set": setname, "n_formulas": len(g), "observed": obs, "expected_from_CV_formula_level": exp_f,
                     "expected_from_CV_entry_level": exp_e, "difference_observed_minus_expected_formula": obs - exp_f})
        log(f"  {setname:9s} n = {len(g):5,}  observed {obs:6.2f} %  expected (CV, per formula) {exp_f:6.2f} %  "
            f"expected (CV, per entry) {exp_e:6.2f} %  difference {obs - exp_f:+6.2f}")
        for c in share.index:
            m = g.t1 == c
            per_cat.append({"set": setname, "category": c, "n_formulas": int(m.sum()), "share_pct": share[c] * 100,
                            "observed": g.ok[m].mean() * 100, "CV_sensitivity_formula": sens_f.get(c, np.nan),
                            "CV_sensitivity_entry": sens_e.get(c, np.nan)})
            log(f"      {c:26s} n = {int(m.sum()):5,} ({share[c] * 100:5.1f} %)  observed {g.ok[m].mean() * 100:6.2f}  "
                f"CV {sens_f.get(c, np.nan):6.2f} (formulas) / {sens_e.get(c, np.nan):6.2f} (entries)")
    pd.DataFrame(rows).to_csv(mc.RES / "91_external_composition.csv", index=False)
    pd.DataFrame(per_cat).to_csv(mc.RES / "91_external_composition_per_category.csv", index=False)

    # ---- (b) prediction sets outside the catalogue
    log("\n(b) 95 % prediction sets on the external sets (final model refitted without the set's formulas, seed 0)")
    ps = []
    cfg = {"category": ("hs_A_formula", yA, True), "main_class": ("hs_C_formula", yC, False)}
    TH = {}
    for level, (tag, y, mond) in cfg.items():
        pred, P, cl = rc.cached_oof(tag, rc.rf_weighted, X, y, forms, 0)
        TH[level] = (cl, thresholds(P, cl, y, mond))
    for setname, level in (("GNPS", "category"), ("MassBank", "category"), ("NIST", "category"), ("NIST", "main_class")):
        g = pr[(pr.set == setname) & (pr.level == level)]
        fs = g.formula.tolist()
        truth = [set(t.split(";")) for t in g.truth]
        keep = ~d.hill.isin(set(fs)).to_numpy()
        y = cfg[level][1]
        cl, thr = TH[level]
        Xte = mc.features(fs, sets=["R2C"])["R2C"]
        classes, yc = np.unique(y[keep], return_inverse=True)
        m = rc.rf_weighted(0)
        m.fit(X[keep], yc)
        P = np.zeros((len(fs), len(cl)))
        P[:, np.searchsorted(cl, classes)] = m.predict_proba(Xte)
        S = (1 - P) <= thr[None, :]
        top = cl[P.argmax(1)]
        sizes = S.sum(1)
        cover = np.array([any(S[i, np.searchsorted(cl, t)] for t in truth[i] if t in cl) for i in range(len(fs))])
        top_ok = np.array([top[i] in truth[i] for i in range(len(fs))])
        top_in_set = np.array([S[i, np.searchsorted(cl, top[i])] for i in range(len(fs))])
        r = {"set": setname, "level": level, "n": len(fs), "top1_accuracy": top_ok.mean() * 100,
             "coverage": cover.mean() * 100, "mean_size": sizes.mean(), "empty_sets_pct": (sizes == 0).mean() * 100,
             "single_label_pct": (sizes == 1).mean() * 100, "size_ge2_pct": (sizes >= 2).mean() * 100,
             "top1_correct_but_set_empty_or_without_it_pct": (top_ok & ~cover).mean() * 100,
             "top1_wrong_but_truth_in_set_pct": (~top_ok & cover).mean() * 100}
        ps.append(r)
        log(f"  {setname:9s} {level:10s} n = {r['n']:5,}  top-1 {r['top1_accuracy']:6.2f} %  coverage {r['coverage']:6.2f} %  "
            f"size {r['mean_size']:.2f}  empty {r['empty_sets_pct']:5.1f} %  single {r['single_label_pct']:5.1f} %  "
            f">=2 labels {r['size_ge2_pct']:5.1f} %  top-1 right but not covered {r['top1_correct_but_set_empty_or_without_it_pct']:5.1f} %")
    pd.DataFrame(ps).to_csv(mc.RES / "91_external_prediction_sets.csv", index=False)
    log("\n  Saved: results/91_external_composition.csv, 91_external_composition_per_category.csv, 91_external_prediction_sets.csv")


if __name__ == "__main__":
    main()
