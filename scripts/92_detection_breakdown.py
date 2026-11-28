#!/usr/bin/env python
"""
92_detection_breakdown.py — Answer to the peer review (point 3): what lies behind the 57.80 % of MassBank
lipid formulas recognised, how precise is lipid detection, and how does it depend on prevalence?

(a) Cross-validation (task B, final model, input list R2C, 5-fold split grouped by formula, seeds 0-2):
    sensitivity, specificity, precision, area under the precision-recall curve; and the sensitivity of the
    lipid formulas by LIPID MAPS category and by whether the formula also appears in a curated non-lipid
    database ("shared": the 3 923 formulas that carry both labels in the permissive scoring).
(b) Precision against prevalence: with the sensitivity and specificity of (a) (and of the MassBank test),
    the precision (positive predictive value) at 1 lipid per 1, 8.4 (the training ratio), 20, 100 non-lipids.
(c) Independent sets (final model refitted without the formulas of the sets, seed 0): GNPS lipids, MassBank
    lipids and measured non-lipids, HMDB non-lipids. For MassBank and GNPS lipids the share recognised by
    category, by shared / not shared with the non-lipid databases and by carbon-number band, set against the
    same breakdown in cross-validation. Precision-recall points for MassBank (lipids against measured non-lipids).

Output: results/92_detection_breakdown.log + 92_detection_*.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import merged_common as mc
import review_common as rc

PREV_RATIOS = [1, 8.4, 20, 100]


def canon(f):
    e, mo = mc.parsear(f)
    return None if mo else mc.hill(e)


def ppv(sens, spec, ratio):
    """Precision when there are `ratio` non-lipids for each lipid."""
    tp, fp = sens, (1 - spec) * ratio
    return tp / (tp + fp) * 100


def main():
    log = mc.logger("92_detection_breakdown")
    log("=" * 88)
    log("92 — LIPID DETECTION: precision, prevalence and the MassBank gap (peer-review point 3)")
    log("=" * 88)
    d, _ = mc.load_entries()
    maj_cat = d.groupby("hill").CATEGORY.agg(lambda s: s.value_counts().index[0]).to_dict()
    n_entries = d.groupby("hill").size().to_dict()
    m, yB, accB = rc.load_detection_audited()
    fB = rc.CACHE / "features_B_R2C.npz"
    if fB.exists() and (np.load(fB, allow_pickle=True)["formula"] == m.formula.to_numpy()).all():
        XB = np.load(fB, allow_pickle=True)["X"]
    else:
        XB = mc.features(m.formula.tolist(), sets=["R2C"])["R2C"]
        np.savez_compressed(fB, formula=m.formula.to_numpy(), X=XB)
    forms = m.formula.to_numpy()
    shared = ((m.label == "lipid") & (m.in_refmet_nonlipid | m.in_chebi | m.in_coconut)).to_numpy()
    cat = np.array([maj_cat.get(f, "") for f in forms])
    cnum = np.array([mc.parsear(f)[0].get("C", 0) for f in forms])
    bands = pd.cut(cnum, [0, 15, 25, 35, 50, 1000], labels=["C<=15", "C16-25", "C26-35", "C36-50", "C>50"]).astype(str)

    # ---------------------------------------------------------------- (a) cross-validation
    sens, spec, prec, apr, rows_cv = [], [], [], [], []
    for s in mc.SEEDS:
        pred, P, cl = rc.cached_oof("hs_B", rc.rf_weighted, XB, yB, forms, s)
        p1 = P[:, 1]
        tp, fn = ((pred == 1) & (yB == 1)).sum(), ((pred == 0) & (yB == 1)).sum()
        tn, fp = ((pred == 0) & (yB == 0)).sum(), ((pred == 1) & (yB == 0)).sum()
        sens.append(tp / (tp + fn) * 100); spec.append(tn / (tn + fp) * 100); prec.append(tp / (tp + fp) * 100)
        apr.append(average_precision_score(yB, p1) * 100)
        if s == 0:
            pr_, rc_, th_ = precision_recall_curve(yB, p1)
            pd.DataFrame({"precision": pr_, "recall": rc_}).iloc[::50].to_csv(mc.RES / "92_detection_pr_curve_cv.csv", index=False)
            lipid = yB == 1
            for name, mask in (("shared with a non-lipid database", shared), ("not shared", lipid & ~shared)):
                rows_cv.append({"group": name, "formulas": int(mask.sum()), "sensitivity": (pred[mask] == 1).mean() * 100})
            for c in sorted(set(cat[lipid]) - {""}):
                mk = lipid & (cat == c)
                rows_cv.append({"group": f"category: {c}", "formulas": int(mk.sum()), "sensitivity": (pred[mk] == 1).mean() * 100})
            for b in ["C<=15", "C16-25", "C26-35", "C36-50", "C>50"]:
                mk = lipid & (bands == b)
                rows_cv.append({"group": f"carbon band: {b}", "formulas": int(mk.sum()), "sensitivity": (pred[mk] == 1).mean() * 100})
    n_l, n_n = int(yB.sum()), int((yB == 0).sum())
    log(f"(a) Cross-validation, task B: {n_l:,} lipid and {n_n:,} non-lipid formulas (ratio 1 : {n_n / n_l:.1f}), 3 seeds")
    log(f"    sensitivity {np.mean(sens):.2f} ± {np.std(sens, ddof=1):.2f} %   specificity {np.mean(spec):.2f} ± "
        f"{np.std(spec, ddof=1):.2f} %   precision {np.mean(prec):.2f} ± {np.std(prec, ddof=1):.2f} %   "
        f"average precision (AUC-PR) {np.mean(apr):.2f} %")
    log("    sensitivity by group, seed 0:")
    for r in rows_cv:
        log(f"      {r['group']:44s} n = {r['formulas']:6,}   {r['sensitivity']:6.2f} %")
    pd.DataFrame(rows_cv).to_csv(mc.RES / "92_detection_cv_breakdown.csv", index=False)

    # ---------------------------------------------------------------- (b) prevalence
    log("\n(b) Precision against prevalence (positive predictive value), from sensitivity and specificity")
    sv, pv = np.mean(sens) / 100, np.mean(spec) / 100
    prev = []
    for r in PREV_RATIOS:
        prev.append({"non_lipids_per_lipid": r, "source": "cross-validation", "precision": ppv(sv, pv, r)})
        log(f"    CV         1 lipid per {r:>5} non-lipids: precision {ppv(sv, pv, r):5.1f} %")

    # ---------------------------------------------------------------- (c) independent sets
    ex = mc.DATOS / "spectra"
    g = pd.read_csv(ex / "formulas_experimental.csv").fillna("")
    g_f = sorted({canon(f) for f in g.formula} - {None})
    mb = pd.read_csv(ex / "formulas_massbank.csv")
    mb["formula"] = mb.formula.map(canon)
    mb = mb[mb.formula.notna()]
    mb_lip = [f for f in mb.formula[mb.en_lipidmaps] if f in maj_cat]
    mb_neg = list(mb.formula[mb.es_negativo_curado & ~mb.en_lipidmaps])
    hm = pd.read_csv(mc.DATOS / "hmdb_test.csv").formula.tolist()
    log(f"\n(c) Independent sets: GNPS {len(g_f):,} lipid formulas; MassBank {len(mb_lip):,} lipids and {len(mb_neg):,} "
        f"measured non-lipids; HMDB {len(hm):,} non-lipids")

    def fit_and_score(fs_withheld, fs_test, seed=0):
        keep = ~m.formula.isin(set(fs_withheld)).to_numpy()
        mdl = rc.rf_weighted(seed)
        mdl.fit(XB[keep], yB[keep])
        return mdl.predict_proba(mc.features(list(fs_test), sets=["R2C"])["R2C"])[:, 1]

    p_mb = fit_and_score(set(mb_lip) | set(mb_neg), mb_lip + mb_neg)
    pl, pn = p_mb[:len(mb_lip)], p_mb[len(mb_lip):]
    pg = fit_and_score(g_f, g_f)
    ph = fit_and_score(hm, hm)
    sens_mb, spec_mb = (pl >= 0.5).mean(), (pn < 0.5).mean()
    log(f"    MassBank (one model without the formulas of both groups): lipids found {sens_mb * 100:.2f} %, non-lipids rejected "
        f"{spec_mb * 100:.2f} %; GNPS lipids found {(pg >= 0.5).mean() * 100:.2f} %; HMDB non-lipids rejected {(ph < 0.5).mean() * 100:.2f} %")
    yy = np.r_[np.ones(len(pl)), np.zeros(len(pn))]
    log(f"    MassBank, lipids against measured non-lipids ({len(pl):,} : {len(pn):,} = 1 : {len(pn) / len(pl):.2f}): "
        f"ROC AUC {roc_auc_score(yy, p_mb) * 100:.2f} %, average precision {average_precision_score(yy, p_mb) * 100:.2f} %; "
        f"precision at the default threshold {(pl >= 0.5).sum() / ((pl >= 0.5).sum() + (pn >= 0.5).sum()) * 100:.2f} %")
    for t in (0.5, 0.4, 0.3, 0.2, 0.1):
        log(f"      threshold {t:.1f}: lipids found {(pl >= t).mean() * 100:5.1f} %  non-lipids called lipid {(pn >= t).mean() * 100:5.2f} %  "
            f"precision {(pl >= t).sum() / max(1, (pl >= t).sum() + (pn >= t).sum()) * 100:5.1f} %")
    pr_, rc_, _ = precision_recall_curve(yy, p_mb)
    pd.DataFrame({"precision": pr_, "recall": rc_}).iloc[::20].to_csv(mc.RES / "92_detection_pr_curve_massbank.csv", index=False)
    for r in PREV_RATIOS:
        prev.append({"non_lipids_per_lipid": r, "source": "MassBank test", "precision": ppv(sens_mb, spec_mb, r)})
        log(f"    MassBank   1 lipid per {r:>5} non-lipids: precision {ppv(sens_mb, spec_mb, r):5.1f} %")
    pd.DataFrame(prev).to_csv(mc.RES / "92_detection_precision_vs_prevalence.csv", index=False)

    # breakdown of the independent lipid sets
    log("\n    Share of lipid formulas recognised, independent sets (seed 0)")
    sh_idx = {f: bool(s_) for f, s_ in zip(forms, shared)}
    rows = []
    for setname, fs, p in (("MassBank", mb_lip, pl), ("GNPS", g_f, pg)):
        fs = list(fs)
        df = pd.DataFrame({"f": fs, "found": p >= 0.5})
        df["cat"] = [maj_cat.get(f, "(not in LIPID MAPS)") for f in fs]
        df["shared"] = [sh_idx.get(f, False) for f in fs]
        df["entries"] = [n_entries.get(f, 0) for f in fs]
        df["C"] = [mc.parsear(f)[0].get("C", 0) for f in fs]
        df["band"] = pd.cut(df.C, [0, 15, 25, 35, 50, 1000], labels=["C<=15", "C16-25", "C26-35", "C36-50", "C>50"]).astype(str)
        groups = [("all", df.found.map(lambda v: True))]
        for c in sorted(df.cat.unique()):
            groups.append((f"category: {c}", df.cat == c))
        groups += [("shared with a non-lipid database", df.shared), ("not shared", ~df.shared)]
        groups += [(f"carbon band: {b}", df.band == b) for b in ["C<=15", "C16-25", "C26-35", "C36-50", "C>50"]]
        groups += [("formula with >= 10 LIPID MAPS entries", df.entries >= 10), ("formula with 1 entry", df.entries == 1)]
        for name, mask in groups:
            if mask.sum():
                rows.append({"set": setname, "group": name, "formulas": int(mask.sum()), "found_pct": df.found[mask].mean() * 100})
                log(f"      {setname:8s} {name:44s} n = {int(mask.sum()):5,}   {df.found[mask].mean() * 100:6.2f} %")
    pd.DataFrame(rows).to_csv(mc.RES / "92_detection_independent_breakdown.csv", index=False)
    log("\n  Saved: results/92_detection_*.csv")


if __name__ == "__main__":
    main()
