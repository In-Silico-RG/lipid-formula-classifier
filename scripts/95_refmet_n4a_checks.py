#!/usr/bin/env python
"""
95_refmet_n4a_checks.py — Answer to the peer review (points 4a, 13 and the small counts).

(a) The 21 836 lipid formulas of RefMet that are not in LIPID MAPS were set aside and never used (01_normalize.py).
    Here the final detection model (trained on the task-B formulas as in the paper) is applied to them: how many are
    recognised as lipids, in cross-validation terms an extra test of sensitivity on formulas absent from the catalogue.
    (Their labels derive from RefMet's own lipid super-classes, not from an independent source of formulas.)
(b) How many of the artificial formulas N4a ("almost lipids", labelled non-lipid) coincide with formulas of lipid-like
    natural products of COCONUT that the pipeline removed from the non-lipid layer (ClassyFire super-class lipids or
    phenylpropanoids and polyketides, or NPClassifier pathway terpenoids, fatty acids, polyketides)? Such a coincidence
    means the "non-lipid" formula is the formula of a real lipid-like compound. Same count for N4b.
(c) Why Table 5 and 48_novelty.log count 83 408 training non-lipids while the Methods say 83 740: formulas that the
    classifier's feature code (classifier/features.py) rejects.
(d) Confusion counts of the final model for the category (out-of-fold, seed 0): the real counts for Fig. 5a.

Output: results/95_refmet_n4a_checks.log + 95_*.csv
"""
import sys
import zipfile

import numpy as np
import pandas as pd

import merged_common as mc
import review_common as rc

sys.path.insert(0, str(mc.BASE / "classifier"))
from features import featurize  # noqa: E402

COCO_SUPERCLASSES = {"Lipids and lipid-like molecules", "Phenylpropanoids and polyketides"}
COCO_PATHWAYS = {"Terpenoids", "Fatty acids", "Polyketides"}


def canon(f):
    e, mo = mc.parsear(f)
    return None if mo else mc.hill(e)


def main():
    log = mc.logger("95_refmet_n4a_checks")
    log("=" * 88)
    log("95 — REFMET LIPIDS, N4a AGAINST COCONUT LIPID-LIKE FORMULAS, COUNTS, CONFUSION MATRIX (peer-review points)")
    log("=" * 88)
    m, yB, _ = rc.load_detection_audited()
    full = pd.read_parquet(mc.DATOS / "master.parquet")

    # ---------------------------------------------------------------- (a) RefMet lipids not in LIPID MAPS
    ref = full[(full.in_refmet_lipid) & (~full.in_lipidmaps) & (~full.formula.isin(set(m.formula)))]
    # the 21 836 set aside as RefMet lipids (label "discarded" in master.parquet); the other 200 are audited formulas (script 55)
    log(f"    candidates {len(ref):,}: {ref.label.value_counts().to_dict()}; kept: the RefMet lipids set aside (label 'discarded')")
    ref = ref[ref.label == "discarded"].formula.tolist()
    log(f"(a) RefMet lipid formulas outside LIPID MAPS and outside the training table: {len(ref):,}")
    fB = rc.CACHE / "features_B_R2C.npz"
    XB = np.load(fB, allow_pickle=True)["X"] if fB.exists() else mc.features(m.formula.tolist(), sets=["R2C"])["R2C"]
    Xr = mc.features(ref, sets=["R2C"])["R2C"]
    calls = []
    for s in mc.SEEDS:
        mdl = rc.rf_weighted(s)
        mdl.fit(XB, yB)
        calls.append(mdl.predict_proba(Xr)[:, 1])
    pr = np.mean(calls, axis=0)
    per_seed = [(c >= 0.5).mean() * 100 for c in calls]
    log(f"    called lipid by the final detection model (3 seeds): {np.mean(per_seed):.2f} ± {np.std(per_seed, ddof=1):.2f} % "
        f"(cross-validation sensitivity on LIPID MAPS formulas: see 92_detection_breakdown.log)")
    cn = np.array([mc.parsear(f)[0].get("C", 0) for f in ref])
    band = pd.cut(cn, [0, 15, 25, 35, 50, 1000], labels=["C<=15", "C16-25", "C26-35", "C36-50", "C>50"])
    rows = [{"group": "all", "formulas": len(ref), "called_lipid_pct": np.mean(per_seed)}]
    for b in ["C<=15", "C16-25", "C26-35", "C36-50", "C>50"]:
        mk = np.asarray(band == b)
        rows.append({"group": f"carbon band: {b}", "formulas": int(mk.sum()), "called_lipid_pct": float((pr[mk] >= 0.5).mean() * 100)})
        log(f"      {b:8s} n = {int(mk.sum()):6,}   {(pr[mk] >= 0.5).mean() * 100:6.2f} %")
    pd.DataFrame(rows).to_csv(mc.RES / "95_refmet_lipids_detection.csv", index=False)

    # ---------------------------------------------------------------- (b) N4a / N4b against COCONUT lipid-like formulas
    n4 = pd.read_csv(mc.DATOS / "n4_adversarial.csv")
    log(f"\n(b) COCONUT lipid-like formulas against the artificial sets (N4a {int((n4.capa == 'N4a').sum()):,}, N4b {int((n4.capa == 'N4b').sum()):,})")
    lip_like, any_coco = set(), set()
    z = zipfile.ZipFile(mc.DATOS / "coconut_csv_lite-09-2026.zip")
    cols = ["molecular_formula", "chemical_super_class", "np_classifier_pathway"]
    seen = {}
    with z.open("coconut_csv_lite-09-2026.csv") as fh:
        for chunk in pd.read_csv(fh, usecols=cols, chunksize=100000, low_memory=False):
            es = chunk.chemical_super_class.isin(COCO_SUPERCLASSES) | chunk.np_classifier_pathway.isin(COCO_PATHWAYS)
            for f, flag in zip(chunk.molecular_formula.astype(str), es):
                seen[f] = seen.get(f, False) or bool(flag)
    for f, flag in seen.items():
        c = canon(f)
        if c is None:
            continue
        any_coco.add(c)
        if flag:
            lip_like.add(c)
    log(f"    COCONUT: {len(seen):,} distinct formula strings, {len(any_coco):,} valid; lipid-like by the paper's rule: {len(lip_like):,}")
    res = []
    for capa in ("N4a", "N4b"):
        fs = n4.formula[n4.capa == capa]
        a, b = fs.isin(lip_like).mean() * 100, fs.isin(any_coco).mean() * 100
        res.append({"set": capa, "formulas": len(fs), "in_coconut_lipid_like_pct": a, "in_coconut_any_pct": b})
        log(f"    {capa}: {len(fs):,} formulas; {a:.2f} % coincide with a COCONUT lipid-like formula ({int(fs.isin(lip_like).sum()):,}); "
            f"{b:.2f} % with any COCONUT formula")
    pd.DataFrame(res).to_csv(mc.RES / "95_n4_in_coconut.csv", index=False)
    log("    (N4a/N4b exclude the 116 354 formulas of the final table and the measured sets, so these coincidences are formulas the "
        "pipeline never counted as non-lipids; COCONUT is only one of the catalogues in which such formulas could appear.)")

    # ---------------------------------------------------------------- (c) 83 408 vs 83 740
    nl = m.formula[yB == 0]
    bad = 0
    for f in nl:
        try:
            featurize(f)
        except ValueError:
            bad += 1
    log(f"\n(c) Non-lipid formulas in the final table {len(nl):,}; rejected by classifier/features.py: {bad:,}; usable {len(nl) - bad:,}")

    # ---------------------------------------------------------------- (d) confusion counts, category, seed 0
    d, _, FE = rc.load_entries_features(("R2", "R2C"))
    y = d.CATEGORY.to_numpy()
    pred, _, cl = rc.cached_oof("hs_A_formula", rc.rf_weighted, FE["R2C"], y, d.hill.to_numpy(), 0)
    ct = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="predicted")).reindex(index=cl, columns=cl, fill_value=0)
    ct.to_csv(mc.RES / "95_confusion_counts_category_seed0.csv")
    err = int((y != pred).sum())
    log(f"\n(d) Category, out-of-fold, seed 0: {err:,} errors in {len(y):,} entries (accuracy {100 - err / len(y) * 100:.2f} %); counts:")
    log(ct.to_string())
    log("\n  Saved: results/95_*.csv")


if __name__ == "__main__":
    main()
