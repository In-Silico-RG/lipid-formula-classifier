#!/usr/bin/env python
"""
44_external.py — Merged pipeline: tests on independent data (spec D9; manuscript §2.7).

Sets (every formula of the set is removed from training before the model is fitted):
  GNPS     formulas of measured lipid spectra (PNNL positive/negative, HCE cell lysate);
           category from the laboratory abbreviation (ABREV of 11_external_validation.py,
           imported, not copied), LIPID MAPS majority when there is no abbreviation.
  MassBank formulas of measured spectra; lipids (in LIPID MAPS, category from LIPID MAPS)
           and measured non-lipids (formulas of our curated non-lipids); unknown excluded.
  NIST     SRM 1950 consensus lipids (data/NIST/nist_srm1950_consensus.csv):
           category and main class from the class names of the interlaboratory report.
  HMDB     non-lipid test set of script 33 (data/hmdb_test.csv), already disjoint from
           training; scored for detection only.
Scores: a prediction is correct if it is one of the labels of the formula in the set
(GNPS may carry several annotations per formula). Prediction sets: the 95 % thresholds
calibrated on LIPID MAPS (out-of-fold probabilities of 40_models.py; class-conditional for the
category, marginal for the class) applied to the external formulas: coverage (a label of the
formula is in the set) and mean set size, a case the conformal guarantee does not cover. Also the leakage scenario for GNPS and
MassBank (model trained on everything). Models trained with seeds 0, 1, 2 (only the model's
own randomness changes); mean ± SD.

Usage:  python 44_external.py --set=R2C --model=RF [--jobs=8]
Output: results/44_external_<set>_<model>.log + .csv + _predictions.csv
"""
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import merged_common as mc  # noqa: E402

m40 = None
OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL = OPT["set"], OPT["model"]
JOBS = int(OPT.get("jobs", 8))
ext11 = mc._load("ext11", "11_external_validation.py")   # ABREV map
SPECTRA = mc.DATOS / "spectra"
NIST = mc.DATOS / "NIST" / "nist_srm1950_consensus.csv"


def make_model(seed, y):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    if MODEL.startswith("RF"):
        m = RandomForestClassifier(n_estimators=500, n_jobs=JOBS, random_state=seed,
                                   class_weight="balanced" if MODEL == "RF-weighted" else None)
        return m, {}
    m = xgb.XGBClassifier(n_estimators=500, tree_method="hist", n_jobs=JOBS, verbosity=0, random_state=seed)
    return m, ({"sample_weight": compute_sample_weight("balanced", y)} if MODEL == "XGB-weighted" else {})


def fit_predict(Xtr, ytr, Xte, seed, all_classes=None):
    """Predicted labels; with all_classes, also probabilities aligned to all_classes."""
    classes, yc = np.unique(ytr, return_inverse=True)
    m, kw = make_model(seed, yc)
    m.fit(Xtr, yc, **kw)
    if all_classes is None:
        return classes[m.predict(Xte).astype(int)]
    P = np.zeros((len(Xte), len(all_classes)))
    P[:, np.searchsorted(all_classes, classes)] = m.predict_proba(Xte)
    return all_classes[P.argmax(1)], P


ALPHA, MIN_CAL = 0.05, 30          # as 45_conformal.py and 49_final_model.py


def lmsd_thresholds(task, mondrian):
    """95 % thresholds calibrated on LIPID MAPS: the out-of-fold probabilities of 40_models.py (seed 0)."""
    f = mc.RES / f"40_oof_{SET}_{task}_{MODEL}.npz"
    if not f.exists():
        return None, None
    z = np.load(f, allow_pickle=True)
    P, cl = z["proba"], z["classes"]
    yi = np.searchsorted(cl, z["y"])
    sc = 1 - P[np.arange(len(yi)), yi]

    def q(v):
        k = int(np.ceil((len(v) + 1) * (1 - ALPHA)))
        return np.sort(v)[min(k, len(v)) - 1]
    marg = q(sc)
    thr = np.array([q(sc[yi == c]) if mondrian and (yi == c).sum() >= MIN_CAL else marg for c in range(len(cl))])
    return cl, thr


def canon(f):
    """Hill form, or None if the parser rejects the formula (same rules as every source)."""
    e, mo = mc.parsear(f)
    return None if mo else mc.hill(e)


def strip(s):
    return mc.RE_CODIGO.sub("", str(s)).strip()


def main():
    log = mc.logger(f"44_external_{SET}_{MODEL}")
    log("=" * 88)
    log(f"44 — INDEPENDENT DATA · input list {SET} · model {MODEL} · formulas of each set withheld")
    log("=" * 88)
    d, acc = mc.load_entries()
    mB, yB, _ = mc.load_detection()
    maj_cat = d.groupby("hill").CATEGORY.agg(lambda s: s.value_counts().index[0]).to_dict()

    # ---------------------------------------------------------------- the sets
    g = pd.read_csv(SPECTRA / "formulas_experimental.csv").fillna("")
    g["formula"] = g.formula.map(canon)
    log(f"GNPS rows rejected by the parser: {int(g.formula.isna().sum()):,}")
    g = g[g.formula.notna()]
    truth_g, source_g, unmapped = defaultdict(set), {}, defaultdict(int)
    for f, ab in zip(g.formula, g.abrev):
        if ab:
            c = ext11.ABREV.get(ab)
            if c is None:
                unmapped[ab] += 1
                continue
            truth_g[f].add(c)
            source_g[f] = "annotation"
        else:
            truth_g.setdefault(f, set())
            source_g.setdefault(f, "LIPID MAPS")
    for f in list(truth_g):
        if not truth_g[f] and f in maj_cat:
            truth_g[f] = {maj_cat[f]}
    gn_all = sorted(truth_g)                       # all formulas: detection (all are lipids)
    gn_cat = [f for f in gn_all if truth_g[f]]     # with a category
    log(f"GNPS: {len(gn_all):,} formulas; with category {len(gn_cat):,} "
        f"({sum(source_g[f] == 'annotation' for f in gn_cat):,} from the laboratory annotation); "
        f"unmapped abbreviations (left out): {dict(unmapped)}")

    mb = pd.read_csv(SPECTRA / "formulas_massbank.csv")
    mb["formula"] = mb.formula.map(canon)
    mb = mb[mb.formula.notna()]
    mb_lip = [f for f in mb.formula[mb.en_lipidmaps] if f in maj_cat]
    mb_neg = list(mb.formula[mb.es_negativo_curado & ~mb.en_lipidmaps])
    log(f"MassBank: {len(mb_lip):,} lipid formulas, {len(mb_neg):,} measured non-lipids, "
        f"{int((~mb.en_lipidmaps & ~mb.es_negativo_curado).sum()):,} unknown (excluded)")

    ni = pd.read_csv(NIST)
    ni["formula"] = ni.formula.map(canon)
    ni["category"], ni["main_class"] = ni.category.map(strip), ni.main_class.astype(str).str.strip()
    ni_truth = {"category": ni.groupby("formula").category.agg(set).to_dict(),
                "main_class": ni.groupby("formula").main_class.agg(set).to_dict()}
    ni_f = sorted(ni_truth["category"])
    log(f"NIST: {len(ni):,} species, {len(ni_f):,} distinct formulas; in LIPID MAPS: "
        f"{sum(f in maj_cat for f in ni_f):,}")
    hm = pd.read_csv(mc.DATOS / "hmdb_test.csv")
    log(f"HMDB: {len(hm):,} non-lipid formulas never used in training")

    FE = mc.features(d.hill.tolist(), sets=[SET])[SET]
    FB = mc.features(mB.formula.tolist(), sets=[SET])[SET]

    def feats(fs):
        return mc.features(list(fs), sets=[SET])[SET]

    rows, preds = [], []

    def record(setname, task, scenario, per_seed, n):
        mu, sd = np.mean(per_seed), np.std(per_seed, ddof=1)
        rows.append({"set": setname, "task": task, "scenario": scenario, "n": n, "score": mu, "score_sd": sd})
        log(f"  {setname:9s} {task:22s} {scenario:22s} n = {n:5,}   {mu:6.2f} ± {sd:.2f} %")

    # ------------------------------------------------- category / class models
    for setname, fs, truth, levels in (
            ("GNPS", gn_cat, {"category": truth_g}, ["category"]),
            ("MassBank", mb_lip, {"category": {f: {maj_cat[f]} for f in mb_lip}}, ["category"]),
            ("NIST", ni_f, ni_truth, ["category", "main_class"])):
        Xte = feats(fs)
        scen = [("no leakage", ~d.hill.isin(set(fs)).to_numpy())]
        if setname != "NIST":
            scen.append(("leakage (all trained)", np.ones(len(d), bool)))
        for level in levels:
            col = "CATEGORY" if level == "category" else "MAIN_CLASS"
            y = d[col].to_numpy()
            cl, thr = lmsd_thresholds("A" if level == "category" else "C", mondrian=(level == "category"))
            for sname, keep in scen:
                per, cov, size = [], [], []
                for s in mc.SEEDS:
                    if cl is not None and sname == "no leakage":
                        p, P = fit_predict(FE[keep], y[keep], Xte, s, all_classes=cl)
                        S = (1 - P) <= thr[None, :]                    # 95 % prediction set of each formula
                        cov.append(np.mean([any(S[i, np.searchsorted(cl, t)] for t in truth[level][f] if t in cl)
                                            for i, f in enumerate(fs)]) * 100)
                        size.append(S.sum(1).mean())
                    else:
                        p = fit_predict(FE[keep], y[keep], Xte, s)
                    per.append(np.mean([pi in truth[level][f] for pi, f in zip(p, fs)]) * 100)
                    if s == 0 and sname == "no leakage":
                        preds += [{"set": setname, "level": level, "formula": f, "predicted": pi,
                                   "truth": ";".join(sorted(truth[level][f]))} for f, pi in zip(fs, p)]
                record(setname, level, sname, per, len(fs))
                if cov:
                    record(setname, f"{level}: 95 % set coverage", sname, cov, len(fs))
                    record(setname, f"{level}: 95 % set size", sname, size, len(fs))

    # ---------------------------------------------------------- detection
    for setname, fs, label in (("GNPS", gn_all, 1), ("MassBank", mb_lip, 1), ("MassBank", mb_neg, 0),
                               ("HMDB", list(hm.formula), 0)):
        keep = ~mB.formula.isin(set(fs)).to_numpy()
        Xte = feats(fs)
        per = []
        for s in mc.SEEDS:
            p = fit_predict(FB[keep], yB[keep], Xte, s)
            per.append(np.mean(p == label) * 100)
        what = "lipids found" if label == 1 else "non-lipids rejected"
        record(setname, f"detection: {what}", "no leakage", per, len(fs))

    pd.DataFrame(rows).to_csv(mc.RES / f"44_external_{SET}_{MODEL}.csv", index=False)
    pd.DataFrame(preds).to_csv(mc.RES / f"44_external_{SET}_{MODEL}_predictions.csv", index=False)
    log(f"\n  Saved: results/44_external_{SET}_{MODEL}.csv and _predictions.csv")


if __name__ == "__main__":
    main()
