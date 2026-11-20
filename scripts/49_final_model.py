#!/usr/bin/env python
"""
49_final_model.py — Merged pipeline: the released classifier (spec D10; manuscript §2.10).

Fits the final model on ALL LIPID MAPS entries for the category and the main class, and on
all task-B formulas for lipid detection; attaches the prediction-set thresholds calibrated on
the out-of-fold probabilities of 40_models.py (seed 0; Mondrian for the category and for lipid detection, with at least 30 calibration
entries per label, marginal for the main class and for labels below 30), and
the lipid-calibrated distance check of 48_novelty.py. The lipid-detection model is also
trained on the shifted-mass formulas N4b (data/n4_adversarial.csv) as non-lipids (§2.9: they
cost 0.3 points of sensitivity); its thresholds are calibrated on out-of-fold probabilities of
that same model (5-fold split of 40_models.py, seed 0, N4b in every training part, scored on
the real formulas only). Everything goes into one file read by
classifier/predict.py.

Usage:  python 49_final_model.py --set=R2C --model=RF [--jobs=8]
Output: classifier/lipid_classifier.joblib (not versioned) + results/49_final_model.log
"""
import datetime
import platform
import sys

import joblib
import numpy as np
import sklearn

import merged_common as mc  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL = OPT["set"], OPT["model"]
JOBS = int(OPT.get("jobs", 8))
ALPHAS = (0.10, 0.05, 0.01)
MIN_CAL = 30
OUT = mc.BASE / "classifier"


def make(seed, y):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    if MODEL.startswith("RF"):
        return RandomForestClassifier(n_estimators=500, n_jobs=JOBS, random_state=seed,
                                      class_weight="balanced" if MODEL == "RF-weighted" else None), {}
    return (xgb.XGBClassifier(n_estimators=500, tree_method="hist", n_jobs=JOBS, verbosity=0, random_state=seed),
            {"sample_weight": compute_sample_weight("balanced", y)} if MODEL == "XGB-weighted" else {})


def thresholds(P, yi, alpha):
    s = 1 - P[np.arange(len(yi)), yi]

    def q(v):
        k = int(np.ceil((len(v) + 1) * (1 - alpha)))
        return float(np.sort(v)[min(k, len(v)) - 1])
    marg = q(s)
    mond = np.array([q(s[yi == c]) if (yi == c).sum() >= MIN_CAL else marg for c in range(P.shape[1])])
    return {"marginal": marg, "mondrian": mond}


def main():
    log = mc.logger("49_final_model")
    OUT.mkdir(exist_ok=True)
    log(f"49 — FINAL MODEL · input list {SET} · model {MODEL}")
    d, _ = mc.load_entries()
    m, yB, _ = mc.load_detection()
    pkg = {"input_list": SET, "model_kind": MODEL, "levels": {},
           "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
           "data": "LIPID MAPS LMSD extended, September 2026; non-lipids N1-N3 (RefMet, ChEBI, COCONUT) "
                    "+ shifted-mass formulas N4b for lipid detection"}
    import pandas as pd
    n4 = pd.read_csv(mc.DATOS / "n4_adversarial.csv")
    n4b = n4.formula[n4[n4.columns[1]] == "N4b"].tolist()
    Xb = mc.features(n4b, sets=[SET])[SET]
    log(f"  N4b shifted-mass formulas added to the non-lipids of LIPID: {len(n4b):,}")
    for level, task, X_src, y in (("CATEGORY", "A", d.hill, d.CATEGORY.to_numpy()),
                                  ("MAIN_CLASS", "C", d.hill, d.MAIN_CLASS.to_numpy()),
                                  ("LIPID", "B", m.formula, np.where(yB == 1, "lipid", "non-lipid"))):
        X = mc.features(X_src.tolist(), sets=[SET])[SET]
        classes, yi = np.unique(y, return_inverse=True)
        if task == "B":                                          # yi: 0 non-lipid, 1 lipid
            yb = np.asarray(yB, int)
            P = np.zeros((len(yb), 2), dtype=np.float32)
            for tr, te in mc.splits(yb, X_src.to_numpy(), 0):
                Xtr, ytr = np.vstack([X[tr], Xb]), np.concatenate([yb[tr], np.zeros(len(Xb), int)])
                mf, kwf = make(0, ytr)
                mf.fit(Xtr, ytr, **kwf)
                P[te] = mf.predict_proba(X[te])
            yo = yb
            Xfit, yfit = np.vstack([X, Xb]), np.concatenate([yb, np.zeros(len(Xb), int)])
            mdl, kw = make(0, yfit)
            mdl.fit(Xfit, yfit, **kw)
            sens = ((P[:, 1] >= 0.5) & (yb == 1)).sum() / (yb == 1).sum() * 100
            log(f"  LIPID out-of-fold with N4b: sensitivity (p >= 0.5) {sens:.2f} %")
        else:
            mdl, kw = make(0, yi)
            mdl.fit(X, yi, **kw)
            z = np.load(mc.RES / f"40_oof_{SET}_{task}_{MODEL}.npz", allow_pickle=True)
            P = z["proba"]
            yo = np.searchsorted(z["classes"], z["y"])
        conf = {str(a): thresholds(P, yo, a) for a in ALPHAS}
        if task == "B":
            classes = np.array(["non-lipid", "lipid"])           # z["classes"] = [0, 1]
        pkg["levels"][level] = {"model": mdl, "classes": classes, "conformal": conf,
                                "default_mode": "marginal" if level == "MAIN_CLASS" else "mondrian"}
        mo = conf["0.05"]["mondrian"]
        log(f"  {level:10s} {len(y):,} units, {len(classes)} labels; default {pkg['levels'][level]['default_mode']}; "
            f"95 % thresholds: marginal {conf['0.05']['marginal']:.4f}, per label {mo.min():.4f}-{mo.max():.4f}")
    nov = mc._load("nov48", "48_novelty.py")
    Xl, lf = nov.featz(sorted(d.hill.unique()))
    oof = np.zeros(len(lf))
    from sklearn.model_selection import KFold
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xl):
        oof[te] = nov.KNNNovelty().fit(Xl[tr]).score(Xl[te])
    pkg["novelty"] = {"detector": nov.KNNNovelty().fit(Xl), "features": nov.NOV,
                      "thresholds": {str(a): nov.cut(oof, a) for a in (0.95, 0.99)}}
    joblib.dump(pkg, OUT / "lipid_classifier.joblib", compress=3)
    log(f"  saved {OUT / 'lipid_classifier.joblib'}")


if __name__ == "__main__":
    main()
