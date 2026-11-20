#!/usr/bin/env python
"""
36_mz_only.py — Merged pipeline, supplement: classification when only the mass is measured
(port of the core of archive/tool_v2/v2/mz_formula.py and mz_library.py).

Test: 1,000 LIPID MAPS entries with only C, H, N, O, P, S drawn from the test part of fold 0
(seed 0) of the merged split; the final model (SET, MODEL) is trained on the other four parts.
The neutral monoisotopic mass of each test lipid is perturbed with a Gaussian error of 1, 2, 5
and 10 ppm and searched within ±3 sigma. Three ways of getting candidate formulas:
  formula known        the true formula (reference)
  all valid formulas   every CHNOPS formula in the window that obeys the valence rules
                       (enumerar_cerca of 15_phase6_adversarial.py, imported)
  library, known       formulas of the master table (LIPID MAPS + curated non-lipids N1-N3,
                       all open sources; HMDB is not used) in the window
  library, novel       the same library without the formulas of the test part, so the true
                       formula can never be found (a compound not yet catalogued)
The category is the one with the highest probability averaged over the candidates. Also
reported: median number of candidates and how often the true formula is among them.

Usage:  python 36_mz_only.py --set=R2C --model=XGB [--jobs=24]
Output: results/36_mz_only_<set>_<model>.log + .csv
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import merged_common as mc  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL = OPT["set"], OPT["model"]
JOBS = int(OPT.get("jobs", 8))
m15 = mc._load("m15", "15_phase6_adversarial.py")   # enumerar_cerca, masa_de (checked mass table)
N_TEST, PPMS = 1000, (1, 2, 5, 10)
CHNOPS = set("CHNOPS")


def make(seed, y):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    if MODEL.startswith("RF"):
        return RandomForestClassifier(n_estimators=500, n_jobs=JOBS, random_state=seed,
                                      class_weight="balanced" if MODEL == "RF-weighted" else None), {}
    return (xgb.XGBClassifier(n_estimators=500, tree_method="hist", n_jobs=JOBS, verbosity=0, random_state=seed),
            {"sample_weight": compute_sample_weight("balanced", y)} if MODEL == "XGB-weighted" else {})


def elements(f):
    e, _ = mc.parsear(f)
    return {k: v for k, v in e.items() if k != "_carga"}


def main():
    log = mc.logger(f"36_mz_only_{SET}_{MODEL}")
    log(f"36 — CLASSIFICATION FROM THE MASS ONLY · {SET} · {MODEL} (supplement)")
    d, _ = mc.load_entries()
    y, g = d.CATEGORY.to_numpy(), d.hill.to_numpy()
    tr, te = next(iter(mc.splits(y, g, 0)))
    classes, yi = np.unique(y, return_inverse=True)
    X = mc.features(d.hill.tolist(), sets=[SET])[SET]
    seen, ytr = np.unique(yi[tr], return_inverse=True)     # labels re-coded within the training part
    mdl, kw = make(0, ytr)
    mdl.fit(X[tr], ytr, **kw)

    chnops = np.array([set(elements(f)) <= CHNOPS for f in g])
    cand_te = te[chnops[te]]
    rng = np.random.default_rng(0)
    test = rng.choice(cand_te, size=min(N_TEST, len(cand_te)), replace=False)
    mass = np.array([m15.masa_de(elements(g[i])) for i in test])
    log(f"test lipids: {len(test):,} (CHNOPS, fold 0 test part); mass check C18H32O2 = "
        f"{m15.masa_de({'C': 18, 'H': 32, 'O': 2}):.6f} Da")

    master = pd.read_parquet(mc.DATOS / "master.parquet")
    lib = [f for f in master.formula[master.label.isin(["lipid", "non_lipid"])] if set(elements(f)) <= CHNOPS]
    lib_mass = np.array([m15.masa_de(elements(f)) for f in lib])
    order = np.argsort(lib_mass)
    lib, lib_mass = np.array(lib)[order], lib_mass[order]
    test_forms = set(g[te])
    log(f"library: {len(lib):,} CHNOPS formulas (LIPID MAPS + N1-N3); in the test part: "
        f"{sum(f in test_forms for f in lib):,}")

    cache = {}

    def proba(fs):
        new = [f for f in fs if f not in cache]
        if new:
            P = np.zeros((len(new), len(classes)))
            P[:, seen] = mdl.predict_proba(mc.features(new, sets=[SET])[SET])
            cache.update(zip(new, P))
        return np.array([cache[f] for f in fs])

    truth = yi[test]
    base = (proba(list(g[test])).argmax(1) == truth).mean() * 100
    rows = [{"scenario": "formula known", "ppm": 0, "accuracy": base, "median_candidates": 1,
             "true_formula_among_candidates": 100.0, "no_candidate": 0.0}]
    log(f"\n  formula known: accuracy {base:.2f} %")
    for ppm in PPMS:
        noisy = mass * (1 + rng.normal(0, ppm, len(mass)) * 1e-6)
        for scen in ("all valid formulas", "library, known", "library, novel"):
            hit, ncand, found, none = [], [], [], 0
            for i, m in zip(test, noisy):
                if scen == "all valid formulas":
                    cands = m15.enumerar_cerca(m, 3 * ppm, set())
                else:
                    lo = np.searchsorted(lib_mass, m * (1 - 3 * ppm * 1e-6))
                    hi = np.searchsorted(lib_mass, m * (1 + 3 * ppm * 1e-6), side="right")
                    cands = [f for f in lib[lo:hi] if not (scen == "library, novel" and f in test_forms)]
                ncand.append(len(cands))
                found.append(g[i] in cands)
                if not cands:
                    none += 1
                    hit.append(False)
                    continue
                hit.append(proba(list(cands)).mean(0).argmax() == yi[i])
            r = {"scenario": scen, "ppm": ppm, "accuracy": np.mean(hit) * 100, "median_candidates": float(np.median(ncand)),
                 "true_formula_among_candidates": np.mean(found) * 100, "no_candidate": none / len(test) * 100}
            rows.append(r)
            log(f"  {ppm:2d} ppm  {scen:20s} accuracy {r['accuracy']:6.2f} %   candidates (median) "
                f"{r['median_candidates']:6.0f}   true formula among them {r['true_formula_among_candidates']:6.2f} %   "
                f"no candidate {r['no_candidate']:5.2f} %")
    pd.DataFrame(rows).to_csv(mc.RES / f"36_mz_only_{SET}_{MODEL}.csv", index=False)


if __name__ == "__main__":
    main()
