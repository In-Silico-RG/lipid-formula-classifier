#!/usr/bin/env python
"""
48_novelty.py — Merged pipeline: flag formulas unlike any known lipid (port of the
lipid-calibrated check of archive/tool_v2/v2/novelty.py; §2.10).

Score = mean distance to the 5 nearest LIPID MAPS formulas in a standardized space of atom
counts, DBE, DBE/C and the ratios H/C, O/C, N/C (tool features and KNNNovelty detector,
imported from classifier/, not copied). Threshold: the out-of-fold scores of the lipid
formulas (5 folds, each distinct formula scored by a detector that never saw it) are cut so
that 95 % (and 99 %) of held-out lipid formulas pass; the acceptance rate is checked on 50
random halves. Fraction flagged is then reported for formulas the detector never saw:
training non-lipids (N1-N3), HMDB test non-lipids (script 33), artificial N4a and N4b, and
20,000 random chemically valid CHNOPS formulas (tool's random_formulas, seed 42).

Output: results/48_novelty.log + .csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import merged_common as mc  # noqa: E402

sys.path.insert(0, str(mc.BASE / "classifier"))
from detectors import KNNNovelty  # noqa: E402
from features import FEATURES, featurize  # noqa: E402
from nonlipid_eval import random_formulas  # noqa: E402

NOV = ["C_COUNT", "H_COUNT", "N_COUNT", "O_COUNT", "P_COUNT", "S_COUNT",
       "DBE", "DBE_PER_C", "H_C_RATIO", "O_C_RATIO", "N_C_RATIO"]
IDX = [FEATURES.index(f) for f in NOV]


def featz(fs):
    X, ok = [], []
    for f in fs:
        try:
            X.append(np.array(featurize(f))[IDX]); ok.append(f)
        except ValueError:
            pass
    return np.array(X), ok


def cut(scores, accept):
    n = len(scores)
    return float(np.sort(scores)[min(int(np.ceil((n + 1) * accept)), n) - 1])


def main():
    log = mc.logger("48_novelty")
    log("=" * 88)
    log("48 — FORMULAS UNLIKE ANY KNOWN LIPID: distance to the 5 nearest LIPID MAPS formulas")
    log("=" * 88)
    d, _ = mc.load_entries()
    Xl, lf = featz(sorted(d.hill.unique()))
    log(f"lipid formulas with tool features: {len(lf):,} of {d.hill.nunique():,}")
    oof = np.zeros(len(lf))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xl):
        oof[te] = KNNNovelty().fit(Xl[tr]).score(Xl[te])
    det = KNNNovelty().fit(Xl)

    m = pd.read_parquet(mc.DATOS / "master.parquet")
    n4 = pd.read_csv(mc.DATOS / "n4_adversarial.csv")
    lay = n4.columns[1]
    hm = pd.read_csv(mc.DATOS / "hmdb_test.csv")
    known = set(m.formula) | set(n4.formula) | set(hm.formula)
    rnd = [f for f in random_formulas(20000, 42)]
    rnd_new = [f for f in rnd if mc.hill(mc.parsear(f)[0]) not in known] if rnd else []
    tests = {"training non-lipids (N1-N3)": list(m.formula[m.label == "non_lipid"]),
             "HMDB test non-lipids": list(hm.formula),
             "artificial N4a (perturbations)": list(n4.formula[n4[lay] == "N4a"]),
             "artificial N4b (+21 Da)": list(n4.formula[n4[lay] == "N4b"]),
             "random valid CHNOPS, not in any table": rnd_new}
    rows = []
    rng = np.random.default_rng(0)
    for acc in (0.95, 0.99):
        t = cut(oof, acc)
        got = []
        for _ in range(50):
            half = rng.random(len(oof)) < 0.5
            got.append((oof[~half] <= cut(oof[half], acc)).mean() * 100)
        log(f"\n--- target acceptance {acc:.0%}: threshold {t:.4f}; held-out lipid formulas accepted "
            f"{np.mean(got):.2f} ± {np.std(got, ddof=1):.2f} % (50 halves)")
        cats = d.drop_duplicates("hill").set_index("hill").CATEGORY.reindex(lf).to_numpy()
        for c in sorted(set(cats)):
            log(f"    lipid formulas flagged, {c:22s} {(oof[cats == c] > t).mean() * 100:6.2f} %")
        rows.append({"target": acc, "set": "held-out lipid formulas", "n": len(lf),
                     "flagged_pct": (oof > t).mean() * 100})
        for name, fs in tests.items():
            X, ok = featz(fs)
            fl = (det.score(X) > t).mean() * 100 if len(ok) else np.nan
            rows.append({"target": acc, "set": name, "n": len(ok), "flagged_pct": fl})
            log(f"  {name:40s} n = {len(ok):6,}   flagged {fl:6.2f} %")
    pd.DataFrame(rows).to_csv(mc.RES / "48_novelty.csv", index=False)
    log("\n  Saved: results/48_novelty.csv")


if __name__ == "__main__":
    main()
