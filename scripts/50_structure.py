#!/usr/bin/env python
"""
50_structure.py — Merged pipeline: what knowing the structure would add (§2.11).

A weighted random forest on Morgan fingerprints (radius 2, 2,048 bits; RDKit, as in
17_structure_oracle.py) against the final formula-only model, on the SAME entries and the SAME
merged split (5-fold grouped by formula, seeds 0, 1, 2), scored molecule by molecule against
the category of each entry. Also on the entries whose formula is shared by several categories.
The structure model is a point of comparison, not a competitor (it is not bound by the limit).

Usage:  python 50_structure.py --set=R2C --model=RF
Output: results/50_structure_<set>_<model>.log + .csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
import merged_common as mc  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL = OPT["set"], OPT["model"]
JOBS = int(OPT.get("jobs", 8))
N_BITS = 2048


def formula_model(seed):
    import xgboost as xgb
    if MODEL.startswith("RF"):
        return RandomForestClassifier(n_estimators=500, n_jobs=JOBS, random_state=seed,
                                      class_weight="balanced" if MODEL == "RF-weighted" else None)
    return xgb.XGBClassifier(n_estimators=500, tree_method="hist", n_jobs=JOBS, verbosity=0, random_state=seed)


def main():
    log = mc.logger(f"50_structure_{SET}_{MODEL}")
    log(f"50 — STRUCTURE (Morgan fingerprints) vs FORMULA ({SET}, {MODEL}), molecule by molecule")
    d, _ = mc.load_entries()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
    fps = {}
    for mol in Chem.SDMolSupplier(str(mc.DATOS / "structures_extended.sdf"), removeHs=False, sanitize=True):
        if mol is not None and mol.HasProp("LM_ID"):
            fps[mol.GetProp("LM_ID")] = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    keep = d.LM_ID.isin(fps).to_numpy()
    log(f"entries {len(d):,}; with a readable structure {int(keep.sum()):,} (others left out)")
    d = d[keep].reset_index(drop=True)
    Xs = np.stack([fps[i] for i in d.LM_ID]).astype(np.float32)
    Xf = mc.features(d.hill.tolist(), sets=[SET])[SET]
    y, g = d.CATEGORY.to_numpy(), d.hill.to_numpy()
    shared = (d.groupby("hill").CATEGORY.transform("nunique") > 1).to_numpy()
    log(f"entries whose formula carries more than one category: {int(shared.sum()):,} ({shared.mean() * 100:.2f} %)")
    rows = []
    for s in mc.SEEDS:
        ps = mc.oof(lambda seed: RandomForestClassifier(n_estimators=500, n_jobs=JOBS, random_state=seed,
                                                        class_weight="balanced"), Xs, y, g, s)
        pf = mc.oof(formula_model, Xf, y, g, s)
        # upper limit on these entries: majority category of each formula, counted per entry
        lim = d.groupby("hill").CATEGORY.agg(lambda x: x.value_counts().iloc[0]).sum() / len(d) * 100
        rows.append({"seed": s, "structure": (ps == y).mean() * 100, "formula": (pf == y).mean() * 100,
                     "structure_shared": (ps == y)[shared].mean() * 100, "formula_shared": (pf == y)[shared].mean() * 100,
                     "upper_limit": lim})
        log(f"  seed {s}: structure {rows[-1]['structure']:.2f}  formula {rows[-1]['formula']:.2f}  "
            f"(shared formulas: {rows[-1]['structure_shared']:.2f} vs {rows[-1]['formula_shared']:.2f}); limit {lim:.2f}")
    t = pd.DataFrame(rows)
    log("\n  mean ± SD: " + "  ".join(f"{c} {t[c].mean():.2f} ± {t[c].std(ddof=1):.2f}" for c in t.columns if c != "seed"))
    t.to_csv(mc.RES / f"50_structure_{SET}_{MODEL}.csv", index=False)


if __name__ == "__main__":
    main()
