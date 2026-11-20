#!/usr/bin/env python
"""
52_relatives.py — Merged pipeline: relatives on both sides of the split (Limitations; checklist §6).

The split by formula keeps every entry of a formula on one side, but formulas of the same
homologous series (differing by CH2, C2H4 or H2) can fall on both sides. For the out-of-fold
predictions of the final model (40_models.py, seed 0, the same split recomputed here), each test
entry is marked if its formula has such a relative in the training part of its fold; accuracy
is compared between entries with and without a relative. If relatives raise the score, the
split by formula is still somewhat optimistic for formulas of new series.

Usage:  python 52_relatives.py --set=R2C --model=XGB
Output: results/52_relatives_<set>_<model>.log + .csv
"""
import sys

import numpy as np
import pandas as pd

import merged_common as mc  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL = OPT["set"], OPT["model"]
STEPS = [{"C": 1, "H": 2}, {"C": -1, "H": -2}, {"C": 2, "H": 4}, {"C": -2, "H": -4}, {"H": 2}, {"H": -2}]


def relatives(f):
    e, _ = mc.parsear(f)
    out = []
    for st in STEPS:
        r = dict(e)
        for k, v in st.items():
            r[k] = r.get(k, 0) + v
        if all(v >= 0 for k, v in r.items() if k != "_carga") and r.get("C", 0) > 0:
            out.append(mc.hill({k: v for k, v in r.items() if k == "_carga" or v > 0}))
    return out


def main():
    log = mc.logger(f"52_relatives_{SET}_{MODEL}")
    log(f"52 — HOMOLOGOUS RELATIVES ON BOTH SIDES OF THE SPLIT · {SET} · {MODEL} · seed 0")
    d, _ = mc.load_entries()
    rel = {f: relatives(f) for f in d.hill.unique()}
    rows = []
    for task, col in (("A", "CATEGORY"), ("C", "MAIN_CLASS")):
        z = np.load(mc.RES / f"40_oof_{SET}_{task}_{MODEL}.npz", allow_pickle=True)
        y, f = z["y"], z["formulas"]
        assert (f == d.hill.to_numpy()).all() and (y == d[col].to_numpy()).all(), "entry order differs from 40"
        p = z["classes"][z["proba"].argmax(1)]
        has = np.zeros(len(y), bool)
        for tr, te in mc.splits(y, f, 0):
            train_f = set(f[tr])
            has[te] = [any(r in train_f for r in rel[x]) for x in f[te]]
        ok = p == y
        r = {"task": task, "share_with_relative": has.mean() * 100, "accuracy_with_relative": ok[has].mean() * 100,
             "accuracy_without_relative": ok[~has].mean() * 100, "n_with": int(has.sum()), "n_without": int((~has).sum())}
        rows.append(r)
        log(f"  task {task}: {r['share_with_relative']:.1f} % of test entries have a homologue (±CH2, ±C2H4, ±H2) in "
            f"training; accuracy {r['accuracy_with_relative']:.2f} % with vs {r['accuracy_without_relative']:.2f} % without")
    pd.DataFrame(rows).to_csv(mc.RES / f"52_relatives_{SET}_{MODEL}.csv", index=False)


if __name__ == "__main__":
    main()
