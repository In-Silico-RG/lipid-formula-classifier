"""
20 — Accuracy of the lookup table under the two splits, measured per entry.

The manuscript claimed that the lookup table «obtains about 90 %» with the per-entry
split; that figure had been inferred from the leakage (89.94 %), not measured. Here it is measured.
Same data (LMSD 2026) and seeds as the leakage in 02_ceiling.py. Unit: the entry.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
t02 = importlib.import_module("02_ceiling")
RES = t02.RES


def tabla_de_busqueda(tr):
    """Most frequent family of each formula in the training set, and the global majority."""
    tab = tr.groupby("hill")["CATEGORY"].agg(lambda x: x.value_counts().index[0]).to_dict()
    return tab, tr["CATEGORY"].value_counts().index[0]


def main():
    fh = open(RES / "20_entry_lookup.log", "w")

    def log(m=""):
        print(m)
        fh.write(m + "\n")

    log("20 — Table lookup: per-entry accuracy under the two splits (5 seeds)")
    d = t02.cargar_lmsd(2026, log)
    filas = []
    for s in t02.SEMILLAS:
        rng = np.random.default_rng(s)
        # (a) per entry, as in the leakage of 02_ceiling.py
        idx = rng.permutation(len(d))
        c = int(0.8 * len(d))
        tr, te = d.iloc[idx[:c]], d.iloc[idx[c:]]
        tab, may = tabla_de_busqueda(tr)
        a = (te["hill"].map(tab).fillna(may).values == te["CATEGORY"].values).mean()
        # (b) per formula
        rng = np.random.default_rng(s)
        formulas = d["hill"].unique()
        perm = rng.permutation(len(formulas))
        tr_f = set(formulas[perm[:int(0.8 * len(formulas))]])
        tr, te = d[d["hill"].isin(tr_f)], d[~d["hill"].isin(tr_f)]
        tab, may = tabla_de_busqueda(tr)
        b = (te["hill"].map(tab).fillna(may).values == te["CATEGORY"].values).mean()
        filas.append({"seed": s, "by_entry_pct": a * 100, "by_formula_pct": b * 100})
    r = pd.DataFrame(filas)
    for col, nom in [("by_entry_pct", "per-entry split"),
                     ("by_formula_pct", "per-formula split")]:
        log(f"  {nom:<24} {r[col].mean():6.2f} % (± {r[col].std(ddof=0):.2f})")
    log(f"  difference               {r['by_entry_pct'].mean() - r['by_formula_pct'].mean():6.2f} points")
    r.to_csv(RES / "20_entry_lookup.csv", index=False)
    log("  Saved: results/20_entry_lookup.csv")


if __name__ == "__main__":
    main()
