#!/usr/bin/env python
"""
02_ceiling.py — Phases 1.1 and 1.2 of the plan.

1.1  Theoretical ceiling: maximum accuracy attainable by ANY method that uses only the
     molecular formula. The Bayes-optimal classifier under formula-only information is
     computed (predicting, for each formula, its majority category) at the
     category (8) and main-class level.

1.2  Leakage: what fraction of the test set was already seen in training, according to the
     split scheme.

Output: results/02_ceiling.log  +  results/02_ceiling.csv
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cargar_lmsd as cargar_lmsd_comun, hill, parsear  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
SEMILLAS = [0, 1, 2, 3, 4]


def cargar_lmsd(anio, log):
    """LIPID MAPS of September 2026 (only version). `anio` is kept for compatibility
    with 20_entry_lookup.py and is ignored."""
    d = cargar_lmsd_comun(DATOS)
    hs = []
    for f in d["FORMULA"]:
        e, motivo = parsear(f)
        hs.append(None if motivo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna() & d["CATEGORY"].notna()].copy()
    log(f"  LMSD September 2026: {len(d):,} valid entries, {d['hill'].nunique():,} unique formulas")
    return d


def techo(d, etiqueta, nombre, log):
    """Bayes-optimal classifier under formula-only information."""
    tab = d.groupby(["hill", etiqueta]).size().rename("n").reset_index()
    por_formula = tab.groupby("hill")["n"]
    total_entradas = tab["n"].sum()
    aciertos_entradas = por_formula.max().sum()          # hits the majority label
    n_formulas = tab["hill"].nunique()
    n_etiq = tab.groupby("hill")[etiqueta].nunique()
    ambiguas = int((n_etiq > 1).sum())

    # weighted per unique formula: hit probability within each formula, averaged
    frac = (por_formula.max() / por_formula.sum())
    techo_formula = frac.mean()

    log(f"\n  --- {nombre} ---")
    log(f"    distinct labels                  {tab[etiqueta].nunique():>10,}")
    log(f"    unique formulas                  {n_formulas:>10,}")
    log(f"    ambiguous formulas (>1 label)   {ambiguas:>10,}  ({ambiguas/n_formulas*100:.2f} %)")
    log(f"    CEILING weighted by entries     {aciertos_entradas/total_entradas*100:>9.2f} %")
    log(f"    CEILING weighted by formula     {techo_formula*100:>9.2f} %")
    return {
        "level": nombre,
        "n_labels": int(tab[etiqueta].nunique()),
        "n_formulas": int(n_formulas),
        "n_ambiguous": ambiguas,
        "pct_ambiguous": round(ambiguas / n_formulas * 100, 3),
        "ceiling_entries": round(aciertos_entradas / total_entradas * 100, 3),
        "ceiling_formulas": round(techo_formula * 100, 3),
    }


def fuga(d, log):
    """Fraction of the test set whose formula already appears in training."""
    rng_res = []
    log("\n  --- Leakage by split scheme (mean of 5 seeds) ---")

    # (a) random 80/20 PER ROW — the usual scheme, and the one that produces the inflated figure
    fugas = []
    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(d))
        corte = int(0.8 * len(d))
        tr, te = d.iloc[idx[:corte]], d.iloc[idx[corte:]]
        vistas = set(tr["hill"])
        fugas.append(te["hill"].isin(vistas).mean())
    log(f"    80/20 random per row            {np.mean(fugas)*100:>9.2f} %  "
        f"(± {np.std(fugas)*100:.2f})")
    rng_res.append({"scheme": "80/20 random by row",
                    "leakage_pct": round(float(np.mean(fugas)) * 100, 3)})

    # (b) 80/20 grouped BY FORMULA — leakage zero by construction
    fugas = []
    formulas = d["hill"].unique()
    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(formulas))
        corte = int(0.8 * len(formulas))
        tr_f = set(formulas[perm[:corte]])
        te = d[~d["hill"].isin(tr_f)]
        fugas.append(te["hill"].isin(tr_f).mean())
    log(f"    80/20 grouped by formula       {np.mean(fugas)*100:>9.2f} %")
    rng_res.append({"scheme": "80/20 grouped by formula",
                    "leakage_pct": round(float(np.mean(fugas)) * 100, 3)})
    return rng_res


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "02_ceiling.log", "w")

    def log(m=""):
        print(m)
        fh.write(m + "\n")

    log("=" * 78)
    log("PHASES 1.1 and 1.2 — Theoretical ceiling and leakage quantification")
    log("=" * 78)
    log("\n[0] DATA")
    d26 = cargar_lmsd(2026, log)

    log("\n[1] THEORETICAL CEILING (LMSD September 2026)")
    log("  Bayes-optimal classifier under formula-only information:")
    log("  for each formula it predicts its majority label. No method that uses")
    log("  only the formula can exceed it.")
    filas = [techo(d26, "CATEGORY", "Category (8)", log),
             techo(d26, "MAIN_CLASS", "Main class", log)]

    log("\n[2] CEILING AT THE LIPID / NON-LIPID DETECTION LEVEL")
    m = pd.read_parquet(DATOS / "master.parquet")
    conflicto = int((m["in_lipidmaps"] & (m["in_refmet_nonlipid"] | m["in_chebi"] | m["in_coconut"])).sum())
    log(f"    formulas in LIPID MAPS and also in some non-lipid database: {conflicto:,}")
    log(f"    (lower bound on lipid/non-lipid ambiguity: includes real ambiguity and")
    log(f"     possible residual contamination of the negative databases)")
    total_lip = int(m["in_lipidmaps"].sum())
    log(f"    out of {total_lip:,} lipid formulas → {conflicto/total_lip*100:.2f} % in conflict")

    log("\n[3] LEAKAGE (LMSD September 2026)")
    fugas = fuga(d26, log)

    pd.DataFrame(filas).to_csv(RES / "02_ceiling.csv", index=False)
    pd.DataFrame(fugas).to_csv(RES / "02_leakage.csv", index=False)
    log("\n  Saved: results/02_ceiling.csv, results/02_leakage.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
