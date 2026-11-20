#!/usr/bin/env python
"""
08_ceiling_extended.py — Redo the ceiling on LMSD_extended and under the two criteria.

Two things:
  1. Convert structures_extended.sdf to a reusable CSV (data/lmsd_extended.csv),
     keeping the alternative labels that the normal SDF does not include.
  2. Recompute the ceiling, which depends on the grading criterion:

     STRICT     "given a specific MOLECULE, do I hit ITS label?"
                The optimum predicts the majority label of the formula. It necessarily
                fails on minority isomers -> ceiling < 100 %.

     PERMISSIVE "given a FORMULA, do I give a label compatible with it?"
                The optimum chooses any one from the admissible set -> ceiling = 100 %.

These are two different questions and both are legitimate. The first matters if one wants to
identify the molecule; the second is the one that corresponds to MS1, where only
the formula is available and the aim is a defensible annotation.

Output: data/lmsd_extended.csv + results/08_ceiling_extended.log + .csv
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hill, parsear  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
VACIO = {"", "-", "N/A", "None", "nan"}
CAMPOS = ["LM_ID", "FORMULA", "CATEGORY", "ALT_CATEGORIES", "MAIN_CLASS",
          "ALT_MAIN_CLASSES", "SUB_CLASS", "ALT_SUB_CLASSES", "CLASS_LEVEL4",
          "EXACT_MASS", "INCHI_KEY"]


def leer_sdf(ruta):
    regs, cur, key = [], {}, None
    with open(ruta) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.startswith("> <"):
                key = ln[3:-1]
                continue
            if ln == "$$$$":
                if cur:
                    regs.append(cur)
                cur, key = {}, None
                continue
            if key and ln.strip():
                if key in CAMPOS:
                    cur[key] = cur.get(key, "") + ln.strip()
                key = None
    return pd.DataFrame(regs)


def limpiar(v):
    return {x.strip() for x in str(v).split(";") if x.strip() not in VACIO}


def techos(d, pri, alt, nombre, log, filas):
    """Ceiling under the two criteria, at the level of the indicated label."""
    admis = defaultdict(set)
    for f, sub in d.groupby("hill"):
        for _, r in sub.iterrows():
            if str(r.get(pri, "")).strip() not in VACIO:
                admis[f].add(str(r[pri]).strip())
            admis[f] |= limpiar(r.get(alt, ""))

    tab = d[d[pri].notna()].groupby(["hill", pri]).size().rename("n").reset_index()
    por_f = tab.groupby("hill")["n"]
    total = tab["n"].sum()
    techo_estricto = por_f.max().sum() / total * 100

    n_f = tab["hill"].nunique()
    amb_estricta = int((tab.groupby("hill")[pri].nunique() > 1).sum())
    amb_permisiva = sum(1 for f in admis if len(admis[f]) > 1)
    n_etiq = tab[pri].nunique()

    log(f"\n  --- {nombre} ---")
    log(f"    distinct labels                        {n_etiq:>8,}")
    log(f"    unique formulas                        {n_f:>8,}")
    log(f"    ambiguous due to isomerism (strict)   {amb_estricta:>8,}  "
        f"({amb_estricta/n_f*100:.2f} %)")
    log(f"    with more than one admissible label   {amb_permisiva:>8,}  "
        f"({amb_permisiva/len(admis)*100:.2f} %)")
    log(f"    CEILING STRICT criterion             {techo_estricto:>7.2f} %")
    log(f"    CEILING PERMISSIVE criterion            100.00 %   (by construction)")
    filas.append({"level": nombre, "n_labels": n_etiq, "n_formulas": n_f,
                  "ambiguous_strict": amb_estricta, "multi_admissible": amb_permisiva,
                  "ceiling_strict": round(techo_estricto, 3), "ceiling_permissive": 100.0})
    return techo_estricto


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "08_ceiling_extended.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    log("=" * 80)
    log("CEILING RECOMPUTED ON LMSD_extended AND UNDER THE TWO CRITERIA")
    log("=" * 80)

    d = leer_sdf(DATOS / "structures_extended.sdf")
    log(f"\n  LMSD_extended: {len(d):,} entries")
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    malas = int(d["hill"].isna().sum())
    d = d[d["hill"].notna()].copy()
    log(f"  unparseable formula: {malas:,}  →  {len(d):,} entries remain, "
        f"{d['hill'].nunique():,} unique formulas")

    d.to_csv(DATOS / "lmsd_extended.csv", index=False)
    log(f"  Saved: data/lmsd_extended.csv")

    log("\n" + "=" * 80)
    log("THE TWO CRITERIA")
    log("=" * 80)
    log("  STRICT     : do I hit the label of THIS molecule? The optimum predicts the")
    log("              majority one of the formula and fails on minority isomers.")
    log("  PERMISSIVE : do I give a label compatible with THIS formula? The optimum chooses")
    log("              any one from the admissible set, so it never fails.")

    filas = []
    tc = techos(d, "CATEGORY", "ALT_CATEGORIES", "Category", log, filas)
    tm = techos(d, "MAIN_CLASS", "ALT_MAIN_CLASSES", "Main class", log, filas)
    ts = techos(d, "SUB_CLASS", "ALT_SUB_CLASSES", "Subclass", log, filas)

    log("\n" + "=" * 80)
    log("COMPARISON WITH PREVIOUS FIGURES (LMSD Apr-2026, strict criterion)")
    log("=" * 80)
    log(f"  {'Level':18s} {'before':>10s} {'now':>10s} {'Δ':>8s}")
    for nom, antes, ahora in [("Category", 97.44, tc), ("Main class", 91.20, tm)]:
        log(f"  {nom:18s} {antes:9.2f}% {ahora:9.2f}% {ahora-antes:+8.2f}")
    log(f"  {'Subclass':18s} {'—':>10s} {ts:9.2f}%")

    log("\n" + "=" * 80)
    log("READING")
    log("=" * 80)
    log("  The ceiling of the strict criterion is NOT a property of the problem: it is a")
    log("  property of HOW IT IS GRADED. Under the permissive criterion -the one that corresponds")
    log("  to MS1, where only the formula is available- there is no ceiling below 100 %.")
    log("")
    log("  Practical consequence: the room for improvement is LARGER than it seemed.")
    log("  The current model gives 85,90 % under the permissive criterion, so 14 points remain,")
    log("  not the 8-9 that the strict ceiling suggested.")

    pd.DataFrame(filas).to_csv(RES / "08_ceiling_extended.csv", index=False)
    log("\n  Saved: results/08_ceiling_extended.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
