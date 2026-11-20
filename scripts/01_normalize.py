#!/usr/bin/env python
"""
01_normalize.py — Phase 0.3-0.5 of the plan.

Normalizes the formulas from all sources to Hill notation, builds the master table of
UNIQUE FORMULAS and removes from the negatives every formula present in LIPID MAPS.

Sources:
  positives  LMSD extended, Sept. 2026   (LIPID MAPS; single version)
  N1         RefMet non-lipids           (real polar metabolites)
  N2         ChEBI                       (chemical coverage)
  N3         COCONUT                     (natural products, hard negatives)

Output: data/master.parquet  +  results/01_normalization.log
"""
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"

CATS = ["Fatty Acyls", "Glycerolipids", "Glycerophospholipids", "Polyketides",
        "Prenol Lipids", "Saccharolipids", "Sphingolipids", "Sterol Lipids"]

from common import cargar_lmsd, hill, parsear  # noqa: E402


# Roots of the ChEBI ontology whose descendants must NOT be used as negatives.
# This is deliberately conservative: it is preferable to lose negatives than to introduce
# lipids labelled as non-lipids.
RAICES_LIPIDICAS = {
    "CHEBI:18059": "lipid",
    "CHEBI:35341": "steroid",
    "CHEBI:26873": "terpenoid",
    "CHEBI:26188": "polyketide",
    "CHEBI:35366": "fatty acid",
    "CHEBI:24913": "isoprenoid",
}
# COCONUT classes that correspond to LIPID MAPS lipid categories.
COCO_SUPERCLASES = {"Lipids and lipid-like molecules", "Phenylpropanoids and polyketides"}
COCO_PATHWAYS = {"Terpenoids", "Fatty acids", "Polyketides"}


def lipidos_chebi(ruta, log):
    """Transitive closure of is_a under the ChEBI lipid roots."""
    hijos = defaultdict(set)
    cur = None
    with gzip.open(ruta, "rt") as fh:
        for ln in fh:
            if ln.startswith("id: CHEBI:"):
                cur = ln[4:].strip()
            elif ln.startswith("is_a:") and cur:
                hijos[ln[5:].strip().split("!")[0].strip()].add(cur)
    vistos = set(RAICES_LIPIDICAS)
    pila = list(RAICES_LIPIDICAS)
    while pila:
        for h in hijos.get(pila.pop(), ()):
            if h not in vistos:
                vistos.add(h)
                pila.append(h)
    log(f"  ChEBI ontology: {len(vistos):,} lipid terms under "
        f"{len(RAICES_LIPIDICAS)} roots ({', '.join(RAICES_LIPIDICAS.values())})")
    return vistos


def normalizar_serie(valores, etiqueta, log):
    """Normalizes an iterable of formulas. Returns (list_of_hill_or_None, counters)."""
    salida, motivos = [], defaultdict(int)
    for v in valores:
        e, motivo = parsear(v)
        if motivo:
            motivos[motivo.split(":")[0]] += 1
            salida.append(None)
        else:
            salida.append(hill(e))
    n, ok = len(salida), sum(x is not None for x in salida)
    log(f"  {etiqueta:26s} {n:>9,} rows → {ok:>9,} valid formulas ({ok/n*100:5.1f} %)")
    for m, c in sorted(motivos.items(), key=lambda x: -x[1]):
        log(f"      discarded for {m:14s} {c:>8,}")
    return salida, motivos


def main():
    RES.mkdir(exist_ok=True)
    salida_log = open(RES / "01_normalization.log", "w")

    def log(msg=""):
        print(msg)
        salida_log.write(msg + "\n")

    log("=" * 78)
    log("PHASE 0.3-0.5 — Formula normalization and master table")
    log("=" * 78)

    # ---------------------------------------------------------------- positives
    log("\n[1] POSITIVES — LIPID MAPS")
    d = cargar_lmsd(DATOS)
    d["hill"], _ = normalizar_serie(d["FORMULA"], "LMSD September 2026", log)
    lmsd = d[d["hill"].notna()].copy()

    # ---------------------------------------------------------------- negatives
    log("\n[2] NEGATIVES")

    ref = pd.read_csv(DATOS / "refmet.csv", low_memory=False)
    ref.columns = [c.strip() for c in ref.columns]
    ref_no = ref[~ref["super_class"].isin(CATS)].copy()
    ref_si = ref[ref["super_class"].isin(CATS)].copy()
    log(f"  RefMet: {len(ref):,} rows → {len(ref_si):,} lipid, {len(ref_no):,} non-lipid")
    ref_no["hill"], _ = normalizar_serie(ref_no["formula"], "N1 RefMet non-lipids", log)
    ref_si["hill"], _ = normalizar_serie(ref_si["formula"], "RefMet lipids (external)", log)

    lip_chebi = lipidos_chebi(DATOS / "chebi_lite.obo.gz", log)
    with gzip.open(DATOS / "chemical_data.tsv.gz", "rt") as fh:
        chebi = pd.read_csv(fh, sep="\t", low_memory=False)
    ids = "CHEBI:" + chebi["compound_id"].astype("Int64").astype(str)
    es_lip = ids.isin(lip_chebi)
    log(f"  ChEBI: {len(chebi):,} rows → {int(es_lip.sum()):,} lipid by ontology, excluded from N2")
    chebi = chebi[~es_lip].copy()
    chebi["hill"], _ = normalizar_serie(chebi["formula"], "N2 ChEBI (non-lipid)", log)

    with gzip.open(DATOS / "coconut_formulas.tsv.gz", "rt") as fh:
        coco = pd.read_csv(fh, sep="\t", low_memory=False)
    es_lip = (coco["chemical_super_class"].isin(COCO_SUPERCLASES)
              | coco["np_classifier_pathway"].isin(COCO_PATHWAYS))
    log(f"  COCONUT: {len(coco):,} rows → {int(es_lip.sum()):,} lipid by ClassyFire/NPClassifier, "
        f"excluded from N3")
    coco = coco[~es_lip].copy()
    coco["hill"], _ = normalizar_serie(coco["molecular_formula"], "N3 COCONUT (non-lipid)", log)

    # ------------------------------------------------------- master table
    log("\n[3] MASTER TABLE — one row per unique formula")

    maestro = defaultdict(lambda: {
        "n_lmsd": 0,
        "categories": set(), "classes": set(),
        "in_refmet_nonlipid": False, "in_refmet_lipid": False,
        "in_chebi": False, "in_coconut": False,
    })

    for f, cat, cls in zip(lmsd["hill"], lmsd["CATEGORY"], lmsd["MAIN_CLASS"]):
        r = maestro[f]
        r["n_lmsd"] += 1
        if pd.notna(cat):
            r["categories"].add(cat)
        if pd.notna(cls):
            r["classes"].add(cls)

    for f in ref_no["hill"].dropna():
        maestro[f]["in_refmet_nonlipid"] = True
    for f in ref_si["hill"].dropna():
        maestro[f]["in_refmet_lipid"] = True
    for f in chebi["hill"].dropna():
        maestro[f]["in_chebi"] = True
    for f in coco["hill"].dropna():
        maestro[f]["in_coconut"] = True

    filas = []
    for f, r in maestro.items():
        filas.append({
            "formula": f,
            "n_lmsd": r["n_lmsd"],
            "categories": "|".join(sorted(r["categories"])),
            "n_categories": len(r["categories"]),
            "classes": "|".join(sorted(r["classes"])),
            "n_classes": len(r["classes"]),
            "in_lipidmaps": r["n_lmsd"] > 0,
            "in_refmet_nonlipid": r["in_refmet_nonlipid"],
            "in_refmet_lipid": r["in_refmet_lipid"],
            "in_chebi": r["in_chebi"],
            "in_coconut": r["in_coconut"],
        })
    m = pd.DataFrame(filas)

    # ------------------------------- Phase 0.5: cleaning of contaminated negatives
    log("\n[4] NEGATIVE CLEANING (Phase 0.5)")
    log("  Rule: no formula present in LIPID MAPS may be used as a negative.")
    for col, nombre in [("in_refmet_nonlipid", "N1 RefMet"), ("in_chebi", "N2 ChEBI"),
                        ("in_coconut", "N3 COCONUT")]:
        total = int(m[col].sum())
        choca = int((m[col] & m["in_lipidmaps"]).sum())
        log(f"  {nombre:12s} {total:>8,} unique formulas · {choca:>7,} also in LIPID MAPS "
            f"({choca/total*100:4.1f} %) → excluded from the negatives")

    m["is_negative"] = (~m["in_lipidmaps"]) & (
        m["in_refmet_nonlipid"] | m["in_chebi"] | m["in_coconut"])
    m["label"] = "discarded"
    m.loc[m["in_lipidmaps"], "label"] = "lipid"
    m.loc[m["is_negative"], "label"] = "non_lipid"

    # ------------------------------------------------------------------ summary
    log("\n[5] SUMMARY")
    log(f"  Total unique formulas          {len(m):>10,}")
    log(f"  Labelled as lipid              {int((m.label=='lipid').sum()):>10,}")
    log(f"  Labelled as non-lipid          {int((m.label=='non_lipid').sum()):>10,}")
    log(f"  Discarded                      {int((m.label=='discarded').sum()):>10,}")

    lip = m[m.label == "lipid"]
    log(f"\n  Of the lipid formulas:")
    log(f"    with a single category       {int((lip.n_categories==1).sum()):>10,}")
    log(f"    ambiguous (>1 category)      {int((lip.n_categories>1).sum()):>10,}")
    amb = (lip.n_categories > 1).sum() / max((lip.n_categories >= 1).sum(), 1) * 100
    log(f"    → category-level ambiguity     {amb:>8.2f} %")

    log(f"\n  Compression by deduplication:")
    log(f"    LMSD entries (Sept. 2026)    {int(m.n_lmsd.sum()):>10,}")
    log(f"    unique formulas              {int((m.n_lmsd>0).sum()):>10,}")
    log(f"    → factor                     {m.n_lmsd.sum()/max((m.n_lmsd>0).sum(),1):>10.1f}x")

    m.to_parquet(DATOS / "master.parquet", index=False)
    log(f"\n  Saved: data/master.parquet ({len(m):,} rows)")
    salida_log.close()


if __name__ == "__main__":
    sys.exit(main())
