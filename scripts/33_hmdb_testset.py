#!/usr/bin/env python
"""
33_hmdb_testset.py — Merged pipeline: HMDB 5.0 as an EXTERNAL non-lipid test set (spec D1).

HMDB is never used for training. This script only builds the test set; the final model is
scored on it by the evaluation script once the model is chosen.

Input:  data/HMDB/hmdb_formula_taxonomy.csv, written by data/HMDB/extract_hmdb.py
        from the official hmdb_metabolites.zip (HMDB 5.0, XML dated 2021-10-23; SHA-256 in
        data/hmdb_sha256.txt). data/master.parquet (script 01) for the formulas already used.
Filter (same as the lipids-class tool, following the LIPID MAPS definition of a lipid):
  - drop metabolites without ClassyFire taxonomy;
  - drop super-classes "Lipids and lipid-like molecules", "Phenylpropanoids and polyketides",
    "Lignans, neolignans and related compounds", "Hydrocarbons" (LIPID MAPS counts many of
    these compounds as lipids);
  - drop formulas our parser rejects (common.parsear, same rules as every other source);
  - drop every formula already in the master table (LIPID MAPS lipids, training non-lipids
    N1-N3, and the set-aside RefMet lipids), so that the test holds only formulas the model
    has never seen.
Output: data/hmdb_test.csv (one row per distinct formula; not versioned) + results/33_hmdb_testset.log
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import hill, parsear  # noqa: E402

BASE = HERE.parent
DATOS, RES = BASE / "data", BASE / "results"
SRC = DATOS / "HMDB" / "hmdb_formula_taxonomy.csv"
EXCLUDED_SUPER = {
    "Lipids and lipid-like molecules",
    "Phenylpropanoids and polyketides",
    "Lignans, neolignans and related compounds",
    "Hydrocarbons",
}


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "33_hmdb_testset.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")

    h = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    log("=" * 84)
    log("33 — HMDB 5.0 AS AN EXTERNAL NON-LIPID TEST SET (never used for training)")
    log("=" * 84)
    log(f"HMDB metabolites read: {len(h):,}  (source: {SRC.name})")
    log(f"  by status: {dict(Counter(h.status))}")

    steps = []
    no_tax = h.super_class.str.strip() == ""
    steps.append(("without ClassyFire taxonomy", int(no_tax.sum())))
    h = h[~no_tax]
    lip = h.super_class.isin(EXCLUDED_SUPER)
    log("  removed by super-class: " + ", ".join(f"{k} {int((h.super_class == k).sum()):,}" for k in sorted(EXCLUDED_SUPER)))
    steps.append(("lipid-like super-class", int(lip.sum())))
    h = h[~lip].copy()

    hs, why = [], Counter()
    for f in h.formula:
        e, mo = parsear(f)
        hs.append(None if mo else hill(e))
        if mo:
            why[mo.split(":")[0]] += 1
    h["hill"] = hs
    steps.append(("formula rejected by the parser", int(h.hill.isna().sum())))
    log(f"  parser rejections by reason: {dict(why)}")
    h = h[h.hill.notna()]

    m = pd.read_parquet(DATOS / "master.parquet", columns=["formula", "label"])
    known = set(m.formula)
    seen = h.hill.isin(known)
    log(f"  formulas already in the master table, by label there: "
        f"{dict(Counter(m.set_index('formula').loc[h.hill[seen].unique(), 'label']))}")
    steps.append(("formula already used (master table)", int(seen.sum())))
    h = h[~seen]

    log("\nfilter (metabolites removed at each step):")
    for s, n in steps:
        log(f"  {s:40s} {n:8,}")
    t = (h.groupby("hill")
           .agg(n_metabolites=("accession", "size"),
                accessions=("accession", lambda s: ";".join(s)),
                super_classes=("super_class", lambda s: ";".join(sorted(set(s)))),
                statuses=("status", lambda s: ";".join(sorted(set(s)))))
           .reset_index().rename(columns={"hill": "formula"}))
    log(f"\nTEST SET: {len(h):,} metabolites, {len(t):,} distinct formulas never seen in training")
    log("  top super-classes (metabolites): " +
        ", ".join(f"{k} {v:,}" for k, v in Counter(h.super_class).most_common(8)))
    t.to_csv(DATOS / "hmdb_test.csv", index=False)
    log("\n  Saved: data/hmdb_test.csv")


if __name__ == "__main__":
    main()
