#!/usr/bin/env python
"""
54_audit_nonlipids.py — Audit of the non-lipid layers: formulas of known lipid classes labelled non-lipid.

Found through the Chlorella application (chlorella-maldi-classification, 2026-11-19): the common
SQDG formulas are absent from LIPID MAPS and present in ChEBI, so the curation labelled them
non_lipid and the classifier learned that sulfolipids are not lipids.

This script generates the neutral formulas of common lipid classes by sum composition (standard
head-group rules; total chain carbons n and double bonds d over wide ranges) and counts how many
formulas labelled non_lipid in master.parquet coincide with them, by class and by source layer
(RefMet, ChEBI, COCONUT). It does not change any data.

Usage:  python 54_audit_nonlipids.py
Output: results/54_audit_nonlipids.log + results/54_audit_nonlipids.csv (one row per suspect formula)
"""
import pandas as pd

import merged_common as mc  # noqa: E402

# class: (C offset, H offset, extra atoms, n range, d range, oxygen additions)
# neutral formula for sum composition n:d (+ox O): C(n+c) H(2n-2d+h) + extra (+ox O)
DI = (range(20, 51), range(0, 13))
MONO = (range(10, 27), range(0, 7))
RULES = {
    "MG": (3, 6, {"O": 4}, *MONO, (0,)), "DG": (3, 4, {"O": 5}, *DI, (0,)),
    "TG": (3, 2, {"O": 6}, range(36, 67), range(0, 16), (0,)),
    "MGDG": (9, 14, {"O": 10}, *DI, (0,)), "DGDG": (15, 24, {"O": 15}, *DI, (0,)),
    "SQDG": (9, 14, {"O": 12, "S": 1}, *DI, (0,)), "MGMG": (9, 16, {"O": 9}, *MONO, (0,)),
    "DGMG": (15, 26, {"O": 14}, *MONO, (0,)), "SQMG": (9, 16, {"O": 11, "S": 1}, *MONO, (0,)),
    "DGTS/DGTA": (10, 17, {"N": 1, "O": 7}, *DI, (0,)),
    "PA": (3, 5, {"O": 8, "P": 1}, *DI, (0,)), "PC": (8, 16, {"N": 1, "O": 8, "P": 1}, *DI, (0,)),
    "PE": (5, 10, {"N": 1, "O": 8, "P": 1}, *DI, (0,)), "PG": (6, 11, {"O": 10, "P": 1}, *DI, (0,)),
    "PI": (9, 15, {"O": 13, "P": 1}, *DI, (0,)), "PS": (6, 10, {"N": 1, "O": 10, "P": 1}, *DI, (0,)),
    "LPA": (3, 7, {"O": 7, "P": 1}, *MONO, (0,)), "LPC": (8, 18, {"N": 1, "O": 7, "P": 1}, *MONO, (0,)),
    "LPE": (5, 12, {"N": 1, "O": 7, "P": 1}, *MONO, (0,)), "LPG": (6, 13, {"O": 9, "P": 1}, *MONO, (0,)),
    "LPI": (9, 17, {"O": 12, "P": 1}, *MONO, (0,)), "LPS": (6, 12, {"N": 1, "O": 9, "P": 1}, *MONO, (0,)),
    "Cer": (0, 1, {"N": 1, "O": 3}, range(28, 53), range(0, 5), (0, 1, 2)),
    "HexCer": (6, 11, {"N": 1, "O": 8}, range(28, 53), range(0, 5), (0, 1, 2)),
    "CE": (27, 44, {"O": 2}, range(12, 25), range(0, 7), (0,)),
}


def hill(comp):
    order = ["C", "H"] + sorted(k for k in comp if k not in ("C", "H"))
    return "".join(f"{k}{comp[k] if comp[k] != 1 else ''}" for k in order if comp.get(k))


def generate():
    out = {}
    for cls, (c0, h0, extra, nr, dr, oxr) in RULES.items():
        for n in nr:
            for d in dr:
                for ox in oxr:
                    h = 2 * n - 2 * d + h0
                    if h <= 0:
                        continue
                    comp = {"C": n + c0, "H": h, **extra}
                    comp["O"] = comp.get("O", 0) + ox
                    out.setdefault(hill(comp), set()).add(cls)
    return out


def main():
    log = mc.logger("54_audit_nonlipids")
    m = pd.read_parquet(mc.DATOS / "master.parquet")
    gen = generate()
    assert "C43H80O10" in gen and "MGDG" in gen["C43H80O10"]          # MGDG 34:1
    assert "C42H82NO8P" in gen and "PC" in gen["C42H82NO8P"]          # PC 34:1
    assert "C43H80O12S" in gen and "SQDG" in gen["C43H80O12S"]        # SQDG 34:1
    log("54 — AUDIT: formulas of known lipid classes labelled non-lipid (master.parquet)")
    log(f"generated formulas: {len(gen):,} over {len(RULES)} classes")
    m["classes_generated"] = m.formula.map(lambda f: "|".join(sorted(gen.get(f, ()))))
    hit = m[m.classes_generated != ""]
    for lab in ("lipid", "non_lipid", "discarded"):
        s = hit[hit.label == lab]
        log(f"  label {lab:10s}: {len(s):6,} of {int((m.label == lab).sum()):,} formulas match a generated lipid formula")
    nl = hit[hit.label == "non_lipid"].copy()
    log(f"\nnon-lipid formulas that match a lipid class: {len(nl):,} "
        f"({100 * len(nl) / (m.label == 'non_lipid').sum():.2f} % of the non-lipid layer)")
    for src in ("in_refmet_nonlipid", "in_chebi", "in_coconut"):
        log(f"  {src:20s}: {int(nl[src].sum()):5,}")
    log(f"  also flagged as lipid by RefMet (in_refmet_lipid): {int(nl.in_refmet_lipid.sum()):,}")
    log("\nby class (a formula may match more than one class):")
    counts = {}
    for cs in nl.classes_generated:
        for c in cs.split("|"):
            counts[c] = counts.get(c, 0) + 1
    for c in RULES:
        if c in counts:
            tot = sum(1 for f, s in gen.items() if c in s)
            log(f"  {c:10s} {counts[c]:5,} non-lipid formulas (of {tot:,} generated)")
    nl[["formula", "classes_generated", "in_refmet_nonlipid", "in_refmet_lipid", "in_chebi", "in_coconut"]] \
        .sort_values("classes_generated").to_csv(mc.RES / "54_audit_nonlipids.csv", index=False)
    log("\n  Saved: results/54_audit_nonlipids.csv")


if __name__ == "__main__":
    main()
