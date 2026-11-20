#!/usr/bin/env python
"""
16_mona_stream.py — The four MoNA numbers (E4, third database), reading the MSP in streaming mode.

MoNA-export-LC-MS-MS_Spectra.msp is 9.46 GB inside the zip; it is not decompressed to disk.
Usage:  unzip -p data/spectra/MoNA-LCMSMS-experimental.msp.zip | python scripts/16_mona_stream.py

Format: NIST-MSP records ("Name:", "Formula:", "Instrument_type:", "Comments:" with SMILES...),
separated by a blank line. MoNA contains MassBank and GNPS; deduplication is by formula.
Output: data/spectra/formulas_mona.csv and results/16_mona.log
"""
import collections
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from common import hill, parsear  # noqa: E402

lm = pd.read_parquet(BASE / "data" / "master.parquet")
en_lm = set(lm.loc[lm.in_lipidmaps, "formula"])
en_neg = set(lm.loc[lm.is_negative, "formula"])
desc = set(lm.loc[lm.label == "discarded", "formula"])
cat_lm = dict(zip(lm.formula, lm.categories))

KEYS = {"NAME", "FORMULA", "INSTRUMENT_TYPE", "SPECTRUM_TYPE", "ION_MODE", "DB#"}
n = 0
formulas = collections.Counter()
insts = collections.Counter()
fuentes = collections.Counter()
ms2 = 0
sin_formula = 0
rechaz = collections.Counter()
cur = {}


def cerrar(cur):
    global n, sin_formula, ms2
    n += 1
    if cur.get("SPECTRUM_TYPE", "").upper().startswith("MS2"):
        ms2 += 1
    insts[cur.get("INSTRUMENT_TYPE", "")] += 1
    dbid = cur.get("DB#", "")
    fuentes[re.sub(r"[0-9].*$", "", dbid)[:12] or "?"] += 1   # identifier prefix ≈ library
    f = cur.get("FORMULA", "")
    if not f:
        sin_formula += 1
        return
    e, mo = parsear(f)
    if mo:
        rechaz[mo.split(":")[0]] += 1
        return
    formulas[hill(e)] += 1


for ln in sys.stdin:
    ln = ln.rstrip("\r\n")
    if not ln.strip():
        if cur:
            cerrar(cur)
            cur = {}
        continue
    if ln[0].isdigit():
        continue
    i = ln.find(":")
    if i > 0:
        k = ln[:i].strip().upper()
        if k in KEYS and k not in cur:
            cur[k] = ln[i + 1:].strip()
if cur:
    cerrar(cur)

u = set(formulas)
out = []
out.append(f"MoNA LC-MS/MS experimental: {n:,} spectra ({ms2:,} MS2) · without FORMULA {sin_formula:,} · rejected {dict(rechaz)}")
out.append(f"  UNIQUE FORMULAS: {len(u):,}")
out.append(f"    ∩ LIPID MAPS (measured lipids):           {len(u & en_lm):,}")
out.append(f"    ∩ curated negatives (measured non-lipids): {len(u & en_neg):,}")
out.append(f"    ∩ discarded (RefMet lipid-like):          {len(u & desc):,}")
out.append(f"    in no table:                              {len(u - en_lm - en_neg - desc):,}")
cats = collections.Counter()
for f in u & en_lm:
    for k in str(cat_lm.get(f, "")).split("|"):
        if k:
            cats[k] += 1
tot = sum(cats.values()) or 1
out.append("  categories of the measured lipids: " + ", ".join(f"{k} {v} ({v / tot * 100:.0f}%)" for k, v in cats.most_common(8)))
out.append("  instruments: " + str(dict(insts.most_common(6))))
out.append("  identifier prefixes (≈ library): " + str(dict(fuentes.most_common(12))))
pd.DataFrame([{"formula": f, "n_spectra": c, "in_lipidmaps": f in en_lm, "is_curated_negative": f in en_neg}
              for f, c in formulas.items()]).to_csv(BASE / "data" / "spectra" / "formulas_mona.csv", index=False)
out.append("  saved data/spectra/formulas_mona.csv")
txt = "\n".join(out)
print(txt)
(BASE / "results").mkdir(exist_ok=True)
(BASE / "results" / "16_mona.log").write_text(txt + "\n")
