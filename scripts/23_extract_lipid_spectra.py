#!/usr/bin/env python
"""
23_extract_lipid_spectra.py — MS/MS spectra of lipids from MoNA, with the REAL class of each molecule.

To know how much adding MS/MS helps the formula classifier, a table of spectra is needed whose
molecule is in LIPID MAPS: the class is taken from the molecule (via the first part of the InChIKey, which
encodes connectivity), not from the formula; this way isomers of the same formula keep their own class.

Usage:  unzip -p data/spectra/MoNA-LCMSMS-experimental.msp.zip | python scripts/23_extract_lipid_spectra.py
Output: data/spectra/ms2_lipids.pkl (one row per spectrum, at most 25 per molecule) and results/23_extract_lipid_spectra.log
"""
import sys, re, pickle, collections
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from common import hill, parsear  # noqa: E402
lm = pd.read_csv(BASE / "data" / "lmsd_extended.csv", usecols=["CATEGORY", "MAIN_CLASS", "SUB_CLASS", "INCHI_KEY", "FORMULA"])
lm["ik"] = lm.INCHI_KEY.str[:14]
mol = lm.drop_duplicates("ik").set_index("ik")
MAX_POR_MOL = 25
cuenta = collections.Counter(); filas = []; leidos = 0; ms2 = 0; casan = 0
cur = {}; picos = []; leyendo = False

def cerrar():
    global leidos, ms2, casan
    leidos += 1
    if not cur.get("Spectrum_type", "").upper().startswith("MS2"): return
    ms2 += 1
    ik = cur.get("InChIKey", "")[:14]
    if ik not in mol.index: return
    casan += 1
    if cuenta[ik] >= MAX_POR_MOL: return
    try: pmz = float(cur.get("PrecursorMZ", "nan"))
    except ValueError: return
    if not picos: return
    a = np.array(picos, dtype=float)
    a = a[np.argsort(-a[:, 1])][:40]
    a[:, 1] = a[:, 1] / a[:, 1].max()
    r = mol.loc[ik]
    cuenta[ik] += 1
    filas.append(dict(ik=ik, formula=r.FORMULA, category=r.CATEGORY, main_class=r.MAIN_CLASS, subclass=r.SUB_CLASS,
                      aducto=cur.get("Precursor_type", ""), modo=cur.get("Ion_mode", ""), pmz=pmz,
                      ce=cur.get("Collision_energy", ""), instr=cur.get("Instrument_type", ""),
                      db=re.sub(r"[0-9].*$", "", cur.get("DB#", ""))[:12], picos=a))

for ln in sys.stdin:
    ln = ln.rstrip("\r\n")
    if leyendo:
        if ln == "":
            leyendo = False; cerrar(); cur = {}; picos = []
        else:
            p = ln.split()
            if len(p) >= 2:
                try: picos.append((float(p[0]), float(p[1])))
                except ValueError: pass
        continue
    if ln == "":
        if cur: cerrar(); cur = {}; picos = []
        continue
    if ":" in ln:
        k, v = ln.split(":", 1); k = k.strip(); v = v.strip()
        if k == "Num Peaks": leyendo = True
        elif k in ("InChIKey", "Precursor_type", "Spectrum_type", "PrecursorMZ", "Ion_mode", "Collision_energy", "Instrument_type", "DB#"): cur[k] = v
if cur: cerrar()
df = pd.DataFrame(filas)
pickle.dump(df, open(BASE / "data" / "spectra" / "ms2_lipids.pkl", "wb"))
msg = (f"records read {leidos:,} · MS2 {ms2:,} · from LIPID MAPS molecules {casan:,} · saved {len(df):,} "
       f"({df.ik.nunique():,} molecules, {df.formula.nunique():,} formulas)\n"
       f"categories: {df.groupby('category').ik.nunique().to_dict()}\nmodes: {df.modo.value_counts().to_dict()}\ntop adducts: {df.aducto.value_counts().head(8).to_dict()}\n")
print(msg); open(BASE / "results" / "23_extract_lipid_spectra.log", "w").write(msg)
