#!/usr/bin/env python
"""
22_extract_spectra_formulas.py — Generates data/spectra/formulas_experimental.csv and
formulas_massbank.csv from the public spectra files.

Until session 34 these two tables were read by scripts 11, 12, 13 and 15 but no
script produced them (they were made by hand on 18-Sep-2026). This script rebuilds them and,
with --comprobar, compares them with the saved tables.

Sources (in data/spectra/):
  · GNPS PNNL positive and negative, and HCE cell lysate (.msp).
  · MassBank EU 2026.03 in NIST format (.msp).
  · master.parquet (columns in_lipidmaps, in_refmet_nonlipid, in_chebi, in_coconut).

formulas_experimental.csv (libreria, formula, abrev)
  PNNL: the FORMULA field is empty; the ION formula is at the end of NAME
        («PC(16:1/0:0); [M+H]+ C24H49N1O7P1»). The adduct is subtracted (M+H: −H; M+NH4: −N −4H;
        M−H: +H) to obtain the neutral formula. One row per spectrum.
        abrev = NAME text before the first «(» or «;» (CoQ10; [M+H]+ … → CoQ10).
  HCE:  the FORMULA field is used (already neutral); empty ones are discarded; one row per distinct formula. abrev empty (HCE does not provide it).
formulas_massbank.csv (formula, in_lipidmaps, is_curated_negative, n_spectra)
  Records with Spectrum_type MS2 and a Formula field; formula normalized (Hill); charged
  ones and those that cannot be interpreted are discarded. n_spectra = number of
  records with that formula. in_lipidmaps = is in LIPID MAPS; is_curated_negative =
  is in any of the three non-lipid layers of the master table.

Usage:  python 22_extract_spectra_formulas.py             (compares with the saved ones)
        python 22_extract_spectra_formulas.py --escribir  (overwrites them)
Result of the check (session 34): the same rows as the saved tables, in
PNNL even in the same order. The ORDER of the HCE formulas and the MassBank ties change
(the saved ones do not retain the criterion); no script depends on that order.
That is why the default does NOT overwrite: the figures in the article came from the
saved tables.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hill, parsear  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
ESP = RAIZ / "data" / "spectra"
MAESTRO = RAIZ / "data" / "master.parquet"

RE_NOMBRE = re.compile(r"^(.*?)\s*;\s*\[(M[^\]]*)\]\S*\s+(\S+)$")
# Atoms subtracted (or added) to the ion formula to obtain the neutral one.
AJUSTE = {"M+H": {"H": -1}, "M+NH4": {"N": -1, "H": -4}, "M-H": {"H": +1}}


def neutra_pnnl(nombre):
    """(abbreviation, neutral Hill formula) from a PNNL NAME, or None if not possible."""
    m = RE_NOMBRE.match(nombre)
    if not m or m.group(2) not in AJUSTE:
        return None
    e, motivo = parsear(m.group(3))
    if e is None:
        return None
    for k, d in AJUSTE[m.group(2)].items():
        e[k] = e.get(k, 0) + d
        if e[k] < 0:
            return None
    e["_carga"] = 0
    return re.split(r"[(;]", nombre)[0].strip(), hill(e)


def registros(ruta):
    """Iterates over the records of an .msp as lists of lines."""
    bloque = []
    with open(ruta, encoding="utf-8", errors="replace") as f:
        for linea in f:
            if linea.strip() == "" and bloque:
                yield bloque
                bloque = []
            elif linea.strip():
                bloque.append(linea.rstrip("\n"))
    if bloque:
        yield bloque


def campo(reg, nombre):
    for l in reg:
        if l.lower().startswith(nombre.lower() + ":"):
            return l.split(":", 1)[1].strip()
    return None


def experimentales():
    filas = []
    for lib in ("PNNL-LIPIDS-POSITIVE", "PNNL-LIPIDS-NEGATIVE"):
        for reg in registros(ESP / f"{lib}.msp"):
            nom = campo(reg, "NAME")
            r = neutra_pnnl(nom) if nom else None
            if r:
                filas.append((lib, r[1], r[0]))
    vistas = set()
    for reg in registros(ESP / "HCE-CELL-LYSATE-LIPIDS.msp"):
        f = campo(reg, "FORMULA")
        if not f:
            continue
        e, _ = parsear(f)
        if e is None or f in vistas:
            continue
        vistas.add(f)
        filas.append(("HCE-CELL-LYSATE-LIPIDS", hill(e), ""))
    return pd.DataFrame(filas, columns=["libreria", "formula", "abrev"])


def massbank():
    cuenta = Counter()
    for reg in registros(ESP / "MassBank_NISTformat.msp"):
        if campo(reg, "Spectrum_type") != "MS2":
            continue
        f = campo(reg, "Formula")
        if not f:
            continue
        e, _ = parsear(f)
        if e is None or e.get("_carga", 0) != 0:
            continue
        cuenta[hill(e)] += 1
    m = pd.read_parquet(MAESTRO).set_index("formula")
    filas = []
    for f, n in cuenta.items():
        en_lm = bool(m.at[f, "in_lipidmaps"]) if f in m.index else False
        neg = bool(m.at[f, "is_negative"]) if f in m.index else False
        filas.append((f, en_lm, neg, n))
    return pd.DataFrame(filas, columns=["formula", "in_lipidmaps", "is_curated_negative",
                                        "n_spectra"]).sort_values(
        ["n_spectra", "formula"], ascending=[False, True], kind="stable")


def comparar(nuevo, guardado, nombre):
    g = pd.read_csv(guardado).fillna("")
    n = nuevo.fillna("")
    print(f"{nombre}: new {len(n):,} rows, saved {len(g):,}")
    cols = list(g.columns)
    a = n[cols].astype(str).value_counts()
    b = g[cols].astype(str).value_counts()
    dif = a.subtract(b, fill_value=0)
    dif = dif[dif != 0]
    print("  distinct rows (multiset):", len(dif))
    if len(dif):
        print(dif.head(10).to_string())


def main():
    comprobar = "--escribir" not in sys.argv
    ex, mb = experimentales(), massbank()
    if comprobar:
        comparar(ex, ESP / "formulas_experimental.csv", "experimental")
        comparar(mb, ESP / "formulas_massbank.csv", "massbank")
        return
    ex.to_csv(ESP / "formulas_experimental.csv", index=False)
    mb.to_csv(ESP / "formulas_massbank.csv", index=False)
    print("written", len(ex), "and", len(mb), "rows")


if __name__ == "__main__":
    main()
