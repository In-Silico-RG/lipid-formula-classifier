"""common.py — Parsing and normalization of molecular formulas.

Shared by all scripts in mitchell_mejorado/. Extracted from 01_normalize.py
in session 8 to avoid duplication.
"""
import re
from collections import defaultdict

# Accepted elements. Any formula with an element outside this list is discarded.
ELEMENTOS = {
    "C", "H", "O", "N", "P", "S", "F", "Cl", "Br", "I", "Si", "Se", "As", "B",
    "Na", "K", "Li", "Mg", "Ca", "Fe", "Zn", "Cu", "Mn", "Co", "Ni", "Mo", "Al",
    "Ag", "Au", "Hg", "Pb", "Sn", "Cd", "Cr", "Ti", "V", "W", "Pt", "Pd", "Ru",
    "Rh", "Ir", "Os", "Re", "Ga", "In", "Tl", "Bi", "Sb", "Te", "Ge", "Ba", "Sr",
    "Cs", "Rb", "Be", "Zr", "Y", "La", "Ce", "Eu", "Gd", "Tc", "U", "D", "T",
}
RE_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
RE_LIMPIA = re.compile(r"^[A-Za-z0-9]+$")
# Charge suffix. NOTE: digits before the sign are NOT captured.
# In "C27H31O16+" the 16 is the oxygen subscript, not the charge magnitude.
# Capturing it produced O(1) with charge 16 (session 7 bug, fixed in session 12).
# The magnitude is expressed by repeating signs ("++" = +2). Verified on
# LMSD 2025/2026 and RefMet: 448 formulas with a digit before the sign, none with
# repeated signs, i.e. all real charges are ±1.
RE_CARGA = re.compile(r"([+-]+)$")


RE_CODIGO = re.compile(r"\s*\[[A-Z0-9]+\]\s*$")


def cargar_lmsd(datos):
    """LIPID MAPS (extended, September 2026) with the columns used by the scripts.

    Only LIPID MAPS version in the project since session 34. Reads data/lmsd_extended.csv
    (written by 08_ceiling_extended.py from structures_extended.sdf). CATEGORY has no
    bracketed code («Fatty Acyls», not «Fatty Acyls [FA]»); the original is kept
    in CATEGORY_RAW. MASA_MONO is EXACT_MASS (six decimals).
    """
    import pandas as pd
    d = pd.read_csv(datos / "lmsd_extended.csv", low_memory=False)
    d["CATEGORY_RAW"] = d["CATEGORY"]
    d["CATEGORY"] = d["CATEGORY"].astype(str).str.replace(RE_CODIGO, "", regex=True)
    d["MASA_MONO"] = d["EXACT_MASS"]
    return d


def separar_carga(f):
    """Split the charge suffix off the formula. Returns (formula_without_charge, charge_int).

    Permanent cations in the LMSD (anthocyanins and flavylium ions, mostly polyketides)
    carry the charge in the formula: 'C27H31O16+'. Their elemental composition is perfectly
    valid and discarding them would bias the set precisely in the hardest category.
    """
    m = RE_CARGA.search(f)
    if not m:
        return f, 0
    signos = m.group(1)
    if len(set(signos)) != 1:              # mixed '+-': meaningless
        return f, None
    return f[: m.start()], len(signos) * (1 if signos[0] == "+" else -1)


def parsear(formula):
    """Returns (element_dict, rejection_reason). If accepted, the reason is None."""
    if formula is None:
        return None, "vacia"
    f = str(formula).strip()
    if not f or f.lower() in ("nan", "none", "null", "-"):
        return None, "vacia"
    f, carga = separar_carga(f)
    if carga is None:
        return None, "sintaxis"
    # Rejects polymers "(C4H6O2)n", hydration dots "C6H12O6.H2O", etc.
    if not RE_LIMPIA.match(f):
        return None, "sintaxis"
    if not f[0].isupper():
        return None, "sintaxis"
    cnt = defaultdict(int)
    consumido = 0
    for el, n in RE_TOKEN.findall(f):
        cnt[el] += int(n) if n else 1
        consumido += len(el) + len(n)
    if consumido != len(f):          # some characters were left uninterpreted
        return None, "sintaxis"
    desconocidos = set(cnt) - ELEMENTOS
    if desconocidos:
        return None, f"elemento:{','.join(sorted(desconocidos))}"
    if cnt.get("C", 0) == 0:         # without carbon it is not organic matter of interest
        return None, "sin_carbono"
    cnt["_carga"] = carga
    return dict(cnt), None


def hill(e):
    """Canonical formula in Hill notation: C, H, remaining elements alphabetically, and charge at the end.

    The charge is kept in the canonical key: C27H31O16 and C27H31O16+ are distinct
    species and must not be collapsed into the same formula.
    """
    carga = e.get("_carga", 0)
    els = {k: v for k, v in e.items() if k != "_carga"}
    orden = ["C", "H"] + sorted(k for k in els if k not in ("C", "H"))
    s = "".join(f"{k}{els[k] if els[k] != 1 else ''}" for k in orden if els.get(k, 0) > 0)
    if carga:
        s += ("+" if carga > 0 else "-") * abs(carga)   # «++» = +2, as read by separar_carga
    return s
