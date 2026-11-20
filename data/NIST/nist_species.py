"""NIST SRM 1950 consensus lipids (NIST IR 8185, Tables 2-6) -> formula, category, main class.

Formulas are computed from the lipid class and its sum composition (total chain carbons
n, double bonds d), using one rule per class. For ether lipids listed as alternatives ("PC O-34:1/P-34:0/33:1")
the first reading is used. 335 of the 339 formulas occur in LMSD (checked 2026-11-13);
the 4 absent are CER d42:3, SM d41:3, SM d42:4, SM d44:3.

Usage:  ~/miniconda3/bin/python -I nist_species.py
Writes: nist_srm1950_consensus.csv
"""
import csv, re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# class -> (C offset, H offset, fixed elements, category, LMSD main class); H = 2n - 2d + offset
RULES = {
    "FFA":    (0, 0,  "O2",      "Fatty Acyls [FA]",          "Fatty Acids and Conjugates [FA01]"),
    "DAG":    (3, 4,  "O5",      "Glycerolipids [GL]",        "Diradylglycerols [GL02]"),
    "TAG":    (3, 2,  "O6",      "Glycerolipids [GL]",        "Triradylglycerols [GL03]"),
    "PC":     (8, 16, "NO8P",    "Glycerophospholipids [GP]", "Glycerophosphocholines [GP01]"),
    "PC O-":  (8, 18, "NO7P",    "Glycerophospholipids [GP]", "Glycerophosphocholines [GP01]"),
    "PC P-":  (8, 16, "NO7P",    "Glycerophospholipids [GP]", "Glycerophosphocholines [GP01]"),
    "LPC":    (8, 18, "NO7P",    "Glycerophospholipids [GP]", "Glycerophosphocholines [GP01]"),
    "LPC O-": (8, 20, "NO6P",    "Glycerophospholipids [GP]", "Glycerophosphocholines [GP01]"),
    "LPC P-": (8, 18, "NO6P",    "Glycerophospholipids [GP]", "Glycerophosphocholines [GP01]"),
    "LPE O-": (5, 14, "NO6P",    "Glycerophospholipids [GP]", "Glycerophosphoethanolamines [GP02]"),
    "LPE P-": (5, 12, "NO6P",    "Glycerophospholipids [GP]", "Glycerophosphoethanolamines [GP02]"),
    "PE":     (5, 10, "NO8P",    "Glycerophospholipids [GP]", "Glycerophosphoethanolamines [GP02]"),
    "PE O-":  (5, 12, "NO7P",    "Glycerophospholipids [GP]", "Glycerophosphoethanolamines [GP02]"),
    "PE P-":  (5, 10, "NO7P",    "Glycerophospholipids [GP]", "Glycerophosphoethanolamines [GP02]"),
    "LPE":    (5, 12, "NO7P",    "Glycerophospholipids [GP]", "Glycerophosphoethanolamines [GP02]"),
    "PS":     (6, 10, "NO10P",   "Glycerophospholipids [GP]", "Glycerophosphoserines [GP03]"),
    "PG":     (6, 11, "O10P",    "Glycerophospholipids [GP]", "Glycerophosphoglycerols [GP04]"),
    "PI":     (9, 15, "O13P",    "Glycerophospholipids [GP]", "Glycerophosphoinositols [GP06]"),
    "SM":     (5, 13, "N2O6P",   "Sphingolipids [SP]",        "Phosphosphingolipids [SP03]"),
    "CER":    (0, 1,  "NO3",     "Sphingolipids [SP]",        "Ceramides [SP02]"),
    "HexCer": (6, 11, "NO8",     "Sphingolipids [SP]",        "Neutral glycosphingolipids [SP05]"),
    "CE":     (27, 44, "O2",     "Sterol Lipids [ST]",        "Sterols [ST01]"),
}
# Species named individually (formulas from their structures).
FIXED = {
    "12-HETE": ("C20H32O3", "Fatty Acyls [FA]", "Eicosanoids [FA03]"),
    "15-HETE": ("C20H32O3", "Fatty Acyls [FA]", "Eicosanoids [FA03]"),
    "5-HETE":  ("C20H32O3", "Fatty Acyls [FA]", "Eicosanoids [FA03]"),
    "Free Cholesterol": ("C27H46O", "Sterol Lipids [ST]", "Sterols [ST01]"),
    "CA":    ("C24H40O5", "Sterol Lipids [ST]", "Bile acids and derivatives [ST04]"),
    "CDCA":  ("C24H40O4", "Sterol Lipids [ST]", "Bile acids and derivatives [ST04]"),
    "DCA":   ("C24H40O4", "Sterol Lipids [ST]", "Bile acids and derivatives [ST04]"),
    "UDCA":  ("C24H40O4", "Sterol Lipids [ST]", "Bile acids and derivatives [ST04]"),
    "LCA":   ("C24H40O3", "Sterol Lipids [ST]", "Bile acids and derivatives [ST04]"),
    "GCA":   ("C26H43NO6", "Sterol Lipids [ST]", "Steroid conjugates [ST05]"),
    "GCDCA": ("C26H43NO5", "Sterol Lipids [ST]", "Steroid conjugates [ST05]"),
    "GDCA":  ("C26H43NO5", "Sterol Lipids [ST]", "Steroid conjugates [ST05]"),
    "GUDCA": ("C26H43NO5", "Sterol Lipids [ST]", "Steroid conjugates [ST05]"),
    "GLCA":  ("C26H43NO4", "Sterol Lipids [ST]", "Steroid conjugates [ST05]"),
    "TCA":   ("C26H45NO7S", "Sterol Lipids [ST]", "Steroid conjugates [ST05]"),
    "TCDCA": ("C26H45NO6S", "Sterol Lipids [ST]", "Steroid conjugates [ST05]"),
    "TDCA":  ("C26H45NO6S", "Sterol Lipids [ST]", "Steroid conjugates [ST05]"),
    "TLCA":  ("C26H45NO5S", "Sterol Lipids [ST]", "Steroid conjugates [ST05]"),
}


def parse_tables(txt):
    rows, tab = [], None
    for line in txt.splitlines():
        m = re.match(r"\s*Table (\d+)\.(?! \(cont)", line)
        if m:
            tab = int(m.group(1))
        if tab in (2, 3, 4, 5, 6):
            r = re.match(r"\s{2,}(\S.*?)\s{2,}(\d+)\s+([nµpm]mol/mL)\s", line)
            if r:
                rows.append((tab, r.group(1).strip(), int(r.group(2))))
    return rows


def species_formula(name):
    if name in FIXED:
        return FIXED[name], name
    first = name.split("/")[0]                                # first reading of alternatives
    m = re.fullmatch(r"(FFA|DAG|TAG|PC|PE|PS|PG|PI|LPC|LPE|SM|CER|HexCer|CE) (O-|P-|d)?(\d+):(\d+)", first)
    if not m:
        raise ValueError(f"cannot parse {name!r}")
    cls, mod, n, d = m.group(1), m.group(2) or "", int(m.group(3)), int(m.group(4))
    key = cls + " " + mod if mod in ("O-", "P-") else cls
    c0, h0, fixed, cat, main = RULES[key]
    c, h = n + c0, 2 * n - 2 * d + h0
    els = dict(re.findall(r"([A-Z][a-z]?)(\d*)", fixed))
    formula = f"C{c}H{h}" + "".join(f"{e}{els[e]}" for e in ("N", "O", "P", "S") if e in els)
    return (formula, cat, main), first


def main():
    rows = parse_tables((HERE / "nist_ir_8185.txt").read_text())
    with open(HERE / "nist_srm1950_consensus.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "nist_name", "reading_used", "n_labs", "formula", "category", "main_class"])
        for tab, name, labs in rows:
            (formula, cat, main), used = species_formula(name)
            w.writerow([tab, name, used, labs, formula, cat, main])
    print(len(rows), "species written")


if __name__ == "__main__":
    main()
