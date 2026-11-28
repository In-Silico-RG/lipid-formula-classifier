#!/usr/bin/env python
"""
97_mass_check_other_elements.py: mass of LIPID MAPS (EXACT_MASS, extended release of data/lmsd_extended.csv) against the
monoisotopic mass computed from the formula, for the formulas with elements other than C, H, N, O, P and S (peer-review
minor point, manuscript Section 2.4). Replaces the statement about "273 formulas up to 13 Da / C18H32Br4O2 omits the bromines",
which cannot be reproduced with this release: the mass of C18H32Br4O2 is 595.91357 Da, equal to ours.
Output: results/97_mass_check_other_elements.log + .csv
"""
import numpy as np
import pandas as pd

import merged_common as mc
from common import cargar_lmsd

MASS = {"C": 12.0, "H": 1.00782503223, "N": 14.0030740048, "O": 15.99491461957, "P": 30.97376199842, "S": 31.9720711744,
        "Br": 78.9183376, "Cl": 34.96885268, "F": 18.99840316, "I": 126.9044719, "D": 2.01410177812, "Na": 22.98976928,
        "K": 38.9637064, "Si": 27.97692653, "Se": 79.9165218, "B": 11.00930536, "As": 74.9215946}


def main():
    log = mc.logger("97_mass_check_other_elements")
    d = cargar_lmsd(mc.DATOS)
    rows = []
    for lm, f, em in zip(d.LM_ID, d.FORMULA.astype(str), d.EXACT_MASS):
        e, mo = mc.parsear(f)
        if mo:
            continue
        ks = [k for k in e if k != "_carga" and e[k]]
        if all(k in "CHNOPS" and len(k) == 1 for k in ks):
            continue
        if any(k not in MASS for k in ks):
            rows.append({"LM_ID": lm, "formula": f, "ours": np.nan, "catalogue": float(em), "abs_error": np.nan})
            continue
        m = sum(MASS[k] * e[k] for k in ks)
        rows.append({"LM_ID": lm, "formula": f, "ours": m, "catalogue": float(em), "abs_error": abs(m - float(em))})
    t = pd.DataFrame(rows)
    ok = t.dropna(subset=["abs_error"])
    log(f"formulas with elements other than CHNOPS: {len(t):,}; comparable (masses of all their elements in the table): {len(ok):,}")
    log(f"agree within 0.01 Da: {int((ok.abs_error <= 0.01).sum()):,}; differ by more: {int((ok.abs_error > 0.01).sum()):,}; largest {ok.abs_error.max():.3f} Da")
    log(ok.sort_values("abs_error", ascending=False).head(10).to_string(index=False))
    r = t[t.formula == "C18H32Br4O2"]
    log(f"C18H32Br4O2: {r.to_dict('records')}")
    t.to_csv(mc.RES / "97_mass_check_other_elements.csv", index=False)

    # entries of the raw extended release without a usable formula (the 606 of Section 2.1)
    import collections
    import io
    import zipfile
    for src in (mc.DATOS / "LMSD_extended.sdf.zip", mc.BASE.parent.parent / "lipid_formula_benchmark_github" / "data" / "LMSD_extended.sdf.zip"):
        if src.exists():
            break
    else:
        log("raw LMSD_extended.sdf.zip not found: the reasons of the 606 rejected entries are not recomputed")
        return
    z = zipfile.ZipFile(src)
    name = [x for x in z.namelist() if x.lower().endswith(".sdf")][0]
    n, reasons, examples, rec = 0, collections.Counter(), {}, []
    with z.open(name) as fh:
        for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
            if not raw.startswith("$$$$"):
                rec.append(raw)
                continue
            n += 1
            txt, rec = "".join(rec), []
            i = txt.find("> <FORMULA>")
            f_ = txt[i:].split("\n")[1].strip() if i >= 0 else ""
            e, mo = mc.parsear(f_)
            if mo:
                k = mo.split(":")[0]
                reasons[k] += 1
                examples.setdefault(k, []).append(f_)
    log(f"raw extended release ({src.name}): {n:,} records; without a usable formula: {sum(reasons.values()):,} {dict(reasons)}; "
        f"examples: { {k: v[:3] for k, v in examples.items()} }")


if __name__ == "__main__":
    main()
