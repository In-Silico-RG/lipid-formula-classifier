#!/usr/bin/env python
"""
21_r1_mitchell_exact.py — R1 reconstructed EXACTLY as the Mitchell et al. 2020 vector.

Our R1 (04_representations.py) followed the text of their §4.1 ("counts, mass and its decimals")
and took six decimal digits. Their figure 2 shows the actual vector:
    <mass, tens, units, tenths, unsaturation, #X, #H+#X, #C, #N, #O, #P, #S, #H>
with unsaturation = 4·C + 3·N + 2·O + 6·P + 6·S − (H + X) − 2·(C + N + O + P + S − 1)
(formula exactly as they write it; their example, C16H32O2, gives 2 — checked below).
Here we train with the SAME splits, seeds, models and tasks as 05_models.py,
to find out whether R2 truly improves on the Mitchell variables.
Output: results/21_r1_mitchell_exact.log + .csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
m05 = importlib.import_module("05_models")
from common import cargar_lmsd, hill, parsear  # noqa: E402

DATOS, RES = m05.DATOS, m05.RES


def insaturacion(c, h, n, o, p, s, x):
    return 4 * c + 3 * n + 2 * o + 6 * p + 6 * s - (h + x) - 2 * (c + n + o + p + s - 1)


def vector_mitchell(R1):
    """R1 = [C, H, O, N, P, S, OTROS, MASA, DEC1..DEC6] → vector of Mitchell's figure 2."""
    c, h, o, n, p, s, x, masa = (R1[:, i].astype(float) for i in range(8))
    decenas = np.floor(masa / 10) % 10
    unidades = np.floor(masa) % 10
    decimas = np.floor(masa * 10) % 10
    ins = insaturacion(c, h, n, o, p, s, x)
    return np.column_stack([masa, decenas, unidades, decimas, ins, x, h + x, c, n, o, p, s, h]).astype(np.float32)


def main():
    fh = open(RES / "21_r1_mitchell_exact.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    log("=" * 84)
    log("MITCHELL EXACT R1 (figure 2) — same splits, seeds and models as 05_models.py")
    log("=" * 84)
    # check with the example from their figure 2: palmitic acid C16H32O2
    ej = insaturacion(16, 32, 0, 2, 0, 0, 0)
    log(f"  check: unsaturation of C16H32O2 = {ej} (Mitchell, figure 2: 2) → {'OK' if ej == 2 else 'ERROR'}")
    masa_ej = 256.240230266
    log(f"  check: digits of {masa_ej}: tens {int(masa_ej // 10 % 10)}, units {int(masa_ej // 1 % 10)}, "
        f"tenths {int(masa_ej * 10 // 1 % 10)} (Mitchell: 5, 6, 2)")

    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    formulas, etiqueta = z["formula"], z["label"]
    M = vector_mitchell(z["R1"])
    log(f"  vector: {M.shape[1]} variables, {M.shape[0]:,} formulas")

    d = cargar_lmsd(DATOS)
    d["hill"] = [None if mo else hill(e) for e, mo in (parsear(x) for x in d["FORMULA"])]
    d = d[d["hill"].notna()]
    may_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0])
    may_cls = d.groupby("hill")["MAIN_CLASS"].agg(lambda s: s.value_counts().index[0])

    filas = []
    cab = f"  {'Representation':14s} {'Model':9s} {'Acc.':>8s} {'Bal.acc':>10s} {'F1 macro':>10s}"
    log("\nTASK A — Category (8 classes)"); log(cab)
    sel = np.array([f in may_cat.index for f in formulas])
    m05.correr(M[sel], np.array([may_cat[f] for f in formulas[sel]]), "A category", "R1 exact", log, filas)
    log("\nTASK B — Lipid / non-lipid detection"); log(cab)
    m05.correr(M, etiqueta, "B detection", "R1 exact", log, filas)
    log("\nTASK C — Main class"); log(cab)
    selC = np.array([f in may_cls.index for f in formulas])
    yC = np.array([may_cls[f] for f in formulas[selC]])
    vc = pd.Series(yC).value_counts()
    keep = np.array([v not in set(vc[vc < 10].index) for v in yC])
    m05.correr(M[selC][keep], yC[keep], "C class", "R1 exact", log, filas)

    t = pd.DataFrame(filas)
    t.to_csv(RES / "21_r1_mitchell_exact.csv", index=False)
    ref = pd.read_csv(RES / "05_models.csv")
    log("\n" + "=" * 84)
    log("COMPARISON (macro F1, best configuration of each representation)")
    log("=" * 84)
    todo = pd.concat([ref[ref["representation"].isin(["R1 Mitchell", "R2 Extended"])], t], ignore_index=True)
    for tar in ["A category", "B detection", "C class"]:
        sub = todo[todo["task"] == tar]
        mej = sub.loc[sub.groupby("representation")["f1_macro"].idxmax()]
        linea = "  ".join(f"{r.representation}: {r.f1_macro:.2f} ({r.model})" for r in mej.itertuples())
        log(f"  {tar:12s} {linea}")
    log("\n  Saved: results/21_r1_mitchell_exact.csv")


if __name__ == "__main__":
    sys.exit(main())
