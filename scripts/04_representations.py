#!/usr/bin/env python
"""
04_representations.py — Phase 2 of the plan.

Generates four ways of describing a molecular formula. ALL are derived
exclusively from the formula: no structural information is used, so that the
premise of the problem (only the formula is available) is kept intact.

  R1  Mitchell  — CHONPS, others, mass and its 6 decimal digits.  Exact replica.
  R2  Extended  — R1 + DBE, elemental ratios, polar atoms, mass/carbon, charge.
  R3  Positional— raw counts per element, one channel each.
  R4  Isotopic  — simulated isotope pattern, 2 channels x 16 nominal masses.  For the CNN.

On R4: a CNN needs local structure in the input, and a vector of fourteen
counts with no natural order does not have it. The isotope pattern does: it is a signal along the
mass axis, it is what a spectrometer actually measures, and it is derived from the formula alone.
Channel 0 = relative intensity of each peak; channel 1 = mass defect of that peak
(its distance to the integer), which is the fine signature of the composition.

Output: data/representations.npz + results/04_representations.log
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cargar_lmsd, parsear  # noqa: E402
from isotopes import patron  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"

ELS6 = ["C", "H", "O", "N", "P", "S"]
ELS_R3 = ELS6 + ["F", "Cl", "Br", "I", "Se", "Si"]
K_MAX = 15                      # 16 nominal masses: M .. M+15

R1_COLS = ELS6 + ["OTROS", "MASA"] + [f"DEC{i}" for i in range(1, 7)]
R2_COLS = R1_COLS + ["CARGA", "DBE", "O_C", "H_C", "N_C", "P_C", "POLARES", "MASA_C"]
R3_COLS = ELS_R3 + ["OTROS", "CARGA"]


def construir(formula):
    """Returns (r1, r2, r3, r4), or None if the formula cannot be parsed."""
    e, motivo = parsear(formula)
    if motivo:
        return None
    carga = e.get("_carga", 0)
    cnt = {k: v for k, v in e.items() if k != "_carga"}
    c, h, o, n, p, s = (cnt.get(x, 0) for x in ELS6)
    otros = sum(v for k, v in cnt.items() if k not in ELS6)

    masas, intens = patron(e, k_max=K_MAX)
    # Small molecules produce an envelope shorter than K_MAX+1: it is
    # padded with zeros so that all inputs have the same length.
    if len(intens) < K_MAX + 1:
        falta = K_MAX + 1 - len(intens)
        masas = np.concatenate([masas, np.zeros(falta)])
        intens = np.concatenate([intens, np.zeros(falta)])
    masa = masas[0]

    # --- R1: Mitchell vector. The decimal digits deliver the mass defect
    # already pre-digested: a tree with axis-aligned splits would take very long to
    # discover on its own a linear combination of six counts.
    frac = masa - int(masa)
    dec = [int(frac * 10 ** i) % 10 for i in range(1, 7)]
    r1 = [c, h, o, n, p, s, otros, masa] + dec

    # --- R2: adds what the Mitchell vector does not express directly
    dbe = c - h / 2 + n / 2 + 1
    r2 = r1 + [carga, dbe,
               o / c if c else 0.0, h / c if c else 0.0,
               n / c if c else 0.0, p / c if c else 0.0,
               o + n + p + s, masa / c if c else 0.0]

    # --- R3: raw counts, without mass or derived features
    r3 = [cnt.get(x, 0) for x in ELS_R3] + \
         [sum(v for k, v in cnt.items() if k not in ELS_R3), carga]

    # --- R4: isotope pattern. Channel 0 intensity, channel 1 relative spacing.
    # Channel 1 is NOT the absolute mass defect: for large masses the decimal part
    # exceeds 0.5 and the rounding changes sign, so the variable loses
    # meaning (error detected in session 14). Instead, the deviation of the
    # spacing from the integer, (m_k - m_0) - k, is used, which is precisely the
    # signature of WHICH isotope contributes to each peak: 13C contributes +1.00335;
    # 2H, +1.00628; 15N, +0.99703. It is also independent of the total mass.
    ks = np.arange(len(masas))
    esp = np.where(intens > 1e-9, (masas - masas[0]) - ks, 0.0)
    r4 = np.stack([intens, esp])

    return r1, r2, r3, r4


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "04_representations.log", "w")

    def log(m=""):
        print(m)
        fh.write(m + "\n")

    log("=" * 78)
    log("PHASE 2 — INPUT REPRESENTATIONS")
    log("=" * 78)

    m = pd.read_parquet(DATOS / "master.parquet")
    m = m[m["label"].isin(["lipid", "non_lipid"])].reset_index(drop=True)
    log(f"\n  formulas to represent: {len(m):,}")

    R1, R2, R3, R4, ok = [], [], [], [], []
    for f in m["formula"]:
        r = construir(f)
        ok.append(r is not None)
        if r:
            R1.append(r[0]); R2.append(r[1]); R3.append(r[2]); R4.append(r[3])
    m = m[ok].reset_index(drop=True)
    R1 = np.array(R1, dtype=np.float32)
    R2 = np.array(R2, dtype=np.float32)
    R3 = np.array(R3, dtype=np.float32)
    R4 = np.array(R4, dtype=np.float32)

    log(f"  successfully represented: {len(m):,}")
    log(f"\n  R1 Mitchell    {R1.shape}  — {', '.join(R1_COLS)}")
    log(f"  R2 Extended    {R2.shape}")
    log(f"  R3 Positional  {R3.shape}")
    log(f"  R4 Isotopic    {R4.shape}  — (n, 2 channels, {K_MAX+1} nominal masses)")

    # ---------------------------------------------------------- physical validation
    log("\n" + "=" * 78)
    log("VALIDATION AGAINST KNOWN PHYSICAL QUANTITIES")
    log("=" * 78)
    log("  Project rule since session 12: no representation is used without")
    log("  checking it against real values, not just against its internal consistency.")

    d = cargar_lmsd(DATOS)
    ref = dict(zip(d["FORMULA"].astype(str), d["MASA_MONO"]))
    idx_masa = R1_COLS.index("MASA")
    # The reference is only reliable for CHONPS formulas: the script that generated it
    # (an earlier version of the dataset script, not part of this repository) uses MASA.get(k, 0), which
    # silently adds ZERO for elements absent from its table. The two cases are
    # separated so as not to confuse an error of our own with an error in the reference.
    solo6 = R1[:, 6] == 0                       # columna OTROS
    e_ok, e_otros = [], []
    for f, fila, s6 in zip(m["formula"], R1, solo6):
        if f in ref and pd.notna(ref[f]):
            (e_ok if s6 else e_otros).append(abs(fila[idx_masa] - ref[f]))
    e_ok, e_otros = np.array(e_ok), np.array(e_otros)
    log(f"\n  [1] Monoisotopic mass vs. the one computed independently in the LMSD")
    log(f"      (a) CHONPS-only formulas — reliable reference")
    log(f"          compared        {len(e_ok):>10,}")
    log(f"          mean error      {e_ok.mean():>10.2e} Da")
    log(f"          max error       {e_ok.max():>10.2e} Da")
    log(f"          → {'CORRECT' if e_ok.max() < 1e-3 else 'CHECK: discrepancy'}")
    if len(e_otros):
        log(f"      (b) formulas with other elements — the REFERENCE is wrong")
        log(f"          compared        {len(e_otros):>10,}")
        log(f"          max error       {e_otros.max():>10.2e} Da")
        log(f"          Checked by hand: for C18H32Br4O2 the LMSD gives 280,240230, which")
        log(f"          is exactly C18H32O2 — it omits the four bromines (315,673 Da).")
        log(f"          Our value, 595,913581, is the correct one.")

    log(f"\n  [2] Height of the M+1 peak vs. the 1,07 % per carbon rule")
    cs = R1[:, 0]
    m1 = R4[:, 0, 1]
    sel = (cs >= 5) & (cs <= 80) & (R1[:, 5] == 0)      # no sulfur, which contaminates M+1
    pred = cs[sel] * 0.0107
    obs = m1[sel]
    rel = np.abs(obs - pred) / np.maximum(pred, 1e-9)
    log(f"      molecules evaluated {int(sel.sum()):>10,}")
    log(f"      median deviation    {np.median(rel)*100:>10.1f} %")
    log(f"      → {'CORRECT' if np.median(rel) < 0.15 else 'REVIEW'}"
        f"  (a small, positive deviation is expected: H, O and N")
    log(f"         also contribute to M+1, so the observed value exceeds the predicted one)")

    log(f"\n  [3] The mass defect must separate lipids from non-lipids")
    esl = (m["label"] == "lipid").values
    masa_col = R1[:, idx_masa]
    frac = masa_col - np.floor(masa_col)
    # compared at comparable mass, because the fractional part grows with mass
    ban = (masa_col > 300) & (masa_col < 600)
    log(f"      in the 300-600 Da band ({int(ban.sum()):,} formulas):")
    log(f"        fractional part, lipids         {frac[ban & esl].mean():>8.4f} Da")
    log(f"        fractional part, non-lipids     {frac[ban & ~esl].mean():>8.4f} Da")
    dif = frac[ban & esl].mean() - frac[ban & ~esl].mean()
    log(f"        difference                      {dif:>+8.4f} Da")
    log(f"      → {'CORRECT' if dif > 0 else 'CHECK: expected to be higher in lipids'}")
    log(f"        lipids are rich in hydrogen (it weighs more: 1,00783) and poor")
    log(f"        in oxygen (it weighs less: 15,99491), so their fractional part is larger")

    log(f"\n  [4] Spacing of the M+1 peak: it must be dominated by 13C (+1,00335)")
    esp1 = R4[:, 1, 1]
    val = esp1[np.abs(esp1) > 1e-6]
    log(f"      median spacing      {np.median(val):>+10.5f} Da   (13C theoretical: +0,00335)")

    np.savez_compressed(DATOS / "representations.npz",
                        formula=m["formula"].values, label=m["label"].values,
                        categories=m["categories"].values, classes=m["classes"].values,
                        n_lmsd=m["n_lmsd"].values,
                        R1=R1, R2=R2, R3=R3, R4=R4,
                        R1_COLS=np.array(R1_COLS), R2_COLS=np.array(R2_COLS),
                        R3_COLS=np.array(R3_COLS))
    tam = (DATOS / "representations.npz").stat().st_size / 1e6
    log(f"\n  Saved: data/representations.npz ({tam:.1f} MB)")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
