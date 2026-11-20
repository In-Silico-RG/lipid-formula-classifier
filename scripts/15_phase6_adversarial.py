#!/usr/bin/env python
"""
15_phase6_adversarial.py — Phase 6: adversarial negatives (N4) and the +21 m/z control.

Mitchell 2020 states that their classifier cannot be used as a quality filter: when
the observed m/z values are shifted by +21 Da and formulas are reassigned, the
classifier still called the resulting formulas "lipid", with a shifted/intact ratio of
1.06. Their diagnosis: the negatives (HMDB) were polar metabolites, and the model
learned "lipid vs polar", not "lipid vs anything else". Sessions 8, 21 and 22
provided the quantitative basis (38.60 % collision; 38.5 % of the error in the
3.95 % ambiguous; 532 measured lipids not detected in MassBank).

This script builds the fourth layer of negatives of the plan and repeats the experiment:

  N4a  PERTURBATION ADVERSARIALS: formulas obtained from real lipids by
       ±CH2, ±H2, ±O, ±H2O, ±CH2O, ±NH, that satisfy valence (integer DBE ≥ 0, nitrogen
       rule) and do NOT exist in any of our tables (LIPID MAPS, RefMet,
       ChEBI, COCONUT). They are "almost lipids" that are nothing.
  N4b  +21 m/z SHIFT (Mitchell's replica in formula space): for
       a sample of real lipids, M + 21.0 Da is taken and the valid CHNOPS
       formulas within ±3 ppm are enumerated — what a formula assigner would
       return on a miscalibrated spectrum. Those that exist in any table are excluded.

Experiment 1 (deployed model, negatives N1-N3): fraction of N4a and N4b that the
detector calls "lipid", and Mitchell's shifted/intact ratio.
Experiment 2 (retrained with N4 as negatives, grouped split, 5 seeds): the
same ratio, plus the sensitivity on real lipids -internal and EXTERNAL (PNNL,
MassBank)- to check that rejecting artifacts does not cost true lipids.

Plan target: ratio clearly < 1 versus the paper's 1.06, without losing sensitivity.
Project rule: masses are validated against a known value before being used.

Output: results/15_phase6_adversarial.log/.csv and data/n4_adversarial.csv
"""
import importlib.util
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, recall_score

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
sys.path.insert(0, str(BASE / "scripts"))
from common import hill, parsear  # noqa: E402

_spec = importlib.util.spec_from_file_location("rep", BASE / "scripts" / "04_representations.py")
_rep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rep)
construir = _rep.construir

SEMILLAS = [0, 1, 2, 3, 4]
N_ARB = 500
PPM = 3.0
DESPLAZAMIENTO = 21.0           # Da, as in Mitchell
N_OBJETIVOS = 2000              # lipids sampled for N4b
MAX_POR_OBJETIVO = 3
MASA = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
        "P": 30.97376163, "S": 31.972071}
ELS = ["C", "H", "N", "O", "P", "S"]


def masa_de(e):
    return sum(MASA[k] * v for k, v in e.items() if k in MASA)


def valida(e):
    """Valence rules for a closed-shell neutral molecule:
    DBE = C − H/2 + (N+P)/2 + 1 must be an integer ≥ 0; H ≥ 0; and H ≤ 2C + N + P + 2."""
    c, h, n, o, p, s = (e.get(k, 0) for k in ELS)
    if c <= 0 or h < 0 or min(n, o, p, s) < 0:
        return False
    dbe2 = 2 * c - h + n + p + 2          # 2·DBE
    if dbe2 < 0 or dbe2 % 2:
        return False
    return h <= 2 * c + n + p + 2


def hill6(e):
    return hill({k: v for k, v in e.items() if v > 0})


PERTURBACIONES = [
    {"C": 1, "H": 2}, {"C": -1, "H": -2}, {"H": 2}, {"H": -2}, {"O": 1}, {"O": -1},
    {"H": 2, "O": 1}, {"H": -2, "O": -1}, {"C": 1, "H": 2, "O": 1}, {"C": -1, "H": -2, "O": -1},
    {"N": 1, "H": 1}, {"N": -1, "H": -1}, {"C": 2, "H": 4}, {"O": 2}, {"H": 4}, {"H": -4},
]


def enumerar_cerca(m_obj, ppm, conocidas, c_max=120, n_max=6, o_max=30, p_max=3, s_max=3):
    """Valid CHNOPS formulas with mass within ±ppm of m_obj, not known. Vectorized."""
    tol = m_obj * ppm * 1e-6
    N, O, P, S = np.meshgrid(np.arange(n_max + 1), np.arange(o_max + 1), np.arange(p_max + 1),
                             np.arange(s_max + 1), indexing="ij")
    N, O, P, S = (a.ravel() for a in (N, O, P, S))
    base = N * MASA["N"] + O * MASA["O"] + P * MASA["P"] + S * MASA["S"]
    out = []
    for c in range(1, c_max + 1):
        resto = m_obj - c * MASA["C"] - base
        h = np.rint(resto / MASA["H"]).astype(int)
        ok = (h >= 0) & (np.abs(resto - h * MASA["H"]) <= tol)
        if not ok.any():
            continue
        for i in np.where(ok)[0]:
            e = {"C": c, "H": int(h[i]), "N": int(N[i]), "O": int(O[i]), "P": int(P[i]), "S": int(S[i])}
            if valida(e):
                f = hill6(e)
                if f not in conocidas:
                    out.append(f)
    return out


def generar_n4(lip, conocidas, log=print):
    """Generates N4a (perturbations) and N4b (+21 Da) from the real lipids `lip` (in this order),
    excluding `conocidas`. Returns n4a, padre_a, n4b, padre_b. Extracted from main() unchanged
    on 2026-11-18 so that 46_adversarial.py (merged pipeline) can reuse it with the parents."""
    # ------------------------------------------------------------ N4a
    n4a, origen, padre_a = set(), Counter(), {}
    for f in lip:
        e0, _ = parsear(f)
        e0 = {k: v for k, v in e0.items() if k != "_carga"}
        if any(k not in MASA for k in e0):
            continue
        for d in PERTURBACIONES:
            e = dict(e0)
            for k, v in d.items():
                e[k] = e.get(k, 0) + v
            if valida(e):
                g = hill6(e)
                if g not in conocidas and g not in n4a:
                    n4a.add(g); padre_a[g] = f; origen["+".join(f"{k}{v:+d}" for k, v in d.items())] += 1
    n4a = sorted(n4a)
    log(f"\n  N4a perturbation: {len(n4a):,} valid, non-existent formulas")
    log("    by perturbation: " + ", ".join(f"{k} {v}" for k, v in origen.most_common(6)))

    # ------------------------------------------------------------ N4b
    rng = np.random.default_rng(0)
    objetivos = rng.choice(lip, size=min(N_OBJETIVOS, len(lip)), replace=False)
    n4b, sin_cand, n_cand, padre_b = [], 0, [], {}
    for f in objetivos:
        e0, _ = parsear(f)
        m0 = masa_de({k: v for k, v in e0.items() if k in MASA})
        cands = enumerar_cerca(m0 + DESPLAZAMIENTO, PPM, conocidas | set(n4a))
        n_cand.append(len(cands))
        if not cands:
            sin_cand += 1; continue
        n4b.extend(cands[:MAX_POR_OBJETIVO])
        for c_ in cands[:MAX_POR_OBJETIVO]:
            padre_b.setdefault(c_, f)
    n4b = sorted(set(n4b))
    log(f"  N4b shift +{DESPLAZAMIENTO:.0f} Da, ±{PPM} ppm: {len(objetivos):,} target lipids → "
        f"{len(n4b):,} formulas; candidates per target median {np.median(n_cand):.0f}, without candidate {sin_cand}")
    return n4a, padre_a, n4b, padre_b


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "15_phase6_adversarial.log", "w")

    def log(m=""):
        print(m, flush=True); fh.write(m + "\n"); fh.flush()

    log("=" * 84); log("PHASE 6 — adversarial negatives (N4) and +21 m/z control"); log("=" * 84)

    # ---- mass validation against a known value (project rule)
    m_ref = masa_de({"C": 18, "H": 32, "O": 2})
    log(f"  mass check: C18H32O2 = {m_ref:.6f} Da (expected 280.240230) → "
        f"{'OK' if abs(m_ref - 280.240230) < 1e-5 else 'ERROR'}")

    maestro = pd.read_parquet(DATOS / "master.parquet")
    conocidas = set(maestro.formula)
    mb = pd.read_csv(DATOS / "spectra" / "formulas_massbank.csv")
    ext = pd.read_csv(DATOS / "spectra" / "formulas_experimental.csv").fillna("")
    conocidas |= set(mb.formula) | set(ext.formula)
    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    F_all, R2_all, et_all = z["formula"], z["R2"], z["label"]
    lip = [f for f, e in zip(F_all, et_all) if e == "lipid"]
    log(f"  known formulas (to exclude from N4): {len(conocidas):,} · real lipids: {len(lip):,}")

    n4a, padre_a, n4b, padre_b = generar_n4(lip, conocidas, log)
    pd.DataFrame([{"formula": f, "layer": "N4a"} for f in n4a] + [{"formula": f, "layer": "N4b"} for f in n4b]
                 ).to_csv(DATOS / "n4_adversarial.csv", index=False)

    # ---- representation
    def r2(fs):
        X, ok = [], []
        for f in fs:
            r = construir(f)
            if r is not None:
                X.append(r[1]); ok.append(f)
        return np.array(X, dtype=np.float32), ok
    Xa, fa = r2(n4a); Xb, fb = r2(n4b)
    # session 33: 80/20 split of the artificial formulas BY PARENT LIPID (previously random by formula: sibling
    # formulas of nearly equal mass could fall on both sides). ga/gb = parent of each artificial formula.
    ga = np.array([padre_a[f] for f in fa]); gb = np.array([padre_b[f] for f in fb])
    def partir_por_padre(g, rng):
        padres = np.array(sorted(set(g))); rng.shuffle(padres); c = int(0.8 * len(padres))
        en_tr = np.isin(g, padres[:c]); return np.where(en_tr)[0], np.where(~en_tr)[0]
    yb_all = (et_all == "lipid").astype(int)
    lip_mb = sorted(mb.loc[mb.in_lipidmaps, "formula"]); neg_mb = sorted(mb.loc[mb.is_curated_negative, "formula"])
    Xmb_l, _ = r2(lip_mb); Xmb_n, _ = r2(neg_mb)
    Xpnnl, _ = r2(sorted(set(ext.formula)))
    ext_set = set(mb.formula) | set(ext.formula)
    interno = np.array([f not in ext_set for f in F_all])   # to avoid contaminating the external validation

    def modelo():
        return xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=0)

    filas = []
    # Third scenario added after seeing the second (session 25): N4a contains homologs (+CH2, ±H2)
    # of real lipids, most of which ARE uncatalogued lipids; labelling them as negatives
    # teaches the model that homologous series are not lipids and sinks the sensitivity. N4b (+21 Da)
    # does not have that problem. Retraining with N4b only is tested.
    DESCR = {"N1-N3": "deployed model (Phase 3)", "N1-N4": "retrained with N4a+N4b as negatives",
             "N1-N3+N4b": "retrained with N4b (+21 Da) only as negatives",
             "N1-N3+N4a_red+N4b": "retrained with N4b and N4a REDUCED to the size of N4b (separates size and chemistry)"}
    for escenario in ("N1-N3", "N1-N4", "N1-N3+N4b", "N1-N3+N4a_red+N4b"):
        log("\n" + "=" * 84)
        log(f"SCENARIO {escenario}: " + DESCR[escenario])
        log("=" * 84)
        acc = []
        for s in SEMILLAS:
            rng = np.random.default_rng(s)
            idx = np.where(interno)[0]; perm = rng.permutation(idx); c = int(0.8 * len(perm))
            itr, ite = perm[:c], perm[c:]
            Xtr, ytr = R2_all[itr], yb_all[itr]
            # N4 is also split 80/20 to evaluate on UNSEEN adversarials
            a_tr, a_te = partir_por_padre(ga, rng); b_tr, b_te = partir_por_padre(gb, rng)
            if escenario == "N1-N3+N4a_red+N4b":
                a_red = rng.choice(a_tr, size=min(len(b_tr), len(a_tr)), replace=False)
                Xtr = np.vstack([Xtr, Xa[a_red], Xb[b_tr]])
                ytr = np.concatenate([ytr, np.zeros(len(a_red) + len(b_tr), dtype=int)])
            elif escenario == "N1-N4":
                Xtr = np.vstack([Xtr, Xa[a_tr], Xb[b_tr]])
                ytr = np.concatenate([ytr, np.zeros(len(a_tr) + len(b_tr), dtype=int)])
            elif escenario == "N1-N3+N4b":
                Xtr = np.vstack([Xtr, Xb[b_tr]])
                ytr = np.concatenate([ytr, np.zeros(len(b_tr), dtype=int)])
            m = modelo(); m.fit(Xtr, ytr)
            p_te = m.predict(R2_all[ite])
            lip_te = yb_all[ite] == 1
            sens_int = p_te[lip_te].mean() * 100
            espec_int = (1 - p_te[~lip_te]).mean() * 100
            fr_a = m.predict(Xa[a_te]).mean() * 100
            fr_b = m.predict(Xb[b_te]).mean() * 100
            sens_pnnl = m.predict(Xpnnl).mean() * 100
            sens_mb = m.predict(Xmb_l).mean() * 100
            espec_mb = (1 - m.predict(Xmb_n)).mean() * 100
            acc.append({"sens_internal": sens_int, "spec_internal": espec_int,
                        "N4a_called_lipid": fr_a, "N4b_called_lipid": fr_b,
                        "ratio_shifted_intact": fr_b / sens_int if sens_int else np.nan,
                        "sens_PNNL": sens_pnnl, "sens_MassBank": sens_mb, "spec_MassBank": espec_mb})
        t = pd.DataFrame(acc); mm = t.mean(); sd = t.std()
        log(f"  real lipids (internal) detected               {mm['sens_internal']:6.2f} %  (± {sd['sens_internal']:.2f})")
        log(f"  curated non-lipids rejected (internal)        {mm['spec_internal']:6.2f} %")
        log(f"  N4a perturbation called 'lipid'              {mm['N4a_called_lipid']:6.2f} %  (± {sd['N4a_called_lipid']:.2f})")
        log(f"  N4b shifted +21 called 'lipid'                {mm['N4b_called_lipid']:6.2f} %  (± {sd['N4b_called_lipid']:.2f})")
        log(f"  RATIO shifted / intact (Mitchell: 1.06)       {mm['ratio_shifted_intact']:6.3f}")
        log(f"  external sensitivity: PNNL {mm['sens_PNNL']:6.2f} %  · MassBank {mm['sens_MassBank']:6.2f} %"
            f"  · MassBank specificity {mm['spec_MassBank']:6.2f} %")
        filas.append({"scenario": escenario, **{k: round(float(v), 3) for k, v in mm.items()}})
    pd.DataFrame(filas).to_csv(RES / "15_phase6_adversarial.csv", index=False)
    log("\n" + "=" * 84)
    log(f"  {'scenario':12s} {'ratio +21':>10s} {'N4a→lipid':>11s} {'int.sens':>9s} {'PNNL':>7s} {'MassBank':>9s} {'spec.MB':>7s}")
    for r in filas:
        log(f"  {r['scenario']:12s} {r['ratio_shifted_intact']:10.3f} {r['N4a_called_lipid']:10.2f}% {r['sens_internal']:8.2f}% "
            f"{r['sens_PNNL']:6.2f}% {r['sens_MassBank']:8.2f}% {r['spec_MassBank']:6.2f}%")
    log("Saved: results/15_phase6_adversarial.csv · data/n4_adversarial.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
