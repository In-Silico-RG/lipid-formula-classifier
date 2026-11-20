#!/usr/bin/env python
"""
24_extra_variables.py — Does accuracy improve when adding variables derived ONLY from the formula? (R5 = R2 + 24 variables)

New variables (all derived from the formula; the premise of the problem does not change):
  · Kendrick mass defect for CH2, H2, O and C2H4 (Lerno et al. 2010 used it as a graphical method for classes);
    remainder of the nominal mass modulo 14 (Z* series)
  · parities: C, H, N, O, H+N
  · linear combinations of counts: H−2C, O−2P, O−4P, (O−4P)/2 (probable acyl chains if there is phosphate), N−P, 2C−H+N, O−2P−N
  · indicators: O and P; N and P; S and O; typical phospholipid (P, N, O≥6); O per 100 C; (H−2C) per heteroatom
Same protocol as 05_models.py: 80/20 split by formula, 5 seeds, 500 trees, RF-bal and XGB-bal; for the main class, classes with fewer than 10 formulas are excluded (57 classes, 9,851 formulas), as in 05.
Output: results/24_extra_variables.csv and .log
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cargar_lmsd, hill, parsear  # noqa: E402
BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
SEMILLAS = [0, 1, 2, 3, 4]

def extra(X2, cols):
    C, H, O, N, P, S = [X2[:, cols.index(k)] for k in ("C", "H", "O", "N", "P", "S")]
    M = X2[:, cols.index("MASA")]
    def kmd(b):
        km = M * round(b) / b
        return np.round(km) - km
    return np.column_stack([kmd(14.01565), kmd(2.01565), kmd(15.9949146), kmd(28.0313), M % 14,
        H % 2, C % 2, (H + N) % 2, N % 2, O % 2,
        H - 2 * C, O - 2 * P, O - 4 * P, (O - 4 * P) / 2, N - P, 2 * C - H + N, O - 2 * P - N,
        (O > 0) & (P > 0), (N > 0) & (P > 0), (S > 0) & (O > 0), (P > 0) & (N > 0) & (O >= 6),
        O / np.maximum(C, 1) * 100, (H - 2 * C) / np.maximum(O + N + P + S, 1)]).astype(float)

def evaluar(X, y, nombres_modelo=("RF-bal", "XGB-bal")):
    clases = np.unique(y); cod = {c: i for i, c in enumerate(clases)}; yc = np.array([cod[v] for v in y])
    out = {}
    for s in SEMILLAS:
        rng = np.random.default_rng(s); perm = rng.permutation(len(y)); c = int(0.8 * len(y)); itr, ite = perm[:c], perm[c:]
        ut = np.unique(yc[itr]); mp = {v: i for i, v in enumerate(ut)}; ytr = np.array([mp[v] for v in yc[itr]])
        for nom in nombres_modelo:
            if nom == "RF-bal":
                m = RandomForestClassifier(n_estimators=500, n_jobs=8, random_state=0, class_weight="balanced"); m.fit(X[itr], ytr)
            else:
                m = xgb.XGBClassifier(n_estimators=500, n_jobs=8, tree_method="hist", verbosity=0, random_state=0)
                m.fit(X[itr], ytr, sample_weight=compute_sample_weight("balanced", ytr))
            p = ut[m.predict(X[ite])]
            out.setdefault(nom, []).append(((p == yc[ite]).mean() * 100, balanced_accuracy_score(yc[ite], p) * 100,
                                            f1_score(yc[ite], p, average="macro", zero_division=0) * 100))
    return {k: np.array(v) for k, v in out.items()}

def main():
    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    formulas, R2, cols = z["formula"], z["R2"], list(z["R2_COLS"])
    d = cargar_lmsd(DATOS); hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x); hs.append(None if mo else hill(e))
    d["hill"] = hs; d = d[d.hill.notna()]
    cat = d.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0])
    cls = d.groupby("hill")["MAIN_CLASS"].agg(lambda s: s.value_counts().index[0])
    sel = np.array([f in cat.index for f in formulas]); F = formulas[sel]; X2 = R2[sel]; X5 = np.hstack([X2, extra(X2, cols)])
    filas = []; log = open(RES / "24_extra_variables.log", "w")
    def p(m=""):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    p("EXTRA VARIABLES DERIVED FROM THE FORMULA — R2 (22 variables) versus R5 (R2 + 23 variables)")
    p(f"lipid formulas: {len(F):,} · 80/20 split by formula · 5 seeds · 500 trees\n")
    for tarea, ser in (("A category", cat), ("C main class", cls)):
        y = np.array([ser[f] for f in F])
        XA, XB = X2, X5
        if tarea.startswith("C"):          # same as 05_models.py: classes with fewer than 10 formulas are excluded
            from collections import Counter
            ct = Counter(y); ok = np.array([ct[c] >= 10 for c in y]); y, XA, XB = y[ok], X2[ok], X5[ok]
        p(f"{tarea}   ({len(np.unique(y))} classes, {len(y):,} formulas)")
        res = {}
        for rep, X in (("R2", XA), ("R5", XB)):
            t0 = time.time(); r = evaluar(X, y)
            for nom, a in r.items():
                mu, sd = a.mean(0), a.std(0); res[(rep, nom)] = a
                p(f"  {rep}  {nom:7s} accuracy {mu[0]:.2f} ± {sd[0]:.2f}   bal.acc {mu[1]:.2f}   macro F1 {mu[2]:.2f} ± {sd[2]:.2f}")
                filas.append(dict(task=tarea, rep=rep, model=nom, accuracy=mu[0], sd_accuracy=sd[0], bal_acc=mu[1], f1_macro=mu[2], sd_f1=sd[2]))
            p(f"  ({time.time()-t0:.0f} s)")
        for nom in ("RF-bal", "XGB-bal"):
            dlt = res[("R5", nom)][:, 0] - res[("R2", nom)][:, 0]
            p(f"  Δ accuracy R5−R2 with {nom}: {dlt.mean():+.2f} points (per seed: {', '.join(f'{x:+.2f}' for x in dlt)}) · improves in {int((dlt>0).sum())} of {len(dlt)} seeds")
        p()
    pd.DataFrame(filas).to_csv(RES / "24_extra_variables.csv", index=False)

if __name__ == "__main__":
    main()
