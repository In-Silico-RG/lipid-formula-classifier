#!/usr/bin/env python
"""
03_baselines.py — Phase 1.3 of the plan: baselines.

Without a baseline it is not known whether a model contributes anything. Mitchell 2020 reports none.

Two tasks, both with a split GROUPED BY FORMULA (leakage 0) and 5 seeds:
  A) Category among lipids (8 classes)
  B) Lipid / non-lipid detection

Baselines:
  1. Majority               — always predicts the most frequent class
  2. Chemical rules         — hand-written from the biochemical scaffold
  3. Database lookup        — if the formula was seen in training, returns its label
  4. k-NN (k=1, 5)          — on the count vector

Output: results/03_baselines.log + .csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cargar_lmsd, hill, parsear  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
SEMILLAS = [0, 1, 2, 3, 4]
ELS = ["C", "H", "O", "N", "P", "S"]


def vector(formula):
    e, motivo = parsear(formula)
    if motivo:
        return None
    v = [e.get(x, 0) for x in ELS]
    carga = e.get("_carga", 0)
    otros = sum(n for k, n in e.items() if k not in ELS and k != "_carga")
    c, h, o = v[0], v[1], v[2]
    dbe = c - h / 2 + v[3] / 2 + 1
    return v + [otros, carga, dbe, o / c if c else 0, h / c if c else 0]


COLS = ELS + ["OTROS", "CARGA", "DBE", "O_C", "H_C"]


def reglas_quimicas(df):
    """Simple rules from the biochemical scaffold of each category.

    Glycerophospholipids carry phosphorus; sphingolipids, nitrogen without phosphorus;
    sterols are oxygen-poor with high DBE; the rest fall into fatty acyls.
    """
    p = np.full(len(df), "Fatty Acyls", dtype=object)
    P, N, O_C, DBE, C = df["P"].values, df["N"].values, df["O_C"].values, df["DBE"].values, df["C"].values
    p[(P >= 1)] = "Glycerophospholipids"
    p[(P == 0) & (N >= 1)] = "Sphingolipids"
    p[(P == 0) & (N == 0) & (O_C < 0.12) & (DBE >= 4) & (C <= 30)] = "Sterol Lipids"
    p[(P == 0) & (N == 0) & (O_C >= 0.12) & (O_C < 0.30) & (C >= 35)] = "Glycerolipids"
    return p


def busqueda_base(tr_x, tr_y, te_x, mayoritaria):
    """Exact lookup by formula. It is the upper limit of pure memorization."""
    tabla = dict(zip(tr_x, tr_y))
    return np.array([tabla.get(f, mayoritaria) for f in te_x], dtype=object)


def metricas(y, p, w):
    return {
        "accuracy": float((y == p).mean() * 100),
        "accuracy_weighted": float(np.average(y == p, weights=w) * 100),
        "balanced_acc": float(balanced_accuracy_score(y, p) * 100),
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0) * 100),
    }


def evaluar(d, etiqueta, nombre, log, techo=None):
    """d: one row per unique formula, with vector columns, 'y' and weight 'w'."""
    log(f"\n{'='*78}\n{nombre}\n{'='*78}")
    log(f"  formulas: {len(d):,} · classes: {d['y'].nunique()} · "
        f"entries represented: {int(d['w'].sum()):,}")
    rep = d["y"].value_counts()
    log(f"  majority class: {rep.index[0]} ({rep.iloc[0]/len(d)*100:.1f} %)")

    X = d[COLS].values.astype(float)
    y = d["y"].values
    w = d["w"].values
    f = d["formula"].values

    acum = {}
    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(d))
        corte = int(0.8 * len(d))
        itr, ite = perm[:corte], perm[corte:]
        may = pd.Series(y[itr]).value_counts().index[0]

        preds = {
            "1. Mayoritaria": np.full(len(ite), may, dtype=object),
            "3. Database look-up": busqueda_base(f[itr], y[itr], f[ite], may),
        }
        if etiqueta == "category":
            preds["2. Chemical rules"] = reglas_quimicas(d.iloc[ite])

        sc = StandardScaler().fit(X[itr])
        for k in (1, 5):
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(sc.transform(X[itr]), y[itr])
            preds[f"4. k-NN (k={k})"] = knn.predict(sc.transform(X[ite]))

        for nom, p in preds.items():
            acum.setdefault(nom, []).append(metricas(y[ite], p, w[ite]))

    filas = []
    log(f"\n  {'Baseline model':24s} {'Acc.':>8s} {'Acc.wtd':>11s} {'Bal.acc':>9s} {'F1 macro':>9s}")
    log(f"  {'-'*24} {'-'*8} {'-'*11} {'-'*9} {'-'*9}")
    for nom in sorted(acum):
        m = {k: np.mean([r[k] for r in acum[nom]]) for k in acum[nom][0]}
        sd = np.std([r["accuracy"] for r in acum[nom]])
        log(f"  {nom:24s} {m['accuracy']:7.2f}% {m['accuracy_weighted']:10.2f}% "
            f"{m['balanced_acc']:8.2f}% {m['f1_macro']:8.2f}%   (±{sd:.2f})")
        filas.append({"task": nombre, "baseline": nom, **{k: round(v, 3) for k, v in m.items()}})
    if techo:
        log(f"  {'-'*24}")
        log(f"  {'THEORETICAL LIMIT':24s} {techo[1]:7.2f}% {techo[0]:10.2f}%")
    return filas


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "03_baselines.log", "w")

    def log(m=""):
        print(m)
        fh.write(m + "\n")

    log("=" * 78)
    log("PHASE 1.3 — BASELINES")
    log("=" * 78)
    log("80/20 split GROUPED BY FORMULA (leakage 0 %), mean of 5 seeds.")
    log("'Acc.wtd' weights each formula by its number of isomers in the LMSD.")

    # ---- Task A: category among lipids
    d = cargar_lmsd(DATOS)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna() & d["CATEGORY"].notna()]

    g = d.groupby("hill")["CATEGORY"]
    may = g.agg(lambda s: s.value_counts().index[0])
    peso = g.size()
    A = pd.DataFrame({"formula": may.index, "y": may.values, "w": peso.values})
    vecs = [vector(x) for x in A["formula"]]
    A = A[[v is not None for v in vecs]].copy()
    A[COLS] = [v for v in vecs if v is not None]

    filas = evaluar(A, "category", "TASK A — Lipid category (8 classes)", log,
                    techo=(97.44, 98.22))

    # ---- Task B: lipid / non-lipid detection
    m = pd.read_parquet(DATOS / "master.parquet")
    m = m[m["label"].isin(["lipid", "non_lipid"])].copy()
    B = pd.DataFrame({
        "formula": m["formula"].values,
        "y": m["label"].values,
        "w": np.maximum(m["n_lmsd"].values, 1),
    })
    vecs = [vector(x) for x in B["formula"]]
    B = B[[v is not None for v in vecs]].copy()
    B[COLS] = [v for v in vecs if v is not None]

    filas += evaluar(B, "detection", "TASK B — Lipid / non-lipid detection", log)

    pd.DataFrame(filas).to_csv(RES / "03_baselines.csv", index=False)
    log("\n  Saved: results/03_baselines.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
