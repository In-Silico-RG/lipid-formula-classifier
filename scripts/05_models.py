#!/usr/bin/env python
"""
05_models.py — Phase 3 (CPU part): Random Forest and XGBoost.

Full grid: 3 tasks x 3 representations x 4 configurations x 5 seeds.
Default hyperparameters except for the number of trees (500, as in Mitchell); the
systematic search is Phase 4. Here a different question is answered: which
representation and which treatment of class imbalance is preferable.

Tasks:
  A  Lipid category (8 classes)       — upper limit 98,22 % / k-NN 90,07 %
  B  Lipid / non-lipid detection      — k-NN 94,35 %
  C  Main class (86 classes)          — upper limit 95,50 %

80/20 split GROUPED BY FORMULA. The selection metric is macro F1, never
accuracy: in task B, always predicting "non-lipid" already gives 89,69 %.

Output: results/05_models.log + .csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cargar_lmsd, hill, parsear  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
SEMILLAS = [0, 1, 2, 3, 4]
N_ARB = 500


def modelos(n_clases):
    return {
        "RF": lambda: RandomForestClassifier(n_estimators=N_ARB, n_jobs=-1, random_state=0),
        "RF-bal": lambda: RandomForestClassifier(n_estimators=N_ARB, n_jobs=-1, random_state=0,
                                                 class_weight="balanced"),
        "XGB": lambda: xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist",
                                         verbosity=0, random_state=0),
        "XGB-bal": lambda: xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist",
                                             verbosity=0, random_state=0),
    }


def correr(X, y, nombre_tarea, nombre_rep, log, filas):
    clases = np.unique(y)
    cod = {c: i for i, c in enumerate(clases)}
    yc = np.array([cod[v] for v in y])
    acum = {}
    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(y))
        c = int(0.8 * len(y))
        itr, ite = perm[:c], perm[c:]
        for nom, fab in modelos(len(clases)).items():
            m = fab()
            t0 = time.time()
            if nom.endswith("-bal") and nom.startswith("XGB"):
                m.fit(X[itr], yc[itr],
                      sample_weight=compute_sample_weight("balanced", yc[itr]))
            else:
                m.fit(X[itr], yc[itr])
            p = m.predict(X[ite])
            acum.setdefault(nom, []).append({
                "accuracy": (p == yc[ite]).mean() * 100,
                "balanced_acc": balanced_accuracy_score(yc[ite], p) * 100,
                "f1_macro": f1_score(yc[ite], p, average="macro", zero_division=0) * 100,
                "seconds": time.time() - t0,
            })
    for nom in sorted(acum):
        r = {k: float(np.mean([x[k] for x in acum[nom]])) for k in acum[nom][0]}
        sd = np.std([x["f1_macro"] for x in acum[nom]])
        log(f"  {nombre_rep:14s} {nom:9s} {r['accuracy']:7.2f}% {r['balanced_acc']:9.2f}% "
            f"{r['f1_macro']:9.2f}% (±{sd:4.2f})  {r['seconds']:6.1f}s")
        filas.append({"task": nombre_tarea, "representation": nombre_rep, "model": nom,
                      **{k: round(v, 3) for k, v in r.items()}})


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "05_models.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    log("=" * 84)
    log("PHASE 3 (CPU) — RANDOM FOREST AND XGBOOST")
    log("=" * 84)
    log(f"{N_ARB} trees, rest default. 80/20 formula-grouped split, "
        f"{len(SEMILLAS)} seeds.")
    log("Selection metric: macro F1. Accuracy is shown only for comparison")
    log("with the literature; in task B, always answering 'non-lipid' already gives 89,69 %.")

    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    formulas, etiqueta = z["formula"], z["label"]
    reps = {"R1 Mitchell": z["R1"], "R2 Extended": z["R2"],
            "R4 Isotopic": z["R4"].reshape(len(formulas), -1)}

    # majority labels per formula, recomputed from the LMSD
    d = cargar_lmsd(DATOS)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna()]
    may_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0])
    may_cls = d.groupby("hill")["MAIN_CLASS"].agg(lambda s: s.value_counts().index[0])

    filas = []
    cab = f"  {'Representation':14s} {'Model':9s} {'Acc.':>8s} {'Bal.acc':>10s} {'F1 macro':>10s}"

    # ---- A: category
    log("\n" + "=" * 84)
    log("TASK A — Lipid category (8 classes)   ·  upper limit 98,22 %  ·  k-NN 90,07 %")
    log("=" * 84)
    log(cab)
    sel = np.array([f in may_cat.index for f in formulas])
    yA = np.array([may_cat.get(f, None) for f in formulas[sel]])
    for nom, M in reps.items():
        correr(M[sel], yA, "A category", nom, log, filas)

    # ---- B: detection
    log("\n" + "=" * 84)
    log("TASK B — Lipid / non-lipid detection   ·  k-NN 94,35 %  ·  majority 89,69 %")
    log("=" * 84)
    log(cab)
    for nom, M in reps.items():
        correr(M, etiqueta, "B detection", nom, log, filas)

    # ---- C: main class
    log("\n" + "=" * 84)
    log("TASK C — Main class (86 classes)   ·  upper limit 95,50 %")
    log("=" * 84)
    log(cab)
    selC = np.array([f in may_cls.index for f in formulas])
    yC = np.array([may_cls.get(f, None) for f in formulas[selC]])
    vc = pd.Series(yC).value_counts()
    raras = set(vc[vc < 10].index)
    keep = np.array([v not in raras for v in yC])
    log(f"  ({len(raras)} classes with fewer than 10 formulas are excluded; "
        f"{pd.Series(yC[keep]).nunique()} classes and {int(keep.sum()):,} formulas remain)")
    for nom, M in reps.items():
        correr(M[selC][keep], yC[keep], "C class", nom, log, filas)

    t = pd.DataFrame(filas)
    t.to_csv(RES / "05_models.csv", index=False)

    log("\n" + "=" * 84)
    log("BEST CONFIGURATION PER TASK (by macro F1)")
    log("=" * 84)
    for tar in t["task"].unique():
        sub = t[t["task"] == tar].sort_values("f1_macro", ascending=False)
        b = sub.iloc[0]
        log(f"  {tar:14s} {b['representation']:14s} {b['model']:9s} "
            f"F1 {b['f1_macro']:6.2f}%  bal.acc {b['balanced_acc']:6.2f}%  "
            f"acc. {b['accuracy']:6.2f}%")
    log("\n  Saved: results/05_models.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
