#!/usr/bin/env python
"""
06_hyperparameters.py — Phase 4 of the plan.

Hyperparameter search with NESTED VALIDATION using a single inner split (it is not
cross-validation: inside each training split there is ONE 75/25 split).

Why nested. If many configurations are tried and the one that scores best on
the test set is chosen, the test set entered a design decision and the reported figure
may be optimistic. For that reason the search never sees the outer test set.
(Mitchell et al. 2020 do not state which data they used to choose their variables; no
error is attributed to them here.)

Scheme for each outer seed:
    1. split 80/20 GROUPED BY FORMULA              -> outer
    2. within the 80, split again 75/25            -> inner
    3. Optuna searches on the inner split, never seeing the outer 20
    4. retrain with the best parameters on the full 80
    5. measure ONCE on the outer 20

Output: results/06_hyperparameters.log + .csv + mejores_params.json
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cargar_lmsd, hill, parsear  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
SEMILLAS = [0, 1, 2, 3, 4]
N_TRIALS = 30


def espacio_xgb(tr):
    return {
        "n_estimators": tr.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth": tr.suggest_int("max_depth", 3, 12),
        "learning_rate": tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": tr.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": tr.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": tr.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": tr.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def espacio_rf(tr):
    return {
        "n_estimators": tr.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": tr.suggest_categorical("max_depth", [None, 10, 20, 30, 40]),
        "min_samples_leaf": tr.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": tr.suggest_int("min_samples_split", 2, 20),
        "max_features": tr.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
    }


def fabricar(nombre, params):
    if nombre == "XGB":
        return xgb.XGBClassifier(**params, n_jobs=12, tree_method="hist",
                                 verbosity=0, random_state=0)
    return RandomForestClassifier(**params, n_jobs=-1, random_state=0,
                                  class_weight="balanced")


def ajustar(m, nombre, X, y):
    if nombre == "XGB":
        m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        m.fit(X, y)
    return m


def anidada(X, y, nombre_modelo, tarea, log, filas, mejores):
    clases = np.unique(y)
    cod = {c: i for i, c in enumerate(clases)}
    yc = np.array([cod[v] for v in y])
    res, params_por_semilla = [], []

    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(yc))
        c = int(0.8 * len(yc))
        i_dev, i_test = perm[:c], perm[c:]              # 1. outer
        c2 = int(0.75 * len(i_dev))
        i_tr, i_val = i_dev[:c2], i_dev[c2:]            # 2. inner

        def objetivo(tr):
            p = espacio_xgb(tr) if nombre_modelo == "XGB" else espacio_rf(tr)
            m = ajustar(fabricar(nombre_modelo, p), nombre_modelo, X[i_tr], yc[i_tr])
            return f1_score(yc[i_val], m.predict(X[i_val]), average="macro", zero_division=0)

        est = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=s))
        est.optimize(objetivo, n_trials=N_TRIALS, show_progress_bar=False)  # 3. search

        m = ajustar(fabricar(nombre_modelo, est.best_params), nombre_modelo,
                    X[i_dev], yc[i_dev])                # 4. retrain on the 80
        p = m.predict(X[i_test])                        # 5. measure ONCE
        res.append({
            "accuracy": (p == yc[i_test]).mean() * 100,
            "balanced_acc": balanced_accuracy_score(yc[i_test], p) * 100,
            "f1_macro": f1_score(yc[i_test], p, average="macro", zero_division=0) * 100,
            "f1_inner": est.best_value * 100,
        })
        params_por_semilla.append(est.best_params)

    r = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
    sd = float(np.std([x["f1_macro"] for x in res]))
    log(f"  {nombre_modelo:5s} {r['accuracy']:7.2f}% {r['balanced_acc']:9.2f}% "
        f"{r['f1_macro']:9.2f}% (±{sd:4.2f})    inner   {r['f1_inner']:6.2f}%  "
        f"optimism {r['f1_inner'] - r['f1_macro']:+5.2f}")
    filas.append({"task": tarea, "model": nombre_modelo,
                  **{k: round(v, 3) for k, v in r.items()},
                  "sd_f1": round(sd, 3),
                  "optimism": round(r["f1_inner"] - r["f1_macro"], 3)})
    mejores[f"{tarea}|{nombre_modelo}"] = params_por_semilla
    return r


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "06_hyperparameters.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    t_ini = time.time()
    log("=" * 88)
    log("PHASE 4 — HYPERPARAMETERS WITH NESTED VALIDATION")
    log("=" * 88)
    log(f"Optuna (TPE), {N_TRIALS} trials per outer seed, {len(SEMILLAS)} seeds.")
    log("Representation R2 (the best of Phase 3). Class weighting always on.")
    log("")
    log("The 'inner' column is the score obtained during the search, and")
    log("'optimism' is by how much it exceeds the score on the real test set. That")
    log("difference is exactly what would have been over-reported without nesting.")

    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    formulas, etiqueta, R2 = z["formula"], z["label"], z["R2"]

    d = cargar_lmsd(DATOS)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna()]
    may_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0])
    may_cls = d.groupby("hill")["MAIN_CLASS"].agg(lambda s: s.value_counts().index[0])

    filas, mejores = [], {}
    cab = f"  {'Model':5s} {'Acc.':>8s} {'Bal.acc':>10s} {'F1 macro':>10s}"

    log("\n" + "=" * 88)
    log("TASK B — Lipid / non-lipid detection         Phase 3: XGB 90,82 %  ·  k-NN 94,35 %")
    log("=" * 88)
    log(cab)
    for nm in ("XGB", "RF"):
        anidada(R2, etiqueta, nm, "B detection", log, filas, mejores)

    log("\n" + "=" * 88)
    log("TASK C — Main class                         Phase 3: XGB 71,52 %  ·  upper limit 95,50 %")
    log("=" * 88)
    selC = np.array([f in may_cls.index for f in formulas])
    yC = np.array([may_cls.get(f, None) for f in formulas[selC]])
    vc = pd.Series(yC).value_counts()
    keep = np.array([v not in set(vc[vc < 10].index) for v in yC])
    log(f"  ({pd.Series(yC[keep]).nunique()} classes, {int(keep.sum()):,} formulas)")
    log(cab)
    for nm in ("XGB", "RF"):
        anidada(R2[selC][keep], yC[keep], nm, "C class", log, filas, mejores)

    log("\n" + "=" * 88)
    log("TASK A — Category (for comparison only)     Phase 3: RF-bal 89,32 %  ·  upper limit 98,22 %")
    log("=" * 88)
    selA = np.array([f in may_cat.index for f in formulas])
    yA = np.array([may_cat.get(f, None) for f in formulas[selA]])
    log(cab)
    for nm in ("XGB", "RF"):
        anidada(R2[selA], yA, nm, "A category", log, filas, mejores)

    pd.DataFrame(filas).to_csv(RES / "06_hyperparameters.csv", index=False)
    with open(RES / "06_best_params.json", "w") as f:
        json.dump(mejores, f, indent=2, default=str)
    log(f"\n  Total time: {(time.time()-t_ini)/60:.1f} min")
    log("  Saved: results/06_hyperparameters.csv, 06_best_params.json")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
