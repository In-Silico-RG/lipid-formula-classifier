#!/usr/bin/env python
"""
42_hyperparameters.py — Merged pipeline: settings search with nested validation (spec D7).

Search spaces and sampler are those of 06_hyperparameters.py (imported, not copied): Optuna's
tree-structured Parzen estimator, 30 trials, weighted classes, objective = macro F1.
Nested protocol: for each outer fold of the merged split (5-fold StratifiedGroupKFold grouped
by formula), the search runs on the outer training part only, comparing settings on an inner
split of that part BY FORMULA (80/20, GroupShuffleSplit); the best settings are then refitted
on the whole outer training part and scored on the outer test part, which the search never saw.
Cost: the search is run for seed 0 only (5 outer folds); the outer scores are pooled out of fold.

Usage:  python 42_hyperparameters.py --set=R2C --model=RF|XGB --task=A|B|C [--trials=30] [--jobs=8]
Output: results/42_hyperparameters_<set>_<model>_<task>.log + .csv
"""
import sys
import warnings

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
import merged_common as mc  # noqa: E402

h06 = mc._load("h06", "06_hyperparameters.py")      # espacio_rf, espacio_xgb
OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL, TASK = OPT["set"], OPT["model"], OPT["task"]
TRIALS, JOBS = int(OPT.get("trials", h06.N_TRIALS)), int(OPT.get("jobs", 8))
SEED = 0


def build(params, seed):
    if MODEL == "RF":
        return RandomForestClassifier(**params, class_weight="balanced", n_jobs=JOBS, random_state=seed)
    return xgb.XGBClassifier(**params, tree_method="hist", n_jobs=JOBS, verbosity=0, random_state=seed)


def fit(m, X, y):
    kw = {"sample_weight": compute_sample_weight("balanced", y)} if MODEL == "XGB" else {}
    return m.fit(X, y, **kw)


def main():
    log = mc.logger(f"42_hyperparameters_{SET}_{MODEL}_{TASK}")
    log(f"42 — SETTINGS SEARCH (nested) · {SET} · {MODEL} · task {TASK} · {TRIALS} trials per outer fold")
    if TASK in ("A", "C"):
        d, acc = mc.load_entries()
        y = (d.CATEGORY if TASK == "A" else d.MAIN_CLASS).to_numpy()
        formulas = d.hill.to_numpy()
        accepted = acc["category" if TASK == "A" else "main_class"]
    else:
        m, y, accepted = mc.load_detection()
        formulas = m.formula.to_numpy()
    X = mc.features(list(formulas), sets=[SET])[SET]
    space = h06.espacio_rf if MODEL == "RF" else h06.espacio_xgb
    pred = np.empty(len(y), dtype=y.dtype)
    best_rows = []
    for k, (tr, te) in enumerate(mc.splits(y, formulas, SEED)):
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(tr, groups=formulas[tr]))
        itr, iva = tr[a], tr[b]
        seen, ytr = np.unique(y[itr], return_inverse=True)
        lut = {c: i for i, c in enumerate(seen)}
        va_ok = np.array([c in lut for c in y[iva]])
        yva = np.array([lut[c] for c in y[iva][va_ok]])

        def objective(trial):
            mdl = fit(build(space(trial), SEED), X[itr], ytr)
            return f1_score(yva, mdl.predict(X[iva][va_ok]), average="macro", zero_division=0)

        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        st.optimize(objective, n_trials=TRIALS, show_progress_bar=False)
        seen2, ytr2 = np.unique(y[tr], return_inverse=True)
        mdl = fit(build(st.best_params, SEED), X[tr], ytr2)
        pred[te] = seen2[mdl.predict(X[te]).astype(int)]
        best_rows.append({"fold": k, "inner_macro_f1": st.best_value * 100, **st.best_params})
        log(f"  fold {k}: inner macro F1 {st.best_value * 100:.2f}; best {st.best_params}")
    sc = mc.scores(y, pred, formulas, accepted)
    log(f"\n  OUTER (seed {SEED}, pooled out of fold): acc {sc['accuracy']:.2f}  macro F1 {sc['macro_f1']:.2f}  "
        f"bal. acc {sc['balanced_accuracy']:.2f}  permissive {sc['permissive_accuracy']:.2f}")
    t = pd.DataFrame(best_rows)
    t.to_csv(mc.RES / f"42_hyperparameters_{SET}_{MODEL}_{TASK}.csv", index=False)
    pd.DataFrame([{"set": SET, "model": MODEL, "task": TASK, **sc}]).to_csv(
        mc.RES / f"42_hyperparameters_{SET}_{MODEL}_{TASK}_outer.csv", index=False)
    log(f"  Saved: results/42_hyperparameters_{SET}_{MODEL}_{TASK}.csv and _outer.csv")


if __name__ == "__main__":
    main()
