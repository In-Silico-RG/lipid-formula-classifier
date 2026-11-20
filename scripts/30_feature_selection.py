#!/usr/bin/env python
"""
30_feature_selection.py — Merged pipeline, decision D4 (docs/PIPELINE_SPEC.md).

Which input features should the final model use? Candidate sets, all computed from the
formula alone:
  R1      Mitchell et al. 2020 vector exactly as in their figure 2 (script 21)           13
  T10     the ten features of the 2025 tool manuscript                                   10
  T12     T10 + C mod 5 + C parity (lipids-class v2 model)                               12
  R2      R1 + charge, DBE, O/C, H/C, N/C, P/C, polar atoms, mass/C                      22
  R2C     R2 + C mod 5 + C parity                                                        24
  R5      R2 + 23 derived variables (script 24)                                          45

Protocol (D2, D3): unit = LIPID MAPS entry (molecule); 5-fold StratifiedGroupKFold grouped
by Hill formula, so no test formula is seen in training; 3 repeats (seeds 0, 1, 2).
Models (one per run, so the choice does not depend on one model):
  rf   random forest, 300 trees, default settings
  xgb  XGBoost, 300 rounds, hist, default settings
  mlp  standardized inputs; 2 hidden layers (256, 128) as in script 09; adam; early stopping
  knn  standardized inputs; k = 5 as in script 03
The CNN (script 09) is not included: its input is the isotope pattern (R4), not a feature set. Tasks: category (8 labels) and main class (all labels).
Metrics on pooled out-of-fold predictions of each repeat: accuracy, macro F1, balanced accuracy.

Selection rule, fixed BEFORE running (D4): keep the smallest set whose macro F1 is within
one SD (across repeats) of the best set, for BOTH tasks. Applied per model; the sets chosen
by the four models are compared by 30_summary at the end.

Usage:  python 30_feature_selection.py {rf|xgb|mlp|knn} [n_jobs] [--task=category|main_class]
                                       [--sets=R1,T12,...] [--smoke]
        --task / --sets split one model over several cluster jobs; 30_summary.py joins the parts
        and applies the rule.
Output: results/30_feature_selection_<model>[_<task>][_<sets>].log + .csv
"""
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import cargar_lmsd, hill, parsear  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = _load("rep04", "04_representations.py")      # construir() -> R1, R2
v24 = _load("v24", "24_extra_variables.py")         # extra() -> the 23 R5 variables
m21 = _load("m21", "21_r1_mitchell_exact.py")       # vector_mitchell() -> exact R1 (13 values)

BASE = HERE.parent
DATOS, RES = BASE / "data", BASE / "results"
SEEDS = [0, 1, 2]
N_TREES = 300
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = ARGS[0] if ARGS else "rf"
N_JOBS = int(ARGS[1]) if len(ARGS) > 1 else 6
SMOKE = "--smoke" in sys.argv
OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
TASKS = [OPT["task"]] if "task" in OPT else ["category", "main_class"]
SETS = OPT["sets"].split(",") if "sets" in OPT else None
TAG = MODEL + ("_" + OPT["task"] if "task" in OPT else "") + ("_" + "-".join(SETS) if SETS else "")
assert MODEL in ("rf", "xgb", "mlp", "knn"), MODEL


def make_model(seed):
    if MODEL == "rf":
        return RandomForestClassifier(n_estimators=N_TREES, n_jobs=N_JOBS, random_state=seed)
    if MODEL == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=N_TREES, tree_method="hist", n_jobs=N_JOBS,
                                 verbosity=0, random_state=seed)
    if MODEL == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                             early_stopping=True, n_iter_no_change=15, random_state=seed))
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS))


def tool10(e):
    """The ten 2025-manuscript features, as in classifier/features.py (DBE from C and H only)."""
    c, h, o, n, p, s = (e.get(x, 0) for x in "CHONPS")
    return [c, h, o, n, p, s, c + h + o + n + p + s, (2 * c + 2 - h) / 2 if c else 0.0,
            o / c if c else 0.0, o + n + p + s]


def build(formulas):
    r1, r2, t10 = [], [], []
    for f in formulas:
        e, _ = parsear(f)
        a, b, _, _ = rep.construir(f)
        r1.append(a); r2.append(b); t10.append(tool10(e))
    r1, r2, t10 = (np.array(x, dtype=float) for x in (r1, r2, t10))
    c = r2[:, 0]
    cnum = np.column_stack([c % 5, c % 2])
    return {
        # exact vector of Mitchell's figure 2; the 6-decimal R1 of script 04 is NOT it (manuscript §2.4)
        "R1": m21.vector_mitchell(r1).astype(float),
        "T10": t10,
        "T12": np.hstack([t10, cnum]),
        "R2": r2,
        "R2C": np.hstack([r2, cnum]),
        "R5": np.hstack([r2, v24.extra(r2, list(rep.R2_COLS))]),
    }


def evaluate(X, y, groups):
    out = []
    for s in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
        pred = np.empty(len(y), dtype=y.dtype)
        for tr, te in cv.split(X, y, groups):
            # labels re-coded per fold: XGBoost needs 0..k-1 and a rare class may be absent
            seen, ytr = np.unique(y[tr], return_inverse=True)
            m = make_model(s)
            m.fit(X[tr], ytr)
            pred[te] = seen[m.predict(X[te]).astype(int)]
        out.append((accuracy_score(y, pred) * 100,
                    f1_score(y, pred, average="macro", zero_division=0) * 100,
                    balanced_accuracy_score(y, pred) * 100))
    return np.array(out)


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / f"30_feature_selection_{TAG}.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    d = cargar_lmsd(DATOS)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    n0 = len(d)
    d = d[d.hill.notna() & d.CATEGORY.notna() & d.MAIN_CLASS.notna()].reset_index(drop=True)
    if SMOKE:
        d = d.sample(3000, random_state=0).reset_index(drop=True)

    log("=" * 84)
    log(f"30 — FEATURE SELECTION FOR THE MERGED PIPELINE (decision D4) — model {MODEL}")
    log("=" * 84)
    log(f"LIPID MAPS entries: {n0:,}; parsed with category and main class: {len(d):,}; "
        f"formulas: {d.hill.nunique():,}")
    log(f"model = {MODEL} · unit = entry · 5-fold StratifiedGroupKFold by formula · seeds {SEEDS}")
    log("rule (fixed before running): smallest set with macro F1 within 1 SD of the best, both tasks\n")

    feats = build(d.FORMULA.tolist())
    if SETS:
        feats = {k: v for k, v in feats.items() if k in SETS}
    groups = d.hill.to_numpy()
    rows = []
    for task, col in (("category", "CATEGORY"), ("main_class", "MAIN_CLASS")):
        if task not in TASKS:
            continue
        y = d[col].astype(str).to_numpy()
        log(f"--- task: {task} ({len(np.unique(y))} labels)")
        for name, X in feats.items():
            t0 = time.time()
            r = evaluate(X, y, groups)
            mu, sd = r.mean(0), r.std(0, ddof=1)
            rows.append({"task": task, "set": name, "n_features": X.shape[1],
                         "accuracy": mu[0], "accuracy_sd": sd[0], "macro_f1": mu[1], "macro_f1_sd": sd[1],
                         "balanced_accuracy": mu[2], "balanced_accuracy_sd": sd[2],
                         "macro_f1_per_seed": " ".join(f"{v:.2f}" for v in r[:, 1])})
            log(f"  {name:4s} ({X.shape[1]:2d} feat.)  acc {mu[0]:6.2f} ± {sd[0]:.2f}   "
                f"macro F1 {mu[1]:6.2f} ± {sd[1]:.2f}   bal. acc {mu[2]:6.2f} ± {sd[2]:.2f}   "
                f"[{time.time() - t0:.0f} s]")
        log()

    t = pd.DataFrame(rows)
    t.insert(0, "model", MODEL)
    t.to_csv(RES / f"30_feature_selection_{TAG}.csv", index=False)

    if SETS or len(TASKS) < 2:
        log(f"\n  partial run ({TAG}); the rule is applied by 30_summary.py. Saved: results/30_feature_selection_{TAG}.csv")
        return
    log("--- selection rule")
    ok = None
    for task in ("category", "main_class"):
        s = t[t.task == task]
        best = s.loc[s.macro_f1.idxmax()]
        within = set(s[s.macro_f1 >= best.macro_f1 - best.macro_f1_sd].set)
        log(f"  {task}: best {best.set} {best.macro_f1:.2f} ± {best.macro_f1_sd:.2f}; "
            f"within 1 SD: {', '.join(sorted(within))}")
        ok = within if ok is None else ok & within
    if ok:
        sizes = t[t.task == "category"].set_index("set").n_features
        chosen = min(ok, key=lambda k: (sizes[k], k))
        log(f"  sets within 1 SD in both tasks: {', '.join(sorted(ok))}")
        log(f"  CHOSEN: {chosen} ({sizes[chosen]} features)")
    else:
        log("  no set is within 1 SD of the best in both tasks: decision goes back to AC")
    log(f"\n  Saved: results/30_feature_selection_{MODEL}.csv")


if __name__ == "__main__":
    main()
