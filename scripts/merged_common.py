"""merged_common.py — Shared definitions of the merged pipeline (docs/PIPELINE_SPEC.md).

One place for what every script 40+ must do identically:
  - data: LIPID MAPS entries (unit for tasks A and C), master table formulas (task B),
    accepted labels per formula (permissive scoring);
  - the six input lists (same definitions as 30_feature_selection.py);
  - the protocol: 5-fold StratifiedGroupKFold grouped by formula, seeds 0, 1, 2;
  - metrics (strict and permissive) and model factories.
"""
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RE_CODIGO, cargar_lmsd, hill, parsear  # noqa: E402

BASE = HERE.parent
DATOS, RES = BASE / "data", BASE / "results"
SEEDS = [0, 1, 2]
N_FOLDS = 5
EMPTY = {"", "-", "nan", "None"}


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = _load("rep04", "04_representations.py")      # construir() -> R1 (6-decimal), R2, R3, R4
v24 = _load("v24", "24_extra_variables.py")         # extra() -> the 23 R5 variables
m21 = _load("m21", "21_r1_mitchell_exact.py")       # vector_mitchell() -> exact R1


# ------------------------------------------------------------------ input lists
def tool10(e):
    """The ten features of the earlier classifier (classifier/features.py; DBE from C and H only)."""
    c, h, o, n, p, s = (e.get(x, 0) for x in "CHONPS")
    return [c, h, o, n, p, s, c + h + o + n + p + s, (2 * c + 2 - h) / 2 if c else 0.0,
            o / c if c else 0.0, o + n + p + s]


def features(formulas, sets=None):
    """{name: matrix} for the six input lists (same definitions as script 30), plus R3 and R4.

    Formulas must already be valid (parsear accepts them)."""
    r1, r2, r3, r4, t10 = [], [], [], [], []
    for f in formulas:
        e, mo = parsear(f)
        if mo:
            raise ValueError(f"invalid formula {f!r}: {mo}")
        a, b, c3, c4 = rep.construir(f)
        r1.append(a); r2.append(b); r3.append(c3); r4.append(c4); t10.append(tool10(e))
    r1, r2, r3, t10 = (np.array(x, dtype=float) for x in (r1, r2, r3, t10))
    c = r2[:, 0]
    cnum = np.column_stack([c % 5, c % 2])
    out = {
        "R1": m21.vector_mitchell(r1).astype(float),
        "T10": t10,
        "T12": np.hstack([t10, cnum]),
        "R2": r2,
        "R2C": np.hstack([r2, cnum]),
        "R5": np.hstack([r2, v24.extra(r2, list(rep.R2_COLS))]),
        "R3": r3,
        "R4": np.array(r4, dtype=np.float32),
    }
    return {k: v for k, v in out.items() if sets is None or k in sets}


# ------------------------------------------------------------------ data
def _split_alt(s):
    return [RE_CODIGO.sub("", a).strip() for a in str(s).split(";") if a.strip() not in EMPTY]


def load_entries():
    """LIPID MAPS entries with a valid formula, category and main class (unit of tasks A and C).

    Adds: hill (canonical formula). Returns (entries, accepted) where accepted[level][formula]
    is the set of labels LIPID MAPS accepts for that formula (all its entries + ALT_ fields).
    """
    d = cargar_lmsd(DATOS)
    d["hill"] = [None if mo else hill(e) for e, mo in (parsear(x) for x in d["FORMULA"])]
    d = d[d.hill.notna() & d.CATEGORY.notna() & d.MAIN_CLASS.notna()].reset_index(drop=True)
    d["CATEGORY"] = d.CATEGORY.astype(str)
    d["MAIN_CLASS"] = d.MAIN_CLASS.astype(str)
    acc = {"category": defaultdict(set), "main_class": defaultdict(set)}
    for f, cat, cls, acat, acls in zip(d.hill, d.CATEGORY, d.MAIN_CLASS,
                                       d.ALT_CATEGORIES, d.ALT_MAIN_CLASSES):
        acc["category"][f].add(cat)
        acc["category"][f].update(_split_alt(acat))
        acc["main_class"][f].add(cls)
        # main classes keep their bracketed code in both MAIN_CLASS and ALT_MAIN_CLASSES
        acc["main_class"][f].update(x.strip() for x in str(acls).split(";") if x.strip() not in EMPTY)
    return d, acc


def load_detection():
    """Task B: distinct formulas of the master table, lipid (1) vs non-lipid (0).

    accepted[f] = {0, 1} for the lipid formulas that also appear in a curated non-lipid source."""
    m = pd.read_parquet(DATOS / "master.parquet")
    m = m[m.label.isin(["lipid", "non_lipid"])].reset_index(drop=True)
    y = (m.label == "lipid").astype(int).to_numpy()
    both = (m.label == "lipid") & (m.in_refmet_nonlipid | m.in_chebi | m.in_coconut)
    accepted = {f: ({0, 1} if b else {int(v)}) for f, v, b in zip(m.formula, y, both)}
    return m, y, accepted


# ------------------------------------------------------------------ protocol
def splits(y, groups, seed):
    """5-fold StratifiedGroupKFold grouped by formula (spec D3)."""
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups)


def oof(make_model, X, y, groups, seed, proba=False, fit_kw=None):
    """Out-of-fold predictions (and class probabilities aligned to np.unique(y))."""
    classes = np.unique(y)
    pred = np.empty(len(y), dtype=y.dtype)
    P = np.zeros((len(y), len(classes)), dtype=np.float32) if proba else None
    for tr, te in splits(y, groups, seed):
        seen, ytr = np.unique(y[tr], return_inverse=True)   # rare classes may be absent
        m = make_model(seed)
        kw = fit_kw(ytr) if fit_kw else {}
        m.fit(X[tr], ytr, **kw)
        pred[te] = seen[m.predict(X[te]).astype(int)]
        if proba:
            cols = np.searchsorted(classes, seen)
            P[np.ix_(te, cols)] = m.predict_proba(X[te])
    return (pred, P) if proba else pred


def scores(y, pred, formulas=None, accepted=None):
    """Strict metrics; permissive accuracy if accepted labels are given."""
    out = {"accuracy": accuracy_score(y, pred) * 100,
           "macro_f1": f1_score(y, pred, average="macro", zero_division=0) * 100,
           "balanced_accuracy": balanced_accuracy_score(y, pred) * 100}
    if accepted is not None:
        out["permissive_accuracy"] = np.mean([p in accepted[f] for p, f in zip(pred, formulas)]) * 100
    return out


def mean_sd(rows):
    """rows: list of dicts (one per seed) -> {metric: (mean, sd)}."""
    t = pd.DataFrame(rows)
    return {c: (t[c].mean(), t[c].std(ddof=1)) for c in t.columns}


def logger(name):
    RES.mkdir(exist_ok=True)
    fh = open(RES / f"{name}.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()
    return log
