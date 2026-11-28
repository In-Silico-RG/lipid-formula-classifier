"""review_common.py — Shared helpers of scripts 90-95 (calculations answering the simulated peer review of V_12).

Everything reuses the protocol of merged_common.py (same data, same input lists, same 5-fold split grouped by
formula, same random forest: 500 trees, balanced class weights, default settings otherwise).
Caches (out-of-fold probabilities, features) go to results/cache/*.npz, which git ignores.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
import merged_common as mc  # noqa: E402

JOBS = 6
CACHE = mc.RES / "cache"
CACHE.mkdir(exist_ok=True)


def rf_weighted(seed):
    """The final model: random forest, 500 trees, class_weight balanced (as 40_models.py, RF-weighted)."""
    return RandomForestClassifier(n_estimators=500, n_jobs=JOBS, random_state=seed, class_weight="balanced")


def rf_plain(seed):
    return RandomForestClassifier(n_estimators=500, n_jobs=JOBS, random_state=seed)


def load_entries_features(sets=("R2", "R2C")):
    """LIPID MAPS entries and the requested input lists (cached: the parser is slow)."""
    d, acc = mc.load_entries()
    f = CACHE / ("features_" + "_".join(sets) + ".npz")
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if (z["hill"] == d.hill.to_numpy()).all():
            return d, acc, {k: z[k] for k in sets}
    FE = mc.features(d.hill.tolist(), sets=list(sets))
    np.savez_compressed(f, hill=d.hill.to_numpy(), **FE)
    return d, acc, FE


def series_key(formula):
    """Homologous (CH2) series: same heteroatoms and same rings plus double bonds, any carbon number."""
    e, _ = mc.parsear(formula)
    c, h, n = e.get("C", 0), e.get("H", 0), e.get("N", 0)
    others = tuple(sorted((k, v) for k, v in e.items() if k not in ("C", "H", "N", "O", "P", "S", "_carga") and v))
    dbe = c - h / 2 + n / 2 + 1
    return (e.get("O", 0), n, e.get("P", 0), e.get("S", 0), others, round(dbe * 2) / 2)


STEPS = [{"C": 1, "H": 2}, {"C": -1, "H": -2}, {"C": 2, "H": 4}, {"C": -2, "H": -4}, {"H": 2}, {"H": -2}]


def relatives(formula):
    """Formulas one step away: ±CH2, ±C2H4, ±H2 (same rule as 52_relatives.py)."""
    e, _ = mc.parsear(formula)
    out = []
    for st in STEPS:
        r = dict(e)
        for k, v in st.items():
            r[k] = r.get(k, 0) + v
        if all(v >= 0 for k, v in r.items() if k != "_carga") and r.get("C", 0) > 0:
            out.append(mc.hill({k: v for k, v in r.items() if k == "_carga" or v > 0}))
    return out


def share_with_relative(formulas, groups, y, seed):
    """For each entry, whether its formula has a relative (±CH2, ±C2H4, ±H2) in the training part of its fold."""
    rel = {f: relatives(f) for f in np.unique(formulas)}
    has = np.zeros(len(y), bool)
    for tr, te in mc.splits(y, groups, seed):
        train_f = set(formulas[tr])
        has[te] = [any(r in train_f for r in rel[x]) for x in formulas[te]]
    return has


def cached_oof(tag, make, X, y, groups, seed):
    """Out-of-fold (predictions, probabilities, classes), cached by tag."""
    f = CACHE / f"oof_{tag}_s{seed}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z["pred"], z["proba"], z["classes"]
    pred, P = mc.oof(make, X, y, groups, seed, proba=True)
    cl = np.unique(y)
    np.savez_compressed(f, pred=pred, proba=P, classes=cl)
    return pred, P, cl


def family_groups(formulas):
    """Connected families: formulas linked by any chain of +-CH2, +-C2H4 or +-H2 steps share a group id.

    With these groups no formula has a one-step relative on the other side of the split (strict homologue-free split)."""
    uf = sorted(set(formulas))
    idx = {f: i for i, f in enumerate(uf)}
    par = list(range(len(uf)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for f in uf:
        for r in relatives(f):
            if r in idx:
                a, b = find(idx[f]), find(idx[r])
                if a != b:
                    par[a] = b
    fam = {f: find(idx[f]) for f in uf}
    return np.array([fam[f] for f in formulas])


def load_detection_audited():
    """Task B exactly as the paper uses it: master.parquet with the 313 audited formulas set aside (script 55).

    data/master.parquet of this checkout is a link to a copy that still holds the 84 053 non-lipids of before the
    audit; the paper (and results/40_models_R2C.log) use 83 740, i.e. 93 675 formulas in all. The exclusion of
    55_exclude_audited.py is applied here in memory (same rule, same generator), and the counts are asserted."""
    m = pd.read_parquet(mc.DATOS / "master.parquet")
    if "audited" not in m.columns:
        a54 = mc._load("a54", "54_audit_nonlipids.py")
        gen = a54.generate()
        hit = (m.label == "non_lipid") & m.formula.isin(gen.keys())
        m.loc[hit, "label"] = "discarded"
    m = m[m.label.isin(["lipid", "non_lipid"])].reset_index(drop=True)
    y = (m.label == "lipid").astype(int).to_numpy()
    assert len(m) == 93675 and int(y.sum()) == 9935, (len(m), int(y.sum()))
    both = (m.label == "lipid") & (m.in_refmet_nonlipid | m.in_chebi | m.in_coconut)
    accepted = {f: ({0, 1} if b else {int(v)}) for f, v, b in zip(m.formula, y, both)}
    return m, y, accepted
