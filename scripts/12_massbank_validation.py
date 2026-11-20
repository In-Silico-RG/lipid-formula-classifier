#!/usr/bin/env python
"""
12_massbank_validation.py — The classifier on MassBank EU (E4, second database).

Complements 11_external_validation.py (GNPS/PNNL) on the two shortcomings that
it had:

  1. MINORITY CATEGORIES. PNNL contained no sterols, prenols or polyketides.
     MassBank provides 1.722 measured lipids covering all eight categories, including
     ~960 formulas from those three.
  2. REAL NEGATIVES. PNNL was all lipids, so it only measured sensitivity.
     MassBank provides 4.041 MEASURED non-lipids (formulas that match
     our curated negatives). With this, detection is evaluated fully:
     sensitivity AND specificity on external data.

Honest difference from 11: here the category label is NOT independent of the
catalogue (MassBank does not annotate lipid class; the ground truth comes from LIPID MAPS). What
is external is the SAMPLE — which molecules were measured — not the label.

MassBank formulas that are in none of our tables (642) are excluded:
it is not known whether they are lipids.

Protocol identical to 11: LEAKAGE-FREE scenario (NO LEAKAGE; all external formulas excluded
from training) and WITH-LEAKAGE scenario (CON LEAKAGE; trained on everything), 5 seeds, unit = formula.
MassBank EU 2026.03: each record carries its own Creative Commons licence (it is not
a single CC BY-SA for everything).

Output: results/12_massbank_validation.log + .csv
"""
import importlib.util
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
sys.path.insert(0, str(BASE / "scripts"))
from common import cargar_lmsd, hill, parsear  # noqa: E402

_spec = importlib.util.spec_from_file_location("rep", BASE / "scripts" / "04_representations.py")
_rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rep)
construir = _rep.construir

SEMILLAS = [0, 1, 2, 3, 4]
N_ARB = 500


def r2_de(fs):
    X, ok = [], []
    for f in fs:
        r = construir(f)
        if r is None:
            continue
        X.append(r[1])
        ok.append(f)
    return np.array(X, dtype=np.float32), ok


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "12_massbank_validation.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    log("=" * 84)
    log("EXTERNAL VALIDATION 2 — MassBank EU 2026.03: minority categories and real negatives")
    log("=" * 84)

    mb = pd.read_csv(DATOS / "spectra" / "formulas_massbank.csv")
    lip_ext = sorted(mb.loc[mb.in_lipidmaps, "formula"])
    neg_ext = sorted(mb.loc[mb.is_curated_negative, "formula"])
    n_esp = dict(zip(mb.formula, mb.n_spectra))
    log(f"\n  MassBank: {len(mb):,} unique formulas with an MS2 spectrum")
    log(f"    measured lipids (∩ LIPID MAPS):          {len(lip_ext):,}")
    log(f"    measured non-lipids (∩ curated negatives): {len(neg_ext):,}")
    log(f"    excluded for unknown status:              {int((~mb.in_lipidmaps & ~mb.is_curated_negative).sum()):,}")

    # ---------------------------------------------------------------- catalogue
    d = cargar_lmsd(DATOS)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna()]
    may_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0]).to_dict()
    todas_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: set(s)).to_dict()

    filas = []
    ext_set = set(lip_ext) | set(neg_ext)

    # ====================================================================== TAREA A
    log("\n" + "=" * 84)
    log("TASK A — Category · balanced RF on R2 · external sample, LIPID MAPS label")
    log("=" * 84)
    X_ext, f_ext = r2_de([f for f in lip_ext if f in may_cat])
    y_ext = np.array([may_cat[f] for f in f_ext])
    dist = Counter(y_ext)
    log("  external distribution (per formula): " + ", ".join(f"{k} {v}" for k, v in dist.most_common()))

    X_cat, f_cat = r2_de(list(may_cat))
    y_cat = np.array([may_cat[f] for f in f_cat])
    en_ext = np.array([f in ext_set for f in f_cat])
    clases = sorted(set(y_cat))
    cod = {c: i for i, c in enumerate(clases)}

    def eval_A(mask, nombre):
        acc = []
        for s in SEMILLAS:
            m = RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=s, class_weight="balanced")
            m.fit(X_cat[mask], [cod[v] for v in y_cat[mask]])
            p = np.array([clases[i] for i in m.predict(X_ext)])
            est = p == y_ext
            per = np.array([pi in todas_cat.get(f, {yi}) for pi, f, yi in zip(p, f_ext, y_ext)])
            w = np.array([n_esp.get(f, 1) for f in f_ext], dtype=float)
            rec = {c: recall_score(y_ext == c, p == c, zero_division=0) * 100 for c in sorted(dist)}
            acc.append({"strict": est.mean() * 100, "permissive": per.mean() * 100,
                        "strict_weighted_spectra": np.average(est, weights=w) * 100,
                        "f1_macro": f1_score(y_ext, p, average="macro", zero_division=0) * 100,
                        "bal_acc": balanced_accuracy_score(y_ext, p) * 100, **rec})
        t = pd.DataFrame(acc)
        m_ = t.mean()
        log(f"\n  {nombre}")
        log(f"    STRICT accuracy      {m_['strict']:6.2f} %  (± {t['strict'].std():.2f})"
            f"   · weighted by number of spectra {m_['strict_weighted_spectra']:6.2f} %")
        log(f"    PERMISSIVE accuracy  {m_['permissive']:6.2f} %  (± {t['permissive'].std():.2f})")
        log(f"    F1 macro {m_['f1_macro']:6.2f} %   · balanced accuracy {m_['bal_acc']:6.2f} %")
        log("    sensitivity per category:")
        for c in sorted(dist, key=lambda k: -dist[k]):
            log(f"      {c:22s} n={dist[c]:4d}   {m_[c]:6.1f} %")
        filas.append({"task": "A", "scenario": nombre, **{k: round(float(v), 2) for k, v in m_.items()}})

    eval_A(~en_ext, "NO LEAKAGE — trained excluding the MassBank formulas")
    eval_A(np.ones(len(f_cat), bool), "WITH LEAKAGE — trained on the whole catalogue")

    # ====================================================================== TAREA B
    log("\n" + "=" * 84)
    log("TASK B — Detection · XGBoost on R2 · external with BOTH classes measured")
    log("=" * 84)
    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    F_all, R2_all, et_all = z["formula"], z["R2"], z["label"]
    yb = (et_all == "lipid").astype(int)
    en_ext_B = np.array([f in ext_set for f in F_all])
    X_lip, f_lip = r2_de(lip_ext)
    X_neg, f_neg = r2_de(neg_ext)
    X_B = np.vstack([X_lip, X_neg])
    y_B = np.array([1] * len(f_lip) + [0] * len(f_neg))
    f_B = f_lip + f_neg
    log(f"  master: {len(F_all):,} formulas; {int(en_ext_B.sum()):,} are in MassBank (excluded in the NO LEAKAGE scenario)")
    log(f"  external: {len(f_lip):,} lipids + {len(f_neg):,} non-lipids = {len(f_B):,}")

    def eval_B(mask, nombre):
        acc = []
        for s in SEMILLAS:
            m = xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=s)
            m.fit(R2_all[mask], yb[mask])
            p = m.predict(X_B)
            acc.append({"sensitivity": recall_score(y_B, p) * 100,
                        "specificity": recall_score(1 - y_B, 1 - p) * 100,
                        "bal_acc": balanced_accuracy_score(y_B, p) * 100,
                        "f1_macro": f1_score(y_B, p, average="macro") * 100,
                        "accuracy": (p == y_B).mean() * 100})
        t = pd.DataFrame(acc)
        m_ = t.mean()
        log(f"\n  {nombre}")
        log(f"    sensitivity (lipids detected)       {m_['sensitivity']:6.2f} %  (± {t['sensitivity'].std():.2f})")
        log(f"    specificity (non-lipids rejected)     {m_['specificity']:6.2f} %  (± {t['specificity'].std():.2f})")
        log(f"    balanced accuracy {m_['bal_acc']:6.2f} %   · F1 macro {m_['f1_macro']:6.2f} %   · accuracy {m_['accuracy']:6.2f} %")
        filas.append({"task": "B", "scenario": nombre, **{k: round(float(v), 2) for k, v in m_.items()}})

    eval_B(~en_ext_B, "NO LEAKAGE — trained excluding the MassBank formulas")
    eval_B(np.ones(len(F_all), bool), "WITH LEAKAGE — trained on everything")
    log("\n  Internal reference for the same model (grouped split, session 21): balanced acc 89,1-90,2 %")

    # ---- where it fails, no leakage, seed 0
    m = xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=0)
    m.fit(R2_all[~en_ext_B], yb[~en_ext_B])
    p = m.predict(X_B)
    fn = [f for f, yi, pi in zip(f_B, y_B, p) if yi == 1 and pi == 0]
    fp = [f for f, yi, pi in zip(f_B, y_B, p) if yi == 0 and pi == 1]
    log(f"\n  real lipids NOT detected: {len(fn):,} of {len(f_lip):,}")
    cat_fn = Counter(may_cat.get(f, "?") for f in fn)
    cat_tot = Counter(may_cat.get(f, "?") for f in f_lip)
    for c, n in cat_fn.most_common():
        log(f"      {c:22s} {n:4d} of {cat_tot[c]:4d}  ({n / cat_tot[c] * 100:5.1f} % missed)")
    log(f"  real non-lipids taken as lipid: {len(fp):,} of {len(f_neg):,}")
    log("    examples: " + ", ".join(sorted(fp)[:10]))

    pd.DataFrame(filas).to_csv(RES / "12_massbank_validation.csv", index=False)
    log("\n" + "=" * 84)
    log("Saved: results/12_massbank_validation.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
