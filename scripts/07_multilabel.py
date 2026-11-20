#!/usr/bin/env python
"""
07_multilabel.py — Is task C wrongly evaluated?

LIPID MAPS acknowledges that 7.14 % of lipids legitimately belong to more than
one main class (field ALT_MAIN_CLASSES of LMSD_extended, which the normal SDF does not
include). Our evaluation counts only one as correct, so if the model
predicts the ALTERNATIVE class -valid according to the database itself- we count it as an error.

This script measures that effect by comparing two evaluations of the SAME model:

  STRICT     hit only if it matches the majority class of the formula
  PERMISSIVE hit if it matches ANY of the classes that LIPID MAPS
             recognizes for that formula

The difference between the two is how much we were over-penalizing. It is also, in
itself, a figure that nobody has published: Mitchell collapsed the multiple labels without
declaring it, which inflates its accuracy without allowing the effect to be measured.

Output: results/07_multilabel.log + .csv
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hill, parsear  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
SEMILLAS = [0, 1, 2, 3, 4]
VACIO = {"", "-", "N/A", "None"}


def leer_sdf(ruta, campos):
    """Reads the indicated fields from an SDF without loading the structures."""
    regs, cur, key = [], {}, None
    with open(ruta) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.startswith("> <"):
                key = ln[3:-1]
                continue
            if ln == "$$$$":
                if cur:
                    regs.append(cur)
                cur, key = {}, None
                continue
            if key and ln.strip():
                if key in campos:
                    cur[key] = cur.get(key, "") + ln.strip()
                key = None
    return pd.DataFrame(regs)


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "07_multilabel.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    log("=" * 84)
    log("IS TASK C WRONGLY EVALUATED? — effect of the alternative classes")
    log("=" * 84)

    campos = {"LM_ID", "FORMULA", "MAIN_CLASS", "ALT_MAIN_CLASSES", "CATEGORY", "ALT_CATEGORIES"}
    d = leer_sdf(DATOS / "structures_extended.sdf", campos)
    log(f"\n  LMSD_extended: {len(d):,} entries")

    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna() & d["MAIN_CLASS"].notna()].copy()
    log(f"  with valid formula and class: {len(d):,}  ·  {d['hill'].nunique():,} unique formulas")

    # ---- per formula: majority class and set of admissible classes
    mayor, admis = {}, defaultdict(set)
    for f, sub in d.groupby("hill"):
        mayor[f] = sub["MAIN_CLASS"].value_counts().index[0]
        for _, r in sub.iterrows():
            admis[f].add(r["MAIN_CLASS"])
            for a in str(r.get("ALT_MAIN_CLASSES", "")).split(";"):
                a = a.strip()
                if a not in VACIO:
                    admis[f].add(a)

    n_multi = sum(1 for f in admis if len(admis[f]) > 1)
    log(f"\n  formulas with MORE THAN ONE admissible class: {n_multi:,} "
        f"({n_multi/len(admis)*100:.2f} %)")
    tam = np.mean([len(v) for v in admis.values()])
    log(f"  admissible classes per formula, mean: {tam:.2f}")

    # ---- join with the already computed representations
    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    formulas, R2 = z["formula"], z["R2"]
    sel = np.array([f in mayor for f in formulas])
    F = formulas[sel]
    X = R2[sel]
    y = np.array([mayor[f] for f in F])

    vc = pd.Series(y).value_counts()
    keep = np.array([v not in set(vc[vc < 10].index) for v in y])
    F, X, y = F[keep], X[keep], y[keep]
    log(f"\n  working set: {len(y):,} formulas, {pd.Series(y).nunique()} classes")

    clases = sorted(set(y))
    cod = {c: i for i, c in enumerate(clases)}
    yc = np.array([cod[v] for v in y])

    log("\n" + "=" * 84)
    log("RESULT — same model, two ways of grading the exam")
    log("=" * 84)

    filas = []
    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(yc))
        c = int(0.8 * len(yc))
        itr, ite = perm[:c], perm[c:]
        m = xgb.XGBClassifier(n_estimators=500, n_jobs=12, tree_method="hist",
                              verbosity=0, random_state=0)
        m.fit(X[itr], yc[itr], sample_weight=compute_sample_weight("balanced", yc[itr]))
        p = m.predict(X[ite])
        pred_txt = np.array([clases[i] for i in p])

        est = (p == yc[ite])
        per = np.array([pt in admis[f] for pt, f in zip(pred_txt, F[ite])])
        filas.append({
            "acc_strict": est.mean() * 100,
            "acc_permissive": per.mean() * 100,
            "f1_strict": f1_score(yc[ite], p, average="macro", zero_division=0) * 100,
            "bal_strict": balanced_accuracy_score(yc[ite], p) * 100,
            "rescued": int((~est & per).sum()),
            "strict_failures": int((~est).sum()),
        })

    t = pd.DataFrame(filas)
    m_ = t.mean()
    log(f"\n  {'Evaluation':22s} {'Accuracy':>12s}")
    log(f"  {'-'*22} {'-'*12}")
    log(f"  {'STRICT':22s} {m_['acc_strict']:11.2f}%   (± {t['acc_strict'].std():.2f})")
    log(f"  {'PERMISSIVE':22s} {m_['acc_permissive']:11.2f}%   (± {t['acc_permissive'].std():.2f})")
    log(f"  {'difference':22s} {m_['acc_permissive']-m_['acc_strict']:+11.2f}")

    log(f"\n  Of the {m_['strict_failures']:.0f} failures counted per seed (mean):")
    log(f"    were actually an official alternative class: {m_['rescued']:.0f}")
    log(f"    → {m_['rescued']/m_['strict_failures']*100:.1f} % of the 'errors' were hits")

    log(f"\n  For reference, with strict evaluation:")
    log(f"    F1 macro          {m_['f1_strict']:6.2f} %")
    log(f"    balanced accuracy {m_['bal_strict']:6.2f} %")

    t.to_csv(RES / "07_multilabel.csv", index=False)
    log("\n  Saved: results/07_multilabel.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
