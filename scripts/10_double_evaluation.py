#!/usr/bin/env python
"""
10_double_evaluation.py — Double evaluation of tasks A and B.

Session 19 (07_multilabel.py) discovered that task C was poorly evaluated: 21.1 % of
its "errors" were alternative official classes. It was corrected only for
task C, which left the project speaking two languages — C with a double criterion, A and B
with the strict criterion. This script unifies it.

The two ways of correcting the exam:

  STRICT      counted correct only if it matches the majority label of the formula
              → answers "do I get the label of THIS molecule right?"
  PERMISSIVE  counted correct if it matches ANY label that the database recognizes for
              that formula
              → answers "do I give a label compatible with THIS formula?", which is the
                question of the real MS1 scenario, where only the formula is available and
                what is sought is a defensible annotation (MSI level 3)

Admissible set in each task:

  A category    all the CATEGORY values in which the formula appears in the LMSD, plus those
                declared in ALT_CATEGORIES (a field only present in LMSD_extended)
  B detection   {lipid} for the LIPID MAPS formulas; and also {non-lipid} for
                the 3,779 that ALSO appear in a curated non-lipid database.
                That is the 38.60 % conflict of session 8: for those formulas
                both answers are defensible, because both molecules exist.

Model: the Phase 3 winner in each task, without retraining anything new
(A: R2 + balanced RF · B: R2 + XGBoost). Same 5 seeds and same split.
What changes between the two columns is NOT the model: it is the correctness criterion.

Output: results/10_double_evaluation.log + .csv
"""
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hill, parsear  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
SEMILLAS = [0, 1, 2, 3, 4]
N_ARB = 500
VACIO = {"", "-", "N/A", "None", "nan"}


def leer_sdf(ruta, campos):
    """Reads the indicated fields of an SDF without loading the structures.
    Copied from 07_multilabel.py to avoid creating dependencies between scripts."""
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


def evaluar(X, y_txt, admisibles, formulas, fabrica, balanceado, log):
    """Trains and returns both evaluations, averaged over the seeds."""
    clases = sorted(set(y_txt))
    cod = {c: i for i, c in enumerate(clases)}
    yc = np.array([cod[v] for v in y_txt])

    filas = []
    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(yc))
        c = int(0.8 * len(yc))
        itr, ite = perm[:c], perm[c:]

        m = fabrica()
        if balanceado:
            m.fit(X[itr], yc[itr], sample_weight=compute_sample_weight("balanced", yc[itr]))
        else:
            m.fit(X[itr], yc[itr])
        p = m.predict(X[ite])
        pred_txt = np.array([clases[i] for i in p])

        est = (p == yc[ite])
        per = np.array([pt in admisibles[f] for pt, f in zip(pred_txt, formulas[ite])])
        ambiguas = np.array([len(admisibles[f]) > 1 for f in formulas[ite]])

        filas.append({
            "acc_strict": est.mean() * 100,
            "acc_permissive": per.mean() * 100,
            "f1_strict": f1_score(yc[ite], p, average="macro", zero_division=0) * 100,
            "bal_strict": balanced_accuracy_score(yc[ite], p) * 100,
            "rescued": int((~est & per).sum()),
            "strict_failures": int((~est).sum()),
            "n_test": len(ite),
            "pct_ambiguous": ambiguas.mean() * 100,
        })
    return pd.DataFrame(filas)


def informar(nombre, t, log, filas_csv, notas=()):
    m = t.mean()
    log(f"\n{'-' * 84}")
    log(f"{nombre}")
    log(f"{'-' * 84}")
    for n in notas:
        log(f"  {n}")
    log(f"  test formulas: {m['n_test']:,.0f}  ·  with more than one admissible "
        f"label: {m['pct_ambiguous']:.2f} %")
    log("")
    log(f"  {'Evaluation':22s} {'Accuracy':>12s}")
    log(f"  {'-' * 22} {'-' * 12}")
    log(f"  {'STRICT':22s} {m['acc_strict']:11.2f}%   (± {t['acc_strict'].std():.2f})")
    log(f"  {'PERMISSIVE':22s} {m['acc_permissive']:11.2f}%   (± {t['acc_permissive'].std():.2f})")
    log(f"  {'difference':22s} {m['acc_permissive'] - m['acc_strict']:+11.2f}")
    if m["strict_failures"] > 0:
        log(f"\n  Of the {m['strict_failures']:.0f} failures per seed (mean), "
            f"{m['rescued']:.0f} were an admissible")
        log(f"  official label → {m['rescued'] / m['strict_failures'] * 100:.1f} % of the "
            f"'errors' were correct answers.")
    log(f"\n  Reference with the strict criterion: macro F1 {m['f1_strict']:.2f} %  ·  "
        f"balanced accuracy {m['bal_strict']:.2f} %")
    filas_csv.append({"task": nombre, **{k: round(float(v), 3) for k, v in m.items()}})


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "10_double_evaluation.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    log("=" * 84)
    log("DOUBLE EVALUATION OF TASKS A AND B — strict versus permissive")
    log("=" * 84)
    log("Same model in both columns. What changes is the correctness criterion.")
    log("Completes what session 19 did only for task C.")

    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    formulas, etiqueta, R2 = z["formula"], z["label"], z["R2"]
    filas_csv = []

    # ------------------------------------------------------------------ TASK A
    campos = {"FORMULA", "CATEGORY", "ALT_CATEGORIES"}
    d = leer_sdf(DATOS / "structures_extended.sdf", campos)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna() & d["CATEGORY"].notna()].copy()

    mayor_cat, admis_cat = {}, defaultdict(set)
    for f, sub in d.groupby("hill"):
        mayor_cat[f] = sub["CATEGORY"].value_counts().index[0]
        for _, r in sub.iterrows():
            admis_cat[f].add(r["CATEGORY"])
            for a in str(r.get("ALT_CATEGORIES", "")).split(";"):
                a = a.strip()
                if a not in VACIO:
                    admis_cat[f].add(a)

    selA = np.array([f in mayor_cat for f in formulas])
    FA, XA = formulas[selA], R2[selA]
    yA = np.array([mayor_cat[f] for f in FA])
    tA = evaluar(XA, yA, admis_cat, FA,
                 lambda: RandomForestClassifier(n_estimators=N_ARB, n_jobs=4,
                                                random_state=0, class_weight="balanced"),
                 balanceado=False, log=log)
    informar("TASK A — Lipid category (8)", tA, log, filas_csv,
             notas=["model: R2 + balanced Random Forest (Phase 3 winner)",
                    "admissible: CATEGORY of all its entries + ALT_CATEGORIES"])

    # ------------------------------------------------------------------ TASK B
    maestro = pd.read_parquet(DATOS / "master.parquet")
    en_neg = maestro.set_index("formula")[
        ["in_refmet_nonlipid", "in_chebi", "in_coconut", "in_lipidmaps"]]

    admis_det = {}
    for f, r in en_neg.iterrows():
        adm = set()
        if r["in_lipidmaps"]:
            adm.add("lipid")
            # collides with a curated non-lipid: both answers are defensible,
            # because both molecules with that same formula exist
            if r["in_refmet_nonlipid"] or r["in_chebi"] or r["in_coconut"]:
                adm.add("non_lipid")
        else:
            adm.add("non_lipid")
        admis_det[f] = adm

    n_col = sum(1 for f in formulas if len(admis_det.get(f, set())) > 1)
    tB = evaluar(R2, etiqueta, admis_det, formulas,
                 lambda: xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=4,
                                           tree_method="hist", verbosity=0, random_state=0),
                 balanceado=False, log=log)
    informar("TASK B — Lipid / non-lipid detection", tB, log, filas_csv,
             notas=["model: R2 + XGBoost (Phase 3 winner)",
                    f"admissible: both labels for the {n_col:,} LIPID MAPS formulas",
                    "that also appear in a curated non-lipid database (the 38.60 %)"])

    pd.DataFrame(filas_csv).to_csv(RES / "10_double_evaluation.csv", index=False)
    log("\n" + "=" * 84)
    log("Saved: results/10_double_evaluation.csv")
    log("Compare with results/07_multilabel.csv (task C: 82.14 % → 85.90 %).")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
