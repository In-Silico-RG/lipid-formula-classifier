#!/usr/bin/env python
"""
14_phase5_evaluation.py — Phase 5: honest evaluation, what was missing.

Already done in Phase 5: strict/permissive double evaluation (07, 10), external
validation (11, 12) and calibrated abstention via conformal prediction (13). What was missing and
is done by this script — everything that Mitchell 2020 does not report:

  1. Sensitivity, precision, F1, support and AUC-PR PER CATEGORY (task A), per class
     (task B) and per main class (task C).
  2. Full confusion matrix, aggregated over the 5 seeds (A and B; that of C, with
     57 classes, is saved to CSV).
  3. End-to-end category → class CASCADE (improvement #4 of the plan): the only figure
     that matters to the real user and that the paper never gives. Two output
     architectures are compared:
       · FLAT: a single 57-class model (the Phase 3 one);
       · HIERARCHICAL (Mitchell's): category model → class model within
         the predicted category. A category error carries over to the class.
     End-to-end accuracy and how the error is distributed are reported.

Protocol identical to the rest: 80/20 formula-grouped split, 5 seeds, Phase 3
winning models on R2 (RF-bal in A; XGB in B; XGB-bal in C). Main metric
macro F1; accuracy is shown only for comparison with the literature.

Output: results/14_phase5_*.log/.csv and figures/14_confusion_*.png (if matplotlib is available)
"""
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
DATOS, RES, FIG = BASE / "data", BASE / "results", BASE / "figures"
sys.path.insert(0, str(BASE / "scripts"))
from common import cargar_lmsd, hill, parsear  # noqa: E402

SEMILLAS = [0, 1, 2, 3, 4]
N_ARB = 500
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAY_MPL = True
except Exception:
    HAY_MPL = False


def particion(n, s):
    rng = np.random.default_rng(s)
    perm = rng.permutation(n)
    c = int(0.8 * n)
    return perm[:c], perm[c:]


def rf_bal():
    return RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=0, class_weight="balanced")


def xgb_(bal):
    return xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=0), bal


def ajustar(modelo_bal, X, y):
    m, bal = modelo_bal if isinstance(modelo_bal, tuple) else (modelo_bal, False)
    if bal:
        m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        m.fit(X, y)
    return m


def por_clase(X, y_txt, fabrica, nombre, log, guardar_cm=True):
    """Per-class metrics aggregated over seeds + accumulated confusion matrix."""
    clases = np.array(sorted(set(y_txt)))
    cod = {c: i for i, c in enumerate(clases)}
    y = np.array([cod[v] for v in y_txt])
    k = len(clases)
    cm = np.zeros((k, k), dtype=int)
    prec, rec, f1s, aps = [], [], [], []
    glob = []
    for s in SEMILLAS:
        itr, ite = particion(len(y), s)
        m = ajustar(fabrica(), X[itr], y[itr])
        p = m.predict(X[ite])
        P = m.predict_proba(X[ite])
        cm += confusion_matrix(y[ite], p, labels=range(k))
        pr, rc, f1, _ = precision_recall_fscore_support(y[ite], p, labels=range(k), zero_division=0)
        prec.append(pr); rec.append(rc); f1s.append(f1)
        ap = np.full(k, np.nan)
        for j in range(k):
            pos = (y[ite] == j)
            if pos.any() and (~pos).any():
                ap[j] = average_precision_score(pos, P[:, j])
        aps.append(ap)
        glob.append({"accuracy": (p == y[ite]).mean() * 100,
                     "bal_acc": balanced_accuracy_score(y[ite], p) * 100,
                     "f1_macro": f1_score(y[ite], p, average="macro", zero_division=0) * 100})
    soporte = np.bincount(y, minlength=k)
    t = pd.DataFrame({"class": clases, "n_formulas": soporte,
                      "sensitivity": np.mean(rec, 0) * 100, "precision": np.mean(prec, 0) * 100,
                      "f1": np.mean(f1s, 0) * 100, "auc_pr": np.nanmean(aps, 0),
                      "sd_sensitivity": np.std(rec, 0) * 100})
    t = t.sort_values("n_formulas", ascending=False)
    g = pd.DataFrame(glob).mean()
    log(f"\n  global: accuracy {g['accuracy']:.2f} %  · bal.acc {g['bal_acc']:.2f} %  · F1 macro {g['f1_macro']:.2f} %")
    log(f"  {'class':28s} {'n':>6s} {'sens.':>7s} {'prec.':>7s} {'F1':>7s} {'AUC-PR':>7s}")
    for _, r in t.iterrows():
        log(f"  {str(r['class'])[:28]:28s} {int(r['n_formulas']):6d} {r['sensitivity']:6.1f}% {r['precision']:6.1f}% "
            f"{r['f1']:6.1f}% {r['auc_pr']:7.3f}")
    t.to_csv(RES / f"14_phase5_{nombre}_per_class.csv", index=False)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    pd.DataFrame(cm, index=clases, columns=clases).to_csv(RES / f"14_phase5_{nombre}_confusion.csv")
    if guardar_cm and HAY_MPL and k <= 12:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(clases, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(clases, fontsize=8)
        ax.set_xlabel("predicted label"); ax.set_ylabel("true (majority)")
        for i in range(k):
            for j in range(k):
                if cm[i, j]:
                    ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of the row")
        ax.set_title(f"{nombre}: confusion matrix (5 seeds, %)")
        fig.tight_layout(); fig.savefig(FIG / f"14_confusion_{nombre}.png", dpi=150); plt.close(fig)
    # main off-diagonal confusions
    off = [(cmn[i, j], clases[i], clases[j], cm[i, j]) for i in range(k) for j in range(k) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    log("  main confusions (true → predicted, % of the row, n):")
    for fr, a, b, n in off[:6]:
        log(f"    {str(a)[:26]:26s} → {str(b)[:26]:26s} {fr*100:5.1f} %  (n={n})")
    return t, g


def main():
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    fh = open(RES / "14_phase5_evaluation.log", "w")

    def log(m=""):
        print(m, flush=True); fh.write(m + "\n"); fh.flush()

    log("=" * 84)
    log("PHASE 5 — per-class evaluation, confusion matrices and category → class cascade")
    log("=" * 84)

    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    F_all, R2, et_all = z["formula"], z["R2"], z["label"]
    d = cargar_lmsd(DATOS)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna()]
    may_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0]).to_dict()
    may_cls = d.groupby("hill")["MAIN_CLASS"].agg(lambda s: s.value_counts().index[0]).to_dict()
    cls2cat = d.groupby("MAIN_CLASS")["CATEGORY"].agg(lambda s: s.value_counts().index[0]).to_dict()

    # ------------------------------------------------------------------ A
    log("\n" + "=" * 84); log("TASK A — Category · balanced RF · R2"); log("=" * 84)
    selA = np.array([f in may_cat for f in F_all])
    yA = np.array([may_cat[f] for f in F_all[selA]])
    tA, gA = por_clase(R2[selA], yA, rf_bal, "A_category", log)

    # ------------------------------------------------------------------ B
    log("\n" + "=" * 84); log("TASK B — Detection · XGB · R2"); log("=" * 84)
    tB, gB = por_clase(R2, et_all, lambda: xgb_(False), "B_detection", log)

    # ------------------------------------------------------------------ C
    log("\n" + "=" * 84); log("TASK C — Main class · balanced XGB · R2"); log("=" * 84)
    selC = np.array([f in may_cls for f in F_all])
    yC_all = np.array([may_cls[f] for f in F_all[selC]])
    vc = pd.Series(yC_all).value_counts()
    keep = np.array([vc[v] >= 10 for v in yC_all])
    FC, XC, yC = F_all[selC][keep], R2[selC][keep], yC_all[keep]
    log(f"  {len(yC):,} formulas · {pd.Series(yC).nunique()} classes (≥ 10 formulas)")
    tC, gC = por_clase(XC, yC, lambda: xgb_(True), "C_class", log, guardar_cm=False)
    log("\n  classes with the worst sensitivity (n ≥ 30):")
    for _, r in tC[tC.n_formulas >= 30].sort_values("sensitivity").head(8).iterrows():
        log(f"    {r['class'][:40]:40s} n={int(r['n_formulas']):4d}  sens. {r['sensitivity']:5.1f} %  F1 {r['f1']:5.1f} %")

    # ------------------------------------------------------------ CASCADE
    log("\n" + "=" * 84)
    log("CASCADE category → class, end to end (improvement #4)")
    log("=" * 84)
    catC = np.array([cls2cat[c] for c in yC])           # true category of each formula in C
    clasesC = sorted(set(yC)); codC = {c: i for i, c in enumerate(clasesC)}
    yCi = np.array([codC[v] for v in yC])
    catsC = sorted(set(catC)); codCat = {c: i for i, c in enumerate(catsC)}
    yCati = np.array([codCat[v] for v in catC])
    filas = []
    for s in SEMILLAS:
        itr, ite = particion(len(yC), s)
        # FLAT
        m_flat = ajustar(xgb_(True), XC[itr], yCi[itr])
        p_flat = m_flat.predict(XC[ite])
        # HIERARCHICAL: category, then class within the predicted category
        m_cat = ajustar(rf_bal(), XC[itr], yCati[itr])
        p_cat = m_cat.predict(XC[ite])
        sub = {}
        for ci, cat in enumerate(catsC):
            mk = yCati[itr] == ci
            ys = yCi[itr][mk]
            if len(set(ys)) == 1:
                sub[ci] = ("const", ys[0])
            else:
                # reindex local classes
                loc = sorted(set(ys)); lmap = {c: i for i, c in enumerate(loc)}
                mm = ajustar(xgb_(True), XC[itr][mk], np.array([lmap[v] for v in ys]))
                sub[ci] = ("model", mm, loc)
        p_hier = np.empty(len(ite), dtype=int)
        for ci in set(p_cat):
            mk = p_cat == ci
            kind = sub[ci][0]
            if kind == "const":
                p_hier[mk] = sub[ci][1]
            else:
                _, mm, loc = sub[ci]
                p_hier[mk] = np.array(loc)[mm.predict(XC[ite][mk])]
        cat_ok = p_cat == yCati[ite]
        flat_ok, hier_ok = p_flat == yCi[ite], p_hier == yCi[ite]
        filas.append({
            "flat_acc": flat_ok.mean() * 100,
            "flat_f1": f1_score(yCi[ite], p_flat, average="macro", zero_division=0) * 100,
            "hier_acc": hier_ok.mean() * 100,
            "hier_f1": f1_score(yCi[ite], p_hier, average="macro", zero_division=0) * 100,
            "cat_acc": cat_ok.mean() * 100,
            "hier_error_by_category": (~cat_ok).mean() * 100,
            "hier_error_within_category": (cat_ok & ~hier_ok).mean() * 100,
            "class_given_cat_ok": hier_ok[cat_ok].mean() * 100,
            "flat_cat_implicit_ok": np.mean([cls2cat[clasesC[a]] == cls2cat[clasesC[b]] for a, b in zip(p_flat, yCi[ite])]) * 100,
        })
    t = pd.DataFrame(filas); m = t.mean(); sd = t.std()
    log(f"\n  {'':34s} {'accuracy':>10s} {'macro F1':>10s}")
    log(f"  {'FLAT (57 classes)':34s} {m['flat_acc']:9.2f}% {m['flat_f1']:9.2f}%   (± {sd['flat_acc']:.2f})")
    log(f"  {'HIERARCHICAL (category → class)':34s} {m['hier_acc']:9.2f}% {m['hier_f1']:9.2f}%   (± {sd['hier_acc']:.2f})")
    log(f"\n  breakdown of the hierarchical model:")
    log(f"    category correct                         {m['cat_acc']:6.2f} %")
    log(f"    class correct GIVEN that the category is    {m['class_given_cat_ok']:6.2f} %")
    log(f"    total error = by category {m['hier_error_by_category']:5.2f} %  +  within the category {m['hier_error_within_category']:5.2f} %")
    log(f"  the FLAT model gets the implicit category right in {m['flat_cat_implicit_ok']:.2f} % of the cases")
    t.to_csv(RES / "14_phase5_cascade.csv", index=False)

    log("\n" + "=" * 84)
    log("Saved: results/14_phase5_*.csv" + (", figures/14_confusion_*.png" if HAY_MPL else " (no matplotlib: no figures)"))
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
