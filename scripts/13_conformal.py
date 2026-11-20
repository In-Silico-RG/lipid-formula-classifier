#!/usr/bin/env python
"""
13_conformal.py — E1: conformal prediction on the already-trained models.

The defining feature of this problem is IRREDUCIBLE AMBIGUITY: identical formulas with
different labels (strict upper limit 97.46 % at category level; 100 % permissive). The
right response is not a larger model (four families already tied) but to stop
predicting ONE label and predict a SET with a coverage guarantee.

Method: split conformal (Vovk; Lei et al.). For each seed:
  · 80/20 formula-grouped split, the same one used throughout Phase 3;
  · the 80 % is further divided into 64 % training and 16 % CALIBRATION;
  · nonconformity score s = 1 − p̂(true label) on the calibration set;
  · q̂ = the ⌈(n+1)(1−α)⌉/n quantile of those scores;
  · prediction set C(x) = { y : 1 − p̂_y(x) ≤ q̂ }.
Guarantee: P(y ∈ C(x)) ≥ 1 − α, with no assumptions on the model or the distribution,
for finite samples, provided calibration and test data are exchangeable.

PREREGISTERED (session 21, E1). Success criterion fixed before running:
  (a) empirical coverage within the sampling tolerance of the target, and
  (b) mean set size CLEARLY larger for ambiguous formulas than for
      non-ambiguous ones. If set size does not discriminate between the two
      populations, the method adds nothing.
Testable prediction (session 22): sets should grow for small fatty acids,
polyketides and prenols.

We also measure what happens to the guarantee under DISTRIBUTION SHIFT (calibrated on the
catalogue, evaluated on measured formulas: PNNL and MassBank). The theory does not promise it;
it is worth knowing how much it degrades.

Tasks: A category (RF-bal, R2) · C class (XGB-bal, R2) · B detection (XGB, R2).
Output: results/13_conformal.log + .csv
"""
import importlib.util
import sys
import warnings
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
sys.path.insert(0, str(BASE / "scripts"))
from common import hill, parsear  # noqa: E402

_spec = importlib.util.spec_from_file_location("rep", BASE / "scripts" / "04_representations.py")
_rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rep)
construir = _rep.construir

SEMILLAS = [0, 1, 2, 3, 4]
ALFAS = [0.10, 0.05, 0.01]
N_ARB = 500
VACIO = {"", "-", "N/A", "None", "nan"}
ABREV = {  # PNNL experimental annotation → category (identical to 11; session 33: previously an incomplete copy)
    **{k: "Glycerophospholipids" for k in
       ["PC", "PE", "PS", "PI", "PG", "PA", "CL", "LPC", "LPE", "LPS", "LPI", "LPG", "LPA",
        "PC O", "PE O", "PC P", "PE P", "PIP", "PIP2", "BMP", "PMeOH", "PEtOH"]},
    **{k: "Glycerolipids" for k in ["TG", "DG", "MG", "DGDG", "MGDG", "SQDG", "DGTS", "TG O", "DG O"]},
    **{k: "Sphingolipids" for k in
       ["SM", "Cer", "GalCer", "GlcCer", "HexCer", "LacCer", "Hex2Cer", "PI_Cer", "PE_Cer",
        "CerP", "SPB", "SPBP", "GM3", "Sulfatide", "SHexCer"]},
    **{k: "Sterol Lipids" for k in ["CE", "ChE", "Chol", "ST", "BA"]},
    **{k: "Fatty Acyls" for k in ["FA", "FAHFA", "CAR", "NAE", "OxFA"]},
    # session 33: ubiquinones (LIPID MAPS PR02, quinones and hydroquinones) and DGTS/DGTA (GL07, betaine diradylglycerols)
    **{k: "Prenol Lipids" for k in ["CoQ4", "CoQ8", "CoQ9", "CoQ10"]},
    "DGTSA": "Glycerolipids",
}


def leer_sdf(ruta, campos):
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


def r2_de(fs):
    X, ok = [], []
    for f in fs:
        r = construir(f)
        if r is not None:
            X.append(r[1])
            ok.append(f)
    return np.array(X, dtype=np.float32), ok


def cuantil_conforme(scores, alfa):
    n = len(scores)
    k = min(1.0, ceil((n + 1) * (1 - alfa)) / n)
    return float(np.quantile(scores, k, method="higher"))


def conjuntos(P, qhat):
    """Boolean matrix (n, classes): labels with 1 − p̂ ≤ q̂."""
    return (1.0 - P) <= qhat


class Conformal:
    """Wraps an sklearn/xgb model: fits, calibrates and produces prediction sets."""

    def __init__(self, fabrica, pesos_balanceados=False):
        self.fabrica, self.bal = fabrica, pesos_balanceados

    def ajustar(self, Xtr, ytr, Xcal, ycal):
        self.m = self.fabrica()
        if self.bal:
            self.m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        else:
            self.m.fit(Xtr, ytr)
        self.clases_ = np.array(self.m.classes_)
        P = self.m.predict_proba(Xcal)
        idx = np.searchsorted(self.clases_, ycal)
        self.scores_ = 1.0 - P[np.arange(len(ycal)), idx]
        self.q_ = {a: cuantil_conforme(self.scores_, a) for a in ALFAS}
        # Mondrian: one quantile per true class → class-CONDITIONAL guarantee.
        # With fewer than 20 calibration examples the class inherits the marginal quantile.
        self.qm_ = {}
        for a in ALFAS:
            qs = np.full(len(self.clases_), self.q_[a])
            for j in range(len(self.clases_)):
                sc = self.scores_[idx == j]
                if len(sc) >= 20:
                    qs[j] = cuantil_conforme(sc, a)
            self.qm_[a] = qs
        return self

    def predecir(self, X, alfa, mondrian=False):
        P = self.m.predict_proba(X)
        if mondrian:
            return (1.0 - P) <= self.qm_[alfa][None, :]
        return conjuntos(P, self.q_[alfa])


def evaluar_conjuntos(S, clases, y, admis, log, etiqueta, grupo=None):
    """S: (n, k) bool. y: encoded majority label. admis: list of sets of names."""
    n = len(y)
    cob_est = S[np.arange(n), y].mean() * 100
    nombres = [set(clases[np.where(r)[0]]) for r in S]
    cob_per = np.mean([len(s & a) > 0 for s, a in zip(nombres, admis)]) * 100
    tam = S.sum(1)
    amb = np.array([len(a) > 1 for a in admis])
    fila = {"coverage_strict": cob_est, "coverage_permissive": cob_per,
            "mean_size": tam.mean(), "singletons": (tam == 1).mean() * 100, "empty": (tam == 0).mean() * 100,
            "size_unambiguous": tam[~amb].mean() if (~amb).any() else np.nan,
            "size_ambiguous": tam[amb].mean() if amb.any() else np.nan}
    if grupo is not None:
        for g in sorted(set(grupo)):
            mk = grupo == g
            fila[f"size|{g}"] = tam[mk].mean()
            fila[f"cov|{g}"] = S[np.where(mk)[0], y[mk]].mean() * 100
    return fila


def resumen(filas, log, titulo, grupos=None):
    t = pd.DataFrame(filas)
    m, sd = t.mean(numeric_only=True), t.std(numeric_only=True)
    log(f"\n  {titulo}")
    log(f"    strict coverage  {m['coverage_strict']:6.2f} %  (± {sd['coverage_strict']:.2f})"
        f"   · permissive {m['coverage_permissive']:6.2f} %")
    log(f"    mean size {m['mean_size']:.3f}   · singletons {m['singletons']:5.1f} %   · empty {m['empty']:4.1f} %")
    log(f"    size on NON-ambiguous {m['size_unambiguous']:.3f}   vs   AMBIGUOUS {m['size_ambiguous']:.3f}"
        f"   → ratio {m['size_ambiguous'] / m['size_unambiguous']:.2f}×")
    if grupos:
        log("    by group (mean size · strict coverage):")
        for g in grupos:
            if f"size|{g}" in m:
                log(f"      {g:22s} {m[f'size|{g}']:.3f}   {m[f'cov|{g}']:6.1f} %")
    return {k: round(float(v), 3) for k, v in m.items()}


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "13_conformal.log", "w")

    def log(msg=""):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    log("=" * 84)
    log("E1 — CONFORMAL PREDICTION: sets with a coverage guarantee, not labels")
    log("=" * 84)
    log("split conformal · s = 1 − p̂(y) · 64/16/20 formula-grouped · 5 seeds · α ∈ {0,10, 0,05, 0,01}")

    # ---------------------------------------------------------------- labels
    campos = {"FORMULA", "CATEGORY", "ALT_CATEGORIES", "MAIN_CLASS", "ALT_MAIN_CLASSES"}
    d = leer_sdf(DATOS / "structures_extended.sdf", campos)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna() & d["CATEGORY"].notna()].copy()
    may_cat, adm_cat, may_cls, adm_cls = {}, defaultdict(set), {}, defaultdict(set)
    for f, sub in d.groupby("hill"):
        may_cat[f] = sub["CATEGORY"].value_counts().index[0]
        mc = sub["MAIN_CLASS"].dropna()
        if len(mc):
            may_cls[f] = mc.value_counts().index[0]
        for _, r in sub.iterrows():
            adm_cat[f].add(r["CATEGORY"])
            for a in str(r.get("ALT_CATEGORIES", "")).split(";"):
                if a.strip() not in VACIO:
                    adm_cat[f].add(a.strip())
            if pd.notna(r.get("MAIN_CLASS")):
                adm_cls[f].add(r["MAIN_CLASS"])
                for a in str(r.get("ALT_MAIN_CLASSES", "")).split(";"):
                    if a.strip() not in VACIO:
                        adm_cls[f].add(a.strip())

    # LMSD extended writes «Glycerophospholipids [GP]»; the PNNL annotation and the LIPID MAPS table
    # write «Glycerophospholipids». The suffix is stripped so that both sources match.
    import re as _re
    _lim = lambda c: _re.sub(r"\s*\[[A-Z]{2}\]$", "", str(c))
    may_cat = {f: _lim(c) for f, c in may_cat.items()}
    adm_cat = defaultdict(set, {f: {_lim(c) for c in v} for f, v in adm_cat.items()})

    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    F_all, R2_all, et_all = z["formula"], z["R2"], z["label"]
    maestro = pd.read_parquet(DATOS / "master.parquet").set_index("formula")
    salida = []

    def particion(n, s):
        rng = np.random.default_rng(s)
        perm = rng.permutation(n)
        c = int(0.8 * n)
        itr_all, ite = perm[:c], perm[c:]
        ncal = int(0.2 * len(itr_all))
        return itr_all[ncal:], itr_all[:ncal], ite

    # ================================================================ TAREA A
    log("\n" + "=" * 84)
    log("TASK A — Category (8) · balanced RF · strict upper limit 97,46 % / permissive 100 %")
    log("=" * 84)
    selA = np.array([f in may_cat for f in F_all])
    FA, XA = F_all[selA], R2_all[selA]
    yA_txt = np.array([may_cat[f] for f in FA])
    clasesA = np.array(sorted(set(yA_txt)))
    yA = np.searchsorted(clasesA, yA_txt)
    admA = [adm_cat[f] for f in FA]
    ambA = np.array([len(a) > 1 for a in admA])
    log(f"  {len(FA):,} formulas · ambiguous {ambA.mean() * 100:.2f} %")
    fabA = lambda: RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=0, class_weight="balanced")

    modelosA = []
    for s in SEMILLAS:
        itr, ical, ite = particion(len(FA), s)
        cf = Conformal(fabA).ajustar(XA[itr], yA[itr], XA[ical], yA[ical])
        modelosA.append((cf, ite))
    for a in ALFAS:
        for mond in (False, True):
            filas = []
            for cf, ite in modelosA:
                S = cf.predecir(XA[ite], a, mondrian=mond)
                filas.append(evaluar_conjuntos(S, clasesA, yA[ite], [admA[i] for i in ite], log,
                                               "A", grupo=yA_txt[ite]))
            var = "MONDRIAN (per class)" if mond else "marginal"
            r = resumen(filas, log, f"α = {a:.2f}  (target ≥ {100 * (1 - a):.0f} %) · {var}", grupos=list(clasesA))
            salida.append({"task": "A", "set": "interno", "variant": var, "alpha": a, **r})

    # reference point: the point classifier of the same model
    exac = np.mean([(cf.m.predict(XA[ite]) == yA[ite]).mean() for cf, ite in modelosA]) * 100
    log(f"\n  reference: point accuracy of the same RF on the same test sets {exac:.2f} %")

    # ---- distribution shift: PNNL (truth = annotation) and MassBank (truth = LIPID MAPS)
    log("\n  --- under DISTRIBUTION SHIFT (calibrated on the catalogue; no leakage) ---")
    ext = pd.read_csv(DATOS / "spectra" / "formulas_experimental.csv").fillna("")
    verdad = defaultdict(set)
    for f, ab in zip(ext.formula, ext.abrev):
        if ab and ab in ABREV:
            verdad[f].add(ABREV[ab])
    pnnl = sorted(verdad)
    mb = pd.read_csv(DATOS / "spectra" / "formulas_massbank.csv")
    mbl = sorted(f for f in mb.loc[mb.in_lipidmaps, "formula"] if f in may_cat)
    externos = {"PNNL (annotation)": (pnnl, lambda f: verdad[f]), "MassBank (LIPID MAPS)": (mbl, lambda f: {may_cat[f]})}
    ext_set = set(pnnl) | set(mbl)
    keep = np.array([f not in ext_set for f in FA])
    for nombre, (fs, truth) in externos.items():
        Xe, fe = r2_de(fs)
        adm_e = [truth(f) | adm_cat.get(f, set()) for f in fe]
        y_e = np.array([np.searchsorted(clasesA, sorted(truth(f))[0]) for f in fe])
        cat_e = np.array([sorted(truth(f))[0] for f in fe])
        cfs = []
        for s in SEMILLAS:
            idx = np.where(keep)[0]
            rng = np.random.default_rng(s)
            perm = rng.permutation(idx)
            ncal = int(0.2 * len(perm))
            ical, itr = perm[:ncal], perm[ncal:]
            cfs.append(Conformal(fabA).ajustar(XA[itr], yA[itr], XA[ical], yA[ical]))
        for a in [0.10, 0.05]:
            for mond in (False, True):
                filas = [evaluar_conjuntos(cf.predecir(Xe, a, mondrian=mond), clasesA, y_e, adm_e, log, nombre, grupo=cat_e)
                         for cf in cfs]
                var = "MONDRIAN (per class)" if mond else "marginal"
                r = resumen(filas, log, f"{nombre} · n={len(fe):,} · α = {a:.2f} · {var}", grupos=sorted(set(cat_e)))
                salida.append({"task": "A", "set": nombre, "variant": var, "alpha": a, **r})

    # ================================================================ TAREA C
    log("\n" + "=" * 84)
    log("TASK C — Main class · balanced XGB · strict upper limit 91,18 %")
    log("=" * 84)
    selC = np.array([f in may_cls for f in F_all])
    FC, XC = F_all[selC], R2_all[selC]
    yC_txt = np.array([may_cls[f] for f in FC])
    vc = pd.Series(yC_txt).value_counts()
    ok = np.array([vc[v] >= 10 for v in yC_txt])
    FC, XC, yC_txt = FC[ok], XC[ok], yC_txt[ok]
    clasesC = np.array(sorted(set(yC_txt)))
    yC = np.searchsorted(clasesC, yC_txt)
    admC = [adm_cls[f] for f in FC]
    log(f"  {len(FC):,} formulas · {len(clasesC)} classes · ambiguous {np.mean([len(a) > 1 for a in admC]) * 100:.2f} %")
    fabC = lambda: xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=0)
    modelosC = []
    for s in SEMILLAS:
        itr, ical, ite = particion(len(FC), s)
        cf = Conformal(fabC, pesos_balanceados=True).ajustar(XC[itr], yC[itr], XC[ical], yC[ical])
        modelosC.append((cf, ite))
    for a in ALFAS:
        for mond in (False, True):
            filas = [evaluar_conjuntos(cf.predecir(XC[ite], a, mondrian=mond), clasesC, yC[ite], [admC[i] for i in ite], log, "C")
                     for cf, ite in modelosC]
            var = "MONDRIAN (per class)" if mond else "marginal"
            r = resumen(filas, log, f"α = {a:.2f}  (target ≥ {100 * (1 - a):.0f} %) · {var}")
            salida.append({"task": "C", "set": "interno", "variant": var, "alpha": a, **r})
    exac = np.mean([(cf.m.predict(XC[ite]) == yC[ite]).mean() for cf, ite in modelosC]) * 100
    log(f"\n  reference: point accuracy of the same XGB {exac:.2f} %  (out of 57 possible classes)")

    # ================================================================ TAREA B
    log("\n" + "=" * 84)
    log("TASK B — Detection · XGB · the set {lipid, non-lipid} means «cannot decide»")
    log("=" * 84)
    yB_txt = et_all
    clasesB = np.array(sorted(set(yB_txt)))
    yB = np.searchsorted(clasesB, yB_txt)
    admB = []
    for f in F_all:
        r = maestro.loc[f]
        s_ = {"lipid"} if r["in_lipidmaps"] else {"non_lipid"}
        if r["in_lipidmaps"] and (r["in_refmet_nonlipid"] or r["in_chebi"] or r["in_coconut"]):
            s_.add("non_lipid")
        admB.append(s_)
    fabB = lambda: xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=0)
    modelosB = []
    for s in SEMILLAS:
        itr, ical, ite = particion(len(F_all), s)
        cf = Conformal(fabB).ajustar(R2_all[itr], yB[itr], R2_all[ical], yB[ical])
        modelosB.append((cf, ite))
    for a in ALFAS:
        for mond in (False, True):
            filas = [evaluar_conjuntos(cf.predecir(R2_all[ite], a, mondrian=mond), clasesB, yB[ite], [admB[i] for i in ite], log, "B",
                                       grupo=yB_txt[ite]) for cf, ite in modelosB]
            var = "MONDRIAN (per class)" if mond else "marginal"
            r = resumen(filas, log, f"α = {a:.2f}  (target ≥ {100 * (1 - a):.0f} %) · {var}", grupos=list(clasesB))
            salida.append({"task": "B", "set": "interno", "variant": var, "alpha": a, **r})
    # how many colliding lipids receive the double set
    cf, ite = modelosB[0]
    amb = np.array([len(admB[i]) > 1 for i in ite])
    for mond in (False, True):
        for a in (0.05, 0.01):
            S = cf.predecir(R2_all[ite], a, mondrian=mond)
            doble = S.sum(1) == 2
            log(f"  α = {a:.2f}, {'Mondrian' if mond else 'marginal'}, seed 0: double set («cannot decide») in the "
                f"{doble[amb].mean() * 100:.1f} % of the formulas that COLLIDE and {doble[~amb].mean() * 100:.1f} % of the others")

    pd.DataFrame(salida).to_csv(RES / "13_conformal.csv", index=False)
    log("\n" + "=" * 84)
    log("Saved: results/13_conformal.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
