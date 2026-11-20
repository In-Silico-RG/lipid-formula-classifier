#!/usr/bin/env python
"""
28_sensitivity.py — sensitivity analysis of the decisions in draft v2 (list `manuscript/revista/correcciones.pdf`, section 2).

Same premise (formula only) and same protocol (80/20 by formula). Each block changes ONE decision:
  1. Threshold of formulas per class in task C: 5, 10 (draft), 20, 30.
  2. Number of seeds: 5 (draft) versus 20, tasks A, B and C.
  3. Nearest-neighbour variables: the 11 of the draft, the 13 of R1 (Mitchell) and the 22 of R2.
  4. Mass shift and tolerance of the +21 test: +7, +14, +21, +28, +42 with 3 ppm; +21 with 1 and 5 ppm.
Output: results/28_sensitivity.log and 28_sensitivity_*.csv   (blocks can be requested: python 28_sensitivity.py 1 3)
"""
import sys, importlib.util, warnings
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
sys.path.insert(0, str(BASE / "scripts"))
from common import cargar_lmsd, hill, parsear  # noqa: E402

N_ARB = 500
bloques = set(sys.argv[1:]) or {"1", "2", "3", "4"}
fh = open(RES / "28_sensitivity.log", "a")
def log(m=""): print(m, flush=True); fh.write(m + "\n"); fh.flush()

z = np.load(DATOS / "representations.npz", allow_pickle=True)
F, R1, R2, et = z["formula"], z["R1"], z["R2"], z["label"]
C2 = list(z["R2_COLS"]); C1 = list(z["R1_COLS"])
d = cargar_lmsd(DATOS); hs = []
for x in d["FORMULA"]:
    e, mo = parsear(x); hs.append(None if mo else hill(e))
d["hill"] = hs; d = d[d["hill"].notna()]
may_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0])
may_cls = d.groupby("hill")["MAIN_CLASS"].agg(lambda s: s.value_counts().index[0])
selA = np.array([f in may_cat.index for f in F]); yA = np.array([may_cat[f] for f in F[selA]])
selC = np.array([f in may_cls.index for f in F]); yC = np.array([may_cls[f] for f in F[selC]])
yB = (et == "lipid").astype(int)

def enc(y):
    cl = np.unique(y); cod = {c: i for i, c in enumerate(cl)}; return np.array([cod[v] for v in y]), cl
def ajusta(nom, X, y, s):
    if nom == "RF-bal": m = RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=s, class_weight="balanced"); m.fit(X, y)
    elif nom == "XGB": m = xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=s); m.fit(X, y)
    else: m = xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=s); m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return m
def partir(n, s):
    perm = np.random.default_rng(s).permutation(n); c = int(.8 * n); return perm[:c], perm[c:]
def resumen(v): v = np.array(v); return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"

# ------------------------------------------------------------------ 1. threshold
if "1" in bloques:
    log("\n" + "=" * 84); log("[1] THRESHOLD OF FORMULAS PER CLASS (task C), 5 seeds"); log("=" * 84)
    vc = pd.Series(yC).value_counts(); comunes = set(vc[vc >= 30].index)
    log(f"  classes with ≥ 30 formulas (comparable across thresholds): {len(comunes)}")
    filas = []
    for k in (5, 10, 20, 30):
        keep = np.array([vc[v] >= k for v in yC]); X, y = R2[selC][keep], yC[keep]; yi, cl = enc(y)
        ic = [i for i, c in enumerate(cl) if c in comunes]
        for nom in ("RF-bal", "XGB-bal"):
            r = []
            for s in range(5):
                itr, ite = partir(len(yi), s); m = ajusta(nom, X[itr], yi[itr], s); p = m.predict(X[ite])
                r.append(((p == yi[ite]).mean() * 100, f1_score(yi[ite], p, average="macro", zero_division=0) * 100,
                          f1_score(yi[ite], p, labels=ic, average="macro", zero_division=0) * 100, int((~np.isin(yi[ite], range(len(cl)))).sum())))
            r = np.array(r)
            log(f"  k={k:2d}  {len(cl):2d} classes {len(yi):5d} formulas  {nom:7s} accuracy {resumen(r[:,0])}  F1 macro {resumen(r[:,1])}  F1 macro (only classes ≥30) {resumen(r[:,2])}")
            filas.append({"threshold": k, "classes": len(cl), "formulas": len(yi), "model": nom, "accuracy": r[:, 0].mean(), "f1_macro": r[:, 1].mean(),
                          "f1_common_classes": r[:, 2].mean(), "sd_f1_common": r[:, 2].std(ddof=1), "min_in_test_per_class_approx": round(k * .2, 1)})
    pd.DataFrame(filas).to_csv(RES / "28_sensitivity_threshold.csv", index=False)

# ------------------------------------------------------------------ 2. seeds
if "2" in bloques:
    log("\n" + "=" * 84); log("[2] NUMBER OF SEEDS: the first 5 versus 20"); log("=" * 84)
    vc = pd.Series(yC).value_counts(); keepC = np.array([vc[v] >= 10 for v in yC])
    tareas = {"A family RF-bal": ("RF-bal", R2[selA], enc(yA)[0]), "A family XGB-bal": ("XGB-bal", R2[selA], enc(yA)[0]),
              "B detection XGB": ("XGB", R2, yB), "C class XGB-bal": ("XGB-bal", R2[selC][keepC], enc(yC[keepC])[0]),
              "C class RF-bal": ("RF-bal", R2[selC][keepC], enc(yC[keepC])[0])}
    filas = []
    for nombre, (nom, X, y) in tareas.items():
        f1 = []
        for s in range(20):
            itr, ite = partir(len(y), s); m = ajusta(nom, X[itr], y[itr], s); p = m.predict(X[ite])
            f1.append(f1_score(y[ite], p, average="macro", zero_division=0) * 100)
        f1 = np.array(f1)
        log(f"  {nombre:18s} 5 seeds {resumen(f1[:5])}   20 seeds {resumen(f1)}   (95 % interval of the mean, 20 seeds: ±{1.96*f1.std(ddof=1)/np.sqrt(20):.2f})")
        filas.append({"task": nombre, "f1_5": f1[:5].mean(), "sd_5": f1[:5].std(ddof=1), "f1_20": f1.mean(), "sd_20": f1.std(ddof=1)})
        # the per-seed difference against the other model of the same task is computed in block 3
    pd.DataFrame(filas).to_csv(RES / "28_sensitivity_seeds.csv", index=False)

# ------------------------------------------------------------------ 3. nearest-neighbour variables
if "3" in bloques:
    log("\n" + "=" * 84); log("[3] NEAREST-NEIGHBOUR VARIABLES (tasks A and B), 5 seeds, best of k=1 and k=5"); log("=" * 84)
    c11 = ["C", "H", "O", "N", "P", "S", "OTROS", "CARGA", "DBE", "O_C", "H_C"]
    X11 = R2[:, [C2.index(c) for c in c11]]
    juegos = {"11 of the draft": X11, "13 of R1 (Mitchell)": R1, "22 of R2": R2}
    filas = []
    for tarea, sel, y in (("A family", selA, enc(yA)[0]), ("B detection", np.ones(len(F), bool), yB)):
        for jn, X in juegos.items():
            Xs = X[sel]; mejor = None
            for k in (1, 5):
                r = []
                for s in range(5):
                    itr, ite = partir(len(y), s); sc = StandardScaler().fit(Xs[itr])
                    p = KNeighborsClassifier(n_neighbors=k, n_jobs=12).fit(sc.transform(Xs[itr]), y[itr]).predict(sc.transform(Xs[ite]))
                    r.append(f1_score(y[ite], p, average="macro", zero_division=0) * 100)
                if mejor is None or np.mean(r) > np.mean(mejor[1]): mejor = (k, r)
            log(f"  {tarea:12s} {jn:20s} k={mejor[0]}  F1 macro {resumen(mejor[1])}")
            filas.append({"task": tarea, "variables": jn, "k": mejor[0], "f1_macro": np.mean(mejor[1]), "sd": np.std(mejor[1], ddof=1)})
    pd.DataFrame(filas).to_csv(RES / "28_sensitivity_neighbour.csv", index=False)

# ------------------------------------------------------------------ 4. shift and tolerance
if "4" in bloques:
    log("\n" + "=" * 84); log("[4] MASS SHIFT AND TOLERANCE (+21 test), XGBoost on R2, 5 seeds"); log("=" * 84)
    sp = importlib.util.spec_from_file_location("adv", BASE / "scripts" / "15_phase6_adversarial.py"); adv = importlib.util.module_from_spec(sp); sp.loader.exec_module(adv)
    maestro = pd.read_parquet(DATOS / "master.parquet"); conocidas = set(maestro.formula)
    mb = pd.read_csv(DATOS / "spectra" / "formulas_massbank.csv"); ext = pd.read_csv(DATOS / "spectra" / "formulas_experimental.csv").fillna("")
    conocidas |= set(mb.formula) | set(ext.formula)
    n4a = set(pd.read_csv(DATOS / "n4_adversarial.csv").query("layer=='N4a'").formula); conocidas |= n4a
    lip = [f for f, e in zip(F, et) if e == "lipid"]
    objetivos = np.random.default_rng(0).choice(lip, size=min(adv.N_OBJETIVOS, len(lip)), replace=False)
    ext_set = set(mb.formula) | set(ext.formula); interno = np.array([f not in ext_set for f in F])
    def r2(fs):
        X, ok = [], []
        for f in fs:
            r = adv.construir(f)
            if r is not None: X.append(r[1]); ok.append(f)
        return np.array(X, dtype=np.float32), ok
    filas = []
    for desp, ppm in ((7, 3.0), (14, 3.0), (21, 3.0), (28, 3.0), (42, 3.0), (21, 1.0), (21, 5.0)):
        nb, padre, sin = [], {}, 0
        for f in objetivos:
            e0, _ = parsear(f); m0 = adv.masa_de({k: v for k, v in e0.items() if k in adv.MASA})
            cand = adv.enumerar_cerca(m0 + desp, ppm, conocidas)
            if not cand: sin += 1; continue
            for c_ in cand[:adv.MAX_POR_OBJETIVO]: nb.append(c_); padre.setdefault(c_, f)
        nb = sorted(set(nb)); Xb, fb = r2(nb); gb = np.array([padre[f] for f in fb])
        rat, acc, sens = [], [], []
        for esc in ("base", "con"):
            R, S, Q = [], [], []
            for s in range(5):
                rng = np.random.default_rng(s); idx = np.where(interno)[0]; perm = rng.permutation(idx); c = int(.8 * len(perm)); itr, ite = perm[:c], perm[c:]
                padres = np.array(sorted(set(gb))); rng.shuffle(padres); cc = int(.8 * len(padres)); b_tr = np.where(np.isin(gb, padres[:cc]))[0]; b_te = np.where(~np.isin(gb, padres[:cc]))[0]
                Xtr, ytr = R2[itr], yB[itr]
                if esc == "con": Xtr = np.vstack([Xtr, Xb[b_tr]]); ytr = np.concatenate([ytr, np.zeros(len(b_tr), int)])
                m = xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=0).fit(Xtr, ytr)
                p = m.predict(R2[ite]); sens_i = p[yB[ite] == 1].mean() * 100; fr = m.predict(Xb[b_te]).mean() * 100
                R.append(fr / sens_i); S.append(fr); Q.append(sens_i)
            rat.append(np.mean(R)); acc.append(np.mean(S)); sens.append(np.mean(Q))
        log(f"  shift +{desp:2d}  ±{ppm:.0f} ppm: {len(nb):5d} formulas (no candidate {sin}); baseline model: accepted {acc[0]:5.2f} %, ratio {rat[0]:.3f}, lipids found {sens[0]:.2f} %"
            f" | with shifted ones as negatives: accepted {acc[1]:5.2f} %, ratio {rat[1]:.3f}, lipids found {sens[1]:.2f} %")
        filas.append({"shift": desp, "ppm": ppm, "formulas": len(nb), "no_candidate": sin, "accepted_base": acc[0], "ratio_base": rat[0], "sens_base": sens[0],
                      "accepted_with": acc[1], "ratio_with": rat[1], "sens_con": sens[1]})
    pd.DataFrame(filas).to_csv(RES / "28_sensitivity_shift.csv", index=False)
log("\ndone"); fh.close()
