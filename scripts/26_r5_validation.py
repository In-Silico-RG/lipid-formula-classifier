#!/usr/bin/env python
"""
26_r5_validation.py — Validation of R5 (R2 + 23 variables derived from the formula) following the accuracy-improvement protocol.

Same premise: formula only. Same split (80/20 by formula), same 5 seeds, same models as 05/11/12.
Compares R2 and R5 on:
  1. Task B (detection), internal split: RF and XGBoost.
  2. Laboratory (GNPS/PNNL/HCE): category (strict/permissive, balanced RF, NO LEAKAGE) and detection sensitivity (XGBoost).
  3. MassBank: category (balanced RF, NO LEAKAGE) and detection (sensitivity and specificity).
  4. Task C: per-class sensitivity (balanced RF) to see WHERE R5 helps.
Output: results/26_r5_validation.log and .csv
"""
import sys, importlib.util, warnings
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
sys.path.insert(0, str(BASE / "scripts"))
from common import cargar_lmsd, hill, parsear  # noqa: E402
def cargar(nombre, archivo):
    sp = importlib.util.spec_from_file_location(nombre, BASE / "scripts" / archivo); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
rep = cargar("rep", "04_representations.py"); v24 = cargar("v24", "24_extra_variables.py"); v11 = cargar("v11", "11_external_validation.py")
construir = rep.construir; SEM = [0, 1, 2, 3, 4]; N_ARB = 500
z = np.load(DATOS / "representations.npz", allow_pickle=True)
COLS = list(z["R2_COLS"]); F_all, R2_all, et_all = z["formula"], z["R2"], z["label"]
def a_r5(X2): return np.hstack([X2, v24.extra(X2, COLS)])
R5_all = a_r5(R2_all); yb = (et_all == "lipid").astype(int)
def r2_de(fs):
    X, ok = [], []
    for f in fs:
        r = construir(f)
        if r is not None: X.append(r[1]); ok.append(f)
    return np.array(X, dtype=np.float32), ok
log_f = open(RES / "26_r5_validation.log", "w")
def log(m=""): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
filas = []
def reg(**k): filas.append(k)
def rf(s): return RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=s, class_weight="balanced")
def xg(s): return xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=s)

log("=" * 84); log("VALIDATION OF R5 (variables derived only from the formula) — R2 versus R5"); log("=" * 84)

# ------------------------------------------------------------------ 1. Internal task B
log("\n[1] TASK B — detection, 80/20 grouped split (93.988 formulas), 5 seeds")
for nom, fab in (("RF", lambda s: RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=s)), ("XGB", xg)):
    res = {}
    for rn, X in (("R2", R2_all), ("R5", R5_all)):
        a = []
        for s in SEM:
            rng = np.random.default_rng(s); perm = rng.permutation(len(yb)); c = int(.8 * len(perm)); itr, ite = perm[:c], perm[c:]
            m = fab(s); m.fit(X[itr], yb[itr]); p = m.predict(X[ite])
            a.append((f1_score(yb[ite], p, average="macro") * 100, balanced_accuracy_score(yb[ite], p) * 100, (p == yb[ite]).mean() * 100, recall_score(yb[ite], p) * 100))
        res[rn] = np.array(a)
        mu, sd = res[rn].mean(0), res[rn].std(0, ddof=1)
        log(f"  {nom:3s} {rn}  F1 macro {mu[0]:.2f} ± {sd[0]:.2f}  bal.acc {mu[1]:.2f}  accuracy {mu[2]:.2f}  lipid sens. {mu[3]:.2f}")
        reg(block="B internal", model=nom, rep=rn, f1=mu[0], sd_f1=sd[0], bal_acc=mu[1], accuracy=mu[2], sens=mu[3])
    d = res["R5"][:, 0] - res["R2"][:, 0]
    log(f"  Δ F1 macro R5−R2 with {nom}: {d.mean():+.2f} (per seed: {', '.join(f'{x:+.2f}' for x in d)}) · improves in {int((d>0).sum())} of 5")

# ------------------------------------------------------------------ catalog
d_ = cargar_lmsd(DATOS); hs = []
for x in d_["FORMULA"]:
    e, mo = parsear(x); hs.append(None if mo else hill(e))
d_["hill"] = hs; d_ = d_[d_["hill"].notna()]
may_cat = d_.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0]).to_dict()
todas_cat = d_.groupby("hill")["CATEGORY"].agg(lambda s: set(s)).to_dict()
cat_f = list(may_cat); Xc2, fc = r2_de(cat_f); Xc5 = a_r5(Xc2); yc_ = np.array([may_cat[f] for f in fc])
clases = sorted(set(yc_)); cod = {c: i for i, c in enumerate(clases)}

# ------------------------------------------------------------------ 2. laboratory
log("\n[2] LABORATORY (GNPS/PNNL/HCE) — NO LEAKAGE")
ext = pd.read_csv(DATOS / "spectra" / "formulas_experimental.csv").fillna("")
verdad = {}
for f, ab in zip(ext.formula, ext.abrev):
    if ab:
        cat = v11.ABREV.get(ab)
        if cat is None: continue
        verdad.setdefault(f, set()).add(cat)
    else: verdad.setdefault(f, set())
formulas_ext = sorted(verdad)
for f in formulas_ext:
    if not verdad[f] and f in may_cat: verdad[f] = {may_cat[f]}
con_v = [f for f in formulas_ext if verdad[f]]
Xe2, fe = r2_de(con_v); Xe5 = a_r5(Xe2); sets = [verdad[f] for f in fe]; ext_set = set(formulas_ext)
en_ext = np.array([f in ext_set for f in fc])
def evalA_lab(Xc, Xe, nombre):
    a = []
    for s in SEM:
        m = rf(s); m.fit(Xc[~en_ext], [cod[v] for v in yc_[~en_ext]]); p = [clases[i] for i in m.predict(Xe)]
        est = np.mean([pi in s_ for pi, s_ in zip(p, sets)]) * 100
        per = np.mean([pi in (s_ | todas_cat.get(f, set())) for pi, s_, f in zip(p, sets, fe)]) * 100
        a.append((est, per))
    a = np.array(a); log(f"  {nombre}: strict {a[:,0].mean():.2f} ± {a[:,0].std(ddof=1):.2f} · permissive {a[:,1].mean():.2f} ± {a[:,1].std(ddof=1):.2f}")
    reg(block="Lab A", rep=nombre, strict=a[:,0].mean(), sd=a[:,0].std(ddof=1), permissive=a[:,1].mean()); return a
a2 = evalA_lab(Xc2, Xe2, "R2"); a5 = evalA_lab(Xc5, Xe5, "R5")
log(f"  Δ strict R5−R2: {(a5[:,0]-a2[:,0]).mean():+.2f} · improves in {int((a5[:,0]>a2[:,0]).sum())} of 5 seeds")
Xe_all2, _ = r2_de(formulas_ext); Xe_all5 = a_r5(Xe_all2)
en_extB = np.array([f in ext_set for f in F_all])
for rn, Xtr_all, Xe in (("R2", R2_all, Xe_all2), ("R5", R5_all, Xe_all5)):
    s_ = [xg(s).fit(Xtr_all[~en_extB], yb[~en_extB]).predict(Xe).mean() * 100 for s in SEM]
    log(f"  detection (sensitivity, 867 formulas) {rn}: {np.mean(s_):.2f} ± {np.std(s_, ddof=1):.2f}")
    reg(block="Lab B", rep=rn, sens=np.mean(s_), sd=np.std(s_, ddof=1))

# ------------------------------------------------------------------ 3. MassBank
log("\n[3] MASSBANK — NO LEAKAGE")
mb = pd.read_csv(DATOS / "spectra" / "formulas_massbank.csv")
lip = sorted(mb.loc[mb.in_lipidmaps, "formula"]); neg = sorted(mb.loc[mb.is_curated_negative, "formula"]); mb_set = set(lip) | set(neg)
Xl2, fl = r2_de([f for f in lip if f in may_cat]); Xl5 = a_r5(Xl2); yl = np.array([may_cat[f] for f in fl])
en_mb = np.array([f in mb_set for f in fc])
def evalA_mb(Xc, Xe, nombre):
    a = []
    for s in SEM:
        m = rf(s); m.fit(Xc[~en_mb], [cod[v] for v in yc_[~en_mb]]); p = np.array([clases[i] for i in m.predict(Xe)])
        est = (p == yl).mean() * 100; per = np.mean([pi in todas_cat.get(f, {yi}) for pi, f, yi in zip(p, fl, yl)]) * 100
        a.append((est, per, f1_score(yl, p, average="macro", zero_division=0) * 100, recall_score(yl == "Prenol Lipids", p == "Prenol Lipids") * 100,
                  recall_score(yl == "Sterol Lipids", p == "Sterol Lipids") * 100))
    a = np.array(a); log(f"  {nombre}: strict {a[:,0].mean():.2f} ± {a[:,0].std(ddof=1):.2f} · permissive {a[:,1].mean():.2f} · F1 macro {a[:,2].mean():.2f} · prenol {a[:,3].mean():.1f} · sterols {a[:,4].mean():.1f}")
    reg(block="MB A", rep=nombre, strict=a[:,0].mean(), sd=a[:,0].std(ddof=1), permissive=a[:,1].mean(), f1=a[:,2].mean(), prenol=a[:,3].mean(), sterols=a[:,4].mean()); return a
b2 = evalA_mb(Xc2, Xl2, "R2"); b5 = evalA_mb(Xc5, Xl5, "R5")
log(f"  Δ strict R5−R2: {(b5[:,0]-b2[:,0]).mean():+.2f} · improves in {int((b5[:,0]>b2[:,0]).sum())} of 5 seeds")
Xn2, fn = r2_de(neg); Xn5 = a_r5(Xn2); Xl_all2, f_l = r2_de(lip); Xl_all5 = a_r5(Xl_all2)
en_mbB = np.array([f in mb_set for f in F_all])
for rn, Xtr_all, XL, XN in (("R2", R2_all, Xl_all2, Xn2), ("R5", R5_all, Xl_all5, Xn5)):
    r = []
    for s in SEM:
        m = xg(s).fit(Xtr_all[~en_mbB], yb[~en_mbB]); r.append((m.predict(XL).mean() * 100, (1 - m.predict(XN)).mean() * 100))
    r = np.array(r); log(f"  detection {rn}: sensitivity {r[:,0].mean():.2f} ± {r[:,0].std(ddof=1):.2f} · specificity {r[:,1].mean():.2f} ± {r[:,1].std(ddof=1):.2f}")
    reg(block="MB B", rep=rn, sens=r[:,0].mean(), spec=r[:,1].mean())

# ------------------------------------------------------------------ 4. task C per class
log("\n[4] TASK C — per-class sensitivity, balanced RF, 5 seeds (classes with ≥ 30 formulas)")
d2 = d_.groupby("hill")["MAIN_CLASS"].agg(lambda s: s.value_counts().index[0]); sel = np.array([f in d2.index for f in F_all])
FF = F_all[sel]; yC = np.array([d2[f] for f in FF]); ct = Counter(yC); ok = np.array([ct[c] >= 10 for c in yC])
FF, yC = FF[ok], yC[ok]; XC2 = R2_all[sel][ok]; XC5 = R5_all[sel][ok]
cl = sorted(set(yC)); cd = {c: i for i, c in enumerate(cl)}; yi = np.array([cd[v] for v in yC])
sens = {rn: {c: [] for c in cl} for rn in ("R2", "R5")}
for s in SEM:
    rng = np.random.default_rng(s); perm = rng.permutation(len(yi)); c_ = int(.8 * len(perm)); itr, ite = perm[:c_], perm[c_:]
    for rn, X in (("R2", XC2), ("R5", XC5)):
        ut = np.unique(yi[itr]); mp = {v: i for i, v in enumerate(ut)}
        m = rf(s); m.fit(X[itr], [mp[v] for v in yi[itr]]); p = ut[m.predict(X[ite])]
        for c in cl:
            msk = yi[ite] == cd[c]
            if msk.sum(): sens[rn][c].append((p[msk] == cd[c]).mean() * 100)
tab = pd.DataFrame({"n": [ct[c] for c in cl], "R2": [np.mean(sens["R2"][c]) for c in cl], "R5": [np.mean(sens["R5"][c]) for c in cl]}, index=cl)
tab["delta"] = tab.R5 - tab.R2; g = tab[tab.n >= 30].sort_values("delta", ascending=False)
log("  largest improvements:"); [log(f"    {i:42s} n={int(r.n):4d}  R2 {r.R2:5.1f} → R5 {r.R5:5.1f}  ({r.delta:+.1f})") for i, r in g.head(8).iterrows()]
log("  largest losses:"); [log(f"    {i:42s} n={int(r.n):4d}  R2 {r.R2:5.1f} → R5 {r.R5:5.1f}  ({r.delta:+.1f})") for i, r in g.tail(5).iterrows()]
for c in ("Quinones and hydroquinones [PR02]", "Aromatic polyketides [PK13]", "Bile acids and derivatives [ST04]", "Other Fatty Acyls [FA00]", "Hopanoids [PR04]"):
    if c in tab.index: log(f"  weak class {c:38s} R2 {tab.loc[c,'R2']:5.1f} → R5 {tab.loc[c,'R5']:5.1f}")
tab.to_csv(RES / "26_r5_per_class.csv"); pd.DataFrame(filas).to_csv(RES / "26_r5_validation.csv", index=False)
log("\nSaved: results/26_r5_validation.csv, results/26_r5_per_class.csv")
