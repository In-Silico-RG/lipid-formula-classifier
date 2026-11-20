#!/usr/bin/env python
"""
25_ms2_contribution.py — How much does adding MS/MS information (and adduct, ionization mode) to the formula help?

Data: data/spectra/ms2_lipids.pkl (script 23): MoNA MS/MS spectra whose molecule is in LIPID MAPS; each spectrum carries the
TRUE category and class of its molecule (not the majority one of the formula), so that the isomers of the same formula
keep their own class.
Three models, the same split (by FORMULA: no test formula appears in training), 5 seeds, XGBoost:
  F  formula only:       R5 (R2 + 23 variables derived from the formula; script 24)
  FI formula + context:  F + ionization mode + adduct type + precursor m/z
  FM formula + MS/MS:    FI + fragments (1 Da bins by m/z) and neutral losses (1 Da bins of precursor m/z − fragment m/z),
                         keeping only bins present in ≥ 100 TRAINING spectra.
Accuracy per spectrum is reported, overall and on spectra with an AMBIGUOUS formula (the formula has more than one category/class in LIPID MAPS).
Output: results/25_ms2_contribution.log y .csv
"""
import sys, pickle, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
import scipy.sparse as sp
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from common import cargar_lmsd, hill, parsear  # noqa: E402
import importlib.util
spec = importlib.util.spec_from_file_location("v24", BASE / "scripts" / "24_extra_variables.py"); v24 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v24)
SEMILLAS = [0, 1, 2, 3, 4]
df = pickle.load(open(BASE / "data" / "spectra" / "ms2_lipids.pkl", "rb"))
z = np.load(BASE / "data" / "representations.npz", allow_pickle=True)
cols = list(z["R2_COLS"]); idx = {f: i for i, f in enumerate(z["formula"])}
hh = []
for f in df.formula:
    e, mo = parsear(f); hh.append(None if mo else hill(e))
df["hill"] = hh
df = df[df.hill.isin(idx)].reset_index(drop=True)
R2 = z["R2"][[idx[f] for f in df.hill]]
XF = np.hstack([R2, v24.extra(R2, cols)])
# ambiguity per formula, according to ALL LIPID MAPS molecules
d = cargar_lmsd(BASE / "data"); hs = []
for x in d["FORMULA"]:
    e, mo = parsear(x); hs.append(None if mo else hill(e))
d["hill"] = hs; d = d[d.hill.notna()]
amb_cat = d.groupby("hill")["CATEGORY"].nunique() > 1
amb_cls = d.groupby("hill")["MAIN_CLASS"].nunique() > 1
# context
modo = (df.modo.str.upper().str[0] == "P").astype(float).values
def tipo(a):
    a = a.replace("[", "").replace("]", "").replace("1", "").replace("+", "").replace("-", "")
    return a
ad = df.aducto.map(tipo); top = ad.value_counts().head(8).index
XI = np.column_stack([XF, modo, df.pmz.values] + [(ad == t).astype(float).values for t in top])
# spectra → sparse bin matrices
def bins(row_peaks, pmz):
    fr = np.clip(np.round(row_peaks[:, 0]).astype(int), 0, 1199)
    nl = np.round(pmz - row_peaks[:, 0]).astype(int); ok = (nl >= 0) & (nl < 800)
    return fr, row_peaks[:, 1], nl[ok], row_peaks[ok, 1]
def construir(ix_train, ix_all):
    f_all, n_all = [], []
    for i in ix_all:
        fr, fi, nl, ni = bins(df.picos[i], df.pmz[i]); f_all.append((fr, fi)); n_all.append((nl, ni))
    def mat(lst, ancho):
        r, c, vv = [], [], []
        for k, (b, w) in enumerate(lst): r += [k] * len(b); c += list(b); vv += list(w)
        return sp.csr_matrix((vv, (r, c)), shape=(len(lst), ancho))
    Mf, Mn = mat(f_all, 1200), mat(n_all, 800)
    return Mf, Mn
Mf_all, Mn_all = construir(None, range(len(df)))
def run(tarea, y):
    cl = np.unique(y); cod = {c: i for i, c in enumerate(cl)}; yc = np.array([cod[v] for v in y])
    ambig = (amb_cat if tarea == "category" else amb_cls).reindex(df.hill).fillna(False).values
    formulas_u = df.hill.unique(); res = {}
    for s in SEMILLAS:
        rng = np.random.default_rng(s); perm = rng.permutation(len(formulas_u)); c = int(0.8 * len(perm))
        ftr = set(formulas_u[perm[:c]]); mtr = df.hill.isin(ftr).values; itr, ite = np.where(mtr)[0], np.where(~mtr)[0]
        ut = np.unique(yc[itr]); mp = {a: i for i, a in enumerate(ut)}; ytr = np.array([mp[a] for a in yc[itr]])
        pres_f = np.asarray((Mf_all[itr] > 0).sum(0)).ravel() >= 100; pres_n = np.asarray((Mn_all[itr] > 0).sum(0)).ravel() >= 100
        Xm = sp.hstack([sp.csr_matrix(XI), Mf_all[:, pres_f], Mn_all[:, pres_n]]).tocsr()
        for nom, X in (("F formula only", sp.csr_matrix(XF)), ("FI formula + context", sp.csr_matrix(XI)), ("FM formula + MS/MS", Xm)):
            m = xgb.XGBClassifier(n_estimators=300, n_jobs=10, tree_method="hist", verbosity=0, random_state=0)
            m.fit(X[itr], ytr); p = ut[m.predict(X[ite])]; ok = p == yc[ite]; am = ambig[ite]
            res.setdefault(nom, []).append((ok.mean() * 100, ok[am].mean() * 100 if am.any() else np.nan, ok[~am].mean() * 100,
                                            f1_score(yc[ite], p, average="macro", zero_division=0) * 100, am.mean() * 100, len(ite), Mf_all[:, pres_f].shape[1] + Mn_all[:, pres_n].shape[1]))
    return res
def main():
    log = open(BASE / "results" / "25_ms2_contribution.log", "w"); filas = []
    def p(m=""): print(m, flush=True); log.write(m + "\n"); log.flush()
    p("CONTRIBUTION OF MS/MS, ADDUCT AND IONIZATION MODE TO THE FORMULA CLASSIFIER")
    p(f"spectra: {len(df):,} · molecules {df.ik.nunique():,} · formulas {df.hill.nunique():,} · 80/20 split by formula · {len(SEMILLAS)} seeds\n")
    for tarea, col in (("category", "category"), ("main class", "main_class")):
        y = df[col].values; t0 = time.time(); res = run("category" if tarea == "category" else "main class", y)
        p(f"{tarea.upper()}  ({len(np.unique(y))} classes)   [{time.time()-t0:.0f} s]")
        p(f"  {'model':24s} {'total acc.':>14s} {'ambiguous':>12s} {'unambiguous':>12s} {'F1 macro':>9s}")
        for nom, a in res.items():
            a = np.array(a); mu = np.nanmean(a, 0); sd = np.nanstd(a, 0)
            p(f"  {nom:24s} {mu[0]:6.2f} ± {sd[0]:4.2f}  {mu[1]:6.2f}±{sd[1]:4.2f}  {mu[2]:8.2f}  {mu[3]:8.2f}")
            filas.append(dict(task=tarea, model=nom, accuracy=mu[0], sd=sd[0], ambiguous=mu[1], unambiguous=mu[2], f1_macro=mu[3], pct_ambiguous_test=mu[4], n_bins_msms=mu[6]))
        a0 = np.array(res["F formula only"]); a2 = np.array(res["FM formula + MS/MS"])
        p(f"  Δ (formula + MS/MS) − (formula only): total {np.mean(a2[:,0]-a0[:,0]):+.2f} · ambiguous {np.nanmean(a2[:,1]-a0[:,1]):+.2f} points · improves in {int((a2[:,0]>a0[:,0]).sum())} of {len(a0)} seeds\n")
    pd.DataFrame(filas).to_csv(BASE / "results" / "25_ms2_contribution.csv", index=False)
if __name__ == "__main__":
    main()
