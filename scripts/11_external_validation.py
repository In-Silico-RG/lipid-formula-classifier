#!/usr/bin/env python
"""
11_external_validation.py — The classifier on MEASURED formulas (E4).

Up to this point all figures came from the LIPID MAPS catalog. Here the evaluation is on the
867 formulas extracted from public experimental spectra (GNPS: PNNL positive and
negative, HCE cell lysate), stored in data/spectra/formulas_experimental.csv.

Two things make this set different from any internal split:

  1. Its DISTRIBUTION is that of what is measured, not that of the catalog (53-83 % of
     glycerophospholipids versus 19 % of the catalog; no sterols or prenols).
  2. For the PNNL formulas the true label comes from the EXPERIMENTAL
     ANNOTATION (the abbreviation in the name: PC, TG, SM...), not from LIPID MAPS.
     It is a ground truth independent of the catalog used for training.

PROTOCOL. Two scenarios, both reported:

  NO LEAKAGE    the 867 formulas are EXCLUDED from training. It measures generalization to
                real molecules never seen. This is the figure that counts.
  WITH LEAKAGE  training uses the whole catalog, which already contains 780 of the 867. It is the
                usual scenario in the literature, and here it serves to show the gap
                on real, not simulated, data.

Tasks:
  A  category — balanced RF on R2 (Phase 3 winner). Truth: experimental
     annotation (PNNL) or LIPID MAPS (HCE, which has no abbreviation). Strict and
     permissive (admissible = annotation ∪ LIPID MAPS categories for that formula).
  B  detection — XGBoost on R2 (Phase 3 winner). The external set is
     all positive, so what is measured is the SENSITIVITY on real lipids.
     It is compared with the internal sensitivity (grouped split) of the same model.

Output: results/11_external_validation.log + .csv
"""
import importlib.util
import sys
import warnings
from collections import Counter, defaultdict
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

# 04_representations.py starts with a digit: it is imported by path
_spec = importlib.util.spec_from_file_location("rep", BASE / "scripts" / "04_representations.py")
_rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rep)
construir = _rep.construir

SEMILLAS = [0, 1, 2, 3, 4]
N_ARB = 500

# Experimental annotation abbreviation → LIPID MAPS category.
# Independent of the catalog: it is assigned by whoever measured the spectrum.
ABREV = {
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


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "11_external_validation.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    log("=" * 84)
    log("EXTERNAL VALIDATION — the classifier on MEASURED formulas (GNPS / PNNL / HCE)")
    log("=" * 84)

    # ------------------------------------------------------------ external set
    ext = pd.read_csv(DATOS / "spectra" / "formulas_experimental.csv").fillna("")
    n_esp = len(ext)
    verdad, fuente, sin_mapa = {}, {}, Counter()
    for f, ab, lib in zip(ext.formula, ext.abrev, ext.libreria):
        if ab:
            cat = ABREV.get(ab)
            if cat is None:
                sin_mapa[ab] += 1
                continue
            # a formula may have several annotations; all are kept
            verdad.setdefault(f, set()).add(cat)
            fuente[f] = "experimental annotation"
        else:
            verdad.setdefault(f, set())
            fuente.setdefault(f, "LIPID MAPS (no annotation)")
    formulas_ext = sorted(verdad)
    log(f"\n  spectra/rows: {n_esp:,}  →  unique formulas: {len(formulas_ext):,}")
    if sin_mapa:
        log(f"  abbreviations without a category map (excluded from task A): {dict(sin_mapa)}")

    # ------------------------------------------------------------ catalog
    d = cargar_lmsd(DATOS)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna()]
    may_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0]).to_dict()
    todas_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: set(s)).to_dict()

    # HCE without abbreviation: truth = LIPID MAPS
    for f in formulas_ext:
        if not verdad[f] and f in may_cat:
            verdad[f] = {may_cat[f]}
    con_verdad = [f for f in formulas_ext if verdad[f]]
    log(f"  with category label: {len(con_verdad):,}  "
        f"({sum(fuente[f].startswith('anot') for f in con_verdad):,} by experimental annotation, "
        f"{sum(not fuente[f].startswith('anot') for f in con_verdad):,} by LIPID MAPS)")

    # annotation vs catalog consistency, over the overlap
    solape = [f for f in con_verdad if f in may_cat and fuente[f].startswith("anot")]
    desac = [f for f in solape if may_cat[f] not in verdad[f]]
    log(f"\n  CONSISTENCY experimental annotation vs LIPID MAPS (over {len(solape):,} formulas in both):")
    log(f"    the catalog majority category does NOT match the annotation: {len(desac):,} "
        f"({len(desac)/max(len(solape),1)*100:.1f} %)")
    for f in desac[:6]:
        log(f"      {f:16s} catalog={may_cat[f]:22s} annotation={'|'.join(sorted(verdad[f]))}")

    # ------------------------------------------------------------ R2 representation
    def r2_de(fs):
        X, ok = [], []
        for f in fs:
            r = construir(f)
            if r is None:
                continue
            X.append(r[1])
            ok.append(f)
        return np.array(X, dtype=np.float32), ok

    X_ext, f_ext = r2_de(con_verdad)
    y_ext_sets = [verdad[f] for f in f_ext]
    ext_set = set(formulas_ext)
    log(f"  R2 computed for {len(f_ext):,} external formulas")
    dist = Counter(c for s in y_ext_sets for c in s)
    log("  external distribution: " + ", ".join(f"{k} {v}" for k, v in dist.most_common()))

    filas = []

    # ====================================================================== TASK A
    log("\n" + "=" * 84)
    log("TASK A — Category · balanced RF on R2 · truth = experimental annotation")
    log("=" * 84)
    cat_formulas = [f for f in may_cat]
    X_cat_all, f_cat_all = r2_de(cat_formulas)
    y_cat_all = np.array([may_cat[f] for f in f_cat_all])
    en_ext = np.array([f in ext_set for f in f_cat_all])
    log(f"  catalog: {len(f_cat_all):,} formulas; of these {int(en_ext.sum()):,} are in the external set")

    clases = sorted(set(y_cat_all))
    cod = {c: i for i, c in enumerate(clases)}

    def eval_A(mask_train, nombre):
        acc = []
        for s in SEMILLAS:
            m = RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=s,
                                       class_weight="balanced")
            m.fit(X_cat_all[mask_train], [cod[v] for v in y_cat_all[mask_train]])
            p = [clases[i] for i in m.predict(X_ext)]
            est = np.array([pi in s_ for pi, s_ in zip(p, y_ext_sets)])
            adm = [s_ | todas_cat.get(f, set()) for s_, f in zip(y_ext_sets, f_ext)]
            per = np.array([pi in a for pi, a in zip(p, adm)])
            # sensitivity per category (truth = first annotation if there are several)
            y1 = [sorted(s_)[0] for s_ in y_ext_sets]
            rec = {c: recall_score([v == c for v in y1], [pi == c for pi in p], zero_division=0) * 100
                   for c in sorted(set(y1))}
            acc.append({"strict": est.mean() * 100, "permissive": per.mean() * 100, **rec})
        t = pd.DataFrame(acc)
        m_ = t.mean()
        log(f"\n  {nombre}")
        log(f"    STRICT accuracy     (pred ∈ annotation):             {m_['strict']:6.2f} %  (± {t['strict'].std():.2f})")
        log(f"    PERMISSIVE accuracy (pred ∈ annotation ∪ catalog):   {m_['permissive']:6.2f} %  (± {t['permissive'].std():.2f})")
        log("    sensitivity per category: " +
            ", ".join(f"{c} {m_[c]:.1f} %" for c in t.columns if c not in ("strict", "permissive")))
        filas.append({"task": "A", "scenario": nombre, **{k: round(float(v), 2) for k, v in m_.items()}})

    eval_A(~en_ext, "NO LEAKAGE — trained excluding the external formulas")
    eval_A(np.ones(len(f_cat_all), bool), "WITH LEAKAGE — trained on the whole catalogue (780 already seen)")

    # ====================================================================== TASK B
    log("\n" + "=" * 84)
    log("TASK B — Detection · XGBoost on R2 · the external set is all lipid → SENSITIVITY")
    log("=" * 84)
    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    F_all, R2_all, et_all = z["formula"], z["R2"], z["label"]
    en_ext_B = np.array([f in ext_set for f in F_all])
    log(f"  master table: {len(F_all):,} formulas; {int(en_ext_B.sum()):,} are in the external set (excluded in the no-leakage scenario)")
    X_ext_B, f_ext_B = r2_de(formulas_ext)          # the 867, with or without category
    yb = (et_all == "lipid").astype(int)

    def eval_B(mask_train, nombre, interna=False):
        acc = []
        for s in SEMILLAS:
            idx = np.where(mask_train)[0]
            if interna:   # internal sensitivity: grouped 80/20 split within the training set
                rng = np.random.default_rng(s)
                perm = rng.permutation(idx)
                c = int(0.8 * len(perm))
                itr, ite = perm[:c], perm[c:]
            else:
                itr, ite = idx, None
            m = xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist",
                                  verbosity=0, random_state=s)
            m.fit(R2_all[itr], yb[itr])
            if interna:
                p = m.predict(R2_all[ite])
                acc.append({"sens_lipid": recall_score(yb[ite], p) * 100,
                            "bal_acc": balanced_accuracy_score(yb[ite], p) * 100})
            else:
                p = m.predict(X_ext_B)
                acc.append({"sens_lipid": p.mean() * 100})
        t = pd.DataFrame(acc)
        m_ = t.mean()
        log(f"\n  {nombre}")
        log(f"    sensitivity on lipids: {m_['sens_lipid']:6.2f} %  (± {t['sens_lipid'].std():.2f})"
            + (f"   · balanced acc {m_['bal_acc']:.2f} %" if interna else ""))
        filas.append({"task": "B", "scenario": nombre, **{k: round(float(v), 2) for k, v in m_.items()}})
        return m_["sens_lipid"]

    eval_B(~en_ext_B, "INTERNAL — grouped 80/20 split, reference", interna=True)
    s_sin = eval_B(~en_ext_B, "EXTERNAL NO LEAKAGE — trained excluding the 867")
    s_con = eval_B(np.ones(len(F_all), bool), "EXTERNAL WITH LEAKAGE — trained on everything (780 already seen)")

    # where does it fail? by library and by annotated class
    m = xgb.XGBClassifier(n_estimators=N_ARB, n_jobs=12, tree_method="hist", verbosity=0, random_state=0)
    m.fit(R2_all[~en_ext_B], yb[~en_ext_B])
    p = m.predict(X_ext_B)
    fall = set(f for f, pi in zip(f_ext_B, p) if pi == 0)
    log(f"\n  real formulas NOT detected as lipid (seed 0, no leakage): {len(fall):,} of {len(f_ext_B):,}")
    por_lib = ext.drop_duplicates("formula").set_index("formula")
    lib_fall = Counter(por_lib.loc[f, "libreria"] for f in fall if f in por_lib.index)
    lib_tot = Counter(por_lib["libreria"])
    log("    by library: " + ", ".join(f"{k} {lib_fall[k]}/{lib_tot[k]}" for k in lib_tot))
    ab_fall = Counter(por_lib.loc[f, "abrev"] for f in fall if f in por_lib.index)
    log("    by annotated class: " + ", ".join(f"{k or '(HCE)'} {v}" for k, v in ab_fall.most_common(8)))
    log("    examples: " + ", ".join(sorted(fall)[:8]))

    pd.DataFrame(filas).to_csv(RES / "11_external_validation.csv", index=False)
    log("\n" + "=" * 84)
    log("Saved: results/11_external_validation.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
