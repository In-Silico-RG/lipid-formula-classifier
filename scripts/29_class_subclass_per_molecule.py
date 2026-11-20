#!/usr/bin/env python
"""
29_class_subclass_per_molecule.py — upper limit and formula-only model PER MOLECULE, at class and subclass level.

Same protocol as 17_structure_oracle.py (which does it for the family): 80/20 split grouped by
formula, seeds 0–4, balanced 500-tree RF on R2 trained with the most frequent label of each
formula, evaluated against the label of EACH test molecule. The oracle (Morgan fingerprint) and the
upper limit on the test molecules of each split are added, separating ambiguous and unambiguous formulas.
The labels are those of data/lmsd_extended.csv (MAIN_CLASS, SUB_CLASS), the same as in the limit computation
(86 classes, 299 subclasses; "-" counts as a label, as there).

Output: results/29_class_subclass_per_molecule.log/.csv
"""
import importlib.util
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"
sys.path.insert(0, str(BASE / "scripts"))
from common import hill, parsear  # noqa: E402

_spec = importlib.util.spec_from_file_location("rep", BASE / "scripts" / "04_representations.py")
_rep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rep)
construir = _rep.construir

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
RDLogger.DisableLog("rdApp.*")

SEMILLAS = [0, 1, 2, 3, 4]
N_ARB = 500
N_BITS = 2048
NIVELES = [("Class", "MAIN_CLASS"), ("Subclass", "SUB_CLASS")]


def leer_moleculas():
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
    sup = Chem.SDMolSupplier(str(DATOS / "structures_extended.sdf"), removeHs=False, sanitize=True)
    filas = []
    for mol in sup:
        if mol is None:
            continue
        p = mol.GetPropsAsDict()
        if not p.get("MAIN_CLASS") or not p.get("SUB_CLASS"):
            continue
        e, mo = parsear(p.get("FORMULA"))
        if mo:
            continue
        filas.append((hill(e), str(p["MAIN_CLASS"]).strip(), str(p["SUB_CLASS"]).strip(),
                      gen.GetFingerprintAsNumPy(mol).astype(np.float32)))
    return filas


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "29_class_subclass_per_molecule.log", "w")

    def log(m=""):
        print(m, flush=True); fh.write(m + "\n"); fh.flush()

    log("=" * 88); log("CLASS AND SUBCLASS PER MOLECULE — upper limit, formula only and structure oracle"); log("=" * 88)
    filas = leer_moleculas()
    formulas = np.array([r[0] for r in filas]); X_est = np.stack([r[3] for r in filas])
    log(f"  molecules with structure, class, subclass and valid formula: {len(filas):,}")
    cache = {}
    for f in set(formulas):
        r = construir(f); cache[f] = r[1] if r is not None else None
    ok = np.array([cache[f] is not None for f in formulas])
    formulas, X_est = formulas[ok], X_est[ok]
    X_for = np.stack([cache[f] for f in formulas]).astype(np.float32)
    log(f"  after building R2: {len(formulas):,} molecules, {len(set(formulas)):,} formulas")
    etiquetas = {"Class": np.array([r[1] for r, o in zip(filas, ok) if o]),
                 "Subclass": np.array([r[2] for r, o in zip(filas, ok) if o])}

    res = []
    for nombre, _ in NIVELES:
        cats = etiquetas[nombre]
        por_f = defaultdict(list)
        for f, c in zip(formulas, cats): por_f[f].append(c)
        frec = Counter(cats)
        mayor = {f: min(set(v), key=lambda c: (-v.count(c), -frec[c], c)) for f, v in por_f.items()}
        ambigua = {f: len(set(v)) > 1 for f, v in por_f.items()}
        u_form = np.array(sorted(por_f))
        clases = np.array(sorted(set(cats))); cod = {c: i for i, c in enumerate(clases)}
        y_mol = np.array([cod[c] for c in cats])
        lim_global = np.mean([cod[mayor[f]] == y for f, y in zip(formulas, y_mol)]) * 100
        log(f"\n--- {nombre}: {len(clases)} labels · ambiguous formulas {sum(ambigua.values()):,} "
            f"({sum(ambigua.values())/len(u_form)*100:.2f} %) · catalog limit {lim_global:.2f} % ---")
        for s in SEMILLAS:
            rng = np.random.default_rng(s)
            perm = rng.permutation(len(u_form)); c = int(0.8 * len(u_form))
            f_tr, f_te = set(u_form[perm[:c]]), set(u_form[perm[c:]])
            itr = np.array([f in f_tr for f in formulas]); ite = ~itr
            m_est = RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=s, class_weight="balanced")
            m_est.fit(X_est[itr], y_mol[itr]); p_est = m_est.predict(X_est[ite])
            ftr = np.array(sorted(f_tr))
            m_for = RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=s, class_weight="balanced")
            m_for.fit(np.stack([cache[f] for f in ftr]), [cod[mayor[f]] for f in ftr])
            p_for = m_for.predict(X_for[ite])
            amb = np.array([ambigua[f] for f in formulas[ite]]); y = y_mol[ite]
            acierta_lim = np.array([cod[mayor[f]] for f in formulas[ite]]) == y
            for modelo, p in (("structure_oracle", p_est), ("formula_R2", p_for)):
                res.append({"level": nombre, "seed": s, "model": modelo,
                            "accuracy_per_molecule": (p == y).mean() * 100,
                            "f1_macro": f1_score(y, p, average="macro") * 100,
                            "acc_unambiguous": (p[~amb] == y[~amb]).mean() * 100,
                            "acc_ambiguous": (p[amb] == y[amb]).mean() * 100,
                            "n_test": int(ite.sum()), "n_ambiguous": int(amb.sum()),
                            "limit_test": acierta_lim.mean() * 100,
                            "limit_ambiguous": acierta_lim[amb].mean() * 100})
            r = res[-1]
            log(f"  seed {s}: test {r['n_test']:,} (ambiguous {r['n_ambiguous']:,}) · limit {r['limit_test']:.2f} % · "
                f"formula {res[-1]['accuracy_per_molecule']:.2f} % · oracle {res[-2]['accuracy_per_molecule']:.2f} %")
            pd.DataFrame(res).to_csv(RES / "29_class_subclass_per_molecule.csv", index=False)

    t = pd.DataFrame(res); t.to_csv(RES / "29_class_subclass_per_molecule.csv", index=False)
    for nombre, _ in NIVELES:
        tn = t[t.level == nombre]
        m = tn.groupby("model").mean(numeric_only=True); sd = tn.groupby("model").std(numeric_only=True)
        o, f = m.loc["structure_oracle"], m.loc["formula_R2"]
        d = (f.limit_test - f.accuracy_per_molecule)
        tf = tn[tn.model == "formula_R2"]; dif = tf.limit_test - tf.accuracy_per_molecule
        log(f"\n=== {nombre} (mean of 5 seeds) ===")
        log(f"  formula only : accuracy {f.accuracy_per_molecule:.2f} % (σ {sd.loc['formula_R2','accuracy_per_molecule']:.2f}; "
            f"min {tf.accuracy_per_molecule.min():.2f}, max {tf.accuracy_per_molecule.max():.2f}) · F1 macro {f.f1_macro:.2f} %")
        log(f"  limit        : {f.limit_test:.2f} % · difference {d:.2f} points (from {dif.min():.2f} to {dif.max():.2f})")
        log(f"  oracle       : accuracy {o.accuracy_per_molecule:.2f} % · F1 macro {o.f1_macro:.2f} %")
        log(f"  unambiguous  : formula {f.acc_unambiguous:.2f} % · oracle {o.acc_unambiguous:.2f} %")
        log(f"  ambiguous    : formula {f.acc_ambiguous:.2f} % · oracle {o.acc_ambiguous:.2f} % · limit {f.limit_ambiguous:.2f} % "
            f"({int(f.n_ambiguous):,} of {int(f.n_test):,} molecules per test)")
    log("\nSaved: results/29_class_subclass_per_molecule.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
