#!/usr/bin/env python
"""
17_structure_oracle.py — GNN-1 / E3: the cost of not knowing the structure (oracle).

Preregistered in session 21. Role: ORACLE, not competitor. A model that sees the
molecular structure is not subject to the formula-only ceiling (it sees each molecule
separately), so it does NOT go in the comparison table: it measures how much would be known if
MS1 gave the structure, and its difference from the formula-only model is "the cost of not
knowing the structure".

Implementation E3 (the cheap one): Morgan fingerprints (radius 2, 2,048 bits) computed with
RDKit from the LMSD extended structures, and a balanced Random Forest. A graph network
is not needed to establish the bound.

Protocol: the same 80/20 split GROUPED BY FORMULA (seeds 0–4). Evaluation unit:
the MOLECULE (each LMSD entry), because that is what the oracle distinguishes.
On the same test set the following are evaluated:
  · ORACLE    RF on the structural fingerprint → category of THAT molecule.
  · FORMULA   RF-bal on R2 → predicts for the formula; compared with the category of
              THAT molecule (strict per-molecule criterion).
And molecules whose formula is ambiguous (several categories) are separated from the rest:
that is where the oracle should win and the formula model cannot.

Output: results/17_structure_oracle.log/.csv
"""
import importlib.util
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

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


def limpia(c):
    import re
    return re.sub(r"\s*\[[A-Z]{2}\]$", "", str(c))


def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "17_structure_oracle.log", "w")

    def log(m=""):
        print(m, flush=True); fh.write(m + "\n"); fh.flush()

    log("=" * 84); log("STRUCTURAL ORACLE (E3) — the cost of not knowing the structure"); log("=" * 84)

    # ---- read molecules with structure from the extended SDF
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
    sup = Chem.SDMolSupplier(str(DATOS / "structures_extended.sdf"), removeHs=False, sanitize=True)
    filas, sin_mol, sin_cat, sin_formula = [], 0, 0, 0
    for mol in sup:
        if mol is None:
            sin_mol += 1; continue
        props = mol.GetPropsAsDict()
        cat = props.get("CATEGORY"); f = props.get("FORMULA")
        if not cat:
            sin_cat += 1; continue
        e, mo = parsear(f)
        if mo:
            sin_formula += 1; continue
        fp = np.zeros(N_BITS, dtype=np.uint8)
        arr = gen.GetFingerprintAsNumPy(mol)
        fp[:] = arr
        filas.append((hill(e), limpia(cat), fp))
    log(f"  molecules with structure, category and valid formula: {len(filas):,}  "
        f"(no structure {sin_mol:,} · no category {sin_cat:,} · invalid formula {sin_formula:,})")

    formulas = np.array([r[0] for r in filas]); cats = np.array([r[1] for r in filas])
    X_est = np.stack([r[2] for r in filas]).astype(np.float32)
    # formula representation (R2) per molecule
    cache = {}
    for f in set(formulas):
        r = construir(f); cache[f] = r[1] if r is not None else None
    ok = np.array([cache[f] is not None for f in formulas])
    formulas, cats, X_est = formulas[ok], cats[ok], X_est[ok]
    X_for = np.stack([cache[f] for f in formulas]).astype(np.float32)

    # majority label and ambiguity per formula
    por_f = defaultdict(list)
    for f, c in zip(formulas, cats): por_f[f].append(c)
    # DETERMINISTIC tie-break (session 31): with max(set(v), key=v.count) the tie between families was
    # resolved according to the order of a Python set, which changes on every run (PYTHONHASHSEED);
    # the result on ambiguous formulas varied between 54 and 62 %. Now: the most frequent family in
    # the formula; if tied, the most frequent in the whole catalog; if still tied, alphabetical order.
    frec_global = Counter(cats)
    mayor = {f: min(set(v), key=lambda c: (-v.count(c), -frec_global[c], c)) for f, v in por_f.items()}
    ambigua = {f: len(set(v)) > 1 for f, v in por_f.items()}
    u_form = np.array(sorted(por_f))
    clases = np.array(sorted(set(cats))); cod = {c: i for i, c in enumerate(clases)}
    y_mol = np.array([cod[c] for c in cats])
    log(f"  unique formulas: {len(u_form):,} · ambiguous: {sum(ambigua.values()):,} "
        f"({sum(ambigua.values())/len(u_form)*100:.2f} %) · molecules with ambiguous formula: "
        f"{int(np.mean([ambigua[f] for f in formulas])*100):d} % approx.")

    res = []
    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(u_form)); c = int(0.8 * len(u_form))
        f_tr, f_te = set(u_form[perm[:c]]), set(u_form[perm[c:]])
        itr = np.array([f in f_tr for f in formulas]); ite = ~itr
        # ORACLE: structure → category of the molecule
        m_est = RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=s, class_weight="balanced")
        m_est.fit(X_est[itr], y_mol[itr]); p_est = m_est.predict(X_est[ite])
        # FORMULA: R2 → category (trained per unique formula, majority label)
        ftr = np.array(sorted(f_tr))
        m_for = RandomForestClassifier(n_estimators=N_ARB, n_jobs=12, random_state=s, class_weight="balanced")
        m_for.fit(np.stack([cache[f] for f in ftr]), [cod[mayor[f]] for f in ftr])
        p_for = m_for.predict(X_for[ite])
        amb = np.array([ambigua[f] for f in formulas[ite]]); y = y_mol[ite]
        # session 33: upper limit ON THE TEST MOLECULES OF THIS SPLIT (the best answer of a
        # method that only sees the formula is its most frequent family; with a formula-grouped split, all the
        # molecules of a test formula are in the test set).
        acierta_lim = np.array([cod[mayor[f]] for f in formulas[ite]]) == y
        lim_te, lim_amb = acierta_lim.mean() * 100, acierta_lim[amb].mean() * 100
        for nombre, p in (("structure_oracle", p_est), ("formula_R2", p_for)):
            res.append({"seed": s, "model": nombre,
                        "accuracy_per_molecule": (p == y).mean() * 100,
                        "f1_macro": f1_score(y, p, average="macro") * 100,
                        "bal_acc": balanced_accuracy_score(y, p) * 100,
                        "acc_unambiguous": (p[~amb] == y[~amb]).mean() * 100,
                        "acc_ambiguous": (p[amb] == y[amb]).mean() * 100,
                        "n_test": int(ite.sum()), "n_ambiguous": int(amb.sum()),
                        "limit_test": lim_te, "limit_ambiguous": lim_amb})
    t = pd.DataFrame(res); t.to_csv(RES / "17_structure_oracle.csv", index=False)
    m = t.groupby("model").mean(numeric_only=True); sd = t.groupby("model").std(numeric_only=True)
    o, f = m.loc["structure_oracle"], m.loc["formula_R2"]
    log(f"\n  test set: {int(o['n_test']):,} molecules per seed, of which "
        f"{int(o['n_ambiguous']):,} have an ambiguous formula")
    log(f"\n  {'':28s} {'accuracy/molecule':>19s} {'F1 macro':>10s} {'bal.acc':>9s} {'unambiguous':>12s} {'ambiguous':>10s}")
    for k, lab in (("structure_oracle", "ORACLE (structure)"), ("formula_R2", "FORMULA ONLY (R2)")):
        r = m.loc[k]
        log(f"  {lab:28s} {r['accuracy_per_molecule']:18.2f}% {r['f1_macro']:9.2f}% {r['bal_acc']:8.2f}% "
            f"{r['acc_unambiguous']:11.2f}% {r['acc_ambiguous']:9.2f}%")
    log(f"\n  COST OF NOT KNOWING THE STRUCTURE (accuracy per molecule): "
        f"{o['accuracy_per_molecule'] - f['accuracy_per_molecule']:+.2f} points overall; "
        f"{o['acc_unambiguous'] - f['acc_unambiguous']:+.2f} on unambiguous formulas; "
        f"{o['acc_ambiguous'] - f['acc_ambiguous']:+.2f} on AMBIGUOUS formulas")
    log(f"  (σ across seeds, accuracy: oracle {sd.loc['structure_oracle','accuracy_per_molecule']:.2f}, "
        f"formula {sd.loc['formula_R2','accuracy_per_molecule']:.2f})")
    log("\n  UPPER LIMIT PER SPLIT (session 33), on the test molecules of each seed:")
    for s_ in SEMILLAS:
        r_ = t[(t.seed == s_) & (t.model == "formula_R2")].iloc[0]
        log(f"    seed {s_}: limit {r_.limit_test:6.2f} % · formula model {r_.accuracy_per_molecule:6.2f} % · "
            f"difference {r_.limit_test - r_.accuracy_per_molecule:5.2f} · on ambiguous: limit {r_.limit_ambiguous:6.2f} %, "
            f"model {r_.acc_ambiguous:6.2f} %")
    tf = t[t.model == "formula_R2"]
    log(f"    mean: limit {tf.limit_test.mean():.2f} %, difference {(tf.limit_test - tf.accuracy_per_molecule).mean():.2f} points "
        f"(from {(tf.limit_test - tf.accuracy_per_molecule).min():.2f} to {(tf.limit_test - tf.accuracy_per_molecule).max():.2f})")
    log("\n  Reading: the strict formula-only ceiling (97,46 %) is the per-molecule accuracy of the best")
    log("  formula classifier; the oracle is not subject to it. The difference on the ambiguous")
    log("  formulas is the value, in points, of the structural elucidation that MS1 does not give.")
    log("\nSaved: results/17_structure_oracle.csv")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
