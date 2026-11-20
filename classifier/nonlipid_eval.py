"""Evaluate the lipid / non-lipid behaviour of lipid_category_nl_v2 (roadmap item 1).

1. Binary view of the out-of-fold predictions (formula-grouped CV, from train.py):
   ROC AUC of P(lipid), precision and recall for non-lipids, category accuracy on lipids.
2. Stress test in the spirit of Mitchell et al. 2020: random CHNOPS formulas that pass
   basic chemical rules, sampled inside the composition box of the training data.
   How many does the model call a lipid? (Some random formulas may match real lipids,
   so this is an upper bound on false lipid calls, not an exact rate.)

Usage:  ~/miniconda3/bin/python nonlipid_eval.py
Writes: nonlipid_eval.json
"""
import json, sys
from pathlib import Path
import numpy as np, joblib
from sklearn.metrics import roc_auc_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import featurize_many, columns

NL = "Non-lipid [NL]"


def random_formulas(n=20000, seed=42):
    """Random CHNOPS formulas passing basic chemical rules (integer DBE >= 0, ratio limits)."""
    rng = np.random.default_rng(seed)
    forms = set()
    while len(forms) < n:
        c = int(rng.integers(2, 80)); nn = int(rng.integers(0, 5)); o = int(rng.integers(0, min(25, 2 * c)))
        p = int(rng.integers(0, 3)) if rng.random() < 0.3 else 0
        s = int(rng.integers(0, 3)) if rng.random() < 0.2 else 0
        dbe = int(rng.integers(0, max(1, c // 2)))
        h = 2 * c + 2 + nn + p - 2 * dbe
        if h < 0 or not (0.2 <= h / c <= 3.1) or nn / c > 1.3 or o / c > 1.2 or p / c > 0.3 or s / c > 0.8:
            continue
        forms.add("".join(f"{e}{v if v > 1 else ''}" for e, v in (("C", c), ("H", h), ("N", nn), ("O", o), ("P", p), ("S", s)) if v))
    return sorted(forms)


def main():
    o = np.load(HERE / "oof_category_nl.npz", allow_pickle=True)
    P, y, classes = o["proba"], o["y"], list(o["classes"])
    k = classes.index(NL)
    is_nl = y == k; p_lipid = 1 - P[:, k]; pred = P.argmax(1)
    res = {"n_lipid": int((~is_nl).sum()), "n_nonlipid": int(is_nl.sum()),
           "auc_lipid_vs_nonlipid": round(float(roc_auc_score(~is_nl, p_lipid)), 4),
           "nonlipid_precision": round(float(precision_score(is_nl, pred == k)), 4),
           "nonlipid_recall": round(float(recall_score(is_nl, pred == k)), 4),
           "lipid_called_nonlipid": round(float((pred[~is_nl] == k).mean()), 4),
           "category_accuracy_on_lipids": round(float((pred[~is_nl] == y[~is_nl]).mean()), 4),
           "overall_accuracy_9_classes": round(float((pred == y).mean()), 4)}
    for c in ("Polyketides [PK]", "Prenol Lipids [PR]", "Sterol Lipids [ST]", "Fatty Acyls [FA]"):
        i = classes.index(c)
        res[f"called_nonlipid_{c[-3:-1]}"] = round(float((pred[y == i] == k).mean()), 4)

    # Stress test: random chemically valid CHNOPS formulas.
    pkg = joblib.load(HERE / "lipid_category_nl_v2.joblib")
    model, le = pkg["model"], pkg["label_encoder"]
    thr = np.asarray(pkg["conformal"]["0.05"]["mondrian"])
    forms = random_formulas(20000, 42)
    Pr = model.predict_proba(featurize_many(forms)[:, columns(pkg["features"])])
    kk = list(le.classes_).index(NL)
    sets = (1 - Pr) <= thr
    res["stress_random_formulas"] = {
        "n": len(forms),
        "called_lipid": round(float((Pr.argmax(1) != kk).mean()), 4),
        "set95_only_nonlipid": round(float((sets[:, kk] & (sets.sum(1) == 1)).mean()), 4),
        "set95_contains_nonlipid": round(float(sets[:, kk].mean()), 4),
        "set95_empty": round(float((sets.sum(1) == 0).mean()), 4),
        "called_as": {le.classes_[i][-3:-1]: round(float(v), 4)
                      for i, v in enumerate(np.bincount(Pr.argmax(1), minlength=len(le.classes_)) / len(forms))},
    }
    (HERE / "nonlipid_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
