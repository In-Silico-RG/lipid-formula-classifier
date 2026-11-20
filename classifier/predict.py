"""Predict the LIPID MAPS category and main class of a molecular formula, and whether it is a lipid.

    from predict import LipidClassifier
    clf = LipidClassifier()                      # reads lipid_classifier.joblib (49_final_model.py)
    clf.predict("C44H82NO8P")

For each level (CATEGORY, MAIN_CLASS, LIPID) it returns the most probable label, its probability
and the prediction set: the labels that together contain the correct one with probability
1 - alpha, as measured on formulas the model had not seen (5-fold split grouped by formula).
CATEGORY and LIPID use one threshold per label ("mondrian"); MAIN_CLASS one shared threshold.
It also returns the distance to the five nearest LIPID MAPS formulas and whether the formula is
lipid-like (95 % of held-out lipid formulas pass); if it is not, the labels should not be trusted.

Command line:  python predict.py C44H82NO8P C27H46O
"""
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import merged_common as mc  # noqa: E402  (same input code as at training time)
from features import FEATURES, featurize  # noqa: E402


class LipidClassifier:
    def __init__(self, path=HERE / "lipid_classifier.joblib", alpha=0.05, accept=0.95):
        self.pkg = joblib.load(path)
        self.alpha, self.accept = alpha, accept

    def predict_many(self, formulas):
        canon = []
        for f in formulas:
            e, why = mc.parsear(f)
            if why:
                raise ValueError(f"formula {f!r} rejected: {why}")
            canon.append(mc.hill(e))
        X = mc.features(canon, sets=[self.pkg["input_list"]])[self.pkg["input_list"]]
        out = [{"formula": f, "hill": h} for f, h in zip(formulas, canon)]
        for level, L in self.pkg["levels"].items():
            P = np.zeros((len(canon), len(L["classes"])))
            P[:, L["model"].classes_] = L["model"].predict_proba(X)
            thr = L["conformal"][str(self.alpha)][L["default_mode"]]
            thr = np.full(len(L["classes"]), thr) if np.isscalar(thr) else np.asarray(thr)
            for r, p in zip(out, P):
                best = int(p.argmax())
                s = np.where(1 - p <= thr)[0]
                s = s[np.argsort(-p[s])]
                r[level] = {"label": str(L["classes"][best]), "probability": round(float(p[best]), 3),
                            f"set_{round((1 - self.alpha) * 100)}": [(str(L["classes"][i]), round(float(p[i]), 3)) for i in s]}
        nov = self.pkg["novelty"]
        idx = [FEATURES.index(c) for c in nov["features"]]
        dist = nov["detector"].score(np.array([np.array(featurize(h))[idx] for h in canon]))
        for r, x in zip(out, dist):
            r["distance_to_lipids"] = round(float(x), 3)
            r["lipid_like"] = bool(x <= nov["thresholds"][str(self.accept)])
        return out

    def predict(self, formula):
        return self.predict_many([formula])[0]


if __name__ == "__main__":
    for r in LipidClassifier().predict_many(sys.argv[1:] or ["C44H82NO8P", "C27H46O", "C18H32O2", "C6H12O6"]):
        print(r)
