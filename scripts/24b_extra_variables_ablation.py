#!/usr/bin/env python
"""
24b_extra_variables_ablation.py — Which group of new variables contributes? (main class, balanced random forest, 3 seeds)
On the main class with the 57 classes of 05 (≥ 10 formulas). Starts from R2 and adds, one by one, the four variable groups of 24_extra_variables.py; finally the full R5.
Output: results/24b_extra_variables_ablation.log
"""
import sys, warnings
from pathlib import Path
import numpy as np, importlib.util
warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from common import cargar_lmsd, hill, parsear  # noqa: E402
spec = importlib.util.spec_from_file_location("v24", BASE / "scripts" / "24_extra_variables.py"); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
z = np.load(BASE / "data" / "representations.npz", allow_pickle=True); formulas, R2, cols = z["formula"], z["R2"], list(z["R2_COLS"])
d = cargar_lmsd(BASE / "data"); hs = []
for x in d["FORMULA"]:
    e, mo = parsear(x); hs.append(None if mo else hill(e))
d["hill"] = hs; d = d[d.hill.notna()]; cls = d.groupby("hill")["MAIN_CLASS"].agg(lambda s: s.value_counts().index[0])
sel = np.array([f in cls.index for f in formulas]); F = formulas[sel]; X2 = R2[sel]; y = np.array([cls[f] for f in F])
from collections import Counter
_ct = Counter(y); _ok = np.array([_ct[c] >= 10 for c in y]); F, X2, y = F[_ok], X2[_ok], y[_ok]    # as in 05: classes with at least 10 formulas
E = v.extra(X2, cols)
G = {"Kendrick mass defect and Z* series": [0, 1, 2, 3, 4], "parities (C, H, N, O, H+N)": [5, 6, 7, 8, 9],
     "linear combinations (H−2C, O−4P, …)": [10, 11, 12, 13, 14, 15, 16], "indicators and ratios": [17, 18, 19, 20, 21, 22]}
v.SEMILLAS = [0, 1, 2]
log = open(BASE / "results" / "24b_extra_variables_ablation.log", "w")
def p(m): print(m, flush=True); log.write(m + "\n"); log.flush()
b = v.evaluar(X2, y, ("RF-bal",))["RF-bal"][:, 0].mean(); p(f"R2 (baseline)                                   accuracy {b:.2f}")
for k, ix in G.items():
    a = v.evaluar(np.hstack([X2, E[:, ix]]), y, ("RF-bal",))["RF-bal"][:, 0].mean(); p(f"R2 + {k:42s} {a:.2f}   Δ {a-b:+.2f}")
a = v.evaluar(np.hstack([X2, E]), y, ("RF-bal",))["RF-bal"][:, 0].mean(); p(f"R5 full                                         {a:.2f}   Δ {a-b:+.2f}")
