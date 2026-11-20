#!/usr/bin/env python
"""
47_rules.py — Merged pipeline: readable rules (port of archive/tool_v2/v2/rules.py; §2.10).

Single decision trees with 4 to 128 leaves, or no limit (min_samples_leaf = 20), on four lists
of increasing size, computed by the tool's own feature code (classifier/features.py,
imported, not copied):
  vK        H/C, O/C (the two axes of the Van Krevelen diagram)
  vK+DBE    plus rings plus double bonds (DBE) and DBE per carbon
  elements  the six atom counts C, H, O, N, P, S
  readable  atom counts + DBE + DBE/C + H/C + O/C + N/C + C mod 5
Protocol of merged_common (entries, 5-fold grouped by formula, seeds 0, 1, 2); task A only.
The 16-leaf tree on the readable list, fitted on all entries, is written as a rule table.

Output: results/47_rules.log + _curve.csv + 47_rules_category.txt
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

import merged_common as mc  # noqa: E402

sys.path.insert(0, str(mc.BASE / "classifier"))
from features import FEATURES, featurize  # noqa: E402

SETS = {
    "vK": ["H_C_RATIO", "O_C_RATIO"],
    "vK+DBE": ["H_C_RATIO", "O_C_RATIO", "DBE", "DBE_PER_C"],
    "elements": ["C_COUNT", "H_COUNT", "O_COUNT", "N_COUNT", "P_COUNT", "S_COUNT"],
    "readable": ["C_COUNT", "H_COUNT", "O_COUNT", "N_COUNT", "P_COUNT", "S_COUNT",
                 "DBE", "DBE_PER_C", "H_C_RATIO", "O_C_RATIO", "N_C_RATIO", "C_MOD5"],
}
LEAVES = [4, 8, 12, 16, 24, 32, 64, 128, None]


def tree(leaves, seed):
    return DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=20, random_state=seed)


def main():
    log = mc.logger("47_rules")
    log("=" * 88)
    log("47 — READABLE RULES: single decision trees, merged protocol, task A (category)")
    log("=" * 88)
    d, _ = mc.load_entries()
    feats, ok = [], []
    for f in d.hill:
        try:
            feats.append(featurize(f)); ok.append(True)
        except ValueError:
            ok.append(False)
    ok = np.array(ok)
    log(f"entries {len(d):,}; without a tool mass for some element (left out): {int((~ok).sum()):,}")
    d = d[ok].reset_index(drop=True)
    F = pd.DataFrame(feats, columns=FEATURES)
    y, g = d.CATEGORY.to_numpy(), d.hill.to_numpy()

    rows = []
    for name, cols in SETS.items():
        X = F[cols].to_numpy()
        for L in LEAVES:
            per = []
            for s in mc.SEEDS:
                pred = mc.oof(lambda seed: tree(L, seed), X, y, g, s)
                per.append({"accuracy": accuracy_score(y, pred) * 100,
                            "macro_f1": f1_score(y, pred, average="macro") * 100})
            ms = mc.mean_sd(per)
            rows.append({"features": name, "max_leaves": L or "unlimited",
                         **{k: v[0] for k, v in ms.items()}, **{k + "_sd": v[1] for k, v in ms.items()}})
            log(f"  {name:9s} leaves {str(L or 'unlimited'):9s} acc {ms['accuracy'][0]:6.2f} ± {ms['accuracy'][1]:.2f}"
                f"   macro F1 {ms['macro_f1'][0]:6.2f}")
    curve = pd.DataFrame(rows)
    curve.to_csv(mc.RES / "47_rules_curve.csv", index=False)

    cols = SETS["readable"]
    X = F[cols].to_numpy()
    classes, yi = np.unique(y, return_inverse=True)
    t = tree(16, 0).fit(X, yi)
    r16 = curve[(curve.features == "readable") & (curve.max_leaves == 16)].iloc[0]
    leaf = t.apply(X)
    lines = ["# Lipid category from the molecular formula: 16-leaf decision tree, readable features",
             f"# Out-of-fold accuracy (5-fold grouped by formula, 3 seeds) {r16.accuracy:.2f} %, macro F1 {r16.macro_f1:.2f} %",
             "# Fitted on all LIPID MAPS entries (September 2026 extended release).", "",
             export_text(t, feature_names=cols, class_names=list(classes), decimals=2, show_weights=False),
             "# Leaves, largest first: predicted category, entries, purity, composition"]
    for lf in sorted(set(leaf), key=lambda k: -(leaf == k).sum()):
        mask = leaf == lf
        cnt = np.bincount(yi[mask], minlength=len(classes))
        comp = ", ".join(f"{classes[i]} {cnt[i]}" for i in np.argsort(-cnt) if cnt[i])
        lines.append(f"leaf {lf:3d}: {classes[cnt.argmax()]}  n={mask.sum():5d}  purity={cnt.max() / mask.sum():.3f}  [{comp}]")
    (mc.RES / "47_rules_category.txt").write_text("\n".join(lines) + "\n")
    log("\n  Saved: results/47_rules_curve.csv and results/47_rules_category.txt")


if __name__ == "__main__":
    main()
