#!/usr/bin/env python
"""
45_conformal.py — Merged pipeline: prediction sets with a guaranteed coverage (spec D8; §2.8).

Split conformal prediction on the out-of-fold probabilities of the final model (written by
40_models.py, seed 0). The distinct formulas are divided at random into a calibration half
and a test half (all entries of a formula in the same half), 50 times. Score s = 1 - p(true);
threshold = the ceil((n+1)(1-alpha))-th smallest calibration score. Variants: marginal (one
threshold) and class-conditional / Mondrian (one per label; labels with fewer than
MIN_CAL calibration entries use the marginal threshold). Main text: Mondrian for the
category, marginal for the main class; both are reported for both levels.

Reported: coverage (true label in the set), coverage of any accepted label, mean set size,
fraction of single-label sets, coverage per label, and set size for formulas with one versus
several accepted labels (criterion fixed beforehand: sets should be larger for the latter).

Usage:  python 45_conformal.py --set=R2C --model=RF
Output: results/45_conformal_<set>_<model>.log + .csv + _per_label.csv
"""
import sys

import numpy as np
import pandas as pd

import merged_common as mc  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL = OPT["set"], OPT["model"]
ALPHAS = (0.10, 0.05, 0.01)
N_REPEATS = 50
MIN_CAL = 30


def threshold(s, alpha):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(s)[k - 1]


def main():
    log = mc.logger(f"45_conformal_{SET}_{MODEL}")
    log("=" * 88)
    log(f"45 — PREDICTION SETS · input list {SET} · model {MODEL} · {N_REPEATS} calibration/test halves")
    log("=" * 88)
    d, acc = mc.load_entries()
    rows, per_label = [], []
    for task, level in (("A", "category"), ("C", "main_class"), ("B", None)):
        f = mc.RES / f"40_oof_{SET}_{task}_{MODEL}.npz"
        if not f.exists():
            log(f"\n  task {task}: {f.name} not found, skipped")
            continue
        z = np.load(f, allow_pickle=True)
        P, classes, y, formulas = z["proba"], z["classes"], z["y"], z["formulas"]
        yi = np.searchsorted(classes, y)
        if level:
            accepted = acc[level]
            n_acc = np.array([len(accepted[x]) for x in formulas])
            acc_idx = [np.isin(classes, list(accepted[x])) for x in formulas]
        else:
            _, _, accB = mc.load_detection()
            n_acc = np.array([len(accB[x]) for x in formulas])
            acc_idx = [np.isin(classes, list(accB[x])) for x in formulas]
        A = np.stack(acc_idx)                      # units x labels: accepted labels of the formula
        s_true = 1 - P[np.arange(len(y)), yi]
        uf = np.unique(formulas)
        log(f"\n--- task {task}: {len(y):,} units, {len(uf):,} formulas, {len(classes)} labels")
        for alpha in ALPHAS:
            for variant in ("marginal", "mondrian"):
                res, cov_lab = [], {c: [] for c in range(len(classes))}
                for r in range(N_REPEATS):
                    rng = np.random.default_rng(r)
                    cal_f = set(rng.permutation(uf)[: len(uf) // 2])
                    cal = np.array([x in cal_f for x in formulas])
                    te = ~cal
                    qm = threshold(s_true[cal], alpha)
                    q = np.full(len(classes), qm)
                    if variant == "mondrian":
                        for c in range(len(classes)):
                            sc = s_true[cal & (yi == c)]
                            if len(sc) >= MIN_CAL:
                                q[c] = threshold(sc, alpha)
                    S = (1 - P[te]) <= q[None, :]
                    yt = yi[te]
                    covered = S[np.arange(len(yt)), yt]
                    any_acc = (S & A[te]).any(1)
                    size = S.sum(1)
                    amb = n_acc[te] > 1
                    res.append({"coverage": covered.mean() * 100, "coverage_any_accepted": any_acc.mean() * 100,
                                "mean_size": size.mean(), "single_label_sets": (size == 1).mean() * 100,
                                "empty_sets": (size == 0).mean() * 100,
                                "size_unambiguous": size[~amb].mean(),
                                "size_ambiguous": size[amb].mean() if amb.any() else np.nan})
                    for c in range(len(classes)):
                        if (yt == c).any():
                            cov_lab[c].append(covered[yt == c].mean() * 100)
                ms = mc.mean_sd(res)
                row = {"task": task, "alpha": alpha, "variant": variant}
                for k, (mu, sd) in ms.items():
                    row[k], row[k + "_sd"] = mu, sd
                rows.append(row)
                worst = min((np.mean(v), classes[c]) for c, v in cov_lab.items() if v)
                log(f"  α={alpha:.2f} {variant:9s} coverage {ms['coverage'][0]:6.2f} ± {ms['coverage'][1]:.2f}  "
                    f"any accepted {ms['coverage_any_accepted'][0]:6.2f}  size {ms['mean_size'][0]:5.2f}  "
                    f"single {ms['single_label_sets'][0]:5.1f} %  size unamb./amb. "
                    f"{ms['size_unambiguous'][0]:4.2f}/{ms['size_ambiguous'][0]:4.2f}  "
                    f"lowest label {worst[1]} {worst[0]:.1f} %")
                for c, v in cov_lab.items():
                    if v:
                        per_label.append({"task": task, "alpha": alpha, "variant": variant, "label": classes[c],
                                          "n": int((yi == c).sum()), "coverage": np.mean(v)})
    pd.DataFrame(rows).to_csv(mc.RES / f"45_conformal_{SET}_{MODEL}.csv", index=False)
    pd.DataFrame(per_label).to_csv(mc.RES / f"45_conformal_{SET}_{MODEL}_per_label.csv", index=False)
    log(f"\n  Saved: results/45_conformal_{SET}_{MODEL}.csv and _per_label.csv")


if __name__ == "__main__":
    main()
