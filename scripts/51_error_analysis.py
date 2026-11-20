#!/usr/bin/env python
"""
51_error_analysis.py — Merged pipeline: where the errors of the final model fall (§3.3, §3.5).

Reads the out-of-fold probabilities of 40_models.py (seed 0) for the final model; no training.
  Task B  share of the test formulas, and of the detection errors, that fall on the lipid
          formulas also present in a curated non-lipid source (both labels accepted); error
          rate on those formulas and on the rest.
  Tasks A and C  fraction of the strict errors whose answer is nevertheless a label that
          LIPID MAPS accepts for that formula (another entry's label or an ALT_ label).

Usage:  python 51_error_analysis.py --set=R2C --model=XGB
Output: results/51_error_analysis_<set>_<model>.log + .csv
"""
import sys

import numpy as np
import pandas as pd

import merged_common as mc  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET, MODEL = OPT["set"], OPT["model"]


def load(task):
    z = np.load(mc.RES / f"40_oof_{SET}_{task}_{MODEL}.npz", allow_pickle=True)
    return z["classes"][z["proba"].argmax(1)], z["y"], z["formulas"]


def main():
    log = mc.logger(f"51_error_analysis_{SET}_{MODEL}")
    log(f"51 — WHERE THE ERRORS FALL · {SET} · {MODEL} · out-of-fold, seed 0")
    rows = []
    _, _, accB = mc.load_detection()
    p, y, f = load("B")
    both = np.array([len(accB[x]) > 1 for x in f])
    err = p != y
    r = {"task": "B", "share_of_units_on_shared_formulas": both.mean() * 100,
         "share_of_errors_on_shared_formulas": err[both].sum() / max(err.sum(), 1) * 100,
         "error_rate_shared": err[both].mean() * 100, "error_rate_rest": err[~both].mean() * 100,
         "errors": int(err.sum()), "units": len(y)}
    rows.append(r)
    log(f"\n  task B: shared formulas are {r['share_of_units_on_shared_formulas']:.2f} % of the formulas and carry "
        f"{r['share_of_errors_on_shared_formulas']:.1f} % of the {r['errors']:,} errors; error rate "
        f"{r['error_rate_shared']:.1f} % on them vs {r['error_rate_rest']:.2f} % on the rest")
    lip = y == 1
    r.update({"sensitivity": (p[lip] == 1).mean() * 100, "specificity": (p[~lip] == 0).mean() * 100,
              "lipids_missed": int((err & lip).sum()), "nonlipids_called_lipid": int((err & ~lip).sum()),
              "sensitivity_shared": (p[lip & both] == 1).mean() * 100,
              "sensitivity_rest": (p[lip & ~both] == 1).mean() * 100})
    log(f"  task B: sensitivity {r['sensitivity']:.2f} %, specificity {r['specificity']:.2f} %; errors = "
        f"{r['lipids_missed']:,} lipids missed + {r['nonlipids_called_lipid']:,} non-lipids called lipid; "
        f"lipids found {r['sensitivity_shared']:.1f} % on shared formulas vs {r['sensitivity_rest']:.1f} % on the rest")
    _, acc = mc.load_entries()
    for task, level in (("A", "category"), ("C", "main_class")):
        p, y, f = load(task)
        wrong = p != y
        ok_perm = np.array([pi in acc[level][x] for pi, x in zip(p, f)])
        r = {"task": task, "strict_accuracy": (~wrong).mean() * 100, "permissive_accuracy": ok_perm.mean() * 100,
             "errors": int(wrong.sum()), "errors_that_are_accepted_labels": (wrong & ok_perm).sum() / max(wrong.sum(), 1) * 100}
        rows.append(r)
        log(f"  task {task}: strict {r['strict_accuracy']:.2f} %, permissive {r['permissive_accuracy']:.2f} %; "
            f"{r['errors_that_are_accepted_labels']:.1f} % of the {r['errors']:,} strict errors are labels LIPID MAPS accepts")
    pd.DataFrame(rows).to_csv(mc.RES / f"51_error_analysis_{SET}_{MODEL}.csv", index=False)


if __name__ == "__main__":
    main()
