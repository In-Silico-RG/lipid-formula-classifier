#!/usr/bin/env python
"""
30_summary.py — joins the parts written by 30_feature_selection.py and applies rule D4 per model.

Rule (fixed before running): for each model, keep the smallest feature set whose macro F1 is
within one SD (across repeats) of that model's best set, in BOTH tasks. Then compares the sets
chosen by the different models.

Output: results/30_summary.log + results/30_feature_selection_all.csv
"""
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
t = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("30_feature_selection_*.csv"))
               if f.name != "30_feature_selection_all.csv"], ignore_index=True)
t = t.drop_duplicates(["model", "task", "set"], keep="last")
t.to_csv(RES / "30_feature_selection_all.csv", index=False)
fh = open(RES / "30_summary.log", "w")


def log(m=""):
    print(m)
    fh.write(m + "\n")


log("30 — FEATURE SELECTION: SUMMARY OVER MODELS (rule D4)\n")
chosen = {}
for model, g in t.groupby("model"):
    log(f"--- {model}")
    log(g.pivot(index="set", columns="task", values="macro_f1").round(2).to_string())
    ok, complete = None, True
    for task in ("category", "main_class"):
        s = g[g.task == task]
        if len(s) < 6:
            complete = False
            log(f"  {task}: only {len(s)} of 6 sets available")
            continue
        best = s.loc[s.macro_f1.idxmax()]
        within = set(s[s.macro_f1 >= best.macro_f1 - best.macro_f1_sd].set)
        log(f"  {task}: best {best.set} {best.macro_f1:.2f} ± {best.macro_f1_sd:.2f}; within 1 SD: {', '.join(sorted(within))}")
        ok = within if ok is None else ok & within
    if complete and ok:
        sizes = g.drop_duplicates("set").set_index("set").n_features
        chosen[model] = min(ok, key=lambda k: (sizes[k], k))
        log(f"  CHOSEN by {model}: {chosen[model]}")
    elif complete:
        log(f"  {model}: no set within 1 SD in both tasks")
    log()
log("--- agreement")
models = sorted(t.model.unique())
for m in models:
    log(f"  {m}: {chosen.get(m, 'no list within 1 SD in both tasks')}")
if len(chosen) == len(models) and len(set(chosen.values())) == 1:
    log("  ALL MODELS AGREE")
else:
    log(f"  NO AGREEMENT: {len(chosen)} of {len(models)} models give a list"
        + (f" ({', '.join(sorted(set(chosen.values())))})" if chosen else "") + "; decision goes back to AC")
