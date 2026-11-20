#!/usr/bin/env python
"""
43_networks.py — Merged pipeline: multilayer perceptron and convolutional network (spec D7).

Architectures, training routine and stopping rule are those of 09_networks.py, imported from
it unchanged: MLP 256-128 with batch normalisation and dropout 0.2; CNN with two convolution
layers (32, 64; kernel 3) on the isotopic pattern R4; Adam, learning rate 1e-3, batches of
256 (512 from 20,000 examples), at most 150 passes, stop after 15 passes without improvement
of the validation macro F1; plain and weighted (inverse class frequency) variants.

What changes is the protocol (merged_common): tasks A and C per entry, task B per formula;
5-fold StratifiedGroupKFold grouped by formula, seeds 0, 1, 2. The validation portion
(10 % of each training part) is taken BY FORMULA as well, so that no validation formula is
in the part used to fit the weights. Inputs are standardized with training-part statistics.

Usage:  python 43_networks.py --set=R2C [--tasks=A,B,C] [--nets=MLP-set,MLP-R3,CNN-R4] [--smoke]
Output: results/43_networks_<set>.log + .csv
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")
import merged_common as mc  # noqa: E402

n09 = mc._load("n09", "09_networks.py")            # MLP, CNN, normalizar, entrenar (unchanged)
OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SET = OPT["set"]
TASKS = OPT.get("tasks", "A,B,C").split(",")
TAG = SET + ("_" + "".join(TASKS) if "tasks" in OPT else "")   # one file per cluster job
NETS = OPT.get("nets", "MLP-set,MLP-R3,CNN-R4").split(",")
SMOKE = "--smoke" in sys.argv
torch.set_num_threads(int(OPT.get("jobs", 8)))
MIN_ENTRIES_C = 20


def run_net(arch, X, y, groups, seed, weighted):
    classes, yc = np.unique(y, return_inverse=True)
    nk = len(classes)
    pred = np.empty(len(y), dtype=int)
    for tr, te in mc.splits(y, groups, seed):
        gss = GroupShuffleSplit(n_splits=1, test_size=n09.FRAC_VAL, random_state=seed)
        a, b = next(gss.split(tr, groups=groups[tr]))
        itr, iva = tr[a], tr[b]
        Xtr, Xva, Xte = n09.normalizar(X[itr], X[iva], X[te])
        w = None
        if weighted:
            cnt = np.bincount(yc[itr], minlength=nk).astype(np.float64)
            w = torch.tensor(len(itr) / (nk * np.maximum(cnt, 1)), dtype=torch.float32)
        args = (Xtr.shape[1], nk) if arch == "MLP" else (Xtr.shape[1], Xtr.shape[2], nk)
        pred[te] = n09.entrenar(n09.MLP if arch == "MLP" else n09.CNN, args,
                                Xtr, yc[itr], Xva, yc[iva], Xte, w, seed)
    return classes[pred]


def main():
    log = mc.logger(f"43_networks_{TAG}")
    log("=" * 88)
    log(f"43 — NEURAL NETWORKS, merged protocol · input list {SET} (MLP), R3 (MLP), R4 (CNN)")
    log("=" * 88)
    d, acc = mc.load_entries()
    m, yB, accB = mc.load_detection()
    if SMOKE:
        d = d.sample(2000, random_state=0).reset_index(drop=True)
        keep = np.random.default_rng(0).random(len(m)) < 0.03
        m, yB = m[keep].reset_index(drop=True), yB[keep]
    need = {SET, "R3", "R4"}
    tasks = {}
    if "A" in TASKS or "C" in TASKS:
        FE = mc.features(d.hill.tolist(), sets=need)
        for t, col, a in (("A", "CATEGORY", "category"), ("C", "MAIN_CLASS", "main_class")):
            if t in TASKS:
                tasks[t] = dict(F=FE, y=d[col].to_numpy(), groups=d.hill.to_numpy(),
                                formulas=d.hill.to_numpy(), accepted=acc[a])
    if "B" in TASKS:
        FB = mc.features(m.formula.tolist(), sets=need)
        tasks["B"] = dict(F=FB, y=yB, groups=m.formula.to_numpy(), formulas=m.formula.to_numpy(),
                          accepted=accB)

    rows = []
    for tname, T in tasks.items():
        y = T["y"]
        big = None
        if tname == "C":
            vc = pd.Series(y).value_counts()
            big = vc[vc >= MIN_ENTRIES_C].index.to_numpy()
        log(f"\n--- task {tname} ({len(np.unique(y))} labels, n = {len(y):,})")
        for net in NETS:
            arch, inp = net.split("-")
            X = T["F"][SET if inp == "set" else inp]
            for weighted in (False, True):
                name = f"{arch}({SET if inp == 'set' else inp})" + ("-weighted" if weighted else "")
                t0, per_seed = time.time(), []
                for s in mc.SEEDS:
                    pred = run_net(arch, X, y, T["groups"], s, weighted)
                    sc = mc.scores(y, pred, T["formulas"], T["accepted"])
                    if big is not None:
                        sc["macro_f1_classes_ge20"] = f1_score(y, pred, labels=big, average="macro",
                                                               zero_division=0) * 100
                    per_seed.append(sc)
                ms = mc.mean_sd(per_seed)
                row = {"set": SET, "task": tname, "method": name}
                for k, (mu, sd) in ms.items():
                    row[k], row[k + "_sd"] = mu, sd
                rows.append(row)
                log(f"  {name:22s} acc {ms['accuracy'][0]:6.2f} ± {ms['accuracy'][1]:.2f}   "
                    f"macro F1 {ms['macro_f1'][0]:6.2f} ± {ms['macro_f1'][1]:.2f}   "
                    f"permissive {ms['permissive_accuracy'][0]:6.2f}   [{time.time() - t0:.0f} s]")
                pd.DataFrame(rows).to_csv(mc.RES / f"43_networks_{TAG}.csv", index=False)
    log(f"\n  Saved: results/43_networks_{TAG}.csv")


if __name__ == "__main__":
    main()
