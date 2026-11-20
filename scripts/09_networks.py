#!/usr/bin/env python
"""
09_networks.py — Phase 3 (the missing part): dense network (MLP) and convolutional network (CNN).

Completes the table of four model families required by the plan. Random Forest and
XGBoost were done in session 15 (05_models.py); the two neural-network families
are covered here.

  MLP  on R1, R2 and R3   — vectors of 14, 22 and 14 numbers
  CNN  on R4              — simulated isotopic pattern, 2 channels x 16 peaks

The CNN only makes sense on R4: a convolution needs local structure in the
input, and R1-R3 are lists of numbers with no natural order. R4 does have it (it is the
isotopic envelope seen by the spectrometer), which is why it was built (session 14).

RUNS ON CPU. There is no GPU on this server and none is needed: the largest input is
22 numbers per example, four orders of magnitude below what justifies a
GPU. The bottleneck of the project is RAM, which a GPU does not fix.

PROTOCOL IDENTICAL TO 05_models.py so that the figures are comparable:
80/20 formula-grouped split, the same 5 seeds, the same permutation,
the same test set. The selection metric is macro F1, never accuracy.

The only addition is a validation cut (10 % of the training set) for early
stopping. It comes from the training block, NEVER from the test block: deciding
when to stop by looking at the test set would be exactly the error this project criticises.

Output: results/09_networks.log + .csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cargar_lmsd, hill, parsear  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATOS, RES = BASE / "data", BASE / "results"

SEMILLAS = [0, 1, 2, 3, 4]      # the same as in Phase 3
LOTE, LOTE_GRANDE = 256, 512    # 512 from 20,000 examples (task B): same
UMBRAL_LOTE = 20_000            # result at half the wall-clock time
EPOCAS_MAX = 150
PACIENCIA = 15                  # epochs without improvement in validation macro F1
LR = 1e-3
FRAC_VAL = 0.10

torch.set_num_threads(8)


# --------------------------------------------------------------------------- networks
class MLP(nn.Module):
    """Dense network. Two hidden layers; the dataset is small and more depth
    only adds overfitting with no gain (verified in preliminary tests)."""

    def __init__(self, n_ent, n_sal):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_ent, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_sal),
        )

    def forward(self, x):
        return self.red(x)


class CNN(nn.Module):
    """1D convolutional network on the isotopic envelope (2 channels x 16 peaks).

    The size-3 kernel spans contiguous peaks M, M+1, M+2: this is precisely the relation
    that encodes the isotopic composition. No causal padding or dilation: the
    signal has 16 points, there is no room for anything more elaborate.
    """

    def __init__(self, n_canales, n_pos, n_sal):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_canales, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.cab = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64 * n_pos, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_sal),
        )

    def forward(self, x):
        return self.cab(self.conv(x))


# ---------------------------------------------------------------- training
def normalizar(Xtr, *otros):
    """Standardises with the mean and standard deviation OF THE TRAINING SET.

    Computing them on the full dataset would be leakage: the test block would have
    influenced the transformation. It is subtle, which is why it is worth writing down.
    For R1/R2 it matters a lot: their columns range from 0 to 8,840.
    """
    eje = tuple(range(Xtr.ndim))[:-1] if Xtr.ndim == 2 else (0, 2)
    mu = Xtr.mean(axis=eje, keepdims=True)
    sd = Xtr.std(axis=eje, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return [(X - mu) / sd for X in (Xtr,) + otros]


def entrenar(cls, args, Xtr, ytr, Xva, yva, Xte, pesos, semilla):
    torch.manual_seed(semilla)
    modelo = cls(*args)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=pesos)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    n = len(ytr_t)
    lote = LOTE_GRANDE if n >= UMBRAL_LOTE else LOTE
    mejor_f1, mejor_estado, sin_mejora = -1.0, None, 0
    g = torch.Generator().manual_seed(semilla)

    for _ in range(EPOCAS_MAX):
        modelo.train()
        for idx in torch.randperm(n, generator=g).split(lote):
            if len(idx) < 2:          # BatchNorm needs at least 2 examples
                continue
            opt.zero_grad()
            crit(modelo(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()

        modelo.eval()
        with torch.no_grad():
            pv = modelo(Xva_t).argmax(1).numpy()
        f1v = f1_score(yva, pv, average="macro", zero_division=0)

        if f1v > mejor_f1 + 1e-5:
            mejor_f1, sin_mejora = f1v, 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    with torch.no_grad():
        return modelo(Xte_t).argmax(1).numpy()


def correr(X, y, tarea, rep, arquitectura, log, filas):
    """Trains an architecture on one representation, in its two variants
    (without and with class weighting), averaging over the 5 seeds."""
    clases = np.unique(y)
    cod = {c: i for i, c in enumerate(clases)}
    yc = np.array([cod[v] for v in y])
    nk = len(clases)

    acum = {}
    for s in SEMILLAS:
        # --- split identical to that of 05_models.py
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(y))
        c = int(0.8 * len(y))
        itr_todo, ite = perm[:c], perm[c:]
        # --- validation cut, taken from the training set
        nv = int(FRAC_VAL * len(itr_todo))
        iva, itr = itr_todo[:nv], itr_todo[nv:]

        Xtr, Xva, Xte = normalizar(X[itr], X[iva], X[ite])
        conteo = np.bincount(yc[itr], minlength=nk).astype(np.float64)
        w = torch.tensor(len(yc[itr]) / (nk * np.maximum(conteo, 1)), dtype=torch.float32)

        for nom, pesos in ((arquitectura, None), (arquitectura + "-bal", w)):
            args = (Xtr.shape[1], nk) if arquitectura == "MLP" else (Xtr.shape[1], Xtr.shape[2], nk)
            t0 = time.time()
            p = entrenar(MLP if arquitectura == "MLP" else CNN, args,
                         Xtr, yc[itr], Xva, yc[iva], Xte, pesos, s)
            acum.setdefault(nom, []).append({
                "accuracy": (p == yc[ite]).mean() * 100,
                "balanced_acc": balanced_accuracy_score(yc[ite], p) * 100,
                "f1_macro": f1_score(yc[ite], p, average="macro", zero_division=0) * 100,
                "seconds": time.time() - t0,
            })

    for nom in sorted(acum):
        r = {k: float(np.mean([x[k] for x in acum[nom]])) for k in acum[nom][0]}
        sd = np.std([x["f1_macro"] for x in acum[nom]])
        log(f"  {rep:14s} {nom:9s} {r['accuracy']:7.2f}% {r['balanced_acc']:9.2f}% "
            f"{r['f1_macro']:9.2f}% (±{sd:4.2f})  {r['seconds']:6.1f}s")
        filas.append({"task": tarea, "representation": rep, "model": nom,
                      **{k: round(v, 3) for k, v in r.items()}})


# --------------------------------------------------------------------------- main
def main():
    RES.mkdir(exist_ok=True)
    fh = open(RES / "09_networks.log", "w")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()

    log("=" * 84)
    log("PHASE 3 (continued) — DENSE NETWORK (MLP) AND CONVOLUTIONAL NETWORK (CNN)")
    log("=" * 84)
    log(f"torch {torch.__version__}, {torch.get_num_threads()} threads, CPU.")
    log(f"Adam lr={LR}, batch={LOTE}, up to {EPOCAS_MAX} epochs with early stopping")
    log(f"(patience {PACIENCIA} on macro F1 of a {FRAC_VAL:.0%} validation set")
    log("taken from the training set, never from the test set).")
    log("80/20 formula-grouped split, the same 5 seeds as in Phase 3.")
    log("Standardisation fitted on the training set only.")

    z = np.load(DATOS / "representations.npz", allow_pickle=True)
    formulas, etiqueta = z["formula"], z["label"]
    reps_mlp = {"R1 Mitchell": z["R1"], "R2 Extended": z["R2"], "R3 Positional": z["R3"]}
    rep_cnn = {"R4 Isotopic": z["R4"]}

    d = cargar_lmsd(DATOS)
    hs = []
    for x in d["FORMULA"]:
        e, mo = parsear(x)
        hs.append(None if mo else hill(e))
    d["hill"] = hs
    d = d[d["hill"].notna()]
    may_cat = d.groupby("hill")["CATEGORY"].agg(lambda s: s.value_counts().index[0])
    may_cls = d.groupby("hill")["MAIN_CLASS"].agg(lambda s: s.value_counts().index[0])

    filas = []
    cab = f"  {'Representation':14s} {'Model':9s} {'Acc.':>8s} {'Bal.acc':>10s} {'F1 macro':>10s}"

    def bloque(titulo, X_sel, y, sel_cnn):
        log("\n" + "=" * 84)
        log(titulo)
        log("=" * 84)
        log(cab)
        for nom, M in reps_mlp.items():
            correr(M[X_sel], y, titulo.split("—")[0].strip(), nom, "MLP", log, filas)
        for nom, M in rep_cnn.items():
            correr(M[sel_cnn], y, titulo.split("—")[0].strip(), nom, "CNN", log, filas)

    # ---- A: category
    sel = np.array([f in may_cat.index for f in formulas])
    yA = np.array([may_cat.get(f, None) for f in formulas[sel]])
    bloque("TASK A — Lipid category (8 classes)   ·  upper limit 98,22 %  ·  k-NN 90,07 %  "
           "·  best Phase 3: 89,32 %", sel, yA, sel)

    # ---- B: detection
    todo = np.ones(len(formulas), dtype=bool)
    bloque("TASK B — Lipid / non-lipid detection   ·  k-NN 94,35 %  "
           "·  best Phase 3: 90,82 %", todo, etiqueta, todo)

    # ---- C: main class
    selC = np.array([f in may_cls.index for f in formulas])
    yC = np.array([may_cls.get(f, None) for f in formulas[selC]])
    vc = pd.Series(yC).value_counts()
    raras = set(vc[vc < 10].index)
    keep = np.array([v not in raras for v in yC])
    idxC = np.where(selC)[0][keep]
    log(f"\n  (task C: {len(raras)} classes with fewer than 10 formulas are excluded)")
    bloque("TASK C — Main class   ·  upper limit 95,50 %  ·  best Phase 3: 71,52 %",
           idxC, yC[keep], idxC)

    t = pd.DataFrame(filas)
    t.to_csv(RES / "09_networks.csv", index=False)

    log("\n" + "=" * 84)
    log("BEST NETWORK PER TASK (by macro F1)")
    log("=" * 84)
    for tar in t["task"].unique():
        b = t[t["task"] == tar].sort_values("f1_macro", ascending=False).iloc[0]
        log(f"  {tar:14s} {b['representation']:14s} {b['model']:9s} "
            f"F1 {b['f1_macro']:6.2f}%  bal.acc {b['balanced_acc']:6.2f}%  "
            f"acc. {b['accuracy']:6.2f}%")
    log("\n  Saved: results/09_networks.csv")
    log("  Compare against results/05_models.csv (RF and XGBoost).")
    fh.close()


if __name__ == "__main__":
    sys.exit(main())
