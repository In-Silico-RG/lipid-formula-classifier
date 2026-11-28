#!/usr/bin/env python
"""
94_per_formula_bootstrap.py — Answer to the peer review (points 8 and the statistics comments): the unit "entry",
the weight of heavily enumerated formulas, uncertainty by resampling formulas, paired comparison of models, and
the nearest-neighbour baseline on unique formulas.

(a) Concentration: share of the entries held by the largest formulas (C22H34O6: 531), and the accuracy of the final
    model on them and without them.
(b) Same accuracy in three units, task A and C (final model, seeds 0-2): per entry (the paper), per formula with every
    formula weighted equally (the unit of L_formula), and per formula against its majority label; with the upper
    limits in the same units (L_entry, L_formula) computed here from the data.
(c) Seeds: accuracy of each seed and its mean (seed 0 is the one used in the descriptive analyses).
(d) Bootstrap over formulas (2000 resamples; entries move with their formula): 95 % interval of the accuracy of the final
    model, XGBoost, and the paired difference between them; the interval of the final model's distance to the limit.
(e) Nearest neighbours fitted on entries (as in the paper) and on unique formulas (one row per formula, majority label),
    R2C and T12, k = 1 and 5, same folds; scored per entry.

Output: results/94_per_formula_bootstrap.log + 94_*.csv
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import merged_common as mc
import review_common as rc

B = 2000


def xgb_model(seed):
    return xgb.XGBClassifier(n_estimators=500, tree_method="hist", n_jobs=rc.JOBS, verbosity=0, random_state=seed)


def formula_table(forms, y, ok_by_seed):
    """One row per formula: entries, correct entries per seed, majority label share."""
    df = pd.DataFrame({"f": forms, "y": y})
    for i, ok in enumerate(ok_by_seed):
        df[f"ok{i}"] = ok
    g = df.groupby("f")
    t = g.size().rename("n").to_frame()
    for i in range(len(ok_by_seed)):
        t[f"ok{i}"] = g[f"ok{i}"].sum()
    t["max_share_n"] = g.y.agg(lambda s: s.value_counts().iloc[0])
    return t


def main():
    log = mc.logger("94_per_formula_bootstrap")
    log("=" * 88)
    log("94 — PER-FORMULA VIEW, BOOTSTRAP OVER FORMULAS, PAIRED MODEL COMPARISON, kNN ON UNIQUE FORMULAS")
    log("=" * 88)
    d, acc, FE = rc.load_entries_features(("R2C", "T12"))
    X, forms = FE["R2C"], d.hill.to_numpy()
    rng = np.random.default_rng(0)
    out = {}
    size_rows = []
    for task, col in (("A", "CATEGORY"), ("C", "MAIN_CLASS")):
        y = d[col].to_numpy()
        oks_rf, oks_xgb = [], []
        for s in mc.SEEDS:
            pred, _, _ = rc.cached_oof(f"hs_{task}_formula", rc.rf_weighted, X, y, forms, s)
            oks_rf.append(pred == y)
        if task == "A":
            for s in mc.SEEDS:
                pred, _, _ = rc.cached_oof("xgb_A_R2C", xgb_model, X, y, forms, s)
                oks_xgb.append(pred == y)
        t_rf = formula_table(forms, y, oks_rf)
        n = t_rf.n.to_numpy()
        L_entry = t_rf.max_share_n.sum() / n.sum() * 100
        L_formula = (t_rf.max_share_n / t_rf.n).mean() * 100
        log(f"\n=== task {task} ({'category' if task == 'A' else 'main class'}) · {len(d):,} entries · {len(t_rf):,} formulas ===")
        log(f"  upper limits computed here (strict): per entry {L_entry:.2f} %, per formula (equal weight) {L_formula:.2f} %")

        # (a) concentration
        top = t_rf.sort_values("n", ascending=False)
        cum5, cum50 = top.n.iloc[:5].sum(), top.n.iloc[:50].sum()
        acc_all = np.mean([ok.mean() for ok in oks_rf]) * 100
        ok_mean = np.mean([t_rf[f"ok{i}"] for i in range(3)], axis=0) if False else None
        okf = pd.DataFrame({f"ok{i}": t_rf[f"ok{i}"] for i in range(3)}).mean(axis=1)
        log(f"  (a) the 5 largest formulas hold {cum5:,} entries ({cum5 / n.sum() * 100:.2f} %), the 50 largest {cum50:,} "
            f"({cum50 / n.sum() * 100:.1f} %); largest: {top.index[0]} ({int(top.n.iloc[0])} entries, "
            f"{top.n.iloc[0] / n.sum() * 100:.2f} %)")
        e5 = top.index[:5]
        acc5 = okf[e5].sum() / top.n.iloc[:5].sum() * 100
        acc_wo5 = (okf.sum() - okf[e5].sum()) / (n.sum() - top.n.iloc[:5].sum()) * 100
        log(f"      accuracy on those 5 formulas {acc5:.2f} %; on the remaining entries {acc_wo5:.2f} % (all entries {acc_all:.2f} %)")

        # (b) units
        per_entry = acc_all
        per_formula_equal = (okf / t_rf.n).mean() * 100
        fm = pd.Series(forms).drop_duplicates()
        maj = pd.DataFrame({"f": forms, "y": y}).groupby("f").y.agg(lambda s: s.value_counts().index[0])
        pred0, _, _ = rc.cached_oof(f"hs_{task}_formula", rc.rf_weighted, X, y, forms, 0)
        pf = pd.Series(pred0, index=forms).groupby(level=0).first()
        per_formula_majority = (pf.reindex(maj.index) == maj).mean() * 100
        perm = np.mean([pf[f] in acc["category" if task == "A" else "main_class"][f] for f in maj.index]) * 100
        log(f"  (b) final model, accuracy per entry {per_entry:.2f} % (limit {L_entry:.2f}); per formula, equal weight "
            f"{per_formula_equal:.2f} % (limit {L_formula:.2f}); per formula against its majority label "
            f"{per_formula_majority:.2f} %; per formula, permissive {perm:.2f} %")
        out[f"{task}_units"] = {"task": task, "per_entry": per_entry, "per_formula_equal": per_formula_equal,
                                "per_formula_majority": per_formula_majority, "per_formula_permissive": perm,
                                "L_entry": L_entry, "L_formula": L_formula}

        # (b2) accuracy by number of entries of the formula (per formula)
        okf_frac = (okf / t_rf.n)
        bins = pd.cut(t_rf.n, [0, 1, 5, 20, 10 ** 6], labels=["1 entry", "2-5 entries", "6-20 entries", "> 20 entries"])
        gb = pd.DataFrame({"n": t_rf.n, "acc": okf_frac, "bin": bins}).groupby("bin", observed=True)
        for name, g in gb:
            size_rows.append({"task": task, "formula_size": name, "formulas": len(g), "share_formulas_pct": len(g) / len(t_rf) * 100,
                              "entries": int(g.n.sum()), "share_entries_pct": g.n.sum() / n.sum() * 100, "accuracy_pct": g.acc.mean() * 100})
            log(f"      formulas with {name:13s}: {len(g):5,} ({len(g) / len(t_rf) * 100:4.1f} % of the formulas, {g.n.sum() / n.sum() * 100:4.1f} % of the entries) "
                f"accuracy {g.acc.mean() * 100:5.1f} %")

        # (c) seeds
        accs = [ok.mean() * 100 for ok in oks_rf]
        log(f"  (c) accuracy by seed: {', '.join(f'{a:.2f}' for a in accs)}; mean {np.mean(accs):.2f} ± {np.std(accs, ddof=1):.2f} "
            f"(seed 0 = {accs[0]:.2f}, {accs[0] - np.mean(accs):+.2f} from the mean)")

        # (d) bootstrap over formulas
        okm = okf.to_numpy()
        idx = rng.integers(0, len(n), size=(B, len(n)))
        def boot(num):
            return num[idx].sum(1) / n[idx].sum(1) * 100
        b_rf = boot(okm)
        lo, hi = np.percentile(b_rf, [2.5, 97.5])
        log(f"  (d) bootstrap over formulas ({B} resamples): final model accuracy {okm.sum() / n.sum() * 100:.2f} % "
            f"[95 % interval {lo:.2f}, {hi:.2f}]; distance to the entry limit {L_entry - okm.sum() / n.sum() * 100:.2f} points "
            f"[{L_entry - hi:.2f}, {L_entry - lo:.2f}]")
        out[f"{task}_boot"] = {"task": task, "accuracy": okm.sum() / n.sum() * 100, "lo": lo, "hi": hi}
        if task == "A":
            t_x = formula_table(forms, y, oks_xgb)
            okx = pd.DataFrame({f"ok{i}": t_x[f"ok{i}"] for i in range(3)}).mean(axis=1).to_numpy()
            b_x = boot(okx)
            dlo, dhi = np.percentile(b_rf - b_x, [2.5, 97.5])
            log(f"      XGBoost accuracy {okx.sum() / n.sum() * 100:.2f} % [{np.percentile(b_x, 2.5):.2f}, {np.percentile(b_x, 97.5):.2f}]; "
                f"paired difference RF-weighted minus XGBoost {(okm.sum() - okx.sum()) / n.sum() * 100:+.2f} points "
                f"[{dlo:+.2f}, {dhi:+.2f}] (share of resamples in which RF-weighted is higher: {np.mean(b_rf > b_x) * 100:.1f} %)")
            out["A_xgb"] = {"xgb_accuracy": okx.sum() / n.sum() * 100, "diff": (okm.sum() - okx.sum()) / n.sum() * 100,
                            "diff_lo": dlo, "diff_hi": dhi}

    # (e) nearest neighbours on entries and on unique formulas, task A, R2C and T12
    log("\n=== (e) nearest neighbours, task A, same folds, scored per entry ===")
    y = d.CATEGORY.to_numpy()
    rows = []
    for fl in ("R2C", "T12"):
        Xk = FE[fl]
        for k in (1, 5):
            for fit_on in ("entries", "unique formulas"):
                accs, f1s = [], []
                for s in mc.SEEDS:
                    pred = np.empty(len(y), dtype=y.dtype)
                    for tr, te in mc.splits(y, forms, s):
                        if fit_on == "entries":
                            Xt, yt = Xk[tr], y[tr]
                        else:
                            df = pd.DataFrame({"f": forms[tr], "i": tr, "y": y[tr]})
                            u = df.groupby("f").agg(i=("i", "first"), y=("y", lambda v: v.value_counts().index[0]))
                            Xt, yt = Xk[u.i.to_numpy()], u.y.to_numpy()
                        mdl = make_pipeline(StandardScaler(), KNeighborsClassifier(k, n_jobs=rc.JOBS))
                        mdl.fit(Xt, yt)
                        pred[te] = mdl.predict(Xk[te])
                    accs.append(accuracy_score(y, pred) * 100)
                    f1s.append(f1_score(y, pred, average="macro", zero_division=0) * 100)
                rows.append({"list": fl, "k": k, "fitted_on": fit_on, "accuracy": np.mean(accs), "accuracy_sd": np.std(accs, ddof=1),
                             "macro_f1": np.mean(f1s)})
                log(f"  {fl:4s} k = {k}  fitted on {fit_on:15s} accuracy {np.mean(accs):6.2f} ± {np.std(accs, ddof=1):.2f}  macro F1 {np.mean(f1s):6.2f}")
    pd.DataFrame(rows).to_csv(mc.RES / "94_knn_unique_formulas.csv", index=False)
    pd.DataFrame([out["A_units"], out["C_units"]]).to_csv(mc.RES / "94_units.csv", index=False)
    pd.DataFrame([out["A_boot"], out["C_boot"]]).to_csv(mc.RES / "94_bootstrap.csv", index=False)
    pd.DataFrame([out["A_xgb"]]).to_csv(mc.RES / "94_paired_rf_xgb.csv", index=False)
    pd.DataFrame(size_rows).to_csv(mc.RES / "94_accuracy_by_formula_size.csv", index=False)
    log("\n  Saved: results/94_*.csv")


if __name__ == "__main__":
    main()
