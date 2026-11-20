# Single pipeline of the paper — specification

*2026-11-17. Status: approved; all recommendations accepted,
including D2 = entry and the D4 rule. Implementation started with script 30.*

Goal: one codebase, one dataset, one evaluation protocol, so that every number in the
RSC Advances paper comes from the same pipeline and traces to one file in `results/`.
Base = `benchmark/` (numbered scripts, `common.py`, provenance). The analyses that exist
only in `tool/` are ported into it as new numbered scripts.

---

## 1. Differences that must be resolved

| | `benchmark/` | `tool/` |
|---|---|---|
| Data | LMSD extended, Sept 2026: 49,966 entries, 9,935 formulas | `lmsd.sdf` (older): 49,267 entries, 9,836 formulas |
| **Unit of training** | one row per **unique formula**, label = its most frequent class | one row per **entry** (molecule) |
| Split | 80/20 grouped by formula × 5 seeds | 5-fold `StratifiedGroupKFold` by formula |
| Features | R1 (Mitchell exact), R2 (R1 + DBE, ratios, polar atoms, mass/C, charge), R5 (R2 + 23 derived) | 12: C H O N P S, total atoms, DBE, O/C, polar atoms, C mod 5, C parity |
| Selection metric | macro F1 | accuracy |
| Trees | 500 (as Mitchell), Optuna in script 06 | 300, fixed |
| Conformal | Mondrian and marginal, α 0.10/0.05/0.01 | the same idea, own code |
| Python | 3.11 (`environment.yml`) | 3.13, scikit-learn 1.7.2 |

The unit of training is the largest of these differences: 94.7 % (tool, accuracy per
entry) and 89.1 % (benchmark, macro F1 per formula) are not comparable numbers.

---

## 2. Decisions

**D1. Data — LMSD extended, September 2026 (benchmark's).** Newer and larger; its
SHA-256 is already in `data/MANIFEST.csv`. The tool's `lmsd.sdf` is retired.
Non-lipids: curated layers N1–N3 (RefMet, ChEBI, COCONUT) and adversarial N4 (benchmark),
**plus HMDB 5.0 as an external negative test set** (tool). *Recommended; low risk.*
**Decision (2026-11-17): HMDB used.** Used as external non-lipid test and as the m/z library
(supplement); NOT in training, so every model can be rebuilt without it.

**D2. Unit of training [decide].**
- **(a) Entry, recommended.** One row per molecule, split grouped by formula. The upper
  limit (97.46 % / 91.18 %), the NIST validation and the prediction sets are all per
  molecule, so training and scoring use the same unit; predicted probabilities then
  reflect how often each class occurs among the isomers of a formula, which is what a
  user of `predict.py` needs. Cost: scripts 03–15 must change unit and be rerun.
- (b) Unique formula with majority label (current benchmark). Less work; but the tool,
  the prediction sets and the per-molecule upper limit then use a different unit.
- Either way, the other unit is reported once in the supplement as a robustness check.

**D3. Split — 5-fold `StratifiedGroupKFold` grouped by formula, repeated with 3 seeds.**
Every entry is tested once per repeat; mean ± SD over repeats. External sets: all their
formulas removed from training (rule already used by both projects).

**D4. Features — chosen by a rule fixed in advance [decide on the rule].**
New script compares, under D2–D3 with random forest: R1 (Mitchell exact), the tool's 12,
R2, R2 + C mod 5 + C parity, R5. **Rule: keep the smallest set whose macro F1 (category
and main class) is within one SD across repeats of the best set.** Fixing the rule now
prevents choosing the set after seeing the numbers. Note: both projects found
independently that carbon-number descriptors (C mod 5, C parity; R5 parities +2.2 points)
carry information beyond the atom counts; this experiment tests it on equal terms.

**D4 decision (2026-11-18): R2C.** The rule gave no agreement (XGBoost: R2C;
RF and k-NN: no list in both tasks), so the choice was made by the authors: **R2C** (24 features =
R2 + C mod 5 + C parity). Best or tied best in category for RF and XGBoost; about 2 points of
main-class macro F1 below R5 with half the features.

**D4 deviation (2026-11-18).** The MLP was left out of
the selection (too slow: >2 h for one list of 6); the choice uses RF, XGBoost and k-NN. The
MLP is still compared in script 43 with the chosen list.

**D5. Labels.** Category: 8. Main class: all 86 in training; metrics for all classes and
for classes with ≥ 20 entries (so small classes do not hide or dominate the picture).
Subclass: upper limit only (82.96 %).

**D6. Metrics.** Accuracy, macro F1 and balanced accuracy, scored strictly per molecule
in the main text; permissive (multi-label) scoring in its own section; detection by F1
and ROC AUC. Selection always by macro F1.

**D7. Models.** Final model = random forest (both projects agree trees win). Compared:
XGBoost, MLP, CNN (R4 isotopic), and the baselines (majority, chemical rules, k-NN,
look-up table). Hyperparameters: nested Optuna for RF and XGBoost only (script 06).

**D8. Prediction sets.** One implementation (split-conformal, score = 1 − p(true class)),
α = 0.05 main, 0.10/0.01 in the supplement. Category: Mondrian (one threshold per
category). Main class: marginal threshold (small classes cannot be calibrated per class).

**D9. External validation — three sets, same final model, formulas withheld.**
GNPS lab spectra (PNNL, HCE), MassBank EU, and **NIST SRM 1950 plasma (339 species)**.
Report for each: n, overlap with LMSD before removal, category and class metrics.
Open: GNPS licence still marked "unverified" in `data/README.md`.

**D10. Tool features kept.** `predict.py` retrained on the final pipeline; readable
16-leaf tree (`rules.py`); novelty check for formulas unlike any lipid (`novelty.py`).

**D11. Moved to the supplement:** m/z-only input (`mz*.py`), structure oracle (17),
sensitivity (28), per-formula robustness check (D2).
**Left out of the paper:** MS/MS contribution (23, 25) and MoNA (16), which break the
formula-only premise; the October 2025 models, which are broken.

**D12. Environment.** One `environment.yml` (Python 3.11, versions pinned); full run
on GUANE via the tool's SLURM scripts.

---

## 3. Script map

| Script | Action |
|---|---|
| 01, 02, 08 | keep (data, upper limit, leakage) |
| 03–07, 09, 10, 13–15 | change to D2 unit and D3 split, rerun |
| 11, 12 | keep protocol, use final model |
| 20, 21 | keep (leakage by entry, Mitchell exact) |
| **30_feature_selection** | new: D4 experiment |
| **31_final_model** | new: trains final RF + sets, writes model for `predict.py` |
| **32_nist_validation** | from `tool/validate_nist.py` |
| **33_hmdb_nonlipids** | from `tool/nonlipid_eval.py` + `data/HMDB/extract_hmdb.py` |
| **34_rules** | from `tool/rules.py` |
| **35_novelty** | from `tool/novelty.py` |
| **36_mz_only** (supplement) | from `tool/mz*.py` |
| 18, 19 | regenerate figures and supplementary tables |
| 16, 23, 25 | not run for the paper |

## 4. Missing inputs

- HMDB 5.0 (`hmdb_metabolites.zip`): not on this machine; download from hmdb.ca and add
  to `MANIFEST.csv`.
- The tool's `lmsd.sdf` and trained models are not on this machine either; not needed
  if D1 is approved (everything is retrained).
- NIST table: present (`tool/data/NIST/nist_srm1950_consensus.csv`).

## 5. Done when

`reproduce_all.sh` runs from raw data to figures on GUANE; every number in the manuscript
appears in `docs/number_provenance.md` with its file; the external sets show no formula
shared with training; the manuscript checklist passes.

## 6. Manuscript outline (RSC benchmark manuscript as base)

1. Introduction — formula-only classification; the gap left by earlier work.
2. Methods — data (D1), unit and split (D2–D3), features (D4), models, sets, validation.
3. Results — 3.1 upper limit · 3.2 leakage · 3.3 features and models · 3.4 lipid vs
   non-lipid · 3.5 external validation (lab, MassBank, NIST) · 3.6 the tool: prediction
   sets, rules, novelty.
4. Discussion — carbon-number descriptors and biosynthesis; m/z-only limits; scope.
5. Data and code availability — GitHub + Zenodo DOI for code, model and results.
