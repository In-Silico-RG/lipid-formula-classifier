# Running the long calculations on a cluster

Guide for the person who runs the calculations of the merged paper on a SLURM cluster.
Contact: Aldo F. Combariza (In Silico group, Universidad de Sucre). What each script does and why: [`docs/PIPELINE_SPEC.md`](docs/PIPELINE_SPEC.md).

Everything runs on CPUs (no graphics card). Every job is independent within its stage, so they
can all be submitted at once.

## 1. Get the code and the environment

```bash
git clone https://github.com/In-Silico-RG/lipid-formula-classification.git
cd lipid-formula-classification
conda env create -f environment.yml        # environment "lipid_p": Python 3.11, scikit-learn 1.8.0, XGBoost 3.2.0, PyTorch 2.14 (CPU), RDKit
conda activate lipid_p
python -c "import sklearn, xgboost, torch, rdkit, optuna; print(sklearn.__version__, xgboost.__version__)"   # must print 1.8.0 3.2.0
```

Use exactly these versions: results are compared with runs made with them.

## 2. Get the data

The data are not in the git history (size and licences). They are attached to the private
release `data-v1` of this repository as one file, `lipid-formula-classification_data.tar.gz`
(40 MB). Download it (you need access to the repository), check it and unpack it into `data/`:

```bash
gh release download data-v1 -R In-Silico-RG/lipid-formula-classification
# or download it from https://github.com/In-Silico-RG/lipid-formula-classification/releases/tag/data-v1
sha256sum lipid-formula-classification_data.tar.gz
# must be 0badf53e549011350f1ed811ddd85fc825e473b18efa54e75f613224b8e7495a
tar -xzf lipid-formula-classification_data.tar.gz -C data/
ls data/        # lmsd_extended.csv master.parquet representations.npz n4_adversarial.csv
                # hmdb_test.csv structures_extended.sdf spectra/ NIST/ HMDB/ README.md ...
```

These are derived files (the LIPID MAPS table, the master table of lipids and non-lipids, the
formulas of the measured spectra, the artificial non-lipids and the HMDB test set). They are for
this calculation only: do not redistribute them. How each one is regenerated from the primary
sources is in `data/README.md`.

Quick check that everything is in place (about one minute):

```bash
cd scripts
python -c "import merged_common as mc; d,a=mc.load_entries(); print(len(d), d.hill.nunique())"   # must print 49966 9935
cd ..
```

## 3. Adapt the SLURM job

Edit `reproducibility/cluster/job.sbatch`: set `--partition` and `--account` for your cluster.
Each job asks for 24 CPUs and 60 GB, up to 3 days; lower `--cpus-per-task` if needed and use the
same number in `--jobs=` of the job lists.

## 4. Stage 1 — 12 jobs, all at the same time

A. Combariza tells you the input list (`SET`, for example `R2C`). From the repository root:

```bash
bash reproducibility/cluster/submit.sh reproducibility/cluster/jobs_stage1.txt R2C
squeue -u $USER
```

| Jobs | Script | What it computes | Rough time on 24 CPUs |
|---|---|---|---|
| 3 | `40_models.py` (tasks A, B, C) | baselines, random forest, XGBoost | 1–6 h each (C is the longest) |
| 3 | `43_networks.py` (tasks A, B, C) | multilayer perceptron and convolutional network | 3–12 h each |
| 6 | `42_hyperparameters.py` (RF/XGB × A, B, C) | settings search, nested validation (30 trials × 5 folds) | 3–24 h each (XGB on C is the longest) |

The times are estimates; the slowest job sets the duration of the stage.

## 5. Stage 2 — 7 jobs, after stage 1

A. Combariza chooses the final model (`MODEL`: `RF`, `RF-weighted`, `XGB` or `XGB-weighted`) from
the stage-1 results. Then:

```bash
bash reproducibility/cluster/submit.sh reproducibility/cluster/jobs_stage2.txt R2C XGB
```

| Script | What it computes |
|---|---|
| `44_external.py` | tests on independent data: GNPS, MassBank, NIST SRM 1950 plasma, HMDB |
| `45_conformal.py` | prediction sets (needs `results/40_oof_*` from stage 1) |
| `46_adversarial.py` | artificial non-lipids and the shifted-mass test |
| `50_structure.py` | comparison with a model that sees the structure |
| `49_final_model.py` | the released classifier (`classifier/lipid_classifier.joblib`) |
| `41_per_class_two_step.py` | results per category and per class, confusion table, two-step model for the class |
| `36_mz_only.py` | classification when only the mass is measured (supplement) |

## 5b. Stage 3 — light, after stage 2 (minutes, no SLURM needed)

`reproducibility/cluster/jobs_stage3.txt`: summary of the feature selection (`30_summary.py`),
where the errors fall (`51`), homologous formulas on both sides of the split (`52`), and the
figures and LaTeX tables of the paper (`53`). A. Combariza can also run this stage himself.

## 6. Send the results back

Everything is written to `results/` (and the classifier to `classifier/`). When a stage ends:

```bash
tar -czf results_stage1.tar.gz results/ reproducibility/cluster/logs/
```

and send that file to A. Combariza (or push a branch to the repository, if you have access).
If a job fails, its log is in `reproducibility/cluster/logs/` and the script's own log in
`results/<script>_*.log`; please send both.

## Notes

- No job reads the output of another job of the same stage, so the order inside a stage does not
  matter. Stage 2 needs the `40_oof_*.npz` files of stage 1 in the same `results/` folder.
- The scripts never download anything and never write outside the repository folder.
- `30_feature_selection.py` and `42_hyperparameters.py` can take a long time without printing;
  their logs in `results/` grow as each part finishes.
