# lipid-formula-classifier

Lipid classification from the molecular formula alone: LIPID MAPS category and main class,
lipid detection, 95 % prediction sets (Mondrian conformal prediction) and a check of whether the
formula resembles any known lipid. Code, results and trained classifier of the paper by
Edwin R. Buelvas, Wilson Castro, Himer Ávila-George and Aldo F. Combariza (RSC Advances,
in preparation), Universidad de Sucre.

## Use the classifier

1. `conda env create -f environment.yml` (Python 3.11, scikit-learn 1.8.0, XGBoost 3.2.0; CPU only).
2. Download `lipid_classifier.joblib` (297 MB) from the
   [v1.0.0 release](https://github.com/In-Silico-RG/lipid-formula-classifier/releases/tag/v1.0.0)
   into `classifier/`. SHA-256:
   `1a6026d51ea1bf62b6308cd1623e2937abd8e2b2455be78d907d2e2448fcae7f`.
3. Then:

```python
from classifier.predict import LipidClassifier
LipidClassifier().predict("C44H82NO8P")
```

It returns, for the category, the main class and lipid detection, the most probable label, its
probability and the 95 % prediction set, plus whether the formula resembles any known lipid.

## Layout

| Folder | Contents |
|---|---|
| `classifier/` | `predict.py` and the feature code it shares with training (`features.py`, `detectors.py`) |
| `scripts/` | the whole pipeline. `01`–`29`: data, upper limit, leakage and benchmark analyses; `merged_common.py` and `30`–`55`: the protocol of the paper (`docs/PIPELINE_SPEC.md`); `49` trains the released classifier |
| `results/` | the log and table written by each script; every number of the paper traces to one of them |
| `figures/` | figures of the paper |
| `data/` | sources are not redistributed (licences, size): `README.md` and `MANIFEST.csv` list them and `download_data.sh` fetches them; `NIST/` and `HMDB/` hold small versioned pieces |
| `reproducibility/` | `reproduce_merged.sh` (pipeline in order), `cluster/` (SLURM jobs); see `RUN_ON_A_CLUSTER.md` |

## Licence

MIT (code, `LICENSE`). Derived datasets, result files (`results/`) and the trained classifier:
Creative Commons Attribution 4.0 (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). The
source databases keep their own licences (`data/README.md`) and are not redistributed.
