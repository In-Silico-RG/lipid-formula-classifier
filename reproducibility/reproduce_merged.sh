#!/bin/bash
# Merged pipeline, in order (docs/PIPELINE_SPEC.md). Usage: ./reproduce_merged.sh <SET> <MODEL> [jobs]
#   SET   input list chosen by script 30 + AC (R1, T10, T12, R2, R2C, R5)
#   MODEL final model chosen from 40/42 (RF, RF-weighted, XGB, XGB-weighted)
# Scripts that do not depend on SET/MODEL (33 HMDB test set, 47 rules, 48 novelty) run first.
set -u
cd "$(dirname "$0")/../scripts"
SET=$1; MODEL=$2; J=${3:-8}
PY=${PY:-python}          # the python of the lipid_p environment (environment.yml)
run() { echo "[$(date +%T)] start $*"; nice -n 10 $PY "$@" > ../results/run_$(echo "$*" | tr -c 'A-Za-z0-9' _).stdout 2>&1 \
        && echo "[$(date +%T)] OK $1" || echo "[$(date +%T)] FAILED $* (see results/run_*.stdout)"; }
# Exclude non-lipid formulas of known lipid classes (audit of script 54); once per data copy
[ -f ../data/master_before_audit.parquet ] || run 55_exclude_audited.py
run 33_hmdb_testset.py
run 47_rules.py
run 48_novelty.py
run 40_models.py --set=$SET --jobs=$J
run 43_networks.py --set=$SET --jobs=$J
for T in A B C; do for M in RF XGB; do run 42_hyperparameters.py --set=$SET --model=$M --task=$T --jobs=$J; done; done
run 44_external.py --set=$SET --model=$MODEL --jobs=$J
run 45_conformal.py --set=$SET --model=$MODEL
run 46_adversarial.py --set=$SET --model=$MODEL --jobs=$J
run 50_structure.py --set=$SET --model=$MODEL --jobs=$J
run 49_final_model.py --set=$SET --model=$MODEL --jobs=$J
run 41_per_class_two_step.py --set=$SET --model=$MODEL --jobs=$J
run 36_mz_only.py --set=$SET --model=$MODEL --jobs=$J
run 30_summary.py
run 51_error_analysis.py --set=$SET --model=$MODEL
run 52_relatives.py --set=$SET --model=$MODEL
run 53_figures_tables.py --set=$SET --model=$MODEL
echo "[$(date +%T)] ALL DONE"
