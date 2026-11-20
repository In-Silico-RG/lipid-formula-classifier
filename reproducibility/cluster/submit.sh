#!/bin/bash
# Submits every line of a jobs file as its own SLURM job.  Run from the repository root:
#   bash reproducibility/cluster/submit.sh reproducibility/cluster/jobs_stage1.txt R2C
#   bash reproducibility/cluster/submit.sh reproducibility/cluster/jobs_stage2.txt R2C XGB
set -e
FILE=$1; export SET=${2:?give the input list, e.g. R2C}; export MODEL=${3:-}
mkdir -p reproducibility/cluster/logs
grep -v '^\s*#' "$FILE" | grep -v '^\s*$' | while read -r cmd; do
  c=$(eval echo "\"$cmd\"")
  sbatch reproducibility/cluster/job.sbatch "$c"
done
