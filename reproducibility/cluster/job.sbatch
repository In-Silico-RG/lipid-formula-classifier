#!/bin/bash
# One calculation of the merged pipeline on a SLURM cluster.
# Usage (from the repository root):  sbatch reproducibility/cluster/job.sbatch "<command>"
# Adjust --partition / --account to your cluster; the scripts use the CPUs given here (--jobs).
#SBATCH --job-name=lipid-formula-classification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
#SBATCH --output=reproducibility/cluster/logs/%x_%j.out
##SBATCH --partition=<your partition>
##SBATCH --account=<your account>
set -e
cd "$SLURM_SUBMIT_DIR"
# the conda environment created from environment.yml (Python 3.11, scikit-learn 1.8.0, XGBoost 3.2.0)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lipid_p
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "start $(date -Is) on $(hostname), $SLURM_CPUS_PER_TASK CPUs: $1"
cd scripts
eval "$1"
echo "end $(date -Is)"
