#!/usr/bin/env bash
# Checks that 17_structure_oracle.py gives the same result with different Python hash randomisation.
cd "$(dirname "$0")/.."
for hash_seed in 1 2; do
  PYTHONHASHSEED=$hash_seed conda run -n lipid_p python scripts/17_structure_oracle.py > /dev/null 2>&1 || exit 1
  cp results/17_structure_oracle.csv reproducibility/17_hash$hash_seed.csv
  cp results/17_structure_oracle.log reproducibility/17_hash$hash_seed.log
done
cmp reproducibility/17_hash1.csv reproducibility/17_hash2.csv && echo "17: IDENTICAL with PYTHONHASHSEED 1 y 2"
