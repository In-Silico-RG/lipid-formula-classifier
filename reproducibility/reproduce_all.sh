#!/usr/bin/env bash
# Reproduces the WHOLE methodology, from the raw data to the figures and the supplement.
# Stops at the first failure. Logs the duration of each script in reproducibility/00_reproduction.log.
# Uses ONE single version of LIPID MAPS (extended, September 2026):
# 08_ceiling_extended.py goes first because it writes data/lmsd_extended.csv. Script 21 (the
# longest in the original design, ~10 h) goes after the others. Do NOT run in parallel: with two heavy processes at once
# (load 26 on 12 cores) 05_models and 21 made no progress in 85 minutes.
# (16_mona_stream.py is not included: MoNA is not used in the manuscript and its licence is unverified.)
set -u
cd "$(dirname "$0")/.."
LOG=reproducibility/00_reproduction.log
echo "Full reproduction — start $(date '+%F %T') — commit $(git rev-parse --short HEAD)" > "$LOG"
correr() {
  local s=$1 t0; t0=$(date +%s)
  echo "[$(date '+%T')] $s ..." >> "$LOG"
  if ! conda run -n lipid_p python "scripts/$s.py" > "reproducibility/output_$s.txt" 2>&1; then
    echo "[$(date '+%T')] FAILURE in $s (see reproducibility/output_$s.txt)" >> "$LOG"; return 1
  fi
  echo "[$(date '+%T')] $s OK in $(( $(date +%s) - t0 )) s" >> "$LOG"
}
for s in 08_ceiling_extended 01_normalize 02_ceiling 03_baselines 04_representations; do correr $s || exit 1; done
for s in 05_models 07_multilabel 09_networks 10_double_evaluation \
         22_extract_spectra_formulas 11_external_validation 12_massbank_validation 13_conformal \
         14_phase5_evaluation 15_phase6_adversarial 17_structure_oracle 20_entry_lookup \
         21_r1_mitchell_exact; do
  correr $s || exit 1
done
# 06 (about 6 h) runs last so that a failure anywhere else shows up early
# 23 streams the 9.46 GB MoNA MSP out of the zip; it feeds 25 (MS/MS contribution)
t0=$(date +%s); echo "[$(date '+%T')] 23_extract_lipid_spectra ..." >> "$LOG"
if ! unzip -p data/spectra/MoNA-LCMSMS-experimental.msp.zip | conda run -n lipid_p python scripts/23_extract_lipid_spectra.py > reproducibility/output_23_extract_lipid_spectra.txt 2>&1; then
  echo "[$(date '+%T')] FAILURE in 23_extract_lipid_spectra (see reproducibility/output_23_extract_lipid_spectra.txt)" >> "$LOG"; exit 1
fi
echo "[$(date '+%T')] 23_extract_lipid_spectra OK in $(( $(date +%s) - t0 )) s" >> "$LOG"
for s in 24_extra_variables 24b_extra_variables_ablation 25_ms2_contribution 26_r5_validation 27_r5_adversarial 28_sensitivity \
         29_class_subclass_per_molecule 06_hyperparameters 18_manuscript_figures 19_supplementary; do correr $s || exit 1; done
echo "Full reproduction — end $(date '+%F %T')" >> "$LOG"
