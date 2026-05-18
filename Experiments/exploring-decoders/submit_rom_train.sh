#!/usr/bin/env bash
# Submit the 3 ROM-faithful retraining jobs.
set -e
cd "$(dirname "$0")"
for CFG in poisson2d_xattn_narrow_cg poisson2d_siren_wide_cg poisson2d_xattn_wide_cg; do
    sbatch run_rom_train.slurm "$CFG"
done
squeue -u "$USER"
