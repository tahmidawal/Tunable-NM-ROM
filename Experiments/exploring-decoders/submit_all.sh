#!/usr/bin/env bash
# Submit the two full-training jobs (siren + xattn). Run after smoke passes.
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p runs

SIREN_JOB=$(sbatch --parsable run_experiment.slurm poisson2d_siren)
XATTN_JOB=$(sbatch --parsable run_experiment.slurm poisson2d_xattn)

echo "SIREN job: $SIREN_JOB"
echo "XATTN job: $XATTN_JOB"
echo "$SIREN_JOB siren" >> runs/submitted_jobs.txt
echo "$XATTN_JOB xattn" >> runs/submitted_jobs.txt
