#!/usr/bin/env bash
# Submit all sweep configs for Heat-3D N=64 and Poisson-3D N=64 to SLURM.
# Captures the submitted job IDs so the status report can list them.
#
# Usage:
#   bash submit_all.sh                 # standalone, no dependency
#   bash submit_all.sh <smoke_jobid>   # gate every training on smoke_jobid completing OK
#
# The second form lets you submit the full sweep RIGHT AFTER the smoke
# test is queued, without waiting for it to finish. SLURM will release
# each training job only if smoke exits 0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/runs"
SUBMITTED="$ROOT/runs/submitted_$(date +%Y%m%d_%H%M%S).txt"

DEP_JOB="${1:-}"
DEP_ARGS=""
if [[ -n "$DEP_JOB" ]]; then
    DEP_ARGS="--dependency=afterok:$DEP_JOB"
    echo "Gating all training jobs on smoke job $DEP_JOB (afterok)"
fi

submit_one() {
    local pkg="$1"
    local config="$2"
    local script="$ROOT/$pkg/scripts/run_experiment.slurm"
    local cwd="$ROOT/$pkg"
    cd "$cwd"
    mkdir -p runs
    local jobid
    jobid=$(sbatch --parsable $DEP_ARGS --job-name="${pkg}_${config}" "$script" "$config" all)
    echo "submitted $pkg/$config -> job $jobid"
    echo "$pkg $config $jobid" >> "$SUBMITTED"
}

for c in poisson3d_n64_A_mirror poisson3d_n64_B_deeper poisson3d_n64_C_wider; do
    submit_one poisson "$c"
done
for c in heat3d_n64_A_mirror heat3d_n64_B_deeper heat3d_n64_C_wider; do
    submit_one heat "$c"
done

echo
echo "Job IDs written to: $SUBMITTED"
echo "Monitor with: squeue -u \$USER"
