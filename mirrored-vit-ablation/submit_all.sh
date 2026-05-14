#!/usr/bin/env bash
# Submit all sweep configs for Heat-3D N=64 and Poisson-3D N=64 to SLURM.
# Captures the submitted job IDs so the status report can list them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/runs"
SUBMITTED="$ROOT/runs/submitted_$(date +%Y%m%d_%H%M%S).txt"

submit_one() {
    local pkg="$1"
    local config="$2"
    local script="$ROOT/$pkg/scripts/run_experiment.slurm"
    local cwd="$ROOT/$pkg"
    cd "$cwd"
    mkdir -p runs
    local jobid
    jobid=$(sbatch --parsable --job-name="${pkg}_${config}" "$script" "$config" all)
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
