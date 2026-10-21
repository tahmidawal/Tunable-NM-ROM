#!/usr/bin/env bash
# spectral-pilot: all timing on ONE core, single-thread flags, nothing else running (DESIGN 6, A1.5). PILOT.
set -u
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
PY=/home/tahmid/Dev/.venv/bin/python
for step in "time_a.py A1 screen" "time_a.py A1 final" "time_a.py A2 screen" "time_a.py A2 final" \
            "time_a.py A3 screen" "time_a.py A3 final" "time_b.py screen" "time_b.py final"; do
  echo "=== $step $(date +%T)"
  timeout 1800 taskset -c 4 $PY -u $step || echo "FAILED/TIMEOUT: $step"
done
echo "=== all timing done $(date +%T)"
