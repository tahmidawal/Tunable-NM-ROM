#!/usr/bin/env bash
# spectral-pilot A6 post-hoc control AB: Part-B descriptors on the box, Part-A pipeline. PILOT.
set -u
cd "$(dirname "$0")"
PY=/home/tahmid/Dev/.venv/bin/python
export JAX_PLATFORMS=cpu
for step in "gen_a.py AB --threads 10" "pad_gate.py AB" "diag_a.py AB" "fom_a.py AB --threads 10" "rom_a.py AB"; do
  echo "=== $step $(date +%T)"; timeout 1800 $PY -u $step || echo "FAILED/TIMEOUT: $step"
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for step in "time_a.py AB screen" "time_a.py AB final"; do
  echo "=== $step $(date +%T)"; timeout 1800 taskset -c 4 $PY -u $step || echo "FAILED/TIMEOUT: $step"
done
echo "=== AB done $(date +%T)"
