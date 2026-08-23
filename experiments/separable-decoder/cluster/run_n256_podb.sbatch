#!/bin/bash
#SBATCH --job-name=n256push_podb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/n256_push/podb/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/n256_push/podb/logs/%j.err

# N=256 push: Burgers POD-floor diagnostic (f64 Gram; DIAGNOSTIC ONLY --
# bounds the separable architecture, no POD enters any model).  Decides R /
# multi-bank feasibility for the PRIMARY Burgers 1e-3 round.  Data
# regenerated from seed in-job; the GB10 attempt was killed as too slow.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/n256_push/podb
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
env PDE=burgers N=256 RS=64,128,256,512,1024,2048 MAX_SNAPS=8192 N_TEST_B=8 \
  OUT="$ROOT/out/pod_floor_N256_burgers.json" \
  "$PY" pod_floor_n256.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
