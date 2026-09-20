#!/bin/bash
#SBATCH --job-name=bf_bfb01
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/bankfloor_20260920/bfb01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/bankfloor_20260920/bfb01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/bankfloor_20260920/bfb01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/bank-floor:$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/b-head-train:$TASK_ROOT/experiments/multiresolution-poisson:$TASK_ROOT/experiments/cost-to-tolerance:$TASK_ROOT/experiments/wave2d-rom-latent-stepping/deps/multistage-precision"
"$PY" experiments/bank-floor/bf_rep.py --config experiments/bank-floor/config-rep-burgers.json --incumbent experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
