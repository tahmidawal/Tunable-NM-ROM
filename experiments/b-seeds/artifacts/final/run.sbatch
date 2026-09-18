#!/bin/bash
#SBATCH --job-name=bsd_final
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b_seeds_20260917/final/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b_seeds_20260917/final/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/b_seeds_20260917/final
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT attempt=final started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
LADDER_PATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/q-ridge:$TASK_ROOT/experiments/b-seeds"

# The sealed cohort is opened here and only here (DESIGN.md section 3). Sequential
# invocations of the SAME driver and configuration shape as the development ladders: the
# incumbent, then every seed checkpoint. Each writes its own output directory.
for CK in incumbent seed1 seed2 seed3; do
  echo "SEALED LADDER $CK $(date -Is)"
  case "$CK" in
    incumbent) CKPT=experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl ;;
    *) CKPT="experiments/b-seeds/checkpoints/sep_hfit_$CK.pkl" ;;
  esac
  sha256sum "$CKPT"
  PYTHONPATH="$LADDER_PATH" "$PY" experiments/b-seeds/seeds_run.py \
    --config "experiments/b-seeds/config-sealed-$CK.json" \
    --checkpoint "$CKPT" --out "output/sealed_$CK"
done
echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
