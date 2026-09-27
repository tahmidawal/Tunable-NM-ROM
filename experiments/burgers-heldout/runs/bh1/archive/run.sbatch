#!/bin/bash
#SBATCH --job-name=bh_bh1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/bheld_20260921/bh1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/bheld_20260921/bh1/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/bheld_20260921/bh1
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
EXPPATH="$TASK_ROOT/exp/experiments/mr-burgers2d:$TASK_ROOT/exp/experiments/separable-decoder:$TASK_ROOT/exp/experiments/head-ablation:$TASK_ROOT/exp/experiments/cheap-corrections:$TASK_ROOT/exp/experiments/b-ladder-top:$TASK_ROOT/exp/experiments/b-panel/speed:$TASK_ROOT/exp/experiments/hires-burgers:$TASK_ROOT/exp/experiments/burgers-heldout"

# ---- 1. seed checkpoint: cat1024 exactly, as ONE separable parameter set (R = 1024)
PYTHONPATH="$EXPPATH" "$PY" exp/experiments/burgers-heldout/bh_compress.py seed1024 --blocks in/burgers2d_cat1024.pkl \
  --incumbent in/sep_hfit_dense_mid_N256_dense.pkl --out in/cat1024_seed.pkl
# ---- 2. the incumbent's extraction, unchanged, on cat1024 (131072 states of 4608 trajectories)
( cd code && env N=256 K=16 R=1024 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=1 \
  EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \
  CKPT="$TASK_ROOT/in/cat1024_seed.pkl" OUT_PREFIX="$TASK_ROOT/output/x1024_" "$PY" sep_coeff_extract.py )
# ---- 3. POD inside the cat1024 span, floors on dev6/sel32, pre-registered selection, R = 512 seed
PYTHONPATH="$EXPPATH" "$PY" exp/experiments/burgers-heldout/bh_compress.py compress --npz output/x1024_sep_coeff_N256_K16_R1024.npz \
  --blocks in/burgers2d_cat1024.pkl --incumbent in/sep_hfit_dense_mid_N256_dense.pkl --out output/compress
# ---- 4. the incumbent's extraction, unchanged, on cpod512
( cd code && env N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=1 \
  EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \
  CKPT="$TASK_ROOT/output/compress/cpod512_seed.pkl" OUT_PREFIX="$TASK_ROOT/output/x512_" "$PY" sep_coeff_extract.py )
# ---- 5. the incumbent's head fit, unchanged recipe (arm mid, from scratch), emitted as the new model
( cd code && env NPZ="$TASK_ROOT/output/x512_sep_coeff_N256_K16_R512.npz" \
  CKPT="$TASK_ROOT/output/compress/cpod512_seed.pkl" OUT="$TASK_ROOT/output/hfit_bh.json" \
  ARMS=mid STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500 ORACLE_ITERS=150 CODEDIAG_N=2048 \
  CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=0 EMIT=mid EMIT_PATH="$TASK_ROOT/output/bh_model.pkl" \
  "$PY" sep_hfit_run.py )
# ---- 6. correction directions by the incumbent's rule (qtd02 configuration)
PYTHONPATH="$EXPPATH" "$PY" exp/experiments/burgers-heldout/bh_dirs.py --checkpoint output/bh_model.pkl --out output/dirs
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
