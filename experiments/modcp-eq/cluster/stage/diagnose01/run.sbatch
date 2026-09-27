#!/bin/bash
#SBATCH --job-name=ctol_modcp_b2d_diagnose01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/diagnose01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/diagnose01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/diagnose01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export COMMIT=$(cat COMMIT.txt)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" code/tests/check_burgers.py
mkdir -p in
cp "/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/validation01/out/fields/validation_cp_cap10_dt0.00125_q4_tol0.0001_L256_case3.npz" "in/validation_cp_cap10_dt0.00125_q4_tol0.0001_L256_case3.npz"
cp "/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/validation01/out/fields/validation_modcp_cap10_dt0.00125_q4_tol0.0001_L256_case3.npz" "in/validation_modcp_cap10_dt0.00125_q4_tol0.0001_L256_case3.npz"
cp "/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/validation01/out/fields/validation_film_cap10_dt0.00125_q4_tol0.0001_L256_case3.npz" "in/validation_film_cap10_dt0.00125_q4_tol0.0001_L256_case3.npz"
cp "/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/validation01/out/fields/validation_film_cap10_dt0.00125_q4_tol0.0001_L256_case9.npz" "in/validation_film_cap10_dt0.00125_q4_tol0.0001_L256_case9.npz"
sha256sum in/*.npz > out/INPUTS.sha256
"$PY" code/burgers/diagnose_initial.py --fields in --checkpoints out/checkpoints --out out/diagnosis.json
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
printf 'ALL-DONE\n'
