#!/bin/bash
#SBATCH --job-name=t3bt_t32048a
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/t3btest_20260925/t32048a/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/t3btest_20260925/t32048a/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/t3btest_20260925/t32048a
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/b-panel/speed:$TASK_ROOT/experiments/quadratic-manifold:$TASK_ROOT/experiments/burgers-compare-hires/lib:$TASK_ROOT/experiments/burgers-compare-hires:$TASK_ROOT/experiments/t3-burgers-test"
cd experiments/t3-burgers-test
"$PY" t3cmp.py --config config-2048.json --checkpoint "$TASK_ROOT/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl" --inputs "$TASK_ROOT/experiments/b-panel/inputs" --out output || echo "T3CMP FAILED (audit still runs)"
"$PY" audit_t3.py output --out output/audit-remote.json || echo "REMOTE AUDIT FAILED"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
