#!/bin/bash
#SBATCH --job-name=bcmp_p2048b
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=30:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/bcmp_20260923/p2048b/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/bcmp_20260923/p2048b/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/bcmp_20260923/p2048b
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
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/b-panel/speed:$TASK_ROOT/experiments/quadratic-manifold:$TASK_ROOT/experiments/burgers-compare-hires/lib:$TASK_ROOT/experiments/burgers-compare-hires"
cd experiments/burgers-compare-hires
"$PY" cmp.py --config config-2048.json --checkpoint "$TASK_ROOT/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl" --inputs "$TASK_ROOT/experiments/b-panel/inputs" --out output || echo "CMP FAILED (operators and audit still run)"
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda', ok, torch.cuda.get_device_name() if ok else None, flush=True); sys.exit(0 if ok else 42)"
mkdir -p output/optiming
"$PY" opdata.py --mesh 2048 --out data && mkdir -p logs && "$PY" ops/worker.py specs/train-fno-injob.json || echo "IN-JOB TRAINING FAILED"
"$PY" cluster/late_ops.py operators-2048.json "$PWD" fno-large 60 output/operators-injob.json || cp operators-2048.json output/operators-injob.json
rm -rf data
[ -f opckpt/fno-large.pt ] && "$PY" ops/optime.py --checkpoint opckpt/fno-large.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name fno-large --role "retrained at 2048^2 (in this allocation)" --repetitions 5 --burn-in 20 || echo "IN-JOB OPERATOR MISSING OR FAILED fno-large"
"$PY" cluster/late_ops.py output/operators-injob.json /cluster/tufts/paralab/tawal01/bcmp_20260923/t2048s/experiments/burgers-compare-hires don-small 28800 output/operators-runtime.json || cp output/operators-injob.json output/operators-runtime.json
[ -f opckpt/don-small.pt ] && "$PY" ops/optime.py --checkpoint opckpt/don-small.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name don-small --role "retrained at 2048^2 (late, t2048s)" --repetitions 5 --burn-in 20 || echo "LATE OPERATOR MISSING OR FAILED don-small"
"$PY" ops/optime.py --checkpoint opckpt/unet-refine.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name unet-refine --role "the 256^2 panel's validation-selected unet configuration, retrained at 2048^2" --repetitions 5 --burn-in 20 || echo "OPERATOR FAILED unet-refine"
"$PY" ops/optime.py --checkpoint opckpt/tsol-refine.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name tsol-refine --role "the 256^2 panel's validation-selected transolver configuration, retrained at 2048^2" --repetitions 5 --burn-in 20 || echo "OPERATOR FAILED tsol-refine"
"$PY" ops/optime.py --checkpoint opckpt/fno-large-256.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name fno-large-256 --role "the published 256^2-trained fno-large checkpoint, evaluated zero-shot at this mesh (FNO is discretisation-agnostic); NOT trained at this mesh" --repetitions 5 --burn-in 20 || echo "OPERATOR FAILED fno-large-256"
"$PY" audit_cmp.py output --operators output/operators-runtime.json --out output/audit-remote.json || echo "REMOTE AUDIT FAILED"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
