#!/bin/bash
#SBATCH --job-name=t2b_t256
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/t2btest_20260925/t256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/t2btest_20260925/t256/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/t2btest_20260925/t256
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
cd experiments/t2-burgers-test
"$PY" tcmp.py --config config-t256.json --checkpoint "$TASK_ROOT/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl" --inputs "$TASK_ROOT/experiments/b-panel/inputs" --out output || echo "CMP FAILED (operators and audit still run)"
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda', ok, torch.cuda.get_device_name() if ok else None, flush=True); sys.exit(0 if ok else 42)"
mkdir -p output/optiming
cp operators-t256.json output/operators-runtime.json
"$PY" ops/optime.py --checkpoint opckpt/fno-large.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name fno-large --role "256^2 checkpoint (selected; source job 3710846)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED fno-large"
"$PY" ops/optime.py --checkpoint opckpt/unet-small.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name unet-small --role "256^2 checkpoint (capacity; source job 3780138)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED unet-small"
"$PY" ops/optime.py --checkpoint opckpt/unet-medium.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name unet-medium --role "256^2 checkpoint (best_validation_worst; source job 3780138)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED unet-medium"
"$PY" ops/optime.py --checkpoint opckpt/unet-large.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name unet-large --role "256^2 checkpoint (capacity; source job 3780138)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED unet-large"
"$PY" ops/optime.py --checkpoint opckpt/unet-refine.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name unet-refine --role "256^2 checkpoint (selected; source job 3780138)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED unet-refine"
"$PY" ops/optime.py --checkpoint opckpt/tsol-small.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name tsol-small --role "256^2 checkpoint (capacity; source job 3780139)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED tsol-small"
"$PY" ops/optime.py --checkpoint opckpt/tsol-medium.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name tsol-medium --role "256^2 checkpoint (capacity; source job 3780139)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED tsol-medium"
"$PY" ops/optime.py --checkpoint opckpt/tsol-large.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name tsol-large --role "256^2 checkpoint (best_validation_worst; source job 3780139)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED tsol-large"
"$PY" ops/optime.py --checkpoint opckpt/tsol-refine.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name tsol-refine --role "256^2 checkpoint (selected; source job 3780139)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED tsol-refine"
"$PY" ops/optime.py --checkpoint opckpt/don-small.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name don-small --role "256^2 checkpoint (selected; source job 4179556)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED don-small"
"$PY" ops/optime.py --checkpoint opckpt/don-medium.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name don-medium --role "256^2 checkpoint (best_validation_worst; source job 4179556)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED don-medium"
"$PY" ops/optime.py --checkpoint opckpt/don-refine.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name don-refine --role "256^2 checkpoint (capacity; source job 4179556)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED don-refine"
"$PY" ops/optime.py --checkpoint opckpt/don-large.pt --index output/opcohort/index.json --out output/optiming --fields output/fields --name don-large --role "256^2 checkpoint (capacity; source job 4179556)" --repetitions 5 --burn-in 20 --sub-points 64 --full-cases 0 || echo "OPERATOR FAILED don-large"
rm -f output/opcohort/*.npz
"$PY" t2audit.py output --operators output/operators-runtime.json --out output/audit-remote.json || echo "REMOTE AUDIT FAILED"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
