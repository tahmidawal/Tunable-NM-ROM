#!/bin/bash
#SBATCH --job-name=osc-test64-close
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/cache" TORCH_HOME="$PWD/cache/torch" TMPDIR="$PWD/tmp" MPLCONFIGDIR="$PWD/cache/matplotlib" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor" TRITON_CACHE_DIR="$PWD/cache/triton" CUDA_CACHE_PATH="$PWD/cache/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
test ! -e results && test ! -e OPENED
sha256sum -c SOURCE.sha256
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' and jax.config.jax_enable_x64 else 42)"
"$PY" -c "import torch; p=torch.cuda.get_device_properties(0); print(torch.__version__,p); assert 'A100' in p.name and p.total_memory >= 75 * 1024**3"
"$PY" - <<'PY'
import hashlib, json
from pathlib import Path
s=json.loads(Path("selection.json").read_text())
assert not s["test_cohort_seen"]
for v in s["runs"].values():
    with (Path(v["path"])/v["checkpoint"]).open("rb") as f:
        assert hashlib.file_digest(f,"sha256").hexdigest()==v["best_sha256"]
for fam,sel in s["selection"].items():
    cand={k:v["validation"]["mean"] for k,v in s["runs"].items() if v["budget"]=="final" and v["family"]==fam}
    assert cand==sel["candidates"] and min(cand,key=cand.get)==sel["selected"]
PY
date -u --iso-8601=seconds > OPENED
"$PY" code/gen_data.py test --mesh 256 --out truth
"$PY" code/evaluate.py --manifest selection.json --cohort-dir truth --cohort test64 --out-dir results --skip-1024
sha256sum truth/* > TRUTH.sha256
sha256sum results/*.json > RESULTS.sha256
echo EVALUATION-COMPLETE
