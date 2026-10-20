#!/bin/bash
#SBATCH --job-name=jg_smk1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcp-regimes/smk1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcp-regimes/smk1/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcp-regimes/smk1
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
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
cd experiments/jcp-regimes
JAX_PLATFORMS=cpu "$PY" test_regimes.py "$TASK_ROOT/output/unit_gates.json"
echo "STEP refjob configs/smk1-ref-C2.json $(date -u +%FT%TZ)"
"$PY" refjob.py --config configs/smk1-ref-C2.json --out "$TASK_ROOT/output/refjob_smk1-ref-C2"
echo "STEP refjob configs/smk1-ref-dev6.json $(date -u +%FT%TZ)"
"$PY" refjob.py --config configs/smk1-ref-dev6.json --out "$TASK_ROOT/output/refjob_smk1-ref-dev6"
echo "STEP refjob configs/smk1-ref-B2.json $(date -u +%FT%TZ)"
"$PY" refjob.py --config configs/smk1-ref-B2.json --out "$TASK_ROOT/output/refjob_smk1-ref-B2"
echo "STEP refjob configs/smk1-ref-Ap.json $(date -u +%FT%TZ)"
"$PY" refjob.py --config configs/smk1-ref-Ap.json --out "$TASK_ROOT/output/refjob_smk1-ref-Ap"
echo "STEP refslim configs/smk1-slim.json $(date -u +%FT%TZ)"
"$PY" refslim.py --config configs/smk1-slim.json --out "$TASK_ROOT/output/refslim_smk1-slim"
echo "STEP qacc configs/smk1-rom.json $(date -u +%FT%TZ)"
"$PY" qacc.py --config configs/smk1-rom.json --out "$TASK_ROOT/output/qacc_smk1-rom"
echo "STEP fverify configs/smk1-fverify.json $(date -u +%FT%TZ)"
"$PY" fverify.py --config configs/smk1-fverify.json --out "$TASK_ROOT/output/fverify_smk1-fverify"
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
