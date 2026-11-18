#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:TERM@120
set -euo pipefail
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" output
manifest() { find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256; }
trap 'manifest; echo RUN_EXIT=TERM; exit 143' TERM
export SOURCE_COMMIT="$(cat COMMIT.txt)"
echo "host=$(hostname) source_commit=$SOURCE_COMMIT job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
sha256sum -c INPUTS.sha256
set +e
"$PY" experiments/jcp-head-mechanism/run_mechanism.py --config experiments/jcp-head-mechanism/configs/pilot.json --out output
RUN_EXIT=$?
set -e
manifest
echo "RUN_EXIT=$RUN_EXIT"
exit "$RUN_EXIT"
