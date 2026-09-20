#!/bin/bash
#SBATCH --job-name=ctol_paper_b3d007
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d007/slurm.%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d007/slurm.%j.err
set -uo pipefail
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export SOURCE_COMMIT=5eb99849c062f72c5dd9f6422f30928a14a0301d
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
cd /cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d007/code
sha256sum -c SOURCE.sha256 || exit 43
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit $?
mkdir -p ../out ../training
finalize() {
    exit_code=$?
    printf '%s\n' "$exit_code" > ../EXIT_CODE
    date -u +%FT%TZ > ../FINISHED_UTC
    (cd .. && find code out training -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256)
}
trap finalize EXIT
date -u +%FT%TZ > ../STARTED_UTC
$PY -u final_campaign.py --config config-final.json --out ../out > ../workflow.log 2>&1
exit $?
