#!/bin/bash
#SBATCH --job-name=ctol_paper_b3d005
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d005/slurm.%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d005/slurm.%j.err
set -uo pipefail
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export SOURCE_COMMIT=178c0e83e0e9f553ea956c3456880a34e536a0b7
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
cd /cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d005/code
sha256sum -c SOURCE.sha256 || exit 43
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit $?
mkdir -p ../out
finalize() {
    exit_code=$?
    printf '%s\n' "$exit_code" > ../EXIT_CODE
    date -u +%FT%TZ > ../FINISHED_UTC
    (cd .. && find code out training  -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256)
}
trap finalize EXIT
date -u +%FT%TZ > ../STARTED_UTC
$PY -u confirmation.py --config config-confirmation.json > ../confirmation.log 2>&1; exit $?
$PY -u train.py --config config-confirmation.json --out ../training > ../training.log 2>&1 || exit $?
$PY -u reference_diagnostic.py --config config-confirmation.json --out ../out/reference_screen > ../reference-screen.log 2>&1 || exit $?
$PY -u operator_panel.py --config config-confirmation.json --training ../training --out ../out --mode train > ../operator-training.log 2>&1 || exit $?
$PY -u run.py --config config-confirmation.json --out ../out --checkpoint ../training/checkpoint.pkl > ../driver.log 2>&1
code=$?
if [ "$code" -eq 0 ]; then $PY -u operator_panel.py --config config-confirmation.json --training ../training --out ../out --mode evaluate > ../operator-evaluate.log 2>&1; code=$?; fi
if [ "$code" -eq 0 ]; then
    $PY -u audit.py ../out --checkpoint ../training/checkpoint.pkl > ../audit.log 2>&1
    code=$?
fi



printf '%s\n' "$code" > ../EXIT_CODE
date -u +%FT%TZ > ../FINISHED_UTC
exit "$code"
