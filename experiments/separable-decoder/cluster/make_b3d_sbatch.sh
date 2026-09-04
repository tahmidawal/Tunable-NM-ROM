#!/bin/bash
# Emit an sbatch file for one Burgers-3D job.
# Usage: ./make_b3d_sbatch.sh <jobname> <gpu: a100|h200> <mem> <time> <script> "<env assignments>"
set -euo pipefail
JOB=$1; GPU=$2; MEM=$3; TIME=$4; SCRIPT=$5; ENVS=$6
ROOT=/cluster/tufts/paralab/tawal01/b3dtensor/$JOB
CONS=""; [ "$GPU" = "a100" ] && CONS="#SBATCH --constraint=a100-80G"
cat <<SB
#!/bin/bash
#SBATCH --job-name=b3d_$JOB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:$GPU:1
$CONS
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$ROOT/logs/%j.out
#SBATCH --error=$ROOT/logs/%j.err
set -euo pipefail
ROOT=$ROOT
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="\$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest
echo "host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "\$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
export COMMIT=\$(cat "\$ROOT/COMMIT.txt")
echo "commit=\$COMMIT"
cd "\$ROOT/code"
env TABLE_DIR="\$ROOT/code/runs/b3dtensor/tables" $ENVS "\$PY" $SCRIPT
cd "\$ROOT/out" && sha256sum \$(ls *.json *.pkl *.npz 2>/dev/null) > RESULTS.sha256
echo ALL-DONE
SB
