#!/bin/bash
# Generate a phase-4 ladder sbatch for one (N, BC): the head/arm/RS that passed W3 are arguments.
# The bank+head cache comes from the SRC_N phase-2 job directory (copied into this job's cache/ at stage time).
# Usage: ./make_ladder_sbatch.sh <N> <BC> <HEAD> <ARM> <RS> <SRC_N>      -> cluster/run_wav2d_ladder_n<N>_<BC>.sbatch
set -euo pipefail
NN=${1:?N}; BCX=${2:?BC}; HEADX=${3:?HEAD}; ARMX=${4:?ARM}; RSX=${5:?RS}; SRCN=${6:?SRC_N}
JOB=ladder_n${NN}_${BCX}
GPU=$([ "$NN" -ge 512 ] && echo h200 || echo a100); MEM=$([ "$NN" -ge 512 ] && echo 200G || echo 96G)
HERE=$(cd "$(dirname "$0")" && pwd)
cat > "$HERE/run_wav2d_${JOB}.sbatch" <<EOS
#!/bin/bash
#SBATCH --job-name=wav2d_${JOB}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:${GPU}:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEM}
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/wav2d/${JOB}/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/wav2d/${JOB}/logs/%j.err
# Wave 2D phase-4 cost ladder at N=${NN} BC=${BCX}: head ${HEADX}, arm ${ARMX}, RS ${RSX}, bank/head from N=${SRCN}.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/wav2d/${JOB}
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="\$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest
echo "host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "\$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
echo "commit=\$(cat \$ROOT/COMMIT.txt)"
cd "\$ROOT/code"
"\$PY" wav2d_ladder.py N=${NN} BC=${BCX} HEAD=${HEADX} ARM=${ARMX} RS=${RSX} SRC_N=${SRCN} R=64 M=64 STEPS=40000 REPS=5 BURN=2 OUT="\$ROOT/out" CACHE="\$ROOT/cache"
cd "\$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
EOS
echo "wrote $HERE/run_wav2d_${JOB}.sbatch"
