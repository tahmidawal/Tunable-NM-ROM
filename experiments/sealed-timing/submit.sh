#!/usr/bin/env bash
# Submit one mesh: bash submit.sh <dim> <mesh> <a100|h200> <mem> <minutes>
set -euo pipefail
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
cd "$(dirname "$0")"
dim=$1 mesh=$2 gpu=$3 mem=$4 mins=$5
[[ $dim =~ ^[23]$ && $mesh =~ ^[0-9]+$ && $mem =~ ^[0-9]+G$ && $mins =~ ^[0-9]+$ ]]
case "$gpu" in
  h200) gres=gpu:h200:1; constraint="" ;;
  a100) gres=gpu:a100:1; constraint="#SBATCH --constraint=a100-80G" ;;
  *) exit 2 ;;
esac
task=t${dim}d${mesh}
ns=/cluster/tufts/paralab/tawal01/sealed-timing-20261115
job=$ns/$task
base=$ns/base/experiments/burgers-quadrature-plateau
commit=$(git rev-parse HEAD)
root=$(git rev-parse --show-toplevel)
git diff --quiet HEAD -- . || { echo "uncommitted source"; exit 2; }
ssh tufts-login 'squeue -u "$USER" -o "%i|%j|%T"'
ssh tufts-login "test ! -e '$job' && mkdir -m 700 '$job' && mkdir '$job/cache' '$job/tmp'"
git -C "$root" archive "$commit" experiments/sealed-timing | ssh tufts-login "tar xf - -C '$job'"
f=$(mktemp)
trap 'rm -f "$f"' EXIT
cat > "$f" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=stm_$task
#SBATCH --partition=gpu
#SBATCH --gres=$gres
$constraint
#SBATCH --cpus-per-task=8
#SBATCH --mem=$mem
#SBATCH --time=$mins
#SBATCH --chdir=$job
#SBATCH --output=$job/%j.out
#SBATCH --error=$job/%j.err
set -euo pipefail
umask 077
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 XDG_CACHE_HOME=$job/cache TMPDIR=$job/tmp PYTHONDONTWRITEBYTECODE=1
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store SOURCE_COMMIT=$commit
unset JCP_OPEN_REGISTRY JCP_APPROVED_REVIEW_SHA256
mkdir -p output
nvidia-smi --query-gpu=name,uuid,memory.total,clocks.max.sm,clocks.max.mem --format=csv,noheader | tee output/gpu-start.txt
nvidia-smi >> output/gpu-start.txt
squeue -w "\$(hostname -s)" -o "%i|%u|%j|%b|%C" > output/node-occupancy-start.txt || true
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' and jax.config.jax_enable_x64 and str(jax.config.jax_default_matmul_precision)=='highest' else 42)"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,clocks_throttle_reasons.active --format=csv,noheader -l 1 > output/gpu-trace.csv &
SMI=\$!
"\$PY" experiments/sealed-timing/timing.py --base $base --dim $dim --mesh $mesh --out output
kill \$SMI || true
squeue -w "\$(hostname -s)" -o "%i|%u|%j|%b|%C" > output/node-occupancy-end.txt || true
rm -rf output/fields
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
EOF
scp -q "$f" "tufts-login:$job/run.sbatch"
ssh tufts-login "cd '$job' && sbatch --parsable run.sbatch"
ssh tufts-login 'squeue -u "$USER" -o "%i|%j|%T"'
