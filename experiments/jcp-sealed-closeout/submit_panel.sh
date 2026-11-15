#!/usr/bin/env bash
set -euo pipefail
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
cd "$(dirname "$0")"
task=$1
cfg=$2
gpu=$3
mem=$4
mins=$5
[[ $task =~ ^[a-zA-Z0-9_-]+$ && $cfg =~ ^(pilot-)?d[23]m[0-9]+$ && $mem =~ ^[0-9]+G$ && $mins =~ ^[0-9]+$ ]]
case "$gpu" in
  h200) gres=gpu:h200:1; constraint="" ;;
  h100) gres=gpu:h100:1; constraint="" ;;
  a100) gres=gpu:a100:1; constraint="#SBATCH --constraint=a100-80G" ;;
  *) exit 2 ;;
esac
namespace=/cluster/tufts/paralab/tawal01/jcp-sealed-closeout-20261115
job=$namespace/$task
base=$namespace/base/experiments/burgers-quadrature-plateau
commit=$(git rev-parse HEAD)
root=$(git rev-parse --show-toplevel)
git diff --quiet HEAD -- . || { echo "uncommitted source"; exit 2; }
review_hash=""
if [[ "$cfg" != pilot-* ]]; then review_hash=$(/home/tahmid/Dev/.venv/bin/python stage_approval.py); fi
ssh tufts-login 'squeue -u "$USER" -o "%i|%j|%T"'
ssh tufts-login "mkdir -m 700 '$job' && mkdir '$job/cache' '$job/tmp'"
git -C "$root" archive "$commit" experiments/jcp-sealed-closeout | ssh tufts-login "tar xf - -C '$job'"
f=$(mktemp /dev/shm/jcp-sealed-closeout/slurm-XXXXXXXX)
trap 'rm -f "$f"' EXIT
cat > "$f" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=jcs_$task
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
export JCP_APPROVED_REVIEW_SHA256=$review_hash
export JCP_OPEN_REGISTRY=/cluster/tufts/paralab/tawal01/jcp-opening-records/3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0
sha256sum -c SOURCE.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' and jax.config.jax_enable_x64 and str(jax.config.jax_default_matmul_precision)=='highest' else 42)"
"\$PY" experiments/jcp-sealed-closeout/panel.py --base $base --config experiments/jcp-sealed-closeout/configs/$cfg.json --out output
bash experiments/jcp-sealed-closeout/witness_audit.sh experiments/jcp-sealed-closeout/configs/$cfg.json $base output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
EOF
scp -q "$f" "tufts-login:$job/run.sbatch"
git -C "$root" ls-tree --full-tree -r --name-only "$commit" experiments/jcp-sealed-closeout | while IFS= read -r path; do
  h=$(git -C "$root" show "$commit:$path" | sha256sum | cut -d' ' -f1)
  printf '%s  %s\n' "$h" "$path"
done | ssh tufts-login "cat > '$job/SOURCE.sha256'"
ssh tufts-login "cd '$job' && sha256sum -c SOURCE.sha256 --quiet && sbatch run.sbatch"
ssh tufts-login 'squeue -u "$USER" -o "%i|%j|%T"'
