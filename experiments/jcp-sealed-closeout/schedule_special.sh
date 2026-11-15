#!/usr/bin/env bash
# Stage and verify a dedicated registry publisher or one native POD evaluation.
set -euo pipefail
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
cd "$(dirname "$0")"
task=$1
kind=$2
[[ "$task" =~ ^[a-z0-9_-]+$ && "$kind" =~ ^(opening|pod2|pod3)$ ]]
n=/cluster/tufts/paralab/tawal01/jcp-sealed-closeout-20261115
remote=$n/$task
base=$n/base/experiments/burgers-quadrature-plateau
root=$(git rev-parse --show-toplevel)
commit=$(git rev-parse HEAD)
git diff --quiet HEAD -- . || exit 2
test -f F1-REVIEW.json
review_hash=$(/home/tahmid/Dev/.venv/bin/python stage_approval.py)
ssh tufts-login 'squeue -u "$USER" -o "%i|%j|%T"'
ssh tufts-login "mkdir -m 700 '$remote' && mkdir '$remote/tmp' '$remote/cache'"
git -C "$root" archive "$commit" experiments/jcp-sealed-closeout | ssh tufts-login "tar xf - -C '$remote'"
f=$(mktemp /dev/shm/jcp-sealed-closeout/special-XXXXXXXX)
trap 'rm -f "$f"' EXIT
cat > "$f" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=jcs_$task
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=40
#SBATCH --chdir=$remote
#SBATCH --output=$remote/%j.out
#SBATCH --error=$remote/%j.err
set -euo pipefail
umask 077
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
L=$remote/experiments/jcp-sealed-closeout
export JCP_APPROVED_REVIEW_SHA256=$review_hash
export JCP_OPEN_REGISTRY=/cluster/tufts/paralab/tawal01/jcp-opening-records/3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_MEM_FRACTION=0.88 OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6
export TMPDIR=$remote/tmp XDG_CACHE_HOME=$remote/cache SOURCE_COMMIT=$commit PYTHONDONTWRITEBYTECODE=1
sha256sum -c SOURCE.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' and jax.config.jax_enable_x64 and str(jax.config.jax_default_matmul_precision)=='highest' else 42)"
EOF
if [[ "$kind" == opening ]]; then
cat >> "$f" <<EOF
mkdir output
"\$PY" "\$L/open_sealed.py" > output/COMPLETE.json
EOF
else
d=${kind:3:1}
cat >> "$f" <<EOF
"\$PY" "\$L/pod_inputs.py" "$d" "$base" "\$L/configs/$kind.json" "$remote/tmp/input.npz"
"\$PY" "\$L/pod_panel.py" --config "\$L/configs/$kind.json" --base "$base" --pod "$n/pod-source" --inputs "$remote/tmp/input.npz" --out "$remote/output"
EOF
fi
cat >> "$f" <<EOF
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
EOF
scp -q "$f" "tufts-login:$remote/run.sbatch"
git -C "$root" ls-tree --full-tree -r --name-only "$commit" experiments/jcp-sealed-closeout | while IFS= read -r path; do
 h=$(git -C "$root" show "$commit:$path"|sha256sum|cut -d' ' -f1)
 printf '%s  %s\n' "$h" "$path"
done | ssh tufts-login "cat > '$remote/SOURCE.sha256'"
ssh tufts-login "cd '$remote' && sha256sum -c SOURCE.sha256 --quiet && sbatch run.sbatch"
ssh tufts-login 'squeue -u "$USER" -o "%i|%j|%T"'
