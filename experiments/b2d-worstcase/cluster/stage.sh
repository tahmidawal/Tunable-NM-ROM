#!/bin/bash
# b2d-worstcase: stage ONE attempt into runs/<attempt> (git-ignored): the vendored byte-identical t256 tree of the
# iso-error lane, the deployed checkpoint, and this lane's committed files under lane/. The command runs in the job
# directory with PYTHONPATH over the vendored modules and lane/.
#   bash cluster/stage.sh <attempt> <HH:MM:SS> <mem e.g. 128G> '<command, may use $PY, $E, $NS>'
set -euo pipefail
A="$1"; WALL="$2"; MEM="$3"; CMD="$4"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
LANE=$(cd "$(dirname "$0")/.." && pwd); ROOT=$(cd "$LANE/../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/b2d-worstcase-20261114
OUT="$LANE/runs/$A"; [ ! -e "$OUT" ]; mkdir -p "$OUT/logs" "$OUT/lane"
cp -r "$LANE/../iso-error/vendor/experiments" "$OUT/experiments"
CK=experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl
mkdir -p "$OUT/$(dirname $CK)"; cp "$ROOT/$CK" "$OUT/$CK"
echo "18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589  $OUT/$CK" | sha256sum -c --quiet
[ -z "$(git -C "$ROOT" status --porcelain -- "$LANE" | grep -v ' runs/' || true)" ] || { echo "uncommitted lane files"; git -C "$ROOT" status --porcelain -- "$LANE"; exit 5; }
(cd "$LANE" && git ls-files -z -- . | xargs -0 -I{} cp --parents {} "$OUT/lane/")
git -C "$ROOT" rev-parse HEAD > "$OUT/COMMIT.txt"
R="$NS/$A"
cat > "$OUT/run.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=wc_$A
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:${WC_GPU:-a100}:1
#SBATCH --constraint=${WC_CONSTRAINT:-a100-80G}
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=$WALL
#SBATCH --output=$R/logs/%j.out
#SBATCH --error=$R/logs/%j.err
set -euo pipefail
TASK_ROOT=$R
NS=$NS
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="\$TASK_ROOT/cache" TMPDIR="\$TASK_ROOT/tmp"
mkdir -p "\$TMPDIR" "\$XDG_CACHE_HOME"
cd "\$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=\$(cat COMMIT.txt)
echo "host=\$(hostname) source_commit=\$SOURCE_COMMIT job=\$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
E=\$TASK_ROOT/experiments
CK=$CK
ROT0=experiments/burgers-bank-knob/inputs/rotation_R512.npz
export PYTHONPATH="\$TASK_ROOT/lane:\$E/mr-burgers2d:\$E/separable-decoder:\$E/head-ablation:\$E/cheap-corrections:\$E/b-ladder-top:\$E/b-panel/speed:\$E/hires-burgers:\$E/burgers-repanel:\$E/burgers-bank-knob:\$E/burgers2d-speed:\$E/burgers2d-test"
SECONDS=0
$CMD
echo "gpu_seconds=\$SECONDS"
echo ALL-DONE
EOF
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
echo "$OUT -> $R"
