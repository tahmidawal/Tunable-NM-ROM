#!/bin/bash
# fp64-query: stage ONE attempt into runs/<attempt> (git-ignored). Text = b2d-query-speed cluster/stage.sh @ c8bd352aa
# with the lane paths, job-name prefix (f64_) and namespace changed; the checkpoint is read in place from the
# query-speed worktree and SHA-checked. cluster/post-<attempt>.sh (committed) runs after the driver, same job.
#   bash cluster/stage.sh <attempt> <HH:MM:SS>
set -euo pipefail
A="$1"; WALL="$2"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
LANE=$(cd "$(dirname "$0")/.." && pwd); ROOT=$(cd "$LANE/../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/fp64-query-20261115
D=experiments/fp64-query
CONFIG=$D/configs/$A.json
OUT="$LANE/runs/$A"; [ ! -e "$OUT" ]; mkdir -p "$OUT/logs"
cp -r "$LANE/vendor/experiments" "$OUT/experiments"
CK=experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl
CKSRC=/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-14-burgers2d-query-speed/$CK
mkdir -p "$OUT/$(dirname $CK)"; cp "$CKSRC" "$OUT/$CK"
echo "18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589  $OUT/$CK" | sha256sum -c --quiet
POST="$D/cluster/post-$A.sh"
for f in $D/src/qsdrive.py $D/src/qsfast.py "$CONFIG" $POST $D/cluster/post-ops.sh $(cd "$ROOT" && ls $D/src/ops/*); do
  [ -z "$(git -C "$ROOT" status --porcelain -- "$f")" ] || { echo "uncommitted: $f"; exit 5; }
  mkdir -p "$OUT/$(dirname $f)"; cp "$ROOT/$f" "$OUT/$f"
done
git -C "$ROOT" rev-parse HEAD > "$OUT/COMMIT.txt"
cp "$LANE/vendor/COMMIT-t256.txt" "$OUT/VENDOR_COMMIT.txt"
R="$NS/$A"
cat > "$OUT/run.sbatch" <<EOT
#!/bin/bash
#SBATCH --job-name=f64_$A
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=$WALL
#SBATCH --output=$R/logs/%j.out
#SBATCH --error=$R/logs/%j.err
set -euo pipefail
TASK_ROOT=$R
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
echo "host=\$(hostname) source_commit=\$SOURCE_COMMIT vendor_commit=\$(cat VENDOR_COMMIT.txt)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
E=\$TASK_ROOT/experiments
export PYTHONPATH="\$E/mr-burgers2d:\$E/separable-decoder:\$E/head-ablation:\$E/cheap-corrections:\$E/b-ladder-top:\$E/b-panel/speed:\$E/hires-burgers:\$E/burgers-repanel:\$E/burgers-bank-knob:\$E/burgers2d-speed:\$E/burgers2d-test"
"\$PY" $D/src/qsdrive.py --config $CONFIG --checkpoint $CK \\
  --rotation experiments/burgers-bank-knob/inputs/rotation_R512.npz \\
  --inputs experiments/b-panel/inputs --out output
export PY; bash $POST || echo "POST-STEP-FAILED rc=\$?"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
EOT
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
echo "$OUT -> $R"
