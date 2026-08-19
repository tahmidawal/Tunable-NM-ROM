#!/usr/bin/env bash
# Build a Burgers weak-mode/EQ ladder for one already-trained decoder.
set -euo pipefail

cell="$1"; checkpoint="$2"
[[ "$cell" =~ ^nda_[A-Za-z0-9_]+$ ]] || { echo "invalid cell" >&2; exit 2; }
[[ -f "$checkpoint" ]] || { echo "missing checkpoint $checkpoint" >&2; exit 3; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
WT="$(cd "$EXP/../.." && pwd)"
BURGERS="$WT/experiments/burgers2d-rom-latent-stepping"
MSP="$WT/../2026-08-14-multistage-precision/experiments/multistage-precision"
BF="$WT/../2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
STAGE="$HERE/stage/$cell"
[[ ! -e "$STAGE" ]] || { echo "stage exists: $STAGE" >&2; exit 4; }

mkdir -p "$STAGE/code/followup" "$STAGE/code/deps/nonlinear-decoder-architecture" \
         "$STAGE/code/deps/multistage-precision" "$STAGE/code/deps/burgers2d-coord-rom" \
         "$STAGE/ckpt" "$STAGE/out" "$STAGE/logs"
cp "$BURGERS"/blat_*.py "$STAGE/code/"
cp "$BURGERS"/followup/fu_*.py "$STAGE/code/followup/"
cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" "$STAGE/code/deps/multistage-precision/"
cp "$BF/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$EXP/nda_arch.py" "$STAGE/code/deps/nonlinear-decoder-architecture/"
cp "$EXP/nda_burgers_eval.py" "$STAGE/code/"
cp "$checkpoint" "$STAGE/ckpt/variant.pkl"
commit="$(git -C "$WT" rev-parse HEAD)"
cat > "$STAGE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o $REMOTE_ROOT/$cell/logs/%j.out
#SBATCH -e $REMOTE_ROOT/$cell/logs/%j.err
set -euo pipefail
cd "$REMOTE_ROOT/$cell"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=$PY
echo "cell=$cell commit=$commit host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
\$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code
export N_TEST=16 FLOOR_BUDGET=60 GN_BUDGET=30 GN_TOL=1e-9 IC_BUDGET=100
export VARIANTS=lspg:full:weak96,lspg:eq384:weak96,lspg:full:weak128,lspg:eq512:weak128
export POD_KS=16 POD_VARIANTS= DO_TIMING=0 TR_FACTOR=0
\$PY -u nda_burgers_eval.py ../ckpt/variant.pkl ../out
\$PY - <<'PY'
import glob, json
p = glob.glob('../out/*.json')
assert len(p) == 1
d = json.load(open(p[0]))
want = {'lspg:full:weak96', 'lspg:eq384:weak96',
        'lspg:full:weak128', 'lspg:eq512:weak128'}
assert d['backend'] == 'gpu' and set(d['rom']) == want
PY
echo ALL-DONE
EOF
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"
