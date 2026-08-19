#!/usr/bin/env bash
# Evaluate one frozen Poisson decoder at one paired weak objective / EQ size.
set -euo pipefail

cell="$1"; checkpoint="$2"; M="$3"; m="$4"
[[ "$cell" =~ ^nda_[A-Za-z0-9_]+$ ]] || { echo "invalid cell" >&2; exit 2; }
[[ -f "$checkpoint" ]] || { echo "missing checkpoint $checkpoint" >&2; exit 3; }
[[ "$M" =~ ^[1-9][0-9]*$ && "$m" =~ ^[1-9][0-9]*$ ]] || { echo "invalid M or m" >&2; exit 2; }
(( M > 16 )) || { echo "M must exceed k=16" >&2; exit 2; }
(( m >= 4 * M )) || { echo "m must be at least 4M" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
WT="$(cd "$EXP/../.." && pwd)"
POISSON="$WT/experiments/poisson2d-rom-objective"
MSP="$WT/../2026-08-14-multistage-precision/experiments/multistage-precision"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
STAGE="$HERE/stage/$cell"
[[ ! -e "$STAGE" ]] || { echo "stage exists: $STAGE" >&2; exit 4; }

mkdir -p "$STAGE/code/deps/nonlinear-decoder-architecture" "$STAGE/code/deps" \
         "$STAGE/ckpt" "$STAGE/out" "$STAGE/logs"
cp "$POISSON/pro_common.py" "$POISSON/pro_colloc.py" "$STAGE/code/"
cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" "$STAGE/code/deps/"
cp "$EXP/nda_arch.py" "$STAGE/code/deps/nonlinear-decoder-architecture/"
cp "$EXP/nda_poisson_eval.py" "$STAGE/code/"
cp "$checkpoint" "$STAGE/ckpt/variant.pkl"
commit="$(git -C "$WT" rev-parse HEAD)"
cat > "$STAGE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
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
export N_TEST=16 GN_ITERS=60 OBJECTIVES=weak_a1_M$M MS=$m SCHEMES=full,nnls INITS=nearest
export EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_FIXED_SNAPS=1 TR_FACTOR=0
\$PY -u nda_poisson_eval.py ../ckpt/variant.pkl ../out/rom.json
\$PY - <<'PY'
import json
d = json.load(open('../out/rom.json'))
assert d['complete'] and d['manifest']['backend'] == 'gpu' and len(d['rows']) == 2
assert {r['scheme'] for r in d['rows']} == {'full', 'nnls'}
PY
echo ALL-DONE
EOF
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"
