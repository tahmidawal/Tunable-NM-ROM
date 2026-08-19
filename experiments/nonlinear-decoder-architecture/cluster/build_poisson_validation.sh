#!/usr/bin/env bash
# Build a fair control-vs-variant Poisson accuracy/trust-region validation job.
set -euo pipefail

cell="$1"; variant="$2"
[[ "$cell" =~ ^nda_[A-Za-z0-9_]+$ ]] || { echo "invalid cell" >&2; exit 2; }
[[ -f "$variant" ]] || { echo "missing variant checkpoint $variant" >&2; exit 3; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
WT="$(cd "$EXP/../.." && pwd)"
POISSON="$WT/experiments/poisson2d-rom-objective"
MSP="$WT/../2026-08-14-multistage-precision/experiments/multistage-precision"
control="$WT/experiments/cost-to-tolerance/ckpt_poisson/autodec_K16_N64_hbc_stages.pkl"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
STAGE="$HERE/stage/$cell"
[[ -f "$control" ]] || { echo "missing control checkpoint $control" >&2; exit 3; }
[[ ! -e "$STAGE" ]] || { echo "stage exists: $STAGE" >&2; exit 4; }

mkdir -p "$STAGE/code/deps/nonlinear-decoder-architecture" "$STAGE/code/deps" \
         "$STAGE/ckpt" "$STAGE/out" "$STAGE/logs"
cp "$POISSON/pro_common.py" "$POISSON/pro_colloc.py" "$STAGE/code/"
cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" "$STAGE/code/deps/"
cp "$EXP/nda_arch.py" "$STAGE/code/deps/nonlinear-decoder-architecture/"
cp "$EXP/nda_poisson_eval.py" "$STAGE/code/"
cp "$control" "$STAGE/ckpt/control.pkl"
cp "$variant" "$STAGE/ckpt/variant.pkl"
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
export N_TEST=16 GN_ITERS=60 OBJECTIVES=weak_a1_M128 MS=512 SCHEMES=full,nnls INITS=nearest
export EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_FIXED_SNAPS=1
for model in control variant; do
  for tr in 0 1; do
    TR_FACTOR=\$tr \$PY -u nda_poisson_eval.py ../ckpt/\$model.pkl ../out/\${model}_tr\${tr}.json
  done
done
\$PY - <<'PY'
import glob, json
p = glob.glob('../out/*.json')
assert len(p) == 4
for f in p:
    d = json.load(open(f))
    assert d['complete'] and d['manifest']['backend'] == 'gpu' and len(d['rows']) == 2
PY
echo ALL-DONE
EOF
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"
