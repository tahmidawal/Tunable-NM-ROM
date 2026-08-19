#!/usr/bin/env bash
# Same-GPU, burn-in-controlled end-to-end Poisson comparison at the selected arm.
set -euo pipefail

cell="$1"; variant="$2"; selected_M="${3:-128}"; selected_m="${4:-512}"
[[ "$cell" =~ ^nda_[A-Za-z0-9_]+$ ]] || { echo "invalid cell" >&2; exit 2; }
[[ -f "$variant" ]] || { echo "missing variant checkpoint $variant" >&2; exit 3; }
[[ "$selected_M" =~ ^[1-9][0-9]*$ && "$selected_m" =~ ^[1-9][0-9]*$ ]] || { echo "invalid M or m" >&2; exit 2; }
(( selected_M > 16 && selected_m >= 4 * selected_M )) || { echo "require M>k and m>=4M" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
WT="$(cd "$EXP/../.." && pwd)"
CTOL="$WT/experiments/cost-to-tolerance"
POISSON="$WT/experiments/poisson2d-rom-objective"
MSP="$WT/../2026-08-14-multistage-precision/experiments/multistage-precision"
control="$WT/experiments/cost-to-tolerance/ckpt_poisson/autodec_K16_N64_hbc_stages.pkl"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
STAGE="$HERE/stage/$cell"
[[ -f "$control" ]] || { echo "missing control checkpoint $control" >&2; exit 3; }
[[ ! -e "$STAGE" ]] || { echo "stage exists: $STAGE" >&2; exit 4; }

mkdir -p "$STAGE/code/deps/poisson2d-rom-objective/followup" \
         "$STAGE/code/deps/poisson2d-rom-objective/deps/nonlinear-decoder-architecture" \
         "$STAGE/ckpt/control" "$STAGE/ckpt/variant" "$STAGE/out" "$STAGE/logs"
cp "$CTOL"/ctol_*.py "$STAGE/code/"
cp "$POISSON/pro_common.py" "$STAGE/code/deps/poisson2d-rom-objective/"
cp "$POISSON/followup/fu_eq.py" "$POISSON/followup/fu_style.py" \
   "$STAGE/code/deps/poisson2d-rom-objective/followup/"
cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" \
   "$STAGE/code/deps/poisson2d-rom-objective/deps/"
cp "$EXP/nda_arch.py" \
   "$STAGE/code/deps/poisson2d-rom-objective/deps/nonlinear-decoder-architecture/"
cp "$control" "$STAGE/ckpt/control/autodec_K16_N64_hbc_stages.pkl"
cp "$variant" "$STAGE/ckpt/variant/autodec_K16_N64_hbc_stages.pkl"
cat > "$STAGE/configs.json" <<EOF
[
  {"pde":"poisson2d","method":"coord","N":64,"k":16,"M":$selected_M,"m":$selected_m,"tau":0.01,"arm":"architecture_e2e"},
  {"pde":"poisson2d","method":"coord","N":64,"k":16,"M":$selected_M,"m":$selected_m,"tau":0.001,"arm":"architecture_e2e"}
]
EOF
/home/tahmid/Dev/.venv/bin/python - "$STAGE/configs.json" "$selected_M" "$selected_m" <<'PY'
import json
import sys

path, expected_M, expected_m = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path) as fp:
    rows = json.load(fp)
assert len(rows) == 2
assert {(row["M"], row["m"]) for row in rows} == {(expected_M, expected_m)}
PY
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
export PY=$PY CTOL_COMMIT=$commit
echo "cell=$cell commit=$commit host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
\$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code
export KS=16 NS=64 TAUS=1e-2,1e-3 M=$selected_M MQ=$selected_m M_BIG=$selected_M K_BIG=99 MQ_4M=$selected_m
export N_TEST=16 GN_ITERS=60 TIME_REPS=9 TIME_WARM=3 BURN_IN_S=1.5
export DO_SUPP=0 POOL_CONTROL=0 CAP_CONTROL=0 DO_POD_DIRECT=0 DO_CEILING=0
export CONFIGS=../configs.json ARM_TAG=architecture_e2e FOM_LADDER=1e-6
export HIDDEN=128 N_LAYERS=4 DECODER_ARCH=film PKL_DIR=../ckpt/control
\$PY -u ctol_poisson.py ../out/control.json
export HIDDEN=98 N_LAYERS=4 DECODER_ARCH=groupfilm FILM_GROUP_SIZE=2 FILM_START=0 Z_WIDTH=64 PKL_DIR=../ckpt/variant
\$PY -u ctol_poisson.py ../out/variant.json
\$PY - <<'PY'
import json
for name in ('control', 'variant'):
    d = json.load(open(f'../out/{name}.json'))
    assert d['complete'] and d['config']['backend'] == 'gpu'
    assert len(d['rows']) == 2 and all(r['method'] == 'coord' for r in d['rows'])
PY
echo ALL-DONE
EOF
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"
