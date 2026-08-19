#!/usr/bin/env bash
# Same-GPU, burn-in-controlled end-to-end Burgers comparison at selected arms.
set -euo pipefail

cell="$1"; variant="$2"; group="${3:-2}"; selected_m="${4:-}"; variant_hidden="${5:-160}"
tau_loose="${6:-0.01}"; tau_tight="${7:-0.001}"
[[ "$cell" =~ ^nda_[A-Za-z0-9_]+$ ]] || { echo "invalid cell" >&2; exit 2; }
[[ -f "$variant" ]] || { echo "missing variant checkpoint $variant" >&2; exit 3; }
[[ "$group" =~ ^[1-9][0-9]*$ ]] || { echo "invalid FiLM group $group" >&2; exit 2; }
[[ -z "$selected_m" || "$selected_m" =~ ^[1-9][0-9]*$ ]] || { echo "invalid selected m" >&2; exit 2; }
[[ "$variant_hidden" =~ ^[1-9][0-9]*$ ]] || { echo "invalid variant width $variant_hidden" >&2; exit 2; }
(( variant_hidden % group == 0 )) || { echo "width must be divisible by FiLM group" >&2; exit 2; }
[[ "$tau_loose" =~ ^0\.[0-9]+$ && "$tau_tight" =~ ^0\.[0-9]+$ ]] || { echo "invalid tolerances" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
WT="$(cd "$EXP/../.." && pwd)"
CTOL="$WT/experiments/cost-to-tolerance"
BURGERS="$WT/experiments/burgers2d-rom-latent-stepping"
MSP="$WT/../2026-08-14-multistage-precision/experiments/multistage-precision"
BF="$WT/../2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom"
control="$WT/experiments/cost-to-tolerance/ckpt_burgers/blat_ad_N64_K16.pkl"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
STAGE="$HERE/stage/$cell"
[[ -f "$control" ]] || { echo "missing control checkpoint $control" >&2; exit 3; }
[[ ! -e "$STAGE" ]] || { echo "stage exists: $STAGE" >&2; exit 4; }

dep="$STAGE/code/deps/burgers2d-rom-latent-stepping"
mkdir -p "$dep/followup" "$dep/deps/burgers2d-coord-rom" \
         "$dep/deps/multistage-precision" "$dep/deps/nonlinear-decoder-architecture" \
         "$STAGE/ckpt/control" "$STAGE/ckpt/variant" "$STAGE/out" "$STAGE/logs"
cp "$CTOL"/ctol_*.py "$STAGE/code/"
cp "$BURGERS/blat_common.py" "$dep/"
cp "$BURGERS/followup/fu_common.py" "$BURGERS/followup/fu_style.py" "$dep/followup/"
cp "$BF/burgers2d_film.py" "$dep/deps/burgers2d-coord-rom/"
cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" "$dep/deps/multistage-precision/"
cp "$EXP/nda_arch.py" "$dep/deps/nonlinear-decoder-architecture/"
cp "$control" "$STAGE/ckpt/control/blat_ad_N64_K16.pkl"
cp "$variant" "$STAGE/ckpt/variant/blat_ad_N64_K16.pkl"
if [[ -n "$selected_m" ]]; then
cat > "$STAGE/configs.json" <<EOF
[
  {"pde":"burgers2d","method":"coord","N":64,"k":16,"M":128,"m":$selected_m,"tau":$tau_loose,"arm":"selected"},
  {"pde":"burgers2d","method":"coord","N":64,"k":16,"M":128,"m":$selected_m,"tau":$tau_tight,"arm":"selected"}
]
EOF
  main_M=128; main_m="$selected_m"; expected_rows=2
else
cat > "$STAGE/configs.json" <<'EOF'
[
  {"pde":"burgers2d","method":"coord","N":64,"k":16,"M":96,"m":384,"tau":0.01,"arm":"weak96"},
  {"pde":"burgers2d","method":"coord","N":64,"k":16,"M":96,"m":384,"tau":0.001,"arm":"weak96"},
  {"pde":"burgers2d","method":"coord","N":64,"k":16,"M":128,"m":512,"tau":0.01,"arm":"weak128"},
  {"pde":"burgers2d","method":"coord","N":64,"k":16,"M":128,"m":512,"tau":0.001,"arm":"weak128"}
]
EOF
  main_M=96; main_m=384; expected_rows=4
fi
commit="$(git -C "$WT" rev-parse HEAD)"
cat > "$STAGE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 18:00:00
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
export N=64 KS=16 NS=64 TAUS=$tau_loose,$tau_tight M=$main_M MQ=$main_m M_BIG=128 K_BIG=99 MQ_4M=$main_m
export CTOL_N_TEST=16 N_POD_TRAJ=512 POD_SLICE_STRIDE=4 GEN_CHUNK=16
export TIME_REPS=9 TIME_WARM=3 BURN_IN_S=1.5 DO_SUPP=0 DO_CEILING=0
export CONFIGS=../configs.json ARM_TAG=architecture_e2e FOM_NEWTON_LADDER=3
export AD_HIDDEN=256 AD_LAYERS=5 DECODER_ARCH=film PKL_DIR=../ckpt/control
\$PY -u ctol_burgers.py ../out/control.json
export AD_HIDDEN=$variant_hidden AD_LAYERS=4 DECODER_ARCH=groupfilm FILM_GROUP_SIZE=$group FILM_START=0 Z_WIDTH=64 PKL_DIR=../ckpt/variant
\$PY -u ctol_burgers.py ../out/variant.json
\$PY - <<'PY'
import json
for name in ('control', 'variant'):
    d = json.load(open(f'../out/{name}.json'))
    assert d['complete'] and d['config']['backend'] == 'gpu'
    assert len(d['rows']) == $expected_rows and all(r['method'] == 'coord' for r in d['rows'])
PY
echo ALL-DONE
EOF
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"
