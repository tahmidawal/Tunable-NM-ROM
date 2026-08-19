#!/usr/bin/env bash
# Build round-1 Tufts cells.  Each cell is a self-contained job directory and
# regenerates its data from the recorded seed on the compute node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
WT="$(cd "$EXP/../.." && pwd)"
POISSON="$WT/experiments/poisson2d-rom-objective"
BURGERS="$WT/experiments/burgers2d-rom-latent-stepping"
MSP="$WT/../2026-08-14-multistage-precision/experiments/multistage-precision"
BF="$WT/../2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom"
STAGE_ROOT="$HERE/stage"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
COMMIT="$(git -C "$WT" rev-parse HEAD)"
DIRTY="$(git -C "$WT" status --porcelain -- . | sha256sum | cut -c1-12)"

make_base() { # cell mem time
  local cell="$1" mem="$2" tlim="$3"
  [[ "$cell" =~ ^nda_[A-Za-z0-9_]+$ ]] || { echo "invalid cell $cell" >&2; exit 2; }
  local d="$STAGE_ROOT/$cell"
  [[ ! -e "$d" ]] || { echo "stage already exists: $d" >&2; exit 3; }
  mkdir -p "$d/code" "$d/out" "$d/logs"
  cat > "$d/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=$mem
#SBATCH -t $tlim
#SBATCH -o $REMOTE_ROOT/$cell/logs/%j.out
#SBATCH -e $REMOTE_ROOT/$cell/logs/%j.err
set -euo pipefail
cd "$REMOTE_ROOT/$cell"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=$PY
echo "cell=$cell commit=$COMMIT dirty_hash=$DIRTY host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
\$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
EOF
  echo "$d"
}

finish_cell() {
  local d="$1"
  cat >> "$d/run.sbatch" <<'EOF'
echo ALL-DONE
EOF
  (cd "$d" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
}

make_poisson() { # cell hidden layers film_start architecture group_size [train_seed] [steps]
  local cell="$1" hidden="$2" layers="$3" film_start="$4" arch="${5:-resfilm}" group="${6:-8}"
  local train_seed="${7:-0}" steps="${8:-20000}" seed_suffix=""
  [[ "$train_seed" =~ ^[0-9]+$ ]] || { echo "invalid training seed $train_seed" >&2; exit 2; }
  [[ "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "invalid step count $steps" >&2; exit 2; }
  [[ "$train_seed" == 0 ]] || seed_suffix="_S$train_seed"
  local d
  d="$(make_base "$cell" 64G 12:00:00)"
  mkdir -p "$d/code/poisson/followup" "$d/code/poisson/deps/nonlinear-decoder-architecture"
  cp "$POISSON/pro_common.py" "$POISSON/pro_colloc.py" "$d/code/poisson/"
  cp "$POISSON/followup/fu_train.py" "$d/code/poisson/followup/"
  cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" "$d/code/poisson/deps/"
  cp "$EXP/nda_arch.py" "$d/code/poisson/deps/nonlinear-decoder-architecture/"
  cat >> "$d/run.sbatch" <<EOF
cd code/poisson
export N=64 N_TRAIN=512 N_VAL=64 STEPS=$steps BATCH=32 P_SUB=1024
export K_LAT=16 HARD_BC=1 GN_ITERS=60 TRAIN_SEED=$train_seed
export HIDDEN=$hidden N_LAYERS=$layers DECODER_ARCH=$arch FILM_GROUP_SIZE=$group FILM_START=$film_start Z_WIDTH=64
\$PY -u followup/fu_train.py ../../out
export PKL=../../out/autodec_K16_N64_hbc${seed_suffix}_stages.pkl
export N_TEST=16 OBJECTIVES=weak_a1_M128 MS=512 SCHEMES=full,nnls INITS=nearest
export EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_FIXED_SNAPS=1
\$PY -u pro_colloc.py ../../out/rom.json
\$PY -c "import json; d=json.load(open('../../out/rom.json')); assert d['complete'] and d['manifest']['backend']=='gpu' and len(d['rows'])==2"
EOF
  finish_cell "$d"
}

make_burgers() { # cell hidden layers n_freq architecture group_size film_start [train_seed] [steps]
  local cell="$1" hidden="$2" layers="$3" n_freq="$4" arch="${5:-resfilm}" group="${6:-8}" film_start="${7:-2}"
  local train_seed="${8:-0}" steps="${9:-60000}" seed_suffix=""
  [[ "$train_seed" =~ ^[0-9]+$ ]] || { echo "invalid training seed $train_seed" >&2; exit 2; }
  [[ "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "invalid step count $steps" >&2; exit 2; }
  [[ "$train_seed" == 0 ]] || seed_suffix="_S$train_seed"
  local d
  d="$(make_base "$cell" 80G 18:00:00)"
  mkdir -p "$d/code/burgers/followup" \
           "$d/code/burgers/deps/nonlinear-decoder-architecture" \
           "$d/code/burgers/deps/multistage-precision" \
           "$d/code/burgers/deps/burgers2d-coord-rom"
  cp "$BURGERS"/blat_*.py "$d/code/burgers/"
  cp "$BURGERS"/followup/fu_*.py "$d/code/burgers/followup/"
  cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" \
     "$d/code/burgers/deps/multistage-precision/"
  cp "$BF/burgers2d_film.py" "$d/code/burgers/deps/burgers2d-coord-rom/"
  cp "$EXP/nda_arch.py" "$d/code/burgers/deps/nonlinear-decoder-architecture/"
  cat >> "$d/run.sbatch" <<EOF
cd code/burgers
export N=64 N_TRAIN=512 N_VAL=64 N_TEST=16 K_LAT=16
export AD_STEPS=$steps AD_BATCH=128 P_SUB=2048 T_SMOOTH=1e-2 LAT_REG=1e-4 LAT_LR=5e-3 PEAK_LR=2e-3
export TRAIN_SEED=$train_seed
export AD_HIDDEN=$hidden AD_LAYERS=$layers AD_N_FREQ=$n_freq
export DECODER_ARCH=$arch FILM_GROUP_SIZE=$group FILM_START=$film_start Z_WIDTH=64
export GN_BUDGET=30 GN_TOL=1e-9 IC_BUDGET=100 FLOOR_BUDGET=60
export VARIANTS=lspg:full:weak64,lspg:eq256:weak64,lspg:eq512:weak64
export POD_KS=16 POD_VARIANTS= DO_TIMING=1 TIME_REPS=7
\$PY -u blat_train_ad.py ../../out
\$PY -u blat_rom.py ../../out/blat_ad_N64_K16${seed_suffix}.pkl ../../out
\$PY -c "import json; d=json.load(open('../../out/blat_rom_N64_K16${seed_suffix}.json')); assert d['backend']=='gpu' and set(d['rom'])=={'lspg:full:weak64','lspg:eq256:weak64','lspg:eq512:weak64'}"
EOF
  finish_cell "$d"
}

round="${1:-round1}"
case "$round" in
  round1)
    make_poisson nda_p96l4_r1 96 4 2
    make_poisson nda_p128l4_r1 128 4 2
    make_burgers nda_b160l4f31_r1 160 4 31
    make_burgers nda_b160l4f16_r1 160 4 16
    make_burgers nda_b128l5f16_r1 128 5 16
    ;;
  round2)
    # Every affine layer is modulated, but adjacent channels share one FiLM
    # coefficient.  group=2 and group=4 bracket the compression/accuracy trade.
    make_poisson nda_pg128l4g2_r2 128 4 0 groupfilm 2
    make_poisson nda_pg128l4g4_r2 128 4 0 groupfilm 4
    make_burgers nda_bg160l4g2f31_r2 160 4 31 groupfilm 2 0
    make_burgers nda_bg192l4g2f31_r2 192 4 31 groupfilm 2 0
    ;;
  round3)
    # Narrow all-layer grouped-FiLM screen: reduce the pointwise trunk cost,
    # which the round-2 same-GPU benchmark identified as the remaining limit.
    make_poisson nda_pg112l4g2_r3 112 4 0 groupfilm 2
    make_poisson nda_pg96l4g2_r3 96 4 0 groupfilm 2
    make_poisson nda_pg112l3g2_r3 112 3 0 groupfilm 2
    ;;
  round4)
    # Interpolate the width knee after round 3: 112 passes the decoder/full-ROM
    # gates, while 96 is the lower bracket.
    make_poisson nda_pg104l4g2_r4 104 4 0 groupfilm 2
    ;;
  *)
    echo "usage: $0 [round1|round2|round3|round4]" >&2
    exit 2
    ;;
esac

echo "built $round cells under $STAGE_ROOT"
