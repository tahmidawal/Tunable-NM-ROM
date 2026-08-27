#!/bin/bash
# N=64 mini-pilot arms of the co-design experiment (design doc:
# understand/2026-08-26-codesign-design.md).  Local GB10, <=3 concurrent
# jaxrun (CLAUDE.md).  Every run certifies BOTH base (frozen h + NNLS nodes)
# and cot (co-trained) with the same instrument, so no standalone baseline
# arm is needed.  Usage:  bash cd_minipilot.sh <arm> <mfactor>
#   arms: n (nodes only) | i (h+nodes, samp) | ii (+jac) | iii (+jac+sob)
#   mfactor: 4 -> m=256, 1 -> m=64
set -euo pipefail
ARM="$1"; MF="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
SEP="$(dirname "$HERE")"
RUN="$HERE/cd_${TAGPFX:-}${ARM}_m$((64 * MF))"
mkdir -p "$RUN/out"
cd "$RUN/out"

case "$ARM" in
  n)   FLAGS="TRAIN_H=0 TRAIN_NODES=1 SAMP_REL=1 JAC_REL=1 SOB_REL=0" ;;
  i)   FLAGS="TRAIN_H=1 TRAIN_NODES=1 SAMP_REL=1 JAC_REL=0 SOB_REL=0" ;;
  ii)  FLAGS="TRAIN_H=1 TRAIN_NODES=1 SAMP_REL=1 JAC_REL=1 SOB_REL=0" ;;
  iii) FLAGS="TRAIN_H=1 TRAIN_NODES=1 SAMP_REL=1 JAC_REL=1 SOB_REL=1" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

source /etc/profile.d/jax-mem.sh
# REC_W=10: smoke2 measured that REC_W=1 lets the optimizer trade +6.4%
# reconstruction drift for mismatch gains and the rollout gets WORSE.
env $FLAGS REC_W="${REC_W_OVR:-10}" \
  CKPT="$SEP/runs/sepdec_r1/out/sep_burgers_N64_K16_R64.pkl" \
  N=64 EQ_M=64 EQ_M_FACTOR="$MF" STEPS=2000 LR=3e-5 LR_NODES=3e-3 \
  REFIT_EVERY=500 REFIT_JAC_STATES=16 EVAL_EVERY=200 N_TEST=4 \
  DATA_CACHE="$HERE/cd_smoke/data_n64.npz" OUT_TAG="${TAGPFX:-}${ARM}_m$((64 * MF))" \
  PYTHONPATH="$SEP" JAX_DEFAULT_MATMUL_PRECISION=highest \
  jaxrun /home/tahmid/Dev/.venv/bin/python "$SEP/sep_codesign.py" \
  > run.log 2>&1
