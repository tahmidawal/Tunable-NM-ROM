#!/bin/bash
cd "$(dirname "$0")/out"
source /etc/profile.d/jax-mem.sh
CKPT=../../sepdec_r1/out/sep_burgers_N64_K16_R64.pkl N=64 EQ_M=64 EQ_M_FACTOR=4 \
STEPS=500 LR=3e-5 TRAIN_H=1 TRAIN_NODES=1 SAMP_REL=1 JAC_REL=1 REFIT_EVERY=200 EVAL_EVERY=100 \
N_TEST=4 DATA_CACHE=../data_n64.npz OUT_TAG=smoke2 \
PYTHONPATH=../../.. JAX_DEFAULT_MATMUL_PRECISION=highest \
jaxrun /home/tahmid/Dev/.venv/bin/python ../../../sep_codesign.py > smoke2.log 2>&1
echo $? > smoke2.rc
