#!/bin/sh
# usage: scripts/decoder_chain.sh burgers siren rbf ...   (train -> rho ladders -> rollouts per decoder)
PY=.venv/bin/python; pde=$1; shift
for arch in "$@"; do
  $PY scripts/train.py $pde --bank $arch --k 32 --suffix _$arch > data/train_${pde}_$arch.log 2>&1
  $PY scripts/quad_study.py $pde --model_suffix _$arch --tag _$arch --states train --meshes 256 --skip_eq --skip_time > results/${pde}_quad_$arch.out 2>&1
  $PY scripts/rollout_study.py $pde --model_suffix _$arch --tag _$arch --meshes 128,256,512 --skip_eq --rules dense,gauss_tensor,fibonacci,sobol,smolyak_cc > results/${pde}_rollout_$arch.out 2>&1
done
echo chain done
