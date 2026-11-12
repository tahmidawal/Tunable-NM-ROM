# Second rebuttal push, PROTOCOL.md A9/A9b: evaluation of frozen models on the extra cases of ONE role.
# Expects ROLE (dev|conf), M (manifest), CASES ("name:Re ..."), LONG ("<name>:<first run>:<continuation run>"),
# FLAGS ("" or --allow-heldout), STAGES (subset of "merge rom ops"), SBANK/SBANK_SHA (frozen stream bank evaluation).
# Nothing is trained or selected here.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
OPS=/cluster/tufts/paralab/tawal01/rebuttal-cyl-ops
EX=$NS/rb-export32/results/export
BANK=$NS/rb-a8b/results/train/trained_bank_eval.npz
echo "b5915d667372345adfedfb15817c8071e346ea1dc49543204d4d50e10f820fd4  $BANK" | sha256sum -c
echo "$SBANK_SHA  $SBANK" | sha256sum -c
DTS=${DTS:-"0.0003125 0.000625 0.00125 0.0025"}
ARMS='[{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"pod_r32","kind":"pod","r":32},{"name":"bank_r32","kind":"bank","r":32,"key":"trained"},{"name":"bank_p24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"}]'
SARMS='[{"name":"stream_r32","kind":"bank","r":32,"key":"trained"},{"name":"stream_p24","kind":"bank","r":24,"key":"trained"},{"name":"stream_p16","kind":"bank","r":16,"key":"trained"}]'
case " $STAGES " in *" merge "*)
  IFS=: read -r LNAME LFIRST LCONT <<< "$LONG"
  mkdir -p $NS/rb-merged
  "$PY" merge_runs.py --first $LFIRST --cont $LCONT --out $NS/rb-merged/$LNAME
esac
case " $STAGES " in *" rom "*)
  "$PY" rom_export.py --manifest $M --bank inputs/run09_bank.npz \
    --bank-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --out results/export \
    --pod-from $EX/pod.npz --pod-sha256 7e7735929a5080dc5c423bf7da216e29be1e4c7f0687f89d2b17a02863c81924 $FLAGS --max-rank 32
  "$PY" -c "import numpy as np,sys;a=np.load('$EX/quadrature.npz');b=np.load('results/export/quadrature.npz');sys.exit(0 if all(np.array_equal(a[k],b[k]) for k in a.files) else 'quadrature differs')"
  for C in $CASES; do
    N=${C%%:*}
    "$PY" rom_project.py --manifest $M --export results/export --bank-eval $BANK --tag trained --case $N --out-dir results/proj_bank $FLAGS
    "$PY" rom_project.py --manifest $M --export results/export --bank-eval $SBANK --tag trained --case $N --out-dir results/proj_stream $FLAGS
    for DT in $DTS; do
      "$PY" rom_r4.py --export results/export --manifest $M --case $N --dt $DT --uncertainty inputs/R1DE.json \
        --out results/r4_${N}_dt$DT --trained-bank $BANK --trained-proj results/proj_bank/proj_${N}_trained.npz --arms "$ARMS"
      "$PY" rom_r4.py --export results/export --manifest $M --case $N --dt $DT --uncertainty inputs/R1DE.json \
        --out results/r4s_${N}_dt$DT --trained-bank $SBANK --trained-proj results/proj_stream/proj_${N}_trained.npz --arms "$SARMS"
      T=$("$PY" -c "import json;print(json.load(open('results/r4_${N}_dt$DT/R4-$N.json'))['reference_cycles']['period'])")
      S=""; for f in results/r4_${N}_dt$DT/series_*.npz results/r4s_${N}_dt$DT/series_*.npz; do b=$(basename $f .npz); S="$S ${b#series_${N}_}=$f"; done
      "$PY" drift.py --period $T --out results/DRIFT-rom-${N}-dt$DT.json $S
    done
  done
  rm -f results/export/basis.npz
esac
case " $STAGES " in *" ops "*)
  export PYTHONPATH="$PWD/ops:$PYTHONPATH"
  "$PY" -c "import torch,sys; print('torch', torch.__version__, torch.cuda.get_device_name(0)); sys.exit(0 if torch.cuda.is_available() else 42)"
  ( cd ops && sha256sum -c <(tail -3 VENDORED.txt) )
  echo "8c35451e12d2ec46bc94d45e588df21fbceee0c2e478ad2a9c5a7d5c4bace8c7  $OPS/prep01/results/data/geom.npz" | sha256sum -c
  "$PY" ops/ops_prep2.py --manifest $M --out results/opsdata --geom-ref $OPS/prep01/results/data/geom.npz $FLAGS
  CA=""; for C in $CASES; do CA="$CA --case $C:$PWD/results/opsdata"; done
  if [ "$ROLE" = dev ]; then CA="--case re95:95:$OPS/prep01/results/data $CA"; fi   # regression check against the operator lane's own numbers
  "$PY" ops/ops_eval2.py --geom $OPS/prep01/results/data/geom.npz --out results/ops --tag $ROLE $CA \
    --arm node60=$OPS/node01/results/node=069f5b8560e3aa39214adfd32931401f0cf07f4a799157fcec87548222518229 --arm node180=$OPS/node02/results/node=2a9b5143121f78a1f7af6c7b63dd9b07786ed3663e5c373794086f9f377f2eae \
    --arm grid60=$OPS/grid01/results/grid=a2af5440073637702257e68e6f6fbc7b8381cab08123428d23adc9ce20fbcc95 --arm grid180=$OPS/grid02/results/grid=1a07c6ad90d416beb1df66c9d55893b8c86bf1ef6bb6041b52fd91b1480e309a \
    --arm don60=$OPS/don01/results/don=a1f161d8c4b253592ab9f02fd5ff0075f35212794015f6bda66b823d5263c191 --arm don180=$OPS/don02/results/don=1aa0d510ab8487097bc760c538be8736b5f741a23006bf5ef36f34389781519b
  for C in $CASES; do
    N=${C%%:*}
    T=$("$PY" -c "import json;print(json.load(open('results/ops/OPS-$ROLE.json'))['reference_readout']['$N']['reference_cycles']['period'])")
    S=""; for f in results/ops/series_*_$N.npz; do b=$(basename $f .npz); b=${b#series_}; S="$S ${b%_$N}=$f"; done
    "$PY" drift.py --period $T --out results/DRIFT-ops-$N.json $S
  done
  rm -f results/opsdata/*_Q.npy results/opsdata/*_n3.npy
esac
