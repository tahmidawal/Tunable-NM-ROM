# Rebuttal push step 3 (PROTOCOL.md, amendments A4/A6): the frozen bank (arm a8b, rank 32, prefixes 24 and 16) and
# POD at the same ranks on held-out Re 105, ONCE; then the paired cost comparison with the FE ladder on Re 105.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
M=tasks/manifest_production.json
CASE=re105
EX=$NS/rb-export32/results/export
ARM=$NS/rb-a8b
BANK=$ARM/results/train/trained_bank_eval.npz
printf '%s  %s\n' b5915d667372345adfedfb15817c8071e346ea1dc49543204d4d50e10f820fd4 $BANK \
  ad9a6bd3ba38e57e71a2a5fad47b259c7a22e3127cd70efe08494f93439b1f3d $ARM/results/train/trained_bank_params.npz | sha256sum -c | tee results/frozen_bank.sha256
ARMS='[{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"pod_r32","kind":"pod","r":32},{"name":"bank_r32","kind":"bank","r":32,"key":"trained"},{"name":"bank_p24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"}]'
TARMS="$ARMS"
DTS="0.0003125 0.000625 0.00125 0.0025"
# 1) export with the frozen FE modes (by hash), held-out projections allowed
"$PY" rom_export.py --manifest $M --bank inputs/run09_bank.npz \
  --bank-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --out results/export_ho \
  --pod-from $EX/pod.npz --pod-sha256 7e7735929a5080dc5c423bf7da216e29be1e4c7f0687f89d2b17a02863c81924 --allow-heldout --max-rank 32
"$PY" -c "import numpy as np,sys;a=np.load('$EX/quadrature.npz');b=np.load('results/export_ho/quadrature.npz');sys.exit(0 if all(np.array_equal(a[k],b[k]) for k in a.files) else 'quadrature differs')"
# 2) frozen-bank projections of the held-out trajectory
"$PY" rom_project.py --manifest $M --export results/export_ho --bank-eval $BANK --tag trained --case re105 \
  --out-dir results/proj --allow-heldout
# 3) held-out rollouts: every arm at every step of the dev ladder (first = the reference step 1/3200)
for DT in $DTS; do
  "$PY" rom_r4.py --export results/export_ho --manifest $M --case re105 --dt $DT --uncertainty inputs/R1DE.json \
    --out results/r4_heldout_dt$DT --trained-bank $BANK --trained-proj results/proj/proj_re105_trained.npz --arms "$ARMS"
done
# 4) paired cost comparison (same script as the dev timing job)
EXPORT=results/export_ho
PROJ=results/proj/proj_re105_trained.npz
FE_LADDER='[[2,200,2],[2,200,20],[2,400,20],[2,400,40],[2,400,80],[2,800,40],[2,800,80]]'
FE_FLAGS="--allow-heldout"
source tasks/rb_cmp_common.sh
# 5) large intermediates are not kept (the export is reproducible from hashed inputs)
rm -f results/export_ho/basis.npz
