# Rebuttal push, amendment A6: paired dev (Re 95) timings of the frozen bank / POD and the accuracy-matched FE
# in one allocation, with the same FE ladder list as the held-out job.  Development data only.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
M=tasks/manifest_production.json
CASE=re95
EXPORT=$NS/rb-export32/results/export
BANK=$NS/rb-a8b/results/train/trained_bank_eval.npz
PROJ=$NS/rb-a8b/results/proj/proj_re95_trained.npz
echo "b5915d667372345adfedfb15817c8071e346ea1dc49543204d4d50e10f820fd4  $BANK" | sha256sum -c
TARMS='[{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"pod_r32","kind":"pod","r":32},{"name":"bank_r32","kind":"bank","r":32,"key":"trained"},{"name":"bank_p24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"}]'
DTS="0.0003125 0.000625 0.00125 0.0025"
FE_LADDER='[[2,200,2],[2,200,20],[2,400,20],[2,400,40],[2,400,80],[2,800,40],[2,800,80]]'
FE_FLAGS=""
source tasks/rb_cmp_common.sh
