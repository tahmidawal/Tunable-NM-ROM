# Second rebuttal push, PROTOCOL.md A7: FE comparator ladder over meshes/orders/steps/schemes on DEVELOPMENT Re 95,
# then (same allocation, CPU, 16 threads) the paired cost comparison: ROM -> five cheapest passing FE entries (5 reps)
# -> ROM.  Development data only.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
M=tasks/manifest_production.json
CASE=re95
MESHES=$NS/rb-meshes/MESHES.json
EXPORT=$NS/rb-export32/results/export
BANK=$NS/rb-a8b/results/train/trained_bank_eval.npz
PROJ=$NS/rb-a8b/results/proj/proj_re95_trained.npz
echo "b5915d667372345adfedfb15817c8071e346ea1dc49543204d4d50e10f820fd4  $BANK" | sha256sum -c
( cd $NS/rb-meshes && sha256sum *.vol.bin MESHES.json ) | tee results/meshes.sha256
TARMS='[{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"pod_r32","kind":"pod","r":32},{"name":"bank_r32","kind":"bank","r":32,"key":"trained"},{"name":"bank_p24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"}]'
DTS="0.0003125 0.000625 0.00125 0.0025"
ENTRIES=$("$PY" -c "
import json
e=[]
for m in ('c4','c3','c2','c1','m0'):
    for k in (2,3):
        e += [[m,k,400,'frozen_oseen',40],[m,k,300,'frozen_oseen',30],[m,k,200,'frozen_oseen',20],[m,k,400,'sbdf2',0],[m,k,800,'sbdf2',0],[m,k,200,'sbdf2',0]]
    if m in ('c1','m0'):
        e.append([m,2,200,'oseen_bdf2',20])
print(json.dumps(e))")
"$PY" fe_ladder2.py --manifest $M --case $CASE --meshes $MESHES --entries "$ENTRIES" --out results/screen --reps 1
TOP=$("$PY" fe_select2.py --ladder results/screen/LADDER2-$CASE.json --top 5 --out results/screen/SELECT-$CASE.json)
echo "five_cheapest_passing=$TOP"
"$PY" rom_time2.py --export $EXPORT --manifest $M --case $CASE --dts "$DTS" --arms "$TARMS" --reps 5 \
  --out results/timing/ROM-before.json --trained-bank $BANK --trained-proj $PROJ
if [ "$TOP" != "[]" ]; then
  "$PY" fe_ladder2.py --manifest $M --case $CASE --meshes $MESHES --entries "$TOP" --out results/timed --reps 5
  "$PY" fe_select2.py --ladder results/timed/LADDER2-$CASE.json --top 1 --out results/timed/SELECT-$CASE.json
fi
"$PY" rom_time2.py --export $EXPORT --manifest $M --case $CASE --dts "$DTS" --arms "$TARMS" --reps 5 \
  --out results/timing/ROM-after.json --trained-bank $BANK --trained-proj $PROJ
