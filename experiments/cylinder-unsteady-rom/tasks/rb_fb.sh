# Second rebuttal push, PROTOCOL.md A7(d) + A8: held-out Re 105.
#  1) FE: the eight cheapest development-passing entries of the rb-fa screening plus the previous comparator are run
#     on Re 105 (one run each); the five cheapest passing there are re-timed (5 reps) between two ROM timing blocks.
#  2) The frozen stream-function bank (rank 32, prefixes 24/16) is run on Re 105 ONCE at the four steps of the ladder
#     (second use of this case, nothing selected here).  The ordinary bank and POD were run on Re 105 in rb-ho1 and are
#     only re-timed here (same rollouts, same start coefficients).
set -euo pipefail
source tasks/stream_frozen.env
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
M=tasks/manifest_production.json
CASE=re105
MESHES=$NS/rb-meshes/MESHES.json
BANK=$NS/rb-a8b/results/train/trained_bank_eval.npz
PROJ=$NS/rb-ho1/results/proj/proj_re105_trained.npz
SBANK=$SARM/results/train/trained_bank_eval.npz
printf '%s  %s\n' b5915d667372345adfedfb15817c8071e346ea1dc49543204d4d50e10f820fd4 $BANK $SBANK_SHA $SBANK | sha256sum -c | tee results/frozen_banks.sha256
# export directory for Re 105: the rb-ho1 export (its basis.npz was removed) with the byte-identical basis of rb-export32
EXPORT=results/export105
mkdir -p $EXPORT
for f in EXPORT.json pod.npz quadrature.npz proj_re105.npz; do ln -s $NS/rb-ho1/results/export_ho/$f $EXPORT/$f; done
ln -s $NS/rb-export32/results/export/basis.npz $EXPORT/basis.npz
"$PY" -c "
import json,hashlib,sys
e=json.load(open('$EXPORT/EXPORT.json'))['files']
h=hashlib.sha256(open('$EXPORT/basis.npz','rb').read()).hexdigest()
print('basis', h); sys.exit(0 if h==e['basis.npz'] else 'basis hash differs from the Re 105 export record')"
TARMS='[{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"pod_r32","kind":"pod","r":32},{"name":"bank_r32","kind":"bank","r":32,"key":"trained"},{"name":"bank_p24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"}]'
SARMS='[{"name":"stream_r32","kind":"bank","r":32,"key":"trained"},{"name":"stream_p24","kind":"bank","r":24,"key":"trained"},{"name":"stream_p16","kind":"bank","r":16,"key":"trained"}]'
DTS="0.0003125 0.000625 0.00125 0.0025"
# stream bank on Re 105, once
"$PY" rom_project.py --manifest $M --export $EXPORT --bank-eval $SBANK --tag trained --case $CASE --out-dir results/proj_stream --allow-heldout
SPROJ=results/proj_stream/proj_${CASE}_trained.npz
for DT in $DTS; do
  "$PY" rom_r4.py --export $EXPORT --manifest $M --case $CASE --dt $DT --uncertainty inputs/R1DE.json \
    --out results/r4s_heldout_dt$DT --trained-bank $SBANK --trained-proj $SPROJ --arms "$SARMS"
done
# FE ladder on Re 105
CAND=$("$PY" fe_select2.py --ladder $NS/rb-fa/results/screen/LADDER2-re95.json --top 8 --always '[["m0",2,400,"frozen_oseen",40]]' --out results/SELECT-dev8.json)
echo "candidates=$CAND"
"$PY" fe_ladder2.py --manifest $M --case $CASE --meshes $MESHES --entries "$CAND" --out results/screen --reps 1 --allow-heldout
TOP=$("$PY" fe_select2.py --ladder results/screen/LADDER2-$CASE.json --top 5 --out results/screen/SELECT-$CASE.json)
echo "five_cheapest_passing=$TOP"
tm() {
  "$PY" rom_time2.py --export $EXPORT --manifest $M --case $CASE --dts "$DTS" --arms "$TARMS" --reps 5 \
    --out results/timing/ROM-$1.json --trained-bank $BANK --trained-proj $PROJ
  "$PY" rom_time2.py --export $EXPORT --manifest $M --case $CASE --dts "$DTS" --arms "$SARMS" --reps 5 \
    --out results/timing/STREAM-$1.json --trained-bank $SBANK --trained-proj $SPROJ
}
tm before
if [ "$TOP" != "[]" ]; then
  "$PY" fe_ladder2.py --manifest $M --case $CASE --meshes $MESHES --entries "$TOP" --out results/timed --reps 5 --allow-heldout
  "$PY" fe_select2.py --ladder results/timed/LADDER2-$CASE.json --top 1 --out results/timed/SELECT-$CASE.json
fi
tm after
