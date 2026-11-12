# Second rebuttal push, PROTOCOL.md A8: development step ladder of the frozen stream-function bank (Re 95), then the
# paired development timing in ONE allocation: reduced models (POD, ordinary bank, stream bank) -> the cheapest passing
# FE entries of the rb-fa timed ladder (5 reps) -> reduced models again.  Development data only.
set -euo pipefail
source tasks/stream_frozen.env
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
M=tasks/manifest_production.json
CASE=re95
MESHES=$NS/rb-meshes/MESHES.json
EXPORT=$NS/rb-export32/results/export
BANK=$NS/rb-a8b/results/train/trained_bank_eval.npz
PROJ=$NS/rb-a8b/results/proj/proj_re95_trained.npz
SBANK=$SARM/results/train/trained_bank_eval.npz
SPROJ=$SARM/results/proj/proj_re95_trained.npz
printf '%s  %s\n' b5915d667372345adfedfb15817c8071e346ea1dc49543204d4d50e10f820fd4 $BANK $SBANK_SHA $SBANK | sha256sum -c | tee results/frozen_banks.sha256
TARMS='[{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"pod_r32","kind":"pod","r":32},{"name":"bank_r32","kind":"bank","r":32,"key":"trained"},{"name":"bank_p24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"}]'
SARMS='[{"name":"stream_r32","kind":"bank","r":32,"key":"trained"},{"name":"stream_p24","kind":"bank","r":24,"key":"trained"},{"name":"stream_p16","kind":"bank","r":16,"key":"trained"}]'
DTS="0.0003125 0.000625 0.00125 0.0025"
for DT in $DTS; do
  "$PY" rom_r4.py --export $EXPORT --manifest $M --case $CASE --dt $DT --uncertainty inputs/R1DE.json \
    --out results/r4s_dev_dt$DT --trained-bank $SBANK --trained-proj $SPROJ --arms "$SARMS"
done
TOP=$("$PY" fe_select2.py --ladder $NS/rb-fa/results/timed/LADDER2-$CASE.json --top 2 --out results/SELECT-dev-top2.json)
echo "fe_entries=$TOP"
tm() {
  "$PY" rom_time2.py --export $EXPORT --manifest $M --case $CASE --dts "$DTS" --arms "$TARMS" --reps 5 \
    --out results/timing/ROM-$1.json --trained-bank $BANK --trained-proj $PROJ
  "$PY" rom_time2.py --export $EXPORT --manifest $M --case $CASE --dts "$DTS" --arms "$SARMS" --reps 5 \
    --out results/timing/STREAM-$1.json --trained-bank $SBANK --trained-proj $SPROJ
}
tm before
"$PY" fe_ladder2.py --manifest $M --case $CASE --meshes $MESHES --entries "$TOP" --out results/timed --reps 5
"$PY" fe_select2.py --ladder results/timed/LADDER2-$CASE.json --top 1 --out results/timed/SELECT-$CASE.json
tm after
