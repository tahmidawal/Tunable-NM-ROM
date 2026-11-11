# Rebuttal push, amendment A4: ROM step ladder on development Re 95 (bank of arm a8b at r32 and its r24, r16 prefixes, POD at the same ranks).
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
EX=$NS/rb-export32/results/export
ARM=$NS/rb-a8b
M=tasks/manifest_production.json
ARMS='[{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"pod_r32","kind":"pod","r":32},{"name":"bank_r32","kind":"bank","r":32,"key":"trained"},{"name":"bank_p24","kind":"bank","r":24,"key":"trained"},{"name":"bank_p16","kind":"bank","r":16,"key":"trained"}]'
for DT in 0.000625 0.00125 0.0025; do
  "$PY" rom_r4.py --export "$EX" --manifest $M --case re95 --dt $DT --uncertainty inputs/R1DE.json --out results/r4_dt$DT \
    --trained-bank $ARM/results/train/trained_bank_eval.npz --trained-proj $ARM/results/proj/proj_re95_trained.npz --arms "$ARMS"
done
