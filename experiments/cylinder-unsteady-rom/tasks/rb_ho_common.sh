# Rebuttal push step 3 (PROTOCOL.md): the frozen arm on held-out Re 105, ONCE.  Expects from the arm script:
#   ARM (attempt dir of the frozen bank), EX (export the bank was trained on), POD_SHA (sha256 of $EX/pod.npz),
#   MAXRANK (24 or 32), ARMS (R4 arm list), TARMS (arms to time), FARMS (arms for figures), DTS (rollout steps, first = 1/3200),
#   FE_ENTRY (accuracy-matched FE entry chosen on Re 95, JSON list of one [k, 1/dt, K]).
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
M=tasks/manifest_production.json
BANK=$ARM/results/train/trained_bank_eval.npz
sha256sum "$BANK" "$ARM/results/train/trained_bank_params.npz" | tee results/frozen_bank.sha256
# 1) export with the frozen FE modes (by hash), held-out projections allowed
"$PY" rom_export.py --manifest $M --bank inputs/run09_bank.npz \
  --bank-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --out results/export_ho \
  --pod-from $EX/pod.npz --pod-sha256 $POD_SHA --allow-heldout $( [ "$MAXRANK" = 24 ] || echo --max-rank $MAXRANK )
"$PY" -c "import numpy as np,sys;a=np.load('$EX/quadrature.npz');b=np.load('results/export_ho/quadrature.npz');sys.exit(0 if all(np.array_equal(a[k],b[k]) for k in a.files) else 'quadrature differs')"
# 2) frozen-bank projections of the held-out trajectory
"$PY" rom_project.py --manifest $M --export results/export_ho --bank-eval $BANK --tag trained --case re105 \
  --out-dir results/proj --allow-heldout
# 3) held-out rollouts: every listed arm at every listed step (first = the reference step 1/3200)
for DT in $DTS; do
  "$PY" rom_r4.py --export results/export_ho --manifest $M --case re105 --dt $DT --uncertainty inputs/R1DE.json \
    --out results/r4_heldout_dt$DT --trained-bank $BANK --trained-proj results/proj/proj_re105_trained.npz --arms "$ARMS"
done
# 4) timings in this one allocation (CPU, 16 threads): ROM, accuracy-matched FE, ROM again
for DT in $DTS; do
  "$PY" rom_time.py --export results/export_ho --manifest $M --case re105 --dt $DT --arms "$TARMS" --reps 5 \
    --out results/timing/ROM-before-dt$DT.json --trained-bank $BANK --trained-proj results/proj/proj_re105_trained.npz
done
"$PY" fe_ladder.py --manifest $M --case re105 --allow-heldout --out results/ladder_heldout --entries "$FE_ENTRY" --reps 5
for DT in $DTS; do
  "$PY" rom_time.py --export results/export_ho --manifest $M --case re105 --dt $DT --arms "$TARMS" --reps 5 \
    --out results/timing/ROM-after-dt$DT.json --trained-bank $BANK --trained-proj results/proj/proj_re105_trained.npz
done
# 5) small field arrays for the figures (one point per element, last two periods)
DT0=${DTS%% *}
"$PY" rom_frames.py --export results/export_ho --manifest $M --case re105 --allow-heldout --series-dir results/r4_heldout_dt$DT0 \
  --arms "$FARMS" --trained-bank $BANK --out results/frames_re105.npz
