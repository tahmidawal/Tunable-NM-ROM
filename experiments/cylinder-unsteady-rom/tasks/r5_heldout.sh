# G5 R5 (DESIGN-G5 5.4, G5-A5 item 6): held-out Re 105 opened ONCE with frozen settings,
# every R4 arm, then paired FE/ROM timings in this one allocation.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
EX=$NS/r2-export01/results/export; TR=$NS/r3-train01/results
sha256sum -c <(grep -E 'quadrature.npz|basis.npz|pod.npz' inputs/EXPORT-SHA256SUMS | sed "s#  ./results/export/#  $EX/#")
sha256sum -c inputs/R3-SHA256SUMS
ARMS_ALL='[{"name":"pod_r8","kind":"pod","r":8},{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"frozen_bank_r4","kind":"bank","r":4,"key":"frozen"},{"name":"bank_r16","kind":"bank","r":16,"key":"trained"}]'
# 1) export with the frozen order-12 FE modes (by hash), held-out projections allowed
"$PY" rom_export.py --manifest tasks/manifest_production.json --bank inputs/run09_bank.npz \
  --bank-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --out results/export_r5 \
  --pod-from $EX/pod.npz --pod-sha256 56c74a805618fe5c13d6ff795aa4125d15fce791401a9e891da7abde0485bf80 --allow-heldout
cmp <("$PY" -c "import numpy as np;print(np.load('$EX/quadrature.npz')['points'].tobytes().hex()[:64])") \
    <("$PY" -c "import numpy as np;print(np.load('results/export_r5/quadrature.npz')['points'].tobytes().hex()[:64])")
# 2) trained-bank projections of the held-out trajectory (bank evaluated on the same Q)
"$PY" rom_project.py --manifest tasks/manifest_production.json --export results/export_r5 \
  --bank-eval $TR/train/trained_bank_eval.npz --tag trained --case re105 --out-dir results/proj --allow-heldout
# 3) held-out rollouts, every R4 arm, frozen settings
"$PY" rom_r4.py --export results/export_r5 --manifest tasks/manifest_production.json --case re105 --dt 0.0003125 \
  --uncertainty inputs/R1DE.json --out results/r4_heldout --trained-bank $TR/train/trained_bank_eval.npz \
  --trained-proj results/proj/proj_re105_trained.npz --arms "$ARMS_ALL"
# 4) paired timings (same allocation, CPU, 16 threads)
"$PY" rom_r5.py --record-run $NS/r1-record-re105/results/record_re105_p4 --export results/export_r5 --case re105 \
  --manifest tasks/manifest_production.json --arms "$ARMS_ALL" --trained-bank $TR/train/trained_bank_eval.npz \
  --trained-proj results/proj/proj_re105_trained.npz --out results/timing
# 5) side diagnostic: magnitude of the near-zero lift-divergence vector BL behind the quadrature-check failure
"$PY" -c "
import numpy as np, json, rom_core as rc
out={}
for tag,d in (('order12','$EX'),('order14','$NS/r4-qcheck02/results/export_q14')):
    a=rc.Arm(rc.Export(d),'pod',16,orthonormalize=False)
    out[tag]={'norm_BL':float(np.linalg.norm(a.BL)),'norm_B':float(np.linalg.norm(a.B)),'norm_KL':float(np.linalg.norm(a.KL))}
open('results/BL_magnitude.json','w').write(json.dumps(out,indent=2)); print(out)"
