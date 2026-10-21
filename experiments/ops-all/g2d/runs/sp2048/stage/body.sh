# g2d speed2048: burgers2d-speed b2speed.py (unmodified, commit fb4a9ff7) at 2048^2 + the lane's NumPy audit, in-job
set -uo pipefail
mkdir -p "$ROOT/arc/output" "$ROOT/arc/logs"
cd experiments/burgers2d-speed
CK=../separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl
ROT=../burgers-bank-knob/inputs/rotation_R512.npz
"$PY" "$ROOT/speed2048/b2speed.py" --config "$ROOT/speed2048/config-2048.json" --checkpoint $CK --rotation $ROT \
  --inputs ../b-panel/inputs --out "$ROOT/arc/output" || echo "B2SPEED FAILED"
cp "$ROOT/job.out" "$ROOT/arc/logs/job.out"
"$PY" "$ROOT/speed2048/audit_b2speed.py" "$ROOT/arc" --checkpoint $CK --rotation $ROT --directions ../b-panel/inputs/directions_qtd02.npz \
  --out "$OUT/b2048-summary.json" || echo "AUDIT FAILED"
cp "$ROOT/arc/output/result.json" "$OUT/result.json"
rm -rf "$ROOT/arc"
