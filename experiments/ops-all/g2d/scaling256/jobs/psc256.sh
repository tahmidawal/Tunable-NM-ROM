# g2d expanded Burgers panel 256^2 (DESIGN.md "Expanded mandate")
set -uo pipefail
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd experiments/burgers-bank-knob
"$PY" bankknob.py --config "$ROOT/configs/psc256.json" --checkpoint ../separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl \
  --rotation inputs/rotation_R512.npz --inputs ../b-panel/inputs --out "$OUT/drv" || echo "DRIVER FAILED"
[ -f "$OUT/drv/opcohort/index.json" ] || { echo "NO OPCOHORT"; exit 1; }
cd ../burgers-compare-hires
mkdir -p "$OUT/drv/optiming" "$OUT/drv/fields"
"$PY" "$ROOT/scripts/stage_ops.py" "$ROOT/configs/ops-psc256.json" "$ROOT/opckpt" > "$ROOT/opckpt.txt"
cp "$ROOT/opckpt/staging.json" "$OUT/drv/staging.json"
NAMES=""
while read -r name path; do
  NAMES="$NAMES $name"
  "$PY" ops/optime.py --checkpoint "$path" --index "$OUT/drv/opcohort/index.json" --out "$OUT/drv/optiming" \
    --fields "$OUT/drv/fields" --name "$name" --role "g2d scaling256" --repetitions 5 --burn-in 20 < /dev/null || echo "OPERATOR FAILED $name"
  "$PY" "$ROOT/scripts/opscore.py" "$OUT/drv" "$name"
done < "$ROOT/opckpt.txt"
true
rm -rf "$ROOT/opckpt" "$ROOT/ckpt" "$OUT/drv/opcohort"/*.npz "$OUT/drv/fields"
