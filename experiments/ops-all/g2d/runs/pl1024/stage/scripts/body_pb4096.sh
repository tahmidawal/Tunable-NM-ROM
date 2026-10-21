export BP_DRIVER=bankknob BP_CONFIG=pb4096.json BP_MESH=4096 BP_OPS="fno-large:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/b4096tc/out/train/fno-large unet-refine:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/b4096tc/out/train/unet-refine tsol-refine:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/b4096tc/out/train/tsol-refine don-small:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/b4096tc/out/train/don-small"
# g2d Burgers panel: BP_DRIVER (b2speed|bankknob), BP_CONFIG, BP_MESH, BP_OPS="arm:<abs dir with result.json+best.pt> ..."
set -uo pipefail
if [ "$BP_DRIVER" = b2speed ]; then cd experiments/burgers2d-speed; DRV=b2speed.py; else cd experiments/burgers-bank-knob; DRV=bankknob.py; fi
"$PY" $DRV --config "$ROOT/configs/$BP_CONFIG" --checkpoint ../separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl \
  --rotation ../burgers-bank-knob/inputs/rotation_R512.npz --inputs ../b-panel/inputs --out "$OUT/drv" || echo "DRIVER FAILED"
[ -f "$OUT/drv/opcohort/index.json" ] || { echo "NO OPCOHORT"; exit 1; }
cd ../burgers-compare-hires
mkdir -p "$OUT/drv/optiming" "$OUT/drv/fields" "$ROOT/opckpt"
NAMES=""
for pair in $BP_OPS; do
  arm=${pair%%:*}; src=${pair#*:}; NAMES="$NAMES $arm"
  want=$("$PY" -c "import json;print(json.load(open('$src/result.json'))['best_checkpoint_sha256'])") || { echo "NO TRAINING RESULT $arm"; continue; }
  cp "$src/best.pt" "$ROOT/opckpt/$arm.pt"; got=$(sha256sum "$ROOT/opckpt/$arm.pt" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "CHECKPOINT HASH MISMATCH $arm"; rm -f "$ROOT/opckpt/$arm.pt"; continue; }
  echo "CHECKPOINT OK $arm $got"
  "$PY" ops/optime.py --checkpoint "$ROOT/opckpt/$arm.pt" --index "$OUT/drv/opcohort/index.json" --out "$OUT/drv/optiming" \
    --fields "$OUT/drv/fields" --name $arm --role "trained at ${BP_MESH}^2 (g2d)" --repetitions 5 --burn-in 20 || echo "OPERATOR FAILED $arm"
done
"$PY" "$ROOT/scripts/opscore.py" "$OUT/drv" $NAMES
rm -rf "$ROOT/opckpt" "$OUT/drv/opcohort"/*.npz
