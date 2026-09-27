export BT_MESH=512 BT_SPEC=spec-tsol-refine.json
# g2d Burgers operator training, one network: BT_MESH, BT_SPEC (env, set by the wrapper line below)
set -uo pipefail
cd experiments/burgers-compare-hires
"$PY" opdata.py --mesh "$BT_MESH" --out data || { echo "OPDATA FAILED"; exit 1; }
cp data/opdata-summary.json "$OUT/"
mkdir -p logs
"$PY" ops/worker.py "$ROOT/configs/$BT_SPEC"; W=$?
cp -a logs "$OUT/logs"
mv out "$OUT/train"
rm -rf data "$OUT"/train/*/*.prediction.npz
echo "worker_exit=$W"
exit $W
