# g2d expanded heat panel 512^2
set -uo pipefail
cd experiments/heat-bank-knob
"$PY" hbk_run.py --config "$ROOT/configs/phl512_hbk.json" --out "$OUT/hbk" || echo "HBK FAILED"
rm -f "$OUT"/hbk/fields_*.npz
cd ../heat-compare-hires
"$PY" panel.py --config "$ROOT/configs/phl512_ops.json" --out "$OUT/ops" || echo "OPS PANEL FAILED"
rm -rf "$OUT/ops/fields" "$ROOT/ckpt"
