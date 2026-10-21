# g2d expanded heat panel 2048^2
set -uo pipefail
cd experiments/heat-bank-knob
"$PY" hbk_run.py --config "$ROOT/configs/phl2048_hbk.json" --out "$OUT/hbk" || echo "HBK FAILED"
rm -f "$OUT"/hbk/fields_*.npz
cd ../heat-compare-hires
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # JAX must not reserve the GPU before the PyTorch operators (phl4096 OOM)
"$PY" panel.py --config "$ROOT/configs/phl2048_ops.json" --out "$OUT/ops" || echo "OPS PANEL FAILED"
rm -rf "$OUT/ops/fields" "$ROOT/ckpt"
