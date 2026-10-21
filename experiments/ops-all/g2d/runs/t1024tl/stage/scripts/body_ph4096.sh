# g2d heat 4096^2 panel: (1) heat-bank-knob hbk_run.py -> Table 1 spans lin_R128_cn / lin_R48_cn + CN-CG grid;
# (2) heat-compare-hires panel.py -> the four operators (+ DST control), second process, same allocation.
set -uo pipefail
cd experiments/heat-bank-knob
"$PY" hbk_run.py --config "$ROOT/configs/ph4096_hbk.json" --out "$OUT/hbk" || echo "HBK FAILED"
cd ../heat-compare-hires
"$PY" panel.py --config "$ROOT/configs/ph4096_ops.json" --out "$OUT/ops" || echo "OPS PANEL FAILED"
rm -f "$OUT"/hbk/fields_*.npz
