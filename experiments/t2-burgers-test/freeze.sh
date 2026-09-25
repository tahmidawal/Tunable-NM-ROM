#!/bin/bash
# Write FROZEN.sha256: every frozen input of this lane (code, configs, model, rules, operator checkpoints).
set -euo pipefail
cd "$(dirname "$0")"
E=..
{
  sha256sum tcmp.py t2audit.py make_configs.py make_operators.py ops/*.py cluster/stage.py config-t256.json config-t1024.json \
    config-t2048.json config-d256.json operators-t256.json operators-t1024.json operators-t2048.json operators-d256.json
  sha256sum $E/burgers-compare-hires/cmp.py $E/burgers-compare-hires/make_configs.py $E/burgers-compare-hires/lib/*.py \
    $E/burgers-compare-hires/sfit.py $E/burgers-compare-hires/gridarm.py $E/burgers-compare-hires/inputs/rotation_R512.npz \
    $E/burgers-compare-hires/inputs/pinned/*.json $E/b-panel/inputs/directions_qtd02.npz \
    $E/b-panel/inputs/rules-eqtop/rule_q0_m1024_qrg304_reachable.npz \
    $E/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl
  for f in operators-t256.json operators-t1024.json operators-t2048.json; do
    python3 -c "import json,sys; [print(o['sha256']+'  '+o['local_path']+'  ['+sys.argv[1]+' '+o['name']+(' TABLE2' if o['table2_row'] else '')+']') for o in json.load(open(sys.argv[1]))['operators']]" $f
  done
} > FROZEN.sha256
wc -l FROZEN.sha256
