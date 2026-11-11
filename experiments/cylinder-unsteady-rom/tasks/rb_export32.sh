# Rebuttal push: export with 32 POD / pressure / supremizer modes (same manifest, rule and cases as r2-export01).
set -euo pipefail
"$PY" rom_export.py --manifest tasks/manifest_production.json --bank inputs/run09_bank.npz \
  --bank-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --out results/export --max-rank 32
"$PY" -c "
import numpy as np
a=np.load('/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/r2-export01/results/export/pod.npz'); b=np.load('results/export/pod.npz')
print('sigma diff', float(np.abs(a['singular_values'][:24]-b['singular_values'][:24]).max()/a['singular_values'][0]))
print('mode diff', float(max(min(np.abs(a['modes'][i]-s*b['modes'][i]).max() for s in (1,-1)) for i in range(24))))"
