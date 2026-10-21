"""spectral-pilot Part A padding gate (DESIGN A3.4): the reference FOM at N_ref with a 2N_ref padded grid against the
default 3N_ref/2, on the two test cases with the largest s, same step count. PILOT: small, not a result.

  python pad_gate.py A2
"""
from __future__ import annotations

import json
import sys

import numpy as np
import jax.numpy as jnp

import common as C
from gen_a import run_batched

lv = sys.argv[1]
d = C.RUNS / 'A' / lv
ref = json.loads((d / 'ref.json').read_text())
Nref, n = ref['Nref'], ref['n_ref']
te = np.asarray(ref['test_desc'])
idx = np.argsort(-C.steep(te))[:2]
Bref = np.load(d / 'ref_coef.npy')[idx]
B2 = run_batched(C.make_fom(Nref, n, 5, lattice=False, M=2 * Nref), te[idx], 2, chunk=1)
e = np.asarray(C.err_coef(jnp.asarray(B2), jnp.asarray(Bref)))
out = dict(level=lv, cases=idx.tolist(), err=e.tolist(), max=float(np.max(e)), ok=bool(np.max(e) <= 2e-5),
           env=C.env_record(), label=C.PILOT)
C.dump(d / 'pad.json', out)
print(lv, 'pad gate', out['max'], out['ok'])
