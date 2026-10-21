"""Build arms/fid/inputs/fid_gate_dev6.npz (the in-job C-B3 references of arm fid's R0 replacement gate) from the E0
ST store via experiments/imp-fronts/fcommon.py load_ref('dev6', c) (manifest-verified refstore). CPU, NumPy only.

    /home/tahmid/Dev/.venv/bin/python experiments/imp-fronts/train2d/arms/fid/make_gate_inputs.py

Contents per case c = 0..5: ref (5, 257, 257) = E0 fields at t = 0.05 k, k = 1..5; n0 = ||u(t0)|| (fcommon.load_ref);
desc = (cx, cy, w, a, nu); eps_ref, eligible; store_version. Deterministic (np.savez, fixed key order).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                   # experiments/imp-fronts
import fcommon as F                                         # noqa: E402

cases = F.e0_cases('dev6')
assert cases == list(range(6)), cases
ref, n0, desc, eps, el = [], [], [], [], []
for c in cases:
    f, m = F.load_ref('dev6', c)
    d = m['desc']
    ref.append(np.asarray(f, np.float64)[1:]); n0.append(m['n0'])
    desc.append([d['cx'], d['cy'], d['w'], d['a'], d['nu']]); eps.append(m['eps_ref']); el.append(m['eligible'])
out = HERE / 'inputs' / 'fid_gate_dev6.npz'
out.parent.mkdir(exist_ok=True)
np.savez(out, ref=np.stack(ref), n0=np.asarray(n0), desc=np.asarray(desc, np.float64), eps_ref=np.asarray(eps, float),
         eligible=np.asarray(el, bool), case=np.asarray(cases), store_version=np.asarray(str(F.RS.store_version())))
print(out, hashlib.sha256(out.read_bytes()).hexdigest(), out.stat().st_size)
