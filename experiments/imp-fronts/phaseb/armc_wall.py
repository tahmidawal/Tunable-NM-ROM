"""Raw wall residual of the trained arm-(c) bank (results-audit item): max over the 512 rotated columns of
RMS(bank column on the 4 x 257 wall nodes) / RMS(column on the 257^2 lattice), wall zeroing switched off. CPU."""
import json, pickle, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'rom')); sys.path.insert(0, str(HERE))
import fcommon as F  # noqa: E402
import npbank_arch  # noqa: E402
npbank_arch.install(F.NC.NB)
bank = json.loads((HERE / 'rom/inputs/banks/sc/BANK.json').read_text())
P = F.NC.NB.load(str(HERE / 'rom/inputs/banks/sc' / bank.get('ckpt', 'ckpt.pkl')))
T = np.load(HERE / 'rom/inputs/banks/sc' / bank.get('rotation', 'rotation_R512.npz'))['T']
x = np.arange(257) / 256.
X = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
V = np.concatenate([F.NC.NB.features_grad(P, X[s:s + 8192])[0] for s in range(0, len(X), 8192)]) @ T
wall = (X[:, 0] == 0) | (X[:, 0] == 1) | (X[:, 1] == 0) | (X[:, 1] == 1)
rel = np.sqrt(np.mean(V[wall] ** 2, 0)) / np.sqrt(np.mean(V ** 2, 0))
out = dict(max_rel_wall_rms=float(rel.max()), median_rel_wall_rms=float(np.median(rel)), registered_bar=1e-14,
           note='round-off of sin(pi x) at x = 1; the construction is exactly zero')
(HERE / 'checks/phaseb/armc_wall.json').write_text(json.dumps(out, indent=1)); print(out)
