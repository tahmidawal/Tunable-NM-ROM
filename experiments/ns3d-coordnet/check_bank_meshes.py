"""Sampling-only diagnostics of a frozen bank file at several meshes (no flow data):
P_n removal, pre-Lowdin orthonormality, Lowdin change, raw derivative mismatch
(autodiff vs spectral), and cross-mesh consistency vs the smallest mesh.
Usage: check_bank_meshes.py <bank npz> <out json> <n> [<n> ...]"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for e in ("experiments/ns3d", "experiments/ns2d", "experiments/separable-decoder", "experiments/ns3d-grok", "experiments/ns3d-shift"):
    sys.path.insert(0, str(ROOT / e))
sys.path.insert(0, str(HERE))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import bankio as BI
bank, outp, meshes = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
p = BI.load_params(bank); T = np.load(bank)["T"]
res, ref = {}, None
for n in sorted(meshes):
    G, info = BI.mesh_bank(p, T, n, abort=1.0); info.pop("lowdin_factor")
    if n <= 64:
        info["derivative_raw"] = BI.derivative_check(p, T, n)
    if ref is None:
        ref = (n, G)
    else:
        n0, G0 = ref
        Gr = np.stack([BI.resample(G[:, j].reshape(3, n, n, n), n0)[0].ravel() for j in range(G.shape[1])], 1) * (n / n0) ** 1.5
        col = np.linalg.norm(Gr - G0, axis=0) / np.linalg.norm(G0, axis=0)
        info["cross_mesh_vs_smallest"] = dict(against=n0, worst_column=float(col.max()), median_column=float(np.median(col)))
    res[str(n)] = info
    print(n, json.dumps(info), flush=True)
Path(outp).write_text(json.dumps(dict(bank=bank, meshes=res), indent=1) + "\n")
