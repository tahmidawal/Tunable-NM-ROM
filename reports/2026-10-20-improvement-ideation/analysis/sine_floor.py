"""Sine-basis truncation floors of the saved 2D Burgers reference fields (CPU, numpy/scipy only).

Question: how many sine modes does a fixed (non-learned) sine basis need to represent the reference solutions as
well as the learned bank does (bank floor e_P), and how does that compare with the spectral FOM (N = 24, 48 modes per
axis) and with our r? Floors use the project's error convention: max over the five output times t>0 of
||P u(t) - u(t)||_2 / ||u(t0)||_2 on the 257^2 scoring lattice (here the interior 255^2 DST-I grid).
References: quadrature-study refdv store (mesh 8192, dt 3.125e-4), 38 dev6+val32 cases, read from
worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/c2budA/refs/ref_ST_*.npz (array f257).
Writes analysis/sine_floor.json next to this script.
"""
import glob, json, hashlib
from pathlib import Path
import numpy as np
from scipy.fft import dstn

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
SRC = ROOT / 'worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/c2budA/refs'
OUT = Path(__file__).with_name('sine_floor.json')
files = sorted(glob.glob(str(SRC / 'ref_ST_*.npz')))
n = 255
k = np.arange(1, n + 1)
KX, KY = np.meshgrid(k, k, indexing='ij')
lap_order = np.argsort((KX ** 2 + KY ** 2).ravel(), kind='stable')
hc_order = np.argsort((KX * KY).ravel() + 1e-6 * (KX ** 2 + KY ** 2).ravel(), kind='stable')
Ns = [16, 24, 32, 48, 64, 96, 128]
Ks = [128, 256, 384, 512, 1024, 1536, 2048, 2304, 4096, 9216]
res = dict(source=str(SRC.relative_to(ROOT)), n_cases=len(files), files_sha256={}, block={N: [] for N in Ns},
           lap={K: [] for K in Ks}, hcross={K: [] for K in Ks}, bestK={K: [] for K in Ks})
for f in files:
    res['files_sha256'][Path(f).name] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    U = np.load(f)['f257']                    # (6, 257, 257), boundary rows/cols are the Dirichlet zeros
    n0 = np.linalg.norm(U[0])
    C = np.stack([dstn(U[t, 1:-1, 1:-1], type=1, norm='ortho') for t in range(6)])   # orthonormal: Parseval
    tot = np.array([np.sum(C[t] ** 2) for t in range(6)])
    def floor(mask_fn):
        return max(np.sqrt(max(tot[t] - mask_fn(C[t]), 0.)) / n0 for t in range(1, 6))
    for N in Ns:
        res['block'][N].append(floor(lambda c: np.sum(c[:N, :N] ** 2)))
    for K in Ks:
        res['lap'][K].append(floor(lambda c: np.sum(c.ravel()[lap_order[:K]] ** 2)))
        res['hcross'][K].append(floor(lambda c: np.sum(c.ravel()[hc_order[:K]] ** 2)))
        res['bestK'][K].append(floor(lambda c: np.sum(np.sort(c.ravel() ** 2)[::-1][:K])))
summ = {}
for key in ('block', 'lap', 'hcross', 'bestK'):
    summ[key] = {str(a): dict(median=float(np.median(v)), worst=float(np.max(v))) for a, v in res[key].items()}
out = dict(source=res['source'], n_cases=res['n_cases'], files_sha256=res['files_sha256'], summary=summ,
           note='floors as fractions (multiply by 100 for %); block = lowest N x N sine modes (N^2 modes); '
                'lap = first K modes by k1^2+k2^2; hcross = first K by k1*k2; bestK = per-field best K-term (oracle)')
OUT.write_text(json.dumps(out, indent=1))
for key in ('block', 'lap', 'hcross', 'bestK'):
    print(key, {a: (round(100 * s['median'], 4), round(100 * s['worst'], 3)) for a, s in summ[key].items()})
