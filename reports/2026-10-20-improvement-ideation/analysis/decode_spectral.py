"""Could the ROM's full-field decode use a fast sine synthesis instead of the dense bank product? (CPU, NumPy only.)

The 2D full-field decode is u = G c with G stored on the output mesh (P1: 0.27 / 4.29 / 68.7 GB at 256^2 / 1024^2 /
4096^2 for R512), which is memory-bandwidth bound. Alternative: take the first K Laplacian-ordered sine coefficients
of the bank span once offline (an r x K matrix S_K), and synthesise u from S_K c with separable fast transforms, as the
spectral FOM does. Error of that truncation, on the bank's own fields G c for the lattice projections c of the 38 saved
references (r = 128, 384), in the project's error convention (max over t1..t5, relative to ||R(t0)||).
Writes analysis/decode_spectral.json.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.fft import dstn

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
sys.path.insert(0, str(ROOT / 'worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes'))
import npcheck as NC  # noqa: E402
SRC = ROOT / 'worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/c2budA/refs'
files = sorted(glob.glob(str(SRC / 'ref_ST_*.npz')))
NC.decode257(np.zeros((1, 512)), 512)
G = NC._G257[512]
n = 255
k = np.arange(1, n + 1)
KX, KY = np.meshgrid(k, k, indexing='ij')
order = np.argsort((KX ** 2 + KY ** 2).ravel(), kind='stable')
Ks = (512, 1024, 1536, 2048, 4096, 9216)
out = dict(source=str(SRC.relative_to(ROOT)), settings={})
for r in (128, 384):
    Gr = G[:, :r]
    errs = {K: [] for K in Ks}
    for f in files:
        Rf = np.load(f)['f257'].reshape(6, -1)
        n0 = np.linalg.norm(Rf[0])
        C = np.linalg.lstsq(Gr, Rf[1:].T, rcond=None)[0]           # (r, 5)
        U = (Gr @ C).T.reshape(5, 257, 257)
        S = np.stack([dstn(U[t, 1:-1, 1:-1], type=1, norm='ortho').ravel()[order] for t in range(5)])
        cum = np.cumsum(S ** 2, 1); tot = cum[:, -1]
        for K in Ks:
            errs[K].append(float(max(np.sqrt(max(tot[t] - cum[t, K - 1], 0.)) for t in range(5)) / n0))
    out['settings'][f'R{r}'] = {str(K): dict(median=float(np.median(v)), max=float(np.max(v))) for K, v in errs.items()}
    print(r, out['settings'][f'R{r}'])
Path(__file__).with_name('decode_spectral.json').write_text(json.dumps(out, indent=1))
