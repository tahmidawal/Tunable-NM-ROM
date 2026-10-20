"""Learned bank against a fixed sine basis, per case, on the 38 saved 2D references (CPU, NumPy only).

For each dev6+val32 reference (quadrature-study refdv store, mesh 8192; same f257 lattice and floor convention as
jcp-regimes rscore.projection_floor: max over t_1..t_5 of ||R - Pi R|| / ||R(t0)||):
  * bank floor at r = 128, 256, 384, 512 (deployed dn256b bank + rotation_R512, NumPy evaluator vendored in jcp-regimes);
  * sine floors (Laplacian-ordered first K modes) and the exact smallest prefix K that matches the bank floor (ratio K/r);
  * hybrid floors: span(bank r=384 + first K Laplacian sines), K = 128, 512 (cheap test of sine enrichment).
Writes analysis/bank_vs_sine.json.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.fft import dstn

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
LANE = ROOT / 'worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes'
sys.path.insert(0, str(LANE))
import npcheck as NC  # noqa: E402  (vendored NumPy bank, read-only use)

SRC = ROOT / 'worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/c2budA/refs'
files = sorted(glob.glob(str(SRC / 'ref_ST_*.npz')))
NC.decode257(np.zeros((1, 512)), 512)
G = NC._G257[512]                                     # (66049, 512), zero walls
bases = {}
for r in (128, 256, 384, 512):
    U, s, _ = np.linalg.svd(G[:, :r], full_matrices=False)
    bases[r] = U[:, s > s[0] * 1e-13]
# sine columns on the lattice (zero walls), Laplacian order, orthonormal on the lattice
n = 255
k = np.arange(1, n + 1)
KX, KY = np.meshgrid(k, k, indexing='ij')
order = np.argsort((KX ** 2 + KY ** 2).ravel(), kind='stable')
x = np.arange(257) / 256.
def sine_cols(K):
    kx, ky = KX.ravel()[order[:K]], KY.ravel()[order[:K]]
    S = np.sin(np.pi * x[:, None] * kx[None]) [:, None, :] * np.sin(np.pi * x[:, None] * ky[None])[None, :, :]
    return S.reshape(257 * 257, K)
hyb = {}
for K in (128, 512):
    Q_, R_ = np.linalg.qr(np.concatenate([bases[384], sine_cols(K)], 1))
    hyb[K] = Q_[:, np.abs(np.diag(R_)) > 1e-10]
Ks = np.unique(np.r_[np.arange(16, 512, 16), np.arange(512, 4097, 64)])
def floor_proj(Rf, Qb):
    R = Rf.reshape(6, -1)
    n0 = np.linalg.norm(R[0])
    res = R[1:].T - Qb @ (Qb.T @ R[1:].T)
    return float(np.max(np.linalg.norm(res, axis=0)) / n0)
rows = []
for f in files:
    Rf = np.load(f)['f257']
    row = dict(case=Path(f).stem)
    for r, Qb in bases.items():
        row[f'bank_{r}'] = floor_proj(Rf, Qb)
    for K, Qb in hyb.items():
        row[f'bank384_plus_sine{K}'] = floor_proj(Rf, Qb)
    n0 = np.linalg.norm(Rf[0])
    C = np.stack([dstn(Rf[t, 1:-1, 1:-1], type=1, norm='ortho').ravel()[order] for t in range(6)])
    cum = np.cumsum(C ** 2, axis=1); tot = cum[:, -1]
    sfl = {int(K): float(max(np.sqrt(max(tot[t] - cum[t, K - 1], 0.)) / n0 for t in range(1, 6))) for K in Ks}
    row['sine'] = sfl
    # exact smallest Laplacian-ordered prefix K whose floor (max over t1..t5) is <= the bank floor
    tail = np.sqrt(np.maximum(tot[1:, None] - cum[1:], 0.)) / n0          # (5, n_modes): floor after keeping K = idx+1
    fl_all = tail.max(axis=0)
    row['sine_all_floor_by_K'] = None
    for r in (128, 256, 384, 512):
        idx = np.nonzero(fl_all <= row[f'bank_{r}'])[0]
        row[f'sineK_to_match_bank_{r}'] = int(idx[0] + 1) if idx.size else None
    row['_fl_all'] = fl_all
    rows.append(row)
summ = {}
for key in [f'bank_{r}' for r in bases] + [f'bank384_plus_sine{K}' for K in hyb]:
    v = np.array([r_[key] for r_ in rows]); summ[key] = dict(median=float(np.median(v)), max=float(v.max()))
for r in (128, 256, 384, 512):
    v = np.array([r_[f'sineK_to_match_bank_{r}'] or 9999 for r_ in rows], float)
    summ[f'ratio_sineK_over_r_{r}'] = dict(median=float(np.median(v) / r), min=float(v.min() / r), max=float(v.max() / r),
                                            n_unmatched_within_4096=int(np.sum(v == 9999)))
# worst-case-driven: K needed for the sine worst floor to reach the bank worst floor
worst_all = np.max(np.stack([r_['_fl_all'] for r_ in rows]), axis=0)
for r in (128, 256, 384, 512):
    bw = summ[f'bank_{r}']['max']
    idx = np.nonzero(worst_all <= bw)[0]
    summ[f'sineK_worstcase_match_bank_{r}'] = int(idx[0] + 1) if idx.size else None
for r_ in rows:
    del r_['_fl_all'], r_['sine_all_floor_by_K']
out = dict(source=str(SRC.relative_to(ROOT)), bank=str(NC.CKPT.relative_to(ROOT)), n_cases=len(rows), summary=summ, rows=rows)
Path(__file__).with_name('bank_vs_sine.json').write_text(json.dumps(out, indent=1))
print(json.dumps(summ, indent=1))
