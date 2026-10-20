"""Controls for the bank-vs-sine diagnosis (CPU, NumPy only; reads the 38 refdv reference fields and the deployed bank).

1. Leave-one-case-out POD: for each held-out case, a POD basis from the 6 output-time snapshots of the other 37 cases
   (222 snapshots, so rank <= 222); held-out floor in the project convention, against the bank prefix and the
   Laplacian-ordered sine prefix of the same size. A data-driven yardstick for "is the bank well trained/ordered?".
2. Steepness control for the upwind-data hypothesis: Spearman correlation with s = a/nu of the bank-384 floor and of the
   sine-384 floor (if a fixed sine basis shows the same correlation, the correlation says nothing about training data).
3. Galerkin mass matrix of the bank (F2): M = sum_q w_q G(x_q)^T G(x_q) at Gauss 128^2 (r = 128, 384, 512), condition
   number raw and after symmetric diagonal scaling; against cond(A)^2 of the sine-tested coefficient matrix A = W Psi^T G.
Writes analysis/pod_control.json.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.fft import dstn
from scipy.stats import spearmanr

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
sys.path.insert(0, str(ROOT / 'worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes'))
import npcheck as NC  # noqa: E402
SRC = ROOT / 'worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/c2budA/refs'
coh = json.loads((ROOT / 'worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/local2dfb/result.json').read_text())['cohorts']
phys = {f'ref_ST_{c}_{i:03d}': p for c in ('dev6', 'val32') for i, p in enumerate(coh[c]['physical'])}
files = sorted(glob.glob(str(SRC / 'ref_ST_*.npz')))
F = np.stack([np.load(f)['f257'].reshape(6, -1) for f in files])          # (38, 6, 66049)
names = [Path(f).stem for f in files]
n0 = np.linalg.norm(F[:, 0], axis=1)

def floor(Fc, n0c, Q):
    res = Fc[1:].T - Q @ (Q.T @ Fc[1:].T)
    return float(np.max(np.linalg.norm(res, axis=0)) / n0c)

ranks = (32, 64, 128, 200)
pod = {r: [] for r in ranks}
for i in range(len(F)):
    S = np.concatenate([F[j] for j in range(len(F)) if j != i]).T             # (66049, 222)
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    for r in ranks:
        pod[r].append(floor(F[i], n0[i], U[:, :r]))
NC.decode257(np.zeros((1, 512)), 512)
G = NC._G257[512]
bank = {r: [] for r in ranks}
for r in ranks:
    U, sv, _ = np.linalg.svd(G[:, :r], full_matrices=False)
    Q = U[:, sv > sv[0] * 1e-13]
    bank[r] = [floor(F[i], n0[i], Q) for i in range(len(F))]
n = 255
k = np.arange(1, n + 1)
KX, KY = np.meshgrid(k, k, indexing='ij')
order = np.argsort((KX ** 2 + KY ** 2).ravel(), kind='stable')
sine = {r: [] for r in ranks + (384,)}
for i in range(len(F)):
    U6 = F[i].reshape(6, 257, 257)
    C = np.stack([dstn(U6[t, 1:-1, 1:-1], type=1, norm='ortho').ravel()[order] for t in range(6)])
    cum = np.cumsum(C ** 2, 1); tot = cum[:, -1]
    for r in sine:
        sine[r].append(float(max(np.sqrt(max(tot[t] - cum[t, r - 1], 0.)) for t in range(1, 6)) / n0[i]))
bvs = json.loads(Path(__file__).with_name('bank_vs_sine.json').read_text())['rows']
b384 = np.array([x['bank_384'] for x in bvs])
s = np.array([phys[nm][3] / phys[nm][4] for nm in names])
out = dict(source=str(SRC.relative_to(ROOT)), n_cases=len(F), loo_pod={}, steepness_control={}, galerkin_mass={})
for r in ranks:
    out['loo_pod'][str(r)] = dict(pod=dict(median=float(np.median(pod[r])), max=float(np.max(pod[r]))),
                                  bank=dict(median=float(np.median(bank[r])), max=float(np.max(bank[r]))),
                                  sine=dict(median=float(np.median(sine[r])), max=float(np.max(sine[r]))))
out['steepness_control'] = dict(spearman_bank384_vs_s=float(spearmanr(b384, s)[0]),
                                spearman_sine384_vs_s=float(spearmanr(np.array(sine[384]), s)[0]))
X, w = NC.gauss_rule(128)
Gq = NC.values_grads(X, 512)[0]
kk = np.arange(1, 200)
K1, K2 = np.meshgrid(kk, kk, indexing='ij')
for r in (128, 384, 512):
    Mm = (Gq[:, :r] * w[:, None]).T @ Gq[:, :r]
    d = 1 / np.sqrt(np.diag(Mm))
    o = np.argsort((K1 ** 2 + K2 ** 2).ravel(), kind='stable')[:4 * r]
    kx, ky = K1.ravel()[o].astype(float), K2.ravel()[o].astype(float)
    WP = w[:, None] * 2. * np.sin(np.pi * X[:, :1] * kx[None]) * np.sin(np.pi * X[:, 1:] * ky[None])
    sA = np.linalg.svd(WP.T @ Gq[:, :r], compute_uv=False)
    out['galerkin_mass'][str(r)] = dict(cond_M=float(np.linalg.cond(Mm)), cond_M_scaled=float(np.linalg.cond(d[:, None] * Mm * d[None])),
                                        cond_A_squared=float((sA[0] / sA[-1]) ** 2))
Path(__file__).with_name('pod_control.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'source'}, indent=1))
