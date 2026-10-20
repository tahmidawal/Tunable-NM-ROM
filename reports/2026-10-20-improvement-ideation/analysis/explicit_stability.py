"""Kill-test proxy for an explicit-advection (IMEX / ETD) Galerkin step: how stiff is the reduced advection operator?

For the deployed 2D bank (dn256b, rotation_R512; NumPy evaluator vendored in jcp-regimes) at r = 128 (M = 512, Gauss 96^2)
and r = 384 (M = 1536, Gauss 128^2), continuum sine tests ordered by k1^2 + k2^2 (production orders by the mesh's
discrete eigenvalues and uses mesh-projected A; the selected sets differ by a few modes, so this is a proxy):
  A = W Psi^T G (M x r), A = Q Rm (thin QR), Galerkin-in-test-coordinates form  Rm c' = -Q^T [N(c) + nu Lambda A c].
  Advection Jacobian  Ja(c) = -Rm^{-1} Q^T dN/dc;  diffusion operator  Ld = -nu Rm^{-1} Q^T Lambda A.
States c: lattice least-squares projections of the saved references (refdv store, 38 dev6+val32 cases) at t = 0, 0.1,
0.25 onto the first r bank functions. Reports the spectral radius of Ja (rho_a), the largest real part, and the
heuristic explicit limit dt_a = 2.8 / rho_a (2.8 is a heuristic constant near the RK4 stability intercepts 2*sqrt(2) on the imaginary axis and about 2.785 on the negative real axis) against the spectral FOM's steps 0.0125, 0.025;
and rho(Ld) to show what an implicit/exponential treatment of diffusion buys. A proxy, not a stability proof:
ETDRK4 stability for L + N is not exactly RK4 on N, and Ja is non-normal.
Writes analysis/explicit_stability.json.
"""
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
LANE = ROOT / 'worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes'
sys.path.insert(0, str(LANE))
import npcheck as NC  # noqa: E402

SRC = ROOT / 'worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/c2budA/refs'
PHYS = ROOT / 'worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/local2dfb/result.json'
coh = json.loads(PHYS.read_text())['cohorts']
phys = {f'ref_ST_{c}_{i:03d}': p for c in ('dev6', 'val32') for i, p in enumerate(coh[c]['physical'])}
files = sorted(glob.glob(str(SRC / 'ref_ST_*.npz')))
NC.decode257(np.zeros((1, 512)), 512)
G257 = NC._G257[512]
out = dict(source=str(SRC.relative_to(ROOT)), bank=str(NC.CKPT.relative_to(ROOT)), settings={})
for r, p in ((128, 96), (384, 128)):
    M = 4 * r
    kk = np.arange(1, 200)
    KX, KY = np.meshgrid(kk, kk, indexing='ij')
    o = np.argsort((KX ** 2 + KY ** 2).ravel(), kind='stable')[:M]
    kx, ky = KX.ravel()[o].astype(float), KY.ravel()[o].astype(float)
    lam = np.pi ** 2 * (kx ** 2 + ky ** 2)
    X, w = NC.gauss_rule(p)
    Gq, Gx, Gy = NC.values_grads(X, r)
    Gs = Gx + Gy
    WP = w[:, None] * 2. * np.sin(np.pi * X[:, :1] * kx[None]) * np.sin(np.pi * X[:, 1:] * ky[None])   # (m, M)
    A = WP.T @ Gq
    Q, Rm = np.linalg.qr(A)
    LA = Q.T @ (lam[:, None] * A)
    Gl = G257[:, :r]
    rows = []
    for f in files:
        name = Path(f).stem
        nu = phys[name][4]
        Rf = np.load(f)['f257'].reshape(6, -1)
        C = np.linalg.lstsq(Gl, Rf[[0, 2, 5]].T, rcond=None)[0].T
        Ld = -nu * np.linalg.solve(Rm, LA)
        rho_d = float(np.max(np.abs(np.linalg.eigvals(Ld))))
        for ti, c in zip((0., .1, .25), C):
            u, s = Gq @ c, Gs @ c
            dN = WP.T @ (s[:, None] * Gq + u[:, None] * Gs)
            Ja = -np.linalg.solve(Rm, Q.T @ dN)
            ev = np.linalg.eigvals(Ja)
            rows.append(dict(case=name, nu=nu, amp=phys[name][3], t=ti, rho_a=float(np.max(np.abs(ev))),
                             max_re=float(np.max(ev.real)), rho_d=rho_d, dt_a=float(2.8 / np.max(np.abs(ev)))))
    ra = np.array([x['rho_a'] for x in rows]); da = np.array([x['dt_a'] for x in rows]); rd = np.array([x['rho_d'] for x in rows])
    out['settings'][f'R{r}_M{M}_gauss{p}'] = dict(
        rho_a=dict(median=float(np.median(ra)), max=float(ra.max())),
        dt_a=dict(median=float(np.median(da)), min=float(da.min())),
        frac_states_dt_a_ge_0p0125=float(np.mean(da >= .0125)), frac_states_dt_a_ge_0p025=float(np.mean(da >= .025)),
        rho_d=dict(median=float(np.median(rd)), max=float(rd.max())),
        dt_d_explicit=dict(median=float(np.median(2.8 / rd)), min=float((2.8 / rd).min())),
        max_re=dict(median=float(np.median([x['max_re'] for x in rows])), max=float(max(x['max_re'] for x in rows))),
        rows=rows)
    print(r, {k: v for k, v in out['settings'][f'R{r}_M{M}_gauss{p}'].items() if k != 'rows'}, flush=True)
Path(__file__).with_name('explicit_stability.json').write_text(json.dumps(out, indent=1))
