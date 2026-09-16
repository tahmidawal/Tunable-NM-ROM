"""Stage 3: the one-output-interval local defect of a saved trajectory.

d_c(t_k) = || f_k - Phi_FOM(f_{k-1}) || / || u_ref_c(t_0) ||,  k = 1..5,

with Phi_FOM ten backward-Euler substeps of the SAME L=256, dt=0.005 upwind/5-point
discretisation the job used (`engines.residual`), solved here in NumPy by Newton with a
DST-I Helmholtz-preconditioned BiCGSTAB. This separates error injected inside interval k
from error inherited at t_{k-1} and amplified. `fft_tight` propagated through the same
operator gives the measurement floor.

Nothing is re-solved in the ROM: the full-order operator is applied to fields that are
already on disk.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.fft import dstn
from scipy.sparse.linalg import LinearOperator, bicgstab


def spatial(u, L):
    p = np.pad(u.reshape(L - 1, L - 1), 1)
    c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    adv = c * L * (np.where(c > 0, c - xm, xp - c) + np.where(c > 0, c - ym, yp - c))
    lap = L ** 2 * (xm + xp + ym + yp - 4 * c)
    return adv.reshape(-1), lap.reshape(-1)


def residual(u, prev, nu, dt, L):
    adv, lap = spatial(u, L)
    return u - prev + dt * (adv - nu * lap)


def jvp(u, v, nu, dt, L):
    """Directional derivative of `residual` in u; the upwind branch is held fixed,
    exactly as jax.jvp does for `jnp.where` with a non-differentiated condition."""
    pu = np.pad(u.reshape(L - 1, L - 1), 1)
    pv = np.pad(v.reshape(L - 1, L - 1), 1)
    c, xm, xp, ym, yp = pu[1:-1, 1:-1], pu[:-2, 1:-1], pu[2:, 1:-1], pu[1:-1, :-2], pu[1:-1, 2:]
    vc, vxm, vxp, vym, vyp = pv[1:-1, 1:-1], pv[:-2, 1:-1], pv[2:, 1:-1], pv[1:-1, :-2], pv[1:-1, 2:]
    pos = c > 0
    A = np.where(pos, c - xm, xp - c) + np.where(pos, c - ym, yp - c)
    dA = np.where(pos, vc - vxm, vxp - vc) + np.where(pos, vc - vym, vyp - vc)
    dadv = vc * L * A + c * L * dA
    dlap = L ** 2 * (vxm + vxp + vym + vyp - 4 * vc)
    return v + dt * (dadv.reshape(-1) - nu * dlap.reshape(-1))


def helmholtz_factory(nu, dt, L):
    k = np.arange(1, L, dtype=float)
    lam = 4 * L ** 2 * np.sin(np.pi * k / (2 * L)) ** 2
    den = 1 + dt * nu * (lam[:, None] + lam[None, :])

    def M(v):
        x = dstn(v.reshape(L - 1, L - 1), type=1, norm='ortho')
        return dstn(x / den, type=1, norm='ortho').reshape(-1)
    return M


def step(prev, nu, dt, L, ntol=1e-11, ltol=1e-10, max_newton=30):
    scale = max(np.linalg.norm(prev), 1e-300)
    M = helmholtz_factory(nu, dt, L)
    n = prev.size
    u = prev.copy()
    rn = np.linalg.norm(residual(u, prev, nu, dt, L))
    it = 0
    while rn > ntol * scale and it < max_newton and np.isfinite(rn):
        r = residual(u, prev, nu, dt, L)
        A = LinearOperator((n, n), matvec=lambda v: jvp(u, v, nu, dt, L), dtype=float)
        P = LinearOperator((n, n), matvec=M, dtype=float)
        delta, _ = bicgstab(A, -r, rtol=ltol, atol=0., maxiter=400, M=P)
        u = u + delta
        rn = np.linalg.norm(residual(u, prev, nu, dt, L))
        it += 1
    return u, it, rn / scale


def advance(field, nu, dt, L, substeps):
    u = np.ascontiguousarray(field[1:-1, 1:-1]).ravel().copy()
    worst = 0.
    for _ in range(substeps):
        u, it, rn = step(u, nu, dt, L)
        worst = max(worst, rn)
    out = np.zeros((L + 1, L + 1))
    out[1:-1, 1:-1] = u.reshape(L - 1, L - 1)
    return out, worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--restore', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--arms', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    R = Path(a.restore) / a.job
    r = json.loads((R / 'output' / 'result.json').read_text())
    L, dt = int(r['intervals']), float(r['config']['dt'])
    substeps = int(round(.05 / dt))
    refs = {e['case']: np.load(R / 'output' / e['artifact'])['fields'] for e in r['reference']}
    inv = {(x['name'], x['case']): x['artifact'] for x in r['invocations']}
    rows = []
    for arm in a.arms.split(','):
        for case in range(len(r['physical_cases'])):
            if (arm, case) not in inv:
                continue
            nu = float(r['physical_cases'][case][4])
            f = np.load(R / 'output' / inv[(arm, case)])['fields']
            n0 = np.linalg.norm(refs[case][0])
            d, w = [], []
            for k in range(1, f.shape[0]):
                adv_, worst = advance(f[k - 1], nu, dt, L, substeps)
                d.append(float(np.linalg.norm(f[k] - adv_) / n0))
                w.append(float(worst))
            rows.append(dict(job=a.job, arm=arm, case=case, local_defect=d,
                             newton_worst_relative_residual=max(w)))
            print(a.job, arm, case, ' '.join(f'{x:.3e}' for x in d), flush=True)
    Path(a.out).write_text(json.dumps(rows))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
