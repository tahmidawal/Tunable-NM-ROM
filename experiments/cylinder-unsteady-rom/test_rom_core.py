"""Synthetic consistency tests for rom_core (no NGSolve): operators, forces, floors, errors.

Functions are analytic sines with exact gradients on random positive-weight points, so
every pre-contracted operator can be compared against direct quadrature of the field.
Run: /home/tahmid/Dev/.venv/bin/python test_rom_core.py
"""
import tempfile
from pathlib import Path

import numpy as np

import rom_core as rc

R = 6
N = 3000
rng = np.random.default_rng(0)


def field(n_fun, pts):
    a = rng.normal(size=(n_fun, 2, 2)) * 3; c = rng.normal(size=(n_fun, 2))
    arg = np.einsum('qd,jcd->qcj', pts, a) + c.T[None]
    V = np.sin(arg)
    G = np.cos(arg)[:, :, None, :] * np.transpose(a, (1, 2, 0))[None]
    return V, G


def make(d):
    pts = rng.uniform([0, 0], [2.2, 0.41], size=(N, 2)); w = rng.uniform(0.5, 1.5, N) * (2.2 * 0.41 / N)
    names = ['lift'] + [f'pod{i}' for i in range(R)] + [f'sup{i}' for i in range(R)] + ['chi_x', 'chi_y']
    V, G = field(len(names), pts)
    P = np.sin(pts @ rng.normal(size=(2, R)) + rng.normal(size=R))
    bV, bG = field(4, pts)
    op = rng.uniform([2.2, 0], [2.2, 0.41], size=(40, 2)); ow = np.full(40, 0.41 / 40)
    oV, _ = field(len(names), op); obV, _ = field(4, op)
    np.savez(d / 'quadrature.npz', points=pts, weights=w, outlet_points=op, outlet_weights=ow,
             inlet_points=op * 0, inlet_weights=ow)
    np.savez(d / 'basis.npz', names=np.array(names), V=V, G=G, P=P, probes=rng.normal(size=(R, 2)),
             outlet_V=oV, frozen_bank_V=bV, frozen_bank_G=bG, frozen_bank_outlet_V=obV)


def direct_conv(arm, a, test_V):
    u = arm.lV + arm.Vb @ a; gu = arm.lG + arm.Gb @ a
    cu = np.einsum('qcd,qd->qc', gu, u)
    return np.einsum('q,qcj,qc->j', arm.ex.w, test_V, cu), u, gu


def main():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); make(d)
        ex = rc.Export(d)
        for kind, r, key in [('pod', 4, None), ('bank', 4, 'frozen')]:
            arm = rc.Arm(ex, kind, r, key)
            assert np.allclose(arm.M, np.eye(arm.d), atol=1e-10), 'orthonormal basis'
            a = rng.normal(size=arm.d) * 0.3
            dc, u, gu = direct_conv(arm, a, arm.Vb)
            e1 = np.max(np.abs(arm.conv(a) - dc)) / np.max(np.abs(dc))
            # Jacobian by central differences
            h = 1e-6; J = np.stack([(arm.conv(a + h * e) - arm.conv(a - h * e)) / (2 * h) for e in np.eye(arm.d)], 1)
            e2 = np.max(np.abs(J - arm.dconv(a))) / np.max(np.abs(J))
            # force functional vs direct quadrature with chi as test function
            nu, adot, c = 0.01, rng.normal(size=arm.d), rng.normal(size=arm.dp)
            p = arm.Pb @ c; ut = arm.Vb @ adot
            fc, _, _ = direct_conv(arm, a, arm.chiV)
            divchi = arm.chiG[:, 0, 0, :] + arm.chiG[:, 1, 1, :]
            Fd = -(np.einsum('q,qcj,qc->j', ex.w, arm.chiV, ut) + nu * np.einsum('q,qcdj,qcd->j', ex.w, arm.chiG, gu)
                   + fc - np.einsum('q,q,qj->j', ex.w, p, divchi)) / 0.05
            e3 = np.max(np.abs(arm.force(nu, a, adot, c) - Fd)) / np.max(np.abs(Fd))
            # floors and errors from streamed inner products of a synthetic reference
            astar = rng.normal(size=arm.d) * 0.2
            uref = arm.lV + arm.Vb @ astar + 1e-3 * np.sin(5 * ex.V[..., ex.col('pod0')])
            pref = arm.Pb @ rng.normal(size=arm.dp)
            proj = {'uv': np.einsum('q,qc,qcj->j', ex.w, uref, ex.V)[None],
                    'uv_frozen': np.einsum('q,qc,qcj->j', ex.w, uref, ex.bank['frozen'][0])[None],
                    'uu': np.array([np.einsum('q,qc,qc->', ex.w, uref, uref)]),
                    'pp': np.array([ex.w @ (pref * pref)]), 'pv': np.einsum('q,q,qk->k', ex.w, pref, ex.P)[None]}
            err = arm.errors(proj, astar[None], np.linalg.solve(arm.Gp, proj['pv'][0, :arm.dp])[None], [0])
            de = np.sqrt(np.einsum('q,qc,qc->', ex.w, uref - arm.lV - arm.Vb @ astar, uref - arm.lV - arm.Vb @ astar) / proj['uu'][0])
            e4 = abs(err['e_u'][0] - de) / de
            fl = arm.floors(proj)
            coef = np.linalg.lstsq((arm.Vb * np.sqrt(ex.w)[:, None, None]).reshape(-1, arm.d),
                                   ((uref - arm.lV) * np.sqrt(ex.w)[:, None]).reshape(-1), rcond=None)[0]
            res = uref - arm.lV - arm.Vb @ coef
            df = np.sqrt(np.einsum('q,qc,qc->', ex.w, res, res) / proj['uu'][0])
            e5 = abs(fl['velocity_with_supremizers'][0] - df) / df
            print(kind, f'conv {e1:.1e} jac {e2:.1e} force {e3:.1e} err {e4:.1e} floor {e5:.1e} p_floor {fl["pressure"][0]:.1e}')
            assert max(e1, e3) < 1e-9 and max(e4, e5) < 1e-7 and e2 < 1e-6  # e4/e5: norm-expansion cancellation
            # Newton consistency on one BDF step
            a1, c1, it, hist = arm.step(nu, 1e3, arm.M @ a * 1e3, a, np.zeros(arm.dp))
            assert hist[-1] < 1e-12 and it < 20
            # consistent initial pressure: residual of the differentiated constraint system
            A, Cc, F, pr, its = arm.rollout(nu, 1e-3, a, 3)
            S = np.block([[arm.M, -arm.B.T], [arm.B, np.zeros((arm.dp, arm.dp))]])
            F0 = nu * (arm.K @ a + arm.KL) + arm.conv(a)
            adot0 = np.linalg.solve(S, np.r_[-F0, np.zeros(arm.dp)])[:arm.d]
            assert np.allclose(arm.M @ adot0 + F0 - arm.B.T @ Cc[0], 0, atol=1e-10 * max(1, np.abs(F0).max()))
            assert np.allclose(arm.B @ adot0, 0, atol=1e-10)
    print('all rom_core synthetic checks passed')


if __name__ == '__main__':
    main()
