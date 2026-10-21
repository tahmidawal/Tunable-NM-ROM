"""A2 (Amendment 1): analytic wall curvature, sine tail and tail wall-share of decoded fields. CPU, NumPy only.

Per case: decoded ROM v = G_384 W (saved r = 384 LSPG-CN-R run), bank projection P_384 u_ref, E0 reference u_ref.
  kappa_a(v)  RMS over the 4 x 255 interior wall nodes and t1..t5 of the normal second derivative of the bank field,
              by a central difference (+-delta, delta = 1e-4) of the bank's analytic gradient, / RMS interior 5-point
              Laplacian of u_ref (the true solution has kappa_a = 0).  Descriptive.
  tau_K(v)    max over t1..t5 of ||(I - S_K) v|| / ||u_ref(t0)||, floors clipped at 1e-14; slope beta over K in [2304, 16384].
  omega(f)    energy of f - S_4096 f in the 8-node band next to the walls / its total energy, pooled over t1..t5.
Controls: identity (closed-form envelope curvature, random-sine zero curvature); empirical C-A2w+ (envelope bump must be
wall-localised, median omega >= 0.5) and C-A2w- (interior bump, median omega <= 0.25).
Decision (heuristic): RUN if median tau4096(v)/tau4096(u_ref) >= 3 and median omega(v) >= 0.5; else DEFER;
INCONCLUSIVE if a control fails. Writes checks/a2_wall.json.
"""
from __future__ import annotations

import json

import numpy as np

import fcommon as F

KS = (1024, 2304, 4096, 9216, 16384, 36864)
FIT = (2304, 4096, 9216, 16384)
COHORTS = ('dev6', 'val32', 'N0s', 'C1', 'C2')
H = 1 / 256.
DELTA = 1e-4
CLIP = 1e-14
BAND = 8
X = np.arange(257) / 256.


def wall_points():
    """(4*255, 2) interior wall nodes and their inward normals (unit vectors)."""
    xi = X[1:-1]
    z, o = np.zeros_like(xi), np.ones_like(xi)
    pts = np.concatenate([np.stack([z, xi], 1), np.stack([o, xi], 1), np.stack([xi, z], 1), np.stack([xi, o], 1)])
    nrm = np.concatenate([np.tile([1., 0.], (255, 1)), np.tile([-1., 0.], (255, 1)),
                          np.tile([0., 1.], (255, 1)), np.tile([0., -1.], (255, 1))])
    return pts, nrm


def bank_dnn(W):
    """Normal second derivative at the interior wall nodes of v = G_384 W for W (6, 384): (6, 1020)."""
    P, T = F.NC.bank()
    pts, nrm = wall_points()
    dd = []
    for sgn in (+1, -1):
        _, gx, gy = F.NC.NB.features_grad(P, pts + sgn * DELTA * nrm)
        dd.append(nrm[:, :1] * gx + nrm[:, 1:] * gy)                    # inward-normal derivative per feature
    D2 = ((dd[0] - dd[1]) / (2 * DELTA)) @ T[:, :384]                   # (n, 384)
    return np.asarray(W) @ D2.T


def lap_rms(u):
    V = np.asarray(u)[1:]
    lap = (V[:, 2:, 1:-1] + V[:, :-2, 1:-1] + V[:, 1:-1, 2:] + V[:, 1:-1, :-2] - 4 * V[:, 1:-1, 1:-1]) / H ** 2
    return float(np.sqrt(np.mean(lap ** 2)))


def tails(V, n0):
    C = F.dst_coef(np.asarray(V)[1:])
    cum = np.cumsum(C ** 2, 1)
    tot = cum[:, -1]
    tau = {K: max(float(np.max(np.sqrt(np.maximum(tot - cum[:, K - 1], 0.)))) / n0, CLIP) for K in KS}
    beta = float(np.polyfit(np.log(FIT), np.log([tau[K] for K in FIT]), 1)[0])
    return tau, beta


_band = np.zeros((257, 257), bool)
_band[:BAND + 1, :] = _band[-BAND - 1:, :] = _band[:, :BAND + 1] = _band[:, -BAND - 1:] = True


def omega(V):
    V = np.asarray(V)[1:]
    hp = V - F.sine_truncate(V, 4096)
    tot = float(np.sum(hp ** 2))
    return float(np.sum(hp[:, _band] ** 2) / tot) if tot > 0 else float('nan')


def identity_checks():
    """(i) closed form of the envelope curvature through the same central-difference-of-gradient path, using an exact
    gradient of e(x,y) = 16 x(1-x) y(1-y); (ii) zero wall curvature of a random sine sum, by the same path."""
    pts, nrm = wall_points()
    def grad_env(p):
        x, y = p[:, 0], p[:, 1]
        return np.stack([16 * (1 - 2 * x) * y * (1 - y), 16 * x * (1 - x) * (1 - 2 * y)], 1)
    d2 = (np.einsum('nd,nd->n', nrm, grad_env(pts + DELTA * nrm)) - np.einsum('nd,nd->n', nrm, grad_env(pts - DELTA * nrm))) / (2 * DELTA)
    t = np.where(nrm[:, 0] != 0, pts[:, 1], pts[:, 0])
    e1 = float(np.max(np.abs(d2 - (-32 * t * (1 - t)))) / 8.)
    rng = np.random.default_rng(7)
    idx = rng.choice(2000, 32, replace=False)
    kx, ky = F._KX.ravel()[F.LAP_ORDER[idx]], F._KY.ravel()[F.LAP_ORDER[idx]]
    cf = rng.standard_normal(32)
    def grad_sine(p):
        x, y = p[:, :1], p[:, 1:]
        gx = (cf * np.pi * kx * np.cos(np.pi * kx * x) * np.sin(np.pi * ky * y)).sum(1)
        gy = (cf * np.pi * ky * np.sin(np.pi * kx * x) * np.cos(np.pi * ky * y)).sum(1)
        return np.stack([gx, gy], 1)
    d2s = (np.einsum('nd,nd->n', nrm, grad_sine(pts + DELTA * nrm)) - np.einsum('nd,nd->n', nrm, grad_sine(pts - DELTA * nrm))) / (2 * DELTA)
    F2 = (cf[None, None] * np.sin(np.pi * kx * X[:, None, None]) * np.sin(np.pi * ky * X[None, :, None])).sum(-1)
    lap = (F2[2:, 1:-1] + F2[:-2, 1:-1] + F2[1:-1, 2:] + F2[1:-1, :-2] - 4 * F2[1:-1, 1:-1]) / H ** 2
    e2 = float(np.max(np.abs(d2s)) / np.sqrt(np.mean(lap ** 2)))
    return dict(envelope_closed_form_rel=e1, envelope_ok=e1 <= 1e-6, sine_zero_rel=e2, sine_ok=e2 <= 1e-8)


def bank_path_check(W, dv=1e-3):
    """Independent path: second difference of bank VALUES across the wall (bank defined outside the box) vs bank_dnn."""
    P, T = F.NC.bank()
    pts, nrm = wall_points()
    val = lambda q: F.NC.NB.features_grad(P, q)[0] @ T[:, :384]
    D2v = (val(pts + dv * nrm) - 2 * val(pts) + val(pts - dv * nrm)) / dv ** 2
    a, b = np.asarray(W) @ D2v.T, bank_dnn(W)
    e = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    return dict(bank_values_vs_gradient_path_rel=e, bank_path_ok=e <= 1e-3)


def main():
    Z = np.load(F.ROMDIR / 'R_R384_LSPG_CNR_1.npz', allow_pickle=True)
    G = F.G(384)
    Q = F.orth(G)
    env = 16 * np.outer(X * (1 - X), X * (1 - X))
    rows, skipped = {}, []
    for i, cid in enumerate(Z['case_ids']):
        cid = str(cid)
        coh, c = cid.split('|')
        if coh not in COHORTS:
            continue
        u, m = F.load_ref(coh, int(c))
        if not m['eligible'] or not bool(Z['valid'][i]):
            skipped.append(dict(case=cid, eligible=m['eligible'], valid=bool(Z['valid'][i])))
            continue
        n0 = m['n0']
        v = (Z['W'][i] @ G.T).reshape(6, 257, 257)
        cp = np.linalg.lstsq(G, u.reshape(6, -1).T, rcond=None)[0].T          # (6, 384) projection coefficients
        pu = (G @ cp.T).T.reshape(6, 257, 257)
        L = lap_rms(u)
        amp = 0.01 * np.abs(u[0]).max()
        d = m['desc']
        hp = amp * np.broadcast_to(env, (6, 257, 257))
        hm = amp * np.broadcast_to(np.exp(-((X[:, None] - 0.5) ** 2 + (X[None] - 0.5) ** 2) / (2 * 0.03 ** 2)), (6, 257, 257))
        row = dict(case=cid, cohort=coh, s=d['a'] / d['nu'])
        for name, W_, V in (('rom', Z['W'][i], v), ('proj', cp, pu)):
            dn = bank_dnn(W_)[1:]
            tau, beta = tails(V, n0)
            row[name] = dict(kappa_a=float(np.sqrt(np.mean(dn ** 2)) / L), beta=beta, tau={str(K): t for K, t in tau.items()},
                             omega=omega(V))
        tau, beta = tails(u, n0)
        row['ref'] = dict(beta=beta, tau={str(K): t for K, t in tau.items()}, omega=omega(u))
        row['ctrl_omega_plus'] = omega(hp)
        row['ctrl_omega_minus'] = omega(hm)
        rows[cid] = row
    def grp(cohs):
        R = [r for r in rows.values() if r['cohort'] in cohs]
        med = lambda f: float(np.median([f(r) for r in R]))
        return dict(n=len(R),
                    kappa_a_med={k: med(lambda r: r[k]['kappa_a']) for k in ('rom', 'proj')},
                    beta_med={k: med(lambda r: r[k]['beta']) for k in ('rom', 'proj', 'ref')},
                    tau4096_med={k: med(lambda r: r[k]['tau']['4096']) for k in ('rom', 'proj', 'ref')},
                    omega_med={k: med(lambda r: r[k]['omega']) for k in ('rom', 'proj', 'ref')},
                    tau4096_ratio_rom_ref_med=med(lambda r: r['rom']['tau']['4096'] / r['ref']['tau']['4096']),
                    tau4096_ratio_proj_ref_med=med(lambda r: r['proj']['tau']['4096'] / r['ref']['tau']['4096']),
                    ctrl_omega_plus_med=med(lambda r: r['ctrl_omega_plus']),
                    ctrl_omega_minus_med=med(lambda r: r['ctrl_omega_minus']))
    groups = {'in_range': grp(F.IN_RANGE), **{c: grp((c,)) for c in ('N0s', 'C1', 'C2')}}
    g = groups['in_range']
    ident = identity_checks()
    ident.update(bank_path_check(Z['W'][0]))
    ctrl = dict(identity=ident, omega_plus_ok=g['ctrl_omega_plus_med'] >= 0.5, omega_minus_ok=g['ctrl_omega_minus_med'] <= 0.25)
    ctrl['controls_pass'] = bool(ident['envelope_ok'] and ident['sine_ok'] and ident['bank_path_ok'] and ctrl['omega_plus_ok'] and ctrl['omega_minus_ok'])
    crit = dict(tail_3x=bool(g['tau4096_ratio_rom_ref_med'] >= 3), wall_share_half=bool(g['omega_med']['rom'] >= 0.5))
    decision = ('INCONCLUSIVE' if not ctrl['controls_pass'] else ('RUN' if crit['tail_3x'] and crit['wall_share_half'] else 'DEFER'))
    out = dict(design='DESIGN.md Amendment 1, A2', provenance=F.provenance(), source=str(F.ROMDIR / 'R_R384_LSPG_CNR_1.npz'),
               skipped=skipped, controls=ctrl, criteria=crit, decision_arm_c=decision, groups=groups, rows=rows)
    F.write_json('a2_wall', out)
    print(json.dumps(dict(controls=ctrl, criteria=crit, decision=decision, groups=groups), indent=1))


if __name__ == '__main__':
    main()
