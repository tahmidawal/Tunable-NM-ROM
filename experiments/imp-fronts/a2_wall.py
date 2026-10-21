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
DELTAS = (1e-4, 5e-5, 2.5e-5)                 # Amendment 2: gradient-difference convergence
OMEGA_FLOOR = 1e-24
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


def bank_dnn(W, delta=DELTA):
    """Normal second derivative at the interior wall nodes of v = G_384 W for W (6, 384): (6, 1020)."""
    P, T = F.NC.bank()
    pts, nrm = wall_points()
    dd = []
    for sgn in (+1, -1):
        _, gx, gy = F.NC.NB.features_grad(P, pts + sgn * delta * nrm)
        dd.append(nrm[:, :1] * gx + nrm[:, 1:] * gy)                    # inward-normal derivative per feature
    D2 = ((dd[0] - dd[1]) / (2 * delta)) @ T[:, :384]                   # (n, 384)
    return np.asarray(W) @ D2.T


def lap_rms(u):
    V = np.asarray(u)[1:]
    lap = (V[:, 2:, 1:-1] + V[:, :-2, 1:-1] + V[:, 1:-1, 2:] + V[:, 1:-1, :-2] - 4 * V[:, 1:-1, 1:-1]) / H ** 2
    return float(np.sqrt(np.mean(lap ** 2)))


def tails(V, n0):
    """Direct summation of the retained DST coefficients (Amendment 6), clipped at CLIP."""
    tn = F.tail_norms(np.asarray(V)[1:], KS)
    tau = {K: max(float(np.max(tn[K])) / n0, CLIP) for K in KS}
    beta = float(np.polyfit(np.log(FIT), np.log([tau[K] for K in FIT]), 1)[0])
    return tau, beta


_band = np.zeros((257, 257), bool)
_band[:BAND + 1, :] = _band[-BAND - 1:, :] = _band[:, :BAND + 1] = _band[:, -BAND - 1:] = True


def omega(V):
    """Wall share of the rank >= 4096 high-pass energy (synthesised directly from its own DST coefficients), pooled over
    t1..t5; None (unavailable) if the field's total energy is zero / non-finite or the tail energy is below
    OMEGA_FLOOR of it (Amendments 2, 3, 6)."""
    V = np.asarray(V, float)[1:]
    E = float(np.sum(V ** 2))
    if not (np.isfinite(E) and E > 0):
        return None
    hp = F.sine_highpass(V, 4096)
    tot = float(np.sum(hp ** 2))
    if not (np.isfinite(tot) and tot >= OMEGA_FLOOR * E):
        return None
    return float(np.sum(hp[:, _band] ** 2) / tot)


def fd_kappa(V):
    """The RETRACTED one-sided FD wall-curvature ratio of the original A2 (DESIGN §2 A2), replayed for disclosure only."""
    V = np.asarray(V)[1:]
    w = []
    for A_ in (V, V[:, ::-1], np.swapaxes(V, 1, 2), np.swapaxes(V, 1, 2)[:, ::-1]):
        w.append((2 * A_[:, 0, 1:-1] - 5 * A_[:, 1, 1:-1] + 4 * A_[:, 2, 1:-1] - A_[:, 3, 1:-1]) / H ** 2)
    return float(np.sqrt(np.mean(np.concatenate(w) ** 2))) / lap_rms(np.concatenate([np.zeros_like(V[:1]), V]))


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


def bank_path_check(W, L):
    """Amendment 6: second difference of bank VALUES across the wall at dv = 1e-4 and 5e-5 vs the gradient-difference
    estimate at delta = 2.5e-5; raw derivatives, absolute infinity norm over interior wall nodes and t1..t5."""
    P, T = F.NC.bank()
    pts, nrm = wall_points()
    val = lambda q: F.NC.NB.features_grad(P, q)[0] @ T[:, :384]
    a = {dv: (np.asarray(W) @ ((val(pts + dv * nrm) - 2 * val(pts) + val(pts - dv * nrm)) / dv ** 2).T)[1:] for dv in (1e-4, 5e-5)}
    b = bank_dnn(W, DELTAS[2])[1:]
    bound = 1e-3 * float(np.max(np.abs(b))) + 1e-6 * L
    aR = (4 * a[5e-5] - a[1e-4]) / 3                                      # Amendment 7: Richardson
    gap = float(np.max(np.abs(aR - b)))
    gap_dv = float(np.max(np.abs(a[1e-4] - a[5e-5])))
    a3 = (np.asarray(W) @ ((val(pts + 1e-3 * nrm) - 2 * val(pts) + val(pts - 1e-3 * nrm)) / 1e-6).T)[1:]
    return dict(bank_path_fixture='dev6|0', bank_path_gap=gap, bank_path_gap_dv=gap_dv, bank_path_bound=bound,
                bank_path_ok=bool(gap <= bound), gap_unextrapolated_5em5=float(np.max(np.abs(a[5e-5] - b))),
                original_dv_1em3_gap=float(np.max(np.abs(a3 - b))),
                note='Amendment 6 respecification; the original dv = 1e-3 check failed (truncation of the value path)')


def main():
    Z = np.load(F.ROMDIR / 'R_R384_LSPG_CNR_1.npz', allow_pickle=True)
    G = F.G(384)
    Gp = np.linalg.pinv(G, rcond=1e-13)                                   # one SVD policy (cut 1e-13, as F.orth)
    env = 16 * np.outer(X * (1 - X), X * (1 - X))
    rows, excluded, invalid = {}, [], []
    idx = {str(c): i for i, c in enumerate(Z['case_ids'])}
    for coh, c in F.all_cases(COHORTS):                                   # eligible references, independently of the ROM archive
        cid = f'{coh}|{c}'
        u, m = F.load_ref(coh, c)
        if not m['eligible']:
            excluded.append(cid)
            continue
        n0 = m['n0']
        L = lap_rms(u)
        L_ok = bool(np.isfinite(L) and L > 0)
        cp = (Gp @ u.reshape(6, -1).T).T                                   # (6, 384) projection coefficients
        pu = (G @ cp.T).T.reshape(6, 257, 257)
        amp = 0.01 * np.abs(u[0]).max()
        d = m['desc']
        hp = amp * np.broadcast_to(env, (6, 257, 257))
        hm = amp * np.broadcast_to(np.exp(-((X[:, None] - 0.5) ** 2 + (X[None] - 0.5) ** 2) / (2 * 0.03 ** 2)), (6, 257, 257))
        i = idx.get(cid)
        reason = None
        if i is None:
            reason = 'no saved ROM run'
        elif not bool(Z['valid'][i]):
            reason = 'archive valid flag false'
        elif not np.isfinite(Z['W'][i]).all():
            reason = 'non-finite coefficients'
        row = dict(case=cid, cohort=coh, s=d['a'] / d['nu'], rom_effective_valid=reason is None, rom_reject_reason=reason,
                   laplacian_scale_ok=L_ok)
        fields = [('proj', cp, pu)]
        if reason is None:
            fields.insert(0, ('rom', Z['W'][i], (Z['W'][i] @ G.T).reshape(6, 257, 257)))
        else:
            invalid.append(cid)
            row['rom'] = None
        for name, W_, V in fields:
            tau, beta = tails(V, n0)
            row[name] = dict(beta=beta, tau={str(K): t for K, t in tau.items()}, omega=omega(V))
            if L_ok:
                kk = [float(np.sqrt(np.mean(bank_dnn(W_, dl)[1:] ** 2)) / L) for dl in DELTAS]
                conv = abs(kk[2] - kk[1]) <= 1e-3 * max(abs(kk[1]), 1e-6)
                row[name].update(kappa_a=kk[2] if conv else None, kappa_a_deltas=kk, kappa_converged=bool(conv))
            else:
                row[name].update(kappa_a=None, kappa_a_deltas=None, kappa_converged=False)
        tau, beta = tails(u, n0)
        row['ref'] = dict(beta=beta, tau={str(K): t for K, t in tau.items()}, omega=omega(u))
        if coh in F.IN_RANGE and L_ok:                                   # retracted-control replay (disclosure)
            vplus = u + amp * env[None]
            row['retracted_fd_control_fires'] = bool(fd_kappa(vplus) >= 10 * fd_kappa(u))
        row['ctrl_omega_plus'] = omega(hp)
        row['ctrl_omega_minus'] = omega(hm)
        rows[cid] = row

    def grp(cohs):
        R = [r for r in rows.values() if r['cohort'] in cohs]
        def med(f):
            v = []
            for r in R:
                try:
                    x = f(r)
                except (TypeError, KeyError):
                    x = None
                if x is not None:
                    v.append(x)
            return float(np.median(v)) if v else None
        get = lambda k, key: (lambda r: r[k][key] if r[k] is not None else None)
        return dict(n_eligible=len(R), n_rom_valid=sum(r['rom'] is not None for r in R),
                    n_laplacian_unavailable=sum(not r['laplacian_scale_ok'] for r in R),
                    kappa_a_med={k: med(get(k, 'kappa_a')) for k in ('rom', 'proj')},
                    kappa_unconverged={k: sum(r[k] is not None and not r[k]['kappa_converged'] for r in R) for k in ('rom', 'proj')},
                    omega_available={k: sum(r[k] is not None and r[k]['omega'] is not None for r in R) for k in ('rom', 'proj', 'ref')},
                    beta_med={k: med(get(k, 'beta')) for k in ('rom', 'proj', 'ref')},
                    tau4096_med={k: med(lambda r, k=k: r[k]['tau']['4096']) for k in ('rom', 'proj', 'ref')},
                    omega_med={k: med(get(k, 'omega')) for k in ('rom', 'proj', 'ref')},
                    tau4096_ratio_rom_ref_med=med(lambda r: r['rom']['tau']['4096'] / r['ref']['tau']['4096']),
                    tau4096_ratio_proj_ref_med=med(lambda r: r['proj']['tau']['4096'] / r['ref']['tau']['4096']),
                    ctrl_omega_plus_med=med(lambda r: r['ctrl_omega_plus']),
                    ctrl_omega_minus_med=med(lambda r: r['ctrl_omega_minus']))
    groups = {'in_range': grp(F.IN_RANGE), **{c: grp((c,)) for c in ('N0s', 'C1', 'C2')}}
    g = groups['in_range']
    ident = identity_checks()
    fx = rows.get('dev6|0')
    if fx is not None and fx['rom'] is not None and fx['laplacian_scale_ok']:
        ident.update(bank_path_check(Z['W'][idx['dev6|0']], lap_rms(F.load_ref('dev6', 0)[0])))
    else:
        ident.update(bank_path_fixture='dev6|0', bank_path_ok=False, note='fixture unavailable')
    ctrl = dict(identity=ident,
                omega_plus_ok=g['ctrl_omega_plus_med'] is not None and g['ctrl_omega_plus_med'] >= 0.5,
                omega_minus_ok=g['ctrl_omega_minus_med'] is not None and g['ctrl_omega_minus_med'] <= 0.25,
                omega_rom_available_half=bool(g['n_eligible'] > 0 and g['omega_available']['rom'] >= g['n_eligible'] / 2))
    ctrl['kappa_reportable'] = bool(ident['envelope_ok'] and ident['sine_ok'] and ident['bank_path_ok'])   # Amendment 7
    ctrl['controls_pass'] = bool(ctrl['omega_plus_ok'] and ctrl['omega_minus_ok'] and ctrl['omega_rom_available_half'])
    if not ctrl['kappa_reportable']:
        for gg in groups.values():
            gg['kappa_a_med'] = 'withheld (curvature identity/path check failed)'
        for r in rows.values():
            for k in ('rom', 'proj'):
                if r[k] is not None:
                    r[k]['kappa_a_diagnostic_unreportable'] = r[k].pop('kappa_a')
    tr, om = g['tau4096_ratio_rom_ref_med'], g['omega_med']['rom']
    crit = dict(tail_3x=bool(tr is not None and tr >= 3), wall_share_half=bool(om is not None and om >= 0.5))
    decision = ('INCONCLUSIVE' if not ctrl['controls_pass'] else ('RUN' if crit['tail_3x'] and crit['wall_share_half'] else 'DEFER'))
    rr = [r['retracted_fd_control_fires'] for r in rows.values() if 'retracted_fd_control_fires' in r]
    retracted = dict(n_in_range=len(rr), n_fired=int(sum(rr)),
                     note='curvature condition of the retracted original control C-A2+ (FD stencil, >= 10x on every in-range case) replayed; its beta condition is not replayed; dropped by Amendment 1')
    out = dict(thresholds=dict(tail_ratio=3, omega=0.5, omega_plus=0.5, omega_minus=0.25),
               design='DESIGN.md A2 as replaced by Amendments 1-7', provenance=F.provenance(), retracted_control_replay=retracted,
               source=str(F.ROMDIR / 'R_R384_LSPG_CNR_1.npz'), excluded_ineligible=excluded, invalid_rom_runs=invalid,
               controls=ctrl, criteria=crit, decision_arm_c=decision, groups=groups, rows=rows)
    F.write_json('a2_wall', out)
    print(json.dumps(dict(controls=ctrl, criteria=crit, decision=decision, groups=groups), indent=1))


if __name__ == '__main__':
    main()
