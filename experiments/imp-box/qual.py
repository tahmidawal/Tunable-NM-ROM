"""imp-box qualification: Q1 for the FOMs, Q2 for every reduced arm (DESIGN A1.5, A2.6, A3.4, A4).

Q1 (reduced arms) is imp-geometry `finalize.q1` (LSPG anchors certified, A2.6); Q1 (FOMs) is here: every stage finite,
contour vs closed-form coefficients <= 1e-10, independent-transcription defect <= 1e-9 at every step (AD: every trial),
certificate replay identical <= 1e-12, AD: no cap-hit / exhaustion.

Q2: rule q vs its successor (q + 2) on the configuration's own reached states (val, 6 outputs): rho = ||N_q - N_succ|| /
||N_succ|| <= 0.116 (max; states with ||N_succ|| <= 1e-12 max skipped, <= 5 %), rollout distance <= 2.5e-4 on every case,
every compared rollout valid. Tensor form 'T12': rho of the rule-12 vector (= tensor contraction, G4) vs rule 14; rollout
tensor(12) vs tensor(14). ECSW: rho of the ECSW vector vs the parent rule (12) vector, rollout ECSW vs PODG on the parent
rule (same r, n), and the parent must pass PODG Q2.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import finalize as FZ
import ladder as LD
import podsine as PS
import rom as RM

Q1_TIGHT = FZ.Q1_TIGHT
RHO_BAR = FZ.RHO_BAR
SUCC = {4: 6, 6: 8, 8: 10, 10: 12, 12: 14, 'T12': 'T14'}


def q1_fom(cfg, build, D, O_prod):
    qc = build(dict(cfg, cert=True))
    Oc, st = FZ.run_unbatched(qc, D)
    st = np.asarray(st)
    it = dict(valid=bool(np.all(np.isfinite(O_prod))),
              replay=float(np.max(np.abs(Oc - O_prod)) / (np.max(np.abs(O_prod)) + 1e-300)),
              nonfinite=float(st[:, 0].sum()), coef_cert_max=float(st[:, 1].max()), defect_max=float(st[:, 4].max()),
              cap_hits=float(st[:, 3].sum()), exhausted=float(st[:, 6].sum()))
    it['passed'] = bool(it['valid'] and it['nonfinite'] == 0 and it['coef_cert_max'] <= 1e-10
                        and it['defect_max'] <= 1e-9 and it['replay'] <= 1e-12 and it['cap_hits'] == 0
                        and it['exhausted'] == 0)
    return it


def nvec(sp, z, r, Mt, w_scale=1.0):
    P, S = sp.P[:, :r], sp.S[:, :r]
    return sp.P[:, :Mt].T @ (w_scale * sp.w * (P @ z) * (S @ z))


def _rho(Z, f1, f2):
    allN = [np.linalg.norm(f2(z)) for z in Z]
    mx = max(allN) if allN else 0.0
    rho, skipped = [], 0
    for z, n2 in zip(Z, allN):
        if n2 == 0.0 or n2 <= 1e-12 * mx:
            skipped += 1
            continue
        rho.append(float(np.linalg.norm(f1(z) - f2(z)) / n2))
    return (max(rho) if rho else float('nan')), skipped / max(len(allN), 1)


def _scaled(sp, w_scale):
    """MF-Q2 / MF-ECSW fault: the configuration's own production rule weights scaled (rollout AND vectors)."""
    if w_scale == 1.0:
        return sp
    return RM.Space(sp.P, sp.S, sp.w * w_scale, sp.M, sp.K, sp.Pi, sp.wi, sp.Pl, name=sp.name + f'-w{w_scale}',
                    meta=sp.meta)


def q2(cfg, spaces, build_on, D, Uref, w_scale=1.0):
    """Generic rule-successor Q2. spaces: {q: Space} incl. the successor; build_on(cfg, space, z_out) -> query.
    Tensor form ('T12'): the nonlinear vectors are the production tensor contractions T12 z z vs T14 z z (code audit 1)."""
    q, r = cfg['q'], cfg['r']
    tensor = str(q).startswith('T')
    qq = 12 if tensor else q
    sq = 14 if tensor else SUCC[q]
    s1, s2 = _scaled(spaces[qq], w_scale), spaces[sq]
    Mt = min(4 * r, s1.R) if cfg.get('st') == 'lspg_cnr' else r
    Z1, st_z = FZ.run_unbatched(build_on(cfg, s1, True), D)
    O1, st1 = FZ.run_unbatched(build_on(cfg, s1, False), D)
    c2 = dict(cfg, q=('T14' if tensor else sq))
    O2, st2 = FZ.run_unbatched(build_on(c2, s2, False), D)
    valid = bool(LD.stats_valid(cfg, st1) and LD.stats_valid(cfg, st2) and np.all(np.isfinite(O1))
                 and np.all(np.isfinite(O2)))
    Zf = np.asarray(Z1).reshape(-1, r)
    if tensor:
        T1 = np.asarray(RM.tensor(jnp.asarray(s1.P[:, :Mt]), jnp.asarray(s1.P[:, :r]), jnp.asarray(s1.S[:, :r]),
                                  jnp.asarray(s1.w)))
        T2 = np.asarray(RM.tensor(jnp.asarray(s2.P[:, :Mt]), jnp.asarray(s2.P[:, :r]), jnp.asarray(s2.S[:, :r]),
                                  jnp.asarray(s2.w)))
        f1 = lambda z: np.einsum('kij,i,j->k', T1, z, z)
        f2 = lambda z: np.einsum('kij,i,j->k', T2, z, z)
    else:
        f1 = lambda z: nvec(s1, z, r, Mt)
        f2 = lambda z: nvec(s2, z, r, Mt)
    rho, skf = _rho(Zf, f1, f2)
    roll = FZ.dist(O1, O2, Uref)
    it = dict(rho_max=rho, skipped_frac=skf, rollout_max=float(np.max(roll)), rollouts_valid=valid, successor=str(sq),
              form='tensor' if tensor else 'point', w_scale=w_scale)
    it['passed'] = bool(valid and np.isfinite(rho) and rho <= RHO_BAR and skf <= 0.05 and it['rollout_max'] <= Q1_TIGHT)
    return it


def q2_ecsw(cfg, sub, parent, build_sub, build_parent, D, Uref, parent_q2, w_scale=1.0):
    r = cfg['r']
    sub = _scaled(sub, w_scale)
    Z1, _ = FZ.run_unbatched(build_sub(cfg, sub, True), D)
    O1, st1 = FZ.run_unbatched(build_sub(cfg, sub, False), D)
    O2, st2 = FZ.run_unbatched(build_parent(cfg, parent, False), D)
    valid = bool(LD.stats_valid(cfg, st1) and LD.stats_valid(cfg, st2) and np.all(np.isfinite(O1))
                 and np.all(np.isfinite(O2)))
    Zf = np.asarray(Z1).reshape(-1, r)
    rho, skf = _rho(Zf, lambda z: nvec(sub, z, r, r), lambda z: nvec(parent, z, r, r))
    roll = FZ.dist(O1, O2, Uref)
    it = dict(rho_max=rho, skipped_frac=skf, rollout_max=float(np.max(roll)), rollouts_valid=valid,
              parent_q2=parent_q2)
    it['passed'] = bool(valid and np.isfinite(rho) and rho <= RHO_BAR and skf <= 0.05 and it['rollout_max'] <= Q1_TIGHT
                        and parent_q2.get('passed', False))
    return it
