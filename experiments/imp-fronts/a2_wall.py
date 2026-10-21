"""A2 wall curvature and sine-tail convergence of decoded fields (DESIGN §2 A2). CPU, NumPy only.

Fields per case: decoded ROM G_384 W (saved r = 384 LSPG-CN-R run), bank projection P_384 u_ref, E0 reference u_ref;
controls v_+ = u_ref + 0.01 max|u_ref(t0)| 16 x(1-x) y(1-y) (must fire) and S_4096 u_ref (must not fire).
Writes checks/a2_wall.json.
"""
from __future__ import annotations

import json

import numpy as np

import fcommon as F

KS = (1024, 2304, 4096, 9216, 16384, 36864)
FIT = (2304, 4096, 9216, 16384)
COHORTS = ('dev6', 'val32', 'N0s', 'C1', 'C2')
H = 1 / 256.


def kappa(V):
    """V (6, 257, 257): RMS one-sided normal second derivative on the wall nodes (corners excluded) over t1..t5,
    divided by the RMS 5-point Laplacian over the interior."""
    V = np.asarray(V)[1:]
    w = []
    for A in (V, V[:, ::-1], np.swapaxes(V, 1, 2), np.swapaxes(V, 1, 2)[:, ::-1]):
        w.append((2 * A[:, 0, 1:-1] - 5 * A[:, 1, 1:-1] + 4 * A[:, 2, 1:-1] - A[:, 3, 1:-1]) / H ** 2)
    lap = (V[:, 2:, 1:-1] + V[:, :-2, 1:-1] + V[:, 1:-1, 2:] + V[:, 1:-1, :-2] - 4 * V[:, 1:-1, 1:-1]) / H ** 2
    return float(np.sqrt(np.mean(np.concatenate(w) ** 2)) / np.sqrt(np.mean(lap ** 2)))


def tails(V, n0):
    C = F.dst_coef(np.asarray(V)[1:])                     # (5, 65025) Laplacian order
    cum = np.cumsum(C ** 2, 1)
    tot = cum[:, -1]
    tau = {K: float(np.max(np.sqrt(np.maximum(tot - cum[:, K - 1], 0.))) / n0) for K in KS}
    x = np.log(FIT)
    y = np.log([max(tau[K], 1e-300) for K in FIT])
    beta = float(np.polyfit(x, y, 1)[0])
    return tau, beta


def raw_wall_value(W):
    """Bank output at wall nodes WITHOUT the wall zeroing (envelope check), relative to the field max."""
    x = np.arange(257) / 256.
    edge = np.concatenate([np.stack([np.zeros(257), x], 1), np.stack([np.ones(257), x], 1),
                           np.stack([x, np.zeros(257)], 1), np.stack([x, np.ones(257)], 1)])
    P, T = F.NC.bank()
    V = F.NC.NB.features_grad(P, edge)[0] @ T[:, :384]
    return float(np.max(np.abs(W @ V.T)))


def main():
    Z = np.load(F.ROMDIR / 'R_R384_LSPG_CNR_1.npz', allow_pickle=True)
    G = F.G(384)
    Q = F.orth(G)
    x = np.arange(257) / 256.
    env = 16 * np.outer(x * (1 - x), x * (1 - x))
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
        pu = (Q @ (Q.T @ u.reshape(6, -1).T)).T.reshape(6, 257, 257)
        vp = u + 0.01 * np.abs(u[0]).max() * env[None]
        s4 = F.sine_truncate(u, 4096)
        row = dict(case=cid, cohort=coh, raw_wall_rel=raw_wall_value(Z['W'][i]) / float(np.abs(v).max()))
        for name, V in (('rom', v), ('proj', pu), ('ref', u), ('pos', vp), ('neg', s4)):
            tau, beta = tails(V, n0)
            row[name] = dict(kappa=kappa(V), beta=beta, tau={str(K): t for K, t in tau.items()})
        rows[cid] = row
    def grp(cohs):
        R = [r for r in rows.values() if r['cohort'] in cohs]
        med = lambda f: float(np.median([f(r) for r in R]))
        return dict(n=len(R),
                    kappa_med={k: med(lambda r: r[k]['kappa']) for k in ('rom', 'proj', 'ref', 'pos', 'neg')},
                    beta_med={k: med(lambda r: r[k]['beta']) for k in ('rom', 'proj', 'ref', 'pos', 'neg')},
                    tau4096_med={k: med(lambda r: r[k]['tau']['4096']) for k in ('rom', 'proj', 'ref', 'pos', 'neg')},
                    tau4096_ratio_rom_ref_med=med(lambda r: r['rom']['tau']['4096'] / r['ref']['tau']['4096']),
                    tau4096_ratio_proj_ref_med=med(lambda r: r['proj']['tau']['4096'] / r['ref']['tau']['4096']),
                    kappa_ratio_rom_ref_med_of_ratios=med(lambda r: r['rom']['kappa'] / r['ref']['kappa']))
    groups = {'in_range': grp(F.IN_RANGE), **{c: grp((c,)) for c in ('N0s', 'C1', 'C2')}}
    IR = [r for r in rows.values() if r['cohort'] in F.IN_RANGE]
    ctrl = dict(pos_kappa_every_case=all(r['pos']['kappa'] >= 10 * r['ref']['kappa'] for r in IR),
                pos_beta=groups['in_range']['beta_med']['pos'] >= groups['in_range']['beta_med']['ref'] + 0.5,
                neg_kappa=groups['in_range']['kappa_med']['neg'] <= 2 * groups['in_range']['kappa_med']['ref'])
    ctrl['controls_pass'] = bool(all(ctrl.values()))
    g = groups['in_range']
    crit_kappa = g['kappa_med']['rom'] >= 5 * g['kappa_med']['ref']
    crit_tail = g['tau4096_ratio_rom_ref_med'] >= 3
    if ctrl['controls_pass']:
        decision = 'RUN arm (c)' if (crit_kappa or crit_tail) else 'DO NOT RUN arm (c)'
    else:
        decision = 'INCONCLUSIVE (control failed): arm (c) only if budget allows after (a), (b)'
    out = dict(design='DESIGN.md §2 A2', provenance=F.provenance(), source=str(F.ROMDIR / 'R_R384_LSPG_CNR_1.npz'),
               skipped=skipped, controls=ctrl, criteria=dict(kappa_5x=bool(crit_kappa), tail_3x=bool(crit_tail)),
               decision_arm_c=decision, groups=groups, rows=rows)
    F.write_json('a2_wall', out)
    print(json.dumps(dict(controls=ctrl, criteria=out['criteria'], decision=decision, groups=groups), indent=1))


if __name__ == '__main__':
    main()
