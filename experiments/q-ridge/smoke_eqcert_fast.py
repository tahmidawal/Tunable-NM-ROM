"""Fast targeted smoke for the parts of the EQ-certification path the full smoke is slowest
to reach: the rho identity, the population ordering and the drop-in into the retained query.

Runs at q = 0 so no direction fit is needed, and at 32 intervals so nothing takes minutes.
`smoke_eqcert.py` covers the fitter and the reachable-state collection at q = 16.

Gates:

  4 rho IS the only thing `arms.weak_eq` approximates: the difference between the empirical
    and the dense weak residual, undone by the row scaling, equals the rho numerator exactly.
  5 rho falls when m grows on the same population, and a rule fitted on REACHABLE states
    beats one fitted on static decoder outputs at the same m, on held-out reachable states.
  6 a certified rule drops into the retained query unchanged and returns a finite six-field
    output of the right shape, with no budget exits.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
for sub in ('experiments/q-ridge', 'experiments/b-ladder-top', 'experiments/cheap-corrections',
            'experiments/head-ablation', 'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import ridge as RG             # noqa: E402
import eqcert as EC            # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    out = {}
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    K, R = Zold.shape[1], int(np.asarray(ck['params']['h_lin']).shape[1])
    L, dt, q = 32, .005, 0
    M = 4 * K
    C = jnp.zeros((R, 0))
    head = RG.corrected_head(params, C, K)
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Phi, lam, _ = e.modes(L, M)
    P = jnp.asarray(Phi)
    dense = dict(A=P.T @ G, lam=jnp.asarray(lam), G=G, Phi=P)
    Zsub = np.asarray(Zold[::max(1, len(Zold) // 512)])
    cold, _ = A.build_cold(bank, head, Zsub, 24)
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    strict = dict(ic_budget=200, step_budget=600, gtol=1e-6)

    collect = EC.make_collect_query(params, C, K, q, L, dt, trust, iters=8,
                                    ic_budget=strict['ic_budget'], gtol=strict['gtol'],
                                    linear='gj', inner_damping=1e-10)
    cf = jax.jit(jax.vmap(head))
    train = e.params_draw(0, 4)
    pops = []
    for p in train:
        seen, _ = collect(jnp.asarray(e.initial(L, p)), float(p[4]), dense, cold)
        pops.append(np.asarray(cf(jnp.asarray(np.asarray(seen).reshape(-1, K)))))
    reach_fit = np.concatenate(pops[:2])
    held = np.concatenate(pops[2:])
    rng = np.random.default_rng(5)
    held = held[np.sort(rng.choice(len(held), 128, replace=False))]
    static = np.asarray(cf(jnp.asarray(Zsub)))
    cand = np.sort(rng.choice((L - 1) ** 2, min(768, (L - 1) ** 2), replace=False))

    rules = {}
    for tag, pop, m in (('reach_m96', reach_fit, 96), ('reach_m192', reach_fit, 192),
                        ('static_m192', static, 192)):
        sel = pop[np.sort(rng.choice(len(pop), min(64, len(pop)), replace=False))]
        rule, info = EC.fit_rule(bank, G, Phi, L, M, m, sel, cand, fitter='gpu', blocks=4)
        cert = EC.certify(G, Phi, L, rule, held, chunk=32)
        rules[tag] = (rule, info, cert)
        print('RULE', tag, 'm', info['m'], 'fit', f"{info['relative_fit']:.3e}",
              'rho_max', f"{cert['rho_max']:.4f}", 'p95', f"{cert['rho_p95']:.4f}", flush=True)
    out['rules'] = {k: dict(m=v[1]['m'], relative_fit=v[1]['relative_fit'], **v[2])
                    for k, v in rules.items()}

    # ---- gate 4 ---------------------------------------------------------
    rule = rules['reach_m192'][0]
    wv = jnp.asarray(reach_fit[0])
    prev = dense['A'] @ wv
    nu = float(train[0][4])
    eqd = dict(A=dense['A'], lam=dense['lam'], G=G, G5=rule['G5'], Pq=rule['Pq'])
    ident = lambda w: w
    row = np.asarray(1. + dt * nu * dense['lam'])
    r_eq = np.asarray(A.weak_eq(wv, prev, nu, eqd, ident, L, dt))
    r_dn = np.asarray(A.weak_dense(wv, prev, nu, dense, ident, L, dt))
    us = np.asarray(jnp.einsum('msr,r->ms', rule['G5'], wv))
    cc, xp, xm, yp, ym = [us[:, i] for i in range(5)]
    aq = cc * L * (np.where(cc > 0, cc - xm, xp - cc) + np.where(cc > 0, cc - ym, yp - cc))
    samp = np.asarray(rule['Pq'].T @ jnp.asarray(aq))
    full = np.asarray(dense['Phi'].T @ e.spatial(G @ wv, L)[0])
    lhs = float(np.linalg.norm((r_eq - r_dn) * row / dt))
    rhs = float(np.linalg.norm(samp - full))
    out['rho_identity'] = dict(weak_difference=lhs, rho_numerator=rhs,
                               relative=abs(lhs - rhs) / max(rhs, 1e-300))
    assert out['rho_identity']['relative'] < 1e-12, out['rho_identity']
    print('GATE 4', out['rho_identity'], flush=True)

    # ---- gate 5 ---------------------------------------------------------
    out['rho_ordering'] = dict(
        rho_falls_with_m=bool(rules['reach_m192'][2]['rho_max']
                              <= rules['reach_m96'][2]['rho_max']),
        reachable_beats_static=bool(rules['reach_m192'][2]['rho_max']
                                    < rules['static_m192'][2]['rho_max']),
        reach_m96=rules['reach_m96'][2]['rho_max'],
        reach_m192=rules['reach_m192'][2]['rho_max'],
        static_m192=rules['static_m192'][2]['rho_max'])
    assert out['rho_ordering']['rho_falls_with_m'], out['rho_ordering']
    print('GATE 5', out['rho_ordering'], flush=True)

    # ---- gate 6 ---------------------------------------------------------
    u0 = jnp.asarray(e.initial(L, train[0]))
    v = jax.device_get(RG.make_query(params, C, K, q, L, dt, trust, 'eq', 0., linear='gj',
                                     inner_damping=1e-10, **strict)(u0, nu, eqd, cold))
    f = np.asarray(v[0])
    d = jax.device_get(RG.make_query(params, C, K, q, L, dt, trust, 'dense', 0., linear='gj',
                                     inner_damping=1e-10, **strict)(u0, nu, dense, cold))
    out['query'] = dict(shape=list(f.shape), finite=bool(np.isfinite(f).all()),
                        max_step_joint_stationarity=float(np.max(v[12])),
                        budget_exits=int(sum(1 for r in v[3].tolist() if r == 0)),
                        relative_to_dense=float(np.linalg.norm(f - np.asarray(d[0]))
                                                / np.linalg.norm(np.asarray(d[0]))))
    assert out['query']['finite'] and f.shape == (6, L + 1, L + 1)
    assert out['query']['budget_exits'] == 0, out['query']
    print('GATE 6', out['query'], flush=True)

    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print('EQCERT FAST SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
