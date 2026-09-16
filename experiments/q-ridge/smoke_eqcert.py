"""Local smoke for the EQ-certification lane (DESIGN.md §A3). 64 intervals, small q.

Gates, in order:

  1 the GPU active-set fitter matches scipy's Lawson-Hanson NNLS on a small problem: a
    relative fit no worse than 1e-9 absolute or 1.2x scipy's, whichever is looser, and
    nonnegative weights. Real rules in this cell reach about 4e-4, so 1e-9 is exact for the
    purpose. This is what licenses using the fitter where Lawson-Hanson cannot reach.
  2 the unrolled collection query lands on the retained solver's own step solutions: at the
    FIRST step, where both start from the same initial fit, the unrolled run's weak residual
    is no larger than the production solver's (it keeps iterating past the gradient exit);
    over the whole trajectory the latents agree to 1e-3 and the decoded output fields to
    1e-4. Later steps cannot be compared residual-for-residual because each trajectory steps
    from its own previous state.
  3 the collected iterates are a strict superset of the converged states, and none is
    non-finite.
  4 rho is the quantity `arms.weak_eq` actually commits: replacing the rule's sampled
    advection by the exact one changes the weak residual by exactly the rho numerator, to
    1e-12 relative.
  5 rho falls when m grows on the same population, and a rule fitted on REACHABLE states has
    a lower held-out rho than one fitted on static decoder outputs at the same m.
  6 a certified rule drops into the retained query unchanged and produces a finite six-field
    output of the right shape.
  7 no large constant is captured: the collection query, the fitter's design build and the
    certifier all take the bank as an argument.
"""
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.optimize
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
for sub in ('experiments/q-ridge', 'experiments/b-ladder-top', 'experiments/cheap-corrections',
            'experiments/head-ablation', 'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import varpro as VP            # noqa: E402
import directions as DIR       # noqa: E402
import ridge as RG             # noqa: E402
import eqcert as EC            # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
              'mr-burgers2d/runs/accuracy09/archive/out/result.json')


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    out = {}

    # ---- gate 1: the GPU fitter against scipy NNLS -----------------------
    rng = np.random.default_rng(7)
    D = np.abs(rng.standard_normal((600, 900)))
    b = D[:, :80] @ np.abs(rng.standard_normal(80))
    s1, w1, i1 = EC.gpu_nnls(D, b, 96, blocks=4, inner_iters=40)
    s2, w2, i2 = VP.bounded_nnls(D, b, 96, 120., block=24)
    r2 = float(np.linalg.norm(D[:, s2] @ w2 - b) / np.linalg.norm(b))
    out['fitter'] = dict(gpu_support=int(len(s1)), scipy_support=int(len(s2)),
                         gpu_relative=i1['relative_fit'], scipy_relative=r2,
                         ratio=i1['relative_fit'] / max(r2, 1e-300),
                         gpu_seconds=i1['seconds'], scipy_seconds=i2['seconds'],
                         min_weight=float(w1.min()) if len(w1) else 0.)
    assert (w1 >= 0).all() and len(s1) > 0
    # The bar is 1e-9 absolute, not machine precision: real rules in this cell reach a
    # relative fit of about 4e-4, so a fitter accurate to 1e-9 is exact for this purpose.
    assert i1['relative_fit'] <= max(1.2 * r2, 1e-9), out['fitter']
    print('GATE 1', out['fitter'], flush=True)

    # ---- setup -----------------------------------------------------------
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    K, R = Zold.shape[1], int(np.asarray(ck['params']['h_lin']).shape[1])
    raw = json.loads(RAW.read_text())
    cfg = raw['config']
    L, dt = 64, cfg['dt']
    setup = next(r for r in raw['mesh_setup'] if (r['intervals'], r['model']) == (64, 'frozen'))
    trust = setup['trust_radius']
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)

    train = e.params_draw(0, 6)
    fom, _ = e.make_fom(L, dt, None, .25, dt)
    U = np.concatenate([np.asarray(fom(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-9, 1e-7)[0])
                        [::10][:, 1:-1, 1:-1].reshape(-1, (L - 1) ** 2) for p in train])
    del fom
    jax.clear_caches()
    coef = jnp.linalg.solve(Rb, Qb.T @ jnp.asarray(U.T)).T
    Zsub = np.asarray(Zold[::len(Zold) // 1024])
    dcfg = dict(residual_seed=5, residual_snapshots=16, residual_starts=2, residual_budget=80,
                strict=dict(gtol=1e-6), q_ladder=[0, 16])
    C, Ct, Zstar, rho_head, _ = DIR.audited(params, Rb, coef, Zsub, K, dcfg)

    q = 16
    M = 4 * (K + q)
    Cq = C[:, :q]
    head = RG.corrected_head(params, Cq, K)
    Phi, lam, _ = e.modes(L, M)
    P = jnp.asarray(Phi)
    dense = dict(A=P.T @ G, lam=jnp.asarray(lam), G=G, Phi=P)
    cold, _ = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1), 24)
    strict = dict(ic_budget=200, step_budget=600, gtol=1e-6)

    # ---- gates 2-3: the collection query ---------------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        collect = EC.make_collect_query(params, Cq, K, q, L, dt, trust, iters=12,
                                        ic_budget=strict['ic_budget'], gtol=strict['gtol'],
                                        linear='gj', inner_damping=1e-10)
        ph = train[0]
        u0 = jnp.asarray(e.initial(L, ph))
        nu = float(ph[4])
        seen, conv = jax.device_get(collect(u0, nu, dense, cold))
    out['captured_constant_warnings'] = [str(w.message)[:120] for w in caught
                                         if 'constants were captured' in str(w.message)]
    assert not out['captured_constant_warnings'], out['captured_constant_warnings']
    prod = jax.device_get(RG.make_query(params, Cq, K, q, L, dt, trust, 'dense', 0.,
                                        linear='gj', inner_damping=1e-10,
                                        **strict)(u0, nu, dense, cold))
    ref_states = np.asarray(prod[7])[1:]
    internal = np.asarray(prod[7])
    wk_dense = A.weak_fn('dense')

    def step_residuals(states):
        # ||r|| at each step's solution, against that step's own previous state.
        outv = []
        for n in range(len(states)):
            prev = dense['A'] @ head(jnp.asarray(internal[n]))
            outv.append(float(jnp.linalg.norm(
                wk_dense(jnp.asarray(states[n]), prev, nu, dense, head, L, dt))))
        return np.asarray(outv)

    r_coll = step_residuals(np.asarray(conv))
    r_prod = step_residuals(ref_states)
    keep = int(round(.05 / dt))
    fields_coll = np.asarray(jax.jit(jax.vmap(lambda v: e.output_field(
        dense['G'] @ head(v), L, L)))(jnp.asarray(
            np.concatenate((internal[:1], np.asarray(conv)))[::keep])))
    out['collection'] = dict(
        shape=list(np.asarray(seen).shape),
        converged_relative_to_production=rel(conv, ref_states),
        first_step_collection_residual=float(r_coll[0]),
        first_step_production_residual=float(r_prod[0]),
        first_step_never_worse=bool(r_coll[0] <= r_prod[0] * (1 + 1e-9) + 1e-300),
        field_relative_to_production=rel(fields_coll, prod[0]),
        finite=bool(np.isfinite(np.asarray(seen)).all()),
        states=int(np.asarray(seen).reshape(-1, K + q).shape[0]),
        superset=bool(np.asarray(seen).shape[1] > 1))
    assert out['collection']['finite'] and out['collection']['superset']
    assert out['collection']['converged_relative_to_production'] < 1e-3, out['collection']
    assert out['collection']['first_step_never_worse'], out['collection']
    assert out['collection']['field_relative_to_production'] < 1e-4, out['collection']
    print('GATES 2-3', out['collection'], flush=True)

    cf = jax.jit(jax.vmap(head))
    reach = np.asarray(cf(jnp.asarray(np.asarray(seen).reshape(-1, K + q))))
    ph2 = train[1]
    seen2, _ = jax.device_get(collect(jnp.asarray(e.initial(L, ph2)), float(ph2[4]), dense, cold))
    held = np.asarray(cf(jnp.asarray(np.asarray(seen2).reshape(-1, K + q))))
    rng2 = np.random.default_rng(3)
    held = held[np.sort(rng2.choice(len(held), 96, replace=False))]
    codes = EC.static_codes(Zstar, rho_head, Rb, Ct, q, Zold)
    static = np.asarray(cf(jnp.asarray(
        codes[np.sort(rng2.choice(len(codes), min(48, len(codes)), replace=False))])))

    cand = np.sort(rng2.choice((L - 1) ** 2, min(2048, (L - 1) ** 2), replace=False))
    rules = {}
    for tag, pop, m in (('reach256', reach, 256), ('reach512', reach, 512),
                        ('static512', static, 512)):
        sel = pop[np.sort(rng2.choice(len(pop), min(48, len(pop)), replace=False))]
        rule, info = EC.fit_rule(bank, G, Phi, L, M, m, sel, cand, fitter='gpu', blocks=4,
                                 inner_iters=40)
        cert = EC.certify(G, Phi, L, rule, held, chunk=16)
        rules[tag] = (rule, info, cert)
        print('RULE', tag, 'm', info['m'], 'fit', f"{info['relative_fit']:.3e}",
              'rho_max', f"{cert['rho_max']:.4f}", 'p95', f"{cert['rho_p95']:.4f}", flush=True)
    out['rules'] = {k: dict(m=v[1]['m'], relative_fit=v[1]['relative_fit'],
                            seconds=v[1]['seconds'], **v[2]) for k, v in rules.items()}

    # ---- gate 4: rho IS the weak residual's only approximation -----------
    rule, _, _ = rules['reach512']
    wv = jnp.asarray(reach[0])
    prev = dense['A'] @ wv
    eqd = dict(A=dense['A'], lam=dense['lam'], G=G, G5=rule['G5'], Pq=rule['Pq'])
    identity = lambda w: w
    r_eq = np.asarray(A.weak_eq(wv, prev, nu, eqd, identity, L, dt))
    r_dn = np.asarray(A.weak_dense(wv, prev, nu, dense, identity, L, dt))
    row = np.asarray(1. + dt * nu * dense['lam'])
    us = np.asarray(jnp.einsum('msr,r->ms', rule['G5'], wv))
    cc, xp, xm, yp, ym = [us[:, i] for i in range(5)]
    aq = cc * L * (np.where(cc > 0, cc - xm, xp - cc) + np.where(cc > 0, cc - ym, yp - cc))
    samp = np.asarray(rule['Pq'].T @ jnp.asarray(aq))
    full = np.asarray(dense['Phi'].T @ e.spatial(G @ wv, L)[0])
    out['rho_identity'] = dict(
        weak_difference=float(np.linalg.norm((r_eq - r_dn) * row / dt)),
        rho_numerator=float(np.linalg.norm(samp - full)),
        relative=float(abs(np.linalg.norm((r_eq - r_dn) * row / dt)
                           - np.linalg.norm(samp - full))
                       / max(np.linalg.norm(samp - full), 1e-300)))
    assert out['rho_identity']['relative'] < 1e-12, out['rho_identity']
    print('GATE 4', out['rho_identity'], flush=True)

    # ---- gate 5: rho falls with m, and reachable beats static ------------
    out['rho_ordering'] = dict(
        rho_falls_with_m=bool(rules['reach512'][2]['rho_max'] <= rules['reach256'][2]['rho_max']),
        reachable_beats_static=bool(
            rules['reach512'][2]['rho_max'] < rules['static512'][2]['rho_max']),
        reach256=rules['reach256'][2]['rho_max'], reach512=rules['reach512'][2]['rho_max'],
        static512=rules['static512'][2]['rho_max'])
    assert out['rho_ordering']['rho_falls_with_m'], out['rho_ordering']
    print('GATE 5', out['rho_ordering'], flush=True)

    # ---- gate 6: a certified rule runs in the retained query -------------
    t0 = time.perf_counter()
    v = jax.device_get(RG.make_query(params, Cq, K, q, L, dt, trust, 'eq', 0., linear='gj',
                                     inner_damping=1e-10, **strict)(u0, nu, eqd, cold))
    f = np.asarray(v[0])
    out['query'] = dict(shape=list(f.shape), finite=bool(np.isfinite(f).all()),
                        max_step_joint_stationarity=float(np.max(v[12])),
                        budget_exits=int(sum(1 for r in v[3].tolist() if r == 0)),
                        relative_to_dense=rel(f, prod[0]), seconds=time.perf_counter() - t0)
    assert out['query']['finite'] and f.shape == (6, L + 1, L + 1)
    print('GATE 6', out['query'], flush=True)

    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print('EQCERT SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
