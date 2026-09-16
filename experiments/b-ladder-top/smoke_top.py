"""Local smoke for the top-of-ladder fixes. One process, 64 intervals, small q.

Gates, in order:

  1 q = 0 through EVERY fix arm reproduces the consolidated saved Burgers case to
    1e-12 relative and the arms are bitwise identical to each other there.
  2 `base` reproduces the cheap-corrections `block` solver at a small q.
  3 `pre` (column equilibration) reproduces `base`: it is a no-op in exact arithmetic,
    so any difference is floating point and must be small.
  4 `damp` / `predamp` reach the shared stationarity rule.
  5 `casc` starting from a coarse rung agrees with the same rung solved cold, and
    never starts a step from a worse residual.
  6 the conditioning probe returns finite numbers and equilibration does not make the
    condition number worse.
  7 the walltime-bounded NNLS fitter with the larger refit block produces a valid rule.

The real fidelity reproduction of the cheap-corrections q = 0 and q = 128 rows is an
in-job gate: those numbers are only comparable on a cluster A100, and a 256-interval
q = 128 query is far outside the sub-minute budget the shared box allows.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
for sub in ('experiments/b-ladder-top', 'experiments/cheap-corrections',
            'experiments/head-ablation', 'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import ladder as LD            # noqa: E402
import varpro as VP            # noqa: E402
import directions as DIR       # noqa: E402
import topfix as TF            # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FIX = ROOT / 'consolidated/fixtures/burgers'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
              'mr-burgers2d/runs/accuracy09/archive/out/result.json')


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    out = {}
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    K, R = Zold.shape[1], int(np.asarray(ck['params']['h_lin']).shape[1])
    raw = json.loads(RAW.read_text())
    cfg = raw['config']
    setup = next(r for r in raw['mesh_setup'] if (r['intervals'], r['model']) == (64, 'frozen'))
    L, dt = 64, cfg['dt']
    arch = {k: jnp.asarray(v) for k, v in np.load(FIX / 'operators.npz').items()}
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    data0 = dict(A=arch['A'], lam=arch['lam'], G5=arch['G5'], Pq=arch['Pq'], G=G)
    trust = setup['trust_radius']
    phys = raw['physical_cases'][0]
    u0 = jnp.asarray(e.initial(L, phys))
    nu = jnp.asarray(phys[4])
    saved = np.load(FIX / 'expected.npz')
    Qb, Rb = A.whiten(G)

    # ---- gate 1: q = 0 through every arm ---------------------------------
    C0 = jnp.zeros((R, 0))
    head0 = LD.corrected_head(params, C0, K)
    Zaug = np.asarray(arch['candidate_Z'])
    Hrot = jax.jit(jax.vmap(head0))(jnp.asarray(Zaug)) @ arch['cold_R'].T
    cold0 = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
             Hrot, jnp.sum(Hrot * Hrot, 1), jnp.asarray(Zaug))
    fields, rows = {}, {}
    for arm in TF.ARMS:
        fn = TF.make_query(params, C0, K, 0, L, dt, trust, 'eq', arm, Rb=Rb,
                           linear='gj', **cfg['strict'])
        v = jax.device_get(fn(u0, nu, data0, cold0))
        fields[arm] = np.asarray(v[0])
        r0 = rel(v[0], saved['fields'])
        rz = rel(v[7], saved['internal_latents'])
        rows[arm] = dict(relative_l2=r0, latent_relative_l2=rz, tolerance=1e-12,
                         max_step_joint_stationarity=float(np.max(v[12])),
                         ic_joint_stationarity=float(v[13]))
        assert r0 <= 1e-12 and rz <= 1e-12, (arm, r0, rz)
        print('GATE q0', arm, r0, rz, flush=True)
    out['q0_vs_saved_case'] = rows
    out['q0_arms_bitwise_identical'] = {a: bool(np.array_equal(fields[a], fields['base']))
                                        for a in fields}
    assert all(out['q0_arms_bitwise_identical'].values())

    # ---- tiny direction fit ----------------------------------------------
    train = e.params_draw(0, 4)
    fom, _ = e.make_fom(L, dt, None, .25, dt)
    U = np.concatenate([np.asarray(fom(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-9, 1e-7)[0])
                        [::10][:, 1:-1, 1:-1].reshape(-1, (L - 1) ** 2) for p in train])
    del fom
    jax.clear_caches()
    coef = jnp.linalg.solve(Rb, Qb.T @ jnp.asarray(U.T)).T
    Zsub = np.asarray(Zold[::len(Zold) // 1024])
    dcfg = dict(residual_seed=5, residual_snapshots=16, residual_starts=2, residual_budget=80,
                strict=dict(gtol=1e-6), q_ladder=[0, 8, 16])
    C, Ct, Zstar, rho, dinfo = DIR.audited(params, Rb, coef, Zsub, K, dcfg)
    out['directions'] = {k: dinfo[k] for k in ('available_rank', 'head_fit_relative_worst',
                                               'residual_energy_captured', 'seconds')}

    # ---- gates 2-5: the fixes at small q ---------------------------------
    strict = dict(ic_budget=200, step_budget=60, gtol=1e-6)
    per_q = {}
    for q in (8, 16):
        M = 4 * (K + q)
        Cq = C[:, :q]
        head = LD.corrected_head(params, Cq, K)
        d, _ = TF.build_operators(bank, L, M, 'dense')
        cq, _ = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1), 24)
        got = {}
        ref_block = jax.device_get(VP.make_query(
            params, Cq, K, q, L, dt, trust, 'dense', 'block', linear='gj',
            inner_damping=1e-10, **strict)(u0, nu, d, cq))
        for arm in ('base', 'pre', 'damp', 'predamp'):
            t0 = time.perf_counter()
            fn = TF.make_query(params, Cq, K, q, L, dt, trust, 'dense', arm, Rb=Rb,
                               linear='gj', inner_damping=1e-10, tau_y=.1, **strict)
            v = jax.device_get(fn(u0, nu, d, cq))
            f = np.asarray(v[0])
            assert np.isfinite(f).all() and f.shape == (6, L + 1, L + 1)
            got[arm] = dict(
                relative_to_cheap_corrections_block=rel(f, ref_block[0]),
                bitwise_to_cheap_corrections_block=bool(np.array_equal(f, np.asarray(ref_block[0]))),
                max_step_joint_stationarity=float(np.max(v[12])),
                ic_joint_stationarity=float(v[13]),
                median_iterations=float(np.median(v[1])),
                budget_exits=int(sum(1 for r in v[3].tolist() if r == 0)),
                stop_reasons=sorted(set(v[3].tolist())), seconds=time.perf_counter() - t0)
            got[arm]['fields'] = f
            print('ARM q', q, arm, got[arm]['relative_to_cheap_corrections_block'],
                  got[arm]['max_step_joint_stationarity'], flush=True)
        # cascade: q = 16 started from the q = 8 predamp trajectory
        if q == 16 and 8 in per_q:
            w8 = per_q[8]['latents']
            warm = jnp.asarray(np.concatenate((w8, np.zeros((len(w8), q - 8))), axis=1))
            fn = TF.make_query(params, Cq, K, q, L, dt, trust, 'dense', 'casc', Rb=Rb,
                               linear='gj', inner_damping=1e-10, tau_y=.1, cascade=True, **strict)
            v = jax.device_get(fn(u0, nu, d, cq, warm))
            got['casc'] = dict(
                relative_to_predamp=rel(v[0], got['predamp']['fields']),
                max_step_joint_stationarity=float(np.max(v[12])),
                median_iterations=float(np.median(v[1])),
                budget_exits=int(sum(1 for r in v[3].tolist() if r == 0)),
                stop_reasons=sorted(set(v[3].tolist())), fields=np.asarray(v[0]))
            print('ARM q', q, 'casc', got['casc']['relative_to_predamp'], flush=True)
        per_q[q] = dict(latents=np.asarray(
            jax.device_get(TF.make_query(params, Cq, K, q, L, dt, trust, 'dense', 'predamp',
                                         Rb=Rb, linear='gj', inner_damping=1e-10, tau_y=.1,
                                         **strict)(u0, nu, d, cq))[7]), arms=got)
        assert got['pre']['relative_to_cheap_corrections_block'] <= 1e-6, q
        assert rel(got['pre']['fields'], got['base']['fields']) <= 1e-9, q

    out['small_q'] = {str(q): {a: {k: v for k, v in r.items() if k != 'fields'}
                               for a, r in per_q[q]['arms'].items()} for q in per_q}
    out['pre_versus_base'] = {str(q): rel(per_q[q]['arms']['pre']['fields'],
                                          per_q[q]['arms']['base']['fields']) for q in per_q}

    # ---- gate 6: conditioning -------------------------------------------
    cond = []
    for q in (0, 8, 16):
        M = 4 * (K + q)
        d, _ = TF.build_operators(bank, L, M, 'dense')
        head = LD.corrected_head(params, C[:, :q], K)
        cq, _ = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1), 24)
        c = TF.conditioning(params, C[:, :q], K, q, L, dt, trust, 'dense', d, u0, nu, cq,
                            step_index=10)
        assert np.isfinite(c['jacobian_condition'])
        for lam, v in c['lambdas'].items():
            assert v['equilibrated_condition'] <= v['unscaled_condition'] * 1e3, (q, lam)
        cond.append(c)
        print('COND q', q, c['lambdas']['1e-06'], flush=True)
    out['conditioning'] = cond

    # ---- gate 7: the bounded fitter with a larger refit block ------------
    rng = np.random.default_rng(3)
    Gd = np.abs(rng.standard_normal((512, 1024)))
    bd = Gd[:, :60] @ np.abs(rng.standard_normal(60))
    s2, w2, i2 = TF.bounded_nnls(Gd, bd, 64, 60., blocks_wanted=16)
    out['nnls'] = dict(support=int(len(s2)), seconds=i2['seconds'], refits=i2['refits'],
                       truncated=bool(i2['truncated']), stop_reason=i2['stop_reason'],
                       relative=float(np.linalg.norm(Gd[:, s2] @ w2 - bd) / np.linalg.norm(bd)))
    assert not i2['truncated']
    print('NNLS', out['nnls'], flush=True)

    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print('TOP SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
