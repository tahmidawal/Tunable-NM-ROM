"""Local smoke and lambda-grid probe for the prior dial. One process, local GPU.

Gate (i): lambda = infinity, THROUGH THE NEW CODE PATH, must reproduce the
consolidated saved Burgers case and the incumbent `accuracy_paths.make_rom` to
1e-12 relative, using the archived operators, exactly as `smoke_arms.py` part 1
does for arm (a).

Then the finite path is checked against that limit at a very large lambda, and a
small coarse-mesh probe measures where the transition in lambda actually lies, so
the sweep grid declared in DESIGN.md can be confirmed or amended once, before any
cluster submission.
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
sys.path.insert(0, str(ROOT / 'experiments/prior-dial'))
sys.path.insert(0, str(ROOT / 'experiments/head-ablation'))
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))

import engines as e            # noqa: E402
import iterative_paths as ip   # noqa: E402
import accuracy_paths as ap    # noqa: E402
import arms as A               # noqa: E402
import prior as PR             # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FIX = ROOT / 'consolidated/fixtures/burgers'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
              'mr-burgers2d/runs/accuracy09/archive/out/result.json')
PROBE_LAMBDAS = [None, 1e0, 1e-2]


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
    trust = setup['trust_radius']
    arch = {k: jnp.asarray(v) for k, v in np.load(FIX / 'operators.npz').items()}
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qg, Rg = A.whiten(G)
    phys = np.asarray(raw['physical_cases'][0])
    u0 = jnp.asarray(e.initial(L, phys))
    nu = float(phys[4])

    # ---- gate (i): lambda = infinity through the new path -----------------
    data0 = dict(A=arch['A'], lam=arch['lam'], G5=arch['G5'], Pq=arch['Pq'], G=G)
    Zaug = np.asarray(arch['candidate_Z'])
    Hrot = jax.jit(jax.vmap(lambda z: A.sc.head(params, z)))(jnp.asarray(Zaug)) @ arch['cold_R'].T
    cold = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
            Hrot, jnp.sum(Hrot * Hrot, 1), jnp.asarray(Zaug))
    qi = PR.make_prior_query(params, K, 0, L, dt, 'eq', linear='gj', **cfg['strict'])
    v = jax.device_get(qi(u0, jnp.asarray(nu), jnp.asarray(0.), jnp.asarray(trust),
                          jnp.asarray(np.inf), data0, cold))
    saved = np.load(FIX / 'expected.npz')
    rel = float(np.linalg.norm(v[0] - saved['fields']) / np.linalg.norm(saved['fields']))
    relz = float(np.linalg.norm(v[7] - saved['internal_latents'])
                 / np.linalg.norm(saved['internal_latents']))
    out['laminf_vs_saved_case'] = dict(relative_l2=rel, latent_relative_l2=relz, tolerance=1e-12,
                                       max_step_stationarity=float(np.max(v[8])),
                                       initial_stationarity=float(v[9]),
                                       max_correction_norm=float(np.max(v[12])))
    print('GATE laminf vs saved', rel, relz, flush=True)
    assert rel <= 1e-12 and relz <= 1e-12, (rel, relz)

    ref = jax.device_get(ap.make_rom(params, L, dt, trust, **cfg['strict'])(
        u0, jnp.asarray(nu),
        (G, arch['A'], arch['lam'], arch['G5'], arch['Pq'], None, None, arch['candidate_Z']),
        (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'], Hrot,
         jnp.sum(Hrot * Hrot, 1))))
    out['laminf_vs_incumbent'] = float(np.linalg.norm(v[0] - ref[0]) / np.linalg.norm(ref[0]))
    out['laminf_bitwise_identical_to_incumbent'] = bool(np.array_equal(v[0], ref[0]))
    print('GATE laminf vs incumbent', out['laminf_vs_incumbent'], flush=True)
    assert out['laminf_vs_incumbent'] <= 1e-12

    # ---- the y-space companions, on the archived operators ----------------
    Phi, lam, _ = e.modes(L, int(arch['A'].shape[0]))
    P = jnp.asarray(Phi)
    out['archived_A_matches_recomputed_modes'] = float(
        jnp.linalg.norm(P.T @ G - arch['A']) / jnp.linalg.norm(arch['A']))
    assert out['archived_A_matches_recomputed_modes'] < 1e-12
    Ay = P.T @ Qg
    sigma = float(jnp.linalg.norm(Ay, 2))
    G5y = jax.scipy.linalg.solve_triangular(
        Rg.T, arch['G5'].reshape(-1, R).T, lower=True).T.reshape(arch['G5'].shape)
    svR = np.asarray(jnp.linalg.svd(Rg, compute_uv=False))
    data = dict(data0, Qg=Qg, Ay=Ay, G5y=G5y)
    out['bank'] = dict(sigma=sigma, R_condition=float(svR[0] / svR[-1]),
                       orthonormality_deviation=float(
                           jnp.linalg.norm(Qg.T @ Qg - jnp.eye(R)) / np.sqrt(R)),
                       factorization_relative=float(jnp.linalg.norm(Qg @ Rg - G) / jnp.linalg.norm(G)),
                       reference_state_norm=float(np.linalg.norm(saved['fields'][0])),
                       intervals=L, M=int(arch['A'].shape[0]))
    print('sigma', sigma, 'cond(R)', out['bank']['R_condition'], flush=True)

    # ---- the finite path at a very large lambda must return to arm (a) ----
    qf = PR.make_prior_query(params, K, R, L, dt, 'eq', linear='lu', **cfg['strict'])
    big = 1e6
    w = jax.device_get(qf(u0, jnp.asarray(nu), jnp.asarray(float(np.sqrt(big)) * sigma),
                          jnp.asarray(trust), jnp.asarray(np.inf), data, cold))
    out['finite_path_large_lambda'] = dict(
        lambda_rel=big, relative_to_laminf=float(np.linalg.norm(w[0] - v[0]) / np.linalg.norm(v[0])),
        max_correction_norm=float(np.max(w[12])),
        max_relative_correction=float(np.max(w[12]) / np.max(w[13])),
        stop_reasons=sorted(set(w[3].tolist())), ic_reason=int(w[6]))
    print('finite@1e6 vs laminf', out['finite_path_large_lambda']['relative_to_laminf'], flush=True)

    # ---- probe: where is the transition in lambda, and in which time? -------
    # Two things have to be measured before spending an A100. (1) The worst error
    # over ALL output times can be pinned at t = 0, because the initializer is arm
    # (a)'s and starts at y = 0, so the t = 0 output is the head's compression of
    # the supplied field and is identical at every lambda. (2) The primary block
    # has M = 4K = 64 < R = 512, so the weak residual is easy to satisfy with a
    # tiny correction; only M > R makes it an overdetermined fit. Both are probed.
    fom, pre = ip.make_fom(L, dt, 'fft')
    probe_cases = [0, 2]
    tight = {c: np.asarray(jax.device_get(fom(jnp.asarray(e.initial(L, np.asarray(raw['physical_cases'][c]))),
                                              float(raw['physical_cases'][c][4]), 1e-6, 1e-8, *pre))[0])
             for c in probe_cases}

    def per_time(f, c):
        g = tight[c]
        n0 = float(np.linalg.norm(g[0]))
        return (np.linalg.norm((np.asarray(f) - g).reshape(len(g), -1), axis=1) / n0).tolist()

    groups = {}
    for M, quadrature in [(64, 'eq'), (1024, 'dense')]:
        if quadrature == 'eq':
            d, i0 = dict(data), dict(sigma=sigma, M=M, m=int(arch['Pq'].shape[0]))
        else:
            d, i0 = PR.build_prior_operators(bank, L, M, quadrature, R, Qg, Rg, params=params)
        groups[(M, quadrature)] = dict(
            data=d, sigma=i0['sigma'], M=M, m=i0.get('m'),
            data0={k: v for k, v in d.items() if k not in ('Qg', 'Ay', 'G5y')},
            qi=PR.make_prior_query(params, K, 0, L, dt, quadrature, linear='gj', **cfg['strict']),
            qf=PR.make_prior_query(params, K, R, L, dt, quadrature, linear='lu', **cfg['strict']))
    rows = []
    for (M, quadrature), g in groups.items():
        for lr in PROBE_LAMBDAS:
            for c in probe_cases:
                ph = np.asarray(raw['physical_cases'][c])
                ui = jnp.asarray(e.initial(L, ph))
                t0 = time.perf_counter()
                if lr is None:
                    z = jax.device_get(g['qi'](ui, jnp.asarray(float(ph[4])), jnp.asarray(0.),
                                               jnp.asarray(trust), jnp.asarray(np.inf), g['data0'], cold))
                else:
                    z = jax.device_get(g['qf'](ui, jnp.asarray(float(ph[4])),
                                               jnp.asarray(float(np.sqrt(lr)) * g['sigma']),
                                               jnp.asarray(trust), jnp.asarray(np.inf), g['data'], cold))
                reasons = z[3].tolist()
                pt = per_time(z[0], c)
                rows.append(dict(M=M, quadrature=quadrature, lambda_rel=lr, case=c,
                                 same_grid_per_time=pt, same_grid_all_times=float(np.max(pt)),
                                 same_grid_evolved_times=float(np.max(pt[1:])),
                                 argmax_time_index=int(np.argmax(pt)),
                                 max_relative_correction=float(np.max(z[12]) / np.max(z[13])),
                                 completed=bool(all(r in (1, 2, 4) for r in reasons)
                                                and int(z[6]) in (1, 2, 4)),
                                 budget_exits=int(sum(1 for r in reasons if r == 0)),
                                 median_iterations=float(np.median(z[1])),
                                 seconds=time.perf_counter() - t0))
                r = rows[-1]
                print('PROBE M', M, quadrature, 'lam', lr, 'case', c,
                      'all', round(r['same_grid_all_times'] * 100, 4),
                      'evolved', round(r['same_grid_evolved_times'] * 100, 4),
                      'argmax', r['argmax_time_index'],
                      'corr', round(r['max_relative_correction'], 5),
                      'done', r['completed'], 'it', r['median_iterations'],
                      's', round(r['seconds'], 2), flush=True)
    out['probe'] = dict(intervals=L, cases=probe_cases,
                        metric='same-grid vs the tight FOM on this mesh, per output time',
                        note=('coarse-mesh development probe for grid and metric calibration only; '
                              'not a result. R = %d, so M = 64 is underdetermined and M = 1024 '
                              'is the overdetermined free-bank limit.' % R),
                        rows=rows)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)
    print('PRIOR SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
