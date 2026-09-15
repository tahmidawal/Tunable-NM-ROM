"""Local smoke: the generic arm machinery must reproduce the retained Burgers case.

Part 1 replays the consolidated saved case through the generic `arms.make_query`
with the neural head and the archived operators: arm (a) is then provably the
retained solver, not a re-implementation of it.
Part 2 exercises every other arm family on a tiny mesh so shapes, quadratures and
stopping paths are covered before any cluster submission.
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
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))
sys.path.insert(0, str(ROOT / 'experiments/head-ablation'))

import engines as e            # noqa: E402
import accuracy_paths as ap    # noqa: E402
import arms as A               # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FIX = ROOT / 'consolidated/fixtures/burgers'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
              'mr-burgers2d/runs/accuracy09/archive/out/result.json')


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

    # -------- part 1: generic machinery on the archived operators, L = 64 -----
    L = 64
    arch = {k: jnp.asarray(v) for k, v in np.load(FIX / 'operators.npz').items()}
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    data = dict(A=arch['A'], lam=arch['lam'], G5=arch['G5'], Pq=arch['Pq'], G=G)
    head = A.neural_head(params)
    Hrot = A.sc.head(params, arch['candidate_Z']) @ arch['cold_R'].T
    cold = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
            Hrot, jnp.sum(Hrot * Hrot, 1), arch['candidate_Z'])
    trust = setup['trust_radius']
    assert abs(trust - .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))) < 1e-12
    query = A.make_query(head, K, L, cfg['dt'], trust, 'eq', linear='gj', **cfg['strict'])
    phys = raw['physical_cases'][0]
    u0 = jnp.asarray(e.initial(L, phys))
    value = jax.device_get(query(u0, jnp.asarray(phys[4]), data, cold))
    fields, internal, gn, icgn = value[0], value[7], value[8], value[9]
    saved = np.load(FIX / 'expected.npz')
    rel = float(np.linalg.norm(fields - saved['fields']) / np.linalg.norm(saved['fields']))
    rel_z = float(np.linalg.norm(internal - saved['internal_latents'])
                  / np.linalg.norm(saved['internal_latents']))
    out['generic_vs_saved_case'] = dict(relative_l2=rel, latent_relative_l2=rel_z,
                                        max_step_stationarity=float(np.max(gn)),
                                        initial_stationarity=float(icgn), tolerance=1e-8)
    print('PART1 fields_rel', rel, 'latents_rel', rel_z, flush=True)
    assert rel <= 1e-8 and rel_z <= 1e-8, (rel, rel_z)

    # also confirm the incumbent solver returns the same thing on this input
    ref = jax.device_get(ap.make_rom(params, L, cfg['dt'], trust, **cfg['strict'])(
        u0, jnp.asarray(phys[4]),
        (G, arch['A'], arch['lam'], arch['G5'], arch['Pq'], None, None, arch['candidate_Z']),
        (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'], Hrot,
         jnp.sum(Hrot * Hrot, 1))))
    out['generic_vs_incumbent'] = float(np.linalg.norm(fields - ref[0]) / np.linalg.norm(ref[0]))
    print('PART1 generic_vs_incumbent', out['generic_vs_incumbent'], flush=True)
    assert out['generic_vs_incumbent'] <= 1e-12

    # ------------- part 2: every other arm family on a tiny mesh -------------
    Ls = 32
    dt = cfg['dt']
    train = e.params_draw(0, 4)
    q, _ = e.make_fom(Ls, dt, None, .25, dt)
    U = np.concatenate([np.asarray(q(jnp.asarray(e.initial(Ls, p)), float(p[4]), 1e-9, 1e-7)[0])
                        [::5][:, 1:-1, 1:-1].reshape(-1, (Ls - 1) ** 2) for p in train])
    del q
    jax.clear_caches()
    banks = A.CoordBank(params, K, R)
    Gs = banks.on_grid(Ls)
    Qb, Rb = A.whiten(Gs)
    Zsub = jnp.asarray(Zold[::len(Zold) // 2048])
    Hdec = A.sc.head(params, Zsub)
    Ut = jnp.asarray(U.T)
    coef = jnp.linalg.solve(Rb, Qb.T @ Ut).T
    gram = Ut.T @ Ut
    w, W = jnp.linalg.eigh(gram)
    w, W = w[::-1], W[:, ::-1]
    V = Ut @ (W[:, :8] / jnp.sqrt(jnp.clip(w[:8], 1e-300, None))[None, :])
    coords = np.asarray(Ut.T @ V)
    lin, Zl, Wt, ilin = A.fit_linear_map(Hdec, Rb, K, seed=7)
    quad, iquad = A.fit_quadratic_map(Hdec, Rb, K, Zl, Wt, seed=7)
    out['fits'] = dict(linear=ilin['relative_projection_rms'], quadratic=iquad['heldout_relative'],
                       quadratic_ridge=iquad['ridge'])
    cases = []
    trials = [('b_linear', A.linear_head(lin), K, banks, Zl, 'eq', 64, 256, 'gj'),
              ('c_quad', A.quadratic_head(quad), K, banks, Zl, 'eq', 64, 256, 'gj'),
              ('c_quad_dense', A.quadratic_head(quad), K, banks, Zl, 'dense', 64, None, 'gj'),
              ('e_pod8', A.identity_head(), 8, A.GridBank(V, Ls), coords, 'eq', 32, 128, 'gj'),
              ('e_pod8_dense', A.identity_head(), 8, A.GridBank(V, Ls), coords, 'dense', 32, None, 'gj'),
              ('d_freebank', A.identity_head(), R, banks, np.asarray(Hdec), 'dense', 768, None, 'lu')]
    phys0 = np.asarray(raw['physical_cases'][0])
    u0s = jnp.asarray(e.initial(Ls, phys0))
    for name, hd, k, bk, cand, quad_kind, M, m, linear in trials:
        t0 = time.perf_counter()
        d, i = A.build_operators(bk, Ls, M, quad_kind, Zcoef=cand, m=m, eq_seed=20259,
                                 candidate_cap=2048, fit_states=8, head=hd)
        c, ci = A.build_cold(bk, hd, cand, 24)
        tr = .01 * float(np.max(np.linalg.norm(np.asarray(cand) - np.asarray(cand).mean(0), axis=1)))
        fn = A.make_query(hd, k, Ls, dt, tr, quad_kind, linear=linear,
                          ic_budget=200, step_budget=60, gtol=1e-6)
        v = jax.device_get(fn(u0s, jnp.asarray(phys0[4]), d, c))
        f = np.asarray(v[0])
        assert np.isfinite(f).all() and f.shape == (6, Ls + 1, Ls + 1), (name, f.shape)
        cases.append(dict(arm=name, k=k, M=M, m=m, quadrature=quad_kind, linear_solve=linear,
                          eq_relative_fit=i['eq_relative_fit'], seconds=time.perf_counter() - t0,
                          max_step_stationarity=float(np.max(v[8])), initial_stationarity=float(v[9]),
                          ic_relative_residual=float(v[10]) / float(v[11]),
                          stop_reasons=sorted(set(v[3].tolist())), ic_reason=int(v[6]),
                          max_iterations=int(np.max(v[1])), field_norm=float(np.linalg.norm(f))))
        print('PART2', name, 'ok', round(time.perf_counter() - t0, 2), flush=True)
    out['arms'] = cases
    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print('SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
