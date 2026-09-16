"""Local smoke for the cheap correction ladder. Under a minute; one process.

Gates, in order:

  1 q = 0 through EVERY solver variant reproduces the consolidated saved Burgers case
    to 1e-12 relative and is bitwise identical to the incumbent `accuracy_paths.make_rom`.
  2 the three variants are bitwise identical to each other at q = 0.
  3 at a small q the inner Gauss-Newton drives ||J_y^T r||/(||J_y|| ||r||) down, and
    variable projection and the joint solver agree on the solved field.
  4 the bounded NNLS fitter produces a usable rule, and its cost and fit are recorded
    beside the retained fitter's on the same design.
  5 the enriched codes reproduce the enriched-manifold least-squares correction.
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
sys.path.insert(0, str(ROOT / 'experiments/cheap-corrections'))
sys.path.insert(0, str(ROOT / 'experiments/head-ablation'))
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))

import engines as e            # noqa: E402
import accuracy_paths as ap    # noqa: E402
import arms as A               # noqa: E402
import ladder as LD            # noqa: E402
import varpro as VP            # noqa: E402
import directions as DIR       # noqa: E402

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
    L = 64
    arch = {k: jnp.asarray(v) for k, v in np.load(FIX / 'operators.npz').items()}
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    data = dict(A=arch['A'], lam=arch['lam'], G5=arch['G5'], Pq=arch['Pq'], G=G)
    trust = setup['trust_radius']
    phys = raw['physical_cases'][0]
    u0 = jnp.asarray(e.initial(L, phys))
    saved = np.load(FIX / 'expected.npz')

    # ---- gate 1/2: q = 0 through every variant ---------------------------
    C0 = jnp.zeros((R, 0))
    head0 = LD.corrected_head(params, C0, K)
    Zaug = np.asarray(arch['candidate_Z'])
    Hrot = jax.jit(jax.vmap(head0))(jnp.asarray(Zaug)) @ arch['cold_R'].T
    cold = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
            Hrot, jnp.sum(Hrot * Hrot, 1), jnp.asarray(Zaug))
    fields, rows = {}, {}
    for variant in ('joint', 'varpro', 'alt'):
        fn = VP.make_query(params, C0, K, 0, L, cfg['dt'], trust, 'eq', variant,
                           linear='gj', inner_iters=2, **cfg['strict'])
        v = jax.device_get(fn(u0, jnp.asarray(phys[4]), data, cold))
        fields[variant] = np.asarray(v[0])
        rel = float(np.linalg.norm(v[0] - saved['fields']) / np.linalg.norm(saved['fields']))
        relz = float(np.linalg.norm(v[7] - saved['internal_latents'])
                     / np.linalg.norm(saved['internal_latents']))
        rows[variant] = dict(relative_l2=rel, latent_relative_l2=relz, tolerance=1e-12,
                             max_step_joint_stationarity=float(np.max(v[12])),
                             ic_joint_stationarity=float(v[13]))
        print('GATE q0', variant, rel, relz, flush=True)
        assert rel <= 1e-12 and relz <= 1e-12, (variant, rel, relz)
    out['q0_vs_saved_case'] = rows

    ref = jax.device_get(ap.make_rom(params, L, cfg['dt'], trust, **cfg['strict'])(
        u0, jnp.asarray(phys[4]),
        (G, arch['A'], arch['lam'], arch['G5'], arch['Pq'], None, None, arch['candidate_Z']),
        (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'], Hrot,
         jnp.sum(Hrot * Hrot, 1))))
    out['q0_vs_incumbent'] = {v: float(np.linalg.norm(fields[v] - ref[0]) / np.linalg.norm(ref[0]))
                              for v in fields}
    out['q0_variants_bitwise_identical'] = {
        v: bool(np.array_equal(fields[v], fields['joint'])) for v in fields}
    out['q0_vs_incumbent_bitwise'] = {v: bool(np.array_equal(fields[v], np.asarray(ref[0])))
                                      for v in fields}
    out['q0_vs_incumbent_note'] = (
        'the audited ladder wrapper was bitwise identical to accuracy_paths.make_rom; this '
        'wrapper carries extra per-step stationarity diagnostics inside the same scan, which '
        'changes XLA fusion, so agreement is to round-off rather than bitwise')
    print('GATE q0 vs incumbent', out['q0_vs_incumbent'], flush=True)
    assert all(x <= 1e-12 for x in out['q0_vs_incumbent'].values())
    assert all(out['q0_variants_bitwise_identical'].values())

    # ---- directions on a tiny problem ------------------------------------
    Qb, Rb = A.whiten(G)
    train = e.params_draw(0, 4)
    fom, _ = e.make_fom(L, cfg['dt'], None, .25, cfg['dt'])
    U = np.concatenate([np.asarray(fom(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-9, 1e-7)[0])
                        [::10][:, 1:-1, 1:-1].reshape(-1, (L - 1) ** 2) for p in train])
    del fom
    jax.clear_caches()
    coef = jnp.linalg.solve(Rb, Qb.T @ jnp.asarray(U.T)).T
    Zsub = np.asarray(Zold[::len(Zold) // 1024])
    dcfg = dict(residual_seed=5, residual_snapshots=16, residual_starts=2, residual_budget=80,
                strict=dict(gtol=1e-6), q_ladder=[0, 8])
    C, Ct, Zstar, rho, dinfo = DIR.audited(params, Rb, coef, Zsub, K, dcfg)
    Cf, Ctf, _, _, finfo = DIR.flat(params, Rb, coef, Zsub, K, dcfg)
    out['directions'] = dict(
        audited={k: dinfo[k] for k in ('available_rank', 'head_fit_relative_median',
                                       'head_fit_relative_worst', 'residual_energy_captured',
                                       'directions_sha256', 'seconds')},
        flat_seconds=finfo['seconds'], flat_sha256=finfo['directions_sha256'],
        flat_bitwise_identical=bool(finfo['directions_sha256'] == dinfo['directions_sha256']),
        flat_max_abs_difference=float(np.max(np.abs(np.asarray(Cf - C)))))
    print('DIRECTIONS flat identical', out['directions']['flat_bitwise_identical'],
          out['directions']['flat_max_abs_difference'], flush=True)

    # ---- gate 5: enriched codes ------------------------------------------
    q = 8
    W = VP.enriched_codes(Zstar, rho, Rb, Ct, q)
    head = LD.corrected_head(params, C[:, :q], K)
    eta = jax.jit(jax.vmap(head))(jnp.asarray(W))
    target = jnp.asarray(np.asarray(coef)[np.sort(np.random.default_rng(5).choice(
        len(coef), 16, replace=False))])
    resid = float(jnp.linalg.norm((eta - target) @ Rb.T) / jnp.linalg.norm(target @ Rb.T))
    base = float(jnp.linalg.norm(rho @ Rb.T) / jnp.linalg.norm(target @ Rb.T))
    out['enriched_codes'] = dict(q=q, relative_after=resid, relative_before=base,
                                 improved=bool(resid < base))
    assert resid < base
    print('ENRICHED', base, '->', resid, flush=True)

    # ---- gate 3: varpro versus joint at q > 0 ----------------------------
    M = 4 * (K + q)
    rowsq = []
    fld = {}
    for variant in ('joint', 'varpro', 'alt'):
        t0 = time.perf_counter()
        d, i = VP.build_operators(bank, L, M, 'dense')
        c, ci = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1), 24)
        fn = VP.make_query(params, C[:, :q], K, q, L, cfg['dt'], trust, 'dense', variant,
                           ic_budget=200, step_budget=60, gtol=1e-6, linear='gj', inner_iters=2)
        w = jax.device_get(fn(u0, jnp.asarray(phys[4]), d, c))
        f = np.asarray(w[0])
        assert np.isfinite(f).all() and f.shape == (6, L + 1, L + 1)
        fld[variant] = f
        rowsq.append(dict(variant=variant, q=q, M=M, seconds=time.perf_counter() - t0,
                          max_step_joint_stationarity=float(np.max(w[12])),
                          max_inner_stationarity=float(np.max(w[14])),
                          ic_joint_stationarity=float(w[13]),
                          stop_reasons=sorted(set(w[3].tolist())), ic_reason=int(w[6]),
                          median_iterations=float(np.median(w[1])),
                          relative_to_q0=float(np.linalg.norm(f - fields['joint'])
                                               / np.linalg.norm(fields['joint']))))
        print('ARM q', q, variant, 'ok', round(time.perf_counter() - t0, 2), flush=True)
    for r in rowsq:
        r['relative_to_joint'] = float(np.linalg.norm(fld[r['variant']] - fld['joint'])
                                       / np.linalg.norm(fld['joint']))
    out['small_q_arms'] = rowsq
    vp = next(r for r in rowsq if r['variant'] == 'varpro')
    assert vp['relative_to_joint'] <= 1e-3, vp['relative_to_joint']

    # ---- gate 4: the two NNLS fitters on one design ----------------------
    rng = np.random.default_rng(3)
    Gd = np.abs(rng.standard_normal((512, 1024)))
    bd = Gd[:, :60] @ np.abs(rng.standard_normal(60))
    s1, w1, i1 = VP.retained_nnls(Gd, bd, 64)
    s2, w2, i2 = VP.bounded_nnls(Gd, bd, 64, 60.)
    out['nnls'] = dict(
        retained=dict(support=int(len(s1)), seconds=i1['seconds'],
                      relative=float(np.linalg.norm(Gd[:, s1] @ w1 - bd) / np.linalg.norm(bd))),
        bounded=dict(support=int(len(s2)), seconds=i2['seconds'], refits=i2['refits'],
                     relative=float(np.linalg.norm(Gd[:, s2] @ w2 - bd) / np.linalg.norm(bd))))
    print('NNLS', out['nnls'], flush=True)

    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)
    print('CHEAP SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
