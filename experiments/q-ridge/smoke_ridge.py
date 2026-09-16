"""Local smoke for the q-ridge lane. One process, 64 intervals, small q, sub-minute.

Gates, in order:

  1 lambda = 0 at q = 0 reproduces the consolidated saved Burgers case to 1e-12 — the
    lambda = 0 branch IS `topfix`/`varpro`'s retained path, so this is the incumbent.
  2 lambda = 0 at q > 0 is BITWISE the retained `varpro.make_block_lm` solver.
  3 the ridge does what it says: ||y|| is non-increasing in lambda, a huge lambda drives
    y to zero, and the field then agrees with the q = 0 solution.
  4 every ridged arm reaches the shared stationarity rule ON THE RIDGED OBJECTIVE.
  5 the t = 0 output field is BITWISE invariant in lambda (the IC fit is not ridged).
  6 the lambda scaling is the claimed one: the exact linear part of dr_w/dy is A C_q
    (the diffusion row scaling cancels), so sigma_q = ||A C_q||_2.
  7 the NumPy R3 kernels reproduce the JAX originals: modes bitwise, advection to 1e-12,
    and u_n = G c_n reproduces the query's own output fields to 1e-12.
  8 the walltime-bounded NNLS fitter returns a valid (untruncated) rule.

The cross-job fidelity reproductions are in-job gates: those numbers are only comparable
on a cluster A100.
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
for sub in ('experiments/q-ridge', 'experiments/b-ladder-top', 'experiments/cheap-corrections',
            'experiments/head-ablation', 'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import ladder as LD            # noqa: E402
import varpro as VP            # noqa: E402
import directions as DIR       # noqa: E402
import topfix as TF            # noqa: E402
import ridge as RG             # noqa: E402
import r3 as R3                # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FIX = ROOT / 'consolidated/fixtures/burgers'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
              'mr-burgers2d/runs/accuracy09/archive/out/result.json')
LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.]


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


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
    L, dt = 64, cfg['dt']
    arch = {k: jnp.asarray(v) for k, v in np.load(FIX / 'operators.npz').items()}
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    data0 = dict(A=arch['A'], lam=arch['lam'], G5=arch['G5'], Pq=arch['Pq'], G=G)
    setup = next(r for r in raw['mesh_setup'] if (r['intervals'], r['model']) == (64, 'frozen'))
    trust = setup['trust_radius']
    phys = raw['physical_cases'][0]
    u0 = jnp.asarray(e.initial(L, phys))
    nu = jnp.asarray(phys[4])
    saved = np.load(FIX / 'expected.npz')
    Qb, Rb = A.whiten(G)

    # ---- gate 1: lambda = 0 at q = 0 is the incumbent ---------------------
    C0 = jnp.zeros((R, 0))
    Zaug = np.asarray(arch['candidate_Z'])
    head0 = LD.corrected_head(params, C0, K)
    Hrot = jax.jit(jax.vmap(head0))(jnp.asarray(Zaug)) @ arch['cold_R'].T
    cold0 = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
             Hrot, jnp.sum(Hrot * Hrot, 1), jnp.asarray(Zaug))
    v0 = jax.device_get(RG.make_query(params, C0, K, 0, L, dt, trust, 'eq', 0.,
                                      linear='gj', **cfg['strict'])(u0, nu, data0, cold0))
    out['q0_vs_saved_case'] = dict(relative_l2=rel(v0[0], saved['fields']),
                                   latent_relative_l2=rel(v0[7], saved['internal_latents']),
                                   tolerance=1e-12)
    assert out['q0_vs_saved_case']['relative_l2'] <= 1e-12, out['q0_vs_saved_case']
    assert out['q0_vs_saved_case']['latent_relative_l2'] <= 1e-12, out['q0_vs_saved_case']
    print('GATE 1', out['q0_vs_saved_case'], flush=True)

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
                                               'seconds')}
    # the field-whitening claim the ridge's metric rests on: G C_q has orthonormal columns
    GC = np.asarray(G @ C[:, :16])
    out['field_whitened_directions_deviation'] = float(
        np.max(np.abs(GC.T @ GC - np.eye(16))))
    assert out['field_whitened_directions_deviation'] < 1e-8, out
    print('WHITENED', out['field_whitened_directions_deviation'], flush=True)

    strict = dict(ic_budget=200, step_budget=120, gtol=1e-6)
    per_q = {}
    for q in (8, 16):
        M = 4 * (K + q)
        Cq = C[:, :q]
        head = LD.corrected_head(params, Cq, K)
        d, _ = TF.build_operators(bank, L, M, 'dense')
        cq, _ = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1), 24)
        sig = RG.sigma_q(d, Cq)

        # ---- gate 6: the linear part of dr_w/dy is exactly A C_q ----------
        wk = A.weak_fn('dense')
        wv = jnp.asarray(np.concatenate((Zsub[0], np.zeros(q))))
        prev = d['A'] @ head(wv)
        lamv = d['lam']
        full = jax.jacfwd(lambda y: wk(jnp.concatenate((wv[:K], y)), prev, nu, d, head, L, dt))(wv[K:])
        advp = jax.jacfwd(lambda y: dt * (d['Phi'].T @ e.spatial(
            d['G'] @ head(jnp.concatenate((wv[:K], y))), L)[0]) / (1 + dt * nu * lamv))(wv[K:])
        linear_dev = float(np.max(np.abs(np.asarray(full - advp) - np.asarray(d['A'] @ Cq))))
        sigma_dev = abs(sig - float(np.linalg.norm(np.asarray(d['A'] @ Cq), 2))) / sig
        assert linear_dev < 1e-9, (q, linear_dev)
        assert sigma_dev < 1e-12, (q, sigma_dev)
        print('GATE 6 q', q, 'linear_dev', linear_dev, 'sigma', sig, flush=True)

        # ---- gates 2-5: lambda = 0 bitwise, then the ridge ----------------
        base = jax.device_get(VP.make_query(params, Cq, K, q, L, dt, trust, 'dense', 'block',
                                            linear='gj', inner_damping=1e-10,
                                            **strict)(u0, nu, d, cq))
        rows = {}
        prev_y = None
        for lam_rel in [0.] + LAMS + [1e6]:
            t0 = time.perf_counter()
            fn = RG.make_query(params, Cq, K, q, L, dt, trust, 'dense', lam_rel * sig ** 2,
                               linear='gj', inner_damping=1e-10, **strict)
            v = jax.device_get(fn(u0, nu, d, cq))
            f = np.asarray(v[0])
            assert np.isfinite(f).all() and f.shape == (6, L + 1, L + 1)
            # The IC fit is deliberately NOT ridged, so ||y|| is measured over the EVOLVED
            # steps only; internal row 0 is the unridged initial fit and is invariant by
            # construction (gate 5 states that as a bitwise fact about the t = 0 field).
            yall = np.linalg.norm(np.asarray(v[7])[:, K:], axis=1)
            yn = float(np.max(yall[1:]))
            rows[f'{lam_rel:g}'] = dict(
                lam_rel=lam_rel, max_y_norm=yn, ic_y_norm=float(yall[0]),
                relative_to_retained_block=rel(f, base[0]),
                bitwise_to_retained_block=bool(np.array_equal(f, np.asarray(base[0]))),
                t0_bitwise_to_lambda0=bool(np.array_equal(f[0], np.asarray(base[0])[0])),
                max_step_joint_stationarity=float(np.max(v[12])),
                ic_joint_stationarity=float(v[13]),
                budget_exits=int(sum(1 for r in v[3].tolist() if r == 0)),
                median_iterations=float(np.median(v[1])), seconds=time.perf_counter() - t0)
            if lam_rel == 0.:
                assert rows['0']['bitwise_to_retained_block'], (q, 'lambda 0 must be the incumbent')
            else:
                assert yn <= prev_y * (1 + 1e-2) + 1e-14, (q, lam_rel, yn, prev_y)   # gate 3
                assert rows[f'{lam_rel:g}']['max_step_joint_stationarity'] <= 1e-6 * (1 + 1e-7), \
                    (q, lam_rel)                                            # gate 4
            assert rows[f'{lam_rel:g}']['t0_bitwise_to_lambda0'], (q, lam_rel)   # gate 5
            prev_y = yn
            print('RIDGE q', q, 'lam_rel', lam_rel, 'max|y|', yn,
                  'rel', rows[f'{lam_rel:g}']['relative_to_retained_block'], flush=True)
            if lam_rel == 1e6:
                per_q[q] = dict(huge=f, rows=rows, coefficients=np.asarray(
                    jax.device_get(RG.coefficients(params, Cq, K)(jnp.asarray(v[7])))),
                    fields=f, latents=np.asarray(v[7]))
        assert rows['1e+06']['max_y_norm'] < 1e-5 * max(1e-300, rows['0']['max_y_norm']), (
            q, rows['1e+06']['max_y_norm'], rows['0']['max_y_norm'])
        assert rows['1e+06']['ic_y_norm'] == rows['0']['ic_y_norm'], q
        out.setdefault('ridge', {})[str(q)] = rows

    # a huge lambda must return the q = 0 answer
    v00 = jax.device_get(RG.make_query(params, C0, K, 0, L, dt, trust, 'dense', 0., linear='gj',
                                       **strict)(u0, nu,
                                                 TF.build_operators(bank, L, 4 * K, 'dense')[0],
                                                 cold0))
    # the q = 0 arm uses its own M = 4K test set, so this is a physical, not bitwise, check
    out['huge_lambda_vs_q0'] = {str(q): rel(per_q[q]['huge'], v00[0]) for q in per_q}
    print('HUGE-LAMBDA vs q0', out['huge_lambda_vs_q0'], flush=True)
    assert all(x < 5e-2 for x in out['huge_lambda_vs_q0'].values()), out['huge_lambda_vs_q0']

    # ---- gate 7: the NumPy R3 kernels ------------------------------------
    Phi_j, lam_j, ids_j = e.modes(L, 64)
    Phi_n, lam_n, ids_n = R3.modes(L, 64)
    held_n, heldlam_n, _ = R3.modes(L, 64, skip=64)
    rng = np.random.default_rng(11)
    uu = rng.standard_normal((L - 1) ** 2) * .1
    adv_j, lap_j = e.spatial(jnp.asarray(uu), L)
    adv_n, lap_n = R3.spatial(uu, L)
    Gn = np.asarray(G)
    coefn = per_q[16]['coefficients']
    decoded = (coefn @ Gn.T)[::10]
    fields_from_coef = np.stack([np.pad(x.reshape(L - 1, L - 1), 1) for x in decoded])
    out['r3_kernels'] = dict(
        modes_bitwise=bool(np.array_equal(Phi_j, Phi_n) and np.array_equal(lam_j, lam_n)
                           and np.array_equal(ids_j, ids_n)),
        held_out_disjoint=bool(not set(map(tuple, ids_n)) & set(map(tuple, R3.modes(L, 64, 64)[2]))),
        advection_relative=rel(adv_n, np.asarray(adv_j)),
        laplacian_relative=rel(lap_n, np.asarray(lap_j)),
        decode_relative=rel(fields_from_coef, per_q[16]['fields']),
        held_out_modes=int(held_n.shape[1]), held_out_lambda_min=float(heldlam_n.min()),
        in_space_lambda_max=float(lam_n.max()))
    assert out['r3_kernels']['modes_bitwise']
    assert out['r3_kernels']['held_out_disjoint']
    assert out['r3_kernels']['advection_relative'] < 1e-12
    assert out['r3_kernels']['decode_relative'] < 1e-12, out['r3_kernels']
    assert out['r3_kernels']['held_out_lambda_min'] >= out['r3_kernels']['in_space_lambda_max']
    print('GATE 7', out['r3_kernels'], flush=True)

    # the residual machinery itself runs and separates the two blocks
    res = R3.trajectory_residuals(Gn, coefn, {'in_space': (Phi_n, lam_n),
                                              'held_out': (held_n, heldlam_n)},
                                  float(nu), dt, L)
    out['r3_example'] = {k: {kk: vv for kk, vv in v.items() if not kk.endswith('per_step')}
                         for k, v in res.items() if isinstance(v, dict)}
    assert np.isfinite(res['in_space']['raw_max']) and np.isfinite(res['held_out']['raw_max'])
    print('R3 EXAMPLE', out['r3_example'], flush=True)

    # ---- gate 8: the bounded fitter --------------------------------------
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
    print('RIDGE SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
