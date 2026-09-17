"""Local smoke for the b-qxm lane. One process, 64 intervals, sub-minute.

Gates, in order:

  1 lambda = 0 at q = 0 reproduces the consolidated saved Burgers case to 1e-12 (the same
    audited fixture the parent q-ridge lane gated on): the retained path is the incumbent.
  2 dense operators are a function of M alone: two fresh builds at the same M are bitwise
    identical, and a fresh build's bank G is bitwise the shared bank.
  3 at fixed q, the t = 0 output field does not depend on M (<= 1e-12 relative; bitwise
    recorded), for q in {0, 8} and M in {64, 96, 128, 256}.
  4 every arm reaches the shared stationarity rule and M enters only through the test
    modes: the arm at M = 4(K+q) built through this lane's explicit-M path is BITWISE the
    arm built through the parent's rule path (`test_count('m4')`).
  5 the error is monotone in M at fixed q on this one case (informational, printed).
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
for sub in ('experiments/b-qxm', 'experiments/q-ridge', 'experiments/b-ladder-top',
            'experiments/cheap-corrections', 'experiments/head-ablation', 'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import ladder as LD            # noqa: E402
import varpro as VP            # noqa: E402
import directions as DIR       # noqa: E402
import topfix as TF            # noqa: E402
import ridge as RG             # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FIX = ROOT / 'consolidated/fixtures/burgers'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
              'mr-burgers2d/runs/accuracy09/archive/out/result.json')


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    t_start = time.perf_counter()
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

    # ---- gate 1: lambda = 0 at q = 0 is the incumbent (audited fixture) ------
    C0 = jnp.zeros((R, 0))
    Zaug0 = np.asarray(arch['candidate_Z'])
    head0 = LD.corrected_head(params, C0, K)
    Hrot = jax.jit(jax.vmap(head0))(jnp.asarray(Zaug0)) @ arch['cold_R'].T
    cold0 = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
             Hrot, jnp.sum(Hrot * Hrot, 1), jnp.asarray(Zaug0))
    v0 = jax.device_get(RG.make_query(params, C0, K, 0, L, dt, trust, 'eq', 0.,
                                      linear='gj', **cfg['strict'])(u0, nu, data0, cold0))
    out['gate1_q0_vs_audited_fixture'] = dict(
        relative_l2=rel(v0[0], saved['fields']),
        latent_relative_l2=rel(v0[7], saved['internal_latents']), tolerance=1e-12)
    out['gate1_q0_vs_audited_fixture'].update(
        ic_iterations=int(v0[5]), iterations_first10=np.asarray(v0[1]).tolist()[:10],
        xla_flags=os.environ.get('XLA_FLAGS'),
        passed=bool(out['gate1_q0_vs_audited_fixture']['relative_l2'] <= 1e-12
                    and out['gate1_q0_vs_audited_fixture']['latent_relative_l2'] <= 1e-12))
    # Recorded here and asserted at the END so gates 2-5 still produce their data when this
    # one fails (it did, at 2.1e-7, on the shared GB10: DESIGN.md §9 A0).
    print('GATE 1', out['gate1_q0_vs_audited_fixture'], flush=True)

    # ---- gate 2: dense operators depend on M alone; the bank is shareable ------
    dA, iA = TF.build_operators(bank, L, 128, 'dense')
    dB, iB = TF.build_operators(bank, L, 128, 'dense')
    out['gate2_operators'] = dict(
        rebuild_A_bitwise=bool(np.array_equal(np.asarray(dA['A']), np.asarray(dB['A']))),
        rebuild_Phi_bitwise=bool(np.array_equal(np.asarray(dA['Phi']), np.asarray(dB['Phi']))),
        rebuild_lam_bitwise=bool(np.array_equal(np.asarray(dA['lam']), np.asarray(dB['lam']))),
        fresh_bank_bitwise_equals_shared=bool(np.array_equal(np.asarray(dA['G']), np.asarray(G))),
        modes_ids=iA['mode_ids_count'])
    assert all(v for k, v in out['gate2_operators'].items() if k != 'modes_ids'), out['gate2_operators']
    print('GATE 2', out['gate2_operators'], flush=True)

    # ---- tiny direction fit (the parent smoke's), then the (q, M) cells --------
    train = e.params_draw(0, 4)
    fom, _ = e.make_fom(L, dt, None, .25, dt)
    U = np.concatenate([np.asarray(fom(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-9, 1e-7)[0])
                        [::10][:, 1:-1, 1:-1].reshape(-1, (L - 1) ** 2) for p in train])
    fom_ref = np.asarray(fom(u0, float(phys[4]), 1e-9, 1e-7)[0])[::10]   # 51 steps -> the 6 output times
    del fom
    jax.clear_caches()
    coef = jnp.linalg.solve(Rb, Qb.T @ jnp.asarray(U.T)).T
    Zsub = np.asarray(Zold[::len(Zold) // 1024])
    dcfg = dict(residual_seed=5, residual_snapshots=16, residual_starts=2, residual_budget=80,
                strict=dict(gtol=1e-6), q_ladder=[0, 8])
    C, Ct, Zstar, rho, dinfo = DIR.audited(params, Rb, coef, Zsub, K, dcfg)
    out['directions'] = {k: dinfo[k] for k in ('available_rank', 'head_fit_relative_worst', 'seconds')}

    strict = dict(ic_budget=200, step_budget=120, gtol=1e-6)
    ops = {}
    Gshared = G
    grid = {}
    n0 = np.linalg.norm(fom_ref[0])
    for q in (0, 8):
        Cq = C[:, :q]
        head = LD.corrected_head(params, Cq, K)
        cq, _ = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1), 24)
        t0f = None
        for M in (64, 96, 128, 256):
            if M not in ops:
                d, _ = TF.build_operators(bank, L, M, 'dense')
                assert np.array_equal(np.asarray(d['G']), np.asarray(Gshared))
                d['G'] = Gshared
                ops[M] = d
            d = ops[M]
            fn = RG.make_query(params, Cq, K, q, L, dt, trust, 'dense', 0., linear='gj',
                               inner_damping=1e-10, **strict)
            v = jax.device_get(fn(u0, nu, d, cq))
            f = np.asarray(v[0])
            assert np.isfinite(f).all() and f.shape == (6, L + 1, L + 1)
            err = np.linalg.norm((f - fom_ref).reshape(6, -1), axis=1) / n0
            row = dict(q=q, M=M, worst_evolved=float(err[1:].max()), t0=float(err[0]),
                       max_step_joint_stationarity=float(np.max(v[12])),
                       budget_exits=int(sum(1 for r in v[3].tolist() if r == 0)),
                       median_iterations=float(np.median(v[1])))
            if t0f is None:
                t0f = f[0]
                row['t0_relative_to_first_M'] = 0.
                row['t0_bitwise_to_first_M'] = True
            else:
                row['t0_relative_to_first_M'] = rel(f[0], t0f)
                row['t0_bitwise_to_first_M'] = bool(np.array_equal(f[0], t0f))
            assert row['t0_relative_to_first_M'] <= 1e-12, row                 # gate 3
            assert row['max_step_joint_stationarity'] <= 1e-6 * (1 + 1e-7), row   # gate 4a
            assert row['budget_exits'] == 0, row
            grid[f'q{q}_M{M}'] = row
            print('CELL', row, flush=True)
            # gate 4b: at M = 4(K+q) this lane's explicit-M cell is BITWISE the parent's rule path
            if M == 4 * (K + q):
                Mp = 4 * (K + q)
                dp, _ = TF.build_operators(bank, L, Mp, 'dense')
                base = jax.device_get(VP.make_query(params, Cq, K, q, L, dt, trust, 'dense',
                                                    'block' if q else 'joint', linear='gj',
                                                    inner_damping=1e-10,
                                                    **({} if q else dict(inner_iters=1)),
                                                    **strict)(u0, nu, dp, cq)) if q else None
                if base is not None:
                    row['bitwise_to_retained_block_at_4x'] = bool(np.array_equal(f, np.asarray(base[0])))
                    assert row['bitwise_to_retained_block_at_4x'], (q, M)
                    print('GATE 4b q', q, 'M', M, 'bitwise to retained block', flush=True)
        ev = [grid[f'q{q}_M{M}']['worst_evolved'] for M in (64, 96, 128, 256)]
        out[f'q{q}_monotone_in_M'] = bool(all(b <= a for a, b in zip(ev, ev[1:])))
        print('MONOTONE in M at q', q, out[f'q{q}_monotone_in_M'], ev, flush=True)
    out['grid'] = grid
    out['seconds'] = time.perf_counter() - t_start
    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    assert out['gate1_q0_vs_audited_fixture']['passed'], out['gate1_q0_vs_audited_fixture']
    print('XM SMOKE OK', round(out['seconds'], 1), 's', flush=True)


if __name__ == '__main__':
    main()
