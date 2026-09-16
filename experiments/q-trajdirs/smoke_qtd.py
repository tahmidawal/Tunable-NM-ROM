"""Local smoke for the trajectory-fitted directions. One process, 64 intervals, tiny q.

Gates, in order:

  1 `trajdirs.pod_from_residual`, handed the INCUMBENT rule's own residual matrix,
    reproduces `directions.audited`'s direction matrix BITWISE. That is what makes
    `old` and `traj` differ only in which residual is decomposed.
  2 the q = 0 query at the retained per-step budget and at budget 600 are bitwise
    identical, and both reproduce the consolidated saved Burgers case to 1e-12
    (routine call 2 of DESIGN.md, checked rather than asserted).
  3 `trajdirs.build` runs end to end on a tiny training cohort: the FOM step count
    matches the ROM internal-latent count, the available rank is the row count, the
    directions are field-orthonormal and nested.
  4 `cross_capture` of a residual matrix by its OWN directions equals that set's
    reported energy captured; `principal_angles` of a set against itself are zero.
  5 a q > 0 arm built exactly as the driver builds it runs, is finite, and its
    enriched quadrature codes have the right shape.

The real fidelity reproductions of the cheap-corrections rows are in-job gates: those
numbers are only comparable on a cluster A100.
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
for sub in ('experiments/q-trajdirs', 'experiments/b-ladder-top',
            'experiments/cheap-corrections', 'experiments/head-ablation',
            'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))

import engines as e            # noqa: E402
import arms as A               # noqa: E402
import ladder as LD            # noqa: E402
import varpro as VP            # noqa: E402
import directions as DIR       # noqa: E402
import topfix as TF            # noqa: E402
import trajdirs as TD          # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FIX = ROOT / 'consolidated/fixtures/burgers'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
              'mr-burgers2d/runs/accuracy09/archive/out/result.json')


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    t_all = time.perf_counter()
    out = {}
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    K, R = int(Zold.shape[1]), int(np.asarray(ck['params']['h_lin']).shape[1])
    raw = json.loads(RAW.read_text())
    rcfg = raw['config']
    L, dt = 64, rcfg['dt']
    setup = next(r for r in raw['mesh_setup'] if (r['intervals'], r['model']) == (64, 'frozen'))
    trust = setup['trust_radius']
    arch = {k: jnp.asarray(v) for k, v in np.load(FIX / 'operators.npz').items()}
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)
    phys = raw['physical_cases'][0]
    u0 = jnp.asarray(e.initial(L, phys))
    nu = jnp.asarray(phys[4])
    saved = np.load(FIX / 'expected.npz')

    # ---- gate 2: the q = 0 query at the retained budget and at 600 ----------
    C0 = jnp.zeros((R, 0))
    head0 = LD.corrected_head(params, C0, K)
    Zaug = np.asarray(arch['candidate_Z'])
    Hrot = jax.jit(jax.vmap(head0))(jnp.asarray(Zaug)) @ arch['cold_R'].T
    cold0 = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
             Hrot, jnp.sum(Hrot * Hrot, 1), jnp.asarray(Zaug))
    data0 = dict(A=arch['A'], lam=arch['lam'], G5=arch['G5'], Pq=arch['Pq'], G=G)
    fields = {}
    rows = {}
    for label, budget in (('retained', rcfg['strict']['step_budget']), ('b600', 600)):
        st = dict(rcfg['strict'])
        st['step_budget'] = budget
        fn = TF.make_query(params, C0, K, 0, L, dt, trust, 'eq', 'base', Rb=Rb, linear='gj', **st)
        v = jax.device_get(fn(u0, nu, data0, cold0))
        fields[label] = np.asarray(v[0])
        r0 = rel(v[0], saved['fields'])
        rows[label] = dict(step_budget=int(budget), relative_l2_to_saved_case=r0,
                           latent_relative_l2=rel(v[7], saved['internal_latents']),
                           budget_exits=int(np.sum(np.asarray(v[3]) == 0)),
                           max_step_joint_stationarity=float(np.max(v[12])))
        assert r0 <= 1e-12, (label, r0)
        print('GATE2', label, r0, flush=True)
    rows['bitwise_identical'] = bool(np.array_equal(fields['retained'], fields['b600']))
    assert rows['bitwise_identical'], 'budget 600 changed the q = 0 trajectory'
    out['q0_budget'] = rows

    # ---- gate 1: the shared POD reproduces the incumbent rule bitwise --------
    train = e.params_draw(0, 4)
    fom, _ = e.make_fom(L, dt, None, .25, dt)
    U = np.concatenate([np.asarray(fom(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-9, 1e-7)[0])
                        [::10][:, 1:-1, 1:-1].reshape(-1, (L - 1) ** 2) for p in train])
    del fom
    jax.clear_caches()
    coef = jnp.linalg.solve(Rb, Qb.T @ jnp.asarray(U.T)).T
    Zsub = np.asarray(Zold[::len(Zold) // 512])
    qlad = [0, 4, 8]
    dcfg = dict(residual_seed=5, residual_snapshots=16, residual_starts=2, residual_budget=80,
                strict=dict(gtol=1e-6), q_ladder=qlad)
    Cold, Ct_old, Zstar, rho_old, dinfo = DIR.audited(params, Rb, coef, Zsub, K, dcfg)
    C2, Ct2, sv2, tilde2, info2 = TD.pod_from_residual(rho_old, Rb, R, qlad)
    out['shared_pod_reproduces_audited'] = dict(
        directions_bitwise=bool(np.array_equal(np.asarray(Cold), np.asarray(C2))),
        field_orthonormal_bitwise=bool(np.array_equal(np.asarray(Ct_old), np.asarray(Ct2))),
        audited_sha256=dinfo['directions_sha256'], shared_sha256=info2['directions_sha256'],
        energy_captured_audited=dinfo['residual_energy_captured'],
        energy_captured_shared=info2['residual_energy_captured'])
    assert out['shared_pod_reproduces_audited']['directions_bitwise']
    assert dinfo['directions_sha256'] == info2['directions_sha256']
    print('GATE1 ok', dinfo['directions_sha256'][:16], flush=True)

    # ---- gate 3: the trajectory rule end to end -----------------------------
    M0 = 4 * K
    d0, _ = TF.build_operators(bank, L, M0, 'dense')
    cq0, _ = A.build_cold(bank, LD.corrected_head(params, C0, K), Zsub, 24)
    qfn0 = TF.make_query(params, C0, K, 0, L, dt, trust, 'dense', 'base', Rb=Rb, linear='gj',
                         ic_budget=200, step_budget=600, gtol=1e-6)
    query0 = lambda a, b: jax.device_get(qfn0(a, b, d0, cq0))
    hz = VP.head_only(params)
    cohort = e.params_draw(0, 128)[[3, 11]]
    Ctraj, Cttraj, Ztr, Ptr, tilde_tr, tinfo = TD.build(
        query0, hz, Qb, Rb, cohort, L, dt, 1e-6, 1e-8, R, qlad, rule='smoke')
    steps = int(round(.25 / dt)) + 1
    assert Ptr.shape == (2 * steps, R), Ptr.shape
    assert Ztr.shape == (2 * steps, K), Ztr.shape
    orth = float(np.max(np.abs(np.asarray(jnp.asarray(Cttraj).T @ jnp.asarray(Cttraj))
                               - np.eye(np.asarray(Cttraj).shape[1]))))
    out['trajectory_rule'] = dict(
        residual_rows=int(Ptr.shape[0]), available_rank=tinfo['available_rank'],
        orthonormality_deviation=orth, seconds=tinfo['seconds'],
        energy_captured=tinfo['residual_energy_captured'],
        relative_residual_worst=tinfo['trajectory']['relative_residual_worst'],
        relative_residual_median=tinfo['trajectory']['relative_residual_median'],
        fom_max_relative_residual=tinfo['trajectory']['fom_max_relative_residual'],
        prefix_sha256=TD.prefix_hashes(np.asarray(Ctraj), qlad))
    assert tinfo['available_rank'] == min(Ptr.shape[0], R)
    assert orth < 1e-10, orth
    print('GATE3 ok rows', Ptr.shape[0], 'rank', tinfo['available_rank'], flush=True)

    # ---- gate 4: the comparison helpers -------------------------------------
    self_cap = TD.cross_capture(tilde_tr, Cttraj, qlad)
    cross = TD.cross_capture(tilde2, Cttraj, qlad)
    ang = TD.principal_angles(Cttraj, Cttraj, [4, 8])
    out['comparisons'] = dict(self_capture=self_cap, reported=tinfo['residual_energy_captured'],
                              traj_residual_by_old_directions=TD.cross_capture(tilde_tr, Ct_old, qlad),
                              old_residual_by_traj_directions=cross,
                              self_angles={k: dict(angle_max_degrees=v['angle_max_degrees'],
                                                   overlap=v['overlap']) for k, v in ang.items()})
    for q in qlad:
        assert abs(self_cap[str(q)] - tinfo['residual_energy_captured'][str(q)]) < 1e-10, q
    # arccos is sqrt-ill-conditioned at sigma = 1, so a self-comparison lands near
    # 1e-6 degrees rather than at zero; the well-conditioned quantity is the overlap.
    assert all(abs(v['overlap'] - 1.) < 1e-12 for v in ang.values()), ang
    assert all(v['angle_max_degrees'] < 1e-3 for v in ang.values()), ang
    print('GATE4 ok', flush=True)

    # ---- gate 5: one q > 0 arm built the way the driver builds it -----------
    q = 8
    M = 4 * (K + q)
    Cq = Ctraj[:, :q]                         # C = Rb^{-1} Ct, already solved inside build()
    head = TF.corrected_head(params, Cq, K)
    W = TF.enriched_codes(Ztr, Ptr, Rb, Cttraj, q)
    assert W.shape == (Ptr.shape[0], K + q), W.shape
    dq, iq = TF.build_operators(bank, L, M, 'eq', head=head, Wcodes=W, m=min(4 * M, 256),
                                fitter='bounded', eq_seed=20259, candidate_cap=2048,
                                fit_states=8, max_fit_rows=2048, eq_seconds=20., blocks_wanted=16)
    cq, _ = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1), 24)
    fn = TF.make_query(params, Cq, K, q, L, dt, trust, 'eq', 'base', Rb=Rb, linear='gj',
                       ic_budget=200, step_budget=200, gtol=1e-6, ic_gtol=1e-6,
                       inner_damping=1e-10, tau_y=.1)
    v = jax.device_get(fn(u0, nu, dq, cq))
    f = np.asarray(v[0])
    assert np.isfinite(f).all() and f.shape == (6, L + 1, L + 1)
    out['arm_q8'] = dict(q=q, M=M, m=iq.get('m'), eq_rule_valid=iq.get('eq_rule_valid'),
                         eq_relative_fit=iq.get('eq_relative_fit'),
                         enriched_codes_shape=list(W.shape),
                         relative_to_q0=rel(f, fields['b600']),
                         max_step_joint_stationarity=float(np.max(v[12])),
                         budget_exits=int(np.sum(np.asarray(v[3]) == 0)),
                         median_iterations=float(np.median(np.asarray(v[1]))))
    print('GATE5 ok', out['arm_q8'], flush=True)

    out['seconds'] = time.perf_counter() - t_all
    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print('Q-TRAJDIRS SMOKE OK', round(out['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
