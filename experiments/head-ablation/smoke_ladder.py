"""Local smoke for the correction ladder. Under a minute; one process.

Gate: the q = 0 ladder arm must reproduce head-ablation arm (a) on the saved
consolidated Burgers case to round-off, through the corrected-head wrapper and
the archived operators. A tiny q > 0 arm is then exercised for shapes and
stopping paths.
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
sys.path.insert(0, str(ROOT / 'experiments/head-ablation'))
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))

import engines as e            # noqa: E402
import accuracy_paths as ap    # noqa: E402
import arms as A               # noqa: E402
import ladder as LD            # noqa: E402

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

    # ---- gate: q = 0 through the corrected-head wrapper -------------------
    C0 = jnp.zeros((R, 0))
    head0 = LD.corrected_head(params, C0, K)
    Zaug = np.asarray(arch['candidate_Z'])
    Hrot = jax.jit(jax.vmap(head0))(jnp.asarray(Zaug)) @ arch['cold_R'].T
    cold = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
            Hrot, jnp.sum(Hrot * Hrot, 1), jnp.asarray(Zaug))
    q0 = A.make_query(head0, K, L, cfg['dt'], trust, 'eq', linear='gj', **cfg['strict'])
    v = jax.device_get(q0(u0, jnp.asarray(phys[4]), data, cold))
    rel = float(np.linalg.norm(v[0] - saved['fields']) / np.linalg.norm(saved['fields']))
    relz = float(np.linalg.norm(v[7] - saved['internal_latents'])
                 / np.linalg.norm(saved['internal_latents']))
    out['q0_vs_saved_case'] = dict(relative_l2=rel, latent_relative_l2=relz, tolerance=1e-12,
                                   max_step_stationarity=float(np.max(v[8])),
                                   initial_stationarity=float(v[9]))
    print('GATE q0 vs saved', rel, relz, flush=True)
    assert rel <= 1e-12 and relz <= 1e-12, (rel, relz)

    # the incumbent solver on the same input, for a second independent comparison
    ref = jax.device_get(ap.make_rom(params, L, cfg['dt'], trust, **cfg['strict'])(
        u0, jnp.asarray(phys[4]),
        (G, arch['A'], arch['lam'], arch['G5'], arch['Pq'], None, None, arch['candidate_Z']),
        (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'], Hrot,
         jnp.sum(Hrot * Hrot, 1))))
    out['q0_vs_incumbent'] = float(np.linalg.norm(v[0] - ref[0]) / np.linalg.norm(ref[0]))
    print('GATE q0 vs incumbent', out['q0_vs_incumbent'], flush=True)
    assert out['q0_vs_incumbent'] == 0.0

    # ---- a small q > 0 arm, dense quadrature, same input ------------------
    Qb, Rb = A.whiten(G)
    rng = np.random.default_rng(11)
    train = e.params_draw(0, 4)
    fom, _ = e.make_fom(L, cfg['dt'], None, .25, cfg['dt'])
    U = np.concatenate([np.asarray(fom(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-9, 1e-7)[0])
                        [::10][:, 1:-1, 1:-1].reshape(-1, (L - 1) ** 2) for p in train])
    del fom
    jax.clear_caches()
    coef = jnp.linalg.solve(Rb, Qb.T @ jnp.asarray(U.T)).T
    Zsub = np.asarray(Zold[::len(Zold) // 1024])
    C, dinfo = LD.residual_directions(params, Rb, coef, Zsub, K, dict(
        residual_seed=5, residual_snapshots=16, residual_starts=2, residual_budget=80,
        strict=dict(gtol=1e-6), q_ladder=[0, 8]))
    out['directions'] = {k: dinfo[k] for k in
                         ('available_rank', 'head_fit_relative_median', 'head_fit_relative_worst',
                          'residual_energy_captured', 'head_fit_reason_counts')}
    rows = []
    for q in (8,):
        M = 4 * (K + q)
        head = LD.corrected_head(params, C[:, :q], K)
        Za = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
        t0 = time.perf_counter()
        d, i = A.build_operators(bank, L, M, 'dense', head=head)
        c, ci = A.build_cold(bank, head, Za, 24)
        fn = A.make_query(head, K + q, L, cfg['dt'], trust, 'dense', linear='gj',
                          ic_budget=200, step_budget=60, gtol=1e-6)
        w = jax.device_get(fn(u0, jnp.asarray(phys[4]), d, c))
        f = np.asarray(w[0])
        assert np.isfinite(f).all() and f.shape == (6, L + 1, L + 1)
        rows.append(dict(q=q, M=M, solved_dimension=K + q, seconds=time.perf_counter() - t0,
                         max_step_stationarity=float(np.max(w[8])),
                         stop_reasons=sorted(set(w[3].tolist())), ic_reason=int(w[6]),
                         relative_to_q0=float(np.linalg.norm(f - v[0]) / np.linalg.norm(v[0]))))
        print('ARM q', q, 'ok', round(time.perf_counter() - t0, 2), flush=True)
    out['small_q_arms'] = rows
    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)
    print('LADDER SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
