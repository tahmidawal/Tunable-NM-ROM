"""Local smoke for per-query head refinement on Burgers. One process, under a minute.

Gate (i): the n = 0 arm, through the NEW code path with theta as a traced runtime
operand, must reproduce the consolidated saved Burgers case and the incumbent
`accuracy_paths.make_rom` to 1e-12 relative. Small V1 and V2 arms are then exercised
for shapes, refinement records and stopping paths.
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
sys.path.insert(0, str(ROOT / 'experiments/head-refine'))
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))

import engines as e            # noqa: E402
import accuracy_paths as ap    # noqa: E402
import arms as A               # noqa: E402
import refine_core as RC       # noqa: E402

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
    theta0, unravel = RC.flatten_theta(params)
    head_of = RC.make_head_of(unravel)
    out['parameters'] = dict(K=K, R=R, theta_count=int(theta0.size),
                             theta_norm=float(jnp.linalg.norm(theta0)))

    # the reconstructed head must be the retained head exactly
    zt = jnp.asarray(Zold[:8])
    d = float(jnp.max(jnp.abs(jax.vmap(lambda z: head_of(theta0, z))(zt)
                              - A.sc.head(params, zt))))
    out['head_roundtrip_max_abs'] = d
    print('HEAD roundtrip', d, flush=True)
    assert d == 0.0

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
    Hrot = A.sc.head(params, arch['candidate_Z']) @ arch['cold_R'].T
    cold = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
            Hrot, jnp.sum(Hrot * Hrot, 1), arch['candidate_Z'])

    # ---------------- gate (i): n = 0 through the new code path --------------
    q0 = RC.make_query(head_of, K, L, cfg['dt'], trust, 'eq', 'baseline', 0,
                       linear='gj', **cfg['strict'])
    v = jax.device_get(q0(u0, jnp.asarray(phys[4]), data, cold, theta0,
                          jnp.asarray(1e-3), jnp.asarray(1e-4)))
    rel = float(np.linalg.norm(v['fields'] - saved['fields']) / np.linalg.norm(saved['fields']))
    relz = float(np.linalg.norm(v['internal'] - saved['internal_latents'])
                 / np.linalg.norm(saved['internal_latents']))
    out['n0_vs_saved_case'] = dict(relative_l2=rel, latent_relative_l2=relz, tolerance=1e-12,
                                   max_step_stationarity=float(np.max(v['stationarity'])),
                                   initial_stationarity=float(v['ic_stationarity']))
    print('GATE n0 vs saved', rel, relz, flush=True)
    assert rel <= 1e-12 and relz <= 1e-12, (rel, relz)

    ref = jax.device_get(ap.make_rom(params, L, cfg['dt'], trust, **cfg['strict'])(
        u0, jnp.asarray(phys[4]),
        (G, arch['A'], arch['lam'], arch['G5'], arch['Pq'], None, None, arch['candidate_Z']),
        (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'], Hrot,
         jnp.sum(Hrot * Hrot, 1))))
    out['n0_vs_incumbent'] = float(np.linalg.norm(v['fields'] - ref[0]) / np.linalg.norm(ref[0]))
    print('GATE n0 vs incumbent', out['n0_vs_incumbent'], flush=True)
    assert out['n0_vs_incumbent'] <= 1e-12

    # ---------------- small refinement arms, both variants -------------------
    rows = []
    for variant, n in (('v1', 2), ('v2', 2)):
        for mu in (1e-3, 1e1):
            t0 = time.perf_counter()
            fn = RC.make_query(head_of, K, L, cfg['dt'], trust, 'eq', variant, n,
                               linear='gj', **cfg['strict'])
            w = jax.device_get(fn(u0, jnp.asarray(phys[4]), data, cold, theta0,
                                  jnp.asarray(mu), jnp.asarray(1e-4)))
            f = np.asarray(w['fields'])
            assert np.isfinite(f).all() and f.shape == (6, L + 1, L + 1), (variant, f.shape)
            assert np.isfinite(w['theta']).all()
            rows.append(dict(variant=variant, n=n, mu=mu, seconds=time.perf_counter() - t0,
                             drift=np.asarray(w['drift']).tolist(),
                             relative_to_n0=float(np.linalg.norm(f - v['fields'])
                                                  / np.linalg.norm(v['fields'])),
                             refine_data_term=np.asarray(w['refine_data_term']).tolist()[:4],
                             refine_anchor_term=np.asarray(w['refine_anchor_term']).tolist()[:4],
                             refine_data_grad=np.asarray(w['refine_data_grad']).tolist()[:4],
                             refine_anchor_grad=np.asarray(w['refine_anchor_grad']).tolist()[:4],
                             stop_reasons=sorted(set(np.asarray(w['reasons']).tolist())),
                             resolve_reasons=sorted(set(np.asarray(w['resolve_reasons']).tolist())),
                             refine_reasons=sorted(set(np.asarray(w['refine_reasons']).tolist())),
                             ic_reason=int(w['ic_reason']),
                             max_step_stationarity=float(np.max(w['stationarity'])),
                             theta_shape=list(np.asarray(w['theta']).shape)))
            print('ARM', variant, 'mu', mu, 'drift', np.asarray(w['drift']),
                  'rel_to_n0', rows[-1]['relative_to_n0'], round(time.perf_counter() - t0, 2),
                  flush=True)
    out['refinement_arms'] = rows

    # the refined manifold reconstruction path must run
    recon = RC.make_reconstruction(head_of, K, budget=100)
    target = jnp.asarray(np.asarray(saved['fields'])[0][1:-1, 1:-1].ravel())
    starts = jnp.asarray(Zold[:8])
    z, rn, it, reason = jax.device_get(recon(starts, target, G, theta0))
    out['reconstruction'] = dict(residual=float(rn), iterations=int(it), reason=int(reason))
    print('RECON ok', out['reconstruction'], flush=True)

    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)
    print('REFINE SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
