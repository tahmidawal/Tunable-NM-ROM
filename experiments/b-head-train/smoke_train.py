"""Local smoke for the TRAINING path. Sub-minute, tiny mesh, local GB10.

Four gates, all of which must pass before a training job is staged:

1. the whitening reparameterisation q = Lam^T h is exact, both ways;
2. identity (*) holds against fields the FOM actually produced;
3. `common.weak_residual` in the whitened parameterisation is the SAME residual
   `arms.weak_dense` evaluates -- the training objective's solve-aware term is
   the ROM's own weak form, not a look-alike;
4. `train.fit_ext` at beta = gamma = 0 reproduces `sep_hfit.fit` step for step
   and draw for draw, so an objective arm differs from a density arm in the
   added term and in nothing else.
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
for d in ('experiments/mr-burgers2d', 'experiments/head-ablation',
          'experiments/separable-decoder', 'experiments/b-head-train'):
    sys.path.insert(0, str(ROOT / d))

import engines as e          # noqa: E402
import arms as A             # noqa: E402
import sep_hfit as hf        # noqa: E402
import common as C           # noqa: E402
import train as T            # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    out = {}
    t0 = time.perf_counter()
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    K = int(np.asarray(ck['Z_tr']).shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L, dt = 64, .005

    bank = A.CoordBank(params, K, R)
    G = jax.block_until_ready(bank.on_grid(L))
    Gram, Lam, Linv = C.bank_algebra(G)

    # ---- gate 1: whitening round trip ---------------------------------------
    qp = hf.to_q(dict(h=params['h'], h_lin=params['h_lin']), Lam)
    rt_h = hf.to_h(qp, Linv)
    rt = float(max(jnp.max(jnp.abs(rt_h['h'][-1][0] - params['h'][-1][0]))
                   / jnp.max(jnp.abs(params['h'][-1][0])),
                   jnp.max(jnp.abs(rt_h['h_lin'] - params['h_lin']))
                   / jnp.max(jnp.abs(params['h_lin']))))
    z8 = jnp.asarray(ck['Z_tr'][:8])
    wt = float(jnp.max(jnp.abs(hf.head_apply(qp, z8) - hf.head_apply(
        dict(h=params['h'], h_lin=params['h_lin']), z8) @ Lam))
        / jnp.max(jnp.abs(hf.head_apply(qp, z8))))
    out['whitening'] = dict(round_trip=rt, q_equals_LT_h=wt, tolerance=1e-10)
    assert rt < 1e-10 and wt < 1e-10, (rt, wt)
    print('GATE1 whitening', rt, wt, flush=True)

    # ---- gate 2: identity (*) against real fields ---------------------------
    project = C.make_projector(G, Lam)
    phys = C.incumbent_draw(4, 0)
    tr, tinfo = C.generate_projected(L, dt, phys, project, 1e-9, 1e-7, stride=1)
    q, _ = e.make_fom(L, dt, None, .25, dt)
    dev = 0.
    for s in (0, 30, 77, 151):
        ti, tj = int(tr['traj'][s]), int(tr['t'][s])
        f = np.asarray(q(jnp.asarray(e.initial(L, phys[ti])), float(phys[ti][4]), 1e-9, 1e-7)[0])
        u = f[tj][1:-1, 1:-1].reshape(-1)
        c = np.asarray(jnp.asarray(tr['a'][s])[None] @ Linv)[0]
        lhs = float(np.linalg.norm(np.asarray(G @ jnp.asarray(c)) - u) ** 2)
        dev = max(dev, abs(lhs - float(tr['fl2'][s])) / max(float(tr['fl2'][s]), 1e-300))
    del q
    jax.clear_caches()
    out['identity_star'] = dict(relative=dev, tolerance=1e-9, states=int(tr['a'].shape[0]))
    assert dev < 1e-9, dev
    print('GATE2 identity', dev, flush=True)

    # ---- gate 3: the training weak residual IS arms.weak_dense --------------
    M = 4 * K
    data, _ = A.build_operators(bank, L, M, 'dense')
    wp = C.weak_pieces(G, Linv, L, M)
    head = A.neural_head(params)
    zt = jnp.asarray(ck['Z_tr'][17])
    zp = jnp.asarray(ck['Z_tr'][18])
    nu = jnp.asarray([float(phys[0][4])])
    prev = data['A'] @ head(zp)
    ref = A.weak_dense(zt, prev, float(nu[0]), data, head, L, dt)
    mine = C.weak_residual(wp, hf.head_apply(qp, zt[None]), prev[None], nu, L, dt)[0]
    rel = float(jnp.linalg.norm(mine - ref) / jnp.linalg.norm(ref))
    out['weak_residual_matches_arms'] = dict(relative=rel, tolerance=1e-11)
    assert rel < 1e-11, rel
    print('GATE3 weak residual', rel, flush=True)

    # ---- gate 4: fit_ext at beta=gamma=0 IS sep_hfit.fit --------------------
    S = tr['a'].shape[0]
    A_tr = jnp.asarray(tr['a'])
    un2, fl2 = jnp.asarray(tr['un2']), jnp.asarray(tr['fl2'])
    prev_i = C.previous_index(tr['traj'], tr['t'])
    nbr_i = C.neighbour_index(phys, tr['traj'], tr['t'])
    steps, bs = 40, 16
    qa, Za, _, trace = T.fit_ext(jax.random.PRNGKey(11), A_tr, un2, fl2, K, R, steps=steps,
                                 lr=1e-3, hidden=32, layers=2, batch=bs, wp=wp,
                                 prev=prev_i, nbr=nbr_i, nu=tr['nu'], res_batch=4, L=L, dt=dt,
                                 log_every=10 ** 9, tag='ref', probe=steps)
    qb, Zb, ib = hf.fit(jax.random.PRNGKey(11), A_tr, un2, fl2, K, R, steps=steps, lr=1e-3,
                        hidden=32, layers=2, batch=bs, log_every=10 ** 9, tag='hf')
    dq = float(max(jnp.max(jnp.abs(x[0] - y[0])) for x, y in zip(qa['h'], qb['h'])))
    dz = float(jnp.max(jnp.abs(Za - Zb)))
    scale = float(max(jnp.max(jnp.abs(qb['h'][-1][0])), 1e-300))
    out['fit_ext_equals_sep_hfit'] = dict(max_weight_delta=dq, max_code_delta=dz,
                                          relative=dq / scale, steps=steps, tolerance=1e-12)
    assert dq / scale < 1e-12 and dz < 1e-12, (dq, dz)
    print('GATE4 fit_ext == sep_hfit.fit', dq, dz, flush=True)

    # ---- the solve-aware terms actually move something ----------------------
    qc, Zc, ic, _ = T.fit_ext(jax.random.PRNGKey(11), A_tr, un2, fl2, K, R, steps=steps,
                              lr=1e-3, hidden=32, layers=2, batch=bs, wp=wp, prev=prev_i,
                              nbr=nbr_i, nu=tr['nu'], beta_w=1., res_batch=4, L=L, dt=dt,
                              log_every=10 ** 9, tag='w')
    moved = float(max(jnp.max(jnp.abs(x[0] - y[0])) for x, y in zip(qa['h'], qc['h'])))
    out['weak_term_changes_the_fit'] = dict(max_weight_delta=moved,
                                            parts=ic['final_parts'])
    assert moved > 0, 'the weak term had no effect at beta_w = 1'
    out['seconds'] = time.perf_counter() - t0
    print('GATE5 weak term moves the fit', moved, flush=True)

    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2, default=float) + '\n')
    print('SMOKE TRAIN OK', round(out['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
