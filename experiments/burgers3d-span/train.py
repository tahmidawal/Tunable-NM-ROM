"""burgers3d-span training job (DESIGN.md section 3): data at 33/65/129 nodes from the TRAINING seed only, a code-free
(variable-projection) coordinate bank trained on all three meshes at once, the one-off ordering rotation, and the heads.

Bank loss (heat3d-bank train_vp.py @55165375, generalised to three point sets): for every group g (mesh),
    e_j = ||(I - Q_g Q_g^T) y_j||^2 / ||y_j||^2,  Q_g from a thin QR of G at the group's points,
    mean_g = sum_k w_k e(v_k) over the top POD modes v_k of the normalised group snapshots,
    objective = log(mean over groups of mean_g) + tail_weight * log(power-mean of e over a minibatch from every group)
                + white_weight * whitening(G at the 65-node grid).
Group 33: every interior node of the 33-node grid; group 65: every interior node of the 65-node grid; group 129: a
fixed random subset of the 129-node interior grid (same size as group 65). Checkpoint selection: lowest worst
bank-validation projection error over the three groups (bank-validation seed only).

Ordering (paper Appendix A.1): QR of the bank on the 65-node grid in the h^3-weighted metric, G = Q_G R_G; rows
R_G a_i / ||u_i|| for every training snapshot i of every group (a_i its least-squares bank coefficients on its own
points); SVD -> V; T = R_G^{-1} V. G_hat = G T. Heads are trained on the ordered coefficients c_i = V^T R_G a_i.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

import common as C


def mlp_init(key, sizes):
    out = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        out.append((jax.random.normal(sub, (a, b), dtype=jnp.float64) * jnp.sqrt(2 / a), jnp.zeros(b)))
    return out


def save(path, value):
    Path(path).write_bytes(pickle.dumps(jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, value)))


def group_points(n, cfg):
    """(interior-node indices into the ni^3 grid, coordinates) of a group."""
    x = C.interior_coords(n)
    cap = cfg.get('group_points_cap')
    if cap and len(x) > cap:
        idx = np.sort(np.random.default_rng(cfg['subset_seed']).choice(len(x), size=cap, replace=False))
    else:
        idx = np.arange(len(x))
    return idx, x[idx]


def generate(cfg, tab, rows, n, steps, idx, log, full_steps=None):
    """Snapshots (len(rows)*len(steps), len(idx)) at the group's points (and optionally full fields)."""
    fom = C.make_fom(n, C.DT, cfg['data_ntol'], cfg['data_ltol'], save_steps=steps)
    t0 = time.perf_counter()
    out, full, worst = [], [], 0.
    idx_j = jnp.asarray(idx)
    for r in rows:
        u0 = jnp.asarray(C.initial_interior(n, tab, r))
        f, it, rn = fom(u0, float(tab['nu'][r]))
        worst = max(worst, float(jnp.max(rn)))
        assert int(jnp.max(it)) < C.MAX_NEWTON and np.isfinite(worst), (n, r, int(jnp.max(it)), worst)
        out.append(np.asarray(f[:, idx_j]))
        if full_steps is not None:
            full.append(np.asarray(f[jnp.asarray(full_steps)]))
    log(f'data n={n} rows={len(rows)} steps={len(steps)} points={len(idx)} worst residual {worst:.2e} '
        f'{time.perf_counter() - t0:.1f}s')
    return np.concatenate(out), (np.stack(full) if full_steps is not None else None), worst


def projection_errors(g, y):
    q, _ = jnp.linalg.qr(g, mode='reduced')
    r = y - q @ (q.T @ y)
    return jnp.sum(r * r, axis=0)


def train_bank(groups, cfg, log):
    """groups: list of dict(name, x (P,3), u (S,P) training, v (V,P) validation)."""
    begin = time.perf_counter()
    k0, k1 = jax.random.split(jax.random.PRNGKey(cfg['model_seed']))
    F = cfg['fourier_features']
    p = dict(freq=jax.random.normal(k0, (3, F), dtype=jnp.float64) * cfg['fourier_scale'],
             net=mlp_init(k1, [2 * F] + [cfg['bank_width']] * cfg['bank_depth'] + [cfg['bank_rank']]))
    allu = np.concatenate([g['u'].ravel() for g in groups])
    p['scale'] = jnp.asarray(float(np.sqrt(np.mean(allu ** 2))))
    del allu
    G = []
    for g in groups:
        un = g['u'] / np.linalg.norm(g['u'], axis=1, keepdims=True)
        S = un.shape[0]
        unj = jnp.asarray(un)
        lam, w = jnp.linalg.eigh(jax.jit(lambda a: a @ a.T)(unj))
        lam, w = lam[::-1], w[:, ::-1]
        K = min(cfg['modes'], S)
        modes = (unj.T @ w[:, :K]) / jnp.sqrt(jnp.maximum(lam[:K], 1e-300))
        weights = jnp.maximum(lam[:K], 0.) / S
        dropped = float(jnp.sum(jnp.maximum(lam[K:], 0.)) / S)
        vn = jnp.asarray((g['v'] / np.linalg.norm(g['v'], axis=1, keepdims=True)).T)
        pod = {}
        for r in cfg['pod_reference_ranks']:
            pod[str(r)] = float(jnp.sqrt(jnp.max(jnp.maximum(1 - jnp.sum((modes[:, :r].T @ vn) ** 2, axis=0), 0.))))
        log(f"group {g['name']}: S={S} P={un.shape[1]} POD modes {K} dropped tail mean {dropped:.3e} "
            f"validation POD floor (worst) by rank {pod}")
        G.append(dict(name=g['name'], x=jnp.asarray(g['x']), un=unj, modes=modes, weights=weights, vn=vn,
                      pod=pod, dropped=dropped))
    opt = optax.chain(optax.clip_by_global_norm(cfg['bank_gradient_clip']),
                      optax.adam(optax.warmup_cosine_decay_schedule(0., cfg['bank_learning_rate'], cfg['warmup'],
                                                                    cfg['bank_steps'],
                                                                    cfg['bank_learning_rate'] * .01)))
    state = opt.init(p['net'])
    pw, tw, ww, B = cfg['tail_power'], cfg['tail_weight'], cfg['white_weight'], cfg['bank_batch_states']
    white_group = [g['name'] for g in G].index(cfg['white_group'])

    def objective(net, p, xs, modes, weights, batches):
        means, tails = [], []
        white = 0.
        for gi, (x, md, wt, bt) in enumerate(zip(xs, modes, weights, batches)):
            g = C.features(dict(p, net=net), x)
            e = projection_errors(g, jnp.concatenate((md, bt), axis=1))
            K = md.shape[1]
            means.append(jnp.sum(wt * e[:K]))
            tails.append(e[K:])
            if gi == white_group:
                a = g.T @ g
                dg = jnp.sqrt(jnp.diag(a))
                white = jnp.mean((a / dg[:, None] / dg[None, :] - jnp.eye(a.shape[0])) ** 2)
        mean = sum(means) / len(means)
        tail = jnp.mean(jnp.concatenate(tails) ** pw) ** (1 / pw)
        return jnp.log(mean) + tw * jnp.log(tail) + ww * white, (mean, tail, white)

    @jax.jit
    def step(net, state, key, p, xs, modes, weights, uns):
        keys = jax.random.split(key, len(uns))
        batches = [u[jax.random.randint(k, (B,), 0, u.shape[0])].T for k, u in zip(keys, uns)]
        (val, aux), grad = jax.value_and_grad(objective, has_aux=True)(net, p, xs, modes, weights, batches)
        upd, state = opt.update(grad, state, net)
        return optax.apply_updates(net, upd), state, val, aux

    @jax.jit
    def validate(net, p, xs, vns):
        worst, rms = [], []
        for x, vn in zip(xs, vns):
            e = projection_errors(C.features(dict(p, net=net), x), vn)
            worst.append(jnp.sqrt(jnp.max(jnp.maximum(e, 0.))))
            rms.append(jnp.sqrt(jnp.mean(jnp.maximum(e, 0.))))
        return jnp.stack(worst), jnp.stack(rms)

    xs = [g['x'] for g in G]
    mods = [g['modes'] for g in G]
    wts = [g['weights'] for g in G]
    uns = [g['un'] for g in G]
    vns = [g['vn'] for g in G]
    key = jax.random.PRNGKey(cfg['model_seed'] + 1)
    net = p['net']
    best = (np.inf, None, None)
    log_rows = []
    for it in range(cfg['bank_steps']):
        key, sub = jax.random.split(key)
        net, state, val, aux = step(net, state, sub, p, xs, mods, wts, uns)
        if (it + 1) % cfg['checkpoint_every'] == 0 or it + 1 == cfg['bank_steps'] or it + 1 == 50:
            worst, rms = validate(net, p, xs, vns)
            worst, rms = np.asarray(worst), np.asarray(rms)
            rec = dict(step=it + 1, objective=float(val), train_mean_sq=float(aux[0]), train_tail=float(aux[1]),
                       whitening=float(aux[2]), validation_worst={g['name']: float(w) for g, w in zip(G, worst)},
                       validation_rms={g['name']: float(w) for g, w in zip(G, rms)},
                       seconds=time.perf_counter() - begin)
            log_rows.append(rec)
            log('BANK ' + json.dumps(rec))
            assert np.isfinite(float(val)) and np.all(np.isfinite(worst)), rec
            if float(worst.max()) < best[0]:
                best = (float(worst.max()), jax.tree_util.tree_map(lambda t: t, net), it + 1)
    params = dict(p, net=best[1])
    info = dict(selected_step=best[2], validation_worst_over_groups=best[0], seconds=time.perf_counter() - begin,
                pod_validation_floor_worst={g['name']: g['pod'] for g in G},
                pod_dropped_tail_mean={g['name']: g['dropped'] for g in G}, log=log_rows)
    return params, info


def train_head(target, norm2, perp, vtarget, vnorm2, vperp, k, cfg, log):
    """heat3d-bank train.train_head (auto-decoder head with linear skip, relative loss + tail), validation
    best-found selection (4 nearest training codes as LM starts), all data as explicit jit arguments."""
    target, norm2, perp = map(jnp.asarray, (target, norm2, perp))
    a, b = jax.random.split(jax.random.PRNGKey(cfg['model_seed'] + 100 + k))
    a0, a1 = jax.random.split(a)
    R = target.shape[1]
    p = dict(net=mlp_init(a0, [k] + [cfg['head_width']] * cfg['head_depth'] + [R]),
             skip=jax.random.normal(a1, (k, R), dtype=jnp.float64) * .1)
    z = .1 * jax.random.normal(b, (len(target), k), dtype=jnp.float64)
    opt = optax.adam(optax.cosine_decay_schedule(cfg['head_learning_rate'], cfg['head_steps'], alpha=.03))
    state = opt.init((p, z))
    batch = min(cfg['head_batch_states'], len(target))

    def objective(pz, idx, target, norm2, perp):
        p, z = pz
        e = (jnp.sum((C.head(p, z[idx]) - target[idx]) ** 2, axis=1) + perp[idx]) / norm2[idx]
        return jnp.mean(e) + .1 * jnp.mean(e ** 2)

    @jax.jit
    def step(pz, state, key, target, norm2, perp):
        value, grad = jax.value_and_grad(objective)(pz, jax.random.randint(key, (batch,), 0, len(target)),
                                                    target, norm2, perp)
        update, state = opt.update(grad, state, pz)
        return optax.apply_updates(pz, update), state, value

    fit = C.make_head_fit(budget=200, gtol=1e-8)
    eye = jnp.eye(R)

    @jax.jit
    def validate(p, z, vt, vn, vp):
        lib = C.head(p, z)
        d2 = jnp.sum(lib * lib, axis=1)[None, :] - 2 * vt @ lib.T
        idx = jax.lax.top_k(-d2, 4)[1]

        def one(t, s4):
            out = jax.vmap(lambda s: fit(p, s, eye, t, R))(z[s4])
            return jnp.min(out[1])
        rn = jax.lax.map(lambda a: one(*a), (vt, idx))
        return jnp.sqrt((rn ** 2 + vp) / vn)

    vt, vn, vp = map(jnp.asarray, (vtarget, vnorm2, vperp))
    key = jax.random.PRNGKey(cfg['model_seed'] + 200 + k)
    pz = (p, z)
    best = (np.inf, None, None)
    begin = time.perf_counter()
    rows = []
    for it in range(cfg['head_steps']):
        key, sub = jax.random.split(key)
        pz, state, value = step(pz, state, sub, target, norm2, perp)
        if (it + 1) % cfg['head_checkpoint_every'] == 0 or it + 1 == cfg['head_steps']:
            worst = float(jnp.max(validate(*pz, vt, vn, vp)))
            rec = dict(k=k, step=it + 1, objective=float(value), validation_best_found_worst=worst,
                       seconds=time.perf_counter() - begin)
            rows.append(rec)
            log('HEAD ' + json.dumps(rec))
            if worst < best[0]:
                best = (worst, pz, it + 1)
    p, z = best[1]
    resid = np.asarray(target) - np.asarray(C.head(p, z))
    err = np.sqrt((np.sum(resid ** 2, axis=1) + np.asarray(perp)) / np.asarray(norm2))
    info = dict(k=k, selected_step=best[2], validation_best_found_worst=best[0], training_error_worst=float(err.max()),
                training_error_mean=float(err.mean()), log=rows)
    return dict(params=p, codes=z, info=info)


def floors_full(bank_params, T, n, fields, ladder):
    """Worst / rms projection error of full native-grid fields (S, N) onto G_hat[:, :R'] at mesh n."""
    G = C.bank_at(bank_params, T, C.interior_coords(n))
    Q, _ = jnp.linalg.qr(G, mode='reduced')
    Y = jnp.asarray(fields)
    nrm = jnp.linalg.norm(Y, axis=1)
    X = Y @ Q                                                     # (S, R) in nested QR coordinates
    out = {}
    for Rp in ladder:
        e = jnp.sqrt(jnp.maximum(nrm ** 2 - jnp.sum(X[:, :Rp] ** 2, axis=1), 0.)) / nrm
        out[str(Rp)] = dict(worst=float(jnp.max(e)), rms=float(jnp.sqrt(jnp.mean(e ** 2))))
    return out


def old_bank_floor(path, n, fields):
    import b3d_common as b3
    ck = pickle.loads(Path(path).read_bytes())
    p = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    x = jnp.asarray(C.interior_coords(n))
    G = jnp.concatenate([b3.features(p, x[s:s + 65536]) for s in range(0, len(x), 65536)])
    Q, _ = jnp.linalg.qr(G, mode='reduced')
    Y = jnp.asarray(fields)
    nrm = jnp.linalg.norm(Y, axis=1)
    e = jnp.sqrt(jnp.maximum(nrm ** 2 - jnp.sum((Y @ Q) ** 2, axis=1), 0.)) / nrm
    return dict(rank=int(G.shape[1]), worst=float(jnp.max(e)), rms=float(jnp.sqrt(jnp.mean(e ** 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    smoke = cfg.get('local_smoke', False)
    assert (smoke or jax.default_backend() == 'gpu') and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    assert jax.config.jax_enable_x64
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest (training)', flush=True)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               gpu=jax.devices()[0].device_kind, backend=jax.default_backend())
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    t0 = time.perf_counter()
    tr = C.table(cfg['train_seed'], cfg['train_count'])
    bv = C.table(cfg['bankval_seed'], cfg['bankval_count'])
    rep['train_table_sha256'] = tr['sha256']
    rep['bankval_table_sha256'] = bv['sha256']
    log(f'tables {time.perf_counter() - t0:.1f}s')
    steps = cfg['train_steps']
    groups, full_val = [], {}
    for n in cfg['meshes']:
        idx, x = group_points(n, cfg)
        u, _, wtr = generate(cfg, tr, range(cfg['train_count']), n, steps, idx, log)
        fs = [0, 10, 20, 30, 40, 50]
        nfull = cfg['full_floor_cases']
        v, vfull, wv = generate(cfg, bv, range(cfg['bankval_count']), n, steps, idx, log)
        _, vfull, _ = generate(cfg, bv, range(nfull), n, fs, np.arange(0), log, full_steps=list(range(len(fs))))
        full_val[n] = vfull.reshape(-1, vfull.shape[-1])
        groups.append(dict(name=str(n), n=n, x=x, idx=idx, u=u, v=v))
        rep.setdefault('data', {})[str(n)] = dict(points=len(idx), train_snapshots=len(u), validation_snapshots=len(v),
                                                   worst_train_residual=wtr, worst_validation_residual=wv)
    params, binfo = train_bank(groups, cfg, log)
    rep['bank'] = {k: v for k, v in binfo.items() if k != 'log'}
    rep['bank_log'] = binfo['log']
    # ---- ordering rotation (training data only)
    n_o = cfg['order_mesh']
    x_o = C.interior_coords(n_o)
    h3 = (1.0 / (n_o - 1)) ** 3
    G_o = np.asarray(C.bank_at(params, np.eye(cfg['bank_rank']), x_o)) * np.sqrt(h3)
    _, RG = np.linalg.qr(G_o, mode='reduced')
    del G_o
    rows, coefs, norms, perps, gname = [], [], [], [], []
    for g in groups:
        Gg = np.asarray(C.bank_at(params, np.eye(cfg['bank_rank']), g['x']))
        hg = (1.0 / (g['n'] - 1)) ** 3 * (len(C.interior_coords(g['n'])) / len(g['x']))   # metric weight per point
        Qg, Rg = np.linalg.qr(Gg * np.sqrt(hg), mode='reduced')
        U = g['u'] * np.sqrt(hg)
        a = np.linalg.solve(Rg, Qg.T @ U.T).T                         # (S, R) raw coefficients
        nrm2 = np.sum(U * U, axis=1)
        prj2 = np.sum((Qg.T @ U.T) ** 2, axis=0)
        coefs.append(a)
        norms.append(nrm2)
        perps.append(np.maximum(nrm2 - prj2, 0.))
        gname += [g['name']] * len(a)
        del Gg, Qg, Rg, U
    a = np.concatenate(coefs)
    nrm2 = np.concatenate(norms)
    perp = np.concatenate(perps)
    Arows = (a @ RG.T) / np.sqrt(nrm2)[:, None]
    _, s, Vt = np.linalg.svd(Arows, full_matrices=False)
    V = Vt.T
    T = np.linalg.solve(RG, V)
    ctarget = (a @ RG.T) @ V                                          # ordered coefficients, (S, R)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    rep['ordering'] = dict(mesh=n_o, singular_values=s.tolist(),
                           cumulative_energy={str(r): float(energy[r - 1]) for r in cfg['ladder']},
                           inverse_check=float(np.linalg.norm(V.T @ RG @ T - np.eye(len(T)))))
    log('ORDER ' + json.dumps(rep['ordering']['cumulative_energy']))
    # validation targets (bank-validation seed) in ordered coordinates, for head selection
    vt, vn2, vperp = [], [], []
    for g in groups:
        Gg = np.asarray(C.bank_at(params, T, g['x']))
        hg = (1.0 / (g['n'] - 1)) ** 3 * (len(C.interior_coords(g['n'])) / len(g['x']))
        Qg, Rg = np.linalg.qr(Gg * np.sqrt(hg), mode='reduced')
        U = g['v'] * np.sqrt(hg)
        c = np.linalg.solve(Rg, Qg.T @ U.T).T
        vt.append(c)
        m2 = np.sum(U * U, axis=1)
        vn2.append(m2)
        vperp.append(np.maximum(m2 - np.sum((Qg.T @ U.T) ** 2, axis=0), 0.))
    vt, vn2, vperp = np.concatenate(vt), np.concatenate(vn2), np.concatenate(vperp)
    cmean = ctarget.mean(0)
    spread = {str(r): float(np.sqrt(np.mean(np.sum((ctarget[:, :r] - cmean[:r]) ** 2, axis=1))))
              for r in cfg['ladder']}
    rep['ordering']['coefficient_rms_spread'] = spread
    save(out / 'bank.pkl', dict(params=params, rotation=T, RG=RG, V=V, cfg=cfg, coefficient_rms_spread=spread,
                                info={k: v for k, v in rep.items() if k in ('bank', 'ordering')}))
    # ---- floors of the ordered bank at full native grids (bank-validation cohort, six output times)
    rep['floors_full'] = {str(n): C.clean(floors_full(params, T, n, full_val[n], cfg['ladder'])) for n in cfg['meshes']}
    log('FLOORS ' + json.dumps(rep['floors_full']))
    if cfg.get('old_checkpoint'):
        rep['old_bank_floor_full'] = {str(n): old_bank_floor(cfg['old_checkpoint'], n, full_val[n])
                                      for n in cfg['meshes']}
        log('OLD BANK FLOORS ' + json.dumps(rep['old_bank_floor_full']))
    C.dump(out / 'training.json', C.clean(rep))
    # ---- heads on ordered coefficients of every training snapshot (all meshes)
    rep['heads'] = {}
    for k in cfg['latent_dimensions']:
        h = train_head(ctarget, nrm2, perp, vt, vn2, vperp, k, cfg, log)
        H = np.asarray(C.head(h['params'], h['codes']))
        save(out / f'head_K{k}.pkl', dict(params=h['params'], codes=h['codes'], library_H=H, cfg=cfg, info=h['info']))
        rep['heads'][str(k)] = h['info']
        C.dump(out / 'training.json', C.clean(rep))
    rep['seconds'] = time.perf_counter() - t0
    rep['complete'] = True
    C.dump(out / 'training.json', C.clean(rep))
    (out / 'training.log').write_text('\n'.join(log_lines) + '\n')
    print('TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
