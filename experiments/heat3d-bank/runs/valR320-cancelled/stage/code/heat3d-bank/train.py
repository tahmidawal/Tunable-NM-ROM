"""Two-stage learned bank + nonlinear head, dimension generic. Port of experiments/paper-h3d/train.py
@230c5410 (same objective, refit, validation selection and correction-direction rule); no POD bank.
Writes inputs/<name>/{bank.pkl,head_K<k>.pkl,training.json}. Training and validation draws only."""
import argparse, json, os, pickle, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax
import core as C


def mlp_init(key, sizes):
    out = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key); out.append((jax.random.normal(sub, (a, b), dtype=jnp.float64) * jnp.sqrt(2 / a), jnp.zeros(b)))
    return out


def save(path, value):
    Path(path).write_bytes(pickle.dumps(jax.tree_util.tree_map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, value)))


def fields(cfg, seed, count, n):
    d = cfg['d']; prop = C.make_propagate(d); lam = C.eig_grid(n, d); t = jnp.asarray(cfg['times'])
    draws = C.family(cfg['family'], seed, count)
    return draws, np.stack([np.asarray(prop(C.initial_grid(n, d, p), lam, t, cfg['diffusivity'])).reshape(len(t), -1) for p in draws])


def train_bank(u, val, cfg, log):
    d, n = cfg['d'], cfg['train_intervals']; x = jnp.asarray(C.coords(n, d)); data = jnp.asarray(u)
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(cfg['model_seed']), 3)
    p = dict(freq=jax.random.normal(k0, (d, cfg['fourier_features']), dtype=jnp.float64) * cfg['fourier_scale'],
             net=mlp_init(k1, [2 * cfg['fourier_features'], cfg['bank_width'], cfg['bank_width'], cfg['bank_rank']]),
             scale=jnp.asarray(float(np.sqrt(np.mean(u * u)))))
    eta = .1 * jax.random.normal(k2, (len(u), cfg['bank_rank']), dtype=jnp.float64)
    norms = jnp.maximum(jnp.mean(data * data, axis=1), 1e-20)
    opt = optax.adam(optax.cosine_decay_schedule(cfg['bank_learning_rate'], cfg['bank_steps'], alpha=.03)); state = opt.init((p, eta))
    b, m = min(cfg['bank_batch_states'], len(u)), min(cfg['bank_batch_points'], u.shape[1])
    def objective(pz, values, points, rows, cols, norms):
        p, z = pz; g = C.mlp_features(p, points[cols]); pred = z[rows] @ g.T
        per = jnp.mean((pred - values[rows[:, None], cols[None, :]]) ** 2 / norms[rows, None], axis=1)
        loss = jnp.mean(per) + cfg['bank_tail_weight'] * jnp.mean(per ** 2)
        gn = g / jnp.maximum(p['scale'], 1e-20); gram = gn.T @ gn / len(cols)
        return loss + 1e-6 * jnp.mean((gram - jnp.eye(g.shape[1])) ** 2), loss
    @jax.jit
    def step(pz, state, key, values, points, norms):
        a, bk = jax.random.split(key)
        rows = jax.random.randint(a, (b,), 0, values.shape[0]); cols = jax.random.randint(bk, (m,), 0, points.shape[0])
        (_, loss), grads = jax.value_and_grad(objective, has_aux=True)(pz, values, points, rows, cols, norms)
        grads[0]['freq'] = jnp.zeros_like(grads[0]['freq']); grads[0]['scale'] = jnp.zeros_like(grads[0]['scale'])
        f = jnp.minimum(1., cfg['bank_gradient_clip'] / jnp.maximum(optax.global_norm(grads), 1e-30))
        updates, state = opt.update(jax.tree_util.tree_map(lambda v: v * f, grads), state, pz)
        return optax.apply_updates(pz, updates), state, loss
    raw = lambda p: np.asarray(jax.jit(C.mlp_features)(p, x))
    key = jax.random.PRNGKey(cfg['model_seed'] + 1); pz = (p, eta); best = (np.inf, None, None); begin = time.perf_counter()
    for it in range(cfg['bank_steps']):
        key, sub = jax.random.split(key); pz, state, loss = step(pz, state, sub, data, x, norms)
        if (it + 1) % cfg['refit_every'] == 0:   # exact training-only free coefficients remove auto-decoder code lag
            # Truncated-SVD refit: an ill-conditioned bank (R=256 in 3D, job 4071535) gave huge exact coefficients and the
            # next Adam steps diverged. Directions below refit_rcond * s_max keep zero coefficient.
            uu, ss, vv = np.linalg.svd(raw(pz[0]), full_matrices=False); keep = ss > cfg.get('refit_rcond', 1e-4) * ss[0]
            fitted = ((u @ uu[:, keep]) / ss[keep]) @ vv[keep]; assert np.isfinite(fitted).all()
            print('REFIT', it + 1, 'kept', int(keep.sum()), 'cond', float(ss[0] / ss[-1]), 'max|coef|', float(np.abs(fitted).max()), flush=True)
            pz = (pz[0], jnp.asarray(fitted)); adam = state[0]
            state = (adam._replace(mu=(adam.mu[0], jnp.zeros_like(pz[1])), nu=(adam.nu[0], jnp.zeros_like(pz[1]))), *state[1:])
        if (it + 1) % cfg['checkpoint_every'] == 0 or it + 1 == cfg['bank_steps']:
            qb, _ = np.linalg.qr(raw(pz[0]), mode='reduced'); vn = np.sum(val * val, axis=1)
            worst = float(np.max(np.sqrt(np.maximum(vn - np.sum((val @ qb) ** 2, axis=1), 0.) / vn)))
            rec = dict(step=it + 1, loss=float(loss), validation_projection_worst=worst, seconds=time.perf_counter() - begin)
            log.append(rec); print('BANK', rec, flush=True); assert np.isfinite(float(loss))
            if worst < best[0]: best = (worst, pz[0], it + 1)
    params = best[1]; q, r = np.linalg.qr(raw(params), mode='reduced'); sv = np.linalg.svd(r, compute_uv=False); assert sv[-1] > sv[0] * 1e-12
    target = u @ q; norm2 = np.sum(u * u, axis=1); perp = np.maximum(norm2 - np.sum(target ** 2, axis=1), 0.)
    info = dict(selected_step=best[2], validation_projection_worst=best[0], condition=float(sv[0] / sv[-1]),
                training_projection_worst=float(np.max(np.sqrt(perp / norm2))), seconds=time.perf_counter() - begin)
    return params, np.linalg.solve(r, np.eye(len(r))), q, target, norm2, perp, info


def train_head(target, norm2, perp, vtarget, vnorm2, vperp, k, cfg, log):
    target, norm2, perp = map(jnp.asarray, (target, norm2, perp)); a, b = jax.random.split(jax.random.PRNGKey(cfg['model_seed'] + 100 + k))
    a0, a1 = jax.random.split(a)
    p = dict(net=mlp_init(a0, [k, cfg['head_width'], cfg['head_width'], target.shape[1]]), skip=jax.random.normal(a1, (k, target.shape[1]), dtype=jnp.float64) * .1)
    z = .1 * jax.random.normal(b, (len(target), k), dtype=jnp.float64)
    opt = optax.adam(optax.cosine_decay_schedule(cfg['head_learning_rate'], cfg['head_steps'], alpha=.03)); state = opt.init((p, z))
    batch = min(cfg['head_batch_states'], len(target))
    def objective(pz, idx, target, norm2, perp):   # data are explicit jit ARGUMENTS (captured constants stalled XLA for >20 min in job 4051298)
        p, z = pz; e = (jnp.sum((C.mlp_head(p, z[idx]) - target[idx]) ** 2, axis=1) + perp[idx]) / norm2[idx]
        return jnp.mean(e) + .1 * jnp.mean(e ** 2)
    @jax.jit
    def step(pz, state, key, target, norm2, perp):
        value, grad = jax.value_and_grad(objective)(pz, jax.random.randint(key, (batch,), 0, len(target)), target, norm2, perp)
        update, state = opt.update(grad, state, pz); return optax.apply_updates(pz, update), state, value
    solve = C.lm(C.mlp_head, 400, 1e-8); eye = jnp.eye(target.shape[1]); vt, vn, vp = map(jnp.asarray, (vtarget, vnorm2, vperp))
    @jax.jit
    def validate(p, z, vt, vn, vp):
        lib = C.mlp_head(p, z)
        def one(t, nrm, pp):
            zs, st = jax.vmap(lambda s: solve(p, eye, t, s))(z[jnp.argsort(jnp.sum((lib - t) ** 2, axis=1))[:4]])
            return jnp.sqrt((jnp.min(st[:, 3]) ** 2 * jnp.sum(t * t) + pp) / nrm)
        return jax.vmap(one)(vt, vn, vp)
    key = jax.random.PRNGKey(cfg['model_seed'] + 200 + k); pz = (p, z); best = (np.inf, None, None); begin = time.perf_counter()
    for it in range(cfg['head_steps']):
        key, sub = jax.random.split(key); pz, state, value = step(pz, state, sub, target, norm2, perp)
        if (it + 1) % cfg['checkpoint_every'] == 0 or it + 1 == cfg['head_steps']:
            worst = float(jnp.max(validate(*pz, vt, vn, vp))); rec = dict(k=k, step=it + 1, objective=float(value), validation_best_found_worst=worst, seconds=time.perf_counter() - begin)
            log.append(rec); print('HEAD', rec, flush=True)
            if worst < best[0]: best = (worst, pz, it + 1)
    p, z = best[1]; resid = np.asarray(target) - np.asarray(C.mlp_head(p, z)); _, sv, vtm = np.linalg.svd(resid, full_matrices=False)
    err = np.sqrt((np.sum(resid ** 2, axis=1) + np.asarray(perp)) / np.asarray(norm2))
    info = dict(k=k, selected_step=best[2], validation_best_found_worst=best[0], training_error_worst=float(err.max()),
                training_error_mean=float(err.mean()), correction_singular_values=sv.tolist())
    return dict(params=p, codes=z, directions=vtm.T, info=info)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text())['training']; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu' and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    print('jax_backend=gpu x64=True precision=highest (training)', flush=True)
    n = cfg['train_intervals']; tdraw, u = fields(cfg, cfg['train_seed'], cfg['train_count'], n); vdraw, v = fields(cfg, cfg['validation_seed'], cfg['validation_count'], n)
    u, v = u.reshape(-1, u.shape[-1]), v.reshape(-1, v.shape[-1]); log = dict(bank=[], head=[])
    params, rotation, q, target, norm2, perp, binfo = train_bank(u, v, cfg, log['bank'])
    save(out / 'bank.pkl', dict(params=params, rotation=rotation, info=binfo, cfg=cfg))
    vtarget = v @ q; vnorm2 = np.sum(v * v, axis=1); vperp = np.maximum(vnorm2 - np.sum(vtarget ** 2, axis=1), 0.); heads = {}
    for k in cfg['latent_dimensions']:
        h = train_head(target, norm2, perp, vtarget, vnorm2, vperp, k, cfg, log['head']); heads[k] = h['info']; save(out / f'head_K{k}.pkl', {**h, 'cfg': cfg})
    C.dump(out / 'training.json', dict(config=cfg, bank=binfo, heads=heads, log=log, train_draws_sha=C.sha(tdraw), validation_draws_sha=C.sha(vdraw)))
    print('TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
