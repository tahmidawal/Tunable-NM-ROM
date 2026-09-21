"""Sub-minute local smoke: WHY did the auto-decoder bank trainer (hires-heat train.py @4fb12a6d) diverge after its
exact coefficient refits? Same objective/step/refit code as hires-heat train_bank, shrunk (d=3, 12 intervals).

Arms (identical data, init, key stream; only the refit handling differs):
  orig      refit codes + zero code Adam moments, shared Adam step count kept   (= hires-heat code)
  resetcnt  refit codes + code optimizer fully re-initialised (own Adam, count=0 -> bias correction restarts)
  keepmom   refit codes, Adam moments left untouched
  norefit   no refit
Records the FULL-BATCH auto-decoder loss (all rows, all points, current codes) every `every` steps and the exact
training projection floor of the current bank, so loss spikes are attributable to the refit handling alone.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hires-heat'))
import core as C
import train as T

cfg = dict(d=3, family='h3d', train_intervals=12, times=[0., .1, .2, .3, .4, .5], diffusivity=.02, train_seed=921000,
           train_count=48, model_seed=921002, fourier_features=16, fourier_scale=1.5, bank_width=64, bank_rank=48,
           bank_batch_states=32, bank_batch_points=512, bank_learning_rate=1e-3, bank_tail_weight=.1, bank_gradient_clip=1.,
           steps=3000, refit_every=1000, every=50, refit_rcond=1e-4)


def run(arm):
    d, n = cfg['d'], cfg['train_intervals']; x = jnp.asarray(C.coords(n, d))
    _, u = T.fields(cfg, cfg['train_seed'], cfg['train_count'], n); u = u.reshape(-1, u.shape[-1]); data = jnp.asarray(u)
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(cfg['model_seed']), 3)
    p = dict(freq=jax.random.normal(k0, (d, cfg['fourier_features']), dtype=jnp.float64) * cfg['fourier_scale'],
             net=T.mlp_init(k1, [2 * cfg['fourier_features'], cfg['bank_width'], cfg['bank_width'], cfg['bank_rank']]),
             scale=jnp.asarray(float(np.sqrt(np.mean(u * u)))))
    eta = .1 * jax.random.normal(k2, (len(u), cfg['bank_rank']), dtype=jnp.float64)
    norms = jnp.maximum(jnp.mean(data * data, axis=1), 1e-20)
    sched = optax.cosine_decay_schedule(cfg['bank_learning_rate'], cfg['steps'], alpha=.03)
    split = arm == 'resetcnt'
    if split:   # separate optimisers so the code optimiser can be fully re-initialised (count too)
        opt_p, opt_z = optax.adam(sched), optax.adam(sched); state = (opt_p.init(p), opt_z.init(eta))
    else:
        opt = optax.adam(sched); state = opt.init((p, eta))
    b, m = cfg['bank_batch_states'], cfg['bank_batch_points']

    def objective(pz, rows, cols):
        p, z = pz; g = C.mlp_features(p, x[cols]); pred = z[rows] @ g.T
        per = jnp.mean((pred - data[rows[:, None], cols[None, :]]) ** 2 / norms[rows, None], axis=1)
        loss = jnp.mean(per) + cfg['bank_tail_weight'] * jnp.mean(per ** 2)
        gn = g / jnp.maximum(p['scale'], 1e-20); gram = gn.T @ gn / len(cols)
        return loss + 1e-6 * jnp.mean((gram - jnp.eye(g.shape[1])) ** 2), loss

    @jax.jit
    def step(pz, state, key):
        a, bk = jax.random.split(key)
        rows = jax.random.randint(a, (b,), 0, data.shape[0]); cols = jax.random.randint(bk, (m,), 0, x.shape[0])
        (_, loss), grads = jax.value_and_grad(objective, has_aux=True)(pz, rows, cols)
        grads[0]['freq'] = jnp.zeros_like(grads[0]['freq']); grads[0]['scale'] = jnp.zeros_like(grads[0]['scale'])
        f = jnp.minimum(1., cfg['bank_gradient_clip'] / jnp.maximum(optax.global_norm(grads), 1e-30))
        grads = jax.tree_util.tree_map(lambda v: v * f, grads)
        if split:
            up, sp = opt_p.update(grads[0], state[0], pz[0]); uz, sz = opt_z.update(grads[1], state[1], pz[1])
            return (optax.apply_updates(pz[0], up), optax.apply_updates(pz[1], uz)), (sp, sz), loss
        updates, state = opt.update(grads, state, pz)
        return optax.apply_updates(pz, updates), state, loss

    full = jax.jit(lambda pz: jnp.mean(jnp.mean((pz[1] @ C.mlp_features(pz[0], x).T - data) ** 2, axis=1) / norms))
    raw = jax.jit(lambda p: C.mlp_features(p, x))
    key = jax.random.PRNGKey(cfg['model_seed'] + 1); pz = (p, eta); curve = []; refits = []
    for it in range(cfg['steps']):
        key, sub = jax.random.split(key); pz, state, _ = step(pz, state, sub)
        if arm != 'norefit' and (it + 1) % cfg['refit_every'] == 0 and it + 1 < cfg['steps']:
            g = np.asarray(raw(pz[0])); uu, ss, vv = np.linalg.svd(g, full_matrices=False); keep = ss > cfg['refit_rcond'] * ss[0]
            fitted = ((u @ uu[:, keep]) / ss[keep]) @ vv[keep]
            before = float(full(pz)); pz = (pz[0], jnp.asarray(fitted)); after = float(full(pz))
            refits.append(dict(step=it + 1, loss_before=before, loss_after_refit=after, kept=int(keep.sum()), cond=float(ss[0] / ss[-1]),
                               max_abs_coef=float(np.abs(fitted).max())))
            if arm == 'orig':
                adam = state[0]
                state = (adam._replace(mu=(adam.mu[0], jnp.zeros_like(pz[1])), nu=(adam.nu[0], jnp.zeros_like(pz[1]))), *state[1:])
            elif arm == 'resetcnt':
                state = (state[0], opt_z.init(pz[1]))
        if (it + 1) % cfg['every'] == 0:
            q, _ = np.linalg.qr(np.asarray(raw(pz[0]))); nrm = np.sum(u * u, 1)
            floor = float(np.max(np.sqrt(np.maximum(nrm - np.sum((u @ q) ** 2, 1), 0) / nrm)))
            curve.append(dict(step=it + 1, full_loss=float(full(pz)), train_floor_worst=floor))
    return dict(curve=curve, refits=refits)


if __name__ == '__main__':
    assert jax.config.jax_enable_x64
    tag = ''
    for kv in sys.argv[1:]:   # overrides, e.g. bank_batch_states=6
        k, v = kv.split('='); cfg[k] = type(cfg[k])(v); tag += '_' + kv.replace('=', '')
    t0 = time.time(); out = dict(config=cfg, backend=jax.default_backend(), arms={})
    for arm in ('orig', 'resetcnt', 'keepmom', 'norefit'):
        out['arms'][arm] = r = run(arm)
        c = {e['step']: e for e in r['curve']}
        for rf in r['refits']:
            s = rf['step']; post = [c[s + k]['full_loss'] for k in (50, 100, 200, 400) if s + k in c]
            rf['full_loss_after_50_100_200_400'] = post
        print(arm, 'refits', json.dumps(r['refits']), flush=True)
        print(arm, 'final full_loss %.3e  final floor %.4f  best floor %.4f' % (r['curve'][-1]['full_loss'], r['curve'][-1]['train_floor_worst'],
              min(e['train_floor_worst'] for e in r['curve'])), flush=True)
    out['seconds'] = time.time() - t0
    Path(__file__).with_name(f'refit_smoke{tag}.json').write_text(json.dumps(out, indent=1))
    print('seconds', out['seconds'])
