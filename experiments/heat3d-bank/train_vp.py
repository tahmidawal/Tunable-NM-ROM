"""Code-free (variable-projection) coordinate-bank trainer + nonlinear head. heat3d-bank lane, 2026-09-21.

Why: the auto-decoder bank trainer (hires-heat train.py @4fb12a6d, ported from paper-h3d) learns per-snapshot codes with
Adam and periodically replaces them by exact least-squares coefficients. At each refit it zeroed the code Adam moments but
kept the shared step count, so bias correction was ~1 and every re-sampled code row took steps of ~3-30x the learning rate
(diagnostics/refit_smoke.py: full-batch loss x3 within 50 steps after a refit, absent when the count is reset or the moments
kept). Together with a bank whose condition number grew to 1e8 (kept rank 112 of 256) this diverged (job 4071535) or degraded
the floor (4072491). Here there are no codes at all: for the current bank G the optimal coefficients are eliminated exactly
(Golub-Pereyra variable projection), so the loss is the true projection error of every snapshot onto span(G).

Loss (f64, full training grid, explicit jit arguments only):
  e_j = ||(I - P_G) y_j||^2 for unit-norm targets y_j, P_G via Cholesky of G^T G (+ tiny trace-relative ridge)
  mean term  = sum_k w_k e(v_k) over the top-K POD modes v_k of the normalised training snapshots, w_k = lambda_k / S
               (exactly the mean training error restricted to those modes; the dropped tail is recorded)
  tail term  = (mean_b e_b^p)^(1/p) over a random minibatch of training snapshots (worst-case pressure)
  whitening  = mean((C - I)^2), C the column-cosine matrix of G (span-invariant; conditions the bank)
  objective  = log(mean) + tail_weight * log(tail) + white_weight * whitening
Selection: the checkpoint with the lowest worst-case VALIDATION projection error (validation seed only; never the final cohort).
Head: train.train_head unchanged (hires-heat @4fb12a6d), on exact coefficients in the training-grid orthonormal bank.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax
import core as C
import train as T


def build(cfg):
    d = cfg['d']; F = cfg['fourier_features']; k0, k1 = jax.random.split(jax.random.PRNGKey(cfg['model_seed']))
    sizes = [2 * F] + [cfg['bank_width']] * cfg.get('bank_depth', 2) + [cfg['bank_rank']]
    return dict(freq=jax.random.normal(k0, (d, F), dtype=jnp.float64) * cfg['fourier_scale'], net=T.mlp_init(k1, sizes))


def projection_errors(g, y, ridge):
    """Squared projection residual of each column of y onto span(g) (both f64 device arrays)."""
    a = g.T @ g; a = a + ridge * jnp.trace(a) / a.shape[0] * jnp.eye(a.shape[0])
    c = jax.scipy.linalg.cho_solve((jnp.linalg.cholesky(a), True), g.T @ y); r = y - g @ c
    return jnp.sum(r * r, axis=0), a


def train_bank_vp(u, v, cfg, log):
    d, n = cfg['d'], cfg['train_intervals']; x = jnp.asarray(C.coords(n, d)); begin = time.perf_counter()
    un = jnp.asarray(u / np.linalg.norm(u, axis=1, keepdims=True)); S = un.shape[0]
    lam, w = jnp.linalg.eigh(jax.jit(lambda a: a @ a.T)(un)); lam, w = lam[::-1], w[:, ::-1]
    K = min(cfg['modes'], S); modes = (un.T @ w[:, :K]) / jnp.sqrt(jnp.maximum(lam[:K], 1e-300)); weights = jnp.maximum(lam[:K], 0.) / S
    dropped = float(jnp.sum(jnp.maximum(lam[K:], 0.)) / S)
    vn = jnp.asarray(v / np.linalg.norm(v, axis=1, keepdims=True)).T
    pod = {}
    for r in cfg.get('pod_reference_ranks', [cfg['bank_rank']]):   # optimal R-subspace of the SAME training data: reference only
        pod[str(r)] = float(jnp.sqrt(jnp.max(jnp.maximum(1 - jnp.sum((modes[:, :r].T @ vn) ** 2, axis=0), 0.))))
    print('POD modes', K, 'dropped tail mean', dropped, 'validation POD floor (worst) by rank', pod, 'seconds', time.perf_counter() - begin, flush=True)
    p = build(cfg); p['scale'] = jnp.asarray(float(np.sqrt(np.mean(u * u))))
    opt = optax.chain(optax.clip_by_global_norm(cfg['bank_gradient_clip']),
                      optax.adam(optax.warmup_cosine_decay_schedule(0., cfg['bank_learning_rate'], cfg.get('warmup', 500), cfg['bank_steps'], cfg['bank_learning_rate'] * .01)))
    state = opt.init(p['net']); ridge = cfg.get('ridge', 1e-13); pw = cfg.get('tail_power', 4.); tw = cfg.get('tail_weight', .5); ww = cfg.get('white_weight', 1e-3)
    B = min(cfg['bank_batch_states'], S)

    def objective(net, p, x, modes, weights, batch):
        g = C.mlp_features(dict(p, net=net), x); K = modes.shape[1]
        e, a = projection_errors(g, jnp.concatenate((modes, batch), axis=1), ridge)
        mean = jnp.sum(weights * e[:K]); tail = jnp.mean(e[K:] ** pw) ** (1 / pw)
        dg = jnp.sqrt(jnp.diag(a)); white = jnp.mean((a / dg[:, None] / dg[None, :] - jnp.eye(a.shape[0])) ** 2)
        return jnp.log(mean) + tw * jnp.log(tail) + ww * white, (mean, tail, white)

    @jax.jit
    def step(net, state, key, p, x, modes, weights, un):
        batch = un[jax.random.randint(key, (B,), 0, un.shape[0])].T
        (val, aux), grad = jax.value_and_grad(objective, has_aux=True)(net, p, x, modes, weights, batch)
        upd, state = opt.update(grad, state, net); return optax.apply_updates(net, upd), state, val, aux

    @jax.jit
    def validate(net, p, x, vn):
        g = C.mlp_features(dict(p, net=net), x); e, a = projection_errors(g, vn, 1e-15); s = jnp.linalg.svd(g, compute_uv=False)
        return jnp.sqrt(jnp.max(jnp.maximum(e, 0.))), jnp.sqrt(jnp.mean(jnp.maximum(e, 0.))), s[0] / s[-1]

    key = jax.random.PRNGKey(cfg['model_seed'] + 1); net = p['net']; best = (np.inf, None, None); every = cfg['checkpoint_every']
    for it in range(cfg['bank_steps']):
        key, sub = jax.random.split(key); net, state, val, aux = step(net, state, sub, p, x, modes, weights, un)
        if (it + 1) % every == 0 or it + 1 == cfg['bank_steps']:
            worst, rms, cond = map(float, validate(net, p, x, vn))
            rec = dict(step=it + 1, objective=float(val), train_mean_sq=float(aux[0]), train_tail=float(aux[1]), whitening=float(aux[2]),
                       validation_projection_worst=worst, validation_projection_rms=rms, condition=cond, seconds=time.perf_counter() - begin)
            log.append(rec); print('BANK', rec, flush=True); assert np.isfinite(float(val)) and np.isfinite(worst)
            if worst < best[0]: best = (worst, jax.tree_util.tree_map(lambda t: t, net), it + 1)
    params = dict(p, net=best[1]); g = np.asarray(jax.jit(C.mlp_features)(params, x)); q, r = np.linalg.qr(g, mode='reduced')
    sv = np.linalg.svd(r, compute_uv=False); assert sv[-1] > sv[0] * 1e-12
    target = u @ q; norm2 = np.sum(u * u, axis=1); perp = np.maximum(norm2 - np.sum(target ** 2, axis=1), 0.)
    info = dict(selected_step=best[2], validation_projection_worst=best[0], condition=float(sv[0] / sv[-1]),
                training_projection_worst=float(np.max(np.sqrt(perp / norm2))), pod_validation_floor_worst=pod, pod_modes=K,
                pod_dropped_tail_mean=dropped, seconds=time.perf_counter() - begin)
    return params, np.linalg.solve(r, np.eye(len(r))), q, target, norm2, perp, info


def floor_at(params, rotation, cfg, seed, count, n):
    """Worst/rms projection error of a cohort onto the bank evaluated at another grid (no fitting, f64, GPU)."""
    model = dict(d=cfg['d'], feature_fn=C.mlp_features, bank_params=jax.tree_util.tree_map(jnp.asarray, params), rotation=rotation)
    g = C.bank_at(model, n); lam = C.eig_grid(n, cfg['d']); prop = C.make_propagate(cfg['d']); t = jnp.asarray(cfg['times'])
    draws = C.family(cfg['family'], seed, count); errs = []; chunk = max(4, min(64, int(3e8 // (len(cfg['times']) * g.shape[0]))))
    f = jax.jit(lambda g, y: projection_errors(g, y, 1e-15)[0])
    for s in range(0, count, chunk):
        y = jnp.concatenate([prop(C.initial_grid(n, cfg['d'], dr), lam, t, cfg['diffusivity']).reshape(len(cfg['times']), -1) for dr in draws[s:s + chunk]])
        y = (y / jnp.linalg.norm(y, axis=1, keepdims=True)).T; errs.append(np.asarray(f(g, y)))
    e = np.sqrt(np.maximum(np.concatenate(errs), 0.)).reshape(count, len(cfg['times']))
    return dict(intervals=n, seed=seed, count=count, worst=float(e.max()), worst_t0=float(e[:, 0].max()), rms=float(np.sqrt(np.mean(e ** 2))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text())['training']; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    smoke = cfg.get('local_smoke', False)
    assert (smoke or jax.default_backend() == 'gpu') and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest' and jax.config.jax_enable_x64
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest (varpro training)', flush=True)
    n = cfg['train_intervals']; tdraw, u = T.fields(cfg, cfg['train_seed'], cfg['train_count'], n); vdraw, v = T.fields(cfg, cfg['validation_seed'], cfg['validation_count'], n)
    assert not any(np.array_equal(a, b) for a in tdraw for b in vdraw)
    u, v = u.reshape(-1, u.shape[-1]), v.reshape(-1, v.shape[-1]); log = dict(bank=[], head=[])
    params, rotation, q, target, norm2, perp, binfo = train_bank_vp(u, v, cfg, log['bank'])
    binfo['validation_floor_other_grids'] = [floor_at(params, rotation, cfg, cfg['validation_seed'], cfg['validation_count'], m) for m in cfg.get('floor_grids', [])]
    print('BANK FLOORS', json.dumps(binfo['validation_floor_other_grids']), flush=True)
    T.save(out / 'bank.pkl', dict(params=params, rotation=rotation, info=binfo, cfg=cfg))
    vtarget = v @ q; vnorm2 = np.sum(v * v, axis=1); vperp = np.maximum(vnorm2 - np.sum(vtarget ** 2, axis=1), 0.); heads = {}
    for k in cfg['latent_dimensions']:
        h = T.train_head(target, norm2, perp, vtarget, vnorm2, vperp, k, dict(cfg, checkpoint_every=cfg.get('head_checkpoint_every', cfg['checkpoint_every'])), log['head']); heads[k] = h['info']; T.save(out / f'head_K{k}.pkl', {**h, 'cfg': cfg})
    C.dump(out / 'training.json', dict(config=cfg, bank=binfo, heads=heads, log=log, train_draws_sha=C.sha(tdraw), validation_draws_sha=C.sha(vdraw)))
    print('TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
