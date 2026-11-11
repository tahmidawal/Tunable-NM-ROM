"""Offline training of the neural-field baselines (arm C auto-encoder, arm L decoder + hyper-network).

  train.py --data DIR --out DIR --configs C16-w128,C16-w256 [--steps 120000] [--budget-hours 3.5]

The training snapshots (trajectories 0..4351 of the pool, 26 time levels) are held on the GPU; each step draws
`--batch` states and `--points` interior nodes per state. Configurations are trained one after another, each to its
full step budget (Adam, warm-up + cosine decay); a run interrupted by the wall budget is resumable from last.pkl.
Validation (trajectories 4352..4415, levels 0,5,..,25, all interior nodes) is the monitoring score only.
"""
import argparse, json, os, pickle, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import nf

NTRAIN, NVAL_TRAJ, LEVELS = 4352, 64, 26
L = nf.L


def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return dict(m=z, v=z, t=jnp.zeros((), jnp.int32))


def adam_update(p, g, s, lr, b1=.9, b2=.999, eps=1e-8):
    t = s['t'] + 1
    m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, s['m'], g)
    v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, s['v'], g)
    c1, c2 = 1 - b1 ** t, 1 - b2 ** t
    p = jax.tree_util.tree_map(lambda a, mm, vv: a - lr * (mm / c1) / (jnp.sqrt(vv / c2) + eps), p, m, v)
    return p, dict(m=m, v=v, t=t)


def schedule(step, steps, lr, warm=1000, floor=1e-5):
    w = jnp.minimum(1., (step + 1) / warm)
    c = floor + .5 * (lr - floor) * (1 + jnp.cos(jnp.pi * jnp.clip((step - warm) / (steps - warm), 0., 1.)))
    return w * c


def predict(cfg, p, xy, coarse, t, mu):
    """xy [B,P,2]; coarse [B,65,65]; t [B]; mu [B,5] -> u [B,P]."""
    if cfg['arm'] == 'C':
        z = nf.encode_c(p, coarse)
        return jax.vmap(lambda x, zz: nf.decode_c(p, x, zz))(xy, z)
    z = nf.hyper_l(p, t, mu)
    return jax.vmap(lambda x, zz: nf.decode_l(p, x, zz))(xy, z)


def make_chunk(cfg, steps, batch, points, chunk, ntrain):
    inner, _ = nf.interior_nodes()
    inner = jnp.asarray(inner, jnp.int32)
    cidx = jnp.asarray((np.arange(0, L + 1, 4)[:, None] * (L + 1) + np.arange(0, L + 1, 4)[None, :]).reshape(-1), jnp.int32)

    def loss(p, data, mu, wgt, key):
        k1, k2, k3 = jax.random.split(key, 3)
        tr = jax.random.randint(k1, (batch,), 0, ntrain)
        lv = jax.random.randint(k2, (batch,), 0, LEVELS)
        node = inner[jax.random.randint(k3, (batch, points), 0, inner.shape[0])]
        target = data[tr[:, None], lv[:, None], node]
        coarse = data[tr[:, None], lv[:, None], cidx[None, :]].reshape(batch, L // 4 + 1, L // 4 + 1)
        pred = predict(cfg, p, nf.node_xy(node).astype(jnp.float32), coarse, lv.astype(jnp.float32) * .01, mu[tr])
        return jnp.mean(jnp.mean((pred - target) ** 2, axis=1) * wgt[tr])

    @jax.jit
    def run(p, opt, key, step0, data, mu, wgt):
        def body(c, i):
            p, opt, key = c
            key, sub = jax.random.split(key)
            l, g = jax.value_and_grad(loss)(p, data, mu, wgt, sub)
            p, opt = adam_update(p, g, opt, schedule(step0 + i, steps, cfg['lr']))
            return (p, opt, key), l
        (p, opt, key), ls = jax.lax.scan(body, (p, opt, key), jnp.arange(chunk))
        return p, opt, key, jnp.mean(ls)
    return run


def make_validate(cfg):
    inner, xy = nf.interior_nodes()
    inner = jnp.asarray(inner, jnp.int32); xy = jnp.asarray(xy, jnp.float32)
    cidx = jnp.asarray((np.arange(0, L + 1, 4)[:, None] * (L + 1) + np.arange(0, L + 1, 4)[None, :]).reshape(-1), jnp.int32)

    @jax.jit
    def err(p, states, t, mu):
        coarse = states[:, cidx].reshape(-1, L // 4 + 1, L // 4 + 1)
        pred = predict(cfg, p, jnp.broadcast_to(xy, (states.shape[0],) + xy.shape), coarse, t, mu)
        return jnp.sqrt(jnp.sum((pred - states[:, inner]) ** 2, axis=1))

    def validate(p, vdata, vmu, vn0, bs=4):
        """vdata [T, 6, N] (levels 0,5,..,25) -> relative errors [T, 6] in the metric's normalisation."""
        T = vdata.shape[0]
        out = np.zeros((T, 6))
        tl = jnp.asarray(np.arange(6) * .05, jnp.float32)
        for a in range(0, T):
            out[a] = np.asarray(err(p, vdata[a], tl, jnp.broadcast_to(vmu[a], (6, 5)))) / vn0[a]
        return out
    return validate


def to_np(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--data', required=True); a.add_argument('--out', required=True)
    a.add_argument('--configs', required=True)
    a.add_argument('--steps', type=int, default=120000)
    a.add_argument('--batch', type=int, default=64); a.add_argument('--points', type=int, default=512)
    a.add_argument('--chunk', type=int, default=500); a.add_argument('--val-every', type=int, default=10000)
    a.add_argument('--budget-hours', type=float, default=1e9)
    a.add_argument('--ntrain', type=int, default=NTRAIN); a.add_argument('--nval', type=int, default=NVAL_TRAJ)
    a.add_argument('--allow-cpu', action='store_true')
    a = a.parse_args()
    t_start = time.time()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    assert a.allow_cpu or jax.default_backend() == 'gpu'
    data_dir, out = Path(a.data), Path(a.out)
    meta = np.load(data_dir / 'pool26.meta.npz')
    params, n0 = meta['params'], meta['n0']
    mm = np.load(data_dir / 'pool26.fields.npy', mmap_mode='r')
    assert mm.shape[1:] == (LEVELS, (L + 1) ** 2) and len(mm) >= a.ntrain + a.nval, mm.shape
    t0 = time.time()
    data = jnp.asarray(np.ascontiguousarray(mm[:a.ntrain]))
    vdata = jnp.asarray(np.ascontiguousarray(mm[a.ntrain:a.ntrain + a.nval, ::5]))
    print(f'LOADED train {data.shape} val {vdata.shape} in {time.time() - t0:.0f}s backend={jax.default_backend()}', flush=True)
    mu = jnp.asarray(params[:a.ntrain], jnp.float32)
    wgt = jnp.asarray((L + 1) ** 2 / n0[:a.ntrain] ** 2, jnp.float32)      # 1 / mean square of the initial field
    vmu = jnp.asarray(params[a.ntrain:a.ntrain + a.nval], jnp.float32); vn0 = n0[a.ntrain:a.ntrain + a.nval]

    for name in a.configs.split(','):
        cfg = dict(nf.CONFIGS[name]); d = out / name; d.mkdir(parents=True, exist_ok=True)
        if (d / 'done.json').exists():
            print('SKIP', name, flush=True); continue
        run = make_chunk(cfg, a.steps, a.batch, a.points, a.chunk, a.ntrain)
        validate = make_validate(cfg)
        if (d / 'last.pkl').exists():
            st = pickle.loads((d / 'last.pkl').read_bytes())
            p, opt, key, step, hist, best, secs = (jax.tree_util.tree_map(jnp.asarray, st['params']), jax.tree_util.tree_map(jnp.asarray, st['opt']),
                                                   jnp.asarray(st['key']), st['step'], st['history'], st['best'], st['seconds'])
            print('RESUME', name, step, flush=True)
        else:
            p = nf.init(jax.random.PRNGKey(0), cfg); opt = adam_init(p); key = jax.random.PRNGKey(1)
            step, hist, best, secs = 0, [], np.inf, 0.
        print('CONFIG', name, json.dumps(cfg), 'parameters', nf.count(p), flush=True)
        tl = []
        stopped = False
        while step < a.steps:
            t0 = time.time()
            p, opt, key, l = run(p, opt, key, jnp.int32(step), data, mu, wgt)
            l = float(l); step += a.chunk; dt = time.time() - t0; secs += dt; tl.append(l)
            assert np.isfinite(l), (name, step, l)
            if step % a.val_every == 0 or step >= a.steps:
                e = validate(p, vdata, vmu, vn0)
                rec = dict(step=step, train_loss=float(np.mean(tl)), train_rel=float(np.sqrt(np.mean(tl))),
                           val_mean=float(e.mean()), val_max=float(e.max()), val_t0_mean=float(e[:, 0].mean()),
                           val_evolved_mean=float(e[:, 1:].mean()), val_evolved_worst=float(e[:, 1:].max()),
                           train_seconds=secs, steps_per_second=a.chunk / dt)
                tl = []; hist.append(rec)
                print('VAL', name, json.dumps(rec), flush=True)
                state = dict(params=to_np(p), cfg=cfg, name=name, step=step)
                if rec['val_mean'] < best:
                    best = rec['val_mean']; (d / 'best.pkl').write_bytes(pickle.dumps(state))
                (d / 'last.pkl.tmp').write_bytes(pickle.dumps(dict(state, opt=to_np(opt), key=np.asarray(key), history=hist, best=best, seconds=secs)))
                os.replace(d / 'last.pkl.tmp', d / 'last.pkl')
                (d / 'history.json').write_text(json.dumps(hist, indent=1) + '\n')
            if (time.time() - t_start) / 3600 > a.budget_hours and step < a.steps:
                stopped = True; break
        if stopped:
            print('BUDGET-STOP', name, step, flush=True); break
        near = [h for h in hist if h['step'] >= .9 * a.steps - 1]
        gain = 1 - hist[-1]['val_mean'] / near[0]['val_mean'] if len(near) > 1 else None
        done = dict(name=name, cfg=cfg, parameters=nf.count(p), steps=a.steps, batch=a.batch, points=a.points,
                    train_states=a.ntrain * LEVELS, val_trajectories=a.nval,
                    final=hist[-1], best_val_mean=best, train_seconds=secs,
                    improvement_over_last_tenth=gain, at_plateau=(gain is not None and gain <= .02),
                    gpu=os.environ.get('NF_GPU_NAME'), job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                    backend=jax.default_backend(), jax=jax.__version__)
        (d / 'done.json').write_text(json.dumps(done, indent=1) + '\n')
        print('DONE', name, json.dumps(done), flush=True)


if __name__ == '__main__':
    main()
