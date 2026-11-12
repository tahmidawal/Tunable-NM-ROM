"""Official CoLoRA (julesberman/CoLoRA @ 7367004) on our Burgers 2D pool: data adapter around the released code.

Unchanged and imported from the repository: colora.build.build_colora, colora.adam.adam_opt (and through them
colora.layers / networks / utils). Copied verbatim from burgers.ipynb: normalize, predict, relative_loss_fn, the
vmap structure. Adapted (PROTOCOL.md section 4): the data (mu = cx, cy, w, a, log nu; t = 26 levels; X = 257^2
nodes), the batch iterator (16 snapshots per step, fresh permutation per epoch, gathered on the GPU).

  colora_ours.py train --name K16-w64 --data DIR --out DIR
  colora_ours.py val   --name K16-w64 --data DIR --out DIR           (32 validation trajectories, float64 and float32)
  colora_ours.py test  --names A,B --models DIR --test PREFIX --out DIR   (the single test evaluation + timing)
"""
import argparse, hashlib, json, os, pickle, time
from pathlib import Path
import numpy as np

L, NTRAIN, NGRID = 256, 4352, 32
TEST_SHA = 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'
TOUT = np.arange(1, 6) * .05

CONFIGS = {   # PROTOCOL.md section 4
    'D7':      dict(u_layers=['P'] + ['C'] * 7, width=25, rank=3, full=False, h_width=15, steps=100_000),
    'D7-600k': dict(u_layers=['P'] + ['C'] * 7, width=25, rank=3, full=False, h_width=15, steps=600_000),
    'K16-w25': dict(u_layers=['P'] + ['C'] * 8, width=25, rank=2, full=True, h_width=15, steps=600_000),
    'K16-w64': dict(u_layers=['P'] + ['C'] * 8, width=64, rank=2, full=True, h_width=64, steps=600_000),
    'K16-w128': dict(u_layers=['P'] + ['C'] * 8, width=128, rank=2, full=True, h_width=64, steps=600_000),
    'K16-deep': dict(u_layers=['P'] + ['C'] * 16, width=25, rank=3, full=False, h_width=15, steps=600_000),
}
LR, BATCH = 5e-3, 16


def mu_of(params):
    p = np.asarray(params, np.float64)
    return np.concatenate([p[:, :4], np.log(p[:, 4:5])], 1)


def grid_X():
    x = np.arange(L + 1) / L
    m = np.meshgrid(x, x, indexing='ij')                 # as prepare_coordinate_data: meshgrid ij, flatten
    return np.asarray([g.flatten() for g in m]).T


def setup(cfg, x64=False):
    import jax
    if x64:
        jax.config.update('jax_enable_x64', True)
    from jax import vmap
    from colora.build import build_colora
    key = jax.random.PRNGKey(1)                         # as the notebook
    u_hat_config = {'width': cfg['width'], 'layers': cfg['u_layers']}
    h_config = {'width': cfg['h_width'], 'layers': ['D', 'D', 'D']}
    u_hat_fn, h_fn, theta_init, psi_init = build_colora(u_hat_config, h_config, 2, 6, 1, lora_filter=['alpha'],
                                                        period=[1.0, 1.0], rank=cfg['rank'], full=cfg['full'], key=key)
    h_v_mu_t = vmap(h_fn, in_axes=(None, 0))
    u_hat_v_x = vmap(u_hat_fn, in_axes=(None, None, 0))
    u_hat_v_x_phi = vmap(u_hat_v_x, in_axes=(None, 0, None))

    def predict(psi_theta, mu_t, X_grid):                # notebook, verbatim
        psi, theta = psi_theta
        phis = h_v_mu_t(psi, mu_t)
        pred = u_hat_v_x_phi(theta, phis, X_grid)
        return pred

    def relative_loss_fn(psi_theta, mu_t, sols, X_grid):  # notebook, verbatim
        pred = predict(psi_theta, mu_t, X_grid)
        loss = jax.numpy.linalg.norm(sols - pred, axis=(1, 2)) / jax.numpy.linalg.norm(sols, axis=(1, 2))
        return loss.mean()
    return (psi_init, theta_init), predict, relative_loss_fn, h_v_mu_t


def train(a):
    import jax, jax.numpy as jnp
    from colora.adam import adam_opt
    assert jax.default_backend() == 'gpu'
    cfg = dict(CONFIGS[a.name])
    if a.smoke_steps:                                 # smoke test only (never used for a reported model)
        cfg['steps'] = a.smoke_steps
    out = Path(a.out) / a.name; out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'done.json').exists()
    meta = np.load(Path(a.data) / 'pool26.meta.npz')
    mm = np.load(Path(a.data) / 'pool26.fields.npy', mmap_mode='r')
    nt = mm.shape[1]
    t = np.arange(nt) * .01
    mus = mu_of(meta['params'][:NTRAIN])
    mu_t = np.concatenate([np.repeat(mus, nt, 0), np.tile(t[:, None], (NTRAIN, 1))], 1)   # (M T) order, as the notebook
    mean, std = mu_t.mean(0), mu_t.std(0)
    t0 = time.time()
    Y = jax.device_put(np.ascontiguousarray(mm[:NTRAIN]).reshape(NTRAIN * nt, (L + 1) ** 2, 1))   # (M T) order, float32
    MU = jax.device_put(((mu_t - mean) / std).astype(np.float32))
    X = jax.device_put(grid_X().astype(np.float32))
    print('data on GPU', Y.shape, round(time.time() - t0), 's', flush=True)
    psi_theta, predict, relative_loss_fn, _ = setup(cfg)
    rng = np.random.default_rng(0)
    state = dict(perm=rng.permutation(len(mu_t)), pos=0)

    def args_fn():                                    # ADAPTER: replaces colora.data.Dataset (memory)
        if state['pos'] + BATCH > len(state['perm']):
            state['perm'], state['pos'] = rng.permutation(len(mu_t)), 0
        idx = jnp.asarray(state['perm'][state['pos']:state['pos'] + BATCH]); state['pos'] += BATCH
        return MU[idx], Y[idx], X
    t1 = time.time()
    opt, loss_history = adam_opt(psi_theta, relative_loss_fn, args_fn, steps=cfg['steps'], learning_rate=LR, verbose=a.verbose)
    secs = time.time() - t1
    pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, opt), cfg=cfg, mean=mean, std=std), open(out / 'final.pkl', 'wb'))
    np.save(out / 'loss_history.npy', np.asarray(loss_history))
    n = len(loss_history)
    info = dict(name=a.name, cfg=cfg, steps=cfg['steps'], lr=LR, batch=BATCH, train_seconds=secs,
                loss_last_10pct=float(np.mean(loss_history[-max(n // 10, 1):])), loss_prev_10pct=float(np.mean(loss_history[-2 * max(n // 10, 1):-max(n // 10, 1)])),
                gpu=os.environ.get('NF_GPU_NAME'), job_id=os.environ.get('SLURM_JOB_ID'), jax=jax.__version__)
    (out / 'done.json').write_text(json.dumps(info, indent=1) + '\n')
    print('TRAINED', json.dumps(info), flush=True)


def load(models, name, dtype, x64):
    import jax, jax.numpy as jnp
    st = pickle.load(open(Path(models) / name / 'final.pkl', 'rb'))
    _, predict, _, _ = setup(st['cfg'], x64=x64)
    p = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype), st['params'])
    return p, st, predict


def make_query(predict, st, dtype):
    """mu (5,) -> fields [5, 257, 257]: one hyper-network call per output time, one decoder pass over the grid."""
    import jax, jax.numpy as jnp
    mean, std = jnp.asarray(st['mean'], dtype), jnp.asarray(st['std'], dtype)
    X = jnp.asarray(grid_X(), dtype)
    tout = jnp.asarray(TOUT, dtype)[:, None]

    @jax.jit
    def q(p, mu):
        mu_t = (jnp.concatenate([jnp.broadcast_to(mu, (5, 5)), tout], 1) - mean) / std
        return predict(p, mu_t, X).reshape(5, L + 1, L + 1)
    return q


def metric(f, ref, n0):
    return np.sqrt(((np.asarray(f, np.float64) - ref) ** 2).sum((-1, -2))).max(-1) / n0


def summary(err):
    return dict(worst_percent=float(100 * np.max(err)), median_percent=float(100 * np.median(err)), mean_percent=float(100 * np.mean(err)),
                nonfinite=int((~np.isfinite(err)).sum()))


def time_fn(fn, args_list, warm, reps):
    import jax
    for x in (args_list * (warm // len(args_list) + 1))[:warm]:
        jax.block_until_ready(fn(*x))
    out = np.zeros((reps, len(args_list)))
    for r in range(reps):
        for i, x in enumerate(args_list):
            t0 = time.perf_counter(); jax.block_until_ready(fn(*x)); out[r, i] = time.perf_counter() - t0
    return out * 1e3


def spread(ms):
    ms = np.asarray(ms).reshape(-1)
    return dict(median=float(np.median(ms)), p5=float(np.percentile(ms, 5)), p95=float(np.percentile(ms, 95)), min=float(ms.min()), max=float(ms.max()), count=int(ms.size))


def val(a):
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu'
    meta = np.load(Path(a.data) / 'pool26.meta.npz')
    mm = np.load(Path(a.data) / 'pool26.fields.npy', mmap_mode='r')
    lo, n = NTRAIN, a.cases
    sl = np.asarray(mm[lo:lo + n, ::5], np.float64).reshape(n, 6, L + 1, L + 1)
    ref, n0, mus = sl[:, 1:], meta['n0'][lo:lo + n], mu_of(meta['params'][lo:lo + n])
    rows = []
    for name in a.names.split(','):
        for dt in (jnp.float64, jnp.float32):
            p, st, predict = load(a.models, name, dt, True)
            q = make_query(predict, st, dt)
            f = np.stack([np.asarray(q(p, jnp.asarray(m, dt))) for m in mus])
            err = metric(f, ref, n0)
            ms = time_fn(q, [(p, jnp.asarray(m, dt)) for m in mus[:3]], 2, 2)
            r = dict(name=name, dtype=np.dtype(dt).name, k=latent_size(st['cfg']), **summary(err), rough_ms=float(np.median(ms)),
                     errors=err.tolist(), cfg=st['cfg'])
            rows.append(r)
            print('VAL', json.dumps({k: v for k, v in r.items() if k not in ('errors', 'cfg')}), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'val-{a.names.replace(",", "+")}.json').write_text(json.dumps(dict(cases=n, val_start=lo, rows=rows, gpu=os.environ.get('NF_GPU_NAME'),
                                                                                       job_id=os.environ.get('SLURM_JOB_ID')), indent=1) + '\n')


def latent_size(cfg):
    """Number of alphas: one per CoLoRA ('C') layer, or `rank` per layer when full=True (colora.layers.CoLoRA)."""
    return sum(1 for l in cfg['u_layers'] if l == 'C') * (cfg['rank'] if cfg['full'] else 1)


def test(a):
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = out / 'test-colora.json'
    assert not tag.exists(), 'the test cohort is evaluated once'
    sel_text = Path(a.selection).read_text()
    sel = json.loads(sel_text)['colora']
    meta = np.load(a.test + '.meta.npz')
    params = meta['params']
    assert hashlib.sha256(np.ascontiguousarray(params).tobytes()).hexdigest() == TEST_SHA
    fld = np.load(a.test + '.fields.npy')
    assert fld.shape == (64, 6, L + 1, L + 1) and fld.dtype == np.float64
    u0, ref = fld[:, 0], fld[:, 1:]
    n0 = np.sqrt((u0 ** 2).sum((1, 2)))
    mus = mu_of(params)
    res = []
    for s in sel:
        dt = jnp.float64 if s['dtype'] == 'float64' else jnp.float32
        p, st, predict = load(a.models, s['name'], dt, True)
        q = make_query(predict, st, dt)
        args = [(p, jnp.asarray(m, dt)) for m in mus]
        err = metric(np.stack([np.asarray(q(*x)) for x in args]), ref, n0)
        ms = time_fn(q, args, 20, 5)
        r = dict(selection=s, test=summary(err), errors=err.tolist(), ms_full=spread(ms), timing_warmup=20, timing_repetitions=5)
        res.append(r)
        print('TEST', json.dumps({k: v for k, v in r.items() if k != 'errors'}), flush=True)
    tag.write_text(json.dumps(dict(cohort='test64 = params_draw(20260916, 64)', cohort_sha256=TEST_SHA, metric='max over t=0.05..0.25 of |u-u_ref|_2/|u0|_2',
                                   timing='batch 1, 20 warm-up, 5 x 64; mu on GPU -> five output fields on GPU', selection_sha256=hashlib.sha256(sel_text.encode()).hexdigest(),
                                   gpu=os.environ.get('NF_GPU_NAME'), job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                                   jax=jax.__version__, results=res), indent=1) + '\n')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('cmd', choices=('train', 'val', 'test'))
    a.add_argument('--name'); a.add_argument('--names'); a.add_argument('--data'); a.add_argument('--out'); a.add_argument('--models')
    a.add_argument('--test'); a.add_argument('--selection'); a.add_argument('--cases', type=int, default=NGRID)
    a.add_argument('--verbose', action='store_true'); a.add_argument('--smoke-steps', type=int, default=0)
    a = a.parse_args()
    dict(train=train, val=val, test=test)[a.cmd](a)


if __name__ == '__main__':
    main()
