"""Online stage of the neural-field baselines and its evaluation.

Residual kinds (arm C, and optionally arm L with z = alpha):
  per backward-Euler step the latent is the Levenberg-Marquardt minimiser of the reference solver's discrete residual
  (burgers_fom.residual: sign-upwind advection, five-point Laplacian, 256 mesh, dt 0.005) at M fixed random interior
  collocation nodes; stencil values come from decoder evaluations at each node and its four neighbours; exact
  forward-mode Jacobian with respect to the latent. No linear bank, no precomputed operators.
Hyper kind (arm L): alpha = hyper(t, mu) at the five output times, one decoder pass over the grid.

  rom.py grid   --models DIR --names A,B --data DIR --out DIR     validation knob grid (first 32 validation trajectories)
  rom.py select --grid DIR --out selection.json                   frozen selection (PROTOCOL.md section 5)
  rom.py test   --selection selection.json --models DIR --test PREFIX --out DIR    the single test evaluation + timing
"""
import argparse, hashlib, json, os, pickle, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import burgers_fom as e          # enables float64
import nf

L, DT, NSTEP = 256, .005, 50
NTRAIN, NGRID = 4352, 32
TEST_SHA = 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'
LAM0, LAM_MIN, LAM_MAX = 1e-6, 1e-9, 1e6


def load_model(models, name, dtype):
    st = pickle.loads((Path(models) / name / 'best.pkl').read_bytes())
    return jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype), st['params']), st['cfg'], st['step']


def collocation(M, seed=7):
    idx = np.random.default_rng(seed).choice((L - 1) ** 2, M, replace=False)
    i, j = idx // (L - 1) + 1, idx % (L - 1) + 1
    ij = np.stack((np.stack((i, j), -1), np.stack((i - 1, j), -1), np.stack((i + 1, j), -1),
                   np.stack((i, j - 1), -1), np.stack((i, j + 1), -1)), axis=1)          # [M, 5, 2]: c, xm, xp, ym, yp
    mask = ((ij >= 1) & (ij <= L - 1)).all(-1)
    return i, j, ij / L, mask


def lm(fun, z, iters):
    """`iters` Levenberg-Marquardt iterations on 0.5 |fun(z)|^2 (Marquardt scaling, accept/reject)."""
    k = z.shape[0]

    def body(s):
        z, lam, it = s
        J, r = jax.jacfwd(lambda q: (fun(q),) * 2, has_aux=True)(z)
        A = J.T @ J
        d = jnp.diagonal(A)
        delta = jnp.linalg.solve(A + jnp.diag(lam * d + 1e-30 * jnp.max(d) + 1e-300), -(J.T @ r))
        cand = z + delta
        rc = fun(cand)
        ok = jnp.isfinite(rc).all() & (jnp.sum(rc * rc) < jnp.sum(r * r))
        return jnp.where(ok, cand, z), jnp.where(ok, jnp.maximum(lam * .1, LAM_MIN), jnp.minimum(lam * 10., LAM_MAX)), it + 1
    z, _, _ = jax.lax.while_loop(lambda s: s[2] < iters, body, (z, jnp.asarray(LAM0, z.dtype), jnp.int32(0)))
    return z


def build(cfg, M, dtype, kind):
    """Returns (latent, full): latent(p,u0,nu,mu,iters,refine)->zs[5,k]; full(...)->fields[5,257,257]."""
    dec = nf.decoder(cfg)
    _, gxy = nf.interior_nodes()
    gxy = jnp.asarray(gxy, dtype)
    tout = jnp.asarray(np.arange(1, 6) * .05, dtype)

    def fields(p, zs):
        return jnp.pad(jax.vmap(lambda z: dec(p, gxy, z))(zs).reshape(5, L - 1, L - 1), ((0, 0), (1, 1), (1, 1)))

    if kind == 'hyper':
        def latent(p, u0, nu, mu, iters, refine):
            return nf.hyper_l(p, tout, jnp.broadcast_to(mu, (5, 5)))
    else:
        ci, cj, xy, mask = collocation(M)
        ci, cj = jnp.asarray(ci), jnp.asarray(cj)
        xyc = jnp.asarray(xy[:, 0], dtype); xy5 = jnp.asarray(xy.reshape(-1, 2), dtype); mask = jnp.asarray(mask, dtype)

        def res(p, z, prev_c, nu):
            v = dec(p, xy5, z).reshape(M, 5) * mask
            c, xm, xp, ym, yp = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4]
            adv = c * L * (jnp.where(c > 0, c - xm, xp - c) + jnp.where(c > 0, c - ym, yp - c))
            lap = L ** 2 * (xm + xp + ym + yp - 4 * c)
            return c - prev_c + DT * (adv - nu * lap)

        def latent(p, u0, nu, mu, iters, refine):
            if cfg['arm'] == 'C':
                z = nf.encode_c(p, u0[::4, ::4][None])[0]
            else:
                z = nf.hyper_l(p, jnp.zeros((), dtype), mu)
            u0c = u0[ci, cj]
            z = lm(lambda q: dec(p, xyc, q) - u0c, z, refine)

            def step(z, _):
                prev_c = dec(p, xyc, z)
                z = lm(lambda q: res(p, q, prev_c, nu), z, iters)
                return z, z
            _, zs = jax.lax.scan(step, z, None, length=NSTEP)
            return zs[9::10]

    def full(p, u0, nu, mu, iters, refine):
        return fields(p, latent(p, u0, nu, mu, iters, refine))
    return latent, full


def metric(f, ref, n0):
    """f, ref [.., 5, 257, 257] float64 -> per-case max over the five output times of |f - ref|_2 / |u0|_2."""
    return np.sqrt(((np.asarray(f, np.float64) - ref) ** 2).sum((-1, -2))).max(-1) / n0


def time_fn(fn, args_list, warm, reps):
    for a in args_list[:warm] if warm <= len(args_list) else (args_list * (warm // len(args_list) + 1))[:warm]:
        jax.block_until_ready(fn(*a))
    out = np.zeros((reps, len(args_list)))
    for r in range(reps):
        for i, a in enumerate(args_list):
            t0 = time.perf_counter(); jax.block_until_ready(fn(*a)); out[r, i] = time.perf_counter() - t0
    return out * 1e3


def spread(ms):
    ms = np.asarray(ms).reshape(-1)
    return dict(median=float(np.median(ms)), p5=float(np.percentile(ms, 5)), p95=float(np.percentile(ms, 95)),
                min=float(ms.min()), max=float(ms.max()), count=int(ms.size))


def summary(err):
    return dict(worst_percent=float(100 * np.nanmax(np.where(np.isfinite(err), err, np.inf))), median_percent=float(100 * np.median(err)),
                mean_percent=float(100 * np.mean(err)), nonfinite=int((~np.isfinite(err)).sum()))


_FN = {}


def run_setting(cfg, p, M, dtype, kind, iters, refine, u0, nu, mu, ref, n0, timing_cases=3, key=None):
    k, W = cfg['k'], cfg['width']
    bs = 32 if kind == 'hyper' else int(2 ** np.clip(np.floor(np.log2(8e9 / (5 * M * (k + 1) * W * 8))), 0, 5))
    bs = min(bs, len(u0))
    ck = (key, M, np.dtype(dtype).name, kind)
    if ck not in _FN:                      # iters / refine are traced, so one compilation serves every setting at this M
        _FN.clear()
        latent, full = build(cfg, M, dtype, kind)
        _FN[ck] = (jax.jit(jax.vmap(full, in_axes=(None, 0, 0, 0, None, None))), jax.jit(full))
    vf, one = _FN[ck]
    it, rf = jnp.int32(iters), jnp.int32(refine)
    out = [np.asarray(vf(p, jnp.asarray(u0[a:a + bs], dtype), jnp.asarray(nu[a:a + bs], dtype), jnp.asarray(mu[a:a + bs], dtype), it, rf))
           for a in range(0, len(u0) - len(u0) % bs, bs)]
    assert len(u0) % bs == 0
    err = metric(np.concatenate(out), ref, n0)
    args = [(p, jnp.asarray(u0[i], dtype), jnp.asarray(nu[i], dtype), jnp.asarray(mu[i], dtype), it, rf) for i in range(min(timing_cases, len(u0)))]
    ms = time_fn(one, args, 2, 2)
    return dict(M=M, iters=iters, refine=refine, dtype=np.dtype(dtype).name, kind=kind, **summary(err),
                rough_ms=float(np.median(ms)), errors=[float(x) for x in err])


def grid(a):
    assert a.allow_cpu or jax.default_backend() == 'gpu'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = np.load(Path(a.data) / 'pool26.meta.npz')
    mm = np.load(Path(a.data) / 'pool26.fields.npy', mmap_mode='r')
    lo = a.val_start; n = a.cases
    sl = np.asarray(mm[lo:lo + n, ::5], np.float64).reshape(n, 6, L + 1, L + 1)
    u0, ref = sl[:, 0], sl[:, 1:]
    params, n0 = meta['params'][lo:lo + n], meta['n0'][lo:lo + n]
    nu = params[:, 4]
    Ms = [int(x) for x in a.M.split(',')]; its = [int(x) for x in a.iters.split(',')]
    for name in a.names.split(','):
        if (out / f'{name}.json').exists():
            print('SKIP', name, flush=True); continue
        p, cfg, step = load_model(a.models, name, jnp.float64)
        rows = []
        t0 = time.time()

        def go(M, iters, refine, dtype=jnp.float64, kind='residual', pp=p):
            r = run_setting(cfg, pp, M, dtype, kind, iters, refine, u0, nu, params, ref, n0, key=name)
            r['name'] = name; rows.append(r)
            print('GRID', name, json.dumps({k: v for k, v in r.items() if k != 'errors'}), round(time.time() - t0), flush=True)
            return r
        if cfg['arm'] == 'L':
            go(0, 0, 0, kind='hyper')
            go(0, 0, 0, dtype=jnp.float32, kind='hyper', pp=jax.tree_util.tree_map(lambda v: v.astype(jnp.float32), p))
        if cfg['arm'] == 'C' or a.residual_l:
            # initial refinement on/off is decided once, at the middle setting, then the rest of the grid uses the better
            mid_M, mid_it = Ms[len(Ms) // 2], its[len(its) // 2]
            r_off, r_on = go(mid_M, mid_it, 0), go(mid_M, mid_it, a.refine_iters)
            refine = a.refine_iters if r_on['worst_percent'] < r_off['worst_percent'] else 0
            for M in [mid_M] + [m for m in Ms if m != mid_M]:
                if cfg['width'] >= 512 and M > 2048:
                    continue
                for iters in its:
                    if (M, iters) != (mid_M, mid_it):
                        go(M, iters, refine)
            # float32 online variant of the most accurate and of the fastest setting under the fast gate
            res = [r for r in rows if r['kind'] == 'residual' and r['nonfinite'] == 0]
            picks = []
            if res:
                picks.append(min(res, key=lambda r: r['worst_percent']))
                ok = [r for r in res if r['worst_percent'] <= a.fast_gate]
                if ok:
                    picks.append(min(ok, key=lambda r: r['rough_ms']))
            p32 = jax.tree_util.tree_map(lambda v: v.astype(jnp.float32), p)
            for r in {(r['M'], r['iters'], r['refine']) for r in picks}:
                go(r[0], r[1], r[2], dtype=jnp.float32, pp=p32)
        (out / f'{name}.json').write_text(json.dumps(dict(name=name, cfg=cfg, checkpoint_step=step, cases=n, val_start=lo, rows=rows,
                                                           gpu=os.environ.get('NF_GPU_NAME'), job_id=os.environ.get('SLURM_JOB_ID'),
                                                           backend=jax.default_backend(), seconds=time.time() - t0), indent=1) + '\n')


def select(a):
    """PROTOCOL.md section 5, mechanically. One accurate and (if any) one fast setting per (arm, k)."""
    rows = []
    for f in sorted(Path(a.grid).glob('*.json')):
        d = json.loads(f.read_text())
        for r in d['rows']:
            rows.append(dict({k: v for k, v in r.items() if k != 'errors'}, arm=d['cfg']['arm'] + ('' if r['kind'] != 'residual' or d['cfg']['arm'] == 'C' else 'R'),
                             k=d['cfg']['k'], checkpoint_step=d['checkpoint_step']))
    sel = []
    for arm in sorted({r['arm'] for r in rows}):
        for k in sorted({r['k'] for r in rows if r['arm'] == arm}):
            c64 = [r for r in rows if r['arm'] == arm and r['k'] == k and r['dtype'] == 'float64' and r['nonfinite'] == 0]
            if not c64:
                continue
            best = min(r['worst_percent'] for r in c64)
            acc = min([r for r in c64 if r['worst_percent'] <= 1.05 * best], key=lambda r: r['rough_ms'])
            chosen = [dict(acc, role='accurate')]
            fast = [r for r in c64 if r['worst_percent'] <= a.fast_gate]
            if fast and arm != 'L':
                f = min(fast, key=lambda r: r['rough_ms'])
                if (f['name'], f['M'], f['iters'], f['refine']) != (acc['name'], acc['M'], acc['iters'], acc['refine']):
                    chosen.append(dict(f, role='fast'))
            for c in chosen:
                sel.append(c)
                tw = [r for r in rows if r['dtype'] == 'float32' and all(r[q] == c[q] for q in ('name', 'M', 'iters', 'refine', 'kind'))]
                if tw and tw[0]['nonfinite'] == 0 and tw[0]['worst_percent'] <= 1.05 * c['worst_percent']:
                    sel.append(dict(tw[0], role=c['role'] + '-float32'))
    Path(a.out).write_text(json.dumps(dict(rule='PROTOCOL.md section 5 (and amendments)', fast_gate_percent=a.fast_gate,
                                           selected=sel, all_validation_rows=rows), indent=1) + '\n')
    for s in sel:
        print('SELECTED', json.dumps(s))


def test(a):
    assert jax.default_backend() == 'gpu' or a.allow_cpu
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'test.json').exists(), 'the test cohort is evaluated once'
    sel_text = Path(a.selection).read_text()
    sel = json.loads(sel_text)['selected']
    meta = np.load(a.test + '.meta.npz')
    params, n0 = meta['params'], meta['n0']
    assert hashlib.sha256(np.ascontiguousarray(params).tobytes()).hexdigest() == TEST_SHA
    assert np.array_equal(params, e.params_draw(20260916, 64))
    fld = np.load(a.test + '.fields.npy')
    assert fld.shape == (64, 6, L + 1, L + 1) and fld.dtype == np.float64
    u0, ref = fld[:, 0], fld[:, 1:]
    n0_meta, n0 = n0, np.sqrt((u0 ** 2).sum((1, 2)))
    print('n0 max relative difference vs stored', float(np.abs(n0 / n0_meta - 1).max()), flush=True)
    res = []
    for s in sel:
        dtype = jnp.float64 if s['dtype'] == 'float64' else jnp.float32
        p, cfg, step = load_model(a.models, s['name'], dtype)
        assert step == s['checkpoint_step']
        latent, full = build(cfg, s['M'], dtype, s['kind'])
        jf, jl = jax.jit(full), jax.jit(latent)
        it, rf = jnp.int32(s['iters']), jnp.int32(s['refine'])
        args = [(p, jnp.asarray(u0[i], dtype), jnp.asarray(params[i, 4], dtype), jnp.asarray(params[i], dtype), it, rf) for i in range(64)]
        err = metric(np.stack([np.asarray(jf(*x)) for x in args]), ref, n0)
        ms = time_fn(jf, args, 20, 5)
        ml = time_fn(jl, args, 20, 5)
        r = dict(selection=s, cfg=cfg, parameters_decoder_and_rest=nf.count(p), test=summary(err), errors=[float(x) for x in err],
                 ms_full=spread(ms), ms_latent_only=spread(ml), ms_full_per_case_median=[float(x) for x in np.median(ms, 0)])
        res.append(r)
        print('TEST', json.dumps({k: v for k, v in r.items() if k not in ('errors', 'ms_full_per_case_median')}), flush=True)
    (out / 'test.json').write_text(json.dumps(dict(
        cohort='test64 = params_draw(20260916, 64)', cohort_sha256=TEST_SHA, mesh=L, metric='max over t=0.05..0.25 of |u-u_ref|_2/|u0|_2',
        timing='batch 1, 20 warm-up queries, 5 repetitions x 64 cases, u0 on GPU -> five output fields on GPU',
        selection_sha256=hashlib.sha256(sel_text.encode()).hexdigest(), gpu=os.environ.get('NF_GPU_NAME'),
        job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'), backend=jax.default_backend(), jax=jax.__version__,
        results=res), indent=1) + '\n')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('cmd', choices=('grid', 'select', 'test'))
    a.add_argument('--models'); a.add_argument('--names'); a.add_argument('--data'); a.add_argument('--out')
    a.add_argument('--grid'); a.add_argument('--selection'); a.add_argument('--test')
    a.add_argument('--M', default='512,2048,8192'); a.add_argument('--iters', default='1,2,4')
    a.add_argument('--refine-iters', type=int, default=4)
    a.add_argument('--val-start', type=int, default=NTRAIN); a.add_argument('--cases', type=int, default=NGRID)
    a.add_argument('--fast-gate', type=float, default=3.85)
    a.add_argument('--residual-l', action='store_true'); a.add_argument('--allow-cpu', action='store_true')
    a = a.parse_args()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    dict(grid=grid, select=select, test=test)[a.cmd](a)


if __name__ == '__main__':
    main()
