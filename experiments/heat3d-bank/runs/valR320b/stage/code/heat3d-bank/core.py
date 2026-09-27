"""Dimension-generic (d=2,3) f64 heat primitives, learned-bank NM-ROM with
analytically eliminated linear corrections, labelled baselines and FOM arms.

Sources (copied ideas, not imports, unless stated): experiments/paper-h3d/rom.py and
common.py at 230c5410 (correction elimination, LM, MLP bank/head format);
experiments/mr-heat2d/heat_core.py and cg_paths.py (2D family, CG recurrence).
The audited 2D checkpoint uses experiments/separable-decoder/sep_common.py directly.
"""
from __future__ import annotations
import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np
import scipy.linalg
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl

HERE = Path(__file__).resolve().parent


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n'); tmp.replace(path)


def sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def block(x):
    jax.block_until_ready(x); return x


# ---------------------------------------------------------------- grid / PDE
def axis_nodes(n):
    return np.arange(1, n, dtype=np.float64) / n


def coords(n, d):
    a = axis_nodes(n)
    return np.stack(np.meshgrid(*([a] * d), indexing='ij'), -1).reshape(-1, d)


def mask(x):
    return 4.0 ** x.shape[-1] * jnp.prod(x * (1 - x), axis=-1)


def family(kind, seed, count):
    """Draw rows are [centre (d), width, amplitude]; identical to the source lanes."""
    if kind == 'mr2d':   # mr-heat2d heat_core.sample_family, config-pilot ranges
        rng = np.random.default_rng(seed)
        return np.column_stack((rng.uniform(.35, .65, (count, 2)), rng.uniform(.10, .15, count),
                                rng.uniform(.8, 1.2, count)))
    assert kind == 'h3d'  # paper-h3d common.family
    a = np.random.default_rng(seed).random((count, 5))
    return a * np.array([.3, .3, .3, .05, .4]) + np.array([.35, .35, .35, .10, .8])


def initial_grid(n, d, draw):
    """Separable evaluation of amp*mask*gaussian on the interior grid (no coordinate table)."""
    a = jnp.asarray(axis_nodes(n)); draw = jnp.asarray(draw); out = draw[d + 1]
    for ax in range(d):
        f = 4 * a * (1 - a) * jnp.exp(-(a - draw[ax]) ** 2 / (2 * draw[d] ** 2))
        out = out * f.reshape((-1,) + (1,) * (d - 1 - ax))
    return out


def dst_axis(u, axis):
    u = jnp.moveaxis(u, axis, -1); n = u.shape[-1] + 1
    z = jnp.zeros(u.shape[:-1] + (1,), dtype=u.dtype)
    odd = jnp.concatenate((z, u, z, -u[..., ::-1]), -1)
    out = -jnp.fft.rfft(odd, axis=-1).imag[..., 1:n] / jnp.sqrt(2.0 * n)
    return jnp.moveaxis(out, -1, axis)


def dstd(u, d):
    for ax in range(1, d + 1): u = dst_axis(u, -ax)
    return u


def lam_axis(n, continuum=False):
    k = jnp.arange(1, n, dtype=jnp.float64)
    return (jnp.pi * k) ** 2 if continuum else 4 * n * n * jnp.sin(jnp.pi * k / (2 * n)) ** 2


def eig_grid(n, d, continuum=False):
    l = lam_axis(n, continuum); out = 0.
    for ax in range(d): out = out + l.reshape((-1,) + (1,) * (d - 1 - ax))
    return out


def negative_laplacian(u, n):
    d = u.ndim; p = jnp.pad(u, ((1, 1),) * d); out = 2 * d * u
    core = tuple(slice(1, -1) for _ in range(d))
    for ax in range(d):
        hi = list(core); lo = list(core); hi[ax] = slice(2, None); lo[ax] = slice(None, -2)
        out = out - p[tuple(hi)] - p[tuple(lo)]
    return n * n * out


def make_propagate(d):
    @jax.jit
    def propagate(u0, lam, times, nu):
        c = dstd(u0, d)
        later = jax.lax.map(lambda t: dstd(c * jnp.exp(-nu * t * lam), d), times[1:])
        return jnp.concatenate((u0[None], later))
    return propagate


def mode_list(spec, n, d):
    if 'per_axis' in spec:   # mr-heat2d tensor ordering
        k = np.arange(1, spec['per_axis'] + 1)
        return np.stack(np.meshgrid(*([k] * d), indexing='ij'), -1).reshape(-1, d)
    k = np.arange(1, min(n, 64))
    t = np.stack(np.meshgrid(*([k] * d), indexing='ij'), -1).reshape(-1, d)
    return t[np.argsort(np.sum(t ** 2, axis=1), kind='stable')[:spec['count']]]


def mode_eigs(n, modes):
    return np.sum(4.0 * n * n * np.sin(np.pi * np.asarray(modes) / (2 * n)) ** 2, axis=1)


def moments(u, modes, n, d):
    """phi^T u / n^d for continuum-normalised sine tests == orthonormal DST / n^(d/2)."""
    c = dstd(u, d)
    return c[(Ellipsis,) + tuple(jnp.asarray(modes[:, ax] - 1) for ax in range(d))] / n ** (d / 2)


def explicit_tests(n, d, modes):
    x = coords(n, d)
    return np.sqrt(2.0) ** d * np.prod(np.sin(np.pi * x[:, None, :] * modes[None]), axis=-1) / n ** d


# ---------------------------------------------------------------- models
def mlp(p, x):
    for w, b in p[:-1]: x = jax.nn.silu(x @ w + b)
    w, b = p[-1]; return x @ w + b


def mlp_features(p, x):
    ang = 2 * jnp.pi * (x @ p['freq'])
    return (p['scale'] * mask(x))[:, None] * mlp(p['net'], jnp.concatenate((jnp.sin(ang), jnp.cos(ang)), -1))


def mlp_head(p, z):
    return mlp(p['net'], z) + z @ p['skip']


def load_model(spec, root):
    """Returns dict(d, feature_fn, head_fn, bank_params, head_params, codes, rotation, directions|None)."""
    root = Path(root); tojax = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
    if spec['kind'] == 'sep2d':
        sys.path.insert(0, str(HERE.parent / 'separable-decoder'))
        import sep_common as sc
        ck = pickle.loads((root / spec['checkpoint']).read_bytes())
        p = tojax(ck['params'])
        return dict(d=2, feature_fn=sc.features, head_fn=sc.head, bank_params=p, head_params=p,
                    codes=jnp.asarray(ck['codes']), rotation=None, directions=None, family='mr2d',
                    sha256={spec['checkpoint']: hashlib.sha256((root / spec['checkpoint']).read_bytes()).hexdigest()})
    bank = pickle.loads((root / spec['bank']).read_bytes()); head = pickle.loads((root / spec['head']).read_bytes())
    d = int(np.asarray(bank['params']['freq']).shape[0])
    return dict(d=d, feature_fn=mlp_features, head_fn=mlp_head, bank_params=tojax(bank['params']),
                head_params=tojax(head['params']), codes=jnp.asarray(head['codes']),
                rotation=np.asarray(bank['rotation']), directions=np.asarray(head['directions']),
                family='mr2d' if d == 2 else 'h3d',
                sha256={k: hashlib.sha256((root / spec[k]).read_bytes()).hexdigest() for k in ('bank', 'head')})


def bank_at(model, n, chunk=1 << 18):
    """[N,R] device array, evaluated in coordinate chunks (never a full activation tensor)."""
    d = model['d']; a = axis_nodes(n); N = (n - 1) ** d
    rot = None if model['rotation'] is None else jnp.asarray(model['rotation'])
    @jax.jit
    def piece(p, idx):
        sub = jnp.stack(jnp.unravel_index(idx, (n - 1,) * d), -1)
        g = model['feature_fn'](p, jnp.asarray(a)[sub])
        return g if rot is None else g @ rot
    parts = [piece(model['bank_params'], jnp.arange(s, min(s + chunk, N))) for s in range(0, N, chunk)]
    return block(jnp.concatenate(parts)) if len(parts) > 1 else block(parts[0])


def tsqr_r(bank, chunk=1 << 20):
    """Triangular factor of the bank by chunked QR (exact; avoids an N x R Q)."""
    rs = [np.linalg.qr(np.asarray(bank[s:s + chunk]), mode='r') for s in range(0, bank.shape[0], chunk)]
    r = np.linalg.qr(np.concatenate(rs), mode='r') if len(rs) > 1 else rs[0]
    return r * np.where(np.diag(r) < 0, -1., 1.)[:, None]


def weak_matrix(bank, modes, n, d, cols=16):
    """a = tests^T bank via DST of bank columns (no N x M test matrix)."""
    f = jax.jit(lambda b: moments(jnp.moveaxis(b.reshape((n - 1,) * d + (b.shape[1],)), -1, 0), modes, n, d))
    return np.concatenate([np.asarray(f(bank[:, s:s + cols])) for s in range(0, bank.shape[1], cols)]).T


def sep2d_directions(model, cfg_train):
    """SVD of field-orthonormal training residuals at the stored codes, mapped to raw
    bank-coefficient coordinates (same rule as paper-h3d train_head; training data only)."""
    n = cfg_train['train_intervals']; d = 2; times = jnp.asarray(cfg_train['times'])
    draws = np.concatenate([family('mr2d', s, c) for s, c in cfg_train['train_draws']])
    prop = make_propagate(d); lam = eig_grid(n, d)
    u = np.concatenate([np.asarray(prop(initial_grid(n, d, p), lam, times, cfg_train['diffusivity'])).reshape(len(times), -1)
                        for p in draws])
    assert len(u) == len(model['codes']), (len(u), len(model['codes']))
    g = np.asarray(bank_at(model, n)); q, r = np.linalg.qr(g, mode='reduced')
    resid = u @ q - np.asarray(model['head_fn'](model['head_params'], model['codes'])) @ r.T
    _, sv, vt = np.linalg.svd(resid, full_matrices=False)
    directions = np.linalg.solve(r, vt.T)
    err = np.sqrt((np.sum(resid ** 2, 1) + np.maximum(np.sum(u * u, 1) - np.sum((u @ q) ** 2, 1), 0)) / np.sum(u * u, 1))
    return directions, dict(singular_values=sv.tolist(), training_error_worst=float(err.max()),
                            training_error_mean=float(err.mean()), draws_hash=sha(draws), direction_hash=sha(directions))


# ---------------------------------------------------------------- reduced solver
def lm(head_fn, budget, tol, cholesky=False):
    """Damped monotone LM on ||matrix h(z) - target||/||target||. reasons: 0 budget 1 stationary 2 tiny 3 damping."""
    def solve(params, matrix, target, z0, scale=None):
        scale = jnp.maximum(jnp.linalg.norm(target), 1e-14) if scale is None else scale
        def parts(z):
            h, d = head_fn(params, z), jax.jacfwd(head_fn, argnums=1)(params, z)
            return (matrix @ h - target) / scale, matrix @ d / scale
        r, j = parts(z0)
        state = (z0, r, j, jnp.dot(r, r), jnp.float64(1e-4), jnp.int32(0), jnp.int32(0), jnp.int32(0))
        def cond(s):
            grad = jnp.linalg.norm(s[2].T @ s[1]) / jnp.maximum(jnp.linalg.norm(s[2]), 1e-30)
            return (s[5] < budget) & (s[7] == 0) & (grad > tol)
        def body(s):
            z, r, j, value, damping, attempts, accepted, _ = s
            gram = j.T @ j; system = gram + damping * jnp.diag(jnp.maximum(jnp.diag(gram), 1e-12))
            dz = jsl.cho_solve((jnp.linalg.cholesky(system), True), -j.T @ r) if cholesky else jnp.linalg.solve(system, -j.T @ r); new = z + dz; rn, jn = parts(new); vn = jnp.dot(rn, rn)
            ok = jnp.isfinite(vn) & (vn < value)
            damping = jnp.where(ok, jnp.maximum(damping / 3, 1e-12), damping * 10)
            reason = jnp.where(jnp.linalg.norm(dz) < 1e-12 * (1 + jnp.linalg.norm(z)), 2,
                               jnp.where(damping > 1e12, 3, 0)).astype(jnp.int32)
            return (jnp.where(ok, new, z), jnp.where(ok, rn, r), jnp.where(ok, jn, j), jnp.where(ok, vn, value),
                    damping, attempts + 1, accepted + ok.astype(jnp.int32), reason)
        z, r, j, value, _, attempts, accepted, reason = jax.lax.while_loop(cond, body, state)
        grad = jnp.linalg.norm(j.T @ r) / jnp.maximum(jnp.linalg.norm(j), 1e-30)
        reason = jnp.where(grad <= tol, 1, reason)
        return z, jnp.array([attempts, accepted, reason, jnp.sqrt(value), grad])
    return solve


def eliminate(a, directions, compress=False):
    """Variable projection of q linear corrections out of ||a (h + D y) - t||.
    compress=True replaces the rows by the exact R-factor coordinates (norm preserving)."""
    l = a @ directions
    if directions.shape[1]:
        qq, rr = np.linalg.qr(l, mode='reduced'); sv = np.linalg.svd(rr, compute_uv=False)
        assert sv[-1] > sv[0] * 1e-12, ('correction rank', sv[[0, -1]])
    else:
        qq, rr = np.zeros((a.shape[0], 0)), np.zeros((0, 0))
    ap = a - qq @ (qq.T @ a); rows = None
    if compress and ap.shape[0] > ap.shape[1]:
        rows, ap = np.linalg.qr(ap, mode='reduced')   # ||ap h - tp|| = ||R h - rows^T tp|| + const
    return dict(a=jnp.asarray(a), qq=jnp.asarray(qq), rr=jnp.asarray(rr), ap=jnp.asarray(ap),
                rows=None if rows is None else jnp.asarray(rows))


def make_stages(model, setup, q, opt):
    """Returns separately callable jitted stages (encode, init, evolve, decode) and the fused query.

    opt: init ('field'|'moments'), stepping ('cn'|'exact_seq'|'exact_direct'), dt, starts, mean_start,
         fit_budget, step_budget, tolerance, compress, direct_starts.
    """
    d, n, modes = model['d'], setup['n'], setup['modes']; head_fn = model['head_fn']; hp = model['head_params']
    codes = model['codes']; times = np.asarray(setup['times']); nu = setup['nu']; lam = jnp.asarray(setup['mode_lam'])
    assert 0 <= q <= np.asarray(setup['directions']).shape[1] and q + codes.shape[1] <= setup['a'].shape[0], ('invalid q', q)
    D = np.asarray(setup['directions'])[:, :q]; Dj = jnp.asarray(D)
    E0 = eliminate(setup['rtri'] if opt['init'] == 'field' else setup['a'], D, opt.get('compress', False))
    E1 = eliminate(setup['a'], D, opt.get('compress', False))
    rtri_t = jnp.asarray(setup['rtri'].T)
    fit0 = lm(head_fn, opt['fit_budget'], opt['tolerance'], opt.get('cholesky', False)); fit1 = lm(head_fn, opt['step_budget'], opt['tolerance'], opt.get('cholesky', False))
    def proj(E, t):   # projected target and the UNCOMPRESSED norm used for LM scaling (compression must not change stopping)
        tp = t - E['qq'] @ (E['qq'].T @ t); scale = jnp.maximum(jnp.linalg.norm(tp), 1e-14)
        return (tp if E['rows'] is None else E['rows'].T @ tp), scale
    lib0 = head_fn(hp, codes) @ E0['ap'].T; lib1 = head_fn(hp, codes) @ E1['ap'].T   # nearest-code search in the (possibly compressed) projected space
    def recover(E, z, t):
        h = head_fn(hp, z)
        return h + Dj @ jsl.solve_triangular(E['rr'], E['qq'].T @ (t - E['a'] @ h), lower=False) if q else h
    def starts(lib, tp, k, mean):
        z = codes[jnp.argsort(jnp.sum((lib - tp) ** 2, axis=1))[:k]]
        return jnp.concatenate((z, jnp.mean(codes, axis=0)[None])) if mean else z
    def best_fit(fit, E, lib, t, k, mean):
        tp, scale = proj(E, t); zs, stats = jax.vmap(lambda z: fit(hp, E['ap'], tp, z, scale))(starts(lib, tp, k, mean))
        best = jnp.argmin(stats[:, 3]); z = zs[best]; return z, recover(E, z, t), stats, stats[best]

    def encode(u0, bank):
        m0 = moments(u0, modes, n, d)
        t0 = jsl.solve_triangular(rtri_t, bank.T @ u0.reshape(-1), lower=True) if opt['init'] == 'field' else m0
        return t0, m0
    def init(t0):
        z, c0, stats, chosen = best_fit(fit0, E0, lib0, t0, opt['starts'], opt.get('mean_start', False))
        return z, c0, jnp.concatenate((stats, chosen[None]))   # all starts, then the SELECTED start last
    nout = len(times) - 1; dtout = float(times[1] - times[0])
    if opt['stepping'] == 'cn':
        dt = opt['dt']; stride = int(round(dtout / dt)); nsteps = stride * nout
        assert np.allclose(np.arange(len(times)) * stride * dt, times)
        factor = (1 - dt * nu * lam / 2) / (1 + dt * nu * lam / 2)
    else:
        stride, nsteps = 1, nout; factor = jnp.exp(-nu * lam * dtout)
    def evolve(z, coef, m0):
        if opt['stepping'] == 'exact_direct':   # independent moment matches of the exact flow of the SUPPLIED field
            targets = jnp.exp(-nu * lam[None] * jnp.asarray(times[1:])[:, None]) * m0[None]
            _, coefs, _, chosen = jax.vmap(lambda t: best_fit(fit1, E1, lib1, t, opt.get('direct_starts', 1), False))(targets)
            return coefs, chosen
        if opt['stepping'] == 'exact_chain':   # same exact supplied-field targets as 'direct', solved in time order with warm starts
            targets = jnp.exp(-nu * lam[None] * jnp.asarray(times[1:])[:, None]) * m0[None]
            def chain(zp, t):
                tp, scale = proj(E1, t); zn, info = fit1(hp, E1['ap'], tp, zp, scale); return zn, (recover(E1, zn, t), info)
            _, (coefs, infos) = jax.lax.scan(chain, z, targets); return coefs, infos
        def step(carry, _):
            zp, cp = carry; t = factor * (E1['a'] @ cp)
            zn, info = fit1(hp, E1['ap'], *proj(E1, t)[:1], zp, proj(E1, t)[1]); cn = recover(E1, zn, t)
            return (zn, cn), (cn, info)
        _, (coefs, infos) = jax.lax.scan(step, (z, coef), None, length=nsteps)
        return coefs[stride - 1::stride], infos
    def decode(coefs, bank):
        return (coefs @ bank.T).reshape((len(times),) + (n - 1,) * d)
    def query(u0, bank):
        t0, m0 = encode(u0, bank); z, c0, s0 = init(t0); coefs, s1 = evolve(z, c0, m0)
        return decode(jnp.concatenate((c0[None], coefs)), bank), s0, s1
    return dict(encode=jax.jit(encode), init=jax.jit(init), evolve=jax.jit(evolve), decode=jax.jit(decode), query=jax.jit(query))


def linear_bank(setup, init, direct=False):
    """LABELLED BASELINE: free bank coefficients, exact reduced weak evolution. Not the NM-ROM."""
    a = setup['a']; left = np.linalg.pinv(a); gen = -setup['nu'] * left @ (setup['mode_lam'][:, None] * a)
    assert np.max(np.linalg.eigvals(gen).real) < 1e-8
    maps = jnp.asarray(np.stack([scipy.linalg.expm(t * gen) for t in setup['times']]))
    n, d, modes = setup['n'], setup['d'], setup['modes']; rt = jnp.asarray(setup['rtri']); leftj = jnp.asarray(left)
    @jax.jit
    def query(u0, bank):
        if init == 'field':
            c = jsl.solve_triangular(rt, jsl.solve_triangular(rt.T, bank.T @ u0.reshape(-1), lower=True), lower=False)
        else:
            c = leftj @ moments(u0, modes, n, d)
        if direct:   # free coefficients fitted to the SAME exact propagated supplied-field moments as the ROM 'direct' arm
            m0 = moments(u0, modes, n, d); lam = jnp.asarray(setup['mode_lam']); ts = jnp.asarray(setup['times'][1:])
            later = (jnp.exp(-setup['nu'] * lam[None] * ts[:, None]) * m0[None]) @ leftj.T
            return (jnp.concatenate((c[None], later)) @ bank.T).reshape((len(setup['times']),) + (n - 1,) * d)
        return ((maps @ c) @ bank.T).reshape((len(setup['times']),) + (n - 1,) * d)
    return query


# ---------------------------------------------------------------- full-order arms
def cg_solve(operator, rhs, x0, tol, budget):
    norm2 = jnp.sum(rhs * rhs); thr = tol ** 2 * norm2
    def cond(s): return (s[2] > thr) & (s[4] < budget) & jnp.isfinite(s[2])
    def step(s):
        x, r, g, p, c = s; ap = operator(p); al = g / jnp.sum(p * ap); rn = r - al * ap; gn = jnp.sum(rn * rn)
        return x + al * p, rn, gn, rn + (gn / g) * p, c + 1
    def run(x0, count):
        r = rhs - operator(x0); g = jnp.sum(r * r)
        x, _, _, _, count = jax.lax.while_loop(cond, step, (x0, r, g, r, count)); return x, count
    true_rel = lambda x: jnp.linalg.norm((rhs - operator(x)).reshape(-1)) / jnp.maximum(jnp.sqrt(norm2), 1e-300)
    accept = lambda t: t <= tol * (1 + 1e-6) + 1e-12
    x, count = run(x0, jnp.int32(0))
    # Recurrence drift: restart from the TRUE residual while budget remains (charged to the FOM, at most 3 restarts).
    def rcond(s): return (~accept(true_rel(s[0]))) & (s[1] < budget) & (s[2] < 3)
    x, count, restarts = jax.lax.while_loop(rcond, lambda s: (*run(s[0], s[1]), s[2] + 1), (x, count, jnp.int32(0)))
    true = true_rel(x)
    return x, jnp.array([count, true, accept(true).astype(jnp.float64)])


def make_cg(n, times, spec, nu):
    """Unpreconditioned warm-started CG on the theta scheme (theta=.5: Crank-Nicolson). Named FOM family."""
    dt, theta = spec['dt'], spec.get('theta', .5); stride = int(round((times[1] - times[0]) / dt))
    assert np.allclose(np.arange(len(times)) * stride * dt, times)
    @jax.jit
    def query(u0):
        op = lambda u: u + theta * dt * nu * negative_laplacian(u, n)
        def inner(u, _):
            rhs = u - (1 - theta) * dt * nu * negative_laplacian(u, n)
            return cg_solve(op, rhs, u, spec['tolerance'], spec.get('max_iterations', 5000))
        def outer(u, _):
            un, info = jax.lax.scan(inner, u, None, length=stride); return un, (un, info)
        _, (later, infos) = jax.lax.scan(outer, u0, None, length=len(times) - 1)
        return jnp.concatenate((u0[None], later)), infos.reshape(-1, 3)
    return query


def upsample(v, nc, n, d):
    """Multilinear interpolation of interior coarse nodal values (zero Dirichlet) onto the fine interior grid."""
    s = n // nc; idx = np.arange(1, n); c = jnp.asarray(idx // s); w = jnp.asarray((idx % s) / s)
    for ax in range(1, d + 1):
        v = jnp.moveaxis(v, -ax, -1); v = jnp.pad(v, [(0, 0)] * (v.ndim - 1) + [(1, 1)])
        v = jnp.moveaxis(v[..., c] * (1 - w) + v[..., c + 1] * w, -1, -ax)
    return v


def make_coarse(n, nc, d, times, spec, nu):
    """Coarse-grid FOM: inject u0 to nc intervals, same CN-CG algorithm there, interpolate every output to n."""
    s = n // nc; assert n % nc == 0; solve = make_cg(nc, times, spec, nu); sl = (slice(s - 1, None, s),) * d
    @jax.jit
    def query(u0):
        fields, info = solve(u0[sl]); return jnp.concatenate((u0[None], upsample(fields[1:], nc, n, d))), info
    return query


def rel_errors(pred, truth):
    ax = tuple(range(1, pred.ndim))
    return jnp.sqrt(jnp.sum((pred - truth) ** 2, axis=ax) / jnp.sum(truth ** 2, axis=ax))


def burn(seconds):
    a = jnp.eye(512, dtype=jnp.float64) + 1e-4; f = jax.jit(lambda x: jnp.tanh(x @ x / 512)); block(f(a)); t = time.perf_counter()
    while time.perf_counter() - t < seconds: a = block(f(a))
