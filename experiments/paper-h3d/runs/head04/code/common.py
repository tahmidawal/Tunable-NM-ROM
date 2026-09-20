"""Self-contained f64 3-spatial-dimensional heat and learned-bank primitives."""
from __future__ import annotations
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def checkpoint(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f: pickle.dump(jax.tree_util.tree_map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, value), f)
    tmp.replace(path)


def sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def coords(n):
    a = np.arange(1, n, dtype=np.float64) / n
    return np.stack(np.meshgrid(a, a, a, indexing='ij'), -1).reshape(-1, 3)


def family(seed, count):
    a = np.random.default_rng(seed).random((count, 5))
    return a * np.array([.3, .3, .3, .05, .4]) + np.array([.35, .35, .35, .10, .8])


def mask(x):
    return 64 * jnp.prod(x * (1 - x), axis=-1)


def initial(x, p):
    return p[4] * mask(x) * jnp.exp(-jnp.sum((x-p[:3])**2, axis=-1)/(2*p[3]**2))


def dst_axis(u, axis):
    u = jnp.moveaxis(u, axis, -1); n = u.shape[-1] + 1
    z = jnp.zeros(u.shape[:-1] + (1,), dtype=u.dtype)
    odd = jnp.concatenate((z, u, z, -u[..., ::-1]), -1)
    out = -jnp.fft.rfft(odd, axis=-1).imag[..., 1:n] / jnp.sqrt(2.0*n)
    return jnp.moveaxis(out, -1, axis)


def dst3(u):
    for a in (-1, -2, -3): u = dst_axis(u, a)
    return u


def eigenvalues(n, continuum=False):
    k = jnp.arange(1, n, dtype=jnp.float64)
    l = (jnp.pi*k)**2 if continuum else 4*n*n*jnp.sin(jnp.pi*k/(2*n))**2
    return l[:, None, None] + l[None, :, None] + l[None, None, :]


def negative_laplacian(u, n):
    p = jnp.pad(u, ((1, 1),)*3)
    return n*n*(6*u-p[2:,1:-1,1:-1]-p[:-2,1:-1,1:-1]-p[1:-1,2:,1:-1]-p[1:-1,:-2,1:-1]-p[1:-1,1:-1,2:]-p[1:-1,1:-1,:-2])


@jax.jit
def propagate(u0, lam, times, nu):
    c = dst3(u0)
    later = jax.vmap(lambda t: dst3(c*jnp.exp(-nu*t*lam)))(times[1:])
    return jnp.concatenate((u0[None], later))


def dataset(n, params, cfg, continuum=False):
    x = jnp.asarray(coords(n)); lam = eigenvalues(n, continuum); ts = jnp.asarray(cfg['times'])
    rows = []
    for p in params:
        u = initial(x, jnp.asarray(p)).reshape((n-1,)*3)
        rows.append(np.asarray(propagate(u, lam, ts, cfg['diffusivity'])))
    return np.stack(rows)


def restrict(u, fine, coarse):
    assert fine % coarse == 0
    s = fine//coarse
    return np.asarray(u)[..., s-1::s, s-1::s, s-1::s]


def modes(count, n):
    k = np.arange(1, n)
    triples = np.stack(np.meshgrid(k, k, k, indexing='ij'), -1).reshape(-1, 3)
    order = np.argsort(np.sum(triples**2, axis=1), kind='stable')
    return triples[order[:count]]


def phi(x, triples):
    return jnp.sqrt(8.) * jnp.prod(jnp.sin(jnp.pi*x[:, None, :]*triples[None, :, :]), axis=-1)


def mode_eigenvalues(n, triples):
    return jnp.sum(4*n*n*jnp.sin(jnp.pi*jnp.asarray(triples)/(2*n))**2, axis=1)


def mlp_init(key, sizes):
    out = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        out.append((jax.random.normal(sub, (a,b), dtype=jnp.float64)*jnp.sqrt(2/a), jnp.zeros(b)))
    return out


def mlp(p, x):
    for w,b in p[:-1]: x = jax.nn.silu(x@w+b)
    w,b = p[-1]
    return x@w+b


def init_bank(key, cfg, scale):
    a,b = jax.random.split(key)
    return dict(freq=jax.random.normal(a,(3,cfg['fourier_features']),dtype=jnp.float64)*cfg['fourier_scale'],
                net=mlp_init(b,[2*cfg['fourier_features'],cfg['bank_width'],cfg['bank_width'],cfg['bank_rank']]),
                scale=jnp.asarray(scale))


def features(p, x):
    ang = 2*jnp.pi*(x@p['freq'])
    return (p['scale']*mask(x))[:,None]*mlp(p['net'],jnp.concatenate((jnp.sin(ang),jnp.cos(ang)),-1))


def init_head(key, k, r, width):
    a,b = jax.random.split(key)
    return dict(net=mlp_init(a,[k,width,width,r]),skip=jax.random.normal(b,(k,r),dtype=jnp.float64)*.1)


def head(p, z):
    return mlp(p['net'],z)+z@p['skip']


def bank_at(p, n, chunk=8192):
    x = coords(n)
    fn = jax.jit(features)
    return np.concatenate([np.asarray(fn(p,jnp.asarray(x[s:s+chunk]))) for s in range(0,len(x),chunk)])


def metrics(prediction, truth):
    a,b = np.asarray(prediction).reshape(len(prediction),-1),np.asarray(truth).reshape(len(truth),-1)
    d = np.linalg.norm(a-b,axis=1); norm = np.linalg.norm(b,axis=1)
    current = d/np.maximum(norm,1e-300); initial = d/max(norm[0],1e-300)
    return dict(current_by_time=current.tolist(),initial_by_time=initial.tolist(),
                current_evolved=float(max(current[1:])),current_all=float(max(current)),
                initial_evolved=float(max(initial[1:])),initial_all=float(max(initial)),
                initial_fit=float(current[0]),energy=(np.sum(a*a,axis=1)/2).tolist())


def burn(seconds):
    a = jnp.eye(256,dtype=jnp.float64)+.0001
    fn = jax.jit(lambda x: x@x); fn(a).block_until_ready()
    start=time.perf_counter()
    while time.perf_counter()-start<seconds: fn(a).block_until_ready()


def verify(cfg):
    from scipy.fft import dstn
    n=8; rng=np.random.default_rng(920301); u=rng.normal(size=(n-1,)*3)
    actual=np.asarray(dst3(jnp.asarray(u))); ref=dstn(u,type=1,norm='ortho')
    rel=lambda a,b:float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b),1e-300))
    checks=dict(dst_scipy=rel(actual,ref),dst_inverse=rel(dst3(jnp.asarray(actual)),u),
                stencil_diagonal=rel(dst3(negative_laplacian(jnp.asarray(u),n)),eigenvalues(n)*actual))
    x=jnp.asarray(coords(n)); triple=jnp.array([[1,2,3]])
    mode=phi(x,triple)[:,0].reshape((n-1,)*3); ts=jnp.array([0.,.1,.2]); nu=.02
    exact=np.exp(-nu*np.asarray(ts)*14*np.pi**2)[:,None,None,None]*np.asarray(mode)
    checks['continuum_mode']=rel(propagate(mode,eigenvalues(n,True),ts,nu),exact)
    tests=phi(x,modes(20,n)); lamb=mode_eigenvalues(n,modes(20,n))
    checks['weak_stencil']=rel(tests.T@negative_laplacian(jnp.asarray(u),n).reshape(-1),lamb*(tests.T@u.reshape(-1)))
    assert max(checks.values())<1e-11, checks
    checks['passed']=True
    return checks
