"""Smooth latent decoders with continuous axis-table interpolation.

Only spatial coordinates and solved latent coordinates enter the decoder.
ModCP's zero-output correction exactly recovers CP, while modulation occurs
before SiLU and cannot be absorbed into the CP coefficient head.
"""
from dataclasses import dataclass, asdict
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


@dataclass(frozen=True)
class DecoderConfig:
    architecture: str = 'cp'
    k: int = 16
    rank: int = 64
    outputs: int = 1
    intervals: int = 256
    boundary: str = 'dirichlet'
    width: int = 64
    inr_width: int = 128
    head_width: int = 256
    frequencies: int = 8

    def to_dict(self):
        return asdict(self)


def _linear(key, din, dout, scale=1.):
    return {'w': scale * jax.random.normal(key, (din, dout), dtype=jnp.float64) / jnp.sqrt(float(din)),
            'b': jnp.zeros((dout,), dtype=jnp.float64)}


def _apply(layer, x):
    return x @ layer['w'] + layer['b']


def features(x, frequencies=8):
    x = jnp.asarray(x)
    phase = jnp.pi * x[..., None] * jnp.arange(1, frequencies + 1)
    return jnp.concatenate((x, jnp.sin(phase).reshape(x.shape[:-1] + (-1,)),
                            jnp.cos(phase).reshape(x.shape[:-1] + (-1,))), axis=-1)


def init_decoder(key, cfg):
    assert cfg.architecture in ('cp', 'modcp', 'film')
    assert cfg.boundary in ('dirichlet', 'free')
    keys = jax.random.split(key, 16)
    if cfg.architecture == 'film':
        nf = 2 * (1 + 2 * cfg.frequencies)
        w = cfg.inr_width
        return {'stem': _linear(keys[0], nf, w), 'layer': _linear(keys[1], w, w),
                'mod0': _linear(keys[2], cfg.k, 2*w, .05),
                'mod1': _linear(keys[3], cfg.k, 2*w, .05),
                'out': _linear(keys[4], w, cfg.outputs, .1),
                'skip': _linear(keys[5], nf, cfg.k*cfg.outputs, .1),
                'bias': jnp.zeros((cfg.outputs,))}
    # Smooth, distinct factor initialization avoids a random mesh-scale gradient
    # floor; every factor-table entry remains independently trainable.
    axis = jnp.linspace(0., 1., cfg.intervals+1)
    freq = (jnp.arange(cfg.rank) % 8 + 1).astype(jnp.float64)
    factors = []
    for d in range(2):
        phase = jax.random.uniform(keys[6+d], (cfg.outputs, cfg.rank, 1), minval=-jnp.pi, maxval=jnp.pi)
        factors.append(jnp.sin(jnp.pi*freq[None,:,None]*axis + phase))
    p = {'factors': jnp.stack(factors),
         'head0': _linear(keys[0], cfg.k, cfg.head_width),
         'head1': _linear(keys[1], cfg.head_width, cfg.head_width),
         'head2': _linear(keys[2], cfg.head_width, cfg.outputs*cfg.rank, .1/jnp.sqrt(cfg.rank)),
         'linear': _linear(keys[3], cfg.k, cfg.outputs*cfg.rank, 1./jnp.sqrt(cfg.rank)),
         'bias': jnp.zeros((cfg.outputs,))}
    if cfg.architecture == 'modcp':
        p = add_modulation(p, cfg, keys[10])
    return p


def add_modulation(cp_params, cfg, key=None):
    """Copy the CP parameters and attach an exactly zero residual correction."""
    key = jax.random.PRNGKey(0) if key is None else key
    keys = jax.random.split(key, 8)
    p = dict(cp_params)
    p['modulation'] = tuple({'stem': _linear(keys[4*d], 1+2*cfg.frequencies, cfg.width),
                             'film': _linear(keys[4*d+1], cfg.k, 2*cfg.width, .05),
                             'out': _linear(keys[4*d+2], cfg.width, cfg.outputs*cfg.rank, 0.)}
                            for d in range(2))
    return p


def coefficients(params, z, cfg):
    h = jax.nn.silu(_apply(params['head0'], z))
    h = jax.nn.silu(_apply(params['head1'], h))
    return (_apply(params['head2'], h) + _apply(params['linear'], z)).reshape(cfg.outputs, cfg.rank)


def factor_values(params, z, axis, d, cfg):
    t = jnp.clip(axis, 0., 1.) * cfg.intervals
    lo = jnp.minimum(jnp.floor(t).astype(jnp.int32), cfg.intervals-1)
    frac = t-lo
    table = params['factors'][d]
    base = jnp.moveaxis(table[..., lo]*(1-frac)+table[..., lo+1]*frac, -1, 0)
    if cfg.architecture == 'modcp':
        q = params['modulation'][d]
        a = _apply(q['stem'], features(axis[:,None], cfg.frequencies))
        gamma, beta = jnp.split(_apply(q['film'], z), 2)
        correction = _apply(q['out'], jax.nn.silu((1.+gamma)*a+beta))
        base = base + correction.reshape((-1, cfg.outputs, cfg.rank))
    return base


def boundary_mask(xy, cfg):
    if cfg.boundary == 'dirichlet':
        # Same binary mask on training nodes, with a linear boundary strip on a
        # refined grid. Masked table endpoints were never trained: interpolating
        # those arbitrary parameters into new near-wall points is invalid.
        t=jnp.minimum(jnp.minimum(xy*cfg.intervals,(1-xy)*cfg.intervals),1.)
        return jnp.prod(jnp.maximum(t,0.),axis=-1)
    return jnp.ones(xy.shape[0], dtype=jnp.float64)


def decode_points(params, z, xy, cfg):
    mask=boundary_mask(xy,cfg)
    if cfg.boundary=='dirichlet':xy=jnp.clip(xy,1./cfg.intervals,1-1./cfg.intervals)
    if cfg.architecture == 'film':
        f = features(xy, cfg.frequencies)
        a = _apply(params['stem'], f)
        g,b = jnp.split(_apply(params['mod0'], z), 2)
        a = jax.nn.silu((1.+g)*a+b)
        a = _apply(params['layer'], a)
        g,b = jnp.split(_apply(params['mod1'], z), 2)
        a = jax.nn.silu((1.+g)*a+b)
        skip = _apply(params['skip'], f).reshape((-1,cfg.outputs,cfg.k))
        u = _apply(params['out'], a)+jnp.einsum('pck,k->pc',skip,z)+params['bias']
    else:
        a = factor_values(params,z,xy[:,0],0,cfg)
        b = factor_values(params,z,xy[:,1],1,cfg)
        u = jnp.einsum('pcr,pcr,cr->pc',a,b,coefficients(params,z,cfg))+params['bias']
    return u*mask[:,None]


def decode_grid(params,z,intervals,cfg):
    axis = jnp.linspace(0.,1.,intervals+1)
    if cfg.architecture == 'film':
        xy = jnp.stack(jnp.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape((-1,2))
        # Scan fixed row chunks to bound dense INR activation memory.
        xy = xy.reshape((intervals+1,intervals+1,2))
        return jax.lax.map(lambda row:decode_points(params,z,row,cfg),xy)
    factor_axis=jnp.clip(axis,1./cfg.intervals,1-1./cfg.intervals) if cfg.boundary=='dirichlet' else axis
    a = factor_values(params,z,factor_axis,0,cfg)
    b = factor_values(params,z,factor_axis,1,cfg)
    u = jnp.einsum('icr,jcr,cr->ijc',a,b,coefficients(params,z,cfg))+params['bias']
    if cfg.boundary == 'dirichlet':
        envelope=jnp.maximum(jnp.minimum(jnp.minimum(axis*cfg.intervals,(1-axis)*cfg.intervals),1.),0.)
        u=u*envelope[:,None,None]*envelope[None,:,None]
    return u


def prepare_points(params,xy,cfg):
    """Offline coordinate-only cache for a frozen checkpoint and point set."""
    mask=boundary_mask(xy,cfg)
    if cfg.boundary=='dirichlet':xy=jnp.clip(xy,1./cfg.intervals,1-1./cfg.intervals)
    if cfg.architecture=='film':
        f=features(xy,cfg.frequencies)
        return (_apply(params['stem'],f),_apply(params['skip'],f).reshape((-1,cfg.outputs,cfg.k)),mask)
    cache=[]
    for d in range(2):
        t=jnp.clip(xy[:,d],0.,1.)*cfg.intervals
        lo=jnp.minimum(jnp.floor(t).astype(jnp.int32),cfg.intervals-1);frac=t-lo
        table=params['factors'][d]
        cache.append(jnp.moveaxis(table[...,lo]*(1-frac)+table[...,lo+1]*frac,-1,0))
        if cfg.architecture=='modcp':
            cache.append(_apply(params['modulation'][d]['stem'],features(xy[:,d,None],cfg.frequencies)))
    return tuple(cache)+(mask,)


def decode_cached(params,z,cache,cfg):
    """Sampled inference; latent-dependent modulation is always recomputed."""
    if cfg.architecture=='film':
        a,skip,mask=cache
        g,b=jnp.split(_apply(params['mod0'],z),2);a=jax.nn.silu((1+g)*a+b)
        a=_apply(params['layer'],a)
        g,b=jnp.split(_apply(params['mod1'],z),2);a=jax.nn.silu((1+g)*a+b)
        u=_apply(params['out'],a)+jnp.einsum('pck,k->pc',skip,z)+params['bias']
    else:
        if cfg.architecture=='cp':a,b,mask=cache
        else:
            a,sa,b,sb,mask=cache
            bases=[]
            for base,stem,q in zip((a,b),(sa,sb),params['modulation']):
                g,shift=jnp.split(_apply(q['film'],z),2)
                bases.append(base+_apply(q['out'],jax.nn.silu((1+g)*stem+shift)).reshape((-1,cfg.outputs,cfg.rank)))
            a,b=bases
        u=jnp.einsum('pcr,pcr,cr->pc',a,b,coefficients(params,z,cfg))+params['bias']
    return u*mask[:,None]


def initial_codes(count,k,seed):
    return .1*jax.random.normal(jax.random.PRNGKey(seed),(count,k),dtype=jnp.float64)
