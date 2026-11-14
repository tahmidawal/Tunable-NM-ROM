"""Neural-field decoders for the two baseline arms (pure JAX, parameters are pytrees of arrays).

Arm C (CROM-style):   u(x; z) = MLP([gamma(x), z])                         latent z concatenated to the coordinate features
                      z = encoder(state on the 65^2 sub-grid)              convolutional encoder, used at t = 0 only online
Arm L (CoLoRA-style): u(x; alpha) = MLP(gamma(x)) with hidden layers W_l + A_l diag(alpha_l) B_l
                      alpha = hyper(t, mu)                                 CoLoRA-D hyper-network

gamma(x) = [sin(2 pi B x), cos(2 pi B x)], B fixed Gaussian (128 x 2, scale 4): the paper's bank embedding.
Every decoder has the signature decode(params, x[P,2], z[k]) -> u[P], so the online solver is arm-agnostic.
"""
import numpy as np
import jax, jax.numpy as jnp

L = 256
NF, FSCALE = 128, 4.0


def _dense(key, n_in, n_out, gain=1.0):
    return dict(W=jax.random.normal(key, (n_in, n_out), jnp.float32) * gain / np.sqrt(n_in), b=jnp.zeros((n_out,), jnp.float32))


def fourier_matrix():
    return np.random.default_rng(20261111).normal(size=(2, NF)) * FSCALE


def gamma(B, x):
    a = (2 * np.pi) * (x @ jax.lax.stop_gradient(B))   # B is fixed, never trained
    return jnp.concatenate((jnp.sin(a), jnp.cos(a)), axis=-1)


# ------------------------------------------------------------------ arm C

def init_c(key, cfg):
    k, W, D = cfg['k'], cfg['width'], cfg['depth']
    keys = jax.random.split(key, D + 8)
    sizes = [2 * NF + k] + [W] * D
    dec = [_dense(keys[i], sizes[i], sizes[i + 1], np.sqrt(2.)) for i in range(D)]
    out = _dense(keys[D], W, 1)
    ch = [1, 32, 64, 128, 128]
    conv = [dict(W=jax.random.normal(keys[D + 1 + i], (3, 3, ch[i], ch[i + 1]), jnp.float32) * np.sqrt(2. / (9 * ch[i])),
                 b=jnp.zeros((ch[i + 1],), jnp.float32)) for i in range(4)]
    enc = dict(conv=conv, fc1=_dense(keys[D + 5], 5 * 5 * 128, 512, np.sqrt(2.)), fc2=_dense(keys[D + 6], 512, k))
    return dict(B=jnp.asarray(fourier_matrix(), jnp.float32), dec=dec, out=out, enc=enc)


def decode_c(p, x, z):
    h = jnp.concatenate((gamma(p['B'], x), jnp.broadcast_to(z, (x.shape[0], z.shape[0]))), axis=-1)
    for layer in p['dec']:
        h = jax.nn.gelu(h @ layer['W'] + layer['b'])
    return (h @ p['out']['W'] + p['out']['b'])[:, 0]


def encode_c(p, coarse):
    """coarse: [B, 65, 65] state restricted to every fourth node -> z [B, k]."""
    h = coarse[..., None]
    for layer in p['enc']['conv']:
        h = jax.lax.conv_general_dilated(h, layer['W'], (2, 2), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + layer['b']
        h = jax.nn.gelu(h)
    h = h.reshape(h.shape[0], -1)
    h = jax.nn.gelu(h @ p['enc']['fc1']['W'] + p['enc']['fc1']['b'])
    return h @ p['enc']['fc2']['W'] + p['enc']['fc2']['b']


# ------------------------------------------------------------------ arm L

N_ADAPT = 4


def init_l(key, cfg):
    k, W, H = cfg['k'], cfg['width'], cfg['hyper']
    r = k // N_ADAPT
    assert r * N_ADAPT == k
    keys = jax.random.split(key, 3 * N_ADAPT + 8)
    first = _dense(keys[0], 2 * NF, W, np.sqrt(2.))
    layers = []
    for i in range(N_ADAPT):
        d = _dense(keys[1 + 3 * i], W, W, np.sqrt(2.))
        d['A'] = jax.random.normal(keys[2 + 3 * i], (W, r), jnp.float32) / np.sqrt(W)
        d['Bm'] = jax.random.normal(keys[3 + 3 * i], (r, W), jnp.float32) / np.sqrt(r)
        layers.append(d)
    out = _dense(keys[3 * N_ADAPT + 1], W, 1)
    hs = [6, H, H, H]
    hyper = [_dense(keys[3 * N_ADAPT + 2 + i], hs[i], hs[i + 1], np.sqrt(2.)) for i in range(3)]
    hyper.append(_dense(keys[3 * N_ADAPT + 5], H, k))
    return dict(B=jnp.asarray(fourier_matrix(), jnp.float32), first=first, layers=layers, out=out, hyper=hyper)


def decode_l(p, x, alpha):
    h = jax.nn.gelu(gamma(p['B'], x) @ p['first']['W'] + p['first']['b'])
    a = alpha.reshape(N_ADAPT, -1)
    for i, layer in enumerate(p['layers']):
        h = jax.nn.gelu(h @ layer['W'] + ((h @ layer['A']) * a[i]) @ layer['Bm'] + layer['b'])
    return (h @ p['out']['W'] + p['out']['b'])[:, 0]


def hyper_inputs(t, mu):
    """t [...], mu [..., 5] = (cx, cy, w, a, nu) -> normalised [..., 6]."""
    return jnp.stack((t / .25, mu[..., 0], mu[..., 1], mu[..., 2] / .2, mu[..., 3] / 2.,
                      (jnp.log(mu[..., 4]) - np.log(.01)) / np.log(10.)), axis=-1)


def hyper_l(p, t, mu):
    h = hyper_inputs(t, mu)
    for layer in p['hyper'][:-1]:
        h = jax.nn.gelu(h @ layer['W'] + layer['b'])
    return h @ p['hyper'][-1]['W'] + p['hyper'][-1]['b']


# ------------------------------------------------------------------ shared helpers

def interior_nodes():
    """Flat indices (into the 257^2 grid) and coordinates of the 255^2 interior nodes."""
    i, j = np.meshgrid(np.arange(1, L), np.arange(1, L), indexing='ij')
    return (i * (L + 1) + j).reshape(-1), np.stack((i / L, j / L), axis=-1).reshape(-1, 2)


def node_xy(node):
    return jnp.stack((node // (L + 1), node % (L + 1)), axis=-1) / L


def count(tree):
    return int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(tree)))


def init(key, cfg):
    return init_c(key, cfg) if cfg['arm'] == 'C' else init_l(key, cfg)


def decoder(cfg):
    return decode_c if cfg['arm'] == 'C' else decode_l


CONFIGS = {
    # arm C: name -> (k, width, depth, lr)
    'C16-w128': dict(arm='C', k=16, width=128, depth=4, lr=1e-3),
    'C16-w256': dict(arm='C', k=16, width=256, depth=5, lr=1e-3),
    'C16-w256-lr3': dict(arm='C', k=16, width=256, depth=5, lr=3e-4),
    'C16-w512': dict(arm='C', k=16, width=512, depth=5, lr=1e-3),
    'C16-w512d6-lr3': dict(arm='C', k=16, width=512, depth=6, lr=3e-4),
    'C32-w128': dict(arm='C', k=32, width=128, depth=4, lr=1e-3),
    'C32-w256': dict(arm='C', k=32, width=256, depth=5, lr=1e-3),
    'C32-w512': dict(arm='C', k=32, width=512, depth=5, lr=1e-3),
    # arm L: decoder = input layer + 4 adapted layers; hyper-network 3 hidden layers
    'L16-w256': dict(arm='L', k=16, width=256, hyper=256, lr=1e-3),
    'L16-w256-lr3': dict(arm='L', k=16, width=256, hyper=256, lr=3e-4),
    'L16-w512': dict(arm='L', k=16, width=512, hyper=512, lr=1e-3),
    'L32-w256': dict(arm='L', k=32, width=256, hyper=256, lr=1e-3),
    'L32-w512': dict(arm='L', k=32, width=512, hyper=512, lr=1e-3),
}
