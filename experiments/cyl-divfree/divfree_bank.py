"""Exactly divergence-free, exactly no-slip four-mode neural velocity bank (stream function).

Per mode k, in dimensionless coordinates X = x/D (D = 0.1 m):

    psi_k(X) = W(X) * N_k(X) + c_k * chi(X)
    u_k = d psi_k / dY,  v_k = - d psi_k / dX

W = prod_j f_j^2 with f_j = 1 - exp(-d_j / ell), ell = 1 (one diameter), over the
inlet (X = 0), the two channel walls (Y = 0, Y = H/D) and the cylinder (rho = |X - C| - R/D).
W vanishes to second order on those boundaries, so grad(W N) = 0 there: u = 0 exactly.
W does not vanish at the outlet (do-nothing boundary). chi is a compactly supported
C^2 radial bump equal to 1 with zero gradient on the cylinder and identically 0 outside
rho < s, so c_k changes the flux split above/below the cylinder without breaking
no-slip anywhere. The velocity gradient is the (explicitly symmetrised) Hessian of psi,
so div u = H_YX - H_XY = 0 bitwise.

Interface matches accuracy_v2.spatial_bank.SpatialBank as consumed by pilot2d:
``geometry``, ``rank``, ``evaluate(points_physical, prefix) -> (values u/U [n,2,r],
physical gradients d(u/U)/dx [n,2,2,r])``.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from geometry import CHANNEL

SCHEMA = 'cyl-divfree-streamfunction-bank-v1'
D = 0.1
LX, H = (CHANNEL.outlet - CHANNEL.inlet) / D, CHANNEL.height / D        # 22, 4.1
C = jnp.asarray(CHANNEL.center) / D                                       # (2, 2)
R = CHANNEL.obstacle_radius / D                                           # 0.5
ELL = 1.0            # wall-factor length (diameters), same scale as run-09 envelope
CHI_S = 0.8          # chi support half-width beyond the cylinder surface (diameters)
FOURIER = (1., 2., 4., 8.)
GAUSS = (1., 2., 4.)        # = run-09 radial widths 0.1,0.2,0.4 m in diameters
WALL_DECAY = (0.1, 0.3, 1.0)  # boundary-layer features exp(-rho/delta), diameters
N_FEATURES = 2 + 2*2*len(FOURIER) + len(GAUSS) + 5 + len(WALL_DECAY)    # 29


def features(X):
    """Single point X (2,) -> (29,). Run-09 features plus wall-aware cylinder features."""
    z = (X - jnp.asarray([LX/2, H/2])) / jnp.asarray([LX/2, H/2])
    out = [z]
    for f in FOURIER:
        out += [jnp.sin(jnp.pi*f*z), jnp.cos(jnp.pi*f*z)]
    d = X - C; r2 = jnp.sum(d*d); r = jnp.sqrt(r2)
    out.append(jnp.stack([jnp.exp(-r2/(2*w*w)) for w in GAUSS]))
    rho = r - R; c, s = d[0]/r, d[1]/r
    out.append(jnp.stack([jnp.tanh(rho/2.), c, s, c*c - s*s, 2*c*s]))
    out.append(jnp.stack([jnp.exp(-rho/dl) for dl in WALL_DECAY]))
    return jnp.concatenate(out)


def wall_factor(X):
    d = X - C; rho = jnp.sqrt(jnp.sum(d*d)) - R
    dist = jnp.stack([X[0], X[1], H - X[1], rho])
    f = -jnp.expm1(-dist/ELL)
    return jnp.prod(f*f)


def chi(X):
    d = X - C; rho = jnp.sqrt(jnp.sum(d*d)) - R; t = rho/CHI_S
    return jnp.where(t < 1., (1. - jnp.minimum(t, 1.)**2)**3, 0.)


def network(params, X):
    h = features(X)
    for i, (w, b) in enumerate(params['layers']):
        h = h @ w + b
        if i < len(params['layers']) - 1:
            h = jnp.tanh(h)
    return h  # (4,)


def psi(params, X):
    return wall_factor(X)*network(params, X) + chi(X)*params['c']


def _point(params, X):
    """Returns velocity (2,4) and dimensionless velocity gradient (2 comp, 2 deriv, 4)."""
    g = lambda Y: jax.jacfwd(psi, argnums=1)(params, Y)          # (4, 2)
    Hs, G = jax.jacfwd(lambda Y: (g(Y), g(Y)), has_aux=True)(X)  # (4,2,2), (4,2)
    Hs = 0.5*(Hs + jnp.swapaxes(Hs, 1, 2))                       # exact symmetry -> div 0 bitwise
    vel = jnp.stack([G[:, 1], -G[:, 0]])                         # (2, 4)
    grad = jnp.stack([jnp.stack([Hs[:, 1, 0], Hs[:, 1, 1]]),     # du/dX, du/dY
                      jnp.stack([-Hs[:, 0, 0], -Hs[:, 0, 1]])])  # dv/dX, dv/dY
    return vel, grad                                             # (2,4), (2,2,4)


fields = jax.vmap(_point, in_axes=(None, 0))   # X (n,2) dimensionless -> (n,2,4),(n,2,2,4)


def initialize(width=160, depth=3, seed=3112026):
    rng = np.random.default_rng(seed); sizes = [N_FEATURES] + [width]*depth + [4]
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        scale = 1./np.sqrt(a) if i < depth else 0.1/np.sqrt(a)
        layers.append((rng.normal(size=(a, b))*scale, np.zeros(b)))
    return {'layers': layers, 'c': np.zeros(4)}


def flatten(params):
    arrays = [a for w, b in params['layers'] for a in (w, b)] + [params['c']]
    shapes = [a.shape for a in arrays]
    return np.concatenate([np.ravel(a) for a in arrays]), shapes


def unflatten(theta, shapes):
    out = []; o = 0
    for s in shapes:
        n = int(np.prod(s)); out.append(theta[o:o+n].reshape(s)); o += n
    layers = list(zip(out[:-1:2], out[1:-1:2]))
    return {'layers': layers, 'c': out[-1]}


_fields_jit = jax.jit(fields)


class DivFreeBank:
    geometry = CHANNEL
    rank = 4

    def __init__(self, params, metadata):
        self.params = {'layers': [(np.asarray(w, np.float64), np.asarray(b, np.float64)) for w, b in params['layers']],
                       'c': np.asarray(params['c'], np.float64)}
        self.metadata = dict(metadata)
        if self.params['layers'][0][0].shape[0] != N_FEATURES or self.params['layers'][-1][0].shape[1] != 4:
            raise ValueError('invalid architecture')
        if not all(np.isfinite(a).all() for w, b in self.params['layers'] for a in (w, b)):
            raise ValueError('nonfinite parameters')

    def evaluate(self, points, prefix=None, block=16384):
        x = np.asarray(points, dtype=np.float64); r = 4 if prefix is None else int(prefix)
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all() or not 1 <= r <= 4:
            raise ValueError('invalid evaluation')
        p = jax.tree_util.tree_map(jnp.asarray, self.params)
        vals = np.empty((len(x), 2, 4)); grads = np.empty((len(x), 2, 2, 4))
        for i in range(0, len(x), block):
            v, g = _fields_jit(p, jnp.asarray(x[i:i+block]/D))
            vals[i:i+block] = np.asarray(v); grads[i:i+block] = np.asarray(g)
        # values u/U; gradients converted to physical coordinates (pilot2d multiplies by D)
        return vals[..., :r], grads[..., :r]/D

    def values_blocked(self, points, block_points=4096):
        return self.evaluate(points, 4, block_points)[0]

    def save(self, path):
        m = dict(self.metadata, schema=SCHEMA, geometry=CHANNEL.as_dict(), ell=ELL, chi_support=CHI_S,
                 fourier=FOURIER, gauss=GAUSS, wall_decay=WALL_DECAY, n_features=N_FEATURES,
                 layers=len(self.params['layers']), value_units='u/U', gradient_coordinates='physical')
        a = {'metadata_json': np.array(json.dumps(m, sort_keys=True)), 'c': self.params['c']}
        for i, (w, b) in enumerate(self.params['layers']):
            a[f'W{i}'] = w; a[f'b{i}'] = b
        np.savez_compressed(path, **a)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            m = json.loads(str(a['metadata_json']))
            if m.get('schema') != SCHEMA or m['ell'] != ELL or m['chi_support'] != CHI_S or m['n_features'] != N_FEATURES:
                raise ValueError('unknown divergence-free bank contract')
            layers = [(a[f'W{i}'].copy(), a[f'b{i}'].copy()) for i in range(m['layers'])]
            return cls({'layers': layers, 'c': a['c'].copy()}, m)
