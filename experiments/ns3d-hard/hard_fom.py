"""Family-parametrised periodic NS3D initial data and spectral comparison tools.

The solver is experiments/ns3d/ns3d_fom.py unchanged. Only the initial-condition
family is generalised: potential sharpness beta (paper: 2), amplitude, separation
range, viscosity range and horizon. Family 'A' reproduces ns3d_fom.parameters /
ns3d_fom.initial_raw exactly (checked in characterise.py).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ns3d"))
import ns3d_fom as F  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def _fam(beta, nu, horizon):
    s = float(np.sqrt(2.0 / beta))          # blob width scale relative to the paper
    return dict(beta=float(beta), amp=0.2 * s, sep=(0.12 * s, 0.24 * s),
                strength=(0.6, 1.4), nu=tuple(nu), horizon=float(horizon))


FAMILIES = {
    "A": _fam(2, (2e-3, 1e-2), 0.2),        # the paper's family
    "B": _fam(8, (5e-4, 2.5e-3), 0.2),
    "C": _fam(16, (2.5e-4, 1.25e-3), 0.2),
    "D": _fam(8, (2.5e-4, 1.25e-3), 0.4),
    "E": _fam(8, (1e-4, 5e-4), 0.4),
}


def parameters(seed, count, fam):
    """Rows [cx,cy,cz,sep,strength,nu]; same draw order as ns3d_fom.parameters."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        center = rng.uniform(0, 1, 3)
        rows.append([*center, rng.uniform(*fam["sep"]), rng.uniform(*fam["strength"]),
                     np.exp(rng.uniform(np.log(fam["nu"][0]), np.log(fam["nu"][1])))])
    return np.asarray(rows, dtype=np.float64)


def initial_raw(n, parameter, fam):
    x = np.arange(n, dtype=np.float64) / n
    xyz = np.stack(np.meshgrid(x, x, x, indexing="ij"))
    c, sep, strength = parameter[:3], parameter[3], parameter[4]
    offset = np.array([1., 1., 1.]) / np.sqrt(3)
    directions = (np.array([0., 0., 1.]), np.array([1., 1., 0.]) / np.sqrt(2))
    beta = fam["beta"]
    u = np.zeros((3, n, n, n), dtype=np.float64)
    for side, weight, direction in ((-1, 1., directions[0]), (1, strength, directions[1])):
        center = c + side * .5 * sep * offset
        angle = 2 * np.pi * (xyz - center[:, None, None, None])
        scalar = np.exp(beta * np.sum(np.cos(angle) - 1, axis=0))
        grad = -2 * np.pi * beta * np.sin(angle) * scalar
        u += weight * np.moveaxis(np.cross(np.moveaxis(grad, 0, -1), direction), -1, 0)
    return fam["amp"] * u


def initial(n, parameter, fam):
    return np.asarray(F.project_field(jnp.asarray(initial_raw(n, parameter, fam)), F.geometry(n)))


# ---------------------------------------------------------------- spectral tools --

def sub_index(n, big):
    """Indices of the n-mesh integer modes inside a big-mesh FFT axis."""
    return (np.fft.fftfreq(n, d=1.0 / n).astype(int)) % big


def keep_mask(n):
    """Modes strictly below the n-mesh Nyquist (the Nyquist plane is dropped)."""
    m = np.abs(np.fft.fftfreq(n, d=1.0 / n)) < n / 2
    return m[:, None, None] & m[None, :, None] & m[None, None, :]


@jax.jit
def _take(hat, idx):
    return hat[..., idx[:, None, None], idx[None, :, None], idx[None, None, :]]


def truncate_hat(hat_big, n):
    """Fourier coefficients (forward norm) of the big field restricted to mesh n."""
    big = hat_big.shape[-1]
    if n == big:
        return hat_big
    return _take(hat_big, jnp.asarray(sub_index(n, big))) * jnp.asarray(keep_mask(n))


def truncate_field(u_big, n):
    """Physical field on mesh n holding the |k|<n/2 Fourier content of u_big."""
    return F.ifft(truncate_hat(F.fft(jnp.asarray(u_big)), n))


@jax.jit
def _err2(pred_hat, ref_sub, ref_tot2):
    """Squared full-field L2 error per frame of a band-limited prediction."""
    d = jnp.sum(jnp.abs(pred_hat - ref_sub) ** 2, axis=(1, 2, 3, 4))
    return d + ref_tot2 - jnp.sum(jnp.abs(ref_sub) ** 2, axis=(1, 2, 3, 4))


def full_errors(pred, ref_hat, ref_tot2, u0_norm2):
    """Relative L2 error on the reference mesh of the trigonometric interpolant
    of pred (T,3,n,n,n). ref_hat (T,3,N,N,N) forward-norm coefficients."""
    n = pred.shape[-1]
    ref_sub = truncate_hat(ref_hat, n)
    e2 = _err2(F.fft(jnp.asarray(pred)) * jnp.asarray(keep_mask(n)), ref_sub, ref_tot2)
    return np.sqrt(np.maximum(np.asarray(e2), 0.0) / u0_norm2)
