"""Exact nested restriction and physically aligned coarse heat output kernels."""
import hashlib

import heat_core as hc
import jax
import jax.numpy as jnp
import numpy as np


def field_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()


def restrict_input(u0, requested, solver):
    assert requested % solver == 0 and u0.shape == (requested-1, requested-1)
    stride = requested//solver
    return np.ascontiguousarray(u0[stride-1::stride, stride-1::stride])


def interpolation_tables(requested, solver):
    assert requested % solver == 0
    positions = np.arange(1, requested, dtype=np.float64)/(requested//solver)
    indices = np.floor(positions).astype(np.int32)
    return indices, positions-indices


def interpolate(fields, indices, weights):
    """Interior nodes at i/N, with known zero boundary; never pixel-center resizing."""
    padded = jnp.pad(fields, ((0, 0), (1, 1), (1, 1)))
    along_x = padded[:, indices, :]*(1-weights)[None, :, None]+padded[:, indices+1, :]*weights[None, :, None]
    return along_x[:, :, indices]*(1-weights)[None, None, :]+along_x[:, :, indices+1]*weights[None, None, :]


def make_fom(requested, solver):
    @jax.jit
    def evolved(u0, lam, later_times, nu, indices, weights):
        spectrum = hc.dst2(u0)
        fields = jax.vmap(lambda t: hc.dst2(spectrum*jnp.exp(-nu*t*lam)))(later_times)
        return fields if requested == solver else interpolate(fields, indices, weights)
    return evolved


def host_outputs(u0, evolved):
    """This complete contiguous allocation/copy is inside the full query timer."""
    outputs = np.empty((len(evolved)+1, *u0.shape), dtype=np.float64)
    outputs[0] = u0
    outputs[1:] = np.asarray(evolved)
    return outputs


def unique_solvers(requested, candidates):
    return sorted(set([requested]+[n for n in candidates if n <= requested]))
