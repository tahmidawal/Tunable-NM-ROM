"""Verbatim excerpt of `output_field` from the Burgers lane's `mr-burgers2d/engines.py`.

Staged so the cluster job can check this lane's NumPy prolongation against the exact
function the Burgers lane uses to lift its coarse-mesh FOM arms onto the evaluation
grid. Nothing here is modified; the source file's SHA256 is recorded below and the
excerpt is asserted byte-identical to those lines at stage time.

SOURCE: /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/mr-burgers2d/engines.py
SOURCE_SHA256 = "820039e5840abaf92dc911f9de8c6a427f97cbd2617f1e7a5169e66c19fa62e4"
SOURCE_LINES = 73-83
"""
import jax.numpy as jnp


def output_field(u, L, target):
    a = jnp.pad(u.reshape(L-1,L-1),1)
    if target == L:
        return a
    # Aligned bilinear interpolation: coarse/fine boundaries and nested nodes exact.
    x = jnp.arange(target+1,dtype=jnp.float64)*L/target
    lo = jnp.minimum(x.astype(jnp.int32),L-1); f=x-lo
    a = (1-f[:,None])*a[lo,:] + f[:,None]*a[lo+1,:]
    return (1-f[None,:])*a[:,lo] + f[None,:]*a[:,lo+1]
