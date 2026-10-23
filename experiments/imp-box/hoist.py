"""imp-box A16: compile queries with their case-independent arrays as executable ARGUMENTS instead of embedded HLO
literals. JAX 0.10 embeds concrete arrays closed over by a jitted function as constants; at A3/A4 those reach hundreds of
MB to GB per query, making every compilation slow and the persistent cache unable to store the executables (seen as
the smk2-A4 / W28_A3 time-stage stalls). `hoisted(q)` traces q once to a jaxpr, passes its constants as arguments of a
jitted evaluator and returns a callable of the descriptor d alone (the timing contract is unchanged: every timed call
runs the same compiled program, all case-dependent work inside it).
IMPBOX_EMBED=1 restores the previous embedded behaviour (used by the G-HOIST control only)."""
from __future__ import annotations

import os

import jax
import jax.core as jcore
import jax.numpy as jnp

D_EXAMPLE = jax.ShapeDtypeStruct((5,), jnp.float64)


class Hoisted:
    def __init__(self, f, example=D_EXAMPLE):
        cj = jax.make_jaxpr(f)(example)
        self.consts = [jnp.asarray(c) for c in cj.consts]
        self._g = jax.jit(lambda d, *cs: jcore.eval_jaxpr(cj.jaxpr, cs, d))
        self._treedef = jax.tree_util.tree_structure(jax.eval_shape(f, example))
        self.info = {}

    def __call__(self, d):
        return jax.tree_util.tree_unflatten(self._treedef, self._g(d, *self.consts))

    def _cache_size(self):
        return self._g._cache_size()


def hoisted(f, embed=None, example=D_EXAMPLE):
    embed = os.environ.get('IMPBOX_EMBED') == '1' if embed is None else embed
    if embed:
        return jax.jit(f)
    return Hoisted(f, example)
