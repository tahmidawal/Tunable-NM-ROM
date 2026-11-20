"""Per-snapshot Adam: only visited embeddings and their clocks advance."""
from common import *


def initialize(codes):
    return dict(m=jnp.zeros_like(codes),v=jnp.zeros_like(codes),count=jnp.zeros(len(codes),jnp.int32))


def update(codes,state,indices,gradient,rate):
    """Indices must be distinct; gradient is only with respect to selected codes."""
    m=.9*state['m'][indices]+.1*gradient
    v=.999*state['v'][indices]+.001*gradient**2
    count=state['count'][indices]+1
    mh=m/(1-.9**count[:,None]);vh=v/(1-.999**count[:,None])
    selected=codes[indices]-rate*mh/(jnp.sqrt(vh)+1e-8)
    return codes.at[indices].set(selected),dict(m=state['m'].at[indices].set(m),v=state['v'].at[indices].set(v),count=state['count'].at[indices].set(count))


def initialize_masked(codes):
    """Independent clocks for coordinates that appear only at some tree depths."""
    return dict(m=jnp.zeros_like(codes), v=jnp.zeros_like(codes),
                count=jnp.zeros(codes.shape, jnp.int32))


def update_masked(codes, state, indices, gradient, rate, mask):
    """Update active coordinates only; indices must be distinct.

    ``mask`` broadcasts to selected rows. Inactive values, moments and clocks
    remain bitwise unchanged even if they accumulated momentum at another depth.
    """
    mask = jnp.broadcast_to(jnp.asarray(mask, dtype=jnp.bool_), gradient.shape)
    old_m, old_v, old_count = state['m'][indices], state['v'][indices], state['count'][indices]
    m = jnp.where(mask, .9*old_m+.1*gradient, old_m)
    v = jnp.where(mask, .999*old_v+.001*gradient**2, old_v)
    count = old_count+mask.astype(jnp.int32)
    # Inactive, never-visited coordinates must not produce a 0/0 intermediate.
    safe_count = jnp.maximum(count, 1)
    mh, vh = m/(1-.9**safe_count), v/(1-.999**safe_count)
    selected = jnp.where(mask, codes[indices]-rate*mh/(jnp.sqrt(vh)+1e-8), codes[indices])
    return codes.at[indices].set(selected), dict(m=state['m'].at[indices].set(m),
        v=state['v'].at[indices].set(v), count=state['count'].at[indices].set(count))
