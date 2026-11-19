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
