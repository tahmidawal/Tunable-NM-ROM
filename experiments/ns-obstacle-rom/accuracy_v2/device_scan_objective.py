"""Prospective dynamic-block reduction; not integrated into any production fit.

Caller supplies a nonempty pytree of equally sized device block stacks. The last
block must be padded by the caller with finite dummy values and zero quadrature
weights. Padding must also keep all intermediate feature/derivative operations
finite: zero weights cannot suppress NaNs. Metrics must be additive block
statistics; global norms/RMS/ratios must be computed after reduction. The
gradient is one flat array, not a generic pytree. There is no captured data or reverse differentiation through the scan:
value_and_grad is applied to each block inside its body. The returned gradient
is already the full parameter gradient; do not differentiate this reduction.
"""

def reduce_blocks(theta, stacked_blocks, block_value_and_grad):
    import jax
    leaves=jax.tree_util.tree_leaves(stacked_blocks)
    if not leaves or not leaves[0].shape or leaves[0].shape[0]<1:raise ValueError('nonempty dynamic block stack required')
    count=leaves[0].shape[0]
    if any(not x.shape or x.shape[0]!=count for x in leaves):raise ValueError('common block count required')
    first=jax.tree_util.tree_map(lambda x:x[0],stacked_blocks)
    (value,metrics),gradient=block_value_and_grad(theta,first)
    def step(carry,block):
        (v,m),g=block_value_and_grad(theta,block)
        old_v,old_g,old_m=carry
        return (old_v+v,old_g+g,jax.tree_util.tree_map(lambda a,b:a+b,old_m,m)),None
    rest=jax.tree_util.tree_map(lambda x:x[1:],stacked_blocks)
    (value,gradient,metrics),_=jax.lax.scan(step,(value,gradient,metrics),rest)
    return (value,metrics),gradient
