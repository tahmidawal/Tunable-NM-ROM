"""Bounded direct-coefficient candidates and exact zero-child transfer.

This module does not select thresholds, run a trajectory or claim adaptation
qualification. Costs are explicit supplied estimates, not inferred speed wins.
"""
from common import np,jnp
import geometry as geo


def transfer(coefficients,old_cells,old_rank,new_cells,new_rank,root):
    old_cells,new_cells=list(map(tuple,old_cells)),list(map(tuple,new_cells))
    geo.valid(old_cells);geo.valid(new_cells)
    assert set(old_cells)<=set(new_cells) and new_rank>=old_rank
    assert len(coefficients)==root+len(old_cells)*old_rank
    result=jnp.zeros(root+len(new_cells)*new_rank,dtype=coefficients.dtype)
    result=result.at[:root].set(coefficients[:root])
    lookup={c:i for i,c in enumerate(new_cells)}
    for i,cell in enumerate(old_cells):
        j=lookup[cell]
        result=result.at[root+j*new_rank:root+j*new_rank+old_rank].set(coefficients[root+i*old_rank:root+(i+1)*old_rank])
    return result


def candidates(cells,rank,root,scores,*,depth,max_variables,next_rank,split_costs,rank_cost):
    """Top two sibling-group additions, one rank increase and global fallback."""
    cells=list(map(tuple,cells));geo.valid(cells)
    proposals=[]
    if root+(len(cells)+4)*rank<=max_variables:
        leaves=[c for c in geo.leaves(cells) if c[0]<depth]
        def score(c):
            assert c in scores and c in split_costs and split_costs[c]>0
            return scores[c]**2/split_costs[c]
        for leaf in sorted(leaves,key=score,reverse=True)[:2]:
            proposals.append(dict(kind='split',leaf=leaf,cells=cells+geo.children(leaf),rank=rank,
                estimated_cost=split_costs[leaf],estimated_reduction_per_cost=score(leaf)))
    if cells and next_rank>rank and root+len(cells)*next_rank<=max_variables:
        assert rank_cost>0
        proposals.append(dict(kind='rank',cells=cells,rank=next_rank,estimated_cost=rank_cost))
    proposals.append(dict(kind='global_fallback',recompute_from_initial=True))
    assert len(proposals)<=4
    return proposals


def admissible_trial(before,after,estimated_cost,cost_cap,*,solver_converged,finite):
    assert before>=0 and estimated_cost>=0 and cost_cap>=0
    return bool(solver_converged and finite and np.isfinite(after) and after>=0
                and after<=.8*before and estimated_cost<=cost_cap)
