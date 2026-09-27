"""Enlarge a vector bank without changing its old physical columns.

Terminal outputs are (component, bank-column), not a flat old/new concatenation.
Numerical rank is measured after the Leray projection, not inferred from width.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_model as D
import ns3d_fom as F


def expand(checkpoint, new_rank, seed, n, validation):
    old=checkpoint['params'];previous=np.asarray(checkpoint['extra']['bank']);old_rank=previous.shape[1]
    assert new_rank>old_rank
    width=old['g'][-1][0].shape[0];nf=old['B'].shape[1]
    params=D.init(jax.random.PRNGKey(seed),64,new_rank,width=width,n_ff=nf)
    terminal=np.asarray(params['g'][-1][0]).reshape(width,3,new_rank).copy()
    bias=np.asarray(params['g'][-1][1]).reshape(3,new_rank).copy()
    terminal[:,:,:old_rank]=np.asarray(old['g'][-1][0]).reshape(width,3,old_rank)
    bias[:,:old_rank]=np.asarray(old['g'][-1][1]).reshape(3,old_rank)
    params['g']=[*old['g'][:-1],(terminal.reshape(width,3*new_rank),bias.ravel())]
    params['B']=old['B'];params['out_scale']=old['out_scale']
    params=jax.tree_util.tree_map(jax.device_put,params)
    G=np.asarray(D.bank(params,D.coords(n),F.geometry(n),n))
    difference=G[:,:old_rank]-previous
    relative=float(np.linalg.norm(difference)/np.linalg.norm(previous))
    absolute=float(np.max(abs(difference)))
    assert relative<1e-10
    metadata=dict(previous_rank=old_rank,new_rank=new_rank,hidden_width=width,
        physical_old_column_relative=relative,physical_old_column_max_absolute=absolute,
        physical_old_columns_byte_identical=bool(np.array_equal(G[:,:old_rank],previous)),
        physical_column_tolerance=1e-10,terminal_layout='component,bank-column',
        rank_rule='numerical rank/conditioning of projected bank measured before any capacity acceptance')
    try:
        _,_,_,perp,info=D.whiten(G,validation)
        metadata['initial_whitening']=dict(passed=True,numerical_rank=new_rank,relative_singular_threshold=1e-10,**info)
    except RuntimeError as error:
        metadata['initial_whitening']=dict(passed=False,error=str(error),
            interpretation='rank-deficient initialization may train; no initial full-rank capacity claim')
    coefficient=np.zeros((len(checkpoint['extra']['train_coefficients']),new_rank),dtype=np.float64)
    coefficient[:,:old_rank]=checkpoint['extra']['train_coefficients']
    return params,coefficient,metadata
