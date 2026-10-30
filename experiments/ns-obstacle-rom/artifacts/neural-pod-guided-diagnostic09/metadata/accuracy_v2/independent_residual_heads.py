"""Prospective four independent residual heads; no trainer or released model.

Frozen-parent values/physical gradients are explicit inputs to the training helper.
The live helper evaluates the actual parent at requested points (no TRAIN fallback).
"""
import numpy as np
from accuracy_v2.spatial_bank import initialize, evaluate_layers


def initialize_heads(seed=30102026):
    if type(seed) is not int or seed < 0:
        raise ValueError('explicit nonnegative integer seed required')
    seeds=np.random.SeedSequence(seed).spawn(4)
    heads=[]
    for child in seeds:
        head=initialize(rank=1,width=64,depth=2,seed=int(child.generate_state(1)[0])).layers
        w,b=head[-1]
        head[-1]=(np.zeros_like(w),np.zeros_like(b))
        heads.append(tuple(head))
    return tuple(heads)


def evaluate_heads(heads,points,xp=np):
    if len(heads)!=4:
        raise ValueError('four independent heads required')
    values=[];gradients=[]
    for layers in heads:
        expected=((21,64),(64,64),(64,2))
        if len(layers)!=3 or any(w.shape!=shape or b.shape!=(shape[1],) for (w,b),shape in zip(layers,expected)):
            raise ValueError('each head must have 21-64-64-2 architecture')
        v,g=evaluate_layers(layers,points,xp)
        values.append(v);gradients.append(g)
    return xp.concatenate(values,axis=-1),xp.concatenate(gradients,axis=-1)


def add_to_frozen_parent(parent_values,parent_physical_gradients,heads,points,xp=np):
    v,g=evaluate_heads(heads,points,xp)
    if parent_values.shape!=v.shape or parent_physical_gradients.shape!=g.shape:
        raise ValueError('frozen parent must have four columns in the same physical layout')
    return parent_values+v,parent_physical_gradients+g


def evaluate_live(parent,heads,points,prefix=None):
    if parent.rank!=4:
        raise ValueError('parent rank must remain four')
    x=np.asarray(points,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all():
        raise ValueError('finite physical points required')
    r=4 if prefix is None else prefix
    if type(r) is not int or not 1<=r<=4:
        raise ValueError('invalid prefix')
    pv,pg=parent.evaluate(x)
    v,g=add_to_frozen_parent(pv,pg,heads,x)
    return v[:,:,:r],g[:,:,:,:r]
