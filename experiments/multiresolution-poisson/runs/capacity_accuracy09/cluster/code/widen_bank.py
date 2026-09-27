"""Function-preserving bank widening; no dependence on solution snapshots."""
import copy
import numpy as np


def widen(params,new_rank,bank_evaluator):
    p=copy.deepcopy(params);old_rank=p['h_lin'].shape[1]
    assert old_rank<new_rank and np.asarray(p['h_lin']).dtype==np.float64
    gw,gb=map(np.asarray,p['g'][-1]);affine=np.vstack((gw,gb));singular=np.linalg.svd(affine,compute_uv=False)
    threshold=singular[0]*max(affine.shape)*np.finfo(float).eps
    assert np.count_nonzero(singular>threshold)==old_rank and new_rank<=affine.shape[0]
    orthogonal=np.linalg.qr(affine,mode='complete')[0][:,old_rank:new_rank]
    p['g'][-1]=(np.concatenate((gw,orthogonal[:-1]),axis=1),np.concatenate((gb,orthogonal[-1])))
    for key in ['h']:
        hw,hb=map(np.asarray,p[key][-1]);p[key][-1]=(np.pad(hw,((0,0),(0,new_rank-old_rank))),np.pad(hb,(0,new_rank-old_rank)))
    p['h_lin']=np.pad(np.asarray(p['h_lin']),((0,0),(0,new_rank-old_rank)))
    G=np.asarray(bank_evaluator(p));norms=np.linalg.norm(G,axis=0);assert np.all(norms>0)
    scales=np.median(norms[:old_rank])/norms[old_rank:]
    gw,gb=p['g'][-1];gw[:,old_rank:]*=scales;gb[old_rank:]*=scales
    return p,dict(old_rank=old_rank,new_rank=new_rank,affine_initial_singular_values=singular.tolist(),complement=orthogonal.tolist(),training_grid_new_column_scales=scales.tolist(),construction='complete QR affine-output complement with training-coordinate feature-norm scaling; appended head/skip coefficients are zero')
