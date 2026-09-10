"""Deterministic positive empirical quadrature; no random PDE collocation."""
import time
import numpy as np
import scipy.optimize


def fit_quadrature(design,target,m,*,candidate_ids=None):
    """Fit already row-scaled decoder-output integrands on candidate columns.

    design and target must describe the SAME full-grid weighted integral;
    returned weights multiply candidate integrands. A constant row belongs in
    the caller's design. Extra selected nodes have exactly zero weight.
    """
    A=np.asarray(design,dtype=np.float64);b=np.asarray(target,dtype=np.float64)
    assert A.ndim==2 and b.shape==(A.shape[0],) and np.isfinite(A).all() and np.isfinite(b).all()
    assert m<=A.shape[1]
    start=time.perf_counter();support=[];weights=np.empty(0);r=b.copy()
    # Block additions reduce expensive host NNLS iterations without changing the
    # positive least-squares objective. Every active support gets a final refit.
    norms=np.maximum(np.linalg.norm(A,axis=0),1e-30)
    for _ in range(4*m):
        if len(support)>=m:break
        score=(A.T@r)/norms
        if support:score[support]=-np.inf
        if np.max(score)<=1e-12*max(np.linalg.norm(b),1e-30):break
        count=min(8,m-len(support))
        order=np.argsort(-score,kind='stable')[:count]
        order=order[score[order]>0]
        support.extend(order.tolist())
        weights,_=scipy.optimize.nnls(A[:,support],b,maxiter=max(1000,20*len(support)))
        active=weights>1e-14
        support=list(np.asarray(support)[active]);weights=weights[active]
        new=b-A[:,support]@weights
        if np.linalg.norm(new)<=1e-12*max(np.linalg.norm(b),1e-30):r=new;break
        # A nonpositive new column can be reselected forever. Deterministic
        # termination makes this deficiency visible in achieved active support.
        if abs(np.linalg.norm(r)-np.linalg.norm(new))<=1e-13*max(np.linalg.norm(b),1e-30):r=new;break
        r=new
    active_count=len(support)
    if len(support)<m:
        remaining=np.setdiff1d(np.arange(A.shape[1]),np.asarray(support,dtype=int))
        chosen=remaining[np.argsort(-norms[remaining],kind='stable')[:m-len(support)]]
        support.extend(chosen.tolist())
        weights=np.pad(weights,(0,len(chosen)))
    ids=np.asarray(support,dtype=int)
    info=dict(target_nodes=m,selected_nodes=len(ids),positive_nodes=active_count,
              relative_fit=float(np.linalg.norm(A[:,ids]@weights-b)/max(np.linalg.norm(b),1e-30)),
              maximum_scaled_row_error=float(np.max(np.abs(A[:,ids]@weights-b))),
              seconds=time.perf_counter()-start,weight_sum=float(weights.sum()),
              minimum_weight=float(weights.min()),candidate_count=A.shape[1],fit_rows=A.shape[0])
    return (ids if candidate_ids is None else np.asarray(candidate_ids)[ids]),weights,info
