"""Deterministic positive empirical quadrature; no random PDE collocation."""
import time
import numpy as np
import scipy.optimize
import scipy.linalg


def solve_nnls(design,target,*,method='direct',maxiter=None):
    """Exact fixed-support positive least squares, optionally after economy QR.

    For a tall matrix A=QR, the original squared objective equals
    ||Rw-Q.T b||² plus the constant ||(I-QQ.T)b||². The QR path preserves the
    feasible set and objective without truncation or regularization. Nonunique
    optima or floating-point ties may produce different weights/supports.
    Returned norm is always ||Aw-b|| on
    the original matrix, never the smaller transformed residual norm.
    """
    A=np.asarray(design,dtype=np.float64);b=np.asarray(target,dtype=np.float64)
    assert A.ndim==2 and b.shape==(A.shape[0],)
    if method=='direct':
        return scipy.optimize.nnls(A,b,maxiter=maxiter)
    if method!='qr':raise ValueError(f'unknown NNLS method {method}')
    if A.shape[0]<A.shape[1]:raise ValueError('economy-QR NNLS requires a tall or square fixed-support matrix')
    Q,R=scipy.linalg.qr(A,mode='economic',overwrite_a=False,check_finite=True)
    w,_=scipy.optimize.nnls(R,Q.T@b,maxiter=maxiter)
    return w,float(np.linalg.norm(A@w-b))


def nnls_diagnostics(design,target,weights):
    """Original-system objective and positive least-squares KKT diagnostics."""
    A=np.asarray(design,dtype=np.float64);b=np.asarray(target,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    residual=A@w-b;gradient=A.T@residual
    scale=max(float(np.linalg.norm(A)*np.linalg.norm(b)),1e-30)
    active=w>1e-12*max(float(np.max(w)),1e-30)
    active_error=float(np.max(np.abs(gradient[active]))) if active.any() else 0.
    inactive_error=float(np.max(np.maximum(-gradient[~active],0.))) if (~active).any() else 0.
    return dict(objective=float(residual@residual),residual_norm=float(np.linalg.norm(residual)),
        relative_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(b),1e-30)),
        minimum_weight=float(np.min(w)),positive_nodes=int(np.sum(active)),
        active_gradient_max=active_error,inactive_negative_gradient_max=inactive_error,
        scaled_kkt=max(active_error,inactive_error)/scale,
        complementarity_max=float(np.max(np.abs(w*gradient))))


def fit_quadrature(design,target,m,*,candidate_ids=None,nnls_method='direct'):
    """Fit already row-scaled decoder-output integrands on candidate columns.

    design and target must describe the SAME full-grid weighted integral;
    returned weights multiply candidate integrands. A constant row belongs in
    the caller's design. Extra selected nodes have exactly zero weight.
    """
    A=np.asarray(design,dtype=np.float64);b=np.asarray(target,dtype=np.float64)
    assert A.ndim==2 and b.shape==(A.shape[0],) and np.isfinite(A).all() and np.isfinite(b).all()
    assert m<=A.shape[1]
    if nnls_method not in ('direct','qr'):raise ValueError(f'unknown NNLS method {nnls_method}')
    if nnls_method=='qr' and m>A.shape[0]:raise ValueError('QR fit requires target support no wider than the row count')
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
        weights,_=solve_nnls(A[:,support],b,method=nnls_method,maxiter=max(1000,20*len(support)))
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
    info['nnls_method']=nnls_method
    return (ids if candidate_ids is None else np.asarray(candidate_ids)[ids]),weights,info
