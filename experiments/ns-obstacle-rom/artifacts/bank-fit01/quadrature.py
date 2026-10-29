"""Positive decoder-output moment fitting. This is not a PDE residual snapshot fit.

Only the quadratic convection tensor is sampled in the first candidate; all linear
operators, pressure/divergence, test Grams and force moments stay on the full rule.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from spaces import positive_weights


def output_moments(rule,training_velocity_coefficients,points,max_bytes=2**30):
    """Moment rows: geometry, u, grad(u), and tested products u_j*d_j(u_i).

    No viscosity, pressure cancellation, PDE residual or validation state enters this fit.
    Reconstructed training-state nonlinear output moments are allowed; provenance must
    identify those states. Jacobian accuracy is independently checked, not assumed.
    """
    a=np.asarray(training_velocity_coefficients,dtype=np.float64)
    x=np.asarray(points,dtype=np.float64)
    n,d=rule.lift.shape
    if a.ndim!=2 or a.shape[1]!=rule.velocity.shape[-1] or x.shape!=(n,d):
        raise ValueError("output moment shapes mismatch")
    rows=1+d+len(a)*(d+d*d+rule.tests.shape[-1])
    if rows*n*8>max_bytes:
        raise MemoryError("moment fit matrix exceeds declared cap; no silent subsampling")
    blocks=[np.ones((1,n)),x.T]
    for c in a:
        u=rule.lift+np.einsum("ndr,r->nd",rule.velocity,c)
        grad=rule.lift_gradient+np.einsum("ndjr,r->ndj",rule.velocity_gradient,c)
        adv=np.einsum("nj,nij->ni",u,grad)
        tested=np.einsum("ni,nim->mn",adv,rule.tests)
        blocks.extend((u.T,grad.reshape(n,-1).T,tested))
    return np.vstack(blocks)


def fit_positive(moments,full_weights,max_support,relative_tolerance=1e-8,kkt_tolerance=1e-8):
    """Deterministic bounded active NNLS; one support cap, never automatic resubmission."""
    from scipy.optimize import nnls
    f=np.asarray(moments,dtype=np.float64)
    if f.ndim!=2 or not np.isfinite(f).all():
        raise ValueError("finite moments matrix required")
    w=positive_weights(full_weights,f.shape[1])
    if not 1<=max_support<=f.shape[1]:
        raise ValueError("invalid support cap")
    scale=np.maximum(np.sum(np.abs(f)*w,axis=1),1e-12)
    matrix=f/scale[:,None]
    target=matrix@w
    target_norm=max(float(np.linalg.norm(target)),1e-12)
    norm=np.linalg.norm(matrix,axis=0)
    selected=[]
    weights=np.empty(0)
    history=[]
    residual=target.copy()
    reason="support_cap"
    for count in range(max_support):
        corr=matrix.T@residual
        score=np.divide(corr,norm,out=np.full_like(corr,-np.inf),where=norm>0)
        score[selected]=-np.inf
        choice=int(np.argmax(score))
        if not np.isfinite(score[choice]) or score[choice]<=0:
            reason="no_positive_correlation";break
        selected.append(choice)
        try:
            weights,_=nnls(matrix[:,selected],target,maxiter=max(100,10*len(selected)))
        except RuntimeError:
            reason="nnls_iteration_limit";break
        residual=target-matrix[:,selected]@weights
        relative=float(np.linalg.norm(residual)/target_norm)
        history.append({"support":len(selected),"relative_moment_residual":relative})
        if relative<=relative_tolerance:
            reason="fit_tolerance";break
    # Preserve selected zero weights in diagnostic support; deploy strictly positive only.
    if len(weights)!=len(selected):
        return {"passed":False,"reason":reason,"selected":selected,"history":history,
                "relative_moment_residual":None,"kkt":None}
    gradient=-matrix[:,selected].T@residual
    positive=weights>0
    stationarity=float(np.max(np.abs(gradient[positive]))) if np.any(positive) else float("inf")
    dual=float(max(0.,-np.min(gradient[~positive]))) if np.any(~positive) else 0.
    complement=float(np.max(np.abs(weights*gradient))) if len(weights) else float("inf")
    normalizer=max(float(np.linalg.norm(matrix[:,selected],2))*target_norm,1e-12) if len(selected) else 1.
    kkt=max(stationarity,dual,complement)/normalizer
    relative=float(np.linalg.norm(residual)/target_norm)
    return {"passed":bool(reason=="fit_tolerance" and kkt<=kkt_tolerance),"reason":reason,
            "selected":selected,"selected_weights":weights.tolist(),
            "support":np.asarray(selected)[positive].tolist(),"weights":weights[positive].tolist(),
            "history":history,"relative_moment_residual":relative,"kkt":kkt,
            "kkt_scope":"restricted selected-support NNLS; no full-dictionary optimality claim",
            "max_support":max_support,"relative_tolerance":relative_tolerance,
            "kkt_tolerance":kkt_tolerance,"row_scaling":scale.tolist(),
            "training_scope":"decoder output/gradient moments only"}


def convection_candidate(full_operator,rule,fit):
    if not fit.get("passed",False):
        raise ValueError("refusing to deploy failed positive rule")
    support=np.asarray(fit["support"],dtype=int)
    w=positive_weights(fit["weights"],len(support))
    u=np.concatenate((rule.lift[:,:,None],rule.velocity),axis=2)[support]
    du=np.concatenate((rule.lift_gradient[:,:,:,None],rule.velocity_gradient),axis=3)[support]
    tensor=np.einsum("n,nim,nja,nijb->mab",w,rule.tests[support],u,du,optimize=True)
    return replace(full_operator,convection=tensor)
