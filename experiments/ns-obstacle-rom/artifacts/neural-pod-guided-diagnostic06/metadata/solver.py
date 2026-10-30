"""Small CPU weak-LS solve, named explicitly; convergence is not PDE admission."""
from __future__ import annotations
import time
import numpy as np


def solve_steady(operator,nu_hat,initial=None,max_iterations=50,gradient_tolerance=1e-9,
                 damping=1e-8,max_backtracks=12,root_tolerance=1e-9):
    z=np.zeros(operator.nv+operator.np) if initial is None else np.array(initial,dtype=np.float64,copy=True)
    history=[]
    start=time.perf_counter()
    calls=0
    initial_gradient=None
    status="iteration_limit"
    for iteration in range(max_iterations+1):
        residual=operator.residual(z,nu_hat)
        calls+=1
        jac=operator.jacobian(z,nu_hat,columns=True)
        sv=np.linalg.svd(jac,compute_uv=False)
        rank=int(np.sum(sv>1e-12*sv[0])) if len(sv) and sv[0]>0 else 0
        condition=float(sv[0]/sv[-1]) if rank==len(sv) else float("inf")
        gradient=jac.T@residual
        gnorm=float(np.linalg.norm(gradient))
        if initial_gradient is None:
            initial_gradient=max(gnorm,1.)
        objective=float(.5*residual@residual)
        row={"iteration":iteration,"objective":objective,"gradient_norm":gnorm,
             "gradient_denominator":initial_gradient,"condition":condition,"rank":rank,
             "damping":damping,"residual_norm":float(np.linalg.norm(residual))}
        history.append(row)
        if not (np.isfinite(residual).all() and np.isfinite(jac).all()):
            status="nonfinite";break
        if rank!=len(z) or condition>1e10:
            status="jacobian_condition_failure";break
        if operator.formulation=="mixed_Galerkin":
            if np.linalg.norm(residual)<=root_tolerance:
                status="root_converged";break
        elif gnorm<=gradient_tolerance*initial_gradient:
            status="stationary";break
        if iteration==max_iterations:
            break
        augmented=np.r_[jac,np.sqrt(damping)*np.eye(len(z))]
        right=np.r_[-residual,np.zeros(len(z))]
        dy,_,_,_=np.linalg.lstsq(augmented,right,rcond=1e-12)
        step=operator.unknown_map@dy
        slope=float(gradient@dy)
        accepted=False
        for bt in range(max_backtracks):
            alpha=2.**(-bt)
            trial=z+alpha*step
            rr=operator.residual(trial,nu_hat)
            calls+=1
            if np.isfinite(rr).all() and .5*rr@rr<=objective+1e-4*alpha*slope:
                z=trial;accepted=True;break
        row.update(accepted=accepted,alpha=alpha,backtracks=bt)
        if not accepted:
            status="line_search_failure";break
    return {"state":z,"status":status,"stationary":status=="stationary", "root_converged":status=="root_converged","history":history,
            "residual_evaluations":calls,"jacobian_evaluations":len(history),
            "seconds":time.perf_counter()-start,"method":operator.formulation,
            "physical_admission":False}


def jacobian_check(operator,state,nu_hat,direction,steps=(1e-4,1e-5,1e-6)):
    """Whitened coefficient direction, independent centered difference; signal retained."""
    v=np.asarray(direction,dtype=np.float64)
    if np.linalg.norm(v)==0:
        raise ValueError("nonzero direction required")
    v=v/np.linalg.norm(v)
    dz=operator.unknown_map@v
    analytic=operator.jacobian(state,nu_hat,columns=True)@v
    results=[]
    for h in steps:
        plus=operator.residual(state+h*dz,nu_hat)
        minus=operator.residual(state-h*dz,nu_hat)
        fd=(plus-minus)/(2*h)
        signal=float(np.linalg.norm(plus-minus))
        noise=20*np.finfo(np.float64).eps*max(np.linalg.norm(plus)+np.linalg.norm(minus),1.)
        error=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-12))
        results.append({"step":h,"relative_error":error,"signal":signal,"roundoff_scale":float(noise),
                        "resolved":bool(signal>noise)})
    resolved=[x for x in results if x["resolved"]]
    # Exact quadratic operators have no truncation term in this centered directional check.
    # A low interior point is the expected roundoff plateau; record all values, not just min.
    interior=results[1] if len(results)==3 else None
    passed=bool(resolved and min(x["relative_error"] for x in resolved)<=1e-6 and
                interior is not None and interior["resolved"] and interior["relative_error"]<=1e-6)
    return {"samples":results,"passed":passed,"interpretation":"quadratic centered difference; roundoff plateau"}
