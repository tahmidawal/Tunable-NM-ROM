"""Prospective affine output-layer polish helpers; no trainer, input or release.

Each head has 130 joint output coordinates in W64x2 then b2 order. Hidden
layers and embedded parent are immutable. Rows target deltas from the actual
current composite; no normal equations, ridge or discarded-mode exactness claim.
"""
import numpy as np
from accuracy_v2.spatial_bank import inputs,envelope


def output_features(hidden_layers,points,xp=np):
    """Envelope-weighted 64 hidden features + bias; physical derivatives."""
    if len(hidden_layers)!=2:raise ValueError('exact two hidden layers required')
    h,dh=inputs(points,xp)
    for w,b in hidden_layers:
        h=xp.tanh(h@w+b)
        dh=xp.einsum('nhd,hk->nkd',dh,w)*(1-h*h)[:,:,None]
    if h.shape[1]!=64:raise ValueError('fixed64-wide head required')
    h=xp.concatenate((h,xp.ones((len(h),1),dtype=h.dtype)),axis=1)
    dh=xp.concatenate((dh,xp.zeros((len(h),1,2),dtype=dh.dtype)),axis=1)
    e,de=envelope(points,xp)
    return e[:,None]*h,e[:,None,None]*dh+h[:,:,None]*de[:,None,:]


def design(values,dimensionless_gradients,xp=np):
    """Linear maps to value and dimensionless gradient of one output head."""
    if values.shape[1:]!=(65,) or dimensionless_gradients.shape!=(len(values),65,2):raise ValueError('exact feature shapes required')
    eye=xp.eye(2,dtype=values.dtype)
    v=xp.einsum('nj,ic->nijc',values,eye).reshape(len(values),2,130)
    g=xp.einsum('njd,ic->nidjc',dimensionless_gradients,eye).reshape(len(values),2,2,130)
    return v,g


def volume_rows(feature_values,feature_gradients_dimless,value_error,gradient_error,weights,mode_weight,value_norm,gradient_norm,trace_weight,xp=np):
    """Rows satisfy ||A delta-b||² = one head's unchanged volume objective.

    error is CURRENT COMPOSITE minus FE teacher. Frozen normalization scalars
    must be validated/bound by the eventual consumer; no local normalization.
    """
    v,g=design(feature_values,feature_gradients_dimless,xp)
    a=xp.concatenate((v*xp.sqrt(weights*mode_weight/value_norm)[:,None,None],
                      g.reshape(len(v),4,130)*xp.sqrt(weights*mode_weight/gradient_norm)[:,None,None],
                      (g[:,0,0]+g[:,1,1])[:,None,:]*xp.sqrt(weights*mode_weight*trace_weight/gradient_norm)[:,None,None]),axis=1)
    b=-xp.concatenate((value_error*xp.sqrt(weights*mode_weight/value_norm)[:,None],
                       gradient_error.reshape(len(v),4)*xp.sqrt(weights*mode_weight/gradient_norm)[:,None],
                       (gradient_error[:,0,0]+gradient_error[:,1,1])[:,None]*xp.sqrt(weights*mode_weight*trace_weight/gradient_norm)[:,None]),axis=1)
    return a.reshape(-1,130),b.reshape(-1)


def boundary_rows(feature_values,feature_gradients_dimless,gradient_error,normals_body,weights,sigma,global_teacher_traction_energy,traction_weight,xp=np):
    """Joint component rows; fixed global sigma² traction denominator, no Re."""
    _,g=design(feature_values,feature_gradients_dimless,xp)
    a=xp.einsum('nijd,nj->nid',g+xp.swapaxes(g,1,2),normals_body)
    b=-xp.einsum('nij,nj->ni',gradient_error+xp.swapaxes(gradient_error,1,2),normals_body)
    scale=xp.sqrt(weights*sigma*sigma*traction_weight/global_teacher_traction_energy)
    return (a*scale[:,None,None]).reshape(-1,130),(b*scale[:,None]).reshape(-1)


def qr_append(previous_factor,rows,right,xp=np):
    """Augmented QR retains the RHS and constant residual; never forms A.T A.

    Sequential original block order. Stack only small prior R with next bounded
    block. xp may be numpy or jax.numpy; future execution must bind GPU/f64.
    """
    if rows.ndim!=2 or rows.shape[1]!=130 or right.shape!=(len(rows),):raise ValueError('130-variable joint affine problem required')
    augmented=xp.concatenate((rows,right[:,None]),axis=1)
    if previous_factor is not None:
        if previous_factor.ndim!=2 or previous_factor.shape[1]!=131 or len(previous_factor)>131:raise ValueError('bounded augmented QR factor required')
        augmented=xp.concatenate((previous_factor,augmented),axis=0)
    return xp.linalg.qr(augmented,mode='r')


def solve_factor(factor,relative_cutoff):
    """Small rank-revealing SVD solution; explicit cutoff, no ridge or exactness promise.

    Returns raw factors/spectrum/delta for retention BEFORE any acceptance gate.
    Truncated singular directions are reported, not silently treated as solved.
    """
    r=np.asarray(factor,dtype=np.float64)
    if r.ndim!=2 or r.shape[1]!=131 or not 1<=len(r)<=131 or not np.isfinite(r).all():raise ValueError('finite bounded augmented factor required')
    if isinstance(relative_cutoff,bool) or not np.isfinite(relative_cutoff) or not 0<relative_cutoff<1:raise ValueError('explicit positive relative SVD cutoff required')
    a,b=r[:,:130],r[:,130];u,s,vt=np.linalg.svd(a,full_matrices=False)
    cutoff=relative_cutoff*s[0] if len(s) else 0.;keep=s>cutoff
    delta=vt[keep].T@((u[:,keep].T@b)/s[keep])
    residual=a@delta-b;gradient=2*a.T@residual
    return {'augmented_factor':r.copy(),'singular_values':s,'right_singular_vectors':vt,'left_singular_vectors':u,'retained_directions':keep,'delta':delta,'factor_residual':residual,'factor_gradient':gradient,
            'relative_cutoff':float(relative_cutoff),'absolute_cutoff':float(cutoff),'numerical_rank':int(np.sum(keep)),
            'full_column_rank':bool(np.sum(keep)==130),'condition_retained':float(s[0]/s[keep][-1]) if np.any(keep) else None,
            'initial_factor_objective':float(b@b),'candidate_factor_objective':float(residual@residual),
            'full_uncached_replay_required':True,'physical_admission':False}
