"""Prospective TRAIN cylinder-traction regression; no chosen parent/weight/trainer.

G is d(u/U)/d(x/D), quadrature weights are ds/D, and supplied normals are
outward from the obstacle (negative of the retained fluid-domain normals).
No Reynolds parameter or pressure enters the spatial-mode teacher. A future
consumer must bind the exact cylinder subset, rule, normals and POD ordering.
"""
import numpy as np


def traction(gradient,normal,xp=np):
    return xp.einsum('nijr,nj->nir',gradient+xp.swapaxes(gradient,1,2),normal)


def teacher_normalization(gradient,normal,weights,singular_values):
    g,n,w,s=[np.asarray(a,dtype=np.float64) for a in (gradient,normal,weights,singular_values)]
    if g.ndim!=4 or g.shape[1:3]!=(2,2) or n.shape!=(len(g),2) or w.shape!=(len(g),) or s.shape!=(g.shape[-1],):raise ValueError('exact boundary mode layout required')
    if any(not np.isfinite(a).all() for a in (g,n,w,s)) or np.any(w<=0) or np.any(s<=0):raise ValueError('finite positive boundary rule and mode energies required')
    if not np.allclose(np.sum(n*n,axis=1),1.,rtol=0.,atol=1e-10):raise ValueError('unit obstacle normals required')
    t=traction(g,n);norms=np.einsum('n,nir,nir->r',w,t,t);denominator=float(np.sum(s*s*norms))
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('positive finite global TRAIN traction energy required')
    return {'mode_traction_norms_squared':norms,'global_training_traction_energy':denominator,'normalized_mode_energy_weights':s*s*norms/denominator}


def block_traction_objective(predicted_gradient,teacher_gradient,normal,weights,singular_values,global_teacher_energy,xp=np):
    """Additive block numerator over one frozen GLOBAL teacher denominator.

    Zero-energy individual modes are permitted: no per-mode norm division.
    No coefficient multiplying this term is selected by this helper.
    """
    predicted=traction(predicted_gradient,normal,xp);target=traction(teacher_gradient,normal,xp);error=predicted-target
    mode_error=xp.einsum('n,nir,nir->r',weights,error,error)
    numerator=xp.sum(singular_values**2*mode_error)
    return numerator/global_teacher_energy,{'error_norm_squared_by_mode':mode_error,'predicted_norm_squared_by_mode':xp.einsum('n,nir,nir->r',weights,predicted,predicted),'teacher_norm_squared_by_mode':xp.einsum('n,nir,nir->r',weights,target,target),'integrated_traction_error_by_component_mode':xp.einsum('n,nir->ir',weights,error),'weighted_numerator':numerator}
