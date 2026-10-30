"""Prospective retained-mode TRAINING reconstruction weighting, never solver loss.

If the retained training coefficient matrix C satisfies C C.T = diag(sigma**2),
these weights reproduce the normalized ensemble value/gradient reconstruction
errors at those ORIGINAL coefficients. They do not describe coefficients found
by an online PDE solve, omitted POD modes, or full-snapshot residual errors.
No parent, lambda, optimizer integration or execution is selected here.
"""
import numpy as np
from accuracy_v2.divergence_objective import block_objective


def training_energy_weights(singular_values, value_norms_squared, gradient_norms_squared):
    sigma,nv,ng=[np.asarray(x,dtype=np.float64) for x in (singular_values,value_norms_squared,gradient_norms_squared)]
    if sigma.ndim!=1 or not len(sigma) or nv.shape!=sigma.shape or ng.shape!=sigma.shape or any(not np.isfinite(x).all() or np.any(x<=0) for x in (sigma,nv,ng)):
        raise ValueError('positive finite matching retained-mode singular values and training norms required')
    def normalized_energy(norm):
        # Log scaling avoids overflow without changing the normalized weights.
        log_energy=2*np.log(sigma)+np.log(norm)
        energy=np.exp(log_energy-np.max(log_energy))
        return energy/energy.sum()
    return {'value':normalized_energy(nv),'gradient_and_trace':normalized_energy(ng)}


def block_energy_objective(values,gradients,targets,target_gradients,weights,
                           value_norms,gradient_norms,value_mode_weights,
                           gradient_mode_weights,divergence_weight,xp):
    """Use separate frozen TRAINING value and derivative energy weights.

    Global denominators remain the original per-mode norms; block contributions
    add exactly. Trace uses the gradient denominator and derivative weights.
    Explicit weight/lambda validation and coefficient-covariance provenance are
    obligations of any separately reviewed future launcher.
    """
    _,parts=block_objective(values,gradients,targets,target_gradients,weights,
                           value_norms,gradient_norms,gradient_mode_weights,
                           divergence_weight,xp)
    value=xp.sum(value_mode_weights*parts['value_relative_squared_by_mode'])
    gradient=xp.sum(gradient_mode_weights*parts['gradient_relative_squared_by_mode'])
    parts['value_component_objective']=value
    parts['gradient_component_objective']=gradient
    parts['base_objective']=value+gradient
    parts['combined_objective']=parts['base_objective']+parts['supervised_divergence_objective_weighted']
    return parts['combined_objective'],parts
