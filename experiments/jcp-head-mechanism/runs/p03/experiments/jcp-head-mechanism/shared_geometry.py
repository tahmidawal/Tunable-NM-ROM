"""Diagnostics at one common reached state; no uniform stability or global manifold claims."""
import numpy as np

def shared_state_geometry(jac_span, head_coeff_jacobian, physical_span_gram, delta_span=None):
    r=head_coeff_jacobian.shape[0]
    chain=np.zeros((r+3,head_coeff_jacobian.shape[1]+3))
    chain[:r,:-3]=head_coeff_jacobian;chain[-3:,-3:]=np.eye(3)
    gram=(physical_span_gram+physical_span_gram.T)/2
    values,vectors=np.linalg.eigh(gram)
    cutoff=max(values[-1]*1e-12,1e-30);kept=values>cutoff
    white=vectors[:,kept]/np.sqrt(values[kept])
    normalized_span=jac_span@white
    _,singular,right=np.linalg.svd(normalized_span,full_matrices=False)
    # Head+frame physical tangent represented in the retained span-orthonormal coordinates.
    head_in_span=np.sqrt(values[kept])[:,None]*(vectors[:,kept].T@chain)
    qh,hs,_=np.linalg.svd(head_in_span,full_matrices=False)
    hcut=max(hs[0]**2*1e-12,1e-30);hkeep=hs**2>hcut;qh=qh[:,hkeep]
    head_singular=np.linalg.svd(normalized_span@qh,compute_uv=False)
    selected=right[-min(5,len(singular)):][::-1]
    outside=np.sqrt(np.maximum(0,1-np.sum((selected@qh)**2,axis=1)))
    discarded=head_coeff_jacobian.shape[1]+3-int(hkeep.sum())
    total_mass=float(np.trace(chain.T@gram@chain))
    retained_mass=float(np.sum(head_in_span*head_in_span))
    result={'scope':'sampled local same-state physical tangent; no global stability constant',
            'span_gram_eigenvalues':values,'span_gram_cutoff':cutoff,'span_rank':int(kept.sum()),
            'head_gram_eigenvalues_in_retained_span':hs**2,'head_gram_cutoff':hcut,'head_rank':int(hkeep.sum()),
            'head_directions_excluded':discarded,'head_mass_lost_to_span_cutoff':max(0,(total_mass-retained_mass)/max(total_mass,1e-300)),
            'span_physical_singular_values':singular,'head_physical_singular_values':head_singular,
            'least_visible_span_values':singular[-len(selected):][::-1],
            'least_visible_span_outside_head_fraction':outside,
            'restriction_inequality_is_algebraic':True}
    if delta_span is not None:
        physical_delta=np.sqrt(values[kept])*(vectors[:,kept].T@delta_span)
        size=np.linalg.norm(physical_delta)
        outside_part=physical_delta-qh@(qh.T@physical_delta)
        result['actual_step_linearized_physical_norm']=float(size)
        result['actual_step_linearized_outside_head_fraction']=float(np.linalg.norm(outside_part)/max(size,1e-300))
        result['actual_step_weakest_direction_projection_fraction']=float(abs(right[-1]@physical_delta)/max(size,1e-300))
    return result
