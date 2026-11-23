"""Shared actual-invocation initializer contract for phase and complete queries."""
import numpy as np
import support as S
import shared_initializer as I
import check_arithmetic as C
from arithmetic_audit import NetworkAudit


def fit_initial_field(initial,basis,params,codes,candidates,gtol,*,normalization=None):
    target,center=S.target_projection(S.jnp.asarray(initial),basis)
    target=np.asarray(target);center=np.asarray(center)
    norm=float(np.linalg.norm(np.asarray(initial))) if normalization is None else float(normalization)
    code,fit=I.fit_head(target,params,codes,candidates,gtol=gtol)
    ranges=C.check_ranges(params,[c['code'] for c in fit['controls']]+[code])
    if ranges['passed'] and np.all(np.isfinite(target)):
        arithmetic=NetworkAudit(params).compare(fit['controls'],code,target,fit['direct_original_objective_guard'])
        increment=C.check_network(params,fit['anchor_code'])
    else:
        arithmetic={'passed':False,'executed':False,'reason':'unsafe range or nonfinite input'}
        increment={'passed':False,'executed':False,'reason':'unsafe range or nonfinite input'}
    coefficients=np.asarray(S.H.head_apply(params,S.jnp.asarray(code)))
    field=np.asarray(S.reconstruct_initial(S.jnp.asarray(coefficients),S.jnp.asarray(center),basis))
    phase_controls=[c for c in fit['controls'] if c['kind']=='phase']
    oldindex,_,_=I.select_fit_candidate([c['original_info'] for c in phase_controls])
    oldcode=phase_controls[oldindex]['code']
    oldcoeff=S.H.head_apply(params,S.jnp.asarray(oldcode))
    oldfield=np.asarray(S.reconstruct_initial(oldcoeff,S.jnp.asarray(center),basis))
    old_new_gap=float(np.linalg.norm(field-oldfield)/norm)
    gates={'core_direct_and_anchored':bool(fit['qualified']),
           'independent_objective_arithmetic':arithmetic['passed'],
           'increment_consistency':increment['passed'],'activation_range':ranges['passed'],
           'old_new_initial_field':old_new_gap<=1e-6}
    # Rich v2 qualification is preserved by conjunction, never overwritten by a reason string.
    qualified=all(gates.values())
    record={'fit':fit,'arithmetic_audit':arithmetic,'increment_checks':increment,
        'activation_ranges':ranges,'qualification':gates,'qualified':qualified,
        'gtol':gtol,'target':target,'center':center,'code':code,'coefficients':coefficients,
        'normalization':norm,'old_phase_selected_control':oldindex,
        'old_new_initial_field_gap':old_new_gap,
        'initial_field_error':float(np.linalg.norm(field-np.asarray(initial))/norm),
        'ordinary_direct_objective_guard':fit['direct_original_objective_guard']}
    return field,record


def save_invocation(path,record,params):
    path=S.Path(path);fit=record['fit'];target=S.jnp.asarray(record['target'])
    dr,dj,_=I.snapshot(record['code'],target,params)
    ar,aj,_=I.snapshot(record['code'],target,params,fit['anchor_code'])
    arrays=path.with_suffix('.npz')
    np.savez_compressed(arrays,returned_code=record['code'],target=record['target'],
        returned_residual=np.asarray(dr),returned_jacobian=np.asarray(dj),
        returned_anchored_residual=np.asarray(ar),returned_anchored_jacobian=np.asarray(aj),
        anchor_code=fit['anchor_code'],control_codes=np.stack([c['code'] for c in fit['controls']]),
        control_direct_residuals=np.stack([c['direct_residual'] for c in fit['controls']]),
        control_direct_jacobians=np.stack([c['direct_jacobian'] for c in fit['controls']]),
        control_anchored_residuals=np.stack([c['anchored_residual'] for c in fit['controls']]),
        control_anchored_jacobians=np.stack([c['anchored_jacobian'] for c in fit['controls']]))
    S.dump(path.with_suffix('.json'),{**record,'audit_arrays':arrays.name,'audit_array_sha256':S.sha(arrays)})
