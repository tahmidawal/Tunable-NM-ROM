"""Untimed scoring of the very coefficient state returned by a reviewed FOM query.

This bridge does not construct/qualify a FOM candidate, load a lift, run Newton or
provide physical admission. Source-bound candidate selection remains the caller's job.
"""
import numpy as np
from pilot2d import score_field_arrays


def score_fom_invocation(query,invocation,reference,common_weights):
    if invocation.get('query_failed'):
        return {'eligible_accuracy':False,'query_failed':True,'state_or_output_available':False,
                'exception':invocation['exception'],'physical_admission':False}
    evidence=invocation['evidence']
    if float(evidence['Re'])!=float(reference['Re']):raise ValueError('FOM invocation/reference Re differs')
    converged=(evidence.get('converged') is True and invocation['work'].get('status')=='root_converged')
    diagnostics=query.diagnostics_from_evidence(evidence)
    actual_weights=np.asarray(diagnostics['volume_weights']);common_weights=np.asarray(common_weights)
    if actual_weights.shape!=common_weights.shape or not np.allclose(actual_weights,common_weights,rtol=1e-12,atol=1e-14):
        raise ValueError('FOM diagnostics must cover the original common full rule')
    full={key:diagnostics[key] for key in ('velocity','pressure','velocity_gradient')}
    boundary={'labels':diagnostics['boundary_labels'],'weights_dimensionless':diagnostics['boundary_weights'],
              'normals_fluid':diagnostics['fluid_normals'],'points_physical':diagnostics['boundary_points']}
    result=score_field_arrays('FOM',full,{'velocity':diagnostics['boundary_velocity']},
                             diagnostics['force_coefficient'],reference,boundary,common_weights)
    result.update(solver_converged=converged,eligible_accuracy=bool(converged and result['accuracy_targets_passed']),
                  actual_invocation_diagnostics=True,diagnostic_timing='outside timed query, using retained coefficients')
    return result
