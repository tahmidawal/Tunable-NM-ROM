"""Successor actual-state diagnostic core; only a separately released wrapper may call it.

No fit or truth-coefficient initializer. No production gate or eligibility is inferred.
"""
from pathlib import Path
import json,time
import numpy as np
import pilot2d as pilot
from fe_adapter import sha_file
from accuracy_v2.spatial_bank import SpatialBank

def diagnose(model,re,reference,boundary,out,guard):
    key='pod_guided_neural_r4_'+str(re);row={'Re':re,'eligible_accuracy':False,'physical_admission':False,'diagnostic_only':True}
    # Only Re and the existing exact operator reach query; its existing initializer is zero.
    try:
        guard();invocation=model.query(re,'full_velocity_pressure_fields',model.operator,False)
    except Exception as exc:
        failure=getattr(exc,'invocation',None)
        if failure is None:failure={'exception':repr(exc),'state_available':False}
        row['raw_invocation']=pilot.save_evidence(out,key+'_raw_invocation',failure);row['failure']=repr(exc)
        return row
    row['raw_invocation']=pilot.save_evidence(out,key+'_raw_invocation',invocation)
    try:
        guard();state=invocation['evidence']['state']
        direct=np.r_[model.rule.direct(state,1./re)];tensor=model.operator.residual(state,1./re,whiten=False)
        terms=model.operator.terms(state,1./re);denominator=max(sum(float(np.linalg.norm(t)) for t in terms.values()),1e-12)
        gradient_fields=model.reconstruct(state,re,gradients=True);boundary_fields=model.reconstruct(state,re,'boundary')
        row['diagnostic_arrays']=pilot.save_evidence(out,key+'_diagnostic_arrays',{'full':gradient_fields,'boundary':boundary_fields,'direct':direct,'tensor':tensor,'terms':terms,'force_coefficient':model.force_output(state,re)})
        scores=pilot.score_state(model,state,reference,boundary,model.rule.weights)
        derivative=pilot.jacobian_check(model.operator,state,1./re,np.ones(len(state)))
        gap=float(np.linalg.norm(direct-tensor)/denominator)
        row.update(scores=scores,solver_status=invocation['work']['status'],full_direct_tensor_relative_gap=gap,
            full_direct_tensor_denominator=denominator,jacobian_check=derivative,
            diagnostic_numerical_targets_passed=bool(invocation['work']['status'] in ('stationary','root_converged') and scores['accuracy_targets_passed'] and derivative['passed'] and gap<=1e-10))
    except Exception as exc:row['diagnostic_failure']=repr(exc)
    return row


def run_verified(inputs, output, expected_model_sha256, guard):
    """Consume independently accepted model and genuine DEV inputs after outer preflight.

    The outer launcher must bind its complete source, input and review closure and
    allocation. This function is deliberately not an executable release bypass.
    """
    if sha_file(inputs['bank']) != expected_model_sha256:
        raise ValueError('accepted successor model differs')
    out=Path(output);out.mkdir(exist_ok=False)
    result={'schema':'ns-obstacle-pod-guided-neural-diagnostic-v2','status':'started',
        'model_sha256':expected_model_sha256,'original_bank_gate_passed':False,
        'new_model_gate_not_yet_admitted':True,'physical_admission':False,
        'eligible_accuracy':False,'validation_seen':False,'reference_fit_initialization_used':False,
        'training_or_quadrature_fit_performed':False,'matched_accuracy_or_speedup_claim':False,'cases':[]}
    serial=0;start=time.monotonic()
    def persist():
        nonlocal serial
        serial+=1;result['elapsed_seconds']=time.monotonic()-start
        receipt=pilot.save_evidence(out,'status_'+str(serial).zfill(3),result)
        (out/'DIAGNOSTIC.json').write_text(json.dumps({'schema':result['schema'],
            'status':result['status'],'latest_evidence':receipt,'completed_case_records':len(result['cases']),
            'physical_admission':False,'eligible_accuracy':False},indent=2)+'\n')
    persist()
    try:
        guard();bank=SpatialBank.load(inputs['bank'])
        if bank.rank!=4:raise ValueError('first successor rank differs')
        arrays,metadata,_=pilot.load_preparation(inputs['prepared'],inputs['preparation_release'])
        probes=pilot._read_arrays(inputs['probes'],2**30);guard()
        model=pilot.build_model('bank',4,arrays,metadata,probes,bank)
        evidence={'source':model.source,'stability':model.stability,'fresh_fingerprint':pilot.model_fingerprint(model),
                  'training_coefficients':model.training_coefficients,'operator':vars(model.operator),'force':model.force}
        result['model']=pilot.save_evidence(out,'pod_guided_neural_r4_model',evidence);persist()
        if not model.stability['passed']:raise RuntimeError('unchanged inf-sup gate rejected')
        rule=pilot.array_digest({k:arrays['volume'][k] for k in ('points_dimensionless','weights_dimensionless')})
        population=pilot.load_reference_population(inputs['development'],sha_file(inputs['development']),'development',rule)
        for reference in population:
            guard()
            # Query is first, receives no reference or projection coefficients.
            row=diagnose(model,reference['Re'],reference,arrays['boundary'],out,guard)
            result['cases'].append(row);persist()
            try:
                guard();row['representation_diagnostic']=pilot.projection_diagnostic(model,reference,arrays['boundary'],model.rule.weights)
            except Exception as exc:row['representation_diagnostic_failure']=repr(exc)
            persist()
        result['status']='completed_diagnostic_pending_independent_review';persist()
    except BaseException as exc:
        result.update(status='failed',exception=repr(exc));persist();raise
    return result
