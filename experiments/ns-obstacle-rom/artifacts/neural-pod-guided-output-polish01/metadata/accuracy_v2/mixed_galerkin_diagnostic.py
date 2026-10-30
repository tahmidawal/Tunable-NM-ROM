"""Prospective same-bank mixed Galerkin diagnostic; separate release and wrapper required.

No fit or truth-coefficient initializer. No production gate or eligibility is inferred.
"""
from pathlib import Path
import json,time
import numpy as np
import pilot2d as pilot
from fe_adapter import sha_file
from accuracy_v2.residual_head_bank import ResidualHeadBank
from weak import WeakRule,WeakOperator

class EnrichedGalerkinRule:
    """Same actual enriched trial fields as tests, accumulated in bounded blocks."""
    def __init__(self,model,block_points=4096):
        if type(block_points) is not int or not 1<=block_points<=4096:raise ValueError('bounded point blocks required')
        self.fields=model.fields['volume'];self.weights=model.rule.weights;self.block_points=block_points
        if self.fields['velocity'].shape[-1]!=8 or self.fields['pressure'].shape[-1]!=4:raise ValueError('exact enriched8/pressure4 required')
    def blocks(self):
        f=self.fields
        for start in range(0,len(self.weights),self.block_points):
            s=slice(start,min(start+self.block_points,len(self.weights)))
            yield WeakRule(self.weights[s],f['lift'][s],f['lift_gradient'][s],f['pressure_lift'][s],
                f['velocity'][s],f['velocity_gradient'][s],f['pressure'][s],
                f['velocity'][s],f['velocity_gradient'][s],f['pressure'][s])
    def tensor(self,guard):
        total={}
        for rule in self.blocks():
            guard()
            for k,v in rule.tensor_moments(formulation='mixed_Galerkin').items():
                if k not in total:total[k]=v.copy()
                else:total[k]+=v
        return WeakOperator(**total,formulation='mixed_Galerkin')
    def direct(self,state,nu,guard):
        total=None
        for rule in self.blocks():
            guard();values=rule.direct(state,nu)
            if total is None:total=[v.copy() for v in values]
            else:
                for i,v in enumerate(values):total[i]+=v
        return tuple(total)

def diagnose(model,operator,rule,re,reference,boundary,out,guard):
    key='composite_mixed_galerkin_r4_'+str(re);row={'Re':re,'eligible_accuracy':False,'physical_admission':False,'diagnostic_only':True,'formulation':'mixed_Galerkin','root_tolerance':1e-9}
    # Only Re and the existing exact operator reach query; its existing initializer is zero.
    try:
        guard();invocation=model.query(re,'full_velocity_pressure_fields',operator,False)
    except Exception as exc:
        failure=getattr(exc,'invocation',None)
        if failure is None:failure={'exception':repr(exc),'state_available':False}
        row['raw_invocation']=pilot.save_evidence(out,key+'_raw_invocation',failure);row['failure']=repr(exc)
        return row
    row['raw_invocation']=pilot.save_evidence(out,key+'_raw_invocation',invocation)
    try:
        guard();state=invocation['evidence']['state']
        direct=np.r_[rule.direct(state,1./re,guard)];tensor=operator.residual(state,1./re,whiten=False)
        terms=operator.terms(state,1./re);denominator=max(sum(float(np.linalg.norm(t)) for t in terms.values()),1e-12)
        gradient_fields=model.reconstruct(state,re,gradients=True);boundary_fields=model.reconstruct(state,re,'boundary')
        row['diagnostic_arrays']=pilot.save_evidence(out,key+'_diagnostic_arrays',{'full':gradient_fields,'boundary':boundary_fields,'direct':direct,'tensor':tensor,'terms':terms,'force_coefficient':model.force_output(state,re),'galerkin_whitened_residual':operator.residual(state,1./re),'original_LS_residual_auxiliary_only':model.operator.residual(state,1./re)})
        scores=pilot.score_state(model,state,reference,boundary,model.rule.weights)
        derivative=pilot.jacobian_check(operator,state,1./re,np.ones(len(state)))
        gap=float(np.linalg.norm(direct-tensor)/denominator)
        root_norm=float(np.linalg.norm(operator.residual(state,1./re)))
        row.update(root_residual_norm=root_norm,scores=scores,solver_status=invocation['work']['status'],full_direct_tensor_relative_gap=gap,
            full_direct_tensor_denominator=denominator,jacobian_check=derivative,
            diagnostic_numerical_targets_passed=bool(invocation['work']['status']=='root_converged' and root_norm<=1e-9 and scores['accuracy_targets_passed'] and derivative['passed'] and gap<=1e-10))
    except Exception as exc:row['diagnostic_failure']=repr(exc)
    return row


def run_verified(inputs, output, expected_model_sha256, guard, bank_loader=ResidualHeadBank.load):
    """Consume independently accepted model and genuine DEV inputs after outer preflight.

    The outer launcher must bind its complete source, input and review closure and
    allocation. This function is deliberately not an executable release bypass.
    """
    if sha_file(inputs['bank']) != expected_model_sha256:
        raise ValueError('accepted successor model differs')
    out=Path(output);out.mkdir(exist_ok=False)
    result={'schema':'ns-obstacle-composite-mixed-galerkin-diagnostic-v1','status':'started',
        'model_sha256':expected_model_sha256,'original_bank_gate_passed':False,
        'new_model_gate_not_yet_admitted':True,'physical_admission':False,
        'eligible_accuracy':False,'validation_seen':False,'reference_fit_initialization_used':False,
        'training_or_quadrature_fit_performed':False,'matched_accuracy_or_speedup_claim':False,'cases':[],'formulation':'mixed_Galerkin','original_LS_solver_not_rerun':True}
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
        guard();bank=bank_loader(inputs['bank']);result['bank_evaluator_class']=type(bank).__module__+'.'+type(bank).__name__
        if bank.rank!=4:raise ValueError('first successor rank differs')
        arrays,metadata,_=pilot.load_preparation(inputs['prepared'],inputs['preparation_release'])
        probes=pilot._read_arrays(inputs['probes'],2**30);guard()
        model=pilot.build_model('bank',4,arrays,metadata,probes,bank)
        evidence={'source':model.source,'stability':model.stability,'fresh_fingerprint':pilot.model_fingerprint(model),
                  'training_coefficients':model.training_coefficients,'operator':vars(model.operator),'force':model.force}
        result['model']=pilot.save_evidence(out,'original_weak_LS_model_context',evidence);persist()
        if not model.stability['passed']:raise RuntimeError('unchanged inf-sup gate rejected')
        galerkin_rule=EnrichedGalerkinRule(model)
        setup_start=time.monotonic();operator=galerkin_rule.tensor(guard)
        operator_arrays={k:v for k,v in vars(operator).items() if isinstance(v,np.ndarray)}
        result['galerkin_operator']=pilot.save_evidence(out,'actual_mixed_galerkin_operator',{
            'operator':vars(operator),'operator_array_fingerprint':pilot.array_digest(operator_arrays),
            'trial_field_fingerprint':pilot.array_digest(galerkin_rule.fields),
            'weights_sha256':pilot.array_digest({'weights':galerkin_rule.weights}),
            'test_definition':'actual enriched velocity8 and pressure4 trial fields, same original order',
            'momentum_rows':8,'continuity_rows':4,'formulation':operator.formulation,
            'construction_point_block':galerkin_rule.block_points,'construction_seconds':time.monotonic()-setup_start,
            'source_sha256':sha_file(__file__),'inherited_trial_inf_sup':model.stability})
        persist()
        rule=pilot.array_digest({k:arrays['volume'][k] for k in ('points_dimensionless','weights_dimensionless')})
        population=pilot.load_reference_population(inputs['development'],sha_file(inputs['development']),'development',rule)
        for reference in population:
            guard()
            # Query is first, receives no reference or projection coefficients.
            row=diagnose(model,operator,galerkin_rule,reference['Re'],reference,arrays['boundary'],out,guard)
            result['cases'].append(row);persist()
            try:
                guard();row['representation_diagnostic']=pilot.projection_diagnostic(model,reference,arrays['boundary'],model.rule.weights)
            except Exception as exc:row['representation_diagnostic_failure']=repr(exc)
            persist()
        result['status']='completed_diagnostic_pending_independent_review';persist()
    except BaseException as exc:
        result.update(status='failed',exception=repr(exc));persist();raise
    return result
