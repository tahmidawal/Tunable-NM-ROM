"""One reviewed frozen ROM model and actual FOM; no training/refit/selection."""
from pathlib import Path
import json,shutil
import numpy as np
from fe_adapter import sha_file
from qualification import require_qualification
from pilot2d import (source_digest,load_preparation,_read_arrays,FrozenBank,load_reference_population,
                     score_state,galerkin_operator,quadrature_state_check,save_evidence)
from frozen_rom2d import load_frozen,rebuild
from fom_comparator2d import FrozenFOMComparator,verify_documents
from fom_bridge2d import score_fom_invocation
from solver import jacobian_check
from timing import paired_cpu


def verify_decisions(inputs):
    review=json.loads(Path(inputs['rom_review']).read_text())
    if (review.get('schema')!='ns-obstacle-frozen-rom-review-v1' or review.get('passed') is not True or
        review.get('source_sha256')!=source_digest() or review.get('validation_seen') is not False or
        review.get('frozen_models_sha256')!=sha_file(inputs['frozen_models'])):
        raise ValueError('independent exact development-frozen ROM review required before validation')
    result=verify_documents(inputs['fom_selection'],inputs['fom_review'],inputs['fom_eligibility'],
                            inputs['truth_admission'],inputs['fom_release'])
    if result[0]['population']!='validation':raise ValueError('explicit paired validation release required')
    if require_qualification(review.get('reference_qualification'))!=result[2]:raise ValueError('ROM/FOM truth scopes differ')
    if review.get('reference_admission_sha256')!=result[0]['reference_admission_sha256']:
        raise ValueError('ROM/FOM truth identity differs')
    target=result[1]['bank_target']
    if target['frozen_models_sha256']!=review['frozen_models_sha256'] or target['rom_review_sha256']!=sha_file(inputs['rom_review']):
        raise ValueError('FOM matching target differs from actual frozen bank evidence')
    return review,result[0]


def failed_call(contract,reason):
    def call():
        exc=RuntimeError(reason)
        exc.invocation={'output_contract':contract,'outputs':None,'outputs_available':False,
            'work':{'status':'construction_or_development_gate_rejected'},'query_failed':True,
            'exception':reason,'evidence':{'state_available':False},'physical_admission':False}
        raise exc
    return call


def score_rom(model,operator,interpolation,invocation,reference,boundary):
    if invocation.get('query_failed'):return {'eligible_accuracy':False,'query_failed':True}
    state=invocation['evidence']['state'];re=reference['Re']
    if invocation['evidence']['reynolds']!=re or invocation['evidence']['model']!=model.name:
        raise ValueError('actual returned ROM state identity differs')
    scores=score_state(model,state,reference,boundary,model.rule.weights)
    direct=np.r_[model.rule.direct(state,1./re)];tensor=model.operator.residual(state,1./re,whiten=False)
    denominator=max(sum(float(np.linalg.norm(v)) for v in model.operator.terms(state,1./re).values()),1e-12)
    gap=float(np.linalg.norm(direct-tensor)/denominator)
    derivative=jacobian_check(operator,state,1./re,np.ones(len(state)))
    converged=interpolation or invocation['work']['status'] in ('stationary','root_converged')
    return {'scores':scores,'solver_converged':converged,'full_direct_tensor_relative_gap':gap,
        'full_direct_tensor_denominator':denominator,'jacobian_check':derivative,
        'eligible_accuracy':bool(converged and scores['accuracy_targets_passed'] and
          (interpolation or (model.stability['passed'] and derivative['passed'] and gap<=1e-10)))}


def run(inputs,release,output):
    review,fom_release=verify_decisions(inputs)
    name=release['model_name'];reynolds=release['reynolds'];contract=release['output_contract']
    if (contract!=fom_release['output_contract'] or not reynolds or
        len(set(reynolds))!=len(reynolds) or reynolds!=fom_release['reynolds']):
        raise ValueError('outer and actual FOM output/population release differ')
    target=fom_release['bank_target']
    if name.startswith('bank_') and name!=target['model']:raise ValueError('FOM selected for a different bank target')
    if name.startswith('POD_') and name!='POD_r'+str(target['prefix']):raise ValueError('POD comparison differs from declared bank target prefix')
    from uncertainty2d import load as load_uncertainty
    arrays,metadata,inner=load_preparation(inputs['prepared'],inputs['preparation_release'])
    for key,field in [('bank','bank_sha256'),('probes','probes_sha256')]:
        if sha_file(inputs[key])!=inner[field]:raise ValueError('released '+key+' differs')
    if metadata['admission_evidence']['reference_admission_sha256']!=fom_release['reference_admission_sha256']:
        raise ValueError('prepared ROM and selected FOM truth identities differ')
    frozen=load_frozen(inputs['frozen_models'],review['frozen_models_sha256'],metadata['reference_qualification'])
    bank=FrozenBank.load(inputs['bank']);probes=_read_arrays(inputs['probes'],2**30)
    out=Path(output);out.mkdir(exist_ok=False)
    model,fitted=rebuild(frozen,name,arrays,metadata,probes,bank)
    save_evidence(out,'model',{'name':name,'fingerprint':frozen['models'][name]['fingerprint'],
        'source':model.source,'stability':model.stability,'choice':frozen['choices'][name],
        'original_training_budget':frozen['input_identity']['original_training_budget']})
    fom=FrozenFOMComparator.open(inputs['fom_selection'],inputs['fom_review'],inputs['fom_eligibility'],
                                inputs['truth_admission'],inputs['fom_release'],out/'fom_setup')
    expected=fom_release['reference_population_sha256']
    if sha_file(inputs['population'])!=expected:raise ValueError('paired reference population differs')
    population_meta=json.loads(Path(inputs['population']).read_text())
    if population_meta.get('reference_admission_sha256')!=fom_release['reference_admission_sha256']:
        raise ValueError('paired reference truth identity differs')
    references={row['Re']:row for row in load_reference_population(inputs['population'],expected,'validation',fom.rule_hash)}
    if not set(reynolds)<=set(references):raise ValueError('released validation state absent')
    deltas=load_uncertainty(inputs['uncertainty'],fom_release['validation_uncertainty_sha256'],
        inputs['uncertainty_review'],fom_release['validation_uncertainty_review_sha256'],metadata['reference_qualification'],
        fom_release['reference_admission_sha256'],fom.rule_hash,'validation')
    receipts=[]
    ledger={'schema':'ns-obstacle-paired-allocation-footprint-v1','model':name,'contract':contract,
        'released_reynolds':reynolds,'completed_blocks':[],'status':'running',
        'minimum_free_reserve_bytes':10*2**30,'external_next_allocation_automatically_authorized':False}
    try:
        for index,re in enumerate(reynolds):
            block=out/f'block_{index:02d}';block.mkdir()
            receipt=run_block(model,fitted,fom,arrays,metadata,review,fom_release,references[re],block,deltas[re])
            receipts.append(receipt)
            footprint=json.loads((block/'BLOCK-FOOTPRINT.json').read_text())
            ledger['completed_blocks'].append(footprint)
            ledger['cumulative_working_output_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            ledger['remaining_free_bytes']=shutil.disk_usage(out).free
            (out/'ALLOCATION-FOOTPRINT.json').write_text(json.dumps(ledger,indent=2)+'\n')
        ledger['status']='pending_retention_audit'
    except Exception as exc:ledger.update(status='failed',exception=repr(exc));raise
    finally:
        ledger['cumulative_working_output_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        (out/'ALLOCATION-FOOTPRINT.json').write_text(json.dumps(ledger,indent=2)+'\n')
    return receipts


def run_block(model,fitted,fom,arrays,metadata,review,fom_release,reference,out,deltas=None):
    name=model.name;re=reference['Re'];contract=fom_release['output_contract']
    expected=fom_release['reference_population_sha256']
    variants={'exact_tensor_weak_LS':(model.operator,False)};construction={}
    if fitted is not None:variants['fitted_convection_weak_LS']=(fitted,False)
    else:construction['fitted_convection_weak_LS']='no development-qualified fitted rule'
    if model.kind=='POD':
        variants['coefficient_PCHIP']=(model.operator,True)
        try:variants['mixed_Galerkin']=(galerkin_operator(model),False)
        except Exception as exc:construction['mixed_Galerkin']=repr(exc)
    calls={'FOM':lambda:fom.query(re)}
    for method,(operator,interpolation) in variants.items():
        calls[method]=(lambda op=operator,interp=interpolation:model.query(re,contract,op,interp)) if (
            interpolation or model.stability['passed']) else failed_call(contract,'inf-sup gate rejected')
    for method,reason in construction.items():calls[method]=failed_call(contract,reason)
    # Reserve worst-case distinct outputs/states before querying; actual duplicate
    # savings are credited only by save_evidence's content hash + byte equality.
    output_values={'full_velocity_pressure_fields':len(fom.points)*3,'fixed_probes':6,'integrated_forces':2}[contract]
    payload=7*(len(calls)*output_values*8+fom.native.X.ndof*8+(len(calls)-1)*(model.operator.nv+model.operator.np)*8)
    next_block_bound=payload+(payload+49)//50+16*2**20
    if shutil.disk_usage(out).free<10*2**30+2*next_block_bound:
        raise OSError('next paired block and durable-copy reserve do not fit')
    save_evidence(out,'block_reservation',{'worst_distinct_payload_bytes':payload,
        'next_block_and_durable_copy_bytes':2*next_block_bound,'minimum_free_reserve_bytes':10*2**30,
        'compression_or_repeat_identity_assumed':False})
    timing=paired_cpu(calls,contract,fom_release['cores'])
    raw_receipt=save_evidence(out,'actual_invocations',timing)
    rows=timing['actual_invocation_outputs'];scored=[]
    controls={row['repetition']:row['invocation'] for row in rows if row['method']=='exact_tensor_weak_LS'}
    for row in rows:
        method=row['method'];invocation=row['invocation']
        try:
            if invocation.get('query_failed'):value={'eligible_accuracy':False,'query_failed':True,'exception':invocation.get('exception')}
            elif method=='FOM':value=score_fom_invocation(fom.native,invocation,reference,fom.weights,fom.points)
            else:
                operator,interpolation=variants[method]
                value=score_rom(model,operator,interpolation,invocation,reference,arrays['boundary'])
                if method=='fitted_convection_weak_LS':
                    control=controls[row['repetition']]
                    statuses={k:v.get('work',{}).get('status') for k,v in [('full',control),('fitted',invocation)]}
                    both=all(v in ('stationary','root_converged') for v in statuses.values())
                    check={'passed':False,'solver_statuses':statuses,'both_solves_converged':both}
                    if not control.get('query_failed') and 'state' in control.get('evidence',{}):
                        check.update(quadrature_state_check(model,fitted,control['evidence']['state'],invocation['evidence']['state'],re))
                        check['numerical_checks_passed']=check['passed'];check['passed']=bool(check['passed'] and both)
                    value['quadrature_reached_state_check']=check;value['eligible_accuracy'] &= check['passed']
        except Exception as exc:value={'eligible_accuracy':False,'diagnostic_failed':True,'exception':repr(exc)}
        scored.append({'method':method,'repetition':row['repetition'],'score':value})
    matches=[]
    if deltas is not None:
        from uncertainty2d import joint_match
        fom_scores={row['repetition']:row['score'] for row in scored if row['method']=='FOM'}
        for row in scored:
            if row['method']=='FOM':continue
            own=row['score'];baseline=fom_scores[row['repetition']]
            match={'matched':False,'reason':'ineligible actual invocation'}
            if own.get('eligible_accuracy') is True and baseline.get('eligible_accuracy') is True:
                match=joint_match(own['scores'],baseline,deltas)
            matches.append({'method':row['method'],'repetition':row['repetition'],**match})
    receipt=save_evidence(out,'paired_'+contract,{'timing_record':raw_receipt,'same_invocation_accuracy':scored,
        'conditional_FOM_vs_current_ROM_invocation':matches,'frozen_bank_target':fom_release.get('bank_target'),
        'model':name,'Re':re,'frozen_models_sha256':review['frozen_models_sha256'],
        'reference_population_sha256':expected,'construction_failures':construction,
        'reference_qualification':metadata['reference_qualification'],'physical_admission':False,
        'training_or_refitting_performed':False})

    measured=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    ledger={'schema':'ns-obstacle-paired-block-footprint-v1','status':'pending_retention_audit',
        'model':name,'Re':re,'contract':contract,'record':receipt,
        'actual_working_output_bytes':measured,'worst_distinct_payload_bytes':payload,
        'minimum_free_reserve_bytes':10*2**30,'remaining_free_bytes':shutil.disk_usage(out).free,
        'compression_or_repeat_identity_assumed':False,'next_block_automatically_authorized':False}
    (out/'BLOCK-FOOTPRINT.json').write_text(json.dumps(ledger,indent=2)+'\n')
    return receipt
