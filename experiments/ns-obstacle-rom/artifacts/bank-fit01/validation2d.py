"""Frozen validation accuracy only; no selection, refit, matched timing or speedup."""
from pathlib import Path
import gc,json,time
from fe_adapter import sha_file
from qualification import require_qualification
from pilot2d import (source_digest,load_preparation,FrozenBank,_read_arrays,load_reference_population,
                     array_digest,evaluate_case,save_evidence)
from frozen_rom2d import load_frozen,rebuild


def verify_review(inputs):
    review=json.loads(Path(inputs['rom_review']).read_text())
    if (review.get('schema')!='ns-obstacle-frozen-rom-review-v1' or review.get('passed') is not True or
        review.get('source_sha256')!=source_digest() or review.get('validation_seen') is not False or
        review.get('frozen_models_sha256')!=sha_file(inputs['frozen_models'])):
        raise ValueError('independent exact development-frozen ROM review required before validation')
    require_qualification(review.get('reference_qualification'))
    return review


def eligible_names(frozen,names):
    if not names or len(set(names))!=len(names) or any(name not in frozen['models'] for name in names):raise ValueError('explicit unique frozen models required')
    accepted=[];skipped={}
    for name in names:
        if frozen['models'][name]['kind']=='bank' and frozen['bank_gate'].get('passed') is not True:
            skipped[name]='frozen bank development floor gate failed; validation not evaluated'
        else:accepted.append(name)
    return accepted,skipped


def run(inputs,release,output):
    started=time.perf_counter();review=verify_review(inputs)
    if (release.get('all_development_decisions_frozen') is not True or release.get('no_matched_timing_or_speedup_claim') is not True or
        release.get('future_validation_driven_selection_changes_forbidden') is not True):
        raise ValueError('explicit frozen accuracy-only validation release required')
    arrays,metadata,inner=load_preparation(inputs['prepared'],inputs['preparation_release'])
    q=require_qualification(metadata['reference_qualification'])
    if q!=require_qualification(review['reference_qualification']) or review['reference_admission_sha256']!=metadata['admission_evidence']['reference_admission_sha256']:
        raise ValueError('actual reviewed ROM truth identity differs')
    for key,field in [('bank','bank_sha256'),('probes','probes_sha256')]:
        if sha_file(inputs[key])!=inner[field]:raise ValueError('released '+key+' differs')
    frozen=load_frozen(inputs['frozen_models'],review['frozen_models_sha256'],q)
    names,skipped=eligible_names(frozen,release['model_names'])
    out=Path(output);out.mkdir(exist_ok=False)
    result={'schema':'ns-obstacle-frozen-validation-accuracy-v1','reference_qualification':q,
        'frozen_models_sha256':review['frozen_models_sha256'],'models':{},'skipped_models':skipped,
        'selection_training_or_refitting_performed':False,'timing_comparison_performed':False,'matched_accuracy_or_speedup_claim':False,
        'physical_admission':False,'validation_opened':False}
    try:
        if names:
            expected=release['reference_population_sha256']
            if sha_file(inputs['population'])!=expected:raise ValueError('released validation population differs')
            population_meta=json.loads(Path(inputs['population']).read_text())
            if population_meta.get('reference_admission_sha256')!=review['reference_admission_sha256']:raise ValueError('validation truth identity differs')
            rule=array_digest({k:arrays['volume'][k] for k in ('points_dimensionless','weights_dimensionless')})
            population=load_reference_population(inputs['population'],expected,'validation',rule);result['validation_opened']=True
            bank=FrozenBank.load(inputs['bank']);probes=_read_arrays(inputs['probes'],2**30)
            for name in names:
                model=None
                try:
                    model,fitted=rebuild(frozen,name,arrays,metadata,probes,bank)
                    rows=[]
                    for reference in population:
                        try:row=evaluate_case(model,reference,arrays['boundary'],frozen['bank_gate'],fitted)
                        except Exception as exc:row={'failure':repr(exc),'physical_admission':False}
                        receipt=save_evidence(out,name+'_validation_'+str(reference['Re']),row)
                        rows.append({'Re':reference['Re'],'evidence':receipt})
                    result['models'][name]={'status':'evaluated_pending_independent_review','records':rows,
                        'fingerprint':frozen['models'][name]['fingerprint'],'choice':frozen['choices'][name]}
                except Exception as exc:
                    result['models'][name]={'status':'construction_failed','exception':repr(exc)}
                finally:model=None;gc.collect()
        result['status']='completed_pending_independent_review'
    except Exception as exc:result.update(status='failed',exception=repr(exc));raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        (out/'VALIDATION-ACCURACY.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
