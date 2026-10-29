"""Select a FOM per frozen bank target using reviewed development evidence and A1.

This consumes scalar metadata already bound by independent raw-evidence reviews. It
does not reopen validation, fit anything, re-solve a state or re-audit cold raw arrays.
"""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
from fe_adapter import sha_file
from qualification import require_qualification
from pilot2d import DEV_RE,OUTPUTS,source_digest as pilot_digest
from frozen_rom2d import load_frozen
from fom_comparator2d import checked_file,read_json,native_digest
from fom_calibrate2d import source_digest as calibration_digest
from uncertainty2d import load as load_uncertainty,joint_match,digest,METRICS

INPUTS=('calibration','calibration_plan','calibration_review','frozen_models','development_summary','rom_review',
        'truth_admission','uncertainty','uncertainty_review')


def source_digest():
    root=Path(__file__).parent
    return hashlib.sha256(json.dumps({'entry':sha_file(__file__),'uncertainty':sha_file(root/'uncertainty2d.py'),
        'calibration':calibration_digest(),'pilot':pilot_digest()},sort_keys=True).encode()).hexdigest()


def scalar_record(path,expected):
    """Read reviewed scalars without claiming another audit of archived arrays."""
    if sha_file(path)!=expected:raise ValueError('reviewed scalar evidence metadata differs')
    wrapped=read_json(path);digest(wrapped['array_sha256'])
    if Path(wrapped['array_file']).name!=wrapped['array_file']:raise ValueError('local reviewed sidecar identity required')
    return wrapped['record']


def receipt_record(base,receipt):
    if Path(receipt['record_file']).name!=receipt['record_file']:raise ValueError('local reviewed record identity required')
    path=base/receipt['record_file'];record=scalar_record(path,receipt['record_sha256'])
    wrapped=read_json(path)
    if wrapped['array_file']!=receipt['array_file'] or wrapped['array_sha256']!=receipt['array_sha256']:raise ValueError('reviewed sidecar receipt differs')
    return record


def verify(plan_path,release_path):
    plan_path=Path(plan_path);p=read_json(plan_path);r=read_json(release_path)
    if (p.get('schema')!='ns-obstacle-bank-matched-fom-selection-plan-v1' or p.get('validation_allowed') is not False or
        set(p.get('inputs',{}))!=set(INPUTS) or r.get('schema')!='ns-obstacle-bank-matched-fom-selection-release-v1' or
        r.get('plan_sha256')!=sha_file(plan_path) or r.get('source_sha256')!=source_digest() or r.get('authorized_stage')!='development_bank_matching'):
        raise ValueError('exact reviewed development-only matching release required')
    files={key:checked_file(plan_path.parent,p['inputs'][key]) for key in INPUTS}
    truth_sha=sha_file(files['truth_admission']);q=require_qualification(read_json(files['truth_admission']))
    if require_qualification(r.get('reference_qualification'))!=q:raise ValueError('matching release truth scope differs')
    calibration=read_json(files['calibration']);cp=read_json(files['calibration_plan']);cr=read_json(files['calibration_review'])
    if (calibration.get('status')!='completed_pending_independent_review' or calibration.get('validation_used') is not False or
        calibration.get('plan_sha256')!=sha_file(files['calibration_plan']) or calibration.get('source_sha256')!=calibration_digest() or
        cr.get('schema')!='ns-obstacle-fom-development-calibration-review-v1' or cr.get('passed') is not True or
        cr.get('calibration_sha256')!=sha_file(files['calibration']) or cr.get('source_sha256')!=calibration_digest()):
        raise ValueError('actual exact-source reviewed development calibration required')
    for record in (calibration,cr):
        if require_qualification(record.get('reference_qualification'))!=q:raise ValueError('calibration qualification differs')
    truth_descriptor=checked_file(files['calibration_plan'].parent,cp['truth_admission'])
    if sha_file(truth_descriptor)!=truth_sha or cr.get('reference_admission_sha256')!=truth_sha:raise ValueError('calibration truth identity differs')
    rr=read_json(files['rom_review']);frozen=load_frozen(files['frozen_models'],sha_file(files['frozen_models']),q)
    if (rr.get('schema')!='ns-obstacle-frozen-rom-review-v1' or rr.get('passed') is not True or rr.get('validation_seen') is not False or
        rr.get('frozen_models_sha256')!=sha_file(files['frozen_models']) or rr.get('development_summary_sha256')!=sha_file(files['development_summary']) or
        rr.get('source_sha256')!=pilot_digest() or rr.get('reference_admission_sha256')!=truth_sha or require_qualification(rr.get('reference_qualification'))!=q):
        raise ValueError('independently reviewed exact bank development/freeze required')
    summary=scalar_record(files['development_summary'],sha_file(files['development_summary']))
    if summary.get('validation_seen') is not False or summary.get('source_sha256')!=pilot_digest() or summary['frozen_models']['record_sha256']!=sha_file(files['frozen_models']):
        raise ValueError('bank development summary/freeze mismatch')
    if frozen['input_identity']['reference_admission_sha256']!=truth_sha:raise ValueError('frozen bank truth identity differs')
    identity=frozen.get('development_identity')
    if identity!={'manifest_sha256':cp['development_population']['sha256'],'reference_admission_sha256':truth_sha,'volume_rule_sha256':cp['volume_rule_sha256']}:
        raise ValueError('bank and FOM must use identical actual development reference population/rule')
    targets=p.get('targets',[])
    if not targets or len({(t['model'],t['method']) for t in targets})!=len(targets):raise ValueError('explicit unique bank targets required')
    for target in targets:
        model=frozen['models'][target['model']]
        if model['kind']!='bank' or model['prefix'] not in (2,4,8) or target['method'] not in ('exact_tensor_weak_LS','fitted_convection_weak_LS'):
            raise ValueError('actual registered frozen bank target required')
    delta=load_uncertainty(files['uncertainty'],sha_file(files['uncertainty']),files['uncertainty_review'],sha_file(files['uncertainty_review']),
                          q,truth_sha,cp['volume_rule_sha256'],'development')
    return p,files,q,calibration,cp,frozen,summary,delta


def select(plan_path,release_path,output):
    plan,files,q,calibration,cp,frozen,summary,deltas=verify(plan_path,release_path)
    out=Path(output);out.mkdir(exist_ok=False);written=[]
    candidates=cp['candidates'];ids=[c['id'] for c in candidates]
    if len(set(ids))!=len(ids):raise ValueError('duplicate calibrated candidate')
    block_records={}
    for block in calibration['blocks']:
        key=(block['candidate'],block['Re'],block['contract'])
        if key in block_records or key[0] not in ids or key[1] not in DEV_RE or key[2] not in OUTPUTS:raise ValueError('unexpected calibration block')
        evidence=checked_file(files['calibration'].parent,block['evidence'])
        scored=scalar_record(evidence,block['evidence']['sha256'])
        timing=receipt_record(evidence.parent,scored['timing_record'])
        if timing['contract']!=block['contract'] or len(timing['repetitions'])!=7 or len(scored['same_invocation_scores'])!=7:
            raise ValueError('all actual timed invocation scores required')
        times=[row['elapsed_seconds'] for row in timing['repetitions']]
        if times!=block['elapsed_seconds'] or any(not math.isfinite(t) or t<=0 for t in times):raise ValueError('raw timing metadata differs')
        block_records[key]=(block,scored,times)
    for target in plan['targets']:
        name=target['model'];method=target['method'];model=frozen['models'][name];bank={};bank_evidence={}
        for re in DEV_RE:
            key=name+'_development_'+str(re);receipt=summary['records'].get(key)
            if receipt is None:bank[re]=None;continue
            record=receipt_record(files['development_summary'].parent,receipt);bank[re]=record.get('methods',{}).get(method)
            bank_evidence[str(re)]=receipt
        bank_target={'model':name,'method':method,'prefix':model['prefix'],'fingerprint':model['fingerprint'],
            'frozen_models_sha256':sha_file(files['frozen_models']),'development_summary_sha256':sha_file(files['development_summary']),
            'rom_review_sha256':sha_file(files['rom_review']),'development_records':bank_evidence}
        for contract in OUTPUTS:
            rankings=[]
            for index,candidate in enumerate(candidates):
                cid=candidate['id'];comparisons=[];times=[];eligible=frozen['bank_gate'].get('passed') is True
                # The complete requested-gradient/boundary proof is mandatory even for probe/force timing.
                for re in DEV_RE:
                    full=block_records.get((cid,re,'full_velocity_pressure_fields'))
                    control=None if full is None else full[1].get('readout_control')
                    if not control or control.get('passed') is not True or not all(k in control for k in (
                        'diagnostic_volume_gradient_relative_gap','diagnostic_boundary_velocity_relative_gap')):eligible=False
                    force=block_records.get((cid,re,'integrated_forces'))
                    force_control=None if force is None else force[1].get('readout_control')
                    if not force_control or force_control.get('passed') is not True or not all(k in force_control for k in (
                        'diagnostic_boundary_gradient_relative_gap','diagnostic_boundary_pressure_relative_gap')):eligible=False
                    entry=block_records.get((cid,re,contract));b=bank[re]
                    if entry is None or b is None or b.get('eligible_accuracy') is not True:
                        eligible=False;comparisons.append({'Re':re,'matched':False,'reason':'missing or ineligible actual development invocation'});continue
                    block,scored,elapsed=entry;times.extend(elapsed)
                    eligible &= block.get('eligible') is True and scored.get('readout_control',{}).get('passed') is True
                    for repetition,score in enumerate(scored['same_invocation_scores']):
                        if score.get('eligible_accuracy') is not True:
                            eligible=False;comparisons.append({'Re':re,'repetition':repetition,'matched':False,'reason':'FOM physics/accuracy failed'});continue
                        match=joint_match(b['scores'],score,deltas[re]);eligible &= match['matched']
                        comparisons.append({'Re':re,'repetition':repetition,**match})
                rankings.append({'candidate':cid,'matched':bool(eligible and len(times)==14),
                    'median_seconds':float(np.median(times)) if len(times)==14 else None,'ladder_index':index,'comparisons':comparisons})
            matches=[row for row in rankings if row['matched']]
            winner=min(matches,key=lambda row:(row['median_seconds'],row['ladder_index']))['candidate'] if matches else None
            selection={'schema':'ns-obstacle-fom-development-selection-v1','selection_frozen':True,'development_only':True,
                'validation_used':False,'matched_accuracy':winner is not None,'unresolved_or_no_match':winner is None,
                'development_Re':list(DEV_RE),'output_contract':contract,'selected_candidate':winner,'rankings':rankings,
                'bank_target':bank_target,'uncertainty_sha256':sha_file(files['uncertainty']),'uncertainty_review_sha256':sha_file(files['uncertainty_review']),
                'matching_source_sha256':source_digest(),'matching_plan_sha256':sha_file(plan_path),'matching_release_sha256':sha_file(release_path),
                'reference_qualification':q,'reference_admission_sha256':sha_file(files['truth_admission']),
                'native_source_sha256':native_digest(),'runtime_versions':cp['runtime_versions'],
                'common_rule_points':cp['common_rule_points'],'volume_rule_sha256':cp['volume_rule_sha256'],
                'development_evidence':{'path':str(files['calibration']),'sha256':sha_file(files['calibration'])},
                'calibration_review_sha256':sha_file(files['calibration_review']),
                'raw_arrays_policy':'existing independent reviews bind retained raw sidecars; no new array audit claimed'}
            if winner is not None:
                candidate=next(c for c in candidates if c['id']==winner);selection['field_rule_mode']=candidate['field_rule_mode']
                for key in ('config','mesh','layout','boundary'):
                    path=checked_file(files['calibration_plan'].parent,candidate[key]);selection[key]={'path':str(path),'sha256':sha_file(path)}
                geometry=checked_file(files['calibration_plan'].parent,cp['common_geometry']);selection['common_geometry']={'path':str(geometry),'sha256':sha_file(geometry)}
            path=out/(name+'__'+method+'__'+contract+'.json');path.write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n');written.append(str(path))
    return written


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','release','out'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(select(args.plan,args.release,args.out),indent=2))
