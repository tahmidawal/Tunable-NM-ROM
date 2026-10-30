"""Freeze diagnostic orchestration only after actual accepted model descriptors exist."""
from pathlib import Path
import ast,json,argparse,shutil
from fe_adapter import sha_file
from accuracy_v2.mixed_galerkin_launch import JOB
ROOT=Path(__file__).resolve().parents[1]

def build(output,model_descriptors,model_review,evaluator_fixture,evaluator_fixture_review,ls_descriptors,ls_review,ls_independent_result):
    model=json.loads(Path(model_descriptors).read_text())
    if set(model)!= {'bank','model_receipt','model_review'} or any(x.get('kind')!='file' or not x.get('sha256') or type(x.get('bytes')) is not int for x in model.values()):raise ValueError('actual reviewed file descriptors required')
    model_review=Path(model_review)
    if sha_file(model_review)!=model['model_review']['sha256'] or model_review.stat().st_size!=model['model_review']['bytes'] or model['model_review']['path']!=JOB+'/MODEL-REVIEW.json':raise ValueError('exact portable actual model review required')
    old=json.loads((ROOT/'staging/model-development01/STAGE-CANDIDATE.json').read_text());base=json.loads((ROOT/'staging/bank-fit01/STAGE-CANDIDATE.json').read_text())
    inputs={k:old['inputs'][k] for k in ('probes','development')};inputs.update({k:base['inputs'][k] for k in ('prepared','preparation_release')});inputs.update(model)
    fixture=Path(evaluator_fixture);fixture_review=Path(evaluator_fixture_review)
    f=json.loads(fixture.read_text());fr=json.loads(fixture_review.read_text())
    if f.get('passed') is not True or f.get('model_sha256')!=model['bank']['sha256'] or fr.get('passed') is not True or fr.get('model_sha256')!=model['bank']['sha256'] or fr.get('fixture_sha256')!=sha_file(fixture):raise ValueError('actual independently accepted composite evaluator fixture required')
    for key,path,name in [('evaluator_fixture',fixture,'EVALUATOR-FIXTURE.json'),('evaluator_fixture_review',fixture_review,'EVALUATOR-FIXTURE-REVIEW.json')]:
        inputs[key]={'kind':'file','path':JOB+'/'+name,'sha256':sha_file(path),'bytes':path.stat().st_size}
    comparison=json.loads(Path(ls_descriptors).read_text());ls_review=Path(ls_review)
    if set(comparison)!= {'same_bank_LS_status','same_bank_LS_review','same_bank_LS_independent_result'} or any(x.get('kind')!='file' or not x.get('sha256') or type(x.get('bytes')) is not int for x in comparison.values()):raise ValueError('exact completed LS comparison descriptors required')
    d=comparison['same_bank_LS_review']
    if d['path']!=JOB+'/SAME-BANK-LS-REVIEW.json' or sha_file(ls_review)!=d['sha256'] or ls_review.stat().st_size!=d['bytes']:raise ValueError('portable exact independent LS review required')
    ls_independent_result=Path(ls_independent_result);d=comparison['same_bank_LS_independent_result']
    if d['path']!=JOB+'/SAME-BANK-LS-INDEPENDENT-RESULT.json' or sha_file(ls_independent_result)!=d['sha256'] or ls_independent_result.stat().st_size!=d['bytes']:raise ValueError('portable accepted independent LS audit required')
    inputs.update(comparison)
    out=Path(output);out.mkdir(exist_ok=False)
    shutil.copyfile(fixture,out/'EVALUATOR-FIXTURE.json');shutil.copyfile(fixture_review,out/'EVALUATOR-FIXTURE-REVIEW.json')
    shutil.copyfile(model_review,out/'MODEL-REVIEW.json');shutil.copyfile(ls_review,out/'SAME-BANK-LS-REVIEW.json');shutil.copyfile(ls_independent_result,out/'SAME-BANK-LS-INDEPENDENT-RESULT.json')
    plan={'schema':'ns-obstacle-composite-mixed-galerkin-diagnostic-plan-v1','rank':4,'inputs':inputs,'physical_admission':False,'reference_fit_initialization_allowed':False,'validation_allowed':False}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    shell=(ROOT/'staging/neural-pod-guided01/native.sh').read_text().replace('ctol_nsobsrom_teacher','ctol_nsobsrom_galerkin').replace('01:00:00','00:15:00').replace('3585s','885s')
    body=(ROOT/'staging/neural-pod-guided01/native-body.sh').read_text().replace('/neural-pod-guided01','/neural-pod-guided-galerkin01').replace('-m accuracy_v2.launch','-m accuracy_v2.mixed_galerkin_launch').replace('TRAINING-RELEASE.json','DIAGNOSTIC-RELEASE.json').replace('results/TRAINING-RESULT.json','results/diagnostic/DIAGNOSTIC.json').replace('training_checkpoint_exported_pending_independent_review','completed_diagnostic_pending_independent_review').replace('training_completed','diagnostic_completed').replace('supervised_neural_teacher','same_bank_mixed_Galerkin')
    (out/'native.sh').write_text(shell);(out/'native-body.sh').write_text(body)
    fixture_modules={'residual_head_bank.py','independent_residual_heads.py','spatial_bank.py','fast_spatial_bank.py'}
    if set(f['source_files'])!=fixture_modules:raise ValueError('exact fixture evaluator source set required')
    for name in fixture_modules:
        if sha_file(ROOT/'accuracy_v2'/name)!=f['source_files'][name]:raise ValueError('fixture evaluator source differs before inventory emission')
    pending=['accuracy_v2/mixed_galerkin_launch.py']+['accuracy_v2/'+name for name in sorted(fixture_modules)];found={'MODEL-CONTRACT.json','FD1-SCOPE-CONTRACT.json'}
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[x.name.replace('.','/')+'.py' for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module.replace('.','/')+'.py']
            elif isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py'):names=[node.value]
            pending.extend(x for x in names if not Path(x).is_absolute() and '..' not in Path(x).parts and (ROOT/x).is_file())
    rows=[{'source':str(ROOT/n),'destination':n,'bytes':(ROOT/n).stat().st_size,'sha256':sha_file(ROOT/n)} for n in sorted(found)]
    for n in ('native.sh','native-body.sh','MODEL-REVIEW.json','EVALUATOR-FIXTURE.json','EVALUATOR-FIXTURE-REVIEW.json','SAME-BANK-LS-REVIEW.json','SAME-BANK-LS-INDEPENDENT-RESULT.json'):
        rows.append({'source':str(out/n),'destination':n,'bytes':(out/n).stat().st_size,'sha256':sha_file(out/n)})
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps({'files':rows,'builder_sha256':sha_file(__file__)},indent=2)+'\n')
    n=1036497;raw=2*10*n*8+128*2**20;archive=raw+raw//50+64*2**20
    (out/'PROPOSAL.json').write_text(json.dumps({'allocation_seconds':900,'outer_timeout_seconds':885,'outer_kill_grace_seconds':10,'allocation_memory_bytes':64*2**30,'raw_plus_archive_reservation_bytes':2**30,'conservative_raw_bytes':raw,'archive_without_compression_credit':archive,'no_timing_comparison':True,'collection_requires_separate_measured_release':True,'wide_successor_build_time_unmeasured':True,'formulation':'mixed_Galerkin','same_bank_LS_job_id':6658937,'momentum_rows':8,'continuity_rows':4,'construction_point_block':4096,'additional_extended_array_payload_bytes':1769472,'additional_setup_headroom_estimate_bytes':64*2**20,'peak_RSS_measured':False,'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),'plan_sha256':sha_file(out/'PLAN.json')},indent=2)+'\n')
    assert raw+archive<2**30
    return out
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--model-descriptors',type=Path,required=True);a.add_argument('--model-review',type=Path,required=True);a.add_argument('--evaluator-fixture',type=Path,required=True);a.add_argument('--evaluator-fixture-review',type=Path,required=True);a.add_argument('--ls-descriptors',type=Path,required=True);a.add_argument('--ls-review',type=Path,required=True);a.add_argument('--ls-independent-result',type=Path,required=True);a=a.parse_args();print(build(a.output,a.model_descriptors,a.model_review,a.evaluator_fixture,a.evaluator_fixture_review,a.ls_descriptors,a.ls_review,a.ls_independent_result))
