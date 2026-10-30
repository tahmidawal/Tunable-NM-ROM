"""Freeze diagnostic orchestration only after actual accepted model descriptors exist."""
from pathlib import Path
import ast,json,argparse,shutil
from fe_adapter import sha_file
from accuracy_v2.diagnostic02_launch import JOB
ROOT=Path(__file__).resolve().parents[1]

def build(output,model_descriptors,model_review):
    model=json.loads(Path(model_descriptors).read_text())
    if set(model)!= {'bank','model_receipt','model_review'} or any(x.get('kind')!='file' or not x.get('sha256') or type(x.get('bytes')) is not int for x in model.values()):raise ValueError('actual reviewed file descriptors required')
    model_review=Path(model_review)
    if sha_file(model_review)!=model['model_review']['sha256'] or model_review.stat().st_size!=model['model_review']['bytes'] or model['model_review']['path']!=JOB+'/MODEL-REVIEW.json':raise ValueError('exact portable actual model review required')
    old=json.loads((ROOT/'staging/model-development01/STAGE-CANDIDATE.json').read_text());base=json.loads((ROOT/'staging/bank-fit01/STAGE-CANDIDATE.json').read_text())
    inputs={k:old['inputs'][k] for k in ('probes','development')};inputs.update({k:base['inputs'][k] for k in ('prepared','preparation_release')});inputs.update(model)
    out=Path(output);out.mkdir(exist_ok=False)
    shutil.copyfile(model_review,out/'MODEL-REVIEW.json')
    plan={'schema':'ns-obstacle-pod-guided-diagnostic-plan-v3','rank':4,'inputs':inputs,'physical_admission':False,'reference_fit_initialization_allowed':False,'validation_allowed':False}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    shell=(ROOT/'staging/neural-pod-guided01/native.sh').read_text().replace('ctol_nsobsrom_teacher','ctol_nsobsrom_teacherdiag').replace('01:00:00','00:15:00').replace('3585s','885s')
    body=(ROOT/'staging/neural-pod-guided01/native-body.sh').read_text().replace('/neural-pod-guided01','/neural-pod-guided-diagnostic02').replace('-m accuracy_v2.launch','-m accuracy_v2.diagnostic02_launch').replace('TRAINING-RELEASE.json','DIAGNOSTIC-RELEASE.json').replace('results/TRAINING-RESULT.json','results/diagnostic/DIAGNOSTIC.json').replace('training_checkpoint_exported_pending_independent_review','completed_diagnostic_pending_independent_review').replace('training_completed','diagnostic_completed').replace('supervised_neural_teacher','successor_neural_weak_LS')
    (out/'native.sh').write_text(shell);(out/'native-body.sh').write_text(body)
    pending=['accuracy_v2/diagnostic02_launch.py'];found={'MODEL-CONTRACT.json','FD1-SCOPE-CONTRACT.json'}
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
    for n in ('native.sh','native-body.sh','MODEL-REVIEW.json'):
        rows.append({'source':str(out/n),'destination':n,'bytes':(out/n).stat().st_size,'sha256':sha_file(out/n)})
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps({'files':rows,'builder_sha256':sha_file(__file__)},indent=2)+'\n')
    n=1036497;raw=2*10*n*8+128*2**20;archive=raw+raw//50+64*2**20
    (out/'PROPOSAL.json').write_text(json.dumps({'allocation_seconds':900,'outer_timeout_seconds':885,'outer_kill_grace_seconds':10,'allocation_memory_bytes':64*2**30,'raw_plus_archive_reservation_bytes':2**30,'conservative_raw_bytes':raw,'archive_without_compression_credit':archive,'no_timing_comparison':True,'collection_requires_separate_measured_release':True,'wide_successor_build_time_unmeasured':True,'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),'plan_sha256':sha_file(out/'PLAN.json')},indent=2)+'\n')
    assert raw+archive<2**30
    return out
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--model-descriptors',type=Path,required=True);a.add_argument('--model-review',type=Path,required=True);a=a.parse_args();print(build(a.output,a.model_descriptors,a.model_review))
