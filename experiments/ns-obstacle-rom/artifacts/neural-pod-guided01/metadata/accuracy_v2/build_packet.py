"""Freeze a reviewable first supervised-neural packet; never submits or admits data."""
from pathlib import Path
import argparse,json,shutil
from build_bank_fit_launch import bank_sources
from fe_adapter import sha_file
from pilot2d import source_digest
ROOT=Path(__file__).resolve().parents[1]
JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided01'

def build(out,scope):
    out=Path(out).resolve();out.mkdir(exist_ok=False)
    prior=json.loads((ROOT/'staging/bank-fit01/STAGE-CANDIDATE.json').read_text())
    metadata=json.loads((ROOT/'accuracy_v2/prepared-metadata.json').read_text());config=json.loads((ROOT/'accuracy_v2/CONFIG.json').read_text())
    shutil.copyfile(scope,out/'SCOPE-AMENDMENT.json');shutil.copyfile(ROOT/'accuracy_v2/CONFIG.json',out/'CONFIG.json')
    plan={'schema':'ns-obstacle-pod-guided-training-plan-v2','job_directory':JOB,'inputs':prior['inputs'],'reference_qualification':prior['reference_qualification'],'prepared_manifest_sha256':metadata['manifest_sha256'],'teacher_singular_values':metadata['pod']['velocity']['small_R_singular_values_padded'][:4],'config':config,'config_sha256':sha_file(out/'CONFIG.json'),'scope_amendment_sha256':sha_file(out/'SCOPE-AMENDMENT.json'),'pilot_source_sha256':source_digest(),'resume_allowed':False,'validation_allowed':False,'physical_admission':False}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    shell=(ROOT/'staging/bank-fit01/native.sh').read_text().replace('ctol_nsobsrom_bank','ctol_nsobsrom_teacher').replace('00:35:00','01:00:00').replace('2085s','3585s')
    body=(ROOT/'staging/bank-fit01/native-body.sh').read_text().replace('/bank-fit01','/neural-pod-guided01').replace('bank_fit_launch2d.py --candidate STAGE-CANDIDATE.json --release STAGE-RELEASE.json','-m accuracy_v2.launch --plan PLAN.json --release TRAINING-RELEASE.json').replace('results/fit/STAGE-RESULT.json','results/TRAINING-RESULT.json').replace('execution_completed_pending_independent_review','training_checkpoint_exported_pending_independent_review').replace('fixed_bank_fit','supervised_neural_teacher')
    body=body.replace('export JAX_DEFAULT_MATMUL_PRECISION=', 'export ROM_ALLOCATION_START_UNIX_SECONDS=$(/usr/bin/date +%s)\nexport JAX_DEFAULT_MATMUL_PRECISION=')
    (out/'native.sh').write_text(shell);(out/'native-body.sh').write_text(body)
    names=set(bank_sources())|{str(p.relative_to(ROOT)) for p in (ROOT/'accuracy_v2').glob('*.py')}
    rows=[{'source':name,'destination':name,'bytes':(ROOT/name).stat().st_size,'sha256':sha_file(ROOT/name)} for name in sorted(names)]
    for name in ('CONFIG.json','SCOPE-AMENDMENT.json','native.sh','native-body.sh'):
        p=out/name;rows.append({'source':str(p.relative_to(ROOT)),'destination':name,'bytes':p.stat().st_size,'sha256':sha_file(p)})
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps({'schema':'ns-obstacle-runtime-inventory-v1','files':rows},indent=2)+'\n')
    (out/'SOURCE-PLAN.json').write_text(json.dumps({'status':'source/input candidate only; no execution release','plan_sha256':sha_file(out/'PLAN.json'),'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),'launcher_sha256':sha_file(ROOT/'accuracy_v2/launch.py'),'runtime_files':len(rows),'staged_bytes':sum(r['bytes'] for r in rows),'prepared_inputs_are_existing_immutable_remote_tree':True,'prepared_bytes':sum(r['bytes'] for r in prior['inputs']['prepared']['files'].values()),'allocation_seconds':3600,'outer_timeout_seconds':3585,'outer_kill_grace_seconds':10,'raw_plus_archive_reservation_bytes':2**30,'resume_allowed':False,'validation_allowed':False},indent=2)+'\n')
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--scope',type=Path,required=True);a=p.parse_args();print(build(a.output,a.scope))
