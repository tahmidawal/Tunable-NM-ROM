"""Build an actual two-case development packet without copying bulk input bytes."""
from pathlib import Path
import argparse,json,shutil
import handoff2d
from fe_adapter import sha_file
from build_preparation_launch import source_closure,ROOT
JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/development-materialize01'


def build(reference,out):
    reference=Path(reference).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'.gitignore').write_text('payload/\n')
    truth_path=ROOT/'datasets/fd1-training01/FD1-REFERENCE-ADMISSION.json';truth=json.loads(truth_path.read_text())
    if sha_file(truth_path)!='5c9402af9ddb2af0ad8925ef224a43ae72cb36209df99865908617c63b5b506f':raise ValueError('genuine complete admission required')
    uncertainty=reference/'g2/population01/uncertainty/development.json';u=json.loads(uncertainty.read_text())
    if u['reference_admission_sha256']!=sha_file(truth_path):raise ValueError('actual development rule/admission differs')
    rows=[];cases=[];closures=[]
    for re,identifier in ((16.875,'re16p875'),(23.125,'re23p125')):
        closure_path=reference/'g2/population01/frozen'/identifier/'ADOPTED-RUNTIME-CLOSURE.json';closure=json.loads(closure_path.read_text())
        adopted=handoff2d.adopted_case(truth,re,reference/closure['case_directory'])
        if adopted['split']!='development' or adopted['storage_mode']!='coefficient_primary' or adopted['export_manifest_sha256']!=closure['export_manifest_sha256']:raise ValueError('actual development adopted identity differs')
        for row in closure['files']:
            p=reference/row['source']
            if p.is_symlink() or not p.resolve().is_relative_to(reference) or sha_file(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('exact warm producer input missing/changed')
            if 'archive_source' in row:
                cold=row['archive_source']
                if sha_file(reference/cold['manifest_source'])!=cold['manifest_sha256']:raise ValueError('immutable recovery binding differs')
            rows.append(dict(row,owner='reference',destination='inputs/reference/'+identifier+'/'+row['destination']))
        cases.append({'Re':re,'directory':JOB+'/inputs/reference/'+identifier})
        shutil.copyfile(closure_path,out/('REFERENCE-CLOSURE-'+identifier+'.json'))
        closures.append({'Re':re,'path':str(closure_path),'sha256':sha_file(closure_path)})
    plan={'schema':'ns-obstacle-fd1-handoff-plan-v1','split':'development',
          'reference_admission':{'path':JOB+'/inputs/FD1-REFERENCE-ADMISSION.json','sha256':sha_file(truth_path)},
          'volume_rule_sha256':u['volume_rule_sha256'],'cases':cases}
    (out/'HANDOFF-PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows.append({'owner':'rom','source':str(truth_path.relative_to(ROOT)),'destination':'inputs/FD1-REFERENCE-ADMISSION.json','sha256':sha_file(truth_path),'bytes':truth_path.stat().st_size})
    names=set(source_closure())|{'handoff2d.py','handoff_primary_worker.py','materialize_reference2d.py','validation2d.py','development_handoff_launch2d.py'}
    names |= {'primary_native/'+p.name for p in (ROOT/'primary_native').iterdir() if p.is_file() and (p.suffix=='.py' or p.name=='SOURCE.json')}
    for name in sorted(names):
        p=ROOT/name;rows.append({'owner':'rom','source':name,'destination':name,'sha256':sha_file(p),'bytes':p.stat().st_size})
    template=ROOT/'staging/training-materialize01'
    for name in ('dependencies.py','DEPENDENCIES.json'):shutil.copyfile(template/name,out/name)
    (out/'REPAIR.json').write_text(json.dumps({'fresh_attempt':'development-materialize01','purpose':'Two admitted development primary reconstructions only; no fitting, calibration or validation'})+'\n')
    outer=(template/'native.sh').read_text().replace('ctol_nsobsrom_materialize','ctol_nsobsrom_devdata').replace('00:07:30','00:02:00').replace('435s','110s').replace('--kill-after=10s','--kill-after=5s')
    body=(template/'native-body.sh').read_text().replace('training-materialize01','development-materialize01')
    body=body.replace('results/BATCH-RESULT.json','results/population/HANDOFF.json').replace("'materialization_completed':r.get('passed',False)","'materialization_completed':r.get('status')=='completed_pending_independent_handoff_review'")
    body=body.replace('18*2**30','14*2**30').replace('eight-GiB workspace/archive envelope','four-GiB workspace/archive envelope')
    body=body.replace('JOB_PHASE=gpu_preflight','"$PY" development_handoff_launch2d.py --plan HANDOFF-PLAN.json --release HANDOFF-RELEASE.json --inventory INPUT-INVENTORY.json --verify-only\nJOB_PHASE=gpu_preflight')
    body=body.replace('"$PY" materialize_batch2d.py --plan BATCH-PLAN.json --release BATCH-RELEASE.json --out results','"$PY" development_handoff_launch2d.py --plan HANDOFF-PLAN.json --release HANDOFF-RELEASE.json --inventory INPUT-INVENTORY.json')
    (out/'native.sh').write_text(outer);(out/'native-body.sh').write_text(body)
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name!='.gitignore':rows.append({'owner':'rom','source':str(p.relative_to(ROOT)),'destination':p.name,'sha256':sha_file(p),'bytes':p.stat().st_size})
    if len({r['destination'] for r in rows})!=len(rows):raise ValueError('duplicate staged destination')
    inventory={'schema':'ns-obstacle-development-materialization-input-inventory-v1','files':rows,'handoff_source_sha256':handoff2d.source_digest(),'launcher_sha256':sha_file(ROOT/'development_handoff_launch2d.py'),'builder_sha256':sha_file(__file__),'input_closures':closures}
    (out/'INPUT-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    proposal={'schema':'ns-obstacle-two-case-development-materialization-proposal-v1','execution_released':False,
       'job_directory':JOB,'output_directory':JOB+'/results/population','allocation_seconds':120,'outer_timeout_seconds':110,'outer_kill_grace_seconds':5,
       'debit_from_existing_development_reservation_seconds':120,'remaining_development_reservation_seconds':3480,
       'fom_reservation_unchanged_seconds':2160,'input_bytes':sum(r['bytes'] for r in rows),'input_inventory_sha256':sha_file(out/'INPUT-INVENTORY.json'),
       'handoff_plan_sha256':sha_file(out/'HANDOFF-PLAN.json'),'handoff_source_sha256':handoff2d.source_digest(),'launcher_sha256':sha_file(ROOT/'development_handoff_launch2d.py'),
       'remote_workspace_archive_envelope_bytes':4*2**30,'local_floor_bytes':10*2**30,'other_local_reserved_bytes':256*2**20,
       'staging_proposal':'Direct upload from every already verified canonical warm file; no duplicate bulk local staging tree. Missing/changed input stops before submission. No automatic cold fallback or unreserved cold extraction.',
       'cold_recovery_bindings_preserved':True,'bulk_copied_or_reconstructed':False,'new_native_or_training_work_executed':False,
       'projection_basis':'Actual nine-case materialization103s including setup; two fresh children use the unchanged worker and the same full rule.120s is a hard bound, not a completion guarantee.',
       'output_review_required_before_ROM_or_FOM_use':True,'collection_requires_actual_measurement':True,
       'common_geometry_later_source':{'path':'/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/bank-fit01/inputs/prepared/volume/geometry.npz','sha256':'45bfc1aef3f47975dc3116da9429feb3eb0a7d4232d42fa8da33ab267beb6bcc'},
       'fom_scope':'Later independently reviewed feasible subset of six existing mesh/order configs at fixed nonlinear tolerance1e-10; no stopping-tolerance optimization or unrestricted tuned-FOM claim.'}
    (out/'PROPOSAL.json').write_text(json.dumps(proposal,indent=2)+'\n');return proposal


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(a.reference_root,a.out),indent=2))
