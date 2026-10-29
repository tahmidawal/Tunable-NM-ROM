"""Freeze actual partial-run recovery inputs without running or authorizing recovery."""
from pathlib import Path
import ast,json
from fe_adapter import sha_file
import pilot2d,stage2d
from pod_recovery2d import JOB,METHODS

ROOT=Path(__file__).resolve().parent


def sources():
    pending=['pod_recovery2d.py'];found={'MODEL-CONTRACT.json','FD1-SCOPE-CONTRACT.json'}
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[r.name.replace('.','/')+'.py' for r in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module.replace('.','/')+'.py']
            elif isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py'):names=[node.value]
            pending.extend(n for n in names if not Path(n).is_absolute() and '..' not in Path(n).parts and (ROOT/n).is_file())
    return sorted(found)


def build():
    out=ROOT/'staging/pod-recovery01';out.mkdir(exist_ok=False)
    old=ROOT/'staging/model-development01';remote=Path(JOB).parent/'model-development01'
    terminal=json.loads((ROOT/'checks/model-development-terminal.json').read_text())
    archive=json.loads((ROOT/'checks/model-development-archive-measurement.json').read_text())
    originals=[{'name':Path(row['path']).name,'sha256':row['sha256'],'bytes':row['bytes']} for row in terminal['outputs']]
    p={'schema':'ns-obstacle-pod-control-recovery-plan-v1','build_prefixes':[2,4,8],
       'new_query_prefixes':[4,8],'development_Re':list(pilot2d.DEV_RE),'methods':sorted(METHODS),
       'original_candidate':{'path':str(remote/'STAGE-CANDIDATE.json'),'sha256':sha_file(old/'STAGE-CANDIDATE.json')},
       'original_release':{'path':str(remote/'STAGE-RELEASE.json'),'sha256':sha_file(old/'STAGE-RELEASE.json')},
       'original_output_directory':str(remote/'results/development-stage/development'),'original_outputs':originals,
       'original_archive':archive,'original_full_operator_fingerprints_available':False,
       'original_bank_gate_must_stay_closed':True,'quadrature_refit_allowed':False,'validation_allowed':False,
       'reuse_r2_without_new_solver_invocations':True,'source_sha256':pilot2d.source_digest(),
       'fresh_fingerprints_only_after_exact_available_original_bindings':True}
    (out/'RECOVERY-PLAN.json').write_text(json.dumps(p,indent=2)+'\n')
    outer=(old/'native.sh').read_text().replace('ctol_nsobsrom_development','ctol_nsobsrom_podrecover').replace('00:56:00','00:20:00').replace('3345s','1185s')
    # Same CPU host as original construction reduces incidental hardware-dependent
    # floating arithmetic; exact checks still reject any actual difference.
    outer=outer.replace('#SBATCH --partition=gpu','#SBATCH --partition=gpu\n#SBATCH --nodelist=pax023')
    (out/'native.sh').write_text(outer)
    body=(old/'native-body.sh').read_text().replace('development_launch2d.py','pod_recovery2d.py').replace('model-development01','pod-recovery01')
    body=body.replace('--candidate STAGE-CANDIDATE.json --release STAGE-RELEASE.json','--plan RECOVERY-PLAN.json --release RECOVERY-RELEASE.json')
    body=body.replace('results/development-stage/STAGE-RESULT.json','results/recovery/RECOVERY.json').replace('execution_completed_pending_independent_review','completed_unsampled_controls_pending_independent_review').replace('JOB_PHASE=frozen_model_development','JOB_PHASE=missing_unsampled_pod_controls')
    body=body.replace('sha256sum -c SOURCE-SHA256SUMS','export ROM_ALLOCATION_START_UNIX_SECONDS=$("$PY" -c "import time; print(time.time())")\nsha256sum -c SOURCE-SHA256SUMS')
    (out/'native-body.sh').write_text(body)
    rows=[{'source':str(ROOT/name),'destination':name,'bytes':(ROOT/name).stat().st_size,'sha256':sha_file(ROOT/name)} for name in sources()]
    rows += [{'source':str(out/name),'destination':name,'bytes':(out/name).stat().st_size,'sha256':sha_file(out/name)} for name in ('native.sh','native-body.sh')]
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps({'files':rows,'builder_sha256':sha_file(__file__)},indent=2)+'\n')
    n=1036497;retained_queries=12;fields=retained_queries*3*n*8
    reused=sum(x['bytes'] for x in originals);raw=2*fields+reused+128*2**20
    archived=raw+raw//50+64*2**20
    assert raw+archived<3*2**30
    proposal={'execution_released':False,'job_directory':JOB,'allocation_seconds':1200,
        'outer_timeout_seconds':1185,'outer_kill_grace_seconds':10,'finalization_reserve_seconds':60,
        'failure_reserve_debit_seconds':1200,'launcher_sha256':sha_file(ROOT/'pod_recovery2d.py'),
        'pilot_source_sha256':pilot2d.source_digest(),'stage_source_sha256':stage2d.source_digest(),
        'plan_sha256':sha_file(out/'RECOVERY-PLAN.json'),'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),
        'measured_previous_three_POD_build_seconds':sum(json.loads((ROOT/f'checks/model-development-progress/POD_r{k}_model.json').read_text())['record']['source']['offline_build_seconds'] for k in (2,4,8)),
        'non_build_budget_including_setup_controls_retention_and_finalization_seconds':814.912,
        'new_query_calls':12,'reused_r2_queries_not_rerun':6,'original_allocation_charge_seconds':3346,
        'storage':{'actual_reused_original_bytes':reused,'new_query_fields_bytes':fields,
            'duplicate_query_first_and_case_retention_bytes':2*fields,'metadata_operator_runtime_allowance':128*2**20,
            'raw_bytes':raw,'archive_without_compression_credit':archived,'combined_bytes':raw+archived,'envelope_bytes':3*2**30,
            'no_unmeasured_deduplication_credit':True,'local_collection_separately_released':True},
        'scope':'Complete only missing unsampled POD controls; original failed bank and rejected/incomplete quadrature remain explicit; no matching, validation, retraining, or refit.'}
    proposal['non_build_budget_including_setup_controls_retention_and_finalization_seconds']=1200-proposal['measured_previous_three_POD_build_seconds']
    (out/'PROPOSAL.json').write_text(json.dumps(proposal,indent=2)+'\n')
    return proposal


if __name__=='__main__':print(json.dumps(build(),indent=2))
