"""Build the user-authorized diagnostic packet, without executing the solver."""
from pathlib import Path
import ast,json
from fe_adapter import sha_file
import pilot2d,stage2d
from neural_diagnostic2d import JOB
ROOT=Path(__file__).resolve().parent


def sources():
    pending=['neural_diagnostic2d.py'];found={'MODEL-CONTRACT.json','FD1-SCOPE-CONTRACT.json'}
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
    out=ROOT/'staging/neural-diagnostic01';out.mkdir(exist_ok=False)
    old=ROOT/'staging/model-development01';remote=Path(JOB).parent/'model-development01'
    terminal=json.loads((ROOT/'checks/model-development-terminal.json').read_text())
    names={s+e for s in ('bank_r8_model','development_floors') for e in ('.json','.npz')}
    originals=[{'name':Path(row['path']).name,'sha256':row['sha256'],'bytes':row['bytes']} for row in terminal['outputs'] if Path(row['path']).name in names]
    amendment=ROOT.parents[2]/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/neural-diagnostic01/SCOPE-AMENDMENT.json'
    p={'schema':'ns-obstacle-neural-diagnostic-plan-v1','prefix':8,'development_Re':list(pilot2d.DEV_RE),
       'original_candidate':{'path':str(remote/'STAGE-CANDIDATE.json'),'sha256':sha_file(old/'STAGE-CANDIDATE.json')},
       'original_release':{'path':str(remote/'STAGE-RELEASE.json'),'sha256':sha_file(old/'STAGE-RELEASE.json')},
       'original_output_directory':str(remote/'results/development-stage/development'),'original_outputs':originals,
       'scope_amendment':{'path':JOB+'/SCOPE-AMENDMENT.json','sha256':sha_file(amendment)},
       'source_sha256':pilot2d.source_digest(),'original_bank_gate_must_stay_false':True,
       'reference_fit_initialization_allowed':False,'full_operator_bitwise_equality_claim':False,
       'fresh_operator_fingerprint_retained':True,'validation_allowed':False}
    (out/'DIAGNOSTIC-PLAN.json').write_text(json.dumps(p,indent=2)+'\n')
    outer=(old/'native.sh').read_text().replace('ctol_nsobsrom_development','ctol_nsobsrom_neuraldiag').replace('00:56:00','00:15:00').replace('3345s','885s')
    (out/'native.sh').write_text(outer)
    body=(old/'native-body.sh').read_text().replace('development_launch2d.py','neural_diagnostic2d.py').replace('model-development01','neural-diagnostic01')
    body=body.replace('--candidate STAGE-CANDIDATE.json --release STAGE-RELEASE.json','--plan DIAGNOSTIC-PLAN.json --release DIAGNOSTIC-RELEASE.json')
    body=body.replace('results/development-stage/STAGE-RESULT.json','results/diagnostic/DIAGNOSTIC.json').replace('execution_completed_pending_independent_review','completed_diagnostic_pending_independent_review').replace('development_completed','diagnostic_completed').replace('JOB_PHASE=frozen_model_development','JOB_PHASE=diagnostic_neural_weak_LS')
    body=body.replace('sha256sum -c SOURCE-SHA256SUMS','export ROM_ALLOCATION_START_UNIX_SECONDS=$("$PY" -c "import time; print(time.time())")\nsha256sum -c SOURCE-SHA256SUMS')
    (out/'native-body.sh').write_text(body)
    rows=[{'source':str(ROOT/name),'destination':name,'bytes':(ROOT/name).stat().st_size,'sha256':sha_file(ROOT/name)} for name in sources()]
    rows += [{'source':str(out/name),'destination':name,'bytes':(out/name).stat().st_size,'sha256':sha_file(out/name)} for name in ('native.sh','native-body.sh')]
    rows.append({'source':str(amendment),'destination':'SCOPE-AMENDMENT.json','bytes':amendment.stat().st_size,'sha256':sha_file(amendment)})
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps({'files':rows,'builder_sha256':sha_file(__file__)},indent=2)+'\n')
    n=1036497;raw=2*(3+7)*n*8+128*2**20;archive=raw+raw//50+64*2**20;assert raw+archive<2**30
    previous=json.loads((ROOT/'checks/model-development-progress/bank_r8_model.json').read_text())['record']['source']['offline_build_seconds']
    proposal={'execution_released':False,'job_directory':JOB,'allocation_seconds':900,'outer_timeout_seconds':885,
       'outer_kill_grace_seconds':10,'finalization_reserve_seconds':60,'launcher_sha256':sha_file(ROOT/'neural_diagnostic2d.py'),
       'pilot_source_sha256':pilot2d.source_digest(),'stage_source_sha256':stage2d.source_digest(),
       'plan_sha256':sha_file(out/'DIAGNOSTIC-PLAN.json'),'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),
       'scope_amendment_sha256':sha_file(amendment),'input_bytes':sum(x['bytes'] for x in rows),
       'measured_previous_bank_r8_build_seconds':previous,'remaining_nonbuild_seconds':900-previous,
       'new_exact_weak_LS_queries':2,'original_gate_passed':False,'no_cross_job_timing_comparison':True,
       'storage':{'raw_bytes':raw,'archive_without_compression_credit':archive,'combined_bytes':raw+archive,'envelope_bytes':2**30,
          'full_raw_invocations_and_gradient_fields_retained':True,'metadata_boundary_operator_runtime_allowance':128*2**20,
          'local_collection_separately_released':True},'local_floor_bytes':10*2**30,'other_local_reserved_bytes':256*2**20,
       'pod_recovery_future_envelope_bytes':3*2**30,'other_future_FOM_allocation_released':False,
       'scope':'User-authorized diagnostic exception only. Existing bank, exact solver, zero/Stokes initializer, inf-sup and checks unchanged; no training, quadrature fit, validation or admission.'}
    (out/'PROPOSAL.json').write_text(json.dumps(proposal,indent=2)+'\n');return proposal


if __name__=='__main__':print(json.dumps(build(),indent=2))
