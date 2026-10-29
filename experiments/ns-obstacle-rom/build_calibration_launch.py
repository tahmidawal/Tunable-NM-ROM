"""Build a real-input, unreleased two-candidate common-target screening packet."""
from pathlib import Path
import ast, json
from fe_adapter import sha_file
from qualification import require_qualification
import fom_calibrate2d as calibration
from calibration_launch2d import JOB, IDS

ROOT=Path(__file__).resolve().parent
WORKTREES=ROOT.parents[2]
REFERENCE=WORKTREES/'2026-10-27-ns-obstacle-2d/experiments/ns-obstacle-2d'
COORDINATOR=WORKTREES/'2026-10-27-ns-obstacle/experiments/ns-obstacle'


def sources():
    pending=['calibration_launch2d.py']; found={'MODEL-CONTRACT.json','FD1-SCOPE-CONTRACT.json'}
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
    found.add('fom_native/SOURCE.json')
    found.update('fom_native/'+n for n in json.loads((ROOT/'fom_native/SOURCE.json').read_text())['files'])
    return sorted(found)


def build():
    out=ROOT/'staging/fom-calibration01';out.mkdir(exist_ok=False)
    rows=[]
    def add(path,destination):
        path=Path(path).resolve()
        rows.append({'source':str(path),'destination':destination,'bytes':path.stat().st_size,'sha256':sha_file(path)})
        return {'path':destination,'sha256':sha_file(path)}
    for name in sources():add(ROOT/name,name)
    previous=ROOT/'staging/development-materialize02'
    for name in ('dependencies.py','DEPENDENCIES.json'):add(previous/name,name)
    dependencies=json.loads((previous/'DEPENDENCIES.json').read_text())
    for package in dependencies['candidates'].values():
        wheel=package['cluster_candidate_wheels'][0]
        path=ROOT/'artifacts/training-materialize01/wheels'/wheel['filename']
        assert sha_file(path)==wheel['digests']['sha256'] and path.stat().st_size==wheel['size']
        add(path,'wheels/'+path.name)
    truth=COORDINATOR/'qualification/FD1-REFERENCE-ADMISSION.json'
    p={'schema':'ns-obstacle-fom-development-calibration-plan-v1','development_Re':[16.875,23.125],
       'validation_allowed':False,'selection_frozen':False,'matched_accuracy':False,
       'output_contracts':list(calibration.OUTPUTS),'native_source_sha256':calibration.native_digest(),
       'selection_policy':'minimum_pooled_development_median_all_invocations_eligible_ties_ladder_order',
       'cores':16,'max_common_points':1300000,'common_rule_points':1036497,'repetitions':7,
       'runtime_versions':{'python':'3.13.2','ngsolve':'6.2.2608','numpy':'2.5.0','scipy':'1.18.0','jax':'0.10.2'},
       'volume_rule_sha256':'1b8cb44f65c64b3e2cbfeefc5848bc3f3dc06c2b30b8123797bb6fe5c74b4590',
       'reference_qualification':require_qualification(json.loads(truth.read_text())),
       'truth_admission':add(truth,'inputs/FD1-REFERENCE-ADMISSION.json'),
       'development_population':{'path':str(Path(JOB).parent/'development-materialize02/results/population/development.json'),
          'sha256':'7b206ba1ed7ce10a4a76ab17a035d13fe3da2fc4f5f5334966b4faa7e7934070'},
       'common_geometry':{'path':str(Path(JOB).parent/'bank-fit01/inputs/prepared/volume/geometry.npz'),
          'sha256':'45bfc1aef3f47975dc3116da9429feb3eb0a7d4232d42fa8da33ab267beb6bcc'},
       'candidates':[], 'scope':'Two tested mesh/order configurations at fixed nonlinear tolerance 1e-10; no stopping-tolerance optimization or matched-accuracy claim.'}
    for identifier,case in zip(IDS,('artifacts/g2-ladder02/results/dfg_h2','artifacts/g2-p4trend/results/dfg_h1')):
        folder=REFERENCE/case
        c={'id':identifier,'field_rule_mode':'certified_requested_points','runtime_workspace_reserve_bytes':4*2**30}
        for key,name in (('config','config.json'),('mesh','mesh.vol.bin'),('layout','fe_layout.json'),('boundary','boundary_rule.json')):
            c[key]=add(folder/name,'inputs/candidates/'+identifier+'/'+name)
        c['eligibility']=add(COORDINATOR/'qualification/fom-development01'/f'{identifier}.json',f'inputs/eligibility/{identifier}.json')
        for name in ('control.json','lift_scaling_arrays.npz','common_stokes_lift.json','common_stokes_lift.npz'):
            add(ROOT/'artifacts/candidate-lift01/results/control'/identifier/name,f'inputs/eligibility/{identifier}/{name}')
        p['candidates'].append(c)
    for name in ('FOM-CANDIDATE-LIFT-RESULT-REVIEW.json','FOM-DEVELOPMENT-ELIGIBILITY-PUBLISHER-REVIEW.json'):
        add(COORDINATOR/'reviews'/name,'inputs/reviews/'+name)
    add(COORDINATOR/'reviews/development-materialize02/POPULATION-REVIEW.json','inputs/reviews/POPULATION-REVIEW.json')
    planning=REFERENCE/'checks/fom-calibration-plan/PLAN.json'
    p['prospective_runtime_planning']=add(planning,'inputs/PROSPECTIVE-RUNTIME-PLAN.json')
    (out/'CALIBRATION-PLAN.json').write_text(json.dumps(p,indent=2)+'\n')
    outer=(previous/'native.sh').read_text().replace('ctol_nsobsrom_devdata','ctol_nsobsrom_fomcal').replace('00:02:00','00:36:00').replace('kill-after=5s 110s','kill-after=10s 2145s')
    (out/'native.sh').write_text(outer)
    body=(previous/'native-body.sh').read_text().replace('development-materialize02','fom-calibration01')
    body=body.replace('development_handoff_launch2d.py','calibration_launch2d.py').replace('HANDOFF-PLAN.json','CALIBRATION-PLAN.json').replace('HANDOFF-RELEASE.json','CALIBRATION-RELEASE.json').replace('INPUT-INVENTORY.json','RUNTIME-INVENTORY.json')
    body=body.replace('results/population/HANDOFF.json','results/calibration/WORKLOAD-COMPLETENESS.json').replace("r.get('status')=='completed_pending_independent_handoff_review'","r.get('complete_planned_workload') is True").replace('materialization_completed','calibration_completed').replace('14*2**30','20*2**30').replace('four-GiB','ten-GiB').replace('JOB_PHASE=native_interface','JOB_PHASE=common_target_screening')
    (out/'native-body.sh').write_text(body)
    (out/'REPAIR.json').write_text(json.dumps({'fresh_attempt':'fom-calibration01','purpose':'Fixed two-candidate development common-target screening only'})+'\n')
    for name in ('native.sh','native-body.sh','REPAIR.json'):add(out/name,name)
    assert len({r['destination'] for r in rows})==len(rows)
    inventory={'schema':'ns-obstacle-fom-screen-runtime-inventory-v1','files':rows,'builder_sha256':sha_file(__file__)}
    (out/'RUNTIME-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    proposal={'execution_released':False,'job_directory':JOB,'allocation_seconds':2160,
       'outer_timeout_seconds':2145,'outer_kill_grace_seconds':10,'finalization_reserve_seconds':180,
       'original_fom_reservation_seconds':2400,'prior_candidate_control_reservation_seconds':240,
       'launcher_sha256':sha_file(ROOT/'calibration_launch2d.py'),'source_sha256':calibration.source_digest(),
       'plan_sha256':sha_file(out/'CALIBRATION-PLAN.json'),'runtime_inventory_sha256':sha_file(out/'RUNTIME-INVENTORY.json'),
       'input_bytes':sum(r['bytes'] for r in rows),'model_development_future_envelope_bytes':12*2**30,
       'calibration_output_archive_envelope_bytes':10*2**30,'miscellaneous_reserved_bytes':256*2**20,
       'global_namespace_limit_bytes':50*2**30,'local_collection_requires_separate_actual_measurement':True,
       'automatic_retry':False,'validation_allowed':False,'matched_accuracy':False}
    n=p['common_rule_points'];max_dofs=61172
    components={'full_field_timed_arrays':4*7*3*n*8,
       'coefficient_state_allowance_four_copies_all_96_calls':96*4*max_dofs*8,
       'four_full_scalar_controls':4*13*n*8,
       'two_mapping_array_sets':2*11*n*8,
       'boundary_controls_and_small_outputs':32*2**20,
       'mapping_failure_and_solver_metadata_allowance':2*2**30,
       'isolated_dependencies_allowance':2**30,
       'copied_input_bytes':proposal['input_bytes']}
    raw=sum(components.values());archive=raw+raw//50+64*2**20
    assert raw+archive<proposal['calibration_output_archive_envelope_bytes']
    proposal['storage_projection']={'components':components,'raw_bytes':raw,'archive_bytes_without_compression_credit':archive,
       'combined_bytes':raw+archive,'no_array_deduplication_credit':True,
       'metadata_allowance_is_a_reservation_not_measured_mapping_evidence':True,
       'existing_shared_geometry_and_population_already_in_namespace_total':True}
    (out/'PROPOSAL.json').write_text(json.dumps(proposal,indent=2)+'\n')
    return proposal


if __name__=='__main__':print(json.dumps(build(),indent=2))
