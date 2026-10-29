"""Build a review candidate from actual admitted training data; never issue a release."""
from pathlib import Path
import argparse,json,shutil,zipfile
from fe_adapter import read_training_manifest,sha_file,CAPS
from qualification import require_qualification
from prepare2d import preparation_digest
import capacity_context2d as capacity
ROOT=Path(__file__).parent
RECONSTRUCTION_CONTROL_SHA='8dd5c0335fd9a103cd09a4553b05e34fb9145a80f1fd7dc16750328d4a9766f6'


def build(training_manifest,measurement,factor_reuse,native_review,reconstruction_control,out,other_active_reservation_bytes):
    if type(other_active_reservation_bytes) is not int or other_active_reservation_bytes<0:raise ValueError('explicit concurrent disk reservation required')
    training_manifest=Path(training_manifest).resolve();admission=read_training_manifest(training_manifest);q=require_qualification(admission)
    paths={'measurement':Path(measurement).resolve(),'factor_reuse':Path(factor_reuse).resolve(),'native_control_review':Path(native_review).resolve()}
    for key,digest in [('measurement',capacity.MEASUREMENT_SHA),('factor_reuse',capacity.FACTOR_SHA),('native_control_review',capacity.CONTROL_REVIEW_SHA)]:
        if sha_file(paths[key])!=digest:raise ValueError('exact measured/reviewed capacity evidence required')
    measured=json.loads(paths['measurement'].read_text());counts=measured['counts']
    reconstruction_control=Path(reconstruction_control).resolve()
    if sha_file(reconstruction_control)!=RECONSTRUCTION_CONTROL_SHA:raise ValueError('accepted actual fine reconstruction control required')
    replay_control=json.loads(reconstruction_control.read_text())
    observed=replay_control['portable_fresh_process_seconds']
    for case in admission['cases']:
        folder=Path(case['resolved_directory'])
        if case['files']['mesh.vol.bin']!=capacity.MESH_SHA or json.loads((folder/'fe_layout.json').read_text())['sha256']!=capacity.LAYOUT_SHA:
            raise ValueError('actual training mesh/layout not the measured target')
    proposed=dict(CAPS,mixed_dofs=counts['velocity_dofs']+counts['pressure_dofs'],elements=counts['elements'],
        volume_points=counts['volume_points'],matrix_nnz=max(row['native_nnz'] for row in measured['metrics'].values()))
    amendment={'schema':'ns-obstacle-rom-count-capacity-amendment-v1','source_sha256':preparation_digest(),'scientific_changes':False,
        'reference_qualification':q,'reference_admission_sha256':admission['reference_admission_sha256'],
        'original_caps':dict(CAPS),'proposed_caps':proposed,
        **{key:{'path':str(path),'sha256':sha_file(path)} for key,path in paths.items()}}
    input_paths={Path(case['resolved_directory'])/name for case in admission['cases'] for name in case['files']}
    lift=admission['resolved_common_stokes_lift'];lift_archive=Path(lift['archive'])
    input_paths.update([lift_archive,lift_archive.with_name('common_stokes_lift.json'),training_manifest,
        (training_manifest.parent/admission['common_stokes_lift']['admission_path']).resolve(),*paths.values()])
    warm_bytes=sum(path.stat().st_size for path in input_paths)
    expanded_case_bytes=[]
    for case in admission['cases']:
        with zipfile.ZipFile(Path(case['resolved_directory'])/'full_rule.npz') as archive:
            expanded_case_bytes.append(sum(info.file_size for info in archive.infolist()))
    projection_path=ROOT/'checks/capacity/PRIMARY-RETENTION-PROJECTION.json';projection=json.loads(projection_path.read_text())
    workspace=projection['prospective_primary_volume_option']['remote_original_primary_restored_workspace_bound_bytes']
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'CAPACITY-AMENDMENT-CANDIDATE.json').write_text(json.dumps(amendment,indent=2)+'\n')
    record={'schema':'ns-obstacle-first-preparation-packet-candidate-v1','execution_released':False,'physical_admission_created':False,
        'source_sha256':sha_file(__file__),'preparation_source_sha256':preparation_digest(),
        'training_manifest':{'path':str(training_manifest),'sha256':sha_file(training_manifest)},
        'reference_admission_sha256':admission['reference_admission_sha256'],'reference_qualification':q,
        'capacity_amendment_sha256':sha_file(out/'CAPACITY-AMENDMENT-CANDIDATE.json'),
        'mandatory_preparation_options':{'point_block_volume':True,'verify_primary_volume':True},
        'original_ROM_subcap_seconds':14400,'original_preparation_reservation_seconds':1800,
        'prospective_subreservations_seconds':{'nine_case_materialization_including_setup_finalization':450,'fine_preparation_and_replay_including_setup_finalization':1350},
        'runtime_assessment':{'control_sha256':RECONSTRUCTION_CONTROL_SHA,'observed_single_state_fresh_replay_seconds':observed,
            'eight_primary_reconstructions_component_projection_seconds':8*observed,'inference_not_measurement':True,
            'unmeasured':['NPZ writes and hashes','legacy derivation','fine basis/eigensolve/export/replay'],
            'hard_bounds_are_not_completion_guarantees':True},
        'materialization_and_previous_preparation_seconds_must_be_deducted':True,
        'disk':{'actual_warm_scientific_input_file_bytes':warm_bytes,'actual_full_rule_expanded_bytes_by_case':expanded_case_bytes,
            'prospective_original_primary_restored_workspace_bytes':workspace,'projection_sha256':sha_file(projection_path),
            'other_active_reservation_bytes':other_active_reservation_bytes,'minimum_free_reserve_bytes':10*2**30,
            'additional_workspace_plus_reserve_bytes':workspace+other_active_reservation_bytes+10*2**30,
            'fresh_staging_inputs_plus_workspace_and_reserve_bytes':warm_bytes+workspace+other_active_reservation_bytes+10*2**30,
            'local_free_bytes_now':shutil.disk_usage(out).free,
            'not_in_envelope':['native wheel/venv staging','Git transport or complete collection copies','JSON/ZIP overhead and other allocated stage outputs'],
            'compression_or_duplicate_savings_assumed':False},
        'required_before_execution':['independent actual training handoff review','independent exact capacity amendment review',
            'measured job/staging and transport disk packet including listed exclusions','remaining original preparation-time ledger and hard-wall release',
            'exact native preparation release binding training manifest, source, reference and capacity review'],
        'required_after_execution':['independent full fine primary replay and native basis/probe evidence review',
            'actual peak RSS, elapsed time and retained bytes','no derived original/restored omission before review',
            'fresh-consumer release binds actual result and native preparation review']}
    (out/'PREPARATION-PACKET.json').write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    p=argparse.ArgumentParser();
    for name in ('training-manifest','measurement','factor-reuse','native-review','reconstruction-control','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--other-active-reservation-bytes',type=int,required=True);a=p.parse_args()
    print(json.dumps(build(a.training_manifest,a.measurement,a.factor_reuse,a.native_review,a.reconstruction_control,a.out,a.other_active_reservation_bytes),indent=2))
