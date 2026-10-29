"""Read-only storage/cost audit; projections never authorize collection or execution."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
ROOT=Path(__file__).parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(ref):
    projection_path=ROOT/'checks/capacity/PRIMARY-RETENTION-PROJECTION.json';p=json.loads(projection_path.read_text())
    resource_path=ref/'g2/population-rest01/RESOURCE-BOUND.json';r=json.loads(resource_path.read_text())
    old=ref/'artifacts/g2-p4trend/results/dfg_h3';new=ref/'artifacts/g2-population-re15-01/results/states/p4_g4_h3_baseline'
    export=json.loads((new/'EXPORT-MANIFEST.json').read_text());runtime_files=[new/k for k in export['runtime_files']]+[new/'EXPORT-MANIFEST.json']
    runtime_bytes=sum(f.stat().st_size for f in runtime_files);runtime_expanded=0
    for f in runtime_files:
        if f.suffix=='.npz':
            with zipfile.ZipFile(f) as z:runtime_expanded+=sum(i.file_size for i in z.infolist())
        else:runtime_expanded+=f.stat().st_size
    from fe_adapter import FILES
    with zipfile.ZipFile(old/'full_rule.npz') as z:full=sum(i.file_size for i in z.infolist())
    other=sum((old/k).stat().st_size for k in FILES if k!='full_rule.npz')
    materialization=9*(full+other)+64*2**20
    prepared=p['prepared_expanded_payload_bytes'];primary=p['prospective_primary_volume_option'];remote=primary['remote_original_primary_restored_workspace_bound_bytes']
    free=shutil.disk_usage(ROOT).free;reserve=10*2**30;scratch=2**30+64*2**20
    return {'schema':'ns-obstacle-handoff-operational-audit-v1','execution_released':False,'numerical_work_performed':False,
      'source_sha256':sha(Path(__file__)),'inputs':{str(projection_path):sha(projection_path),str(resource_path):sha(resource_path),
          str(old/'EXPORT-MANIFEST.json'):sha(old/'EXPORT-MANIFEST.json'),str(new/'EXPORT-MANIFEST.json'):sha(new/'EXPORT-MANIFEST.json')},
      'observed_local_free_bytes':free,'minimum_free_reserve_bytes':reserve,
      'existing_Git_and_working_files':'already included in observed disk usage; do not add them again',
      'canonical_reference_runtime':{'actual_Re15_runtime_file_bytes':runtime_bytes,'actual_Re15_uncompressed_NPZ_and_other_bytes':runtime_expanded,
          'fourteen_same_layout_runtime_projection_bytes':14*runtime_expanded,'projection_not_future_measurement':True,
          'shared_immutable_dedup_credit_assumed':False,'old_Re20_already_retained':True},
      'reference_audit_storage':{'declared_remaining14_persistent_worst_bytes':r['cumulative14_persistent_worst_case_bytes'],
          'all14_worst_case_fits_current_free_above_reserve':r['cumulative14_persistent_worst_case_bytes']<=free-reserve,
          'note':'References are conditionally released per actual collection. This bound includes audit archives/Git, not merely canonical adopted runtime.'},
      'materialization_phase':{'nine_case_shape_projection_bytes':materialization,'basis':'actual legacy uncompressed full_rule+other FE input sizes times9, plus64MiB; not future measured bytes',
          'remote_runtime_inputs_and_derived_outputs':'coexist remotely; input staging is not automatically a second local copy',
          'local_supported_two_copy_collection_plus_scratch_reserve_bytes':2*materialization+scratch+reserve},
      'fine_preparation_remote_disk':{'prepared_originals_bytes':prepared['total'],'primary_bytes':primary['fine_primary_conservative_bytes'],
          'restored_volume_bytes':prepared['volume'],'conservative_block_allowance_bytes':remote-prepared['total']-primary['fine_primary_conservative_bytes']-prepared['volume'],
          'total_bytes':remote,'not_local_disk_or_RAM':True,'RAM_process_guard_bytes':16*2**30},
      'collection_phase':{'completed_remote_archive_size_must_be_measured':True,'proposed_chunk_workspace_bytes':64*2**20,
          'proposed_single_member_audit_workspace_bytes':2**30,'concurrent_collection_or_reference_recovery_permitted':False,
          'supported_peak_model':'archive chunks A + additional Git objects <= A + bounded scratch +10GiB reserve; actual other reservations are additional',
          'supported_no_compression_worst_peak_bytes':2*remote+scratch+reserve,
          'supported_worst_peak_fits_current_free':2*remote+scratch+reserve<=free,
          'largest_archive_allowed_by_current_free_before_other_reservations_bytes':max(0,(free-reserve-scratch)//2),
          'expanded_tree_plus_archive_plus_Git_not_permitted_by_plan':True,
          'incremental_stream_to_Git':'prospective only; requires independently reviewed exact manifest/recovery procedure before using reduced peak credit',
          'proof_required':'all remote file/chunk hashes, ordered archive SHA, exact source/runtime and complete original/restored fine replay; durable source/mirror reachability before retiring duplicates'},
      'after_independent_fine_review':{'warm_primary_plus_nonvolume_prepared_projection_bytes':primary['prepared_retained_after_successful_fine_replay_bytes'],
          'cold_raw_original_restored_evidence_retained':True,'do_not_reexpand_all_cold_evidence_simultaneously':True,
          'unique_data_deletion_authorized':False},
      'cost_reporting':{'online_comparison':'same-process/same-allocation paired_cpu, fixed CPU cores, alternating methods,7 repetitions; accuracy from each actual saved invocation',
          'report_required':['per-contract paired medians and outlier/failure counts','same-job ratios only','finite frozen ladder and conditional accuracy intervals','CPU/GPU identity and job/release provenance'],
          'offline_records':['reference/materialization/preparation actual job receipts','training CONFIG/FINAL/checkpoints','development quadrature and model-build records','FOM calibration/setup records','validation qualification and warmups'],
          'historical_job_seconds':'campaign resource accounting, not transferable benchmark speed',
          'same_allocation_setup':'ROM rebuild and FOM setup recorded separately; excludes historical training/data/qualification',
          'full_break_even':'unresolved without complete valid same-platform offline cost basis; do not combine heterogeneous historical wall times with online savings',
          'nonpositive_paired_saving':'no break-even',
          'final_report_assembler':'not yet implemented; no speedup or break-even result exists'},
      'readiness':'source handoff ready; actual full admission, measured stage/archive peaks, exact disk release and cost-report basis remain required'}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reference-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    result=audit(a.reference_root);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
