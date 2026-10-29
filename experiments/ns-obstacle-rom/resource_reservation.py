"""Source-only bounded reservations. No runtime/compression/dedup predictions."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).parent
coordinator_commit='762d74044'
coordinator_raw=subprocess.check_output(['git','-c','gc.auto=0','show',coordinator_commit+':experiments/ns-obstacle/reviews/ROM-ALLOCATION-RESERVATION.json'])
coordinator=json.loads(coordinator_raw);schedule=coordinator['population_schedule_proposal']
p=ROOT/'checks/capacity/PRIMARY-RETENTION-PROJECTION.json';base=json.loads(p.read_text())
m=ROOT/'artifacts/native-capacity04/results/interface/result.json';measurement=json.loads(m.read_text())
from pilot2d import DEV_RE,VALIDATION_RE,PREFIXES,QUADRATURE_SUPPORTS
seconds={'fine_preparation':1800,'fixed_bank_allocation':2100,'ROM_development_quadrature':3600,
         'FOM_development_calibration':2400,'validation_and_same_job_paired':3300,'failures':1200}
field=base['per_full_field_output_bytes'];dofs=(measurement['counts']['velocity_dofs']+measurement['counts']['pressure_dofs'])
methods=len(PREFIXES)*6;models=len(PREFIXES)*2
candidates=methods*len(QUADRATURE_SUPPORTS)*len(DEV_RE)
dev=(candidates+methods*len(DEV_RE))*field+base['accuracy_driver_maximum_logical_payload']['selected_fit_column_bytes']
paired_fields=(models+methods)*len(VALIDATION_RE)*7*field
paired_states=models*len(VALIDATION_RE)*7*3*dofs*8+methods*len(VALIDATION_RE)*7*3*24*8
readout_control_per_candidate=len(DEV_RE)*measurement['counts']['volume_points']*13*8
calibration_per_candidate=len(DEV_RE)*7*(field+3*dofs*8)+readout_control_per_candidate+8*2**20
prepared=base['prospective_primary_volume_option']['prepared_retained_after_successful_fine_replay_bytes']
checkpoint_allowance=128*2**20
validation_accuracy_fallback=methods*len(VALIDATION_RE)*field
unique_base=prepared+dev+paired_fields+paired_states+checkpoint_allowance+validation_accuracy_fallback
record={'status':'source_only_reservation_not_execution_release','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'capacity_projection_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'measurement_sha256':hashlib.sha256(m.read_bytes()).hexdigest(),
 'allocation_seconds':seconds,'conditional_ROM_subcap_seconds':14400,'total_ROM_reservation_seconds':sum(seconds.values()),
 'fixed_bank_optimizer_seconds':1800,'bank_setup_retention_allowance_seconds':300,
 'coordinator_reservation_commit':coordinator_commit,'coordinator_reservation_sha256':hashlib.sha256(coordinator_raw).hexdigest(),
 'reference_plan_including_spent_seconds':schedule['maximum_infrastructure_total_seconds'],
 'total_plan_seconds':schedule['maximum_infrastructure_total_seconds']+sum(seconds.values()),
 'remaining_reference_infrastructure_headroom_seconds':coordinator['campaign_cap_seconds']-schedule['maximum_infrastructure_total_seconds']-sum(seconds.values()),
 'disk_bytes':{'prepared_primary_after_fine_replay':prepared,'development_audit_maximum':dev,
   'all_registered_paired_full_outputs':paired_fields,'accuracy_only_validation_fallback_maximum':validation_accuracy_fallback,'paired_returned_states_upper_bound':paired_states,
   'training_checkpoint_allowance':checkpoint_allowance,'FOM_calibration_per_candidate':calibration_per_candidate,'native_scalar_readout_per_candidate':readout_control_per_candidate,
   'base_new_archive_only_payload':unique_base,'minimum_free_reserve':10*2**30,
   'fine_remote_original_primary_restored_workspace':base['prospective_primary_volume_option']['remote_original_primary_restored_workspace_bound_bytes']},
 'not_in_disk_payload':['JSON/ZIP headers and Git transport workspaces','native FOM lookup/certification arrays for each actual constructed candidate','candidate Stokes-control evidence not already retained by reference lane','reference populations and common immutable inputs already counted in the reference plan'],
 'calibration_ladder_scenarios':[{'candidate_count':count,'new_archive_only_payload_bytes':unique_base+count*calibration_per_candidate,
   'payload_plus_reserve_before_transport_bytes':unique_base+count*calibration_per_candidate+10*2**30} for count in (2,4,6)],
 'local_free_bytes_at_generation':shutil.disk_usage(ROOT).free,
 'disk_flow':['Keep actual FD1 input/cache hashes and paths warm; materialize derived full-rule/prepared fields only in fresh allocated jobs.',
  'Fine preparation must complete exact primary replay; originals remain until independent acceptance.',
  'After review, development raw fields/failed candidates/fit evidence and all paired outputs/states are audit-only complete archives; do not also commit their expanded arrays.',
  'Keep frozen operator/model decisions, bank/checkpoint metadata and active preparation manifests warm.',
  'Collect bounded per-model/per-contract archives, verify every file/chunk and mirror history before retiring only recoverable working duplicates.',
  'Before each next stage reserve its worst-case remaining output and transport workspace above10GiB; stop for capacity review if it does not fit.'],
 'unresolved':['Actual eligible finite FOM ladder size must be frozen; scenarios are not a selected ladder.',
  'Worst-case uncompressed complete paired retention can exceed current free space even with archive-only storage; no blanket full-pipeline disk release is implied.',
  'Transport/Git staging overhead and actual compressed sizes require measured per-stage release accounting; no identical-output or compression savings assumed.',
  'Fine eigensolve/NNLS/runtime costs remain unmeasured. Reservations are hard stops, not successful-completion predictions.'],
 'failure_policy':'No automatic retry, fit reset or numerical relaxation. Any later attempt requires explicit release within remaining original budgets.',
 'physical_admission':False,'training_executed':False}
assert record['total_ROM_reservation_seconds']==14400
print(json.dumps(record,indent=2))
