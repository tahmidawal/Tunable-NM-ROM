"""Prospective shape accounting, not measured RSS or execution authorization."""
import argparse,json,hashlib
from pathlib import Path

def estimate(n):
 rows=1+2+9*(2+4+64); b=8; block=min(n,16384)
 gpu_primary={'input':2,'envelope':1,'flattened_weights':2,'target_nine_states':18,
  'two_hidden_activations':128,'two_hidden_reverse_arrays':128,'raw_output':16,
  'weighted_output':16,'thin_Q':16,'projection_residual':18,'QR_reverse_arrays':32}
 gpu_workspace={'three_additional_hidden_array_pairs':384,'two_Q_copies':32,'residual_copy':18}
 volume={'geometry':9,'lift_u':6,'lift_p_per_nu':3,'pod_velocity':8*6,
  'pressure':8*3,'supremizers':8*6,'momentum_tests':64*6,'continuity_tests':16*3}
 arrays={name:count*n*b for name,count in volume.items()}
 return {'points':n,'source':'projected point count; replace with actual retained rule count',
  'volume_payload_components_bytes':arrays,'volume_expanded_archive_bytes':sum(arrays.values()),
  'largest_volume_array_momentum_gradients_bytes':n*2*2*64*b,
  'training_expanded_archive_bytes':n*(5+9*(2+4+1))*b+9*b,
  'maximum_velocity_candidate_values_and_gradients_bytes':n*(96+8)*6*b,
  'maximum_velocity_candidate_gradient_single_array_bytes':n*(96+8)*4*b,
  'blocked_candidate_TSQR_workspace_envelope_bytes':8*4096*6*104*b,
  'blocked_candidate_FE_coefficient_storage_not_included':'104 times actual velocity DOF count times 8 bytes, plus native GridFunction ownership; measure on selected mesh',
  'maximum_pressure_candidate_values_and_gradients_bytes':n*(1+8+32)*3*b,
  'legacy_six_models_retained_enriched_velocity_fields_bytes':n*2*sum((2,4,8))*2*6*b,
  'current_one_model_max_trial_fields_bytes':n*(16*6+8*3+9)*b,
  'bank_evaluation_conservative_temporary_envelope_bytes':n*64*2*b*8,
  'dense_moment_rows':rows,'dense_moment_bytes':rows*n*b,
  'legacy_disk_moment_matrix_bytes_not_used_by_current_RAM_fitter':rows*n*b,
  'current_RAM_moment_matrix_bytes':rows*n*b,
  'blocked_fit_resident_workspace_envelope_bytes':4*rows*block*b+4*n*b+4*rows*2560*b,
  'sparse_metric_capacity':{'matrix_nnz_cap':12000000,'CSR_bytes_per_matrix_at_cap':12000000*12+100001*4,
     'retained_metric_matrices':6,'factor_estimate_cap_bytes':8*2**30},
  'remaining_limits':{'point_cap_current':100000,'archive_expanded_cap_current_bytes':2**30,
     'single_array_cap_unchanged_bytes':2**30,'process_cap_unchanged_bytes':16*2**30},
  'required_before_production':['separately reviewed measured point and DOF ceilings',
    'native fine-population integration and measured phase RSS/GPU memory/time',
    'exact retained artifact, archive and Git duplication disk envelope',
    'actual full-support NNLS elapsed/memory measurement'],
  'implemented_source_capacity_changes':[
    'point-blocked volume archives with original hashes/units; per-array split insufficient at h3',
    'mapped-point block candidate evaluation and all-point TSQR selection',
    'one model at a time and blocked bank derivative evaluation',
    'exact RAM-block moment fitting with immutable input provenance and per-block hashes'],
  'training_memory_shape_accounting':{
    'full_GPU_QR_objective_unchanged':True,
    'GPU_primary_array_projection_bytes':{key:value*n*b for key,value in gpu_primary.items()},
    'GPU_primary_forward_and_reverse_arrays_estimate_bytes':sum(gpu_primary.values())*n*b,
    'GPU_workspace_array_projection_bytes':{key:value*n*b for key,value in gpu_workspace.items()},
    'GPU_additional_compiler_workspace_projection_bytes':sum(gpu_workspace.values())*n*b,
    'GPU_projection_not_measured_peak_bytes':(sum(gpu_primary.values())+sum(gpu_workspace.values()))*n*b,
    'host_training_archive_bytes':n*68*b+9*b,
    'postfit_full_bank_values_bytes':n*16*b,
    'postfit_QR_ordering_conservative_array_envelope_bytes':n*16*b*8,
    'bank_derivative_temporary_block_projection_bytes':4096*64*2*b*8,
    'warning':'XLA aliasing, QR workspace, native allocator and host/device lifetimes must be measured; estimates are not caps or authorization.'},
  'retained_disk_projection':{
    'one_preparation_volume_plus_training_bytes':sum(arrays.values())+n*68*b+9*b,
    'archive_plus_expanded_plus_git_three_copy_upper_projection_bytes':3*(sum(arrays.values())+n*68*b+9*b),
    'timing_full_field_arrays_seven_reps_four_cases_six_methods_bytes':7*4*6*n*3*b,
    'reference_archives_common_lift_mesh_and_coefficients':'additional measured inputs, never silently excluded',
    'chosen_policy':'immutable preparation once; no redundant moment matrices; one model at a time; preserve every actual timing invocation',
    'minimum_free_reserve_bytes':10*2**30},
  'not_a_total_peak_guarantee':'Phase lifetimes, native factor storage, JAX autodiff/QR and compression workspaces require actual allocated-node RSS measurement before admission.',
  'physical_admission':False,'execution_released':False}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--points',type=int,nargs='+',required=True)
 p.add_argument('--case-metadata',type=Path);a=p.parse_args()
 if any(n<=0 for n in a.points):raise ValueError('positive counts required')
 rows=[estimate(n) for n in a.points]
 if a.case_metadata is not None:
  if len(rows)!=1:raise ValueError('one point count per actual case metadata')
  path=a.case_metadata;record=json.loads(path.read_text());g=record['geometry'];v=g['velocity_ndof'];q=g['pressure_ndof']
  rows[0]['case_metadata']={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
   'elements':g['element_count'],'velocity_dofs':v,'pressure_dofs':q,'mixed_dofs':g['ndof'],
   'point_count_status':'caller supplied; verify against retained full_rule array before release',
   'FOM_process_peak_rss_bytes_not_ROM_peak':record['peak_rss_bytes']}
  rows[0]['native_coefficient_arrays']={
   'velocity_104_candidate_columns_bytes':104*v*8,
   'velocity_104_GridFunction_storage_additional_bytes':104*v*8,
   'pressure_41_candidate_columns_bytes':41*q*8,
   'two_scalar_48_eigenvector_copies_bytes':2*48*(v//2)*8,
   'six_sparse_metrics_and_factorization':'actual metric nnz and retained-factor lifetimes still require measurement; mixed FOM Jacobian is not a substitute'}
 print(json.dumps(rows,indent=2))
