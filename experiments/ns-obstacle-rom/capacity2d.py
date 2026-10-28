"""Prospective shape accounting, not measured RSS or execution authorization."""
import argparse,json
from pathlib import Path

def estimate(n):
 rows=1+2+9*(2+4+64); b=8; block=min(n,16384)
 volume={'geometry':9,'lift_u':6,'lift_p_per_nu':3,'pod_velocity':8*6,
  'pressure':8*3,'supremizers':8*6,'momentum_tests':64*6,'continuity_tests':16*3}
 arrays={name:count*n*b for name,count in volume.items()}
 return {'points':n,'source':'projected point count; replace with actual retained rule count',
  'volume_payload_components_bytes':arrays,'volume_expanded_archive_bytes':sum(arrays.values()),
  'largest_volume_array_momentum_gradients_bytes':n*2*2*64*b,
  'training_expanded_archive_bytes':n*(5+9*(2+4+1))*b+9*b,
  'maximum_velocity_candidate_values_and_gradients_bytes':n*(96+8)*6*b,
  'maximum_pressure_candidate_values_and_gradients_bytes':n*(1+8+32)*3*b,
  'six_models_retained_enriched_velocity_fields_bytes':n*2*sum((2,4,8))*2*6*b,
  'bank_evaluation_conservative_temporary_envelope_bytes':n*64*2*b*8,
  'dense_moment_rows':rows,'dense_moment_bytes':rows*n*b,
  'blocked_matrix_persistent_disk_bytes':rows*n*b,
  'blocked_fit_resident_workspace_envelope_bytes':4*rows*block*b+4*n*b+4*rows*2560*b,
  'sparse_metric_capacity':{'matrix_nnz_cap':12000000,'CSR_bytes_per_matrix_at_cap':12000000*12+100001*4,
     'retained_metric_matrices':6,'factor_estimate_cap_bytes':8*2**30},
  'remaining_limits':{'point_cap_current':100000,'archive_expanded_cap_current_bytes':2**30,
     'single_array_cap_unchanged_bytes':2**30,'process_cap_unchanged_bytes':16*2**30},
  'required_prospective_changes':['separately reviewed measured point ceiling',
    'per-array volume archives with original hashes/units instead of one oversized archive',
    'exact blocked moment fitting; preserve every raw point/moment block'],
  'not_a_total_peak_guarantee':'Phase lifetimes, native factor storage, JAX autodiff/QR and compression workspaces require actual allocated-node RSS measurement before admission.',
  'physical_admission':False,'execution_released':False}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--points',type=int,nargs='+',required=True);a=p.parse_args()
 if any(n<=0 for n in a.points):raise ValueError('positive counts required')
 print(json.dumps([estimate(n) for n in a.points],indent=2))
