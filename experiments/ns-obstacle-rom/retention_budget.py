"""Conservative retained-byte accounting; no compression/dedup savings assumed."""
from pathlib import Path
import hashlib,json,shutil
from capacity2d import estimate
from pilot2d import PREFIXES,DEV_RE,VALIDATION_RE,QUADRATURE_SUPPORTS

def budget(measurement,boundary_path):
 r=json.loads(measurement.read_text());c=r['counts'];n=c['volume_points'];v=c['velocity_dofs'];q=c['pressure_dofs']
 b=json.loads(boundary_path.read_text());base=estimate(n)
 volume=base['volume_expanded_archive_bytes'];training=base['training_expanded_archive_bytes']
 metrics=sum(m['CSC_bytes'] for m in r['metrics'].values());free=r['existing_factor_estimates']
 replay=metrics+c['scalar_velocity_dofs']*48*8+q*32*8+free['velocity']['free_dofs']*8*8+free['velocity_scalar']['free_dofs']*8+free['velocity']['free_dofs']*8+v*8
 coefficients=(v*(1+8+8+64)+q*(1+8+16))*8
 boundary=len(b)*((561+10)*8+4*max(len(row['label']) for row in b))
 fields=n*3*8;methods=len(PREFIXES)*(4+2)
 candidate_invocations=methods*len(QUADRATURE_SUPPORTS)*len(DEV_RE)
 regular_invocations=methods*(len(DEV_RE)+len(VALIDATION_RE))
 candidate_fields=candidate_invocations*fields;regular_fields=regular_invocations*fields
 fit_columns=6*sum(QUADRATURE_SUPPORTS)*(1+2+9*(2+4+64))*8
 prepared=volume+training+replay+coefficients+boundary
 accuracy=candidate_fields+regular_fields+fit_columns
 free_bytes=shutil.disk_usage(measurement.parent).free
 # Expanded files plus Git objects; archive copies and transient push workspaces
 # are deliberately NOT silently absorbed into this subtotal.
 subtotal=2*(prepared+accuracy)+10*2**30
 return {'status':'prospective lower bound on conservative release envelope; not authorization',
  'measurement_sha256':hashlib.sha256(measurement.read_bytes()).hexdigest(),
  'boundary_sha256':hashlib.sha256(boundary_path.read_bytes()).hexdigest(),
  'points':n,'boundary_points':len(b),'per_full_field_output_bytes':fields,
  'prepared_expanded_payload_bytes':{'volume':volume,'training':training,'spectral_replay':replay,'coefficients':coefficients,'boundary':boundary,'total':prepared},
  'accuracy_driver_maximum_logical_payload':{'quadrature_candidate_invocations':candidate_invocations,'regular_invocations':regular_invocations,'candidate_field_bytes':candidate_fields,'regular_field_bytes':regular_fields,'selected_fit_column_bytes':fit_columns,'subtotal':accuracy},
  'two_copy_preparation_and_accuracy_plus_reserve_bytes':subtotal,
  'local_free_bytes_at_generation':free_bytes,'shortfall_before_other_required_payload_bytes':max(0,subtotal-free_bytes),
  'timing_full_field_bytes_additional':{'FOM_plus_one_ROM_per_model':6*2*len(VALIDATION_RE)*7*fields,'FOM_plus_all_registered_methods_per_model':(6+methods)*len(VALIDATION_RE)*7*fields},
  'not_in_subtotal':['qualified family/common-lift inputs','training checkpoints and history','timing outputs and returned FOM coefficients','JSON/ZIP/file-system overhead','extra chunks for files exceeding mirror size limit','transient Git transport workspace'],
  'retention_policy':['no assumed byte-identical outputs or compression savings','one immutable preparation; no whole duplicate collection archive','exact file checksums and only lossless duplicate elimination','mirror large results in bounded incremental batches; final manifest only after complete verification','preserve ten-GiB reserve throughout'],
  'physical_admission':False,'execution_released':False}

if __name__=='__main__':
 root=Path(__file__).parent
 source=json.loads((root/'native04/SOURCE-CANDIDATE.json').read_text())
 folder=Path(source['producer_inputs'][0]['source']).parent
 print(json.dumps(budget(root/'artifacts/native-capacity04/results/interface/result.json',folder/'boundary_rule.json'),indent=2))
