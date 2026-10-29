"""Bounded retained-native-basis primary replay. No PDE/factor/eigen/training."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,os,resource,sys,time
import numpy as np
from fe_adapter import TrainingFEAdapter,CAPS,ESSENTIAL,MESH_FILENAME,layout_record,sha_file
from point_store import write_volume,PointVolumeStore
from primary_volume import write_primary,PrimaryVolumeStore,materialize,source_digest

def run(case,basis,release_path,output):
 started=time.perf_counter();case=Path(case);basis=Path(basis);output=Path(output)
 release=json.loads(Path(release_path).read_text())
 if (release.get('schema')!='ns-obstacle-primary-volume-replay-control-v1' or
     release.get('primary_source_sha256')!=source_digest() or release.get('fixture_source_sha256')!=sha_file(__file__) or
     release.get('physical_admission') is not False or release.get('training_released') is not False):
  raise ValueError('exact-source primary replay control release required')
 if sha_file(case/'EXPORT-MANIFEST.json')!=release['producer_manifest_sha256'] or sha_file(basis/'manifest.json')!=release['basis_manifest_sha256']:
  raise ValueError('released retained inputs differ')
 producer=json.loads((case/'EXPORT-MANIFEST.json').read_text())
 for name in ('config.json',MESH_FILENAME,'fe_layout.json','full_rule.npz'):
  if sha_file(case/name)!=producer['files'][name]:raise ValueError('producer input hash differs')
 fixed=json.loads((basis/'manifest.json').read_text())
 if fixed['physical_admission'] is not False or fixed['producer_manifest_sha256']!=release['producer_manifest_sha256'] or sha_file(basis/'arrays.npz')!=fixed['array_file_sha256']:
  raise ValueError('retained basis binding differs')
 if (sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID') or
     os.environ.get('JAX_ENABLE_X64')!='true' or os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest'):
  raise RuntimeError('reviewed native cluster allocation required')
 output.mkdir(parents=True,exist_ok=False)
 report={'passed':False,'physical_admission':False,'training_released':False,'PDE_solves':0,'factorizations':0,'eigenproblems':0,
  'scope':'accepted native-batch03 basis/mesh field serialization and complete replay only; no fine-P4 extrapolation',
  'field_aliases':'seven production storage shapes; retained eigen/test columns are not claimed as Stokes/POD/supremizer constructions',
  'source_sha256':source_digest(),'fixture_sha256':sha_file(__file__),'phases':[]}
 stage='native_setup'
 def guard():
  peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  payload=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
  report['phases'].append({'stage':stage,'elapsed_seconds':time.perf_counter()-started,'peak_rss_bytes':peak,'result_bytes':payload})
  if peak>CAPS['process_rss_bytes'] or payload>2**30:raise MemoryError('bounded native replay resource cap')
 try:
  import ngsolve as ng
  cfg=json.loads((case/'config.json').read_text())
  if cfg['kind']!='dfg' or cfg['release']!='G2_steady_control_candidate':raise ValueError('retained unadmitted control required')
  ng.SetNumThreads(cfg['threads']);mesh=ng.Mesh(str(case/MESH_FILENAME))
  if mesh.dim!=2 or mesh.ne>CAPS['elements'] or int(mesh.GetCurveOrder())!=cfg['geometry_order']:raise ValueError('immutable mesh order/cap')
  V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order'])
  if V.ndof+Q.ndof>CAPS['mixed_dofs']:raise MemoryError('coarse DOF cap')
  layout,digest=layout_record(mesh,V,Q,ng)
  if json.loads((case/'fe_layout.json').read_text())!={'layout':layout,'sha256':digest}:raise ValueError('actual layout differs')
  a=SimpleNamespace(ng=ng,mesh=mesh,V=V,Q=Q,cfg=cfg,U=cfg['U'],D=cfg['D'],layout_sha256=digest)
  a.volume=TrainingFEAdapter.mapped_rule(a,False)
  with np.load(case/'full_rule.npz',allow_pickle=False) as z:
   gaps={k:float(np.max(np.abs(a.volume[k]-z[k]))) for k in ('points','weights','jacobians')}
  if max(gaps.values())>1e-12:raise ArithmeticError('retained complete producer rule differs')
  with np.load(basis/'arrays.npz',allow_pickle=False) as z:
   cv=z['velocity_candidates'];cq=z['pressure_candidates'];tv=z['velocity_transform'];tq=z['pressure_transform']
  if cv.shape!=(V.ndof,97) or cq.shape!=(Q.ndof,34) or tv.shape!=(64,64) or tq.shape!=(16,16):
   raise ValueError('retained complete native candidate widths differ')
  if any(a.dtype!=np.dtype('float64') or not np.isfinite(a).all() for a in (cv,cq,tv,tq)):
   raise ValueError('retained original float64 basis required')
  momentum=cv[:,fixed['selected_indices']['velocity']]@tv
  continuity=cq[:,fixed['selected_indices']['pressure']]@tq
  a.coefficients={'lift_u':cv[:,0].copy(),'lift_p_per_nu':cq[:,0].copy(),
   'pod_velocity':cv[:,:8].copy(),'pressure':cq[:,:8].copy(),'supremizers':cv[:,8:16].copy(),
   'momentum_tests':momentum,'continuity_tests':continuity}
  report['counts']={'elements':mesh.ne,'mixed_dofs':V.ndof+Q.ndof,'points':len(a.volume['points'])}
  report['producer_rule_max_abs_gaps']=gaps;guard()
  stage='original_volume';t=time.perf_counter();original=write_volume(a,output/'original')
  report['original_volume_seconds']=time.perf_counter()-t;guard()
  stage='primary_capture';primary=write_primary(a,output/'primary',case,output/'original')
  report['primary_capture_seconds']=primary['creation_seconds'];guard()
  stage='exact_restoration';store=PrimaryVolumeStore(output/'primary',sha_file(output/'primary/manifest.json'),ng)
  report['restoration']=materialize(store,output/'restored');report['replay_accounting']=store.replay_accounting;guard()
  stage='independent_array_comparison';left=PointVolumeStore(output/'original',sha_file(output/'original/manifest.json'))
  right=PointVolumeStore(output/'restored',sha_file(output/'restored/manifest.json'));count=0;comparisons=0
  for (start,end,x),(rs,re,y) in zip(left.iter_blocks(),right.iter_blocks(),strict=True):
   if (start,end)!=(rs,re) or set(x)!=set(y):raise ValueError('restored structure differs')
   for key in x:
    if not np.array_equal(x[key],y[key]):raise ArithmeticError('restored array bytes differ: '+key)
    comparisons+=1
   count+=end-start
  if count!=len(a.volume['points']):raise ValueError('incomplete comparison')
  report['all_array_comparisons_exact']=comparisons;report['original_manifest_sha256']=sha_file(output/'original/manifest.json')
  report['primary_manifest_sha256']=sha_file(output/'primary/manifest.json')
  report['primary_directory_bytes']=sum(p.stat().st_size for p in (output/'primary').rglob('*') if p.is_file())
  report['original_directory_bytes']=sum(p.stat().st_size for p in (output/'original').rglob('*') if p.is_file())
  guard();report['passed']=True
 except Exception as exc:report['failure']={'stage':stage,'exception':repr(exc)};raise
 finally:
  report['elapsed_seconds']=time.perf_counter()-started;report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--basis',required=True);p.add_argument('--release',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 print(json.dumps(run(a.case,a.basis,a.release,a.out),indent=2),flush=True)
