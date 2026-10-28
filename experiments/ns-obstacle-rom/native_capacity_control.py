"""Fine candidate metric capacity measurement only; no admission, factor or training."""
from pathlib import Path
import argparse,hashlib,json,resource,time,traceback
import numpy as np
from scipy.sparse import coo_matrix,save_npz
from fe_adapter import layout_record,sparse_digest,adapter_digest,CAPS,ESSENTIAL

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

PROBE={'elements':30000,'mixed_dofs':500000,'matrix_nnz':30000000,
       'single_array_bytes':2**30,'process_rss_bytes':16*2**30,'output_bytes':3*2**30}

def run(case,release,out):
 release=json.loads(release.read_text());manifest=json.loads((case/'EXPORT-MANIFEST.json').read_text())
 if (release.get('schema')!='ns-obstacle-fine-metric-capacity-control-v1' or
     release.get('fixture_source_sha256')!=sha(__file__) or
     release.get('adapter_source_sha256')!=adapter_digest() or
     release.get('producer_manifest_sha256')!=sha(case/'EXPORT-MANIFEST.json') or
     release.get('probe_caps')!=PROBE or release.get('training_released') is not False):
  raise RuntimeError('exact reviewed capacity-only release required')
 for name in ('config.json','mesh.vol.bin','fe_layout.json','result.json'):
  if sha(case/name)!=manifest['files'][name]:raise ValueError('candidate input hash mismatch: '+name)
 cfg=json.loads((case/'config.json').read_text())
 if (cfg['velocity_order'],cfg['pressure_order'],cfg['geometry_order'],cfg['diagnostic_order'])!=(4,3,4,12):
  raise ValueError('only exact prospective P4/g4 rule permitted')
 import ngsolve as ng
 if ng.__version__!=manifest['versions']['ngsolve']:raise ValueError('native version mismatch')
 ng.SetNumThreads(cfg['threads'])
 out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();phases=[];matrices={}
 def guard(stage,extra=0):
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  if rss+extra>PROBE['process_rss_bytes']:raise MemoryError('capacity probe process envelope')
  phases.append({'stage':stage,'elapsed_seconds':time.perf_counter()-started,'peak_rss_bytes':rss,
                 'prospective_extra_bytes':int(extra)})
  (out/'phases.json').write_text(json.dumps(phases,indent=2)+'\n')
 def convert(form,size):
  nnz=int(form.mat.nze)
  if nnz>PROBE['matrix_nnz'] or nnz*8>PROBE['single_array_bytes']:raise MemoryError('native sparse payload bound')
  guard('before_COO_conversion',extra=nnz*8*9)
  rows,cols,values=form.mat.COO()
  matrix=coo_matrix((np.asarray(values),(np.asarray(rows),np.asarray(cols))),shape=(size,size)).tocsc()
  matrix.sum_duplicates();matrix.sort_indices()
  if not np.isfinite(matrix.data).all():raise ArithmeticError('nonfinite metric')
  guard('before_symmetry_check',extra=3*(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes))
  gap=matrix-matrix.T
  if gap.nnz and np.max(abs(gap.data))>1e-12*max(np.max(abs(matrix.data)),1e-300):raise ArithmeticError('nonsymmetric metric')
  return matrix,nnz
 try:
  mesh=ng.Mesh(str(case/'mesh.vol.bin'))
  if mesh.ne>PROBE['elements'] or int(mesh.GetCurveOrder())!=4:raise MemoryError('candidate geometry capacity')
  V=ng.VectorH1(mesh,order=4,dirichlet=ESSENTIAL);Q=ng.H1(mesh,order=3);S=ng.H1(mesh,order=4,dirichlet=ESSENTIAL)
  if V.ndof+Q.ndof>PROBE['mixed_dofs']:raise MemoryError('candidate DOF capacity')
  layout,digest=layout_record(mesh,V,Q,ng);saved=json.loads((case/'fe_layout.json').read_text())
  if saved['sha256']!=digest or saved['layout']!=layout:raise ValueError('actual layout differs')
  rules={kind:ng.IntegrationRule(kind,cfg['diagnostic_order']) for kind in {el.type for el in mesh.Elements(ng.VOL)}}
  n=sum(len(rules[el.type]) for el in mesh.Elements(ng.VOL));dx=ng.dx(intrules=rules)
  guard('layout_and_original_rule');metrics={}
  for label,space in [('velocity',V),('pressure',Q),('velocity_scalar',S)]:
   u,v=space.TnT();product=ng.InnerProduct(u,v);derivative=ng.InnerProduct(ng.Grad(u),ng.Grad(v))
   local_sizes=[len(space.GetDofNrs(el)) for el in mesh.Elements(ng.VOL)]
   structural_upper_entries=sum(size*size for size in local_sizes)
   if structural_upper_entries>PROBE['matrix_nnz']:raise MemoryError('pre-assembly structural bound')
   assembly_workspace=2*structural_upper_entries*16*3+16*max(local_sizes)**2*8*cfg['threads']
   guard(label+'_before_native_assembly',extra=assembly_workspace)
   mass=ng.BilinearForm(space);h1=ng.BilinearForm(space)
   mass+=product/cfg['D']**2*dx;h1+=(product/cfg['D']**2+derivative)*dx
   with ng.TaskManager():mass.Assemble();h1.Assemble()
   guard(label+'_native_assembly')
   for suffix,form in [('h1',h1),('mass',mass)]:
    key=label+'_'+suffix;matrix,native_nnz=convert(form,space.ndof);matrices[key]=matrix
    path=out/(key+'.npz');save_npz(path,matrix,compressed=True)
    metrics[key]={'shape':list(matrix.shape),'native_nnz':native_nnz,'CSC_nnz':matrix.nnz,
      'CSC_bytes':matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes,
      'matrix_sha256':sparse_digest(matrix),'file_sha256':sha(path),'file_bytes':path.stat().st_size}
    guard(key+'_saved')
    if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>PROBE['output_bytes']:raise MemoryError('output capacity')
   del mass,h1,form
  estimates={}
  for label,space in [('velocity',V),('pressure',Q),('velocity_scalar',S)]:
   parent=matrices[label+'_h1']
   parent_bytes=parent.data.nbytes+parent.indices.nbytes+parent.indptr.nbytes
   guard(label+'_before_free_submatrix',extra=3*parent_bytes+space.ndof*16)
   ids=np.flatnonzero(np.array([bool(v) for v in space.FreeDofs()]))
   matrix=matrices[label+'_h1'][ids][:,ids];estimate=int(matrix.nnz*16*100+matrix.shape[0]*8*32)
   estimates[label]={'free_dofs':len(ids),'free_nnz':matrix.nnz,'existing_factor_estimate_bytes':estimate,
    'existing_matrix_cap_would_pass':bool(matrix.nnz<=CAPS['matrix_nnz']),
    'existing_factor_cap_would_pass':bool(estimate<=CAPS['factor_estimate_bytes']),
    'existing_combined_process_estimate_would_pass':bool(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+estimate<=CAPS['process_rss_bytes'])}
   np.save(out/(label+'_free_ids.npy'),ids);del matrix;guard(label+'_free_submatrix_estimate')
  result={'passed':True,'scope':'candidate fine metric capacity measurement only','physical_admission':False,
   'training_released':False,'factorizations_performed':0,'eigenproblems_performed':0,
   'counts':{'elements':mesh.ne,'velocity_dofs':V.ndof,'pressure_dofs':Q.ndof,'scalar_velocity_dofs':S.ndof,'volume_points':n},
   'layout_sha256':digest,'metrics':metrics,'existing_factor_estimates':estimates,'probe_caps':PROBE,
   'production_caps_unchanged':CAPS,'elapsed_seconds':time.perf_counter()-started,
   'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'phases':phases,
   'versions':{'ngsolve':ng.__version__,'numpy':np.__version__}}
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 except Exception as exc:
  (out/'failure.json').write_text(json.dumps({'error':repr(exc),'traceback':traceback.format_exc(),
    'phases':phases,'physical_admission':False,'training_released':False},indent=2)+'\n');raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 run(a.case,a.release,a.out)
