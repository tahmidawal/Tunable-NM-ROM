"""Coarse native full-vector/scalar-component factor equivalence; no training."""
from pathlib import Path
import argparse,json,resource,time,traceback
import numpy as np
from scipy.sparse import save_npz
from fe_adapter import TrainingFEAdapter,CAPS,ESSENTIAL,adapter_digest,sha_file,layout_record,bounded_factor,sparse_digest
from fe_pod_a6 import check_component_blocks
from component_factor import ComponentFactorSolve

class NativeInterface:
 mapped_rule=TrainingFEAdapter.mapped_rule
 iter_mapped=TrainingFEAdapter.iter_mapped
 control_indices=TrainingFEAdapter.control_indices
 scalar_component_maps=TrainingFEAdapter.scalar_component_maps
 metric_matrices=TrainingFEAdapter.metric_matrices

def run(case,release_path,out):
 release=json.loads(release_path.read_text());manifest=json.loads((case/'EXPORT-MANIFEST.json').read_text())
 if (release.get('schema')!='ns-obstacle-component-factor-control-v1' or
     release.get('fixture_source_sha256')!=sha_file(__file__) or
     release.get('adapter_source_sha256')!=adapter_digest() or
     release.get('component_factor_source_sha256')!=sha_file(Path(__file__).with_name('component_factor.py')) or
     release.get('producer_manifest_sha256')!=sha_file(case/'EXPORT-MANIFEST.json') or
     release.get('training_released') is not False):raise RuntimeError('reviewed exact coarse factor control release required')
 for name in ('config.json','mesh.vol.bin','fe_layout.json','result.json'):
  if sha_file(case/name)!=manifest['files'][name]:raise ValueError('candidate input changed')
 cfg=json.loads((case/'config.json').read_text())
 if (cfg['velocity_order'],cfg['pressure_order'],cfg['geometry_order'],cfg['diagnostic_order'])!=(4,3,4,12):
  raise ValueError('exact coarse P4/g4 control required')
 import ngsolve as ng
 if ng.__version__!=manifest['versions']['ngsolve']:raise ValueError('native version differs')
 ng.SetNumThreads(cfg['threads']);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 try:
  a=NativeInterface();a.ng=ng;a.cfg=cfg;a.D=cfg['D'];a.U=cfg['U'];a.mesh=ng.Mesh(str(case/'mesh.vol.bin'))
  if a.mesh.ne>2000 or int(a.mesh.GetCurveOrder())!=4:raise ValueError('coarse control mesh bound')
  a.V=ng.VectorH1(a.mesh,order=4,dirichlet=ESSENTIAL);a.Q=ng.H1(a.mesh,order=3);a.S=ng.H1(a.mesh,order=4,dirichlet=ESSENTIAL)
  if a.V.ndof+a.Q.ndof>40000:raise MemoryError('coarse control DOF bound')
  layout,digest=layout_record(a.mesh,a.V,a.Q,ng)
  if json.loads((case/'fe_layout.json').read_text())!={'layout':layout,'sha256':digest}:raise ValueError('layout differs')
  a.free=np.array(layout['velocity_free'],dtype=bool);a.volume=a.mapped_rule()
  maps,map_evidence=a.scalar_component_maps();sf=np.array([bool(v) for v in a.S.FreeDofs()])
  hv,mv,dx=a.metric_matrices(a.V,True);hs,ms,_=a.metric_matrices(a.S,False)
  block_evidence=check_component_blocks(hv,mv,hs,ms,maps)
  vids=np.flatnonzero(a.free);sids=np.flatnonzero(sf);vfree=hv[vids][:,vids];sfree=hs[sids][:,sids]
  factor_v,meta_v=bounded_factor(vfree)
  rng=np.random.default_rng(27102028);rhs=rng.normal(size=(len(vids),8));pressure_coefficients=rng.normal(size=(a.Q.ndof,4))
  test=a.V.TestFunction()
  for j in range(4):
   p=ng.GridFunction(a.Q);p.vec.FV().NumPy()[:]=pressure_coefficients[:,j]
   form=ng.LinearForm(a.V);form+=p*ng.div(test)/a.D*dx
   with ng.TaskManager():form.Assemble()
   rhs[:,4+j]=form.vec.FV().NumPy()[vids]
  full=factor_v.solve(rhs);del factor_v
  factor_s,meta_s=bounded_factor(sfree);solver=ComponentFactorSolve(factor_s.solve,maps,a.free,sf)
  reused=solver.solve(rhs)
  gaps={'solution_relative':float(np.linalg.norm(reused-full)/np.linalg.norm(full)),
        'full_relative_residual':float(np.linalg.norm(vfree@full-rhs)/np.linalg.norm(rhs)),
        'reused_relative_residual':float(np.linalg.norm(vfree@reused-rhs)/np.linalg.norm(rhs))}
  if max(gaps.values())>1e-10:raise ArithmeticError('unchanged full solve/component solve equivalence failed')
  matrices={}
  for name,matrix in [('velocity_h1',hv),('velocity_mass',mv),('scalar_h1',hs),('scalar_mass',ms)]:
   path=out/(name+'.npz');save_npz(path,matrix);matrices[name]={'sha256':sha_file(path),'matrix_sha256':sparse_digest(matrix),'nnz':matrix.nnz}
  np.savez_compressed(out/'solve_arrays.npz',component_maps=maps,vector_free=a.free,scalar_free=sf,rhs=rhs,
   full_solution=full,component_solution=reused,pressure_coefficients=pressure_coefficients)
  result={'passed':True,'physical_admission':False,'training_released':False,'scope':'coarse actual native matrices; factor reuse equivalence only',
   'seed':27102028,'matrix_block_evidence':block_evidence,'component_map_evidence':map_evidence,
   'scalar_reuse':solver.evidence,'full_factor':meta_v,'scalar_factor':meta_s,'gaps':gaps,'tolerance':1e-10,
   'matrices':matrices,'solve_arrays_sha256':sha_file(out/'solve_arrays.npz'),'layout_sha256':digest,
   'counts':{'elements':a.mesh.ne,'velocity_dofs':a.V.ndof,'scalar_dofs':a.S.ndof,'pressure_dofs':a.Q.ndof},
   'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'elapsed_seconds':time.perf_counter()-started,
   'production_caps_unchanged':CAPS,'versions':{'ngsolve':ng.__version__,'numpy':np.__version__}}
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 except Exception as exc:
  (out/'failure.json').write_text(json.dumps({'error':repr(exc),'traceback':traceback.format_exc(),'physical_admission':False},indent=2)+'\n');raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.case,a.release,a.out)
