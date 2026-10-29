"""Bounded retained-Re20 Stokes scaling controls; no solve, fit or admission."""
from pathlib import Path
import argparse,hashlib,json,os,resource,sys,time
import numpy as np
ROOT=Path(__file__).parent;NATIVE=ROOT/'fom_native'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def source_digest():
    manifest=json.loads((NATIVE/'SOURCE.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(NATIVE/name)!=digest:raise ValueError('native closure changed')
    return hashlib.sha256(json.dumps({'entry':sha(__file__),'native_manifest':sha(NATIVE/'SOURCE.json')},sort_keys=True).encode()).hexdigest()
def safe_json(value):
    if isinstance(value,float) and not np.isfinite(value):return {'nonfinite':repr(value)}
    if isinstance(value,dict):return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,list):return [safe_json(v) for v in value]
    return value
def guard(extra=0):
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+extra>16*2**30:raise MemoryError('existing16GiB process bound')
def control(item,out,ng,rex,split_lift,export_lift):
    folder=Path(item['directory']);export=json.loads((folder/'EXPORT-MANIFEST.json').read_text())
    for name,expected in item['files'].items():
        if sha(folder/name)!=expected or (name!='EXPORT-MANIFEST.json' and export['files'].get(name)!=expected):raise ValueError('retained source member differs')
    cfg=json.loads((folder/'config.json').read_text());result=json.loads((folder/'result.json').read_text())
    if cfg!=result['configuration'] or cfg['kind']!='dfg' or cfg['outlet']!='gradient' or cfg['nu']!=.001 or cfg['precision']!='float64':raise ValueError('exact original Re20 producer configuration required')
    if json.loads((folder/'reload_checks.json').read_text()).get('passed') is not True:raise ValueError('accepted native reload missing')
    out.mkdir();started=time.perf_counter();ng.SetNumThreads(cfg['threads'])
    cfg=dict(cfg,immutable_mesh={'mesh_path':str(folder/rex.MESH_FILENAME),'mesh_sha256':sha(folder/rex.MESH_FILENAME),'identity_path':str(folder/'mesh_identity.json'),'identity_sha256':sha(folder/'mesh_identity.json')})
    with ng.TaskManager():
        guard(2*2**30);mesh,origin=rex.make_or_reuse_mesh(cfg,out,None,ng)
        V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=rex.ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
        layout,digest=rex.layout_record(mesh,V,Q,ng);saved=json.loads((folder/'fe_layout.json').read_text())
        if saved!={'layout':layout,'sha256':digest}:raise ValueError('original native layout changed')
        n=V.ndof+Q.ndof;local_sizes=[len(V.GetDofNrs(e))+len(Q.GetDofNrs(e)) for e in mesh.Elements(ng.VOL)]
        assembly_bound=sum(k*k for k in local_sizes)*3*16+2**30+n*128
        guard(assembly_bound)
        state=ng.GridFunction(X);state.Load(str(folder/'stokes.sol'));coeff=state.vec.FV().NumPy().copy();u,pnu=split_lift(coeff,V.ndof,cfg['nu'])
        (trial,pressure),(test,q)=X.TnT();free=np.array([bool(v) for v in X.FreeDofs()],dtype=bool)
        U,D=cfg['U'],cfg['D'];scale=np.r_[np.full(V.ndof,U**2*D),np.full(Q.ndof,U*D)]
        trace=ng.GridFunction(X);trace.components[0].Set(ng.CF((6*U*ng.y*(cfg['H']-ng.y)/cfg['H']**2,0)),definedon=mesh.Boundaries('inlet'))
        records=[];states=[];residuals=[];forcing=[]
        for Re in (15.,20.,25.):
            guard(assembly_bound);nu=U*D/Re;form=ng.BilinearForm(X)
            form+=(nu*ng.InnerProduct(ng.Grad(trial),ng.Grad(test))-pressure*ng.div(test)-q*ng.div(trial))*ng.dx(bonus_intorder=cfg['assembly_bonus_order']);form.Assemble()
            state.vec.FV().NumPy()[:]=np.r_[u,nu*pnu];res=state.vec.CreateVector();res.data=form.mat*state.vec
            rhs=state.vec.CreateVector();rhs.data=form.mat*trace.vec;raw=res.FV().NumPy().copy();force=rhs.FV().NumPy().copy()
            numerator=float(np.linalg.norm((raw/scale)[free]));denominator=max(float(np.linalg.norm((force/scale)[free])),1e-30)
            records.append({'Re':Re,'nu':nu,'numerator':numerator,'denominator':denominator,'relative_residual':numerator/denominator})
            states.append(state.vec.FV().NumPy().copy());residuals.append(raw);forcing.append(force)
            del form,res,rhs
        arrays={'reference_coefficients':coeff,'states':np.asarray(states),'residuals':np.asarray(residuals),'free_mask':free,
            'row_scale':scale,'forcing_vectors':np.asarray(forcing),'trace_coefficients':trace.vec.FV().NumPy().copy()}
        np.savez_compressed(out/'lift_scaling_arrays.npz',**arrays)
        passed=all(np.isfinite(a).all() for a in arrays.values()) and all(np.isfinite(row['relative_residual']) and row['relative_residual']<=cfg['linear_relative_tolerance'] for row in records)
        record={'passed':bool(passed),'physical_admission':False,'mesh_origin':origin,'layout_sha256':digest,
            'source_case_manifest_sha256':item['files']['EXPORT-MANIFEST.json'],'source_sha256':source_digest(),'records':records,
            'no_new_nonlinear_solves':True,'no_new_linear_solves':True,'no_fabricated_admission':True,
            'elapsed_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'assembly_workspace_bound_bytes':assembly_bound}
        (out/'control.json').write_text(json.dumps(safe_json(record),indent=2,allow_nan=False)+'\n')
        if not passed:raise ArithmeticError('retained scaled lift failed unchanged tolerance; raw arrays retained')
        export_lift(out,cfg,sha(out/rex.MESH_FILENAME),saved,coeff,result['metrics']['stokes_linear_residual'],result['reference_phase_seconds']['stokes_lift'])
        return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());release=json.loads(a.release.read_text())
    if (release.get('schema')!='ns-obstacle-candidate-lift-control-release-v1' or release.get('source_sha256')!=source_digest() or
        release.get('plan_sha256')!=sha(a.plan) or release.get('input_inventory_sha256')!=sha(Path('INPUT-INVENTORY.json')) or release.get('allocation_seconds')!=240 or release.get('dev_FOM_remaining_seconds')!=2160 or
        release.get('job_directory')!=str(Path.cwd()) or not os.environ.get('SLURM_JOB_ID')):raise ValueError('exact bounded native control release required')
    if sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python':raise RuntimeError('cluster only')
    if [row['id'] for row in plan['cases']]!=['p3g3h1','p3g3h2','p3g3h3','p4g4h1','p4g4h2']:raise ValueError('fixed retained five-control list required')
    with Path('CONTROL-CLAIM.json').open('x') as f:json.dump({'release_sha256':sha(a.release),'job_id':os.environ['SLURM_JOB_ID']},f)
    if any(name[:-3] in sys.modules for name in json.loads((NATIVE/'SOURCE.json').read_text())['files']):raise RuntimeError('fresh native helper namespace required')
    sys.path.insert(0,str(NATIVE));import ngsolve as ng,reference_export as rex,jax
    from shared_lift import split_common_lift,export_common_lift
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or jax.config.jax_default_matmul_precision!='highest':raise RuntimeError('GPU/f64/highest required')
    print('jax_backend=gpu',flush=True);a.out.mkdir(parents=True,exist_ok=False);records=[];start=float(os.environ['ROM_ALLOCATION_START_UNIX_SECONDS'])
    try:
        for item in plan['cases']:
            if time.time()-start>=210:raise TimeoutError('control work deadline; retain30s finalization')
            records.append({'id':item['id'],**control(item,a.out/item['id'],ng,rex,split_common_lift,export_common_lift)})
    finally:
        (a.out/'RESULT.json').write_text(json.dumps({'passed':len(records)==5 and all(r['passed'] for r in records),'records':records,
            'physical_admission':False,'eligibility_created':False,'allocation_elapsed_seconds':time.time()-start,
            'plan_sha256':sha(a.plan),'release_sha256':sha(a.release),'source_sha256':source_digest(),
            'control_allocation_debit_seconds':240,'remaining_dev_FOM_reservation_seconds':2160},indent=2)+'\n')
if __name__=='__main__':main()
