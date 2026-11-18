"""Truth-free fixed-tree weak PDE pilot; dense integration, no runtime-win claim."""
import argparse
from common import *
from training_io import read_checkpoint,checkpoint
from build_local_cache import cohort,expand_matrix
import data,local_polynomial as LP,tensor_metric as TM,discrete_weak as W


def visibility(cache,M):
    A=cache['A'][:M];G=cache['gram'];C=jnp.linalg.cholesky(G+1e-12*jnp.eye(G.shape[0]))
    whiten=jax.scipy.linalg.solve_triangular(C,A.T,lower=True).T
    ev=jnp.linalg.eigvalsh(whiten.T@whiten)
    return dict(tests=M,minimum=float(ev[0]),maximum=float(ev[-1]),effective_directions=int(jnp.sum(ev>1e-8)),variables=G.shape[0])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--cache',required=True)
    ap.add_argument('--span',required=True);ap.add_argument('--dev',required=True);ap.add_argument('--admission',required=True)
    ap.add_argument('--L',type=int,default=1024);ap.add_argument('--dt',type=float,default=.0025)
    ap.add_argument('--labels',default='dev000,dev003,align002');ap.add_argument('--test-factor',type=int,default=4)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    admission=json.loads((Path(a.admission)/'result.json').read_text());assert admission['representation_gate']
    cachepath=Path(a.cache);cache_rec=json.loads((cachepath/'result.json').read_text())
    assert digest(cachepath/'local_POD.pkl')==cache_rec['local_POD_sha256'];rotation=read_checkpoint(cachepath/'local_POD.pkl')['rotation'][:,:24]
    spanpath=Path(a.span);spanrec=json.loads((spanpath/'result.json').read_text());assert digest(spanpath/'pod_basis.npz')==spanrec['basis_sha256']
    allbasis=np.load(spanpath/'pod_basis.npz')['coefficients'];assert len(allbasis)==768
    refmap={r['case']:(p,r) for p,r in cohort(a.dev)};metric=TM.make_geometry(6);rule=TM.rule()
    result=dict(meta=meta,status='fixed_tree_pilot_running',L=a.L,dt=a.dt,labels=a.labels.split(','),cases=[],
        operator='incumbent scaled tested weak residual; exact fine-grid sign-upwind; discrete sine eigen-diffusion; backward Euler',
        integration='all fine-grid interior nodes; exact dense tested sum through orthonormal DST',
        input_contract='shared initial-field callable only; no reference in initialization, tree selection or stepping',
        admission_sha256=digest(Path(a.admission)/'result.json'),final_test_opened=False,
        timing_claim='diagnostic query wall times include assembly/compilation; no paired performance inference')
    save(out/'result.json',result)
    for label in result['labels']:
        folder,row=refmap[label];p=row['parameters'];initial=lambda xy:data.initial(xy,p)
        query_start=time.perf_counter()
        x,w=data.grid(128,True);cells=LP.callable_tree(x,w,initial,52,4,'swept')
        Gfull,tr,_,_=metric(jnp.asarray(allbasis[:128]),jnp.asarray(cells),*rule)
        T=expand_matrix(jnp.asarray(rotation),52,128);Gtree=T.T@Gfull@T
        models={};M0=a.test_factor*(128+52*24);Mmax=4*M0
        for name,basis,cc,tt,U,G in [('tree_direct24',allbasis[:128],cells,tr,rotation,Gtree),
                                    ('global_POD768',allbasis,[],[],np.empty((36,0)),jnp.eye(768))]:
            ca,slices=W.prepare(basis,cc,tt,U,a.L,Mmax,G);audits=[];chosen=Mmax
            for M in (M0,2*M0,4*M0):
                v=visibility(ca,M);audits.append(v)
                if v['minimum']>=1e-6:chosen=M;break
            models[name]=dict(cache=ca,slices=slices,tests=chosen,visibility=audits,basis=basis)
        commonM=max(m['tests'] for m in models.values());setup=time.perf_counter()-query_start
        assert (a.L-1)**2>=4*commonM
        rowout=dict(case=label,parameters=p,common_tests=commonM,dense_points=(a.L-1)**2,shared_allocation_setup_seconds=setup,methods={})
        # Reference files are opened only below, after both truth-free solves finish.
        for name,model in models.items():
            ca=model['cache'];ca=dict(ca,A=ca['A'][:commonM],indices=ca['indices'][:,:commonM],lam=ca['lam'][:commonM])
            field=W.make_field(model['slices'],24 if name.startswith('tree') else 0)
            initial_fit=W.make_initial_fit(field);step,residual=W.make_solver(field,a.L)
            start=time.perf_counter();axis=jnp.arange(1,a.L)/a.L;xy=jnp.stack(jnp.meshgrid(axis,axis,indexing='ij'),axis=-1)
            u0=initial(xy);c,e0,lin=initial_fit(u0,ca);jax.block_until_ready(c)
            cold=time.perf_counter()-start;states=[np.asarray(c)];statuses=[];valid=True
            for it in range(1,round(.25/a.dt)+1):
                c,status=step(c,ca,a.dt,data.NU,1e-8,1e-10);jax.block_until_ready(c)
                statuses.append(np.asarray(status))
                if not bool(jnp.all(jnp.isfinite(c))) or float(jnp.sqrt(jnp.maximum(c@ca['gram']@c,0)))>10*float(jnp.linalg.norm(u0)/a.L):
                    valid=False;break
                if it%round(.05/a.dt)==0:
                    states.append(np.asarray(c));print('weak_step',label,name,it,'station',float(status[1]),flush=True)
                    checkpoint(out/f'{label}_{name}_partial.pkl',dict(coefficients=np.asarray(states),step_status=np.asarray(statuses),last_time=it*a.dt))
            query=time.perf_counter()-start+setup
            method=dict(active_variables=len(c),initial_fit_error=float(e0),initial_linear_residual=float(lin),
                cold_start_including_compilation_seconds=cold,query_before_output_seconds=query,visibility=model['visibility'],
                finite_trajectory=valid,step_status=np.asarray(statuses),status_columns=['scaled_tested_residual','scaled_stationarity','relative_update','relative_linear_residual','LM_iterations','accepted_steps'],
                all_steps_stationary=bool(np.max(np.asarray(statuses)[:,1])<=1e-8))
            model['states']=np.asarray(states);rowout['methods'][name]=method
            save(out/'result.json',dict(result,cases=result['cases']+[rowout]))
        # Independent physical scoring; it never feeds the solver or policy.
        refpath=folder/row['levels'][-1]['file'];assert digest(refpath)==row['levels'][-1]['sha256']
        b=jnp.asarray(np.load(refpath)['coefficients'][::2]);norm2=jnp.sum(b*b,axis=(1,2))*.25
        for name,model in models.items():
            method=rowout['methods'][name];C=jnp.asarray(model['states']);count=len(C);basis=jnp.asarray(model['basis'])
            if name.startswith('tree'):
                _,_,mx,my=metric(basis,jnp.asarray(cells),*rule);br=TM.rhs(basis,b[:count],mx,my);full=C@T.T
                err2=jnp.einsum('ti,ij,tj->t',full,Gfull,full)-2*jnp.sum(full*br,axis=1)+norm2[:count]
            else:
                N=basis.shape[-1];br=.25*jnp.einsum('tij,kij->tk',b[:count,:N,:N],basis)
                err2=jnp.sum(C*C,axis=1)-2*jnp.sum(C*br,axis=1)+norm2[:count]
            method['physical_error']=np.asarray(jnp.sqrt(jnp.maximum(err2,0)/norm2[:count]));method['full_trajectory_returned']=count==6
            if count==6:
                t0=time.perf_counter();axis=np.linspace(0,1,4096)
                if name.startswith('tree'):pred=LP.field_grid(basis,full[-1],cells,tr,6,axis)[0]
                else:pred=data.synth_grid(jnp.einsum('k,kij->ij',C[-1],basis),axis)
                jax.block_until_ready(pred);method['requested4096_reconstruction_including_compilation_seconds']=time.perf_counter()-t0
                method['diagnostic_total_query_seconds']=method['query_before_output_seconds']+method['requested4096_reconstruction_including_compilation_seconds']
                truth=data.synth_grid(b[-1],axis);method['actual4096_final_error']=float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth))
                method['output_shape']=list(pred.shape)
                np.savez_compressed(out/f'{label}_{name}_4096.npz',prediction=np.asarray(pred),reference=np.asarray(truth),case=label,time=.25)
            method['reference_allowance_max']=float(max(row['allowance'])) if 'allowance' in row else float(max(row['reference_allowance']))
            method['physical_accuracy_pass']=count==6 and bool(np.max(method['physical_error'][1:])+method['reference_allowance_max']<=.01)
            checkpoint(out/f'{label}_{name}_states.pkl',dict(coefficients=np.asarray(C),cells=cells if name.startswith('tree') else [],
                transforms=np.asarray(tr) if name.startswith('tree') else [],method=method))
            print('weak_result',label,name,method['physical_error'],flush=True)
        result['cases'].append(rowout);save(out/'result.json',result)
    result['status']='complete_fixed_tree_diagnostic_pilot';save(out/'result.json',result)

if __name__=='__main__':main()
