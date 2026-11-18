"""Truth-free fixed-tree weak PDE pilot; dense integration, no runtime-win claim."""
import argparse
from common import *
from training_io import read_checkpoint,checkpoint
from build_local_cache import cohort,expand_matrix
import data,local_polynomial as LP,tensor_metric as TM,discrete_weak as W,weak_quadrature as QW


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
    ap.add_argument('--compare-previous')
    ap.add_argument('--integrator',choices=['be','cn'],default='be')
    ap.add_argument('--allow-integrator-comparison',action='store_true')
    ap.add_argument('--quadrature',type=int,default=0,help='0 means exact dense integration; otherwise structured fine-stencil integration')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    admission=json.loads((Path(a.admission)/'result.json').read_text());assert admission['representation_gate']
    cachepath=Path(a.cache);cache_rec=json.loads((cachepath/'result.json').read_text())
    assert digest(cachepath/'local_POD.pkl')==cache_rec['local_POD_sha256'];rotation=read_checkpoint(cachepath/'local_POD.pkl')['rotation'][:,:24]
    spanpath=Path(a.span);spanrec=json.loads((spanpath/'result.json').read_text());assert digest(spanpath/'pod_basis.npz')==spanrec['basis_sha256']
    allbasis=np.load(spanpath/'pod_basis.npz')['coefficients'];assert len(allbasis)==768
    refmap={r['case']:(p,r) for p,r in cohort(a.dev)};metric=TM.make_geometry(6);rule=TM.rule()
    result=dict(meta=meta,status='fixed_tree_pilot_running',L=a.L,dt=a.dt,labels=a.labels.split(','),cases=[],
        operator='incumbent scaled tested weak residual; exact fine-grid sign-upwind; discrete sine eigen-diffusion; '+('backward Euler' if a.integrator=='be' else 'Crank-Nicolson'),
        time_integrator=a.integrator,
        integration=('all fine-grid interior nodes; exact dense tested sum through orthonormal DST' if not a.quadrature else
                     'uniform composite weak integration; sign-upwind stencil retains fine spacing 1/L'),
        quadrature_intervals=a.quadrature or a.L,
        input_contract='shared initial-field callable only; no reference in initialization, tree selection or stepping',
        admission_sha256=digest(Path(a.admission)/'result.json'),global_basis_sha256=spanrec['basis_sha256'],
        local_rotation_sha256=cache_rec['local_POD_sha256'],final_test_opened=False,
        timing_claim='diagnostic query wall times include assembly/compilation; no paired performance inference',timing_burn_in=True)
    if a.compare_previous:
        previous_path=Path(a.compare_previous);assert str(previous_path).startswith(os.environ['STUDY_NAMESPACE']+'/')
        previous_result=json.loads((previous_path/'result.json').read_text())
        assert previous_result['status']=='complete_fixed_tree_diagnostic_pilot'
        if previous_result['operator']!=result['operator']:
            assert a.allow_integrator_comparison and previous_result['operator'].rsplit(';',1)[0]==result['operator'].rsplit(';',1)[0]
            result['comparison_kind']='different_integrator_diagnostic_not_a_step_convergence_check'
        else:result['comparison_kind']='same_integrator_numerical_refinement'
        result['previous_result_sha256']=digest(previous_path/'result.json')
    save(out/'result.json',result)
    for label in result['labels']:
        folder,row=refmap[label];p=row['parameters'];initial=lambda xy:data.initial(xy,p)
        burn_gpu();query_start=time.perf_counter()
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
        if a.quadrature:
            assert a.L%a.quadrature==0 and (a.quadrature-1)**2>=4*commonM
            for name,model in models.items():
                qc,qs=QW.prepare(model['basis'],cells if name.startswith('tree') else [],tr if name.startswith('tree') else [],rotation,a.L,a.quadrature)
                model['cache']=dict(model['cache'],quadrature=qc);model['quadrature_slices']=qs
                assert int(jnp.max(model['cache']['indices'][:,:commonM]))<a.quadrature-1
            setup=time.perf_counter()-query_start
        rowout=dict(case=label,parameters=p,common_tests=commonM,fine_interior_nodes=(a.L-1)**2,
            integration_points=((a.quadrature or a.L)-1)**2,dense_points=(a.L-1)**2 if not a.quadrature else None,
            shared_allocation_setup_seconds=setup,methods={})
        # Reference files are opened only below, after both truth-free solves finish.
        for name,model in models.items():
            ca=model['cache'];ca=dict(ca,A=ca['A'][:commonM],indices=ca['indices'][:,:commonM],lam=ca['lam'][:commonM])
            field=W.make_field(model['slices'],24 if name.startswith('tree') else 0)
            advection=None;fitfield=field;fitN=a.L
            if a.quadrature:
                qfields=QW.make_fields(model['quadrature_slices'],24 if name.startswith('tree') else 0)
                advection=QW.make_tested(qfields,a.L,a.quadrature);fitfield=lambda c,ca:qfields(c,ca)[0];fitN=a.quadrature
            initial_fit=W.make_initial_fit(fitfield);step,residual=W.make_solver(field,a.L,theta=1. if a.integrator=='be' else .5,advection_tests=advection)
            burn_gpu();start=time.perf_counter();axis=jnp.arange(1,fitN)/fitN;xy=jnp.stack(jnp.meshgrid(axis,axis,indexing='ij'),axis=-1)
            u0=initial(xy);c,e0,lin=initial_fit(u0,ca);jax.block_until_ready(c)
            cold=time.perf_counter()-start;states=[np.asarray(c)];statuses=[];valid=True
            for it in range(1,round(.25/a.dt)+1):
                c,status=step(c,ca,a.dt,data.NU,1e-8,1e-10);jax.block_until_ready(c)
                statuses.append(np.asarray(status))
                if not bool(jnp.all(jnp.isfinite(c))) or float(jnp.sqrt(jnp.maximum(c@ca['gram']@c,0)))>10*float(jnp.linalg.norm(u0)/fitN):
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
                Gram=Gtree;rhs=br@T
                optimum=jnp.linalg.solve(Gram+1e-10*jnp.eye(Gram.shape[0]),rhs.T).T
            else:
                N=basis.shape[-1];br=.25*jnp.einsum('tij,kij->tk',b[:count,:N,:N],basis)
                err2=jnp.sum(C*C,axis=1)-2*jnp.sum(C*br,axis=1)+norm2[:count]
                Gram=jnp.eye(len(basis));rhs=br;optimum=br
            method['physical_error']=np.asarray(jnp.sqrt(jnp.maximum(err2,0)/norm2[:count]));method['full_trajectory_returned']=count==6
            proj2=jnp.einsum('ti,ij,tj->t',optimum,Gram,optimum)-2*jnp.sum(optimum*rhs,axis=1)+norm2[:count]
            delta=C-optimum;dyn2=jnp.einsum('ti,ij,tj->t',delta,Gram,delta)
            method['projection_error']=np.asarray(jnp.sqrt(jnp.maximum(proj2,0)/norm2[:count]))
            method['distance_from_best_projection']=np.asarray(jnp.sqrt(jnp.maximum(dyn2,0)/norm2[:count]))
            method['squared_error_decomposition_defect']=float(jnp.max(jnp.abs(err2-proj2-dyn2)/norm2[:count]))
            if a.compare_previous:
                oldrow=next(v for v in previous_result['cases'] if v['case']==label);assert np.array_equal(oldrow['parameters'],p)
                oldfile=previous_path/f'{label}_{name}_states.pkl';oldstate=read_checkpoint(oldfile)
                oldC=jnp.asarray(oldstate['coefficients']);assert oldC.shape==C.shape
                if name.startswith('tree'):assert np.array_equal(oldstate['cells'],cells) and np.allclose(oldstate['transforms'],np.asarray(tr),rtol=0,atol=1e-12)
                change=C-oldC
                method['previous_state_relative_change']=np.asarray(jnp.sqrt(jnp.maximum(jnp.einsum('ti,ij,tj->t',change,Gram,change),0)/norm2[:count]))
                method['previous_state_file_sha256']=digest(oldfile)
            if count==6:
                burn_gpu();t0=time.perf_counter();axis=np.linspace(0,1,4096)
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
