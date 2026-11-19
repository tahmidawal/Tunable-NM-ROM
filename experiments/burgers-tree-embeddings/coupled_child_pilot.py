"""Truth-free joint root/child evolution with the common verified weak PDE."""
import argparse
from common import *
from training_io import read_checkpoint,checkpoint
from build_local_cache import cohort
import data,local_polynomial as LP,tensor_metric as TM,shared_child_head as H
import discrete_weak as W,weak_quadrature as QW,coupled_child as C


def visibility(A,G):
    scale=jnp.sqrt(jnp.maximum(jnp.diag(G),1e-30))
    Gn=G/(scale[:,None]*scale[None]);An=A/scale[None]
    chol=jnp.linalg.cholesky(Gn+1e-12*jnp.eye(len(G)))
    white=jax.scipy.linalg.solve_triangular(chol,An.T,lower=True).T
    ev=jnp.linalg.eigvalsh(white.T@white)
    return dict(minimum=float(ev[0]),maximum=float(ev[-1]),directions=int(jnp.sum(ev>1e-8)),variables=len(G),
                normalization='physical tangent column norms before Cholesky;1e-12 scaled Gram ridge')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    for key in ['model','cache','admission','global-control','dev']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--L',type=int,default=2048);ap.add_argument('--dt',type=float,default=.0025)
    ap.add_argument('--tests',type=int,default=11008);ap.add_argument('--labels',default='dev000,dev003,align002')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    mp=Path(a.model);mr=json.loads((mp/'result.json').read_text());assert digest(mp/'shared_child.pkl')==mr['model_sha256']
    head=read_checkpoint(mp/'shared_child.pkl');assert head['latent']==16
    ad=Path(a.admission);ar=json.loads((ad/'result.json').read_text())
    assert ar['representation_gate'] and ar['nodes']==64 and ar['model_sha256']==mr['model_sha256']
    cp=Path(a.cache);assert digest(cp/'result.json')==head['cache_sha256'];cache_record=json.loads((cp/'result.json').read_text())
    assert digest(cp/'global_basis.npz')==ar['basis_sha256'];root=np.load(cp/'global_basis.npz')['coefficients'];assert len(root)==128
    gp=Path(a.global_control);gr=json.loads((gp/'result.json').read_text())
    assert gr['status']=='complete_stable_global_controls' and digest(gp/'stable_global_basis.npz')==gr['basis_sha256']
    assert {r['case']:r['sha256'] for r in gr['training_sources']}=={r['case']:r['reference_sha256'] for r in cache_record['cases']['train']}
    assert gr['dev_labels']==head['dev_labels']
    global_basis=np.load(gp/'stable_global_basis.npz')['coefficients'][:960]
    refs={r['case']:(p,r) for p,r in cohort(a.dev)};rule=TM.rule();metric=TM.make_geometry(6)
    result=dict(meta=meta,status='coupled_child_pilot_running',L=a.L,dt=a.dt,time_integrator='cn',quadrature_intervals=a.L,
        requested_initial_tests=a.tests,model_sha256=mr['model_sha256'],admission_sha256=digest(ad/'result.json'),
        root_basis_sha256=ar['basis_sha256'],global_basis_sha256=gr['basis_sha256'],nodes=64,child_latent=16,
        operator='incumbent scaled tested weak residual; exact fine-grid sign-upwind; discrete sine eigen-diffusion; Crank-Nicolson',
        integration='dense fine-grid tested sum through orthonormal DST',input_contract='shared initial-field callable only',
        labels=a.labels.split(','),cases=[],final_test_opened=False,matched_training_reference_hashes=True,
        timing_claim='diagnostic wall time including compilation; joint comparison setup is charged to each method, so these are not method-specific performance measurements',
        stationarity_tolerance=1e-8,update_tolerance=1e-10,initial_fit_attempts=2,initial_fit_max_iterations=250)
    save(out/'result.json',result)
    for label in result['labels']:
        folder,ref=refs[label];p=ref['parameters'];initial=lambda xy:data.initial(xy,p)
        burn_gpu();setup_start=time.perf_counter();x,w=data.grid(128,True)
        cells=LP.callable_tree(x,w,initial,64,4,'swept');assert len(cells)==64
        gram,transforms,_,_=metric(jnp.asarray(root),jnp.asarray(cells),*rule)
        g,h=H.geometry(cells);models={}
        for name,basis,cc,tt,G in [('tree_child16_n64',root,cells,transforms,gram),('global_stable960',global_basis,[],[],jnp.eye(960))]:
            ca,slices=W.prepare(basis,cc,tt,jnp.eye(36),a.L,2*a.tests,G)
            field=W.make_field(slices,36 if cc is cells else 0)
            if name.startswith('tree'):
                ca.update(child_head=jax.tree.map(jnp.asarray,head['params']['head']),child_encoder=jax.tree.map(jnp.asarray,head['params']['encoder']),
                          child_geometry=g,child_width=h,child_scale=jnp.asarray(head['scale']))
                q,qs=QW.prepare(basis,cc,tt,jnp.eye(36),a.L,a.L)
                ca['quadrature']=q;ca['input_gram']=QW.make_gram(qs)(q)
                zero=jnp.zeros(1152);J=C.coefficient_jacobian(zero,ca);A=C.tested_mass_jacobian(zero,ca);tangent_gram=J.T@G@J
            else:A=ca['A'];tangent_gram=G
            audits=[];chosen=2*a.tests
            for M in (a.tests,2*a.tests):
                v=visibility(A[:M],tangent_gram);v['tests']=M;audits.append(v)
                if v['minimum']>=1e-6:chosen=M;break
            models[name]=dict(cache=ca,field=field,basis=basis,tests=chosen,visibility=audits)
        commonM=max(m['tests'] for m in models.values());assert (a.L-1)**2>=4*commonM
        row=dict(case=label,parameters=p,common_tests=commonM,shared_setup_seconds=time.perf_counter()-setup_start,methods={})
        if any(m['visibility'][-1]['minimum']<1e-6 for m in models.values()):
            result.update(status='failed_coupled_visibility',failure=dict(case=label,models={n:m['visibility'] for n,m in models.items()}))
            save(out/'result.json',result);return
        for name,model in models.items():
            ischild=name.startswith('tree');ca=model['cache'];ca=dict(ca,A=ca['A'][:commonM],indices=ca['indices'][:,:commonM],lam=ca['lam'][:commonM])
            field=model['field'];decode=C.coefficients if ischild else lambda z,ca:z
            step,_=W.make_solver(field,a.L,theta=.5,coefficient_map=C.coefficients if ischild else None,
                                 tested_mass_jacobian=C.tested_mass_jacobian if ischild else None)
            initialfit=C.make_initial_fit(field) if ischild else W.make_initial_fit(field)
            burn_gpu();start=time.perf_counter();axis=jnp.arange(1,a.L)/a.L;xy=jnp.stack(jnp.meshgrid(axis,axis,indexing='ij'),axis=-1)
            u0=initial(xy);fit=initialfit(u0,ca);z,e0,fitstation=fit[:3];jax.block_until_ready(z)
            if not bool(jnp.all(jnp.isfinite(z))) or not bool(jnp.isfinite(e0)) or not bool(jnp.isfinite(fitstation)):
                failure=checkpoint(out/f'{label}_{name}_failed_initial.pkl',dict(fit=jax.device_get(fit)))
                result.update(status='failed_coupled_initial_fit',failure=dict(case=label,method=name,checkpoint_sha256=failure))
                save(out/'result.json',result);return
            cold=time.perf_counter()-start;states=[np.asarray(decode(z,ca))];codes=[np.asarray(z)];statuses=[];valid=True
            for it in range(1,round(.25/a.dt)+1):
                z,status=step(z,ca,a.dt,data.NU,1e-8,1e-10);jax.block_until_ready(z);statuses.append(np.asarray(status));coef=decode(z,ca)
                if not bool(jnp.all(jnp.isfinite(coef))) or float(jnp.sqrt(jnp.maximum(coef@ca['gram']@coef,0)))>10*float(jnp.linalg.norm(u0)/a.L):
                    valid=False;break
                if it%round(.05/a.dt)==0:
                    states.append(np.asarray(coef));codes.append(np.asarray(z))
                    print('coupled_step',label,name,it,float(status[1]),flush=True)
                    checkpoint(out/f'{label}_{name}_partial.pkl',dict(coefficients=np.asarray(states),codes=np.asarray(codes),step_status=np.asarray(statuses),last_time=it*a.dt))
            wall=time.perf_counter()-start+row['shared_setup_seconds']
            method=dict(active_variables=len(z),initial_fit_error=float(e0),initial_fit_stationarity_or_linear_residual=float(fitstation),
                initial_fit_accepted=np.asarray(fit[3]) if ischild else None,cold_start_including_compilation_seconds=cold,
                query_before_output_seconds=wall,finite_trajectory=valid,all_steps_stationary=bool(np.max(np.asarray(statuses)[:,1])<=1e-8),
                visibility=model['visibility'],step_status=np.where(np.isfinite(statuses),np.asarray(statuses,dtype=object),None).tolist(),
                status_columns=['scaled_tested_residual','scaled_stationarity','relative_update','relative_linear_residual','LM_iterations','accepted_steps'])
            # This post-solve diagnostic is outside query timing and never alters states.
            if ischild:
                audits=[]
                for code in codes:
                    zz=jnp.asarray(code);J=C.coefficient_jacobian(zz,ca)
                    audits.append(visibility(C.tested_mass_jacobian(zz,ca),J.T@ca['gram']@J))
                method['requested_state_tangent_visibility']=audits
                method['tangent_visibility_below_safeguard']=any(v['minimum']<1e-6 for v in audits)
            model.update(states=np.asarray(states),codes=np.asarray(codes),cache=ca);row['methods'][name]=method
            save(out/'result.json',dict(result,cases=result['cases']+[row]))
        # Only after both truth-free trajectories: open reference fields and offline fits.
        refpath=folder/ref['levels'][-1]['file'];assert digest(refpath)==ref['levels'][-1]['sha256']
        b=jnp.asarray(np.load(refpath)['coefficients'][::2]);norm2=.25*jnp.sum(b*b,axis=(1,2))
        assert digest(ad/'growth_fits.pkl')==ar['fit_sha256'];offline=next(v for v in read_checkpoint(ad/'growth_fits.pkl') if v['case']==label)
        assert np.array_equal(offline['cells'],cells) and np.allclose(offline['transforms'],np.asarray(transforms),atol=1e-12,rtol=0)
        for name,model in models.items():
            m=row['methods'][name];c=jnp.asarray(model['states']);count=len(c);basis=jnp.asarray(model['basis']);G=model['cache']['gram']
            if name.startswith('tree'):
                _,_,mx,my=metric(basis,jnp.asarray(cells),*rule);rhs=TM.rhs(basis,b[:count],mx,my)
                best=jnp.asarray(offline['coefficients'][::2][:count]);projection_label='saved nonlinear reference-assisted fit; not certified minimum'
            else:
                N=basis.shape[-1];rhs=.25*jnp.einsum('tij,kij->tk',b[:count,:N,:N],basis);best=rhs;projection_label='orthogonal linear projection'
            err2=jnp.einsum('ti,ij,tj->t',c,G,c)-2*jnp.sum(c*rhs,axis=1)+norm2[:count]
            proj2=jnp.einsum('ti,ij,tj->t',best,G,best)-2*jnp.sum(best*rhs,axis=1)+norm2[:count]
            delta=c-best;distance2=jnp.einsum('ti,ij,tj->t',delta,G,delta)
            m.update(physical_error=np.asarray(jnp.sqrt(jnp.maximum(err2,0)/norm2[:count])),full_trajectory_returned=count==6,
                offline_projection_error=np.asarray(jnp.sqrt(jnp.maximum(proj2,0)/norm2[:count])),projection_label=projection_label,
                distance_from_offline_fit=np.asarray(jnp.sqrt(jnp.maximum(distance2,0)/norm2[:count])))
            if count==6:
                burn_gpu();t0=time.perf_counter();axis=np.linspace(0,1,4096)
                pred=LP.field_grid(basis,c[-1],cells,transforms,6,axis)[0] if name.startswith('tree') else data.synth_grid(jnp.einsum('k,kij->ij',c[-1],basis),axis)
                jax.block_until_ready(pred);m['requested4096_reconstruction_including_compilation_seconds']=time.perf_counter()-t0
                m['diagnostic_total_query_seconds']=m['query_before_output_seconds']+m['requested4096_reconstruction_including_compilation_seconds']
                truth=data.synth_grid(b[-1],axis);m['actual4096_final_error']=float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth));m['output_shape']=list(pred.shape)
                np.savez_compressed(out/f'{label}_{name}_4096.npz',prediction=np.asarray(pred),reference=np.asarray(truth),case=label,time=.25)
            m['reference_allowance_max']=float(max(ref.get('allowance',ref.get('reference_allowance'))))
            m['physical_accuracy_pass']=count==6 and float(np.max(m['physical_error'][1:]))+m['reference_allowance_max']<=.01
            checkpoint(out/f'{label}_{name}_states.pkl',dict(coefficients=np.asarray(c),codes=model['codes'],cells=cells if name.startswith('tree') else [],transforms=np.asarray(transforms) if name.startswith('tree') else [],method=m))
            print('coupled_result',label,name,m['physical_error'],flush=True)
        result['cases'].append(row);save(out/'result.json',result)
    result['status']='complete_coupled_child_diagnostic_pilot'
    result['pilot_gate']=all(m['physical_accuracy_pass'] and m['all_steps_stationary'] for r in result['cases'] for m in r['methods'].values())
    save(out/'result.json',result)


if __name__=='__main__':main()
