"""Independent NumPy/SciPy wave checks; no JAX or runner imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.fft import dstn,idstn
from audit_dynamics import recompute,compare_metrics,geometry,NAMES,energy2,mass


def modal_fields(u0,v0,c,n,times,dt=None):
    eigen=4*n*n*np.sin(np.arange(1,n)*np.pi/(2*n))**2
    omega=c*np.sqrt(eigen[:,None]+eigen[None,:])
    freq=omega if dt is None else 2*np.arctan(omega*dt/2)/dt
    a=dstn(u0,type=1,norm='ortho');b=dstn(v0,type=1,norm='ortho')
    for t in times:
        co,si=np.cos(freq*t),np.sin(freq*t)
        yield idstn(a*co+b*si/omega,type=1,norm='ortho'),idstn(b*co-a*omega*si,type=1,norm='ortho')


def sample_bank(inputs,mesh,n,bc):
    with np.load(inputs/'bank_parameters.npz') as f:p={k:f[k] for k in f.files}
    with np.load(inputs/'coordinates.npz') as f:inverse=np.linalg.inv(f['qr_r'])@np.linalg.inv(mesh['transform'])
    axis=np.linspace(0,1,n+1)
    if bc=='dirichlet':axis=axis[1:-1]
    ix=np.unique(np.linspace(0,len(axis)-1,min(31,len(axis)),dtype=int))
    x,y=np.meshgrid(axis[ix],axis[ix],indexing='ij');xy=np.stack((x,y),axis=-1).reshape(-1,2)
    phase=2*np.pi*(xy@p['frequency'].T)
    h=np.concatenate((2*xy-1,np.sin(phase),np.cos(phase)),axis=1)
    for layer in ('l1','l2'):
        h=h@p[f'p/{layer}/w']+p[f'p/{layer}/b'];h=h/(1+np.exp(-h))
    raw=h@p['p/out/w']+p['p/out/b']
    if bc=='dirichlet':raw*=np.prod(np.sin(np.pi*xy),axis=1)[:,None]
    indices=(ix[:,None]*len(axis)+ix[None,:]).reshape(-1)
    error=float(np.max(abs(raw@inverse-mesh['g'][indices])))
    assert error<1e-8,error
    return error


def extra_audit(record,data):
    cluster=record/'cluster';native=cluster/'out/pilot';cfg=data['config']
    times=np.arange(round(cfg['end_time']/cfg['observation_dt'])+1)*cfg['observation_dt']
    origin=json.loads((cluster/'in/ORIGIN.json').read_text())
    lineage_checks=0
    root=Path(__file__).resolve().parents[2]
    for commit,sources in ((origin['checkpoint_commit'],origin['sources']),(origin['head32_origin']['commit'],origin['head32_origin']['sources'])):
        for path,expected in sources.items():
            if '/training_ladder_' in path:
                proof=origin['head32_origin']
                result_bytes=subprocess.check_output(['git','show',commit+':'+proof['result_path']],cwd=root)
                assert hashlib.sha256(result_bytes).hexdigest()==proof['result_sha256']
                assert json.loads(result_bytes)['output_sha256'][Path(path).name]==expected
                value=(root/path).read_bytes()
                bc=Path(path).stem.removeprefix('training_ladder_')
                with np.load(root/path) as original,np.load(cluster/'in'/bc/'initializer32.npz') as supplied:
                    np.testing.assert_array_equal(original['standardized_linear'][:,:32],supplied['linear'])
                    np.testing.assert_array_equal(original['center'],supplied['center'])
            else:
                value=subprocess.check_output(['git','show',commit+':'+path],cwd=root)
            assert hashlib.sha256(value).hexdigest()==expected
            lineage_checks+=1
    all_groups=[];bank_checks=[];ref_checks=[];rom_refinements=[];parities=[]
    for bc in cfg['boundaries']:
      for n in cfg['meshes']:
        with np.load(native/f'mesh_{bc}_{n}.npz') as f:mesh={k:f[k] for k in f.files}
        bank_error=sample_bank(cluster/'in'/bc,mesh,n,bc)
        bank_checks.append(dict(boundary=bc,intervals=n,sampled_bank_max_absolute=bank_error,samples_per_axis=min(31,n-1 if bc=='dirichlet' else n+1)))
        for ci in cfg['validation_indices']:
            with np.load(native/f'reference_{bc}_{n}_{ci}.npz') as ref:
                ut,vt,u0,v0,c=ref['u'],ref['v'],ref['u0'],ref['v0'],float(ref['parameters'][5])
            rr=next(r for r in data['references'] if (r['boundary'],r['intervals'],r['case'])==(bc,n,ci))
            if bc=='absorbing':
                with np.load(native/f'reference_coarse_{bc}_{n}_{ci}.npz') as coarse:
                    independent=recompute(coarse['u'],coarse['v'],ut,vt,n,bc,c)
                delta=compare_metrics(independent,rr['temporal_refinement']);assert delta<1e-10
                maximum=max(float(np.max(independent[k]['initial_normalized'])) for k in NAMES)
                assert (maximum<=cfg['reference_refinement_target'])==rr['refinement_passed']
                ref_checks.append(dict(boundary=bc,intervals=n,case=ci,maximum_initial_normalized_refinement=maximum,metric_discrepancy=delta,passed=rr['refinement_passed']))
            else:
                reference_delta=0.
                for j,(uu,vv) in enumerate(modal_fields(u0,v0,c,n,times)):
                    reference_delta=max(reference_delta,float(np.max(abs(uu-ut[j]))),float(np.max(abs(vv-vt[j]))))
                assert reference_delta<1e-10,reference_delta
                ref_checks.append(dict(boundary=bc,intervals=n,case=ci,modal_reference_max_absolute=reference_delta,passed=True))
            for method in [cfg['primary_method']]+[f'cg_{x:g}' for x in cfg['cg_tolerances']]+(['dst'] if bc=='dirichlet' else ['rk4']):
                rows=[r for r in data['invocations'] if (r['boundary'],r['intervals'],r['case'],r['method'])==(bc,n,ci,method)]
                assert len(rows)==cfg['repetitions']
                with np.load(native/rows[0]['field_artifact']) as f:
                    u,v=f['u'],f['v'];rec=dict(boundary=bc,intervals=n,case=ci,method=method)
                    if method.startswith('cg_'):
                        for row in rows:
                            its=np.asarray(row['cg_iterations']); recursive=np.asarray(row['cg_recursive_relative']);true=np.asarray(row['cg_true_relative'])
                            assert its.shape==true.shape==recursive.shape==(round(cfg['end_time']/cfg['primary_dt']),)
                            assert np.all(its>=0)&np.all(its<=cfg['cg_max_iterations'])
                            assert row['cg_all_converged']==bool(np.all(true<=1.01*row['setting']))
                            assert np.all(recursive<=row['setting']*1.000001)|(row['cg_cap_exits']>0)
                        rec.update(cg_max_true_relative=float(max(rows[0]['cg_true_relative'])),cg_max_iterations=max(rows[0]['cg_iterations']),cg_converged=rows[0]['cg_all_converged'])
                        if bc=='dirichlet':
                            denom=np.sqrt(energy2(u0,v0,n,bc,c));modal_err=0.
                            for j,(uu,vv) in enumerate(modal_fields(u0,v0,c,n,times,dt=cfg['primary_dt'])):
                                modal_err=max(modal_err,float(np.sqrt(max(0,energy2(u[j]-uu,v[j]-vv,n,bc,c)))/denom))
                            rec['maximum_initial_energy_difference_from_exact_midpoint']=modal_err
                    if method.startswith('frozen_mlp'):
                        with np.load(cluster/'in'/bc/'head32.npz') as h:head={k:h[k] for k in h.files}
                        aa=[];bb=[];rank=[]
                        for z,w in zip(f['rollout_z'],f['rollout_w']):
                            a,b,jac,curvature=geometry(head,mesh['transform'],z,w)
                            aa.append(a);bb.append(b);sv=np.linalg.svd(jac,compute_uv=False);rank.append(float(sv[-1]/sv[0]))
                        a0,b0,j0,curve0=geometry(head,mesh['transform'],f['rollout_z'][0],f['rollout_w'][0])
                        target=mesh['g'].T@(mesh['mass']*u0.reshape(-1))
                        vtarget=mesh['g'].T@(mesh['mass']*v0.reshape(-1))
                        scale=np.sqrt(np.sum(mesh['mass']*u0.reshape(-1)**2))
                        residual=(a0-target)/scale;jscaled=j0/scale
                        gradient=float(np.max(abs(jscaled.T@residual))/max(1.,np.linalg.norm(jscaled)*np.linalg.norm(residual)))
                        q,_=np.linalg.qr(jscaled,mode='reduced')
                        stationarity=float(np.linalg.norm(q.T@residual)/max(np.linalg.norm(residual),1e-10))
                        fits=rows[0]['cold_fit'];selected=fits['selected']
                        selected_objective=float(residual@residual)
                        fit_delta=max(abs(gradient-fits['gradient'][selected]),abs(stationarity-fits['stationarity'][selected]),abs(selected_objective-fits['objective'][selected]))
                        assert fit_delta<1e-9,fit_delta
                        np.testing.assert_allclose(np.linalg.lstsq(j0,vtarget,rcond=None)[0],f['rollout_w'][0],atol=1e-9,rtol=1e-9)
                        rec.update(selected_fit_independent_gradient=gradient,selected_fit_independent_stationarity=stationarity,selected_fit_diagnostic_discrepancy=fit_delta)
                        discrepancy=max(float(np.max(abs(np.asarray(aa)-f['coefficients']))),float(np.max(abs(np.asarray(bb)-f['velocity_coefficients']))))
                        assert discrepancy<1e-8,discrepancy
                        assert np.all(f['rollout_completed'])==rows[0]['completed']
                        assert min(rank)>1e-8
                        rec.update(head_coefficient_max_absolute=discrepancy,minimum_observed_rank_ratio=min(rank),minimum_all_stages_recorded_rank_ratio=rows[0]['minimum_dynamic_rank_ratio'])
                        refined=next(r for r in data['accuracy_controls'] if (r['boundary'],r['intervals'],r['case'],r['method'])==(bc,n,ci,method))
                        with np.load(native/refined['field_artifact']) as f2:
                            dd=recompute(u,v,f2['u'],f2['v'],n,bc,c)
                        tm=next(r for r in data['time_refinement'] if (r['boundary'],r['intervals'],r['case'],r['method'])==(bc,n,ci,method))
                        assert compare_metrics(dd,tm['differences'])<1e-10
                        maximum={k:float(np.max(dd[k]['absolute']/refined['same_grid_discrepancy'][k]['initial_scale'])) for k in NAMES}
                        for k in NAMES:assert abs(maximum[k]-tm['maxima_on_reference_initial_scales'][k])<1e-10
                        rom_refinements.append(dict(boundary=bc,intervals=n,case=ci,method=method,maxima=maximum,passed=tm['passed']))
                        prior=record.parent/'device04/cluster/out/pilot'/f'{bc}_{n}_{ci}_{method}.npz'
                        if prior.exists():
                            with np.load(prior) as old:
                                parity=recompute(u,v,old['u'],old['v'],n,bc,c)
                            parities.append(dict(boundary=bc,intervals=n,case=ci,max_initial_differences={k:float(np.max(parity[k]['initial_normalized'])) for k in NAMES}))
                    energy=energy2(u,v,n,bc,c)/2
                    rec['maximum_energy_over_initial']=float(np.max(energy/max(energy[0],1e-300)))
                    if method.startswith('frozen_mlp'):
                        rec['maximum_relative_energy_balance_defect']=float(np.max(abs(energy+f['rollout_outflux']-energy[0]))/energy[0])
                    if bc=='absorbing':
                        w=np.ones(n+1)/n;w[[0,-1]]*=.5
                        boundary=(u[:,0,:]+u[:,-1,:])@w+(u[:,:,0]+u[:,:,-1])@w
                        invariant=np.sum(mass(n,bc)*v,axis=(-2,-1))+c*boundary
                        truth_boundary=(ut[:,0,:]+ut[:,-1,:])@w+(ut[:,:,0]+ut[:,:,-1])@w
                        truth_inv=np.sum(mass(n,bc)*vt,axis=(-2,-1))+c*truth_boundary
                        rec['maximum_absorbing_invariant_drift']=float(np.max(abs(invariant-invariant[0])))
                        rec['initial_absorbing_invariant_error']=float(abs(invariant[0]-truth_inv[0]))
                    all_groups.append(rec)
        del mesh
    return dict(passed=True,lineage_hashes_checked=lineage_checks,bank_checks=bank_checks,reference_checks=ref_checks,method_checks=all_groups,rom_refinement=rom_refinements,prior_frozen_trajectory_parity=parities,limitation='Independent CG large-grid checks score saved fields and modal solutions; per-internal-step true residuals are retained runner evidence, not independently reconstructed because internal full fields are not archived. Bank identity is sampled at fixed physical indices; output fields and physical errors are checked in full.')
