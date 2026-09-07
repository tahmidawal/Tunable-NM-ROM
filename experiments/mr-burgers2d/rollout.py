"""Fixed-physical cold fitting in a measured full rollout; frozen development cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import engines as e


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def host(x):return jax.tree_util.tree_map(np.asarray,x)
def save_json(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--cases',type=int,default=4);p.add_argument('--reps',type=int,default=3)
    p.add_argument('--seed',type=int,default=7090702);p.add_argument('--meshes',default='256,512,1024')
    p.add_argument('--study',choices=['cold','timestep'],default='cold');p.add_argument('--observation-intervals',type=int,default=256)
    p.add_argument('--reference-mesh',type=int,default=4096);p.add_argument('--reference-dt',type=float,default=.0003125)
    p.add_argument('--order-audit',action='store_true',default=True);p.add_argument('--ic-starts',default='1')
    p.add_argument('--candidate-cap',type=int,default=8192);p.add_argument('--fit-states',type=int,default=64)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=='gpu';assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    start=time.perf_counter();meshes=[int(x) for x in args.meshes.split(',')];obs=args.observation_intervals
    assert all(L%obs==0 for L in meshes) and args.reference_mesh//4>=max(meshes)
    physical=e.params_draw(args.seed,args.cases)
    dense_obs=max(meshes);timing_seed=89004 if args.study=='cold' else 89005;timing_rng=np.random.default_rng(timing_seed)
    arms=[('edge',60,.005,.01),('fixed_gauss',60,.005,.01),('fixed_gauss',180,.005,.01),('fixed_gauss',180,.005,.001),('fixed_gauss',180,.0025,.01)] if args.study=='cold' else [('fixed_gauss',180,dt,.01) for dt in [.005,.01,.025,.05]]
    profiles=[('edge',60,.005,.01),('fixed_gauss',180,.005,.01)] if args.study=='cold' else arms
    ck=pickle.load(open(args.checkpoint,'rb'));params=jax.tree_util.tree_map(jnp.asarray,ck['params']);Z=ck['Z_tr']
    K,R=Z.shape[1],ck['params']['h_lin'].shape[1];M=4*K;m=4*M
    report=dict(config=vars(args),commit=os.environ.get('COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),
        gpu=jax.devices()[0].device_kind,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,
        matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],jax_version=jax.__version__,
        checkpoint_sha256=sha(args.checkpoint),checkpoint_training_nodes=ck['cfg']['N'],
        checkpoint_training_intervals=ck['cfg']['N']-1,checkpoint_cfg={k:v for k,v in ck['cfg'].items() if k not in ['hfit_pick','train']},
        network_weights_frozen=True,physical_cases=physical.tolist(),cohort_role='fixed validation development cohort; final cohort unopened',
        output_times=[0,.05,.10,.15,.20,.25],observation_intervals=obs,
        timing_contract='host dense initial field and viscosity to six host dense output fields; all transfers included; compile/setup excluded',
        reference=[],mesh_setup=[],invocations=[],component_profiles=[],declared_subjects=[],timing_order=[],complete=False,
        experiment='fixed_gauss_rollout' if args.study=='cold' else 'gauss_timestep_study',dense_observation_intervals=dense_obs,timing_order_seed=timing_seed,
        arm_grid=[dict(cold_rule=rule,ic_budget=icb,ic_starts=1,dt=dt,stall=stall) for rule,icb,dt,stall in
            arms],
        component_profile_configs=[dict(cold_rule=rule,ic_budget=icb,ic_starts=1,dt=dt,stall=stall) for rule,icb,dt,stall in profiles],
        arm_rationale='cold03 supports one-start and budget180 quality control; retain budget60 matched edge/Gauss and original evolution settings' if args.study=='cold' else 'rollout04 validates fixed-Gauss fitting; test temporal over-resolution with unchanged initialization, weak modes and FOM envelope')
    path=out/'pilot.json'
    def save():save_json(path,report)
    save()
    references={};same_grid={};ref_fields={};ref_dense={};same_dense={}
    # Three independently converged FOM settings: spatial and temporal comparisons.
    settings=[(args.reference_mesh//2,args.reference_dt),(args.reference_mesh,2*args.reference_dt),(args.reference_mesh,args.reference_dt)]
    if args.order_audit:
        settings += [(args.reference_mesh//4,2*args.reference_dt),(args.reference_mesh//2,2*args.reference_dt),(args.reference_mesh,4*args.reference_dt)]
    for L,dt in settings:
        print(f'REFERENCE L={L} dt={dt}',flush=True);q,_=e.make_fom(L,dt)
        for case,phys in enumerate(physical):
            t=time.perf_counter();f,it,res=host(q(jnp.asarray(e.initial(L,phys)),float(phys[4]),1e-11,1e-9))
            worst=float(np.max(res));assert np.isfinite(f).all() and worst<=2e-11,(L,dt,case,worst)
            coarse=np.array(f[:,::L//obs,::L//obs],copy=True)
            ref_fields[L,dt,case]=coarse
            ref_dense[L,dt,case]=np.array(f[:,::L//dense_obs,::L//dense_obs],copy=True)
            np.savez_compressed(out/f'ref_L{L}_dt{dt}_case{case}.npz',fields=coarse,dense_fields=ref_dense[L,dt,case],iterations=it,residuals=res)
            report['reference'].append(dict(intervals=L,dt=dt,case=case,wall_seconds=time.perf_counter()-t,
                max_relative_residual=worst,iterations=it.tolist(),residuals=res.tolist(),minimum=float(f.min())))
            save()
        del q;jax.clear_caches()
    for case in range(args.cases):references[case]=ref_fields[args.reference_mesh,args.reference_dt,case]
    report['reference_uncertainty']=[]
    for case in range(args.cases):
        fine=references[case]
        space=e.errors(ref_fields[args.reference_mesh//2,args.reference_dt,case],fine,obs)
        temporal=e.errors(ref_fields[args.reference_mesh,2*args.reference_dt,case],fine,obs)
        report['reference_uncertainty'].append(dict(case=case,space_difference=space,time_difference=temporal,
            conservative_difference_sum=space['fixed_initial_max']+temporal['fixed_initial_max'],
            interpretation='refinement estimate, not a rigorous continuum error bound; tighter targets remain provisional'))
    if args.order_audit:
        report['reference_order_audit']=[]
        for case in range(args.cases):
            Rf=args.reference_mesh;dtr=args.reference_dt
            # Three same-dt spatial levels, and three same-mesh temporal levels.
            spatial_coarse=e.errors(ref_fields[Rf//4,2*dtr,case],ref_fields[Rf//2,2*dtr,case],obs)['fixed_initial_max']
            spatial_fine=e.errors(ref_fields[Rf//2,2*dtr,case],ref_fields[Rf,2*dtr,case],obs)['fixed_initial_max']
            temporal_coarse=e.errors(ref_fields[Rf,4*dtr,case],ref_fields[Rf,2*dtr,case],obs)['fixed_initial_max']
            temporal_fine=e.errors(ref_fields[Rf,2*dtr,case],ref_fields[Rf,dtr,case],obs)['fixed_initial_max']
            ps=float(np.log2(spatial_coarse/spatial_fine));pt=float(np.log2(temporal_coarse/temporal_fine))
            report['reference_order_audit'].append(dict(case=case,spatial_coarse_difference=spatial_coarse,
                spatial_fine_difference=spatial_fine,observed_spatial_order=ps,temporal_coarse_difference=temporal_coarse,
                temporal_fine_difference=temporal_fine,observed_temporal_order=pt,
                asymptotic_decrease_observed=bool(spatial_fine<spatial_coarse and temporal_fine<temporal_coarse),
                empirical_richardson_estimate=(spatial_fine/(2**ps-1)+temporal_fine/(2**pt-1)) if ps>0 and pt>0 else None))
    # Each requested-grid norm receives its own empirical reference margin.
    dense_references={case:ref_dense[args.reference_mesh,args.reference_dt,case] for case in range(args.cases)}
    report['reference_metrics_by_output']={}
    for target in sorted(set(meshes+[obs])):
        fields={key:f[:,::dense_obs//target,::dense_obs//target] for key,f in ref_dense.items()}
        uncertainty=[];orders=[]
        for case in range(args.cases):
            rf=args.reference_mesh;dtr=args.reference_dt;fine=fields[rf,dtr,case]
            space=e.errors(fields[rf//2,dtr,case],fine,target);temporal=e.errors(fields[rf,2*dtr,case],fine,target)
            uncertainty.append(dict(case=case,space_difference=space,time_difference=temporal,
                conservative_difference_sum=space['fixed_initial_max']+temporal['fixed_initial_max']))
            sc=e.errors(fields[rf//4,2*dtr,case],fields[rf//2,2*dtr,case],target)['fixed_initial_max']
            sf=e.errors(fields[rf//2,2*dtr,case],fields[rf,2*dtr,case],target)['fixed_initial_max']
            tc=e.errors(fields[rf,4*dtr,case],fields[rf,2*dtr,case],target)['fixed_initial_max']
            tf=temporal['fixed_initial_max'];ps=float(np.log2(sc/sf));pt=float(np.log2(tc/tf))
            orders.append(dict(case=case,observed_spatial_order=ps,observed_temporal_order=pt,
                spatial_coarse_difference=sc,spatial_fine_difference=sf,temporal_coarse_difference=tc,temporal_fine_difference=tf,
                asymptotic_decrease_observed=bool(sf<sc and tf<tc),
                empirical_richardson_estimate=(sf/(2**ps-1)+tf/(2**pt-1)) if ps>0 and pt>0 else None))
        report['reference_metrics_by_output'][str(target)]=dict(uncertainty=uncertainty,order_audit=orders)
    save()
    for L in meshes:
        print(f'BUILD ROM L={L} K={K} R={R} M={M} m={m}',flush=True)
        data,info=e.build_rom(params,Z,L,M,m,candidate_cap=args.candidate_cap,fit_states=args.fit_states)
        G=data[0]
        cold,cold_info=e.build_gauss_cold(params,data[7]);info['gauss_cold_setup']=cold_info
        # Numerical bank rank on deterministic rows, not the declared output dimension.
        sample=np.asarray(G[::max(1,len(G)//4096)])
        sv=np.linalg.svd(sample,compute_uv=False)
        info['sampled_bank_singular_values']=sv.tolist();info['sampled_bank_rank_relative_1e12']=int(np.sum(sv>sv[0]*1e-12))
        # Holdout from training codes, excluded from the quadrature fit, used only for operator diagnostics.
        held=np.setdiff1d(np.arange(min(len(Z),512)),info['eq_fit_rows'])[:12]
        Phi,_,_=e.modes(L,M);P=jnp.asarray(Phi)
        full_adv=jax.jit(lambda G,P,z:P.T@e.spatial(G@e.sc.head(params,z),L)[0])
        def sample_adv(z):
            h=e.sc.head(params,z);us=jnp.einsum('msr,r->ms',data[3],h);c,xp,xm,yp,ym=[us[:,j] for j in range(5)]
            return data[4].T@(c*L*(jnp.where(c>0,c-xm,xp-c)+jnp.where(c>0,c-ym,yp-c)))
        sample_adv=jax.jit(sample_adv)
        op=[]
        for zi in held:
            z=jnp.asarray(Z[zi]);exact=np.asarray(full_adv(G,P,z));approx=np.asarray(sample_adv(z));u=np.asarray(G@e.sc.head(params,z))
            op.append(dict(training_code=int(zi),relative_advection_error=float(np.linalg.norm(approx-exact)/(np.linalg.norm(exact)+1e-300)),
                           decoded_minimum=float(u.min()),negative_fraction=float(np.mean(u<0))))
        info['held_training_operator_audit']=op;report['mesh_setup'].append(info);save()
        del full_adv,sample_adv,P,Phi,sample
        # Tight same-grid FOM at each ROM timestep; same physical cases, no fitting on these fields.
        for dt in sorted({r['dt'] for r in report['arm_grid']}):
            q,_=e.make_fom(L,dt)
            for case,phys in enumerate(physical):
                f,it,res=host(q(jnp.asarray(e.initial(L,phys)),float(phys[4]),1e-11,1e-9))
                assert np.max(res)<2e-11
                same_grid[L,dt,case]=f[:,::L//obs,::L//obs]
                same_dense[L,dt,case]=f
                np.savez_compressed(out/f'same_L{L}_dt{dt}_case{case}.npz',fields=f,iterations=it,residuals=res)
            del q
        subjects=[]
        for arm in report['arm_grid']:
            rule,icb,dt,stall=arm['cold_rule'],arm['ic_budget'],arm['dt'],arm['stall']
            name=f'rom_L{L}_{rule}_ic{icb}_dt{dt}_stall{stall}_starts1'
            fun=e.make_rom(params,L,dt,info['trust_radius'],stall=stall) if rule=='edge' else e.make_gauss_rom(params,L,dt,info['trust_radius'],stall=stall,ic_budget=icb)
            subjects.append((name,dict(method='rom',solver_intervals=L,output_intervals=L,**arm),fun))
        for grid in [g for g in [128,256,512,1024] if g<=L]:
            for dt,ntol in [(.01,.01),(.01,.003),(.005,.01),(.005,.003),(.0025,.003)]:
                name=f'fom_L{grid}_out{L}_dt{dt}_ntol{ntol}'
                fun,_=e.make_fom(grid,dt,target=L)
                subjects.append((name,dict(method='fom',solver_intervals=grid,output_intervals=L,dt=dt,newton_tolerance=ntol,linear_tolerance=.5),fun))
        report['declared_subjects'] += [dict(name=name,**cfg) for name,cfg,_ in subjects]
        input_fields=[e.initial(L,p) for p in physical]
        dense_truth={case:dense_references[case][:,::dense_obs//L,::dense_obs//L] for case in range(args.cases)}
        first_dense={}
        def invoke(subject,case):
            name,cfg,fun=subject;nu=float(physical[case,4]);u=jnp.asarray(input_fields[case])
            if cfg['method']=='rom':value=fun(u,nu,data) if cfg['cold_rule']=='edge' else fun(u,nu,data,cold)
            else:value=fun(u,nu,cfg['newton_tolerance'],cfg['linear_tolerance'])
            return host(value)
        # Compile every shape and complete warmups before any timed block.
        compile_start=time.perf_counter()
        for sub in subjects:invoke(sub,0)
        info['compilation_warmup_seconds']=time.perf_counter()-compile_start;save()
        for rep in range(args.reps):
            for case in range(args.cases):
                order=[subjects[i] for i in timing_rng.permutation(len(subjects))]
                report['timing_order'].append(dict(intervals=L,case=case,rep=rep,names=[x[0] for x in order]))
                for sub in order:
                    e.burn(.25)
                    name,cfg,_=sub;t=time.perf_counter();value=invoke(sub,case);elapsed=time.perf_counter()-t
                    f,it,rn=value[:3]
                    finite=bool(np.isfinite(f).all() and np.isfinite(rn).all())
                    row=dict(name=name,**cfg,case=case,rep=rep,seconds=elapsed,
                        physical_error=e.errors(f,references[case],obs) if finite else None,
                        dense_physical_error=e.errors(f,dense_truth[case],L) if finite else None,iterations=it.tolist(),residuals=[float(x) if np.isfinite(x) else None for x in rn],
                        field_sha256=hashlib.sha256(f.tobytes()).hexdigest(),minimum=float(f.min()) if finite else None,
                        output_bytes=int(f.nbytes),finite=finite)
                    if cfg['method']=='rom':
                        row['same_grid_error']=e.errors(f,same_grid[L,cfg['dt'],case],obs) if finite else None
                        row['dense_same_grid_error']=e.errors(f,same_dense[L,cfg['dt'],case],L) if finite else None
                        row['stop_reasons']=value[3].tolist();row['latent_states']=value[4].tolist() if np.isfinite(value[4]).all() else None;row['ic_iterations']=int(value[5]);row['ic_reason']=int(value[6])
                    else:row['nonlinear_tolerance_satisfied']=bool(finite and np.max(rn)<=cfg['newton_tolerance']*(1+1e-9))
                    # Preserve each invocation's actual common-grid fields and full-output hash.
                    artifact=f'{name}_case{case}_rep{rep}.npz'
                    np.savez_compressed(out/artifact,fields=f[:,::L//obs,::L//obs],iterations=it,residuals=rn)
                    row['observation_artifact']=artifact
                    # Archive one actual dense output per configuration/case; exact-hash
                    # matches permit repeat reuse. Nonmatching repeats retain their own fields.
                    key=(name,case)
                    if key not in first_dense or first_dense[key][0]!=row['field_sha256']:
                        dense_artifact=f'{name}_case{case}_rep{rep}.dense.npz'
                        np.savez_compressed(out/dense_artifact,fields=f)
                        row['dense_artifact']=dense_artifact
                        if key not in first_dense:first_dense[key]=(row['field_sha256'],dense_artifact)
                    else:row['dense_artifact']=first_dense[key][1]
                    row['matches_first_dense_sha256']=row['field_sha256']==first_dense[key][0]
                    report['invocations'].append(row)
            print(f'TIMED L={L} rep={rep} subjects={len(subjects)} elapsed={time.perf_counter()-start:.1f}s',flush=True);save()
        for profile in report['component_profile_configs']:
            rule,icb,dt,stall=profile['cold_rule'],profile['ic_budget'],profile['dt'],profile['stall']
            maker=e.make_rom if rule=='edge' else e.make_gauss_rom
            extra={} if rule=='edge' else dict(ic_budget=icb)
            fused,parts=maker(params,L,dt,info['trust_radius'],stall=stall,return_parts=True,**extra)
            def init(u):return parts['initialize'](u,data) if rule=='edge' else parts['initialize'](u,data,cold)
            def query(u,nu):return fused(u,nu,data) if rule=='edge' else fused(u,nu,data,cold)
            warm=jnp.asarray(input_fields[0]);zi,ii,ir,scale=init(warm)
            Zp,itp,rnp,rsp=parts['evolve'](zi,float(physical[0,4]),scale,data)
            jax.block_until_ready(parts['decode'](Zp,data));jax.block_until_ready(query(warm,float(physical[0,4])))
            for case in range(args.cases):
                e.burn(.4);t=time.perf_counter();uj=jnp.asarray(input_fields[case]);jax.block_until_ready(uj);t1=time.perf_counter()
                zi,ii,ir,scale=jax.block_until_ready(init(uj));t2=time.perf_counter()
                Zp,itp,rnp,rsp=jax.block_until_ready(parts['evolve'](zi,float(physical[case,4]),scale,data));t3=time.perf_counter()
                Fp=jax.block_until_ready(parts['decode'](Zp,data));t4=time.perf_counter()
                fp=np.asarray(Fp);t5=time.perf_counter();fused_value=host(query(uj,float(physical[case,4])))
                parity=float(np.linalg.norm(fp-fused_value[0])/(np.linalg.norm(fused_value[0])+1e-300));assert parity<1e-8
                report['component_profiles'].append(dict(intervals=L,case=case,**profile,
                    input_transfer_s=t1-t,cold_fit_s=t2-t1,evolution_s=t3-t2,dense_decode_s=t4-t3,output_transfer_s=t5-t4,
                    staged_total_s=t5-t,fused_output_relative_difference=parity,physical_error=e.errors(fp,references[case],obs),
                    time_steps=int(len(itp)),lm_attempts=int(np.asarray(itp).sum()),ic_selected_attempts=int(ii),ic_selected_reason=int(ir),
                    interpretation='separately synchronized components of one staged invocation; diagnostic timings, not substituted into fused query results'))
            save()
        del data,G,subjects,same_grid,same_dense,cold
        same_grid={};same_dense={};jax.clear_caches()
    report['checkpoint_sha256_after']=sha(args.checkpoint)
    assert report['checkpoint_sha256_after']==report['checkpoint_sha256']
    report['complete']=True;report['total_wall_seconds']=time.perf_counter()-start;save()
    print(f'ALL-DONE {path} seconds={report["total_wall_seconds"]:.1f}',flush=True)


if __name__=='__main__':main()
