"""Bounded, validation-only frozen-network transfer and solver tuning pilot."""
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
    p.add_argument('--seed',type=int,default=7090702);p.add_argument('--meshes',default='256,512')
    p.add_argument('--reference-mesh',type=int,default=1024);p.add_argument('--reference-dt',type=float,default=.000625)
    p.add_argument('--candidate-cap',type=int,default=8192);p.add_argument('--fit-states',type=int,default=64)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=='gpu';assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    start=time.perf_counter();meshes=[int(x) for x in args.meshes.split(',')];obs=min(meshes)
    physical=e.params_draw(args.seed,args.cases)
    ck=pickle.load(open(args.checkpoint,'rb'));params=jax.tree_util.tree_map(jnp.asarray,ck['params']);Z=ck['Z_tr']
    K,R=Z.shape[1],ck['params']['h_lin'].shape[1];M=4*K;m=4*M
    report=dict(config=vars(args),commit=os.environ.get('COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),
        gpu=jax.devices()[0].device_kind,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,
        matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],jax_version=jax.__version__,
        checkpoint_sha256=sha(args.checkpoint),checkpoint_training_nodes=ck['cfg']['N'],
        checkpoint_training_intervals=ck['cfg']['N']-1,checkpoint_cfg={k:v for k,v in ck['cfg'].items() if k not in ['hfit_pick','train']},
        network_weights_frozen=True,physical_cases=physical.tolist(),cohort_role='new validation-only development cohort; final cohort unopened',
        output_times=[0,.05,.10,.15,.20,.25],observation_intervals=obs,
        timing_contract='host dense initial field and viscosity to six host dense output fields; all transfers included; compile/setup excluded',
        reference=[],mesh_setup=[],invocations=[],complete=False)
    path=out/'pilot.json'
    def save():save_json(path,report)
    save()
    references={};same_grid={};ref_fields={}
    # Three independently converged FOM settings: spatial and temporal comparisons.
    settings=[(args.reference_mesh//2,args.reference_dt),(args.reference_mesh,2*args.reference_dt),(args.reference_mesh,args.reference_dt)]
    for L,dt in settings:
        print(f'REFERENCE L={L} dt={dt}',flush=True);q,_=e.make_fom(L,dt)
        for case,phys in enumerate(physical):
            t=time.perf_counter();f,it,res=host(q(jnp.asarray(e.initial(L,phys)),float(phys[4]),1e-11,1e-9))
            worst=float(np.max(res));assert np.isfinite(f).all() and worst<=2e-11,(L,dt,case,worst)
            coarse=f[:,::L//obs,::L//obs]
            ref_fields[L,dt,case]=coarse
            np.savez_compressed(out/f'ref_L{L}_dt{dt}_case{case}.npz',fields=coarse,iterations=it,residuals=res)
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
    save()
    for L in meshes:
        print(f'BUILD ROM L={L} K={K} R={R} M={M} m={m}',flush=True)
        data,info=e.build_rom(params,Z,L,M,m,candidate_cap=args.candidate_cap,fit_states=args.fit_states)
        G=data[0]
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
        for dt in [.005,.0025]:
            q,_=e.make_fom(L,dt)
            for case,phys in enumerate(physical):
                f,it,res=host(q(jnp.asarray(e.initial(L,phys)),float(phys[4]),1e-11,1e-9))
                assert np.max(res)<2e-11
                same_grid[L,dt,case]=f[:,::L//obs,::L//obs]
            del q
        subjects=[]
        for dt in [.005,.0025]:
            for stall in [1e-3,1e-2]:
                name=f'rom_L{L}_dt{dt}_stall{stall}'
                fun=e.make_rom(params,L,dt,info['trust_radius'],stall=stall)
                subjects.append((name,dict(method='rom',solver_intervals=L,output_intervals=L,dt=dt,stall=stall),fun))
        for grid in [g for g in [128,256,512] if g<=L]:
            for dt,ntol in [(.01,.01),(.01,.003),(.005,.01),(.005,.003),(.0025,.003)]:
                name=f'fom_L{grid}_out{L}_dt{dt}_ntol{ntol}'
                fun,_=e.make_fom(grid,dt,target=L)
                subjects.append((name,dict(method='fom',solver_intervals=grid,output_intervals=L,dt=dt,newton_tolerance=ntol,linear_tolerance=.5),fun))
        input_fields=[e.initial(L,p) for p in physical]
        def invoke(subject,case):
            name,cfg,fun=subject;nu=float(physical[case,4]);u=jnp.asarray(input_fields[case])
            value=fun(u,nu,data) if cfg['method']=='rom' else fun(u,nu,cfg['newton_tolerance'],cfg['linear_tolerance'])
            return host(value)
        # Compile every shape and complete warmups before any timed block.
        compile_start=time.perf_counter()
        for sub in subjects:invoke(sub,0)
        info['compilation_warmup_seconds']=time.perf_counter()-compile_start;save()
        for rep in range(args.reps):
            e.burn(.75)
            order=subjects if rep%2==0 else subjects[::-1]
            for case in range(args.cases):
                for sub in order:
                    name,cfg,_=sub;t=time.perf_counter();value=invoke(sub,case);elapsed=time.perf_counter()-t
                    f,it,rn=value[:3]
                    finite=bool(np.isfinite(f).all() and np.isfinite(rn).all())
                    row=dict(name=name,**cfg,case=case,rep=rep,seconds=elapsed,
                        physical_error=e.errors(f,references[case],obs) if finite else None,iterations=it.tolist(),residuals=[float(x) if np.isfinite(x) else None for x in rn],
                        field_sha256=hashlib.sha256(f.tobytes()).hexdigest(),minimum=float(f.min()) if finite else None,
                        output_bytes=int(f.nbytes),finite=finite)
                    if cfg['method']=='rom':
                        row['same_grid_error']=e.errors(f,same_grid[L,cfg['dt'],case],obs) if finite else None
                        row['stop_reasons']=value[3].tolist();row['latent_states']=value[4].tolist() if np.isfinite(value[4]).all() else None;row['ic_iterations']=int(value[5]);row['ic_reason']=int(value[6])
                    else:row['nonlinear_tolerance_satisfied']=bool(finite and np.max(rn)<=cfg['newton_tolerance']*(1+1e-9))
                    # Preserve each invocation's actual common-grid fields and full-output hash.
                    artifact=f'{name}_case{case}_rep{rep}.npz'
                    np.savez_compressed(out/artifact,fields=f[:,::L//obs,::L//obs],iterations=it,residuals=rn)
                    row['observation_artifact']=artifact;report['invocations'].append(row)
            print(f'TIMED L={L} rep={rep} subjects={len(subjects)} elapsed={time.perf_counter()-start:.1f}s',flush=True);save()
        del data,G,subjects,same_grid
        same_grid={};jax.clear_caches()
    report['checkpoint_sha256_after']=sha(args.checkpoint)
    assert report['checkpoint_sha256_after']==report['checkpoint_sha256']
    report['complete']=True;report['total_wall_seconds']=time.perf_counter()-start;save()
    print(f'ALL-DONE {path} seconds={report["total_wall_seconds"]:.1f}',flush=True)


if __name__=='__main__':main()
