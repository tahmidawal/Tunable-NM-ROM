"""Fixed-family development comparison: genuine NS3D ROMs, operators and FOMs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_fom as F
import ns3d_verify as V
import ns3d_model as D
import ns3d_rom as R
import dataset_adapter as A
import operator_adapter as O
from operators import training as OT
from pilot import generate,write,summary,sha


def project_coefficients(G,X):
    Q,Rb,coeff,perp,info=D.whiten(G,X)
    return Q,Rb,coeff,perp,info


def projection_floor(X,P):
    # GPU contractions, directly measured error (no subtraction of squared norms).
    xx=jnp.asarray(X).reshape(len(X),-1);basis=jnp.asarray(P)
    error=jnp.linalg.norm(xx-(xx@basis)@basis.T,axis=1)/jnp.linalg.norm(xx,axis=1)
    return np.asarray(error)


def raw_projection_check(params,coeff,U,n):
    xy=D.coords(n);angle=2*jnp.pi*(xy@params['B'])
    features=jnp.concatenate((jnp.sin(angle),jnp.cos(angle)),axis=1)
    raw=params['out_scale']*D.S.apply_mlp(params['g'],features)
    rank=raw.shape[-1]//3
    raw=raw.reshape(n,n,n,3,rank).transpose(3,0,1,2,4).reshape(3*n**3,rank)
    G=D.bank(params,xy,F.geometry(n),n)
    c=jnp.asarray(coeff);truth=jnp.asarray(U).reshape(len(U),-1)
    raw_error=np.asarray(jnp.linalg.norm(c@raw.T-truth,axis=1))
    projected=np.asarray(jnp.linalg.norm(c@G.T-truth,axis=1))
    return dict(passed=bool(np.all(projected<=raw_error+1e-10)),raw_error=raw_error,
                projected_error=projected,max_excess=float(np.max(projected-raw_error)))


def burn(seconds=1.):
    a=jnp.ones((1024,1024),dtype=jnp.float64)*.001
    fn=jax.jit(lambda x: x@x+.00001)
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        a=fn(a);a.block_until_ready()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    result=out/'result.json'
    report=dict(complete=False,stage='preflight',kind='ns3d_fixed_family_development_comparison',config=cfg,
                source_commit=os.environ.get('SOURCE_COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),
                backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),jax_version=jax.__version__,
                precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),final_cohort_opened=False,
                limitations=['one training seed and opened development cases','dense cold fit and full field readout charged',
                             'dense diagnostic does not establish grid-independent initialization',
                             'budget termination does not establish training convergence'])
    def stage(name):
        report['stage']=name;write(report,result);print('STAGE',name,flush=True)
    try:
        report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
        assert report['backend']=='gpu' and report['x64'] and report['precision']=='highest'
        assert cfg['initial_amplitude']==F.INITIAL_AMPLITUDE
        print('jax_backend=gpu',flush=True);stage('reference_verification')
        report['operator_verification']=V.operator_checks()
        report['reference_verification']=V.reference_checks(cfg['n'],cfg['dt'],cfg['horizon'],cfg['verification_cases'],out/'reference_N32')
        write(report,result)
        assert report['operator_verification']['passed'] and report['reference_verification']['passed']
        n=cfg['n'];k=cfg['k'];geom=F.geometry(n)
        stage('data')
        Utr,report['train_data']=generate(n,cfg['dt'],cfg['horizon'],cfg['train_cases'],cfg['train_seed'],out/'train_data.npz')
        Udev,report['dev_data']=generate(n,cfg['dt'],cfg['horizon'],cfg['dev_cases'],cfg['dev_seed'],out/'dev_data.npz')
        train=Utr.reshape(-1,3,n,n,n);dev=Udev.reshape(-1,3,n,n,n)
        stage('training_POD_capacity_screen')
        start=time.monotonic();P,scores,singular=D.pod_gpu(train,max(cfg['rank_candidates']))
        report['POD_setup_seconds']=time.monotonic()-start
        report['training_POD_floors']={str(r):summary(projection_floor(train,P[:,:r])) for r in [64,128]+cfg['rank_candidates']}
        report['development_POD_floors']={str(r):summary(projection_floor(dev,P[:,:r])) for r in [16,32,48,64,128]+cfg['rank_candidates']}
        rank=cfg['rank_candidates'][-1]
        for r in cfg['rank_candidates']:
            if report['training_POD_floors'][str(r)]['worst']<=cfg['rank_train_worst_target']:
                rank=r;break
        report['selected_rank']=rank;report['rank_selection']='training worst snapshot POD floor only; family fixed'
        np.savez_compressed(out/'pod.npz',basis=P,scores=scores,singular=singular)
        stage('free_coefficient_bank_training')
        params,z,report['bank_training'],free_coeff=D.train_free_bank(train,n,k,rank,cfg['model_seed'],cfg['bank_steps'],
            cfg['bank_seconds'],out/'free_bank.pkl',scores,batch=cfg['bank_batch'],width=rank,n_ff=cfg['n_ff'],
            spatial_batch=cfg['spatial_batch'],callback=lambda curve:(report.update(bank_learning_curve=curve),write(report,result)))
        G=np.asarray(D.bank(params,D.coords(n),geom,n))
        Qb,Rb,ctr,ptr,report['bank_whitening']=project_coefficients(G,train)
        Xdev=dev.reshape(len(dev),-1);rot=Xdev@Qb;cdev=np.linalg.solve(Rb,rot.T).T
        pdev=np.sum((Xdev-cdev@G.T)**2,axis=1)
        report['projection_nonexpansion']=raw_projection_check(params,cdev,dev,n)
        assert report['projection_nonexpansion']['passed']
        report['training_bank_floor']=summary(np.sqrt(ptr)/np.linalg.norm(train.reshape(len(train),-1),axis=1))
        stage('head_training')
        params,z,report['head_training']=D.train_head(params,z,ctr,Rb,cfg['head_steps'],cfg['head_seconds'],cfg['model_seed'],out/'checkpoint.pkl',cfg)
        D.checkpoint(out/'checkpoint.pkl',params,z,cfg,dict(bank=report['bank_training'],head=report['head_training']),
                     extra=dict(bank=G,qr_Q=Qb,qr_R=Rb,train_coefficients=ctr))
        stage('representation')
        fit=D.representation(params,G,Rb,cdev,pdev,dev,z,starts=cfg['fit_starts'],budget=cfg['fit_budget'])
        report['bank_floor']=summary(fit['bank_error']);report['head_fit']=summary(fit['head_error'])
        report['head_fit']['stationary_count']=int(np.sum(fit['stationary']))
        report['head_fit']['metric']='snapshot vector norm, representation diagnostic'
        initial_norms=np.linalg.norm(Udev[:,0].reshape(len(Udev),-1),axis=1)
        state_norms=np.linalg.norm(dev.reshape(len(dev),-1),axis=1)
        initial_errors=fit['head_error']*state_norms/np.repeat(initial_norms,6)
        report['head_initial_normalized']=dict(all_times=summary(initial_errors.reshape(-1,6).max(1)),
            evolved=summary(initial_errors.reshape(-1,6)[:,1:].max(1)),initial=summary(initial_errors.reshape(-1,6)[:,0]))
        Cq,correction_singular=D.correction_directions(params,Rb,ctr,z)
        pred=np.asarray(D.head(params,jnp.asarray(fit['z'])))@G.T
        np.savez_compressed(out/'representation_fields.npz',truth=dev,prediction=pred.reshape(dev.shape),**fit,
                            correction_directions=Cq,correction_singular_values=correction_singular)
        # Train both architectures even if the representation target is missed.
        stage('operator_training')
        trds=dict(initial=Utr[:,0],viscosity=F.parameters(cfg['train_seed'],len(Utr))[:,-1:],targets=Utr[:,1:])
        dvds=dict(initial=Udev[:,0],viscosity=F.parameters(cfg['dev_seed'],len(Udev))[:,-1:],targets=Udev[:,1:])
        statistics=A.training_statistics(trds);tx,ty=A.common_model_arrays(trds,statistics);vx,vy=A.common_model_arrays(dvds,statistics)
        td=np.repeat(np.sum(trds['initial']**2,axis=(1,2,3,4))[:,None]/statistics['output_scale']**2,5,axis=1)
        vd=np.repeat(np.sum(dvds['initial']**2,axis=(1,2,3,4))[:,None]/statistics['output_scale']**2,5,axis=1)
        op_models=[];report['operators']=[]
        for index,spec in enumerate(cfg['operators']):
            ocfg=dict(cfg['operator_train'],seed=cfg['operator_train']['seed']+index)
            model,info=OT.train(tx,ty,vx,vy,spec,ocfg,out/spec['kind'],lambda path,obj:write(obj,path),O.checkpoint,td,vd)
            op_models.append((spec,model));report['operators'].append(info);write(report,result)
        O.checkpoint(out/'operator_statistics.pkl',statistics)
        del tx,ty,vx,vy,td,vd
        stage('reduced_operator_assembly')
        methods=[];assembly={}
        def add(name,fn,extra,kind):methods.append(dict(name=name,fn=fn,extra=extra,kind=kind))
        h=cfg['horizon'];dt=cfg['rom_dt'];steps=round(h/dt);every=steps//5
        Phi,lam,ids=R.test_modes(n,cfg['test_modes'])
        report['tensor_verification']=R.tensor_checks(G[:,:min(rank,16)],n,32)
        assert report['tensor_verification']['passed']
        start=time.monotonic();T=R.build_tensor(G,n,ids);Amat=Phi.T@G
        assembly['learned_weak']=dict(seconds=time.monotonic()-start,bytes=int(T.nbytes))
        full_parity=[]
        for ii in range(3):
            c=D.head(params,jnp.asarray(fit['z'][ii]));expected=R.dense_projection(Phi,G,c,n)
            full_parity.append(float(jnp.linalg.norm(R.contract(jnp.asarray(T),c)-expected)/jnp.linalg.norm(expected)))
        report['full_tensor_parity']=full_parity;assert max(full_parity)<1e-9
        np.savez_compressed(out/'weak_operators.npz',A=Amat,T=T,lam=lam,C=Cq)
        theta={name:params[name] for name in ('h','h_lin')}
        shared=tuple(jax.tree_util.tree_map(jnp.asarray,x) for x in (G,Qb,Rb,Amat,T,lam))
        shared_theta=jax.tree_util.tree_map(jnp.asarray,theta);shared_z=jnp.asarray(z)
        for q in cfg['q_values']:
            fn=R.make_run(dt,steps,every,k,q,budget=cfg['rom_budget'])
            extra=shared+(jnp.asarray(Cq[:,:q]),shared_theta,shared_z)
            add(f'nmrom_q{q}',fn,extra,'weak_lm')
        # Shared smooth tests comfortably exceed all online dimensions.
        pr=max(cfg['pod_weak_ranks']);start=time.monotonic();PT=R.build_tensor(P[:,:pr],n,ids)
        assembly['pod_weak']=dict(seconds=time.monotonic()-start,bytes=int(PT.nbytes))
        for r in cfg['pod_weak_ranks']:
            basis=P[:,:r];fn=R.make_run(dt,steps,every,0,r,linear=True,budget=cfg['rom_budget'])
            extra=tuple(jax.tree_util.tree_map(jnp.asarray,x) for x in (basis,basis,np.eye(r),Phi.T@basis,PT[:,:r,:r],lam,np.empty((r,0)),{},np.empty((1,0))))
            add(f'pod_weak_{r}',fn,extra,'weak_lm')
        # A single maximal tensor is sliced for the nested POD control ranks.
        start=time.monotonic();PL,PG=R.build_galerkin(P,n)
        assembly['pod_galerkin']=dict(seconds=time.monotonic()-start,bytes=int(PG.nbytes))
        np.savez_compressed(out/'pod_galerkin.npz',L=PL,T=PG)
        for r in cfg['pod_galerkin_ranks']:
            add(f'pod_galerkin_{r}',R.make_galerkin_run(dt,steps,every),tuple(jnp.asarray(x) for x in (P[:,:r],PL[:r,:r],PG[:r,:r,:r])),'galerkin')
        start=time.monotonic();BL,BG=R.build_galerkin(Qb,n)
        assembly['free_bank_galerkin']=dict(seconds=time.monotonic()-start,bytes=int(BG.nbytes))
        np.savez_compressed(out/'bank_galerkin.npz',L=BL,T=BG)
        add('free_bank_galerkin',R.make_galerkin_run(dt,steps,every),tuple(jnp.asarray(x) for x in (Qb,BL,BG)),'galerkin')
        for fdt in cfg['fom_dts']:
            ns=round(h/fdt);add(f'fom_dt{fdt}',F.make_solver(fdt,ns,ns//5),(geom,),'fom')
        for spec,model in op_models:
            for projected in (False,True):
                extra=(model,jnp.asarray(statistics['input_mean']),jnp.asarray(statistics['input_std']),jnp.asarray(statistics['output_scale']),geom)
                add(spec['kind']+('_projected' if projected else '_raw'),O.make_query(spec,n,projected),extra,'operator')
        report['assembly']=assembly;report['method_names']=[m['name'] for m in methods];write(report,result)
        stage('timed_reference_fields')
        count=cfg['timed_cases'];parameters=F.parameters(cfg['dev_seed'],len(Udev))[:count]
        refs=[];fine=[]
        rs=round(h/cfg['reference_dt']);rf=F.make_solver(cfg['reference_dt'],rs,rs//5)
        fs=round(h/cfg['fine_reference_dt']);ff=F.make_solver(cfg['fine_reference_dt'],fs,fs//5);fg=F.geometry(cfg['fine_reference_n'])
        for par in parameters:
            refs.append(np.asarray(rf(jnp.asarray(F.initial(n,par)),par[-1],geom)))
            fine.append(np.asarray(ff(jnp.asarray(F.initial(cfg['fine_reference_n'],par)),par[-1],fg)))
        refs=np.stack(refs);fine=np.stack(fine)
        np.savez_compressed(out/'timing_references.npz',same_grid=refs,fine_grid=fine,parameters=parameters)
        stage('compile_complete_queries')
        for method in methods:
            st=time.perf_counter();value=method['fn'](jnp.asarray(Udev[0,0]),parameters[0,-1],*method['extra']);jax.block_until_ready(value)
            method['compile_warm_seconds']=time.perf_counter()-st
            print('COMPILED',method['name'],method['compile_warm_seconds'],flush=True)
        stage('matched_timing')
        rows=[];saved={};metadata={};burn(3.)
        for case in range(count):
            u0=jnp.asarray(Udev[case,0]);nu=parameters[case,-1]
            for repetition in range(cfg['repetitions']):
                burn(1.)
                invocation_fields=[]
                order=np.roll(np.arange(len(methods)),case+repetition)
                if repetition%2:order=order[::-1]
                for order_index,j in enumerate(order):
                    method=methods[j];st=time.perf_counter();value=method['fn'](u0,nu,*method['extra']);jax.block_until_ready(value)
                    gpu=time.perf_counter()-st
                    host=jax.tree_util.tree_map(np.asarray,value);total=time.perf_counter()-st
                    field=host[0] if method['kind']=='weak_lm' else host
                    field=field.reshape(6,3,n,n,n)
                    row=dict(case=case,repetition=repetition,order_index=order_index,method=method['name'],gpu_seconds=gpu,
                             with_host_seconds=total,field_sha256=sha(field),finite=bool(np.isfinite(field).all()))
                    if method['kind']=='weak_lm':
                        row['cold']=host[1];row['steps']=host[2]
                    # Accuracy uses this invocation's actual retained result.
                    row['same_grid_errors']=np.linalg.norm((field-refs[case]).reshape(6,-1),axis=1)/np.linalg.norm(Udev[case,0])
                    rows.append(row)
                    invocation_fields.append((row,field))
                    if repetition==0:saved[f'{method["name"]}__case{case}']=field
                # Compute physical error from this same timed invocation. Host
                # refinement sits after the GPU block; the next block burns in.
                from scipy.signal import resample
                for row,field in invocation_fields:
                    finefield=field
                    for axis in (-3,-2,-1):finefield=resample(finefield,cfg['fine_reference_n'],axis=axis)
                    row['fine_grid_errors']=np.linalg.norm((finefield-fine[case]).reshape(6,-1),axis=1)/np.linalg.norm(fine[case,0])
                    key=f'{row["method"]}__case{case}'
                    if sha(saved[key])!=row['field_sha256']:
                        saved[key+f'__rep{repetition}']=field
                write(rows,out/'timing_rows.json')
                print('TIMED',case,repetition,flush=True)
        np.savez_compressed(out/'timed_fields.npz',**saved)
        report['timing_summary']={}
        for method in methods:
            subset=[r for r in rows if r['method']==method['name']]
            report['timing_summary'][method['name']]=dict(kind=method['kind'],compile_warm_seconds=method['compile_warm_seconds'],
                gpu_seconds=summary([r['gpu_seconds'] for r in subset]),with_host_seconds=summary([r['with_host_seconds'] for r in subset]),
                worst_evolved_same_grid=summary([max(r['same_grid_errors'][1:]) for r in subset]),
                worst_evolved_fine_grid=summary([max(r['fine_grid_errors'][1:]) for r in subset]),
                nonfinite_invocations=sum(not r['finite'] for r in subset),invocations=len(subset))
        report['complete']=True;stage('complete');print('ALL-DONE',flush=True)
    except Exception as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc();write(report,result)
        print(report['traceback'],flush=True);raise


if __name__=='__main__':main()
