"""Gated first NS3D verification + learned bank/head pilot, no final cohort."""
from __future__ import annotations
import argparse
import hashlib
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


def normalize(value):
    if isinstance(value,dict):
        return {str(k):normalize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [normalize(v) for v in value]
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    return value


def write(report,path):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(normalize(report),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def summary(a):
    a=np.asarray(a,dtype=float)
    return dict(mean=float(np.mean(a)),median=float(np.median(a)),worst=float(np.max(a)),
                over_5pct=int(np.sum(a>.05)),count=int(a.size))


def generate(n,dt,horizon,count,seed,dest):
    parameters=F.parameters(seed,count)
    nsteps=round(horizon/dt)
    run=F.make_solver(dt,nsteps,nsteps//5)
    geom=F.geometry(n)
    states=[]
    records=[]
    start=time.monotonic()
    for i,param in enumerate(parameters):
        u0=F.initial(n,param)
        trajectory=np.asarray(run(jnp.asarray(u0),param[-1],geom))
        if not np.all(np.isfinite(trajectory)):
            raise RuntimeError(f'nonfinite trajectory cohort {seed}, index {i}')
        diagnostics=tuple(np.asarray(a) for a in F.diagnostics(jnp.asarray(trajectory),geom))
        if float(np.max(diagnostics[2]))>1e-10:
            raise RuntimeError('trajectory divergence exceeded the gate')
        if np.max(np.diff(diagnostics[0]))>1e-10:
            raise RuntimeError('unforced trajectory energy increased')
        states.append(trajectory)
        records.append(dict(index=i,field_sha256=sha(trajectory),energy=diagnostics[0],
                            divergence=diagnostics[2],initial_component_rms=diagnostics[3][0]))
        if i%8==0 or i+1==count:
            print(f'data seed={seed} {i+1}/{count} elapsed={time.monotonic()-start:.1f}',flush=True)
    states=np.stack(states)
    np.savez_compressed(dest,parameters=parameters,states=states)
    return states,dict(seed=seed,count=count,n=n,dt=dt,horizon=horizon,
                       parameter_sha256=sha(parameters),states_sha256=sha(states),
                       records=records,seconds=time.monotonic()-start)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report=dict(complete=False,stage='preflight',kind='ns3d_verified_representation_pilot',
                config=config,source_commit=os.environ.get('SOURCE_COMMIT'),
                job_id=os.environ.get('SLURM_JOB_ID'),backend=jax.default_backend(),
                jax_version=jax.__version__,x64=bool(jax.config.jax_enable_x64),
                precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
                metric='velocity L2, initial norm for trajectories; state norm for snapshot reconstruction',
                final_cohort_opened=False,
                limitations=['development pilot, one training seed',
                             'dense initial projection is not hyper-reduced',
                             'no neural-operator or matched-cost comparison in this first attempt'])
    result=out/'result.json'
    try:
        report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total',
                                               '--format=csv,noheader'],text=True).strip()
        if report['backend']!='gpu' or not report['x64'] or report['precision']!='highest':
            raise RuntimeError('mandatory GPU/precision preflight failed')
        if config['initial_amplitude'] != F.INITIAL_AMPLITUDE:
            raise RuntimeError('initial amplitude differs from the committed family specification')
        print('jax_backend=gpu',flush=True)
        write(report,result)
        report['operator_verification']=V.operator_checks()
        write(report,result)
        if not report['operator_verification']['passed']:
            raise RuntimeError('operator verification failed; training not permitted')
        report['stage']='reference_verification'
        write(report,result)
        report['requested_config']=dict(config)
        report['reference_attempts']=[]
        for candidate_n in config['reference_mesh_candidates']:
            attempt=V.reference_checks(candidate_n,config['dt'],config['horizon'],
                                       count=config['verification_cases'],outdir=out/f'reference_N{candidate_n}')
            report['reference_attempts'].append(attempt)
            report['reference_verification']=attempt
            write(report,result)
            if attempt['passed']:
                config=dict(config,n=candidate_n)
                report['config']=config
                write(config,out/'effective_config.json')
                break
        if not report['reference_verification']['passed']:
            raise RuntimeError('physical reference budget failed; training not permitted')
        n=config['n'];k=config['k'];rank=config['rank']
        report['stage']='data';write(report,result)
        Utr,tr_info=generate(n,config['dt'],config['horizon'],config['train_cases'],config['train_seed'],
                            out/'train_data.npz')
        report['train_data']=tr_info;write(report,result)
        Udev,dev_info=generate(n,config['dt'],config['horizon'],config['dev_cases'],config['dev_seed'],
                              out/'dev_data.npz')
        report['dev_data']=dev_info;write(report,result)
        train=Utr.reshape(-1,3,n,n,n)
        dev=Udev.reshape(-1,3,n,n,n)
        report['stage']='bank_training';write(report,result)
        def callback(curve):
            report['bank_learning_curve']=curve
            write(report,result)
        params,z,bank_info=D.train_bank(train,n,k,rank,config['model_seed'],config['bank_steps'],
                                       config['bank_seconds'],out/'joint_checkpoint.pkl',
                                       batch=config['batch'],width=config['width'],callback=callback)
        report['bank_training']=bank_info;write(report,result)
        G=np.asarray(D.bank(params,D.coords(n),F.geometry(n),n))
        Qb,Rb,ctr,ptr,whitening=D.whiten(G,train)
        _,_,cdev,pdev,_=D.whiten(G,dev)
        report['bank_whitening']=whitening
        report['stage']='head_refinement';write(report,result)
        params,z,hinfo=D.train_head(params,z,ctr,Rb,config['head_steps'],config['head_seconds'],
                                    config['model_seed'],out/'checkpoint.pkl',config)
        report['head_training']=hinfo;write(report,result)
        # Saved checkpoint includes exact grid bank/QR; future fields can be audited.
        D.checkpoint(out/'checkpoint.pkl',params,z,config,dict(bank=bank_info,head=hinfo),
                     extra=dict(bank=G,qr_R=Rb,train_coefficients=ctr))
        report['stage']='representation';write(report,result)
        fit=D.representation(params,G,Rb,cdev,pdev,dev,z,starts=config['fit_starts'],
                              budget=config['fit_budget'])
        report['bank_floor']=summary(fit['bank_error'])
        report['head_fit']=summary(fit['head_error'])
        report['head_fit']['stationary_count']=int(np.sum(fit['stationary']))
        report['head_fit']['stop_reason_counts']={str(int(reason)):int(np.sum(fit['reasons']==reason))
                                                 for reason in np.unique(fit['reasons'])}
        # Initial-normalized view matches trajectory metrics and prevents denominator drift.
        initial_norm=np.linalg.norm(Udev[:,0].reshape(len(Udev),-1),axis=1)
        field_norm=np.linalg.norm(dev.reshape(len(dev),-1),axis=1)
        initial_error=fit['head_error']*field_norm/np.repeat(initial_norm,6)
        report['head_initial_normalized']=dict(all_times=summary(initial_error.reshape(-1,6).max(1)),
                                               evolved=summary(initial_error.reshape(-1,6)[:,1:].max(1)),
                                               initial=summary(initial_error.reshape(-1,6)[:,0]))
        prediction=np.asarray(D.head(params,jnp.asarray(fit['z'])))@G.T
        train_prediction=np.asarray(D.head(params,jnp.asarray(z)))@G.T
        train_error=np.linalg.norm(train_prediction-train.reshape(len(train),-1),axis=1)/np.linalg.norm(
            train.reshape(len(train),-1),axis=1)
        report['training_reconstruction']=summary(train_error)
        Cq,singular=D.correction_directions(params,Rb,ctr,z)
        P,pod_singular=D.pod(train,rank)
        X=dev.reshape(len(dev),-1)
        report['pod_floors']={}
        for dim in config['pod_ranks']:
            dim=min(dim,P.shape[1])
            err=np.linalg.norm(X-(X@P[:,:dim])@P[:,:dim].T,axis=1)/np.linalg.norm(X,axis=1)
            report['pod_floors'][str(dim)]=summary(err)
        # Full bank, exact cold fit and correction directions are independent controls.
        np.savez_compressed(out/'representation_fields.npz',truth=dev,prediction=prediction.reshape(dev.shape),
                            **fit,initial_normalized_errors=initial_error,
                            correction_directions=Cq,correction_singular_values=singular,
                            pod_basis=P,pod_singular_values=pod_singular)
        report['stage']='tensor_verification';write(report,result)
        report['tensor_verification']=R.tensor_checks(G[:,:min(rank,16)],n,m=32)
        if not report['tensor_verification']['passed']:
            raise RuntimeError('weak tensor verification failed')
        # Prepare the complete current-method q/POD/free-bank controls for the next panel.
        Phi,lam,ids=R.test_modes(n,config['test_modes'])
        T=R.build_tensor(G,n,ids)
        full_parity=[]
        for index in range(min(3,len(fit['z']))):
            c=D.head(params,jnp.asarray(fit['z'][index]))
            actual=R.contract(jnp.asarray(T),c)
            expected=R.dense_projection(Phi,G,c,n)
            full_parity.append(float(jnp.linalg.norm(actual-expected)/jnp.maximum(
                jnp.linalg.norm(expected),1e-300)))
        report['full_tensor_heldout_parity']=full_parity
        if max(full_parity)>1e-9:
            raise RuntimeError('full-rank weak tensor parity failed')
        np.savez_compressed(out/'reduced_operators.npz',bank=G,qr_Q=Qb,qr_R=Rb,A=Phi.T@G,
                            T=T,lam=lam,C=Cq,pod_basis=P)
        report['reduced_operator_bytes']=int(T.nbytes)
        report['head_representation_target_passed']=bool(report['head_fit']['worst']<=.05)
        report['stage']='complete';report['complete']=True
        write(report,result)
        print('ALL-DONE',flush=True)
    except Exception as error:
        report['error']=str(error)
        report['traceback']=traceback.format_exc()
        write(report,result)
        print(report['traceback'],flush=True)
        raise


if __name__=='__main__':
    main()
