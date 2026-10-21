"""Bounded larger-bank and wider-head diagnostic on the unchanged NS3D family."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_fom as F
import ns3d_model as D
from comparison import projection_floor,raw_projection_check
from pilot import summary,write


def run(Utr,Udev,cfg,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);report=dict(complete=False,stage='POD_capacity',family_amplitude=cfg['initial_amplitude'],
       target='initial-normalized5pct',final_cohort_opened=False,selection='training-only rank screen, opened-development head diagnostics')
    def save():write(report,out/'screen.json')
    save();n=cfg['n'];train=Utr.reshape(-1,3,n,n,n);dev=Udev.reshape(-1,3,n,n,n)
    trnorm=np.repeat(np.linalg.norm(Utr[:,0].reshape(len(Utr),-1),axis=1),6);dvnorm=np.repeat(np.linalg.norm(Udev[:,0].reshape(len(Udev),-1),axis=1),6)
    tnorm=np.linalg.norm(train.reshape(len(train),-1),axis=1);vnorm=np.linalg.norm(dev.reshape(len(dev),-1),axis=1)
    start=time.monotonic();P,scores,singular=D.pod_gpu(train,max(cfg['capacity_ranks']));report['POD_seconds']=time.monotonic()-start;report['POD']={}
    for rank in cfg['capacity_ranks']:
        tf=projection_floor(train,P[:,:rank])*tnorm/trnorm;vf=projection_floor(dev,P[:,:rank])*vnorm/dvnorm
        report['POD'][str(rank)]=dict(train_initial_normalized=summary(tf),development_initial_normalized=summary(vf))
    # Training-only mean-square criterion is capacity guidance, not a proof that
    # any bank meets a worst-case target. The larger candidate is retained if needed.
    rank=cfg['capacity_ranks'][-1]
    for candidate in cfg['capacity_ranks']:
        if report['POD'][str(candidate)]['train_initial_normalized']['mean']<=.02:rank=candidate;break
    report['selected_rank']=rank;report['stage']='free_bank';save()
    initial_variance=trnorm**2/(3*n**3)
    params,z,info,coeff=D.train_free_bank(train,n,32,rank,cfg['capacity_seed'],cfg['capacity_bank_steps'],cfg['capacity_bank_seconds'],out/'free_bank.pkl',scores,
         batch=16,width=rank,n_ff=256,spatial_batch=1024,snapshot_variance=initial_variance,checkpoint_every=2000,
         callback=lambda curve:(report.update(bank_curve=curve),save()))
    report['bank_training']=info;G=np.asarray(D.bank(params,D.coords(n),F.geometry(n),n));Q,Rb,ctr,ptr,whitening=D.whiten(G,train)
    Xdev=dev.reshape(len(dev),-1);cdev=np.linalg.solve(Rb,(Xdev@Q).T).T;pdev=np.sum((Xdev-cdev@G.T)**2,axis=1)
    report['whitening']=whitening;report['training_bank_initial_normalized']=summary(np.sqrt(ptr)/trnorm);report['development_bank_initial_normalized']=summary(np.sqrt(pdev)/dvnorm)
    report['projection_nonexpansion']=raw_projection_check(params,cdev,dev,n);assert report['projection_nonexpansion']['passed'];save()
    np.savez_compressed(out/'pod_capacity.npz',singular=singular)
    report['heads']={}
    for k in cfg['capacity_head_k']:
        report['stage']=f'head_K{k}';save();hp=D.init(jax.random.PRNGKey(cfg['capacity_seed']+k),k,rank,width=rank,n_ff=256)
        candidate={**params,'h':hp['h'],'h_lin':hp['h_lin']};zk=.25*scores[:,:k]/max(np.sqrt(np.mean(scores[:,:k]**2)),1e-12)
        candidate,zk,training=D.train_head(candidate,zk,ctr,Rb,cfg['capacity_head_steps'],cfg['capacity_head_seconds'],cfg['capacity_seed']+k,out/f'head_K{k}.pkl',cfg,
            snapshot_variance=initial_variance)
        fit=D.representation(candidate,G,Rb,cdev,pdev,dev,zk,starts=4,budget=250)
        errors=fit['head_error']*vnorm/dvnorm
        record=dict(training=training,head_current_normalized=summary(fit['head_error']),head_initial_normalized=summary(errors),stationary_count=int(np.sum(fit['stationary'])),
            error_by_case_time=errors.reshape(len(Udev),6),fits=fit,head_terminal_hidden_width=rank)
        report['heads'][str(k)]=record
        D.checkpoint(out/f'head_K{k}.pkl',candidate,zk,cfg,record)
        save()
    D.checkpoint(out/'frozen_bank.pkl',params,z,cfg,dict(bank=info,rank=rank),extra=dict(bank=G,qr_Q=Q,qr_R=Rb,train_coefficients=ctr))
    report['complete']=True;report['stage']='complete';save();return report
