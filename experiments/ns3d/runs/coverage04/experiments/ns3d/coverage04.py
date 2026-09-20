"""Shared periodic translation coverage, initialized heads and residual operators.

This is a development training/screening job. Any later paper final panel must
freeze checkpoints, configurations and efficient FOM controls before test access.
"""
from __future__ import annotations
import argparse,gc,json,os,pickle,subprocess,time,traceback
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_fom as F
import ns3d_model as D
import ns3d_rom as R
import translation_cohort as TC
import head_pca as HP
import dataset_adapter as A
import operator_adapter as O
from operators import training as OT
from comparison import projection_floor,raw_projection_check,burn
from extra03 import with_coordinates
from pilot import generate,write,summary,sha


def capacity(Utr,Udev,cfg,out):
    out.mkdir(parents=True,exist_ok=True); n=cfg['n']; rank=cfg['coverage_rank']
    report=dict(complete=False,stage='POD',rank=rank,final_cohort_opened=False,selection='rank fixed before augmented training')
    def save():write(report,out/'screen.json')
    train=Utr.reshape(-1,3,n,n,n); dev=Udev.reshape(-1,3,n,n,n)
    tn=np.repeat(np.linalg.norm(Utr[:,0].reshape(len(Utr),-1),axis=1),6)
    vn=np.repeat(np.linalg.norm(Udev[:,0].reshape(len(Udev),-1),axis=1),6)
    variance=tn**2/(3*n**3); start=time.monotonic();save()
    P,scores,singular=D.pod_gpu(train,rank)
    report['POD_seconds']=time.monotonic()-start;report['POD']={}
    for r in cfg['pod_screen_ranks']:
        tr=projection_floor(train,P[:,:r])*np.linalg.norm(train.reshape(len(train),-1),axis=1)/tn
        dv=projection_floor(dev,P[:,:r])*np.linalg.norm(dev.reshape(len(dev),-1),axis=1)/vn
        report['POD'][str(r)]=dict(train_initial_normalized=summary(tr),development_initial_normalized=summary(dv))
        save()
    np.savez_compressed(out/'pod.npz',basis=P,singular=singular)
    del P;gc.collect();report['stage']='free_bank';save()
    params,z,info,_=D.train_free_bank(train,n,64,rank,cfg['model_seed'],cfg['coverage_bank_steps'],cfg['coverage_bank_seconds'],out/'free_bank.pkl',scores,
        batch=16,width=rank,n_ff=cfg['n_ff'],spatial_batch=1024,snapshot_variance=variance,checkpoint_every=5000,
        callback=lambda curve:(report.update(bank_curve=curve),save()))
    report['bank_training']=info
    G=np.asarray(D.bank(params,D.coords(n),F.geometry(n),n));Q,Rb,ctr,ptr,whitening=D.whiten(G,train)
    Xdev=dev.reshape(len(dev),-1);cdev=np.linalg.solve(Rb,(Xdev@Q).T).T;pdev=np.sum((Xdev-cdev@G.T)**2,axis=1)
    report['whitening']=whitening
    report['training_bank_initial_normalized']=summary(np.sqrt(ptr)/tn)
    report['development_bank_initial_normalized']=summary(np.sqrt(pdev)/vn)
    report['projection_nonexpansion']=raw_projection_check(params,cdev,dev,n);assert report['projection_nonexpansion']['passed']
    D.checkpoint(out/'frozen_bank.pkl',params,z,{**cfg,'k':64,'r':rank},dict(bank=info,rank=rank,head_status='untrained placeholder; use separately saved selected head'),extra=dict(bank=G,qr_Q=Q,qr_R=Rb,train_coefficients=ctr))
    np.savez_compressed(out/'bank_fields.npz',development_truth=dev,development_prediction=(cdev@G.T).reshape(dev.shape),
        train_floor=np.sqrt(ptr)/tn,development_floor=np.sqrt(pdev)/vn)
    report['heads']={};save()
    for spec in cfg['head_variants']:
        label=spec['label'];k=spec['k'];report['stage']=label;save()
        candidate,codes,initialization=HP.initialize(params,ctr,Rb,k,cfg['model_seed']+k,variance,n)
        span=np.linalg.qr(Rb@np.asarray(candidate['h_lin']).T,mode='reduced')[0]
        center=Rb@np.asarray(candidate['h'][-1][1]);Ydev=cdev@Rb.T
        linear_residual=(Ydev-center)-((Ydev-center)@span)@span.T
        initial_dev=np.sqrt(pdev+np.sum(linear_residual**2,axis=1))/vn
        initial_label=f'pca{k}_initial'
        if initial_label not in report['heads']:
            initial_fit=D.representation(candidate,G,Rb,cdev,pdev,dev,codes,starts=8,budget=300)
            initial_prediction=np.asarray(D.head(candidate,jnp.asarray(initial_fit['z'])))@G.T
            initial_errors=np.linalg.norm(initial_prediction-Xdev,axis=1)/vn
            initial_coefficient_residual=(np.asarray(D.head(candidate,jnp.asarray(codes)))-ctr)@Rb.T
            initial_train=np.sqrt(ptr+np.sum(initial_coefficient_residual**2,axis=1))/tn
            initial_spec=dict(label=initial_label,k=k,free_codes=False,initialization_only=True)
            initial_record=dict(initialization=initialization,training=dict(steps=0,stage='affine initialization candidate',converged_claim=False),
                head_initial_normalized=summary(initial_errors),training_stored_code_initial_normalized=summary(initial_train),
                affine_PCA_development_initial_normalized=summary(initial_dev),error_by_case_time=initial_errors.reshape(len(Udev),6),
                stationary_count=int(np.sum(initial_fit['stationary'])),fits=initial_fit,configuration=initial_spec,final_cohort_opened=False)
            report['heads'][initial_label]=initial_record
            D.checkpoint(out/(initial_label+'.pkl'),candidate,codes,{**cfg,'k':k,'r':rank,'head_variant':initial_label,'free_codes':False},initial_record)
            np.savez_compressed(out/(initial_label+'_fields.npz'),prediction=initial_prediction.reshape(dev.shape),training_stored_code_error=initial_train,affine_PCA_development_error=initial_dev)
            save()
        hcfg={**cfg,'k':k,'r':rank,'head_variant':label,'free_codes':spec['free_codes']}
        candidate,codes,training=HP.train(candidate,codes,ctr,Rb,hcfg,out/(label+'.pkl'),variance,cfg['model_seed']+k)
        code_residual=(np.asarray(D.head(candidate,jnp.asarray(codes)))-ctr)@Rb.T
        train_errors=np.sqrt(ptr+np.sum(code_residual**2,axis=1))/tn
        fit=D.representation(candidate,G,Rb,cdev,pdev,dev,codes,starts=8,budget=300)
        prediction=np.asarray(D.head(candidate,jnp.asarray(fit['z'])))@G.T
        errors=np.linalg.norm(prediction-Xdev,axis=1)/vn
        record=dict(initialization=initialization,training=training,head_initial_normalized=summary(errors),
            training_stored_code_initial_normalized=summary(train_errors),
            affine_PCA_development_initial_normalized=summary(initial_dev),
            error_by_case_time=errors.reshape(len(Udev),6),stationary_count=int(np.sum(fit['stationary'])),
            fits=fit,configuration=spec,final_cohort_opened=False)
        report['heads'][label]=record
        D.checkpoint(out/(label+'.pkl'),candidate,codes,hcfg,record)
        np.savez_compressed(out/(label+'_fields.npz'),prediction=prediction.reshape(dev.shape),training_stored_code_error=train_errors,
            affine_PCA_development_error=initial_dev)
        save()
    report['stage']='complete';report['complete']=True;save();return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    report=dict(complete=False,kind='NS3D matched translated-coverage development screen',config=cfg,
        source_commit=os.environ.get('SOURCE_COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),
        backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
        final_cohort_opened=False,limitations=['opened development data','a training budget is not convergence',
        'augmentation reuses exact PDE symmetries, not independent new physical samples',
        'head fits are representation diagnostics, not online rollout errors'])
    def save():write(report,out/'result.json')
    def stage(name):report['stage']=name;save();print('STAGE',name,flush=True)
    try:
        report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
        assert report['backend']=='gpu' and report['x64'] and report['precision']=='highest'
        assert cfg['initial_amplitude']==F.INITIAL_AMPLITUDE
        stage('symmetry_verification');report['symmetry']=TC.verify();assert report['symmetry']['passed'];save()
        n=cfg['n'];h=cfg['horizon'];stage('base_data_regeneration')
        base,report['train_data']=generate(n,cfg['dt'],h,cfg['train_cases'],cfg['train_seed'],out/'train_data.npz')
        dev,report['dev_data']=generate(n,cfg['dt'],h,cfg['dev_cases'],cfg['dev_seed'],out/'dev_data.npz')
        params=F.parameters(cfg['train_seed'],len(base));table=TC.membership(len(base),n,cfg['augmentation_copies'],cfg['augmentation_seed'])
        report['augmentation']=TC.manifest(table,n,cfg['augmentation_seed'],base,params);write(report['augmentation'],out/'membership.json')
        stage('shared_training_augmentation');train=TC.augment(base,table);report['augmentation']['augmented_states_sha256']=sha(train);del base;gc.collect();save()
        stage('coverage_capacity_and_heads')
        report['capacity']=capacity(train,dev,cfg,out/'capacity');save();gc.collect();jax.clear_caches()
        stage('shared_operator_training')
        trds=dict(initial=train[:,0],viscosity=params[table[:,0],-1:],targets=train[:,1:]-train[:,0,None])
        dvds=dict(initial=dev[:,0],viscosity=F.parameters(cfg['dev_seed'],len(dev))[:,-1:],targets=dev[:,1:]-dev[:,0,None])
        statistics=A.training_statistics(trds);O.checkpoint(out/'operator_statistics.pkl',statistics)
        tx,ty=A.common_model_arrays(trds,statistics);vx,vy=A.common_model_arrays(dvds,statistics)
        td=np.repeat(np.sum(trds['initial']**2,axis=(1,2,3,4))[:,None]/statistics['output_scale']**2,5,1)
        vd=np.repeat(np.sum(dvds['initial']**2,axis=(1,2,3,4))[:,None]/statistics['output_scale']**2,5,1)
        del train,trds;gc.collect();models=[];report['operators']=[]
        for index,spec in enumerate(cfg['operators']):
            assert spec['output_residual_initial']
            extra=spec['kind'] in ('deeponet3d','transolver3d')
            xx=with_coordinates(tx) if extra else tx;vv=with_coordinates(vx) if extra else vx
            model,info=OT.train(xx,ty,vv,vy,spec,{**cfg['operator_train'],'seed':cfg['operator_train']['seed']+index},out/spec['kind'],lambda path,obj:write(obj,path),O.checkpoint,td,vd)
            models.append((spec,model));report['operators'].append(info);save();del xx,vv;gc.collect()
        del tx,ty,vx,vy,td,vd;gc.collect();jax.clear_caches()
        stage('complete_query_development_controls');geom=F.geometry(n);methods=[]
        for spec,model in models:
            for projected in (False,True):
                extra=(model,statistics['input_mean'],statistics['input_std'],statistics['output_scale'],geom)
                methods.append((spec['kind']+'_increment'+('_projected' if projected else '_raw'),O.make_query(spec,n,projected),jax.tree_util.tree_map(jax.device_put,extra)))
        heads=report['capacity']['heads'];selected=min(heads,key=lambda key:heads[key]['head_initial_normalized']['worst'])
        eligible=(report['capacity']['development_bank_initial_normalized']['worst']<=cfg.get('rollout_bank_threshold',.05)
            and heads[selected]['head_initial_normalized']['worst']<=cfg.get('rollout_head_threshold',.08))
        report['larger_rollout_gate']=dict(eligible=eligible,selected_head=selected,
            bank_threshold=cfg.get('rollout_bank_threshold',.05),head_threshold=cfg.get('rollout_head_threshold',.08),
            gate_scope='development representation permits a measured rollout; no final acceptance implied')
        if eligible:
            stage('larger_dense_weak_rollout_assembly')
            with (out/'capacity/frozen_bank.pkl').open('rb') as f:frozen=pickle.load(f)
            with (out/'capacity'/(selected+'.pkl')).open('rb') as f:head=pickle.load(f)
            G=frozen['extra']['bank'];Q=frozen['extra']['qr_Q'];Rb=frozen['extra']['qr_R']
            theta={key:head['params'][key] for key in ('h','h_lin')};z=head['codes']
            C,singular=D.correction_directions(head['params'],Rb,frozen['extra']['train_coefficients'],z)
            Phi,lam,ids=R.test_modes(n,cfg['test_modes']);Amat=Phi.T@G
            selector=(np.asarray([x['wave'] for x in ids])%n,np.asarray([x['polarization'] for x in ids]),np.asarray([x['kind']=='cos' for x in ids]),geom)
            shared=jax.tree_util.tree_map(jax.device_put,(G,Q,Rb,Amat,selector,lam));theta=jax.tree_util.tree_map(jax.device_put,theta);zc=jax.device_put(z)
            k=heads[selected]['configuration']['k'];steps=round(h/cfg['rom_dt'])
            np.savez_compressed(out/'dense_weak_operators.npz',A=Amat,lam=lam,C=C,correction_singular=singular)
            write(ids,out/'dense_weak_test_modes.json')
            for q in cfg.get('larger_rollout_q_values',[0,32,64]):
                assert cfg['test_modes']>=4*(k+q)
                methods.append((f'nmrom_{selected}_q{q}',R.make_run(cfg['rom_dt'],steps,steps//5,k,q,budget=cfg['rom_budget'],retain_states=True,dense_n=n),
                    shared+(jax.device_put(C[:,:q]),theta,zc)))
            with np.load(out/'capacity/pod.npz') as f:P=f['basis']
            for rank in sorted({k+q for q in cfg.get('larger_rollout_q_values',[0,32,64])}):
                basis=P[:,:rank]
                extra=(basis,basis,np.eye(rank),Phi.T@basis,selector,lam,np.empty((rank,0)),{},np.empty((1,0)))
                methods.append((f'pod_weak_{rank}',R.make_run(cfg['rom_dt'],steps,steps//5,0,rank,linear=True,budget=cfg['rom_budget'],retain_states=True,dense_n=n),jax.tree_util.tree_map(jax.device_put,extra)))
            del P,Phi,G,Q,Rb,C,frozen,head;gc.collect()
        save()
        for dt in cfg['fom_dts']:
            steps=round(h/dt);assert steps%5==0
            methods.append((f'fom_dt{dt}',F.make_solver(dt,steps,steps//5),geom))
        reference=F.make_solver(cfg['reference_dt'],round(h/cfg['reference_dt']),round(h/cfg['reference_dt'])//5)
        pars=F.parameters(cfg['dev_seed'],len(dev));refs=[]
        for case in range(cfg['timed_cases']):
            refs.append(np.asarray(reference(jnp.asarray(dev[case,0]),pars[case,-1],geom)))
        refs=np.stack(refs);np.savez_compressed(out/'timing_references.npz',same_grid=refs,parameters=pars[:len(refs)])
        for name,fn,extra in methods:
            args=(extra,) if name.startswith('fom_') else extra
            value=fn(jnp.asarray(dev[0,0]),pars[0,-1],*args)
            jax.block_until_ready(value)
        rows=[];fields={};histories={};burn(3.)
        for case in range(cfg['timed_cases']):
            u0=jnp.asarray(dev[case,0]);nu=pars[case,-1]
            for repetition in range(cfg['repetitions']):
                burn(1.);order=np.roll(np.arange(len(methods)),case+repetition)
                if repetition%2:order=order[::-1]
                for oi,index in enumerate(order):
                    name,fn,extra=methods[index];args=(extra,) if name.startswith('fom_') else extra
                    start=time.perf_counter();value=fn(u0,nu,*args);jax.block_until_ready(value);gpu=time.perf_counter()-start
                    weak=name.startswith(('nmrom_','pod_weak_'));host=jax.tree_util.tree_map(np.asarray,value)
                    field=(host[0] if weak else host).reshape(6,3,n,n,n);total=time.perf_counter()-start;finite=bool(np.isfinite(field).all());key=f'{name}__case{case}'
                    if repetition and sha(fields[key])!=sha(field):key+=f'__rep{repetition}'
                    fields[key]=field
                    errors=np.linalg.norm((field-refs[case]).reshape(6,-1),axis=1)/np.linalg.norm(dev[case,0])
                    row=dict(case=case,repetition=repetition,order_index=oi,method=name,gpu_seconds=gpu,with_host_seconds=total,
                        field_sha256=sha(field),finite=finite,same_grid_errors=errors if finite else None)
                    if weak:
                        row.update(cold=host[1],steps=host[2],state_sha256=sha(host[3]));histories[f'{name}__case{case}__rep{repetition}']=host[3]
                    rows.append(row)
                write(rows,out/'timing_rows.json');print('TIMED',case,repetition,flush=True)
        np.savez_compressed(out/'timed_fields.npz',**fields)
        if histories:np.savez_compressed(out/'latent_histories.npz',**histories)
        report['method_names']=[x[0] for x in methods];report['timing_summary']={}
        for name,_,_ in methods:
            arm=[r for r in rows if r['method']==name];times=np.asarray([r['gpu_seconds'] for r in arm]);valid=[r for r in arm if r['finite']]
            report['timing_summary'][name]=dict(gpu_seconds_median=float(np.median(times)),invocations=len(arm),nonfinite_invocations=len(arm)-len(valid),
                timing_outliers_above_three_median=int(np.sum(times>3*np.median(times))),
                worst_evolved_same_grid=max(max(r['same_grid_errors'][1:]) for r in valid) if len(valid)==len(arm) else None)
        report['complete']=True;stage('complete');print('ALL-DONE',flush=True)
    except Exception as error:
        report['error']=str(error);report['traceback']=traceback.format_exc();save();print(report['traceback'],flush=True);raise


if __name__=='__main__':main()
