"""Matched heat training-cohort expansion and independent initialization seeds.

This driver produces frozen training artifacts and development diagnostics only.
Its results are not timed trajectory comparisons and never open final data.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, time
from pathlib import Path
import numpy as np
import jax
import common as C
import train as T
import rom as R


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def train_seed(cfg,out,trainp,valp,data,valid,metadata):
    out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='heat3d-head-pca-diagnostic-v1',config=cfg,**metadata,
        complete=False,final_cohort_opened=False,bank={},heads=[],operators=[],meshes=[],candidates=[],
        status='training_bank',scope='training and development representation diagnostics; no paired timing claim')
    save=lambda:C.dump(out/'result.json',record);save()
    C.dump(out/'cohorts.json',dict(training_parameters=trainp.tolist(),validation_parameters=valp.tolist(),
        train_sha256=C.sha(trainp),validation_sha256=C.sha(valp),reserved_final_seed=cfg['reserved_final_seed'],
        final_cohort_opened=False,original_128_training_prefix_verified=bool(np.array_equal(trainp[:128],C.family(cfg['train_seed'],min(128,len(trainp)))))))
    params,rotation,bank,target,norm2,perp,info=T.train_bank(data,cfg,out,valid)
    record['bank']=info;save()
    qb,rb=np.linalg.qr(bank,mode='reduced');v=valid.reshape(-1,len(bank));vt=v@qb;vn=np.sum(v*v,axis=1)
    validation=dict(matrix=rb,target=vt,norm2=vn,perpendicular2=np.maximum(vn-np.sum(vt*vt,axis=1),0.),shape=valid.shape[:2])
    np.savez_compressed(out/'coordinates.npz',bank=bank,training_target=target,training_norm2=norm2,
        training_perpendicular2=perp,validation_target=vt,validation_norm2=vn,
        validation_perpendicular2=validation['perpendicular2'],validation_matrix=rb)
    np.savez_compressed(out/'reference.npz',validation=valid)
    models=[]
    for k in cfg['latent_dimensions']:
        record['status']=f'training_head_K{k}';save()
        model=T.train_head(target,norm2,perp,k,cfg,out,validation);models.append(model)
        record['heads'].append(model['info'])
        prediction,stats,latents=R.best_found_fields(model,bank,valid,cfg,return_latents=True)
        path=f'head_K{k}_development.npz'
        np.savez_compressed(out/path,prediction=prediction,stats=stats,latents=latents)
        record['candidates'].append(dict(name=f'random_K{k}',info=model['info'],newly_trained=True,
            checkpoint_relative_path=f'head_K{k}.pkl',fits=[dict(starts=cfg['representation_fit_starts'],
                metrics=[C.metrics(a,b) for a,b in zip(prediction,valid)],
                nonstationary_fits=int(np.count_nonzero(stats[...,2]!=1)),field_file=path)]));save()
    del models
    from operators import training as OT, heat_adapter as OH, models3d as OM
    scale=float(np.sqrt(np.mean(data[:,0]**2)))
    train_x,train_y=OH.arrays(data,scale);valid_x,valid_y=OH.arrays(valid,scale)
    for setting in cfg['operators']:
        name=setting['name'];spec=setting['model'];directory=out/'operators'/name
        record['status']=f'training_{name}';save();initial=None;initial_source=None
        if setting.get('pretraining'):
            from operators.pretrained_deeponet import pretrain
            initial,initial_source=pretrain(train_x,train_y,spec,setting['training'],setting['pretraining'],
                directory/'pretraining',C.dump,C.checkpoint)
        op,info=OT.train(train_x,train_y,valid_x,valid_y,spec,setting['training'],directory,C.dump,C.checkpoint,
            initial_params=initial,initial_source=initial_source)
        info.update(name=name,physical_scale=scale,training_parameters_sha256=C.sha(trainp),validation_parameters_sha256=C.sha(valp),
            input='supplied initial field divided by training RMS plus three coordinate channels; no generator descriptors',
            output='five evolved fields in physical units; exact supplied initial field prepended at inference')
        C.checkpoint(directory/'adapter.pkl',dict(params=op,spec=spec,physical_scale=scale,info=info))
        @jax.jit
        def predict(p,x):return OM.apply_model(p,x,spec)
        for label,x,y in [('training',train_x,train_y),('development',valid_x,valid_y)]:
            errors=[];predictions=[]
            for i in range(len(x)):
                pred=np.asarray(predict(op,x[i:i+1]))[0]
                errors.append(np.linalg.norm((pred-y[i]).reshape(-1,pred.shape[-1]),axis=0)/np.linalg.norm(y[i].reshape(-1,pred.shape[-1]),axis=0))
                if label=='development':predictions.append(pred)
            info[label+'_error_by_case_time']=np.asarray(errors).tolist()
            if predictions:np.savez_compressed(directory/'development.npz',prediction=np.asarray(predictions),target=valid_y)
        np.testing.assert_allclose(max(map(max,info['development_error_by_case_time'])),info['best_validation_worst'],rtol=1e-10,atol=1e-12)
        C.dump(directory/'diagnostics.json',info);record['operators'].append(info);save()
    record['status']='complete';record['complete']=True;save()
    paths=['bank.pkl']+[f'head_K{k}.pkl' for k in cfg['latent_dimensions']]+[f'operators/{s["name"]}/adapter.pkl' for s in cfg['operators']]
    C.dump(out/'FROZEN.json',dict(source_commit=metadata['source_commit'],job_id=metadata['job_id'],
        role=cfg['seed_role'],config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        training_cohort_sha256=C.sha(trainp),development_cohort_sha256=C.sha(valp),
        checkpoints={p:sha(out/p) for p in paths},final_cohort_opened=False))
    return dict(directory=out.name,role=cfg['seed_role'],config=cfg,bank=record['bank'],heads=record['heads'],
        operators=record['operators'],frozen_manifest_sha256=sha(out/'FROZEN.json'))


def run(cfg,out):
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    print('jax_backend=gpu x64=True precision=highest',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    metadata=dict(source_commit=os.environ.get('SOURCE_COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),
        gpu=jax.devices()[0].device_kind,backend='gpu',x64=True,matmul_precision='highest')
    record=dict(schema='heat3d-coverage-training-v1',config=cfg,**metadata,complete=False,
        final_cohort_opened=False,seeds=[],status='verification')
    save=lambda:C.dump(out/'result.json',record);save()
    record['verification']=dict(heat=C.verify(cfg),rom=R.verify());save()
    trainp=C.family(cfg['train_seed'],cfg['train_count']);valp=C.family(cfg['validation_seed'],cfg['validation_count'])
    assert np.array_equal(trainp[:128],C.family(cfg['train_seed'],min(128,len(trainp))))
    assert not any(np.array_equal(a,b) for a in trainp for b in valp)
    data=C.dataset(cfg['train_intervals'],trainp,cfg);valid=C.dataset(cfg['train_intervals'],valp,cfg)
    for setting in cfg['seeds']:
        scfg=copy.deepcopy(cfg);scfg.pop('seeds');scfg.update(setting['overrides']);scfg['seed_role']=setting['role']
        for operator in scfg['operators']:operator['training']['seed']+=setting['operator_seed_offset']
        if setting.get('nmrom_only'):scfg['operators']=[]
        count=scfg['train_count'];record['status']='training_'+setting['name'];save()
        result=train_seed(scfg,out/setting['name'],trainp[:count],valp,data[:count],valid,metadata)
        record['seeds'].append(result);save();jax.clear_caches()
    record.update(complete=True,status='complete');save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),Path(a.out))
