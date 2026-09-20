"""Train-only PCA head initialization on one frozen learned Heat3D bank."""
import argparse, json, os, pickle, time
from pathlib import Path
import numpy as np
import jax
import common as C
import train as T
import rom as R


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    print('jax_backend=gpu x64=True precision=highest',flush=True)
    import frozen as F
    trainp=C.family(cfg['train_seed'],cfg['train_count']);valp=C.family(cfg['validation_seed'],cfg['validation_count'])
    saved,old_models,operator_models,origin,reuse=F.load(cfg['frozen_input_directory'],cfg,trainp,valp,out)
    record=dict(schema='heat3d-head-pca-diagnostic-v1',config=cfg,source_commit=os.environ.get('SOURCE_COMMIT'),
        job_id=os.environ.get('SLURM_JOB_ID'),gpu=jax.devices()[0].device_kind,backend='gpu',x64=True,
        matmul_precision='highest',complete=False,final_cohort_opened=False,frozen_checkpoints=reuse,candidates=[])
    save=lambda:C.dump(out/'result.json',record);save()
    bank=C.bank_at(saved['params'],cfg['train_intervals'],cfg['field_chunk'])@saved['rotation']
    data=C.dataset(cfg['train_intervals'],trainp,cfg);valid=C.dataset(cfg['train_intervals'],valp,cfg)
    from operators import models3d as OM, heat_adapter as OH, training as OT
    scale=float(np.sqrt(np.mean(data[:,0]**2)));train_x,train_y=OH.arrays(data,scale);valid_x,valid_y=OH.arrays(valid,scale)
    record['operator_diagnostics']=[];record['operators']=[]
    def operator_diagnostic(name,spec,params,stage):
        @jax.jit
        def predict(p,x):return OM.apply_model(p,x,spec)
        row=dict(name=name,stage=stage)
        for label,x,y in [('training',train_x,train_y),('validation',valid_x,valid_y)]:
            es=[]
            for i in range(len(x)):
                pred=np.asarray(predict(params,x[i:i+1]))[0]
                es.append(np.sqrt(np.sum((pred-y[i])**2,axis=(0,1,2))/np.sum(y[i]**2,axis=(0,1,2))))
            row[label+'_current_error_by_case_time']=np.asarray(es).tolist()
        if spec['kind']=='deeponet3d':
            from operators import extra_models3d as E
            features=E._trunk_features(OH.coordinates(cfg['train_intervals']),spec)
            for layer in params['trunk'][:-1]:features=jax.numpy.tanh(E._linear(layer,features))
            trunk=np.asarray(E._linear(params['trunk'][-1],features)).reshape(-1,spec['rank'])
            q,r=np.linalg.qr(trunk,mode='reduced');u,singular,_=np.linalg.svd(r,full_matrices=False)
            row['trunk_singular_values']=singular.tolist();row['trunk_rank']=int(np.count_nonzero(singular>singular[0]*1e-12))
            row['trunk_rank_relative_threshold']=1e-12;q=q@u[:,:row['trunk_rank']]
            for label,y in [('training',train_y),('validation',valid_y)]:
                shape=y.shape;flat=np.moveaxis(y,-1,1).reshape(-1,len(q));bias=np.tile(np.asarray(params['bias']),len(y))
                centered=flat-bias[:,None];norm=np.sum(flat*flat,axis=1)
                projected=(centered@q)@q.T+bias[:,None]
                error=np.sqrt(np.sum((projected-flat)**2,axis=1)/norm).reshape(len(y),shape[-1])
                row[label+'_trunk_projection_current_error_by_case_time']=error.tolist()
        record['operator_diagnostics'].append(row);save()
        print('OPERATOR_DIAGNOSTIC',name,stage,{k:float(np.max(v)) for k,v in row.items() if k.endswith('by_case_time')},flush=True)
    for name,spec,params,physical_scale,info in operator_models:
        assert abs(physical_scale-scale)<1e-12
        if spec['kind'] in ('deeponet3d','transolver3d'):operator_diagnostic(name,spec,params,'before_continuation')
    u=data.reshape(-1,len(bank));target=u@bank;norm2=np.sum(u*u,axis=1)
    perp=np.maximum(norm2-np.sum(target*target,axis=1),0.)
    qb,rb=np.linalg.qr(bank,mode='reduced');v=valid.reshape(-1,len(bank));vt=v@qb;vn=np.sum(v*v,axis=1)
    validation=dict(matrix=rb,target=vt,norm2=vn,perpendicular2=np.maximum(vn-np.sum(vt*vt,axis=1),0.),shape=valid.shape[:2])
    np.savez_compressed(out/'coordinates.npz',bank=bank,training_target=target,training_norm2=norm2,training_perpendicular2=perp,
        validation_target=vt,validation_norm2=vn,validation_perpendicular2=validation['perpendicular2'],validation_matrix=rb)
    np.savez_compressed(out/'reference.npz',validation=valid)
    C.dump(out/'cohorts.json',dict(train_sha256=C.sha(trainp),validation_sha256=C.sha(valp),training_parameters=trainp.tolist(),validation_parameters=valp.tolist(),reserved_final_seed=cfg['reserved_final_seed'],final_cohort_opened=False))
    candidates=[(f'pca_K{k}',k,'weighted_pca_linear_skip') for k in cfg['diagnostic_latent_dimensions']]
    candidates += [(f'matched_random_K{k}',k,'random_joint_codes') for k in cfg.get('diagnostic_random_control_dimensions',[])]
    for name,k,initialization in candidates:
        directory=out/name;directory.mkdir()
        record['status']=f'training_{name}';save()
        model=T.train_head(target,norm2,perp,k,{**cfg,'head_initialization':initialization},directory,validation)
        record['candidates'].append(dict(name=name,info=model['info'],fits=[],newly_trained=True,
            checkpoint_relative_path=f'{name}/head_K{k}.pkl'));save()
        for start_count in cfg['diagnostic_fit_starts']:
            record['status']=f'evaluating_{name}_starts{start_count}';save()
            prediction,stats,latents=R.best_found_fields(model,bank,valid,{**cfg,'representation_fit_starts':start_count},return_latents=True)
            metrics=[C.metrics(a,b) for a,b in zip(prediction,valid)]
            np.savez_compressed(directory/f'validation_starts{start_count}.npz',prediction=prediction,stats=stats,latents=latents)
            row=dict(starts=start_count,metrics=metrics,nonstationary_fits=int(np.count_nonzero(stats[...,2]!=1)),field_file=f'{name}/validation_starts{start_count}.npz')
            record['candidates'][-1]['fits'].append(row);save()
    for model in old_models:
        k=model['info']['k'];record['candidates'].append(dict(name=f'old_random_K{k}',info=model['info'],fits=[],newly_trained=False,
            checkpoint_relative_path=f'head_K{k}.pkl'))
        for start_count in cfg['diagnostic_fit_starts']:
            prediction,stats,latents=R.best_found_fields(model,bank,valid,{**cfg,'representation_fit_starts':start_count},return_latents=True)
            metrics=[C.metrics(a,b) for a,b in zip(prediction,valid)]
            np.savez_compressed(out/f'old_K{k}_starts{start_count}.npz',prediction=prediction,stats=stats,latents=latents)
            record['candidates'][-1]['fits'].append(dict(starts=start_count,metrics=metrics,nonstationary_fits=int(np.count_nonzero(stats[...,2]!=1)),field_file=f'old_K{k}_starts{start_count}.npz'));save()
    for setting in cfg.get('operator_continuations',[]):
        name=setting['name'];old=next(item for item in operator_models if item[0]==name)
        _,spec,params,physical_scale,old_info=old;record['status']=f'continuing_{name}';save()
        source=dict(source_attempt=origin['source_attempt'],source_commit=origin['source_commit'],
            checkpoint_sha256=origin['files'][f'operators/{name}/adapter.pkl'],original_training=old_info)
        op,info=OT.train(train_x,train_y,valid_x,valid_y,spec,setting['training'],out/'operators_continued'/name,C.dump,C.checkpoint,
            initial_params=params,initial_source=source)
        info.update(name=name,physical_scale=scale,training_parameters_sha256=C.sha(trainp),validation_parameters_sha256=C.sha(valp),
            input=old_info['input'],output=old_info['output'])
        C.checkpoint(out/'operators_continued'/name/'adapter.pkl',dict(params=op,spec=spec,physical_scale=scale,info=info))
        record['operators'].append(info);save();operator_diagnostic(name,spec,op,'after_continuation')
    record.update(status='complete',complete=True);save()
    print('HEAD_PCA_COMPLETE',[(c['name'],[(f['starts'],max(m['current_evolved'] for m in f['metrics'])) for f in c['fits']]) for c in record['candidates']],flush=True)

if __name__=='__main__':main()
