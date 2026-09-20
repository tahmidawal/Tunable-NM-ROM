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
    saved,old_models,_,origin,reuse=F.load(cfg['frozen_input_directory'],cfg,trainp,valp,out)
    record=dict(schema='heat3d-head-pca-diagnostic-v1',config=cfg,source_commit=os.environ.get('SOURCE_COMMIT'),
        job_id=os.environ.get('SLURM_JOB_ID'),gpu=jax.devices()[0].device_kind,backend='gpu',x64=True,
        matmul_precision='highest',complete=False,final_cohort_opened=False,frozen_checkpoints=reuse,candidates=[])
    save=lambda:C.dump(out/'result.json',record);save()
    bank=C.bank_at(saved['params'],cfg['train_intervals'],cfg['field_chunk'])@saved['rotation']
    data=C.dataset(cfg['train_intervals'],trainp,cfg);valid=C.dataset(cfg['train_intervals'],valp,cfg)
    u=data.reshape(-1,len(bank));target=u@bank;norm2=np.sum(u*u,axis=1)
    perp=np.maximum(norm2-np.sum(target*target,axis=1),0.)
    qb,rb=np.linalg.qr(bank,mode='reduced');v=valid.reshape(-1,len(bank));vt=v@qb;vn=np.sum(v*v,axis=1)
    validation=dict(matrix=rb,target=vt,norm2=vn,perpendicular2=np.maximum(vn-np.sum(vt*vt,axis=1),0.),shape=valid.shape[:2])
    np.savez_compressed(out/'coordinates.npz',bank=bank,training_target=target,training_norm2=norm2,training_perpendicular2=perp,
        validation_target=vt,validation_norm2=vn,validation_perpendicular2=validation['perpendicular2'],validation_matrix=rb)
    np.savez_compressed(out/'reference.npz',validation=valid)
    C.dump(out/'cohorts.json',dict(train_sha256=C.sha(trainp),validation_sha256=C.sha(valp),training_parameters=trainp.tolist(),validation_parameters=valp.tolist(),reserved_final_seed=cfg['reserved_final_seed'],final_cohort_opened=False))
    for k in cfg['diagnostic_latent_dimensions']:
        directory=out/f'pca_K{k}';directory.mkdir()
        record['status']=f'training_pca_K{k}';save()
        model=T.train_head(target,norm2,perp,k,cfg,directory,validation)
        record['candidates'].append(dict(name=f'pca_K{k}',info=model['info'],fits=[]));save()
        models=[(f'pca_K{k}',model)]
        for start_count in cfg['diagnostic_fit_starts']:
            record['status']=f'evaluating_pca_K{k}_starts{start_count}';save()
            prediction,stats=R.best_found_fields(model,bank,valid,{**cfg,'representation_fit_starts':start_count})
            metrics=[C.metrics(a,b) for a,b in zip(prediction,valid)]
            np.savez_compressed(directory/f'validation_starts{start_count}.npz',prediction=prediction,stats=stats)
            row=dict(starts=start_count,metrics=metrics,nonstationary_fits=int(np.count_nonzero(stats[...,2]!=1)))
            record['candidates'][-1]['fits'].append(row);save()
    for model in old_models:
        k=model['info']['k'];record['candidates'].append(dict(name=f'old_random_K{k}',info=model['info'],fits=[]))
        for start_count in cfg['diagnostic_fit_starts']:
            prediction,stats=R.best_found_fields(model,bank,valid,{**cfg,'representation_fit_starts':start_count})
            metrics=[C.metrics(a,b) for a,b in zip(prediction,valid)]
            np.savez_compressed(out/f'old_K{k}_starts{start_count}.npz',prediction=prediction,stats=stats)
            record['candidates'][-1]['fits'].append(dict(starts=start_count,metrics=metrics,nonstationary_fits=int(np.count_nonzero(stats[...,2]!=1))));save()
    record.update(status='complete',complete=True);save()
    print('HEAD_PCA_COMPLETE',[(c['name'],[(f['starts'],max(m['current_evolved'] for m in f['metrics'])) for f in c['fits']]) for c in record['candidates']],flush=True)

if __name__=='__main__':main()
