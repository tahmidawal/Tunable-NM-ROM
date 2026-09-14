"""Fresh matched-data bank/head screen; validation is never opened in training."""
import argparse
import json
from pathlib import Path
import time
import os
import numpy as np
from dataset import preflight,sha_file,HERE,ROOT


def run(index_path, output, rank, smoke=False):
    core,jax,runtime=preflight(smoke)
    import jax.numpy as jnp
    import staged_training as st
    cfg=json.loads((HERE/'training-protocol.json').read_text())
    if smoke:
        for phase in cfg['phases']: phase['steps']=2
        cfg['common'].update(source_batch=4,point_batch=32,timing_block_updates=1,log_every_updates=1,training_burn_seconds=.001)
    index_path=Path(index_path); index=json.loads(index_path.read_text())
    assert index['pde']=='poisson' and index['split']=='train' and index['complete']
    fields=[]
    for record in index['records']:
        path=index_path.parent/record['path']; assert sha_file(path)==record['sha256']
        with np.load(path,allow_pickle=False) as a:
            assert set(a.files)=={'input','target','parameters','times'}
            fields.append(a['target'][0,0,1:-1,1:-1].ravel())
    U=jnp.asarray(np.stack(fields)); n=index['intervals']
    x=jnp.arange(1,n,dtype=jnp.float64)/n;xx,yy=jnp.meshgrid(x,x,indexing='ij')
    coords=jnp.column_stack((xx.ravel(),yy.ravel()))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    scale=float(jnp.sqrt(jnp.mean(U*U)))
    params=core.sc.init_separable(jax.random.PRNGKey(cfg['initialization_seed']),16,rank,
        n_ff=64,ff_scales=[1.,4.,16.,32.],g_hidden=256,g_layers=2,
        h_hidden=256,h_layers=2,out_scale=scale)
    # Training-field PCA scores seed the trainable latent codes, no descriptors.
    normalized=U/jnp.linalg.norm(U,axis=1)[:,None]
    eig,vectors=jnp.linalg.eigh(normalized@normalized.T)
    codes=np.asarray(vectors[:,-16:])*np.sqrt(len(U))
    if codes.shape[1]<16:
        codes=np.pad(codes,((0,0),(0,16-codes.shape[1])))
    result=dict(complete=False,runtime=runtime,config=cfg,rank=rank,k=16,
        dataset_index_sha256=sha_file(index_path),training_case_ids=[r['case_id'] for r in index['records']],
        phases=[],checkpoints=[],initialization='fresh shared-seed weights; normalized training-field PCA latent scores',
        validation_access=False,comparison_status='common-width matched-data capacity control; r only varies; historical checkpoint separate')
    def persist():
        p=out/'training.tmp';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');p.replace(out/'training.json')
    def checkpoint(tag,p,z):
        path=out/(tag+'.pkl');core.sc.save_pkl(path,p,z,dict(k=16,r=rank,training_index_sha256=sha_file(index_path),training_protocol=cfg,phase=tag,training_case_ids=result['training_case_ids']))
        entry=dict(tag=tag,path=path.name,sha256=sha_file(path));result['checkpoints'].append(entry);persist()
        return path
    checkpoint('initial',params,codes)
    for phase in cfg['phases']:
        options={**cfg['common'],**phase}; name=phase['name'];kind=phase['phase']
        extra={}
        if kind in ('bank','head'):
            G,Q,R,T,C,perp,diagnostic=st.bank_qr(params,U,coords)
            result.setdefault('bank_checks',[]).append(dict(before_phase=name,**diagnostic));persist()
            assert diagnostic['rank_valid']
            if kind=='bank': latent=np.asarray(C)
            else:latent=codes;extra=dict(R=R,T=T,perpendicular=perp)
        else:latent=codes
        params,trained,record=st.train_phase(params,latent,U,coords,options,kind,f'r{rank}_{name}',**extra)
        if kind!='bank':codes=trained
        result['phases'].append(record);persist()
        checkpoint(name,params,codes)
        if not record['finite']:raise RuntimeError('Nonfinite training phase retained; stopping')
    G,Q,R,T,C,perp,diagnostic=st.bank_qr(params,U,coords)
    assert diagnostic['rank_valid']
    H=core.sc.head(params,jnp.asarray(codes))
    residual=(T-H@R.T)/jnp.linalg.norm(U,axis=1)[:,None]
    _,singular,Vt=jnp.linalg.svd(residual,full_matrices=False)
    assert Vt.shape[0]>=32,'Need enough independent training cases for q32 correction fit'
    directions=jax.scipy.linalg.solve_triangular(R,Vt[:32].T,lower=False)
    orth=float(jnp.linalg.norm(directions.T@R.T@R@directions-jnp.eye(32)))
    if orth>1e-9:raise RuntimeError('Invalid correction metric')
    basis=out/'basis.npz'
    np.savez_compressed(basis,coefficient_directions=np.asarray(directions),R=np.asarray(R),training_latents=codes,
                        training_residual_singular_values=np.asarray(singular))
    final=checkpoint('final',params,codes)
    evaluation=json.loads((HERE/'protocol.json').read_text())
    evaluation.update(checkpoint=os.path.relpath(final.resolve(),ROOT),checkpoint_sha256=sha_file(final),basis=os.path.relpath(basis.resolve(),ROOT),basis_sha256=sha_file(basis),
        comparison_status=result['comparison_status'],training_index_sha256=sha_file(index_path),training_rank=rank)
    (out/'evaluation-protocol.json').write_text(json.dumps(evaluation,indent=2)+'\n')
    result.update(complete=True,final_bank_check=diagnostic,correction_metric_orthogonality_error=orth,
        basis_sha256=sha_file(basis),correction_fit_scope='top32 normalized training-field coefficient residual directions; no validation input',
        training_metrics=st.reconstruction_metrics(params,codes,U,coords,float(jnp.mean(U*U))))
    persist();print('MATCHED TRAINING COMPLETE',rank,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--rank',required=True,type=int,choices=[128,256]);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.index,a.output,a.rank,a.smoke)
