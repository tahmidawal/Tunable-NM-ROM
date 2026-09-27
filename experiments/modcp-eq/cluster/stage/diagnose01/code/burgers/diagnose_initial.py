"""Accuracy-only validation diagnosis; does not change any query or checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from common.decoders import DecoderConfig,decode_grid,prepare_points,decode_cached
from common.lm import make_lm


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fields',required=True);parser.add_argument('--checkpoints',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(args.fields);checkpoints=Path(args.checkpoints)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    report=dict(accuracy_only=True,validation_only=True,checkpoints_modified=False,query_algorithms_modified=False,
        interpretation='offline oracle reconstruction upper bounds, excluded from selection and online comparisons; no speed claim',
        provenance=dict(commit=os.environ.get('COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),gpu=jax.devices()[0].device_kind,
                        backend=jax.default_backend(),x64=jax.config.jax_enable_x64,matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION']),rows=[])
    for arm in ('cp','modcp','film'):
        path=checkpoints/f'{arm}.pkl';before=sha(path)
        with path.open('rb') as f:ck=pickle.load(f)
        cfg=DecoderConfig(**ck['config']);params=jax.tree_util.tree_map(jnp.asarray,ck['params']);Z=jnp.asarray(ck['Z']);L=cfg.intervals
        ids=np.unique(np.rint(np.linspace(1,L-1,32)).astype(int));ii,jj=np.meshgrid(ids,ids,indexing='ij');ii=ii.ravel();jj=jj.ravel()
        xy=jnp.asarray(np.stack((ii,jj),axis=1)/L);cache=prepare_points(params,xy,cfg)
        sampled=jax.jit(lambda p,c,z:jax.vmap(lambda q:decode_cached(p,q,c,cfg)[:,0])(z))
        all_predictions=np.asarray(sampled(params,cache,Z))
        def residual(z,target,scale,p,c):return (decode_cached(p,z,c,cfg)[:,0]-target)/scale/jnp.sqrt(target.size)
        fit=jax.jit(make_lm(residual,cap=120));decode=jax.jit(lambda p,z:decode_grid(p,z,L,cfg)[...,0])
        for case in ([3,9] if arm=='film' else [3]):
            artifact=root/f'validation_{arm}_cap10_dt0.00125_q4_tol0.0001_L256_case{case}.npz'
            data=np.load(artifact);u=data['u'];truth=data['truth_u'];same=data['same_grid_u']
            scale0=max(float(np.linalg.norm(truth[0])),1e-30)
            row=dict(method=arm,case=case,intervals=L,input_artifact_sha256=sha(artifact),checkpoint_sha256=before,
                checkpoint_dtype=sorted(set(str(a.dtype) for a in jax.tree_util.tree_leaves(ck['params']))),
                recorded_query_fine_error=(np.linalg.norm((u-truth).reshape(6,-1),axis=1)/scale0).tolist(),
                independently_same_grid_error=(np.linalg.norm((u-same).reshape(6,-1),axis=1)/max(np.linalg.norm(same[0]),1e-30)).tolist(),
                same_grid_vs_finer_reference=(np.linalg.norm((same-truth).reshape(6,-1),axis=1)/scale0).tolist(),
                query_evolution_norm=float(np.linalg.norm(u[-1]-u[0])/scale0),snapshots=[])
            for frame,target_field in enumerate(truth):
                target=target_field[ii,jj];scale=max(float(np.sqrt(np.mean(target**2))),1e-12)
                distances=np.sum((all_predictions-target[None])**2,axis=1)
                initial_codes=np.arange(0,len(Z),6);online_index=int(initial_codes[np.argmin(distances[initial_codes])])
                indices=[online_index]+[int(x) for x in np.argsort(distances,kind='stable')[:4] if int(x)!=online_index]
                fits=[]
                for index in indices:
                    zz,it,reason,stat,rn=fit(Z[index],(jnp.asarray(target),jnp.asarray(scale),params,cache),jnp.asarray(1e-6))
                    field=np.asarray(decode(params,zz));error=float(np.linalg.norm(field-target_field)/scale0)
                    fits.append(dict(training_code_index=index,start_role='supplied_field_nearest_training_initial' if index==online_index else 'oracle_best_training_snapshot_start',
                                     physical_error=error,iterations=int(it),reason=int(reason),stationarity=float(stat),sampled_residual=float(rn)))
                row['snapshots'].append(dict(frame=frame,time=.05*frame,fits=fits,best_oracle_error=min(x['physical_error'] for x in fits)))
            if arm=='cp':
                W=np.asarray(ck['params']['factors']);bank=np.einsum('ri,rj->ijr',W[0,0],W[1,0]);bank[[0,-1],:,:]=0.;bank[:,[0,-1],:]=0.
                mask=np.ones((L+1,L+1));mask[[0,-1],:]=0.;mask[:,[0,-1]]=0.
                G=bank.reshape(-1,cfg.rank);offset=float(ck['params']['bias'][0])*mask.ravel()
                beta,_,rank,sv=np.linalg.lstsq(G,truth.reshape(6,-1).T-offset[:,None],rcond=1e-12)
                projection=(G@beta+offset[:,None]).T.reshape(truth.shape)
                row['unconstrained_CP_spatial_span']=dict(rank=int(rank),singular_values=sv.tolist(),
                    errors=(np.linalg.norm((projection-truth).reshape(6,-1),axis=1)/scale0).tolist(),
                    interpretation='unconstrained R-coefficient full-grid least-squares floor; a lower bound on this fixed-CP manifold error, not an online ROM result')
            assert sha(path)==before
            report['rows'].append(row)
            Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(f'DIAG {arm} case={case} initial={row["recorded_query_fine_error"][0]} best_oracle={row["snapshots"][0]["best_oracle_error"]}',flush=True)
    print('ALL-DONE accuracy-only validation diagnosis',flush=True)


if __name__=='__main__':main()
