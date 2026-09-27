"""Saved train/development errors and a learned DeepONet trunk-span diagnostic.

The unrestricted trunk fit is a representation diagnostic, never an operator
prediction. It sees each target solely to measure a best linear approximation.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import core as c
from run import dump
from operator_panel import input_fields
from operators import models3d as M


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--training',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    training=Path(a.training);out=Path(a.out);meta=json.loads((out/'operator_metadata.json').read_text())
    assert jax.default_backend()=='gpu'
    ni=cfg['nodes']-2;nt=len(cfg['train_steps'])
    truth={
        'training':np.load(training/'training_fields.npy',mmap_mode='r').reshape(-1,nt,ni,ni,ni),
        'development':np.load(training/'validation_observed.npz')['fields'].reshape(-1,nt,ni,ni,ni)}
    nu=np.load(training/'physical_inputs.npz');viscosities={'training':nu['training_nu'],'development':nu['validation_nu']}
    xyz=c.b3.grid_coords_3d(cfg['nodes'])[c.b3.interior_indices_3d(cfg['nodes'])].reshape(ni,ni,ni,3)*2-1
    report=dict(normalization='supplied initial-field norm; seven learned evolved training times',final_cohort_unopened=True,models=[])
    for model in cfg['operators']:
        ck=pickle.loads((out/model['name']/'best.pkl').read_bytes());params=jax.tree_util.tree_map(jnp.asarray,ck['params'])
        spec=model['spec'];apply=jax.jit(lambda p,x:M.apply_model(p,x,spec))
        row=dict(name=model['name'],checkpoint_step=ck['step'],splits={})
        Q=None
        if spec['kind']=='deeponet3d':
            # Independently reconstruct the coordinate trunk in NumPy.
            t=xyz.reshape(-1,3);features=[t]
            for frequency in spec.get('trunk_frequencies',[1.,2.,4.]):
                features.extend([np.sin(np.pi*frequency*t),np.cos(np.pi*frequency*t)])
            t=np.concatenate(features,axis=-1)
            for layer in ck['params']['trunk'][:-1]:t=np.tanh(t@layer['w']+layer['b'])
            layer=ck['params']['trunk'][-1];t=t@layer['w']+layer['b']
            augmented=jnp.asarray(np.column_stack((t,np.ones(len(t)))))
            q,rr=jnp.linalg.qr(augmented,mode='reduced');u,s,_=jnp.linalg.svd(rr,full_matrices=False)
            singular=np.asarray(s);rank=int(np.sum(singular>singular[0]*np.finfo(float).eps*max(augmented.shape)))
            Q=q@u[:,:rank]
            row['trunk']=dict(rank=rank,requested_rank=spec['rank'],singular_values=singular.tolist(),
                scope='unrestricted learned spatial trunk plus constant; per-target least squares; not a network prediction')
        for split,fields in truth.items():
            errors=[];floors=[]
            for case in range(len(fields)):
                u=jnp.asarray(fields[case,0]);x=input_fields(u,float(viscosities[split][case]),jnp.asarray(xyz),
                    meta['scale'],meta['nu_center'],meta['nu_scale'])
                prediction=np.asarray(apply(params,x[None]))[0]*meta['scale']
                target=np.moveaxis(fields[case,1:],0,-1);den=float(np.sum(fields[case,0]**2))
                errors.append(np.sqrt(np.sum((prediction-target)**2,axis=(0,1,2))/den))
                if Q is not None:
                    y=jnp.asarray(target.reshape(-1,nt-1));residual=y-Q@(Q.T@y)
                    floors.append(np.sqrt(np.asarray(jnp.sum(residual**2,axis=0))/den))
            e=np.asarray(errors)
            values=dict(error_by_case_time=e.tolist(),mean=float(e.mean()),median_case_worst=float(np.median(e.max(axis=1))),worst=float(e.max()))
            if floors:
                f=np.asarray(floors);values['trunk_projection']=dict(error_by_case_time=f.tolist(),mean=float(f.mean()),
                    median_case_worst=float(np.median(f.max(axis=1))),worst=float(f.max()))
            row['splits'][split]=values
        report['models'].append(row);dump(out/'operator-diagnostic.json',report)
        print('OPERATOR DIAGNOSTIC',model['name'],{k:(v['worst'],v.get('trunk_projection',{}).get('worst')) for k,v in row['splits'].items()},flush=True)
        del params,apply,Q;jax.clear_caches()


if __name__=='__main__':main()
