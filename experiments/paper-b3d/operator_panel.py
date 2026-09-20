"""Shared-data B3D operators, with explicit sparse-time interpolation control."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import core as c
from run import burn, dump, host
from train import checkpoint
from operators import models3d as M
from operators.training import train


def interpolation_matrix(observed,steps):
    observed=np.asarray(observed);eye=np.eye(len(observed))
    return np.stack([np.interp(np.arange(steps+1),observed,eye[:,j]) for j in range(len(observed))],axis=1)


def input_fields(u0,nu,xyz,scale,nu_center,nu_scale):
    return jnp.concatenate((u0[...,None]/scale,xyz,
         jnp.broadcast_to((nu-nu_center)/nu_scale,u0.shape)[...,None]),axis=-1)


def make_query(spec,n,steps,observed):
    weights=jnp.asarray(interpolation_matrix(observed,steps))
    def query(p,u0,nu,data):
        field=u0.reshape((n-2,)*3)
        x=input_fields(field,nu,data['xyz'],data['scale'],data['nu_center'],data['nu_scale'])
        evolved=M.apply_model(p,x[None],spec)[0]*data['scale']
        knots=jnp.concatenate((field[...,None],evolved),axis=-1).reshape(-1,len(observed)).T
        return weights@knots,knots
    return jax.jit(query)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--training',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--mode',choices=['train','evaluate'],required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    training=Path(args.training);out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    n=cfg['nodes'];nt=len(cfg['train_steps']);ni=n-2
    if args.mode=='train':
        raw=np.load(training/'training_fields.npy',mmap_mode='r').reshape(cfg['train_trajectories'],nt,ni,ni,ni)
        valid=np.load(training/'validation_observed.npz')['fields'].reshape(len(cfg['validation_rows']),nt,ni,ni,ni)
        nu=np.load(training/'physical_inputs.npz');tnu=nu['training_nu'];vnu=nu['validation_nu']
        scale=float(np.sqrt(np.mean(raw[:,0]**2)));nu_center=float(tnu.mean());nu_scale=float(tnu.std())
        xyz=c.b3.grid_coords_3d(n)[c.b3.interior_indices_3d(n)].reshape(ni,ni,ni,3)*2-1
        def pack(fields,viscosities):
            x=np.stack([np.asarray(input_fields(jnp.asarray(u),float(v),jnp.asarray(xyz),scale,nu_center,nu_scale))
                        for u,v in zip(fields[:,0],viscosities)])
            y=np.moveaxis(fields[:,1:]/scale,1,-1)
            norm=np.sum((fields[:,0]/scale)**2,axis=(1,2,3))
            return x,y,np.repeat(norm[:,None],nt-1,axis=1)
        tx,ty,td=pack(raw,tnu);vx,vy,vd=pack(valid,vnu)
        metadata=dict(training_rows=list(range(cfg['train_trajectories'])),validation_rows=cfg['validation_rows'],
            observed_training_steps=cfg['train_steps'],output_steps=list(range(cfg['steps']+1)),
            output_contract='seven learned evolved knots plus exact supplied initial field, linear time interpolation to all 51 fields',
            scale=scale,nu_center=nu_center,nu_scale=nu_scale,final_cohort_unopened=True,
            normalization='each time error divided by supplied initial-field norm',input='supplied initial interior field, viscosity and fixed coordinates',
            training_fields_sha256=hashlib.sha256((training/'training_fields.npy').read_bytes()).hexdigest(),
            shared_operator_source=json.loads(Path('operators/VENDOR.json').read_text()),models=[])
        dump(out/'operator_metadata.json',metadata)
        for model in cfg['operators']:
            params,info=train(tx,ty,vx,vy,model['spec'],model['training'],out/model['name'],dump,checkpoint,td,vd)
            metadata['models'].append(dict(name=model['name'],**info));dump(out/'operator_metadata.json',metadata)
            del params;jax.clear_caches()
        return
    result=json.loads((out/'result.json').read_text());metadata=json.loads((out/'operator_metadata.json').read_text())
    data=dict(xyz=c.b3.grid_coords_3d(n)[c.b3.interior_indices_3d(n)].reshape(ni,ni,ni,3)*2-1,
              scale=metadata['scale'],nu_center=metadata['nu_center'],nu_scale=metadata['nu_scale'])
    data=jax.tree_util.tree_map(jnp.asarray,data);rng=np.random.default_rng(cfg['timing_seed']+93)
    result['operators']=metadata;result['operator_invocations']=[];result['interpolation_controls']=[]
    weights=interpolation_matrix(cfg['train_steps'],cfg['steps'])
    for case in range(len(cfg['validation_rows'])):
        truth=np.load(out/f'reference_case{case}.npz')['fields'];interpolated=weights@truth[cfg['train_steps']]
        artifact=f'interpolation_control_case{case}.npz'
        np.savez_compressed(out/artifact,fields=interpolated,knots=truth[cfg['train_steps']])
        result['interpolation_controls'].append(dict(case=case,artifact=artifact,**c.metrics(interpolated,truth)))
    for model in cfg['operators']:
        name=model['name'];ck=pickle.loads((out/name/'best.pkl').read_bytes());params=jax.tree_util.tree_map(jnp.asarray,ck['params'])
        query=make_query(model['spec'],n,cfg['steps'],cfg['train_steps'])
        first=np.load(out/'reference_case0.npz');u0=jnp.asarray(first['u0']);nu=float(first['nu'])
        for _ in range(2):jax.block_until_ready(query(params,u0,nu,data))
        for case in rng.permutation(len(cfg['validation_rows'])):
            case=int(case);ref=np.load(out/f'reference_case{case}.npz');u0=jnp.asarray(ref['u0']);nu=float(ref['nu'])
            for rep in range(cfg['repetitions']):
                burn();begin=time.perf_counter();answer=query(params,u0,nu,data);jax.block_until_ready(answer)
                gpu_ms=(time.perf_counter()-begin)*1000;fields,knots=host(answer)
                artifact=f'{name}_case{case}_rep{rep}.npz'
                np.savez_compressed(out/artifact,fields=fields,knots=knots)
                row=dict(method=name,case=case,row=cfg['validation_rows'][case],repetition=rep,
                         gpu_ms=gpu_ms,artifact=artifact,**c.metrics(fields,ref['fields']))
                result['operator_invocations'].append(row);dump(out/'result.json',result)
                print('OPERATOR TIMING',name,case,rep,gpu_ms,row['worst_evolved'],flush=True)
        del query,params;jax.clear_caches()
    result['operator_complete']=True;dump(out/'result.json',result)


if __name__=='__main__':main()
