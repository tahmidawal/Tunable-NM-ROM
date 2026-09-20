"""A train-only POD-informed initialization of the unchanged learned DeepONet.

Data are regenerated on the cluster. Both independent initializations receive
the identical training set and schedule. Expanded development fields are used
only after training; no final seed is accessed here.
"""
import argparse
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import core as c
from run import dump, host, reference_audit
from train import checkpoint
from operator_panel import input_fields, make_query
from operators.training import train
from operators.pretrained_deeponet import pretrain


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=='gpu' and cfg['final_cohort_unopened']
    n=cfg['nodes'];observed=cfg['train_steps'];nt=len(observed);ni=n-2
    tab=c.b3.draw_param_table(cfg['seed'],max(cfg['development_rows'])+1);tab['s_star']=c.b3.peak_on_reference_grid(tab)
    xyz=c.b3.grid_coords_3d(n);idx=c.b3.interior_indices_3d(n);fom=c.b3.make_newton_tol_rollout(n,'fft')
    training=np.lib.format.open_memmap(out/'training_fields.npy',mode='w+',dtype=np.float64,
        shape=(cfg['train_trajectories'],nt,ni,ni,ni))
    valid=[];records=[]
    for row in list(range(cfg['train_trajectories']))+cfg['development_rows']:
        u=c.b3.blob_ic_3d(n,tab,row,xyz)[idx];nu=float(tab['nu'][row]);fields,its,rn=host(fom(jnp.asarray(u),nu,1e-10,1e-11))
        defect=reference_audit(fields,nu,n,cfg['dt']);assert defect<2e-9 and np.isfinite(fields).all()
        records.append(dict(row=row,nu=nu,maximum_relative_residual=defect))
        if row<cfg['train_trajectories']:training[row]=fields[observed].reshape(nt,ni,ni,ni)
        else:
            valid.append(fields[observed].reshape(nt,ni,ni,ni));np.savez_compressed(out/f'reference_row{row}.npz',fields=fields,u0=u,nu=nu)
        if row%64==0:print('DATA',row,flush=True)
    training.flush();valid=np.asarray(valid);dump(out/'data.json',dict(parameter_seed=cfg['seed'],records=records,final_cohort_unopened=True))
    tnu=tab['nu'][:cfg['train_trajectories']];vnu=tab['nu'][cfg['development_rows']]
    scale=float(np.sqrt(np.mean(training[:,0]**2)));center=float(tnu.mean());spread=float(tnu.std())
    coords=xyz[idx].reshape(ni,ni,ni,3)*2-1
    def pack(fields,nu):
        x=np.stack([np.asarray(input_fields(jnp.asarray(u),float(v),jnp.asarray(coords),scale,center,spread)) for u,v in zip(fields[:,0],nu)])
        y=np.moveaxis(fields[:,1:]/scale,1,-1);den=np.sum((fields[:,0]/scale)**2,axis=(1,2,3))
        return x,y,np.repeat(den[:,None],nt-1,axis=1)
    tx,ty,td=pack(training,tnu);vx,vy,vd=pack(valid,vnu)
    selected=[cfg['development_rows'].index(v) for v in cfg['validation_rows']]
    for seed_index,seed in enumerate(cfg['model_seeds']):
        dst=out/f'seed{seed_index}';dst.mkdir();tc=copy.deepcopy(cfg['joint_training']);tc['seed']=seed
        params,info=pretrain(tx,ty,cfg['spec'],tc,cfg['pretraining'],dst/'pretraining',dump,checkpoint,td)
        params,trained=train(tx,ty,vx[selected],vy[selected],cfg['spec'],tc,dst,dump,checkpoint,td,vd[selected],initial_params=params)
        metadata=dict(name='deeponet3d',**trained,pretraining=info,model_seed_index=seed_index,
            scale=scale,nu_center=center,nu_scale=spread,final_cohort_unopened=True)
        dump(dst/'metadata.json',metadata)
        query=make_query(cfg['spec'],n,cfg['steps'],observed)
        data=jax.tree_util.tree_map(jnp.asarray,dict(xyz=coords,scale=scale,nu_center=center,nu_scale=spread))
        rows=[]
        for row in cfg['development_rows']:
            ref=np.load(out/f'reference_row{row}.npz');fields,knots=host(query(params,jnp.asarray(ref['u0']),float(ref['nu']),data))
            name=f'prediction_row{row}.npz';np.savez_compressed(dst/name,fields=fields,knots=knots)
            rows.append(dict(row=row,artifact=name,**c.metrics(fields,ref['fields'])))
        dump(dst/'development.json',dict(rows=rows,worst_evolved=max(v['worst_evolved'] for v in rows),
            selection_cases=cfg['validation_rows'],expanded_development_cases=cfg['development_rows'],final_cohort_unopened=True))
        print('CONDITIONED_DEEPONET',seed_index,max(v['worst_evolved'] for v in rows),flush=True)
        del params,query,data;jax.clear_caches()
    dump(out/'complete.json',dict(complete=True,final_cohort_unopened=True))


if __name__=='__main__':main()
