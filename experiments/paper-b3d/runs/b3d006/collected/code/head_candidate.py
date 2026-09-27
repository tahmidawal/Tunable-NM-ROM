"""Prospective wider K64 head on the unchanged, checksum-frozen learned bank.

Training sees only the original 512 trajectories at eight times. Held-out fields
are not used by this optimizer; best-found validation fits are in the panel.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax
import core as c
from run import dump
from train import checkpoint


def train_candidate(cfg, training, out):
    training=Path(training);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tc=cfg['head_candidate'];k=tc['latent_dimension'];n=cfg['nodes']
    raw=(training/'checkpoint.pkl').read_bytes();old=pickle.loads(raw)
    params=jax.tree_util.tree_map(jnp.asarray,old['params']);r=old['cfg']['r']
    coords=c.b3.grid_coords_3d(n);idx=c.b3.interior_indices_3d(n);x=jnp.asarray(coords[idx])
    bank=np.concatenate([np.asarray(c.b3.features(params,x[s:s+2048])) for s in range(0,len(x),2048)])
    q,rb=np.linalg.qr(bank,mode='reduced');assert np.linalg.norm(q.T@q-np.eye(r))<1e-9
    data=np.load(training/'training_fields.npy',mmap_mode='r')
    target=np.asarray(jnp.asarray(data)@jnp.asarray(q));norm2=np.sum(data*data,axis=1)
    # Explicit full-space residual avoids cancellation in the small floor.
    floor=np.sum((data-target@q.T)**2,axis=1)
    _,_,vt=np.linalg.svd(target,full_matrices=False);scale=float(np.sqrt(np.mean(target*target)))
    normalized=target/scale;z=jnp.asarray(normalized@vt[:k].T)
    fresh=c.b3.init_separable_3d(jax.random.PRNGKey(tc['seed']),k,r,n_ff=cfg['training']['fourier_features'],
        ff_scale=cfg['training']['fourier_scale'],g_hidden=cfg['training']['bank_width'],g_layers=2,
        h_hidden=tc['head_width'],h_layers=2,out_scale=float(old['params']['out_scale']))
    hp={name:fresh[name] for name in ['h','h_lin']};hp['h_lin']=jnp.asarray(vt[:k])
    hp['h'][-1]=(hp['h'][-1][0]*.01,hp['h'][-1][1]*.01)
    tt=jnp.asarray(normalized);nn=jnp.asarray(norm2/scale**2);ff=jnp.asarray(floor/scale**2)
    opt=optax.adam(optax.cosine_decay_schedule(tc['head_learning_rate'],tc['head_steps'],alpha=.03))
    pz=(hp,z);state=opt.init(pz);key=jax.random.PRNGKey(tc['seed']+1)
    def loss(pz,targets,norm,perp,key):
        hp,z=pz;rows=jax.random.randint(key,(min(tc['head_batch'],len(targets)),),0,len(targets))
        residual=c.b3.head(hp,z[rows])-targets[rows]
        errors=(jnp.sum(residual**2,axis=1)+perp[rows])/norm[rows]
        return jnp.mean(errors)+.1*jnp.mean(errors**2)
    @jax.jit
    def step(pz,state,key,targets,norm,perp):
        value,grad=jax.value_and_grad(loss)(pz,targets,norm,perp,key)
        update,state=opt.update(grad,state,pz)
        return optax.apply_updates(pz,update),state,value
    curve=[];begin=time.perf_counter()
    for it in range(tc['head_steps']):
        key,sub=jax.random.split(key);pz,state,value=step(pz,state,sub,tt,nn,ff)
        if it==0 or (it+1)%100==0 or it+1==tc['head_steps']:
            objective=float(value);assert np.isfinite(objective)
            curve.append(dict(step=it+1,objective=objective,seconds=time.perf_counter()-begin))
        if (it+1)%tc['checkpoint_every']==0 or it+1==tc['head_steps']:
            checkpoint(out/'head_partial.pkl',dict(pz=pz,state=state,key=key,step=it+1,cfg=cfg,coeff_scale=scale))
            dump(out/'head_curve.json',curve);print('HEAD CANDIDATE',curve[-1],flush=True)
    hp,z=pz;pred=np.asarray(c.b3.head(hp,z))*scale
    error=np.sqrt((np.sum((pred-target)**2,axis=1)+floor)/norm2)
    info=dict(seconds=time.perf_counter()-begin,mean=float(error.mean()),median=float(np.median(error)),worst=float(error.max()),
              steps=tc['head_steps'],latent_dimension=k,head_width=tc['head_width'],
              source_bank_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
              normalization='current training snapshot norm',selection='prospectively fixed training step budget; no validation gradients',converged_claim=False)
    dump(out/'head_info.json',info)
    transform=jnp.asarray(np.linalg.inv(rb).T*scale);w,b=hp['h'][-1]
    hp['h'][-1]=(w@transform,b@transform);hp['h_lin']=hp['h_lin']@transform;params.update(hp)
    reconstructed=np.asarray(c.b3.head(params,z))@rb.T
    parity=float(np.linalg.norm(reconstructed-pred)/np.linalg.norm(pred));assert parity<1e-10
    candidate=copy.deepcopy(cfg);candidate.pop('operators',None);candidate.pop('reuse_attempt',None);candidate.pop('head_candidate',None)
    candidate['attempt']=cfg['attempt']+'_head64';candidate['refinement_cases']=[]
    candidate['training'].update(tc);candidate['pod_ranks']=[64,128,192,256]
    candidate['candidate_parent_checkpoint_sha256']=hashlib.sha256(raw).hexdigest()
    dump(out/'config.json',candidate)
    checkpoint(out/'checkpoint.pkl',dict(params=params,Z_tr=z,cfg=dict(k=k,r=r,training=candidate),
               training_rows=old['training_rows'],training_steps=old['training_steps'],bank_info=old['bank_info'],head_info=info,
               source='K64 wider-head candidate on unchanged frozen B003 learned spatial bank',coefficient_conversion_relative=parity))
    np.savez_compressed(out/'coordinates.npz',target=target,norm2=norm2,floor=floor,codes=np.asarray(z))
    print('HEAD CANDIDATE COMPLETE',info,flush=True)
    return info


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--training',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();assert jax.default_backend()=='gpu'
    train_candidate(json.loads(Path(args.config).read_text()),args.training,args.out)


if __name__=='__main__':main()
