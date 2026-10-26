"""Exactly one training-only NNLS path for the predeclared rank512/k32 support."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'jcp-hr-baselines'))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import hrcore as H
import build as B
import nnls as NN
import pipeline as P


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    return h.hexdigest()


def ah(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def conv(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(type(v).__name__)


def run(cfg,artifacts,out):
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    assert cfg['driver']=='fit' and cfg['target_points']==16384 and cfg['rank']==512 and cfg['L']==64
    assert cfg['training_seed']==923701 and cfg['training_count']==1536
    assert cfg['trajectory_seed']==B.ECSW_SEED_TRAJ==20261017 and B.N_ECSW_TRAJ==24
    assert cfg['candidate_seed']==B.ECSW_SEED_CAND==20261018 and cfg['candidate_count']==32768
    assert cfg['nnls_seconds']==5400 and cfg['nnls_iterations']==40000 and cfg['max_drops']==3
    assert not out.exists();out.mkdir(parents=True)
    start=time.perf_counter()
    pin=json.loads((HERE/'inputs/MANIFEST.json').read_text())['files']['Phi.npz']
    assert sha(artifacts/'Phi.npz')==pin
    source=json.loads((out.parent/'PROVENANCE.json').read_text())
    rep=dict(config=cfg,source_commit=source['commit'],source_manifest_sha256=sha(out.parent/'PROVENANCE.json'),
             job_id=os.environ['SLURM_JOB_ID'],backend=jax.default_backend(),precision='f64/highest',
             gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip(),
             basis_sha256=pin,complete=False,attempts=0)
    def save():
        (out/'fit.json.tmp').write_text(json.dumps(rep,indent=2,default=conv)+'\n')
        os.replace(out/'fit.json.tmp',out/'fit.json')
    save()
    z=np.load(artifacts/'Phi.npz');Phi=jax.block_until_ready(jnp.asarray(z['Phi'][:,:512]));del z
    assert 0<P.device_peak()<=120e9,'missing or excessive GPU memory telemetry'
    fam=H.Family(3,'train')
    states,info=B.ecsw_states(3,64,fam,Phi)
    assert fam.n==1536
    rep['training']=dict(seed=923701,count=fam.n,table_sha256=fam.sha256,**info,
                         descriptors={j:fam.descriptor(j).tolist() for j in info['trajectories']},
                         states=[dict(traj=s['traj'],option=s['option'],step=s['k'],nu=s['nu'],
                                      u_sha256=ah(s['u']),un_sha256=ah(s['un']),um_sha256=ah(s['um'])) for s in states])
    G,b,cand,ci=B.ecsw_contrib(states,Phi,3,64,seed=20261018,n_cand=32768)
    assert len(cand)==32768 and len(set(cand.tolist()))==32768
    assert G.shape==(len(states)*512,32768) and b.shape==(len(states)*512,)
    # Persist the exact training matrix and targets. They can reproduce the single fit without new FOM trajectories.
    # This is offline fitting data, never a validation cohort.
    np.save(out/'candidate_nodes.npy',cand)
    np.save(out/'target.npy',np.asarray(b))
    gh=np.asarray(G);np.save(out/'contributions.npy',gh);del gh
    rep['contributions']=dict(shape=list(G.shape),candidate_sha256=sha(out/'candidate_nodes.npy'),
                              target_sha256=sha(out/'target.npy'),matrix_sha256=sha(out/'contributions.npy'),
                              info=ci,setup_seconds=time.perf_counter()-start)
    save()
    if time.perf_counter()-start >= cfg['budget_hours']*3600-cfg['nnls_seconds']:
        raise TimeoutError('insufficient registered fitting time after setup')
    print('training_inputs_frozen',rep['contributions']['matrix_sha256'],flush=True)
    rep['attempts']=1;save()
    result=NN.greedy_nnls(G,b,taus=(),supports=(16384,),cap=16384,max_drops=3,max_iter=40000,
                          max_seconds=5400,log=lambda *x:print(*x,flush=True))
    np.savez(out/'NNLS_terminal.npz',support=result['support'],weights=result['weights'],
             nodes=cand[np.asarray(result['support'],int)],residual=result['terminal']['resvec'])
    rec=result['records'].get(('s',16384))
    optimum=result['reason']=='kkt' and result['kkt']['ok']
    # A final cleanup can record a target after a time/max-iteration cut. Such a record is retained, never admitted.
    normal_target=result['reason']=='taus' and rec is not None
    if not normal_target and not optimum:rec=None
    if rec is None and optimum and len(result['support']):
        pos=result['weights']>0
        rec=dict(support=result['support'][pos],weights=result['weights'][pos],size=int(pos.sum()),optimum=True)
    quality=bool(result.get('drift_ok',False) and 0<P.device_peak()<=120e9
                 and result['banned']<=.01*len(cand))
    reached=rec is not None
    rep['path']={k:v for k,v in result.items() if k not in ('records','terminal','path','support','weights')}
    rep['path']['path']=result['path']
    rep.update(target_reached=reached,earlier_certified_optimum=optimum,quality_ok=quality,
               peak_device_bytes=P.device_peak(),elapsed_seconds=time.perf_counter()-start)
    if reached:
        nodes=cand[np.asarray(rec['support'],int)];weights=np.asarray(rec['weights'],float)
        positive=bool(np.isfinite(weights).all() and np.all(weights>0) and len(nodes)>0)
        rep['positive_weights']=positive
        if quality and positive:
            np.savez(out/'HR~ecsw~512~k32.npz',nodes=nodes,weights=weights)
            rep['support_sha256']=sha(out/'HR~ecsw~512~k32.npz')
            rep['support_points']=len(nodes)
    rep['accepted']=bool(reached and quality and rep.get('positive_weights',False))
    rep['complete']=True;save()
    print('fit_complete',rep['accepted'],rep.get('support_sha256'),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--artifacts',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();run(json.loads(Path(a.config).read_text()),Path(a.artifacts),Path(a.out))
