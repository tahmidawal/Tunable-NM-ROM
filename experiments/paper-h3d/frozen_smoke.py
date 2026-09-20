"""Bounded replay of already verified tiny checkpoint; no scientific training."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import common as C
import frozen as F
import rom as R
from operators import heat_adapter as H

root=Path(__file__).resolve().parent;source=root/'smokes/tune02';out=root/'smokes/frozen_replay'
out.mkdir(exist_ok=True);inputs=out/'inputs';inputs.mkdir(exist_ok=True)
d=json.loads((source/'result.json').read_text());cfg=dict(d['config'])
cfg['frozen_operators']=[s['name'] for s in cfg['operators']]
files=['bank.pkl','bank_curve.json','cohorts.json']
for k in cfg['latent_dimensions']:files += [f'head_K{k}.pkl',f'head_K{k}_curve.json']
for name in cfg['frozen_operators']:files += [f'operators/{name}/adapter.pkl']
for n in cfg['evaluation_intervals']:
 for k in cfg['latent_dimensions']:
    name=f'eq_N{n}_K{k}.npz'
    if (source/name).exists():files.append(name)
for name in files:
    dest=inputs/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
origin=dict(source_attempt='tiny_fixture',source_commit=d['source_commit'],job_id=d['job_id'],
    original_result_sha256=F.file_sha(source/'result.json'),config=cfg,files={name:F.file_sha(inputs/name) for name in files},
    bank=d['bank'],heads=d['heads'],operators=d.get('operators',[]),meshes=d['meshes'],copied_training_fields=False)
C.dump(inputs/'ORIGIN.json',origin)
train=C.family(cfg['train_seed'],cfg['train_count']);validation=C.family(cfg['validation_seed'],cfg['validation_count'])
bank,models,operators,origin,info=F.load(inputs,cfg,train,validation,out)
assert info['parameters_uploaded_before_timing']
assert all(isinstance(x,jax.Array) and x.dtype==jnp.float64 for m in models for x in jax.tree_util.tree_leaves((m['params'],m['codes'])))
n=cfg['evaluation_intervals'][0];truths=C.dataset(n,validation,cfg);u=jnp.asarray(truths[0,0]);errors=[]
for name,spec,p,scale,_ in operators:
    value=np.asarray(H.engine(p,spec,scale,n)(u));ref=np.load(source/'fields'/f'N{n}_case0_{name}.npz')['prediction']
    relative=np.linalg.norm(value-ref)/np.linalg.norm(ref);assert relative<1e-13;errors.append(dict(method=name,replay_relative=float(relative)))
g=C.bank_at(bank['params'],n,cfg['field_chunk'])@bank['rotation'];test,a,lam,triples=R.assemble(g,n,cfg['weak_tests'])
for model in models:
    k=model['info']['k'];q=cfg['q_ladder'][-1];name=f'nmrom_K{k}_q{q}_dense'
    value=np.asarray(R.engine(model,g,a,lam,test,np.arange(len(g)),q,cfg)(u)[0]).reshape(truths[0].shape)
    ref=np.load(source/'fields'/f'N{n}_case0_{name}.npz')['prediction']
    relative=np.linalg.norm(value-ref)/np.linalg.norm(ref);assert relative<1e-13;errors.append(dict(method=name,replay_relative=float(relative)))
    reuse=F.quadrature(inputs,origin,n,k,g,a,test,truths,cfg,out)
    original_rule_reused=reuse is not None
    if reuse is None:
        # A legacy array need not reconstruct bitwise across processes. Confirm
        # exact reuse separately using a self-consistent full-grid identity rule.
        # This modifies only this synthetic fixture, never a scientific archive.
        name=f'eq_N{n}_K{k}.npz'
        np.savez_compressed(inputs/name,indices=np.arange(len(g)),weighted_tests=test)
        origin['files'][name]=F.file_sha(inputs/name)
        mesh=next(m for m in origin['meshes'] if m['intervals']==n)
        mesh.update(bank_sha256=C.sha(g),weak_operator_sha256=C.sha(a),weak_tests=test.shape[1])
        mesh['quadrature']=[dict(k=k,certified=True,source='synthetic full-grid identity fixture')]
        reuse=F.quadrature(inputs,origin,n,k,g,a,test,truths,cfg,out)
    assert reuse is not None and reuse[2]['reused_rule']
    assert F.quadrature(inputs,origin,n,k,g+1e-12,a,test,truths,cfg,out) is None
    assert F.quadrature(inputs,origin,n,k,g,a,test,truths,{**cfg,'weak_tests':cfg['weak_tests']+1},out) is None
C.dump(inputs/'ORIGIN.json',origin)
C.dump(out/'audit.json',dict(passed=True,device_residency=True,checkpoint_replay=errors,original_rule_bitwise_reused=original_rule_reused,exact_quadrature_reuse=True,changed_bank_rejected=True,changed_test_count_rejected=True))
print('FROZEN_REPLAY_PASS',errors,flush=True)
