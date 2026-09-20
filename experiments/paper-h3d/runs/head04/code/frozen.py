"""Hash-checked reuse of trained heat checkpoints, never regenerated field data."""
from __future__ import annotations
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_inputs(root,source_attempt,destination,cfg):
    assert source_attempt.isalnum()
    run=Path(root)/'experiments/paper-h3d/runs'/source_attempt
    assert json.loads((run/'COLLECTED.json').read_text())['checksums_verified']
    assert json.loads((run/'audit-local.json').read_text())['passed']
    source=run/'archive/out';record=json.loads((source/'result.json').read_text())
    assert record['complete'] and not record['final_cohort_opened']
    files=['bank.pkl','bank_curve.json','cohorts.json']
    dimensions=cfg['latent_dimensions']+cfg.get('frozen_additional_heads',[])
    assert len(dimensions)==len(set(dimensions))
    for k in dimensions:files += [f'head_K{k}.pkl',f'head_K{k}_curve.json']
    for name in cfg.get('frozen_operators',[]):
        files += [f'operators/{name}/adapter.pkl',f'operators/{name}/best.pkl',f'operators/{name}/training.json',f'operators/{name}/curve.json']
    for n in cfg['evaluation_intervals']:
        for k in dimensions:
            name=f'eq_N{n}_K{k}.npz'
            if (source/name).exists():files.append(name)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    hashes={};copied_paths={}
    for name in files:
        original=source/name
        if not original.exists():original=source/'frozen_training'/name
        if not original.exists() and record['config'].get('frozen_input_directory'):
            original=source.parent/record['config']['frozen_input_directory']/name
        assert original.exists(),original
        dest=destination/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,dest)
        hashes[name]=file_sha(dest);assert hashes[name]==file_sha(original)
        copied_paths[name]=str(original.relative_to(run/'archive'))
    # Small derived descriptor; original raw result remains in its immutable archive.
    info=dict(source_attempt=source_attempt,source_commit=record['source_commit'],job_id=record['job_id'],
              original_result_sha256=file_sha(source/'result.json'),config=record['config'],files=hashes,
              bank=record['bank'],heads=record['heads'],operators=record.get('operators',[]),
              meshes=[{k:m[k] for k in ('intervals','bank_sha256','weak_operator_sha256','weak_tests','quadrature')} for m in record['meshes']],
              independent_source_audit=json.loads((run/'audit-local.json').read_text()),
              copied_from_original_archive_paths=copied_paths,
              copied_training_fields=False)
    (destination/'ORIGIN.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    return info


def load(directory,cfg,train_parameters,validation_parameters,out):
    import jax
    import common as C
    start=time.perf_counter();directory=Path(directory);origin=json.loads((directory/'ORIGIN.json').read_text())
    for name,expected in origin['files'].items():assert file_sha(directory/name)==expected,(name,'frozen checkpoint hash')
    physical_keys=('train_intervals','train_count','train_seed','validation_count','validation_seed','diffusivity','times',
                   'bank_rank','latent_dimensions','fourier_features','fourier_scale','bank_width','head_width')
    for key in physical_keys:assert cfg[key]==origin['config'][key],(key,'frozen scientific configuration differs')
    cohorts=json.loads((directory/'cohorts.json').read_text())
    assert C.sha(train_parameters)==cohorts['train_sha256']
    assert C.sha(validation_parameters)==cohorts['validation_sha256']
    def checkpoint(name):
        value=pickle.loads((directory/name).read_bytes())
        # Model weights are uploaded once during setup. Keep explicit JIT args.
        value['params']=jax.device_put(value['params']);jax.block_until_ready(value['params'])
        dest=Path(out)/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(directory/name,dest)
        return value
    bank=checkpoint('bank.pkl');models=[]
    bank['info']={**bank['info'],'reused_checkpoint':True,'original_training_source':origin['bank'].get('original_training_source',origin['source_commit']),
                  'immediate_reuse_source':origin['source_commit'],
                  'checkpoint_sha256':origin['files']['bank.pkl']}
    for k in cfg['latent_dimensions']+cfg.get('frozen_additional_heads',[]):
        model=checkpoint(f'head_K{k}.pkl');model['codes']=jax.device_put(model['codes']);jax.block_until_ready(model['codes'])
        original_head=next(h for h in origin['heads'] if h['k']==k)
        model['info']={**model['info'],'reused_checkpoint':True,'original_training_source':original_head.get('original_training_source',origin['source_commit']),
                       'immediate_reuse_source':origin['source_commit'],
                       'checkpoint_sha256':origin['files'][f'head_K{k}.pkl']}
        models.append(model)
    operators=[]
    for name in cfg.get('frozen_operators',[]):
        item=checkpoint(f'operators/{name}/adapter.pkl')
        original_operator=next(o for o in origin['operators'] if o['name']==name)
        info={**item['info'],'reused_checkpoint':True,'original_training_source':original_operator.get('original_training_source',origin['source_commit']),
              'immediate_reuse_source':origin['source_commit'],
              'checkpoint_sha256':origin['files'][f'operators/{name}/adapter.pkl']}
        operators.append((name,item['spec'],item['params'],item['physical_scale'],info))
    for name in origin['files']:
        if name.endswith('_curve.json') or name.endswith('/curve.json') or name.endswith('/training.json'):
            dest=Path(out)/'frozen_training'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(directory/name,dest)
    info={k:origin[k] for k in ('source_attempt','source_commit','job_id','original_result_sha256','files','copied_training_fields')}
    info.update(checkpoint_loading_seconds=time.perf_counter()-start,parameters_uploaded_before_timing=True,
                training_cohort_sha256=C.sha(train_parameters),validation_cohort_sha256=C.sha(validation_parameters))
    return bank,models,operators,origin,info


def quadrature(directory,origin,n,k,bank,a,test,truths,cfg,out):
    import common as C
    name=f'eq_N{n}_K{k}.npz';path=Path(directory)/name
    mesh=next((m for m in origin['meshes'] if m['intervals']==n),None)
    if not path.exists() or mesh is None:return None
    if mesh['bank_sha256']!=C.sha(bank) or mesh['weak_operator_sha256']!=C.sha(a) or mesh['weak_tests']!=len(test.T):return None
    for key in ('weak_tests','quadrature_seed','quadrature_candidates','quadrature_decoder_snapshots','quadrature_fit_rows','quadrature_scale_floor'):
        if cfg.get(key)!=origin['config'].get(key):return None
    original=next((q for q in mesh['quadrature'] if q['k']==k),None)
    if original is None or 'error' in original:return None
    assert file_sha(path)==origin['files'][name]
    data=np.load(path);indices=data['indices'];weights=data['weighted_tests']
    exact=np.asarray([u[0].reshape(-1)@test for u in truths]);sample=np.asarray([u[0].reshape(-1)[indices]@weights for u in truths])
    errors=np.linalg.norm(sample-exact,axis=1)/np.maximum(np.linalg.norm(exact,axis=1),1e-300)
    metadata={**original,'reused_rule':True,'original_source_commit':origin['source_commit'],'rule_file_sha256':file_sha(path),
              'validation_moment_errors':errors.tolist(),'certificate_threshold':cfg['quadrature_certificate'],
              'certified':bool(max(errors)<cfg['quadrature_certificate']),
              'reuse_gate':'exact N/M/bank/weak-operator/checkpoint/file/config hashes; held-out moments recomputed'}
    shutil.copy2(path,Path(out)/name)
    return indices,weights,metadata
