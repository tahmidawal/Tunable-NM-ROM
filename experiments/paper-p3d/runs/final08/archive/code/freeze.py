"""Enforce a committed final-selection record before drawing final parameters.

Preparing the record reads checkpoints/configuration only, never final fields.
The cluster stager then proves the exact record is in the submitted Git commit.
"""
from pathlib import Path
import argparse
import hashlib
import json


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def configuration(cfg):
    return {k:v for k,v in cfg.items() if k!='final_freeze_path'}


def checkpoint_names(cfg):
    from offline_assets import names
    return ['bank.pkl']+[f'head_K{k}.pkl' for k in cfg['latent_dimensions']]+[
        f"operators/{entry['name']}/best.pkl" for entry in cfg['operators']]+names(cfg)


def prepare(cfg, source):
    assert cfg['evaluation_cohort']=='final' and cfg['final_count']>0
    assert cfg['final_cohort_opened'] is True and cfg['reserved_final_seed']==920499
    assert cfg.get('frozen_offline_assets') is True,'final POD assets must be frozen from a real development replay'
    assert {x['spec']['kind'] for x in cfg['operators']}=={'fno3d','unet3d','deeponet3d','transolver3d'}
    assert cfg.get('reuse_checkpoint_directory') and all(x.get('reuse') for x in cfg['operators'])
    assert cfg['reserved_final_seed'] not in [cfg['train_seed'],cfg['validation_seed']]
    prior=json.loads((source/'result.json').read_text())
    assert prior['complete'] and not prior['final_cohort_opened']
    assert cfg.get('plain_cg_control') is True and prior['config'].get('plain_cg_control') is True,'efficient CG needs a real paired development replay'
    assert cfg.get('iterative_cg') and cfg['iterative_cg']==prior['config'].get('iterative_cg'), 'CG settings must have a real paired development replay'
    for mesh in prior['meshes']:
        assert {f'cg_identity_rtol{tol:.0e}' for tol in cfg['iterative_cg']['relative_tolerances']} <= set(mesh['methods'])
        assert {f'cg_identity_plain_rtol{tol:.0e}' for tol in cfg['iterative_cg']['relative_tolerances']} <= set(mesh['methods'])
    return dict(schema='poisson3d-final-freeze-v1',configuration=configuration(cfg),
        checkpoint_sha256={name:digest(source/name) for name in checkpoint_names(cfg)},
        selection_result_sha256=digest(source/'result.json'),selection_source_commit=prior['source_commit'],
        selection_job_id=prior['job_id'],final_fields_generated=False,
        policy='All configurations fixed before final parameter generation; no final-based retraining or selection.')


def verify_final_freeze(cfg):
    path=Path(cfg['final_freeze_path']);freeze=json.loads(path.read_text())
    assert freeze['configuration']==configuration(cfg),'final configuration differs from committed freeze'
    assert freeze['final_fields_generated'] is False
    source=Path(cfg['reuse_checkpoint_directory'])
    assert freeze['selection_result_sha256']==digest(source/'result.json')
    assert cfg.get('plain_cg_control') is True and json.loads((source/'result.json').read_text())['config'].get('plain_cg_control') is True
    prior=json.loads((source/'result.json').read_text())
    assert cfg.get('iterative_cg') and cfg['iterative_cg']==prior['config'].get('iterative_cg'), 'CG settings lack paired development replay'
    assert freeze['checkpoint_sha256']=={name:digest(source/name) for name in checkpoint_names(cfg)}
    assert all(x.get('reuse') for x in cfg['operators']),'final evaluation cannot train operators'
    assert cfg['final_cohort_opened'] is True and cfg['reserved_final_seed']==920499
    assert {x['spec']['kind'] for x in cfg['operators']}=={'fno3d','unet3d','deeponet3d','transolver3d'}
    assert cfg.get('representation_oracles',True) is False,'final model-selection oracle disabled'
    assert cfg.get('frozen_offline_assets') is True,'final POD assets cannot be refitted'
    return dict(path=str(path),sha256=digest(path),selection_source_commit=freeze['selection_source_commit'],
                selection_job_id=freeze['selection_job_id'],verified_before_final_parameter_generation=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.output)
    assert not out.exists(),'a freeze record must not be silently overwritten'
    out.write_text(json.dumps(prepare(cfg,Path(a.source)),indent=2)+'\n')
