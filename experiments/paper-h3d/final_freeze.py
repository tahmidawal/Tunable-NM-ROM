"""Freeze exact scientific settings and model bytes before final parameter draw."""
import argparse,hashlib,json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def configuration(cfg):return {k:v for k,v in cfg.items() if k!='final_freeze_path'}


def names(cfg):
    return ['bank.pkl']+[f'head_K{k}.pkl' for k in cfg['latent_dimensions']]+[
        f'operators/{name}/adapter.pkl' for name in cfg['frozen_operators']]+[
        f'pod_N{n}.pkl' for n in cfg['evaluation_intervals'] if cfg.get('include_linear_controls',True)]


def gates(cfg):
    assert cfg['evaluation_cohort']=='final' and cfg['reserved_final_seed']==920399 and cfg['reserved_final_count']==64
    assert cfg['reserved_final_seed'] not in (cfg['train_seed'],cfg['validation_seed'])
    assert cfg.get('frozen_input_directory') and not cfg.get('operators') and not cfg.get('additional_latent_dimensions')
    assert not cfg.get('representation_oracles',True) and cfg.get('retain_solver_states')
    assert len(cfg['frozen_operators'])==4 and not cfg.get('include_uncertified_quadrature',True)
    assert not cfg.get('fit_quadrature',True),'final sampled-arm membership must not be chosen from final certificate results'


def prepare(cfg,source):
    gates(cfg);source=Path(source);record=json.loads((source/'result.json').read_text())
    assert record['complete'] and not record['final_cohort_opened']
    if cfg.get('confirmation_role')=='independent_seed_accuracy':
        assert cfg.get('primary_freeze_sha256') and not cfg.get('include_linear_controls',True)
        assert record['config']['seed_role'].startswith('independent initialization')
    else:assert record['invocations']
    assert {op['spec']['kind'] for op in record['operators']}=={'fno3d','unet3d','deeponet3d','transolver3d'}
    return dict(schema='heat3d-final-freeze-v1',configuration=configuration(cfg),
        checkpoint_sha256={name:digest(source/name) for name in names(cfg)},
        selection_result_sha256=digest(source/'result.json'),selection_source_commit=record['source_commit'],
        selection_job_id=record['job_id'],final_fields_generated=False,
        policy='Final cohort is evaluated once after development-only recipe/head/q/transfer/control selection; no final-based retraining.')


def verify(cfg):
    gates(cfg);path=Path(cfg['final_freeze_path']);f=json.loads(path.read_text())
    assert f['configuration']==configuration(cfg) and f['final_fields_generated'] is False
    source=Path(cfg['frozen_input_directory']);origin=json.loads((source/'ORIGIN.json').read_text())
    assert origin['original_result_sha256']==f['selection_result_sha256']
    assert f['checkpoint_sha256']=={name:digest(source/name) for name in names(cfg)}
    assert {op['spec']['kind'] for op in origin['operators']}=={'fno3d','unet3d','deeponet3d','transolver3d'}
    companion=None
    if cfg.get('companion_config_file'):
        other=json.loads(Path(cfg['companion_config_file']).read_text())
        assert other['confirmation_role']=='independent_seed_accuracy' and other['primary_freeze_sha256']==digest(path)
        assert not other.get('companion_config_file')
        for key in ('train_seed','train_count','validation_seed','validation_count','reserved_final_seed','reserved_final_count',
                    'diffusivity','times','dt','bank_rank','latent_dimensions','weak_tests','q_ladder','lm_budget','lm_tolerance','initial_starts','initial_lm_budget'):
            assert other[key]==cfg[key],(key,'seed comparison scientific configuration mismatch')
        assert other['evaluation_intervals']==[cfg['train_intervals']]
        companion=verify(other)
    return dict(path=str(path),sha256=digest(path),selection_source_commit=f['selection_source_commit'],
        selection_job_id=f['selection_job_id'],verified_before_final_parameter_generation=True,companion_verified_before_first_draw=companion)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);assert not out.exists();out.write_text(json.dumps(prepare(json.loads(Path(a.config).read_text()),Path(a.source)),indent=2)+'\n')
