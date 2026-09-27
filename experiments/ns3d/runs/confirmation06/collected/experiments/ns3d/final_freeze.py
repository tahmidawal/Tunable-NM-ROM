"""Content-bound model and solver selection before the final NS3D draw."""
import argparse,json
from pathlib import Path
from extra03 import file_hash


def configuration(cfg):return {k:v for k,v in cfg.items() if k!='final_freeze_path'}


def prepare(cfg,source):
    source=Path(source);raw=json.loads((source/'result.json').read_text())
    assert raw['complete'] and not raw['final_cohort_opened'] and raw['checkpoint_replay']['passed']
    audit_root=source.parent.parent;audits={}
    for name in ('audit.json','history_audit.json','source_audit.json','teacher_audit.json'):
        value=json.loads((audit_root/name).read_text());assert value['passed'],name
        audits[name]=file_hash(audit_root/name)
    assert cfg['evaluation_cohort']=='final' and cfg['final_seed_unopened']==202609203 and cfg['timed_cases']==32
    assert {s['kind'] for s in cfg['operators']}=={'fno3d','unet3d','deeponet3d','transolver3d'}
    fixed=('k','r','selected_head','q_values','test_modes','rom_dt','rom_budget','cold_starts','gtol','operators','operator_projection_variants','fom_dts','reference_dt','fine_reference_n','fine_reference_dt','refinement_n','refinement_dt','refinement_budget','repetitions')
    assert all(cfg[k]==raw['config'][k] for k in fixed),'Final online choices must have a real development replay'
    assets=json.loads((source/'assets/manifest.json').read_text())
    assert all(file_hash(source/'assets'/name)==value for name,value in assets['files'].items())
    passing=[(name,row) for name,row in raw['timing_summary'].items() if name.startswith('fom_') and row['nonfinite_invocations']==0 and row['worst_evolved_physical']<=cfg['physical_accuracy_target']]
    assert passing,'No validation-passing FOM control to freeze'
    cheapest=min(passing,key=lambda item:item[1]['gpu_seconds_median'])[0]
    return dict(schema='ns3d-final-freeze-v1',configuration=configuration(cfg),assets=assets,accepted_development_audit_sha256=audits,
        asset_manifest_sha256=file_hash(source/'assets/manifest.json'),selection_source_commit=raw['source_commit'],selection_job_id=raw['job_id'],
        selection_files={name:file_hash(source/name) for name in ('result.json','timing_rows.json','timed_fields.npz')},
        cheapest_passing_development_fom=cheapest,final_fields_generated=False,
        policy='Actual offline assets and every solver choice are fixed before final generation; no final-based training, selection or exclusions.')


def verify(cfg):
    freeze=json.loads(Path(cfg['final_freeze_path']).read_text());source=Path(cfg['frozen_source_directory'])
    assert freeze['configuration']==configuration(cfg) and freeze['final_fields_generated'] is False
    assert freeze['asset_manifest_sha256']==file_hash(source/'assets/manifest.json')
    assert all(file_hash(source/'assets'/name)==value for name,value in freeze['assets']['files'].items())
    assert all(file_hash(source/name)==value for name,value in freeze['selection_files'].items())
    assert cfg['evaluation_cohort']=='final' and cfg['final_seed_unopened']==202609203 and cfg['timed_cases']==32
    return freeze


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    target=Path(a.output);assert not target.exists(),'Do not replace a sealed selection record'
    target.write_text(json.dumps(prepare(json.loads(Path(a.config).read_text()),a.source),indent=2)+'\n')
