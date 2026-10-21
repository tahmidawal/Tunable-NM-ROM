"""CPU-only hash/config tamper checks on a tiny fixture; no final generation."""
import copy,json
from pathlib import Path
from final_freeze import verify,configuration
from extra03 import file_hash


def main():
    source=Path('experiments/ns3d/checks/frozen_panel_v2/output');out=Path('experiments/ns3d/checks/freeze_guard');out.mkdir(exist_ok=True)
    cfg=json.loads((source/'result.json').read_text())['config'];cfg.update(evaluation_cohort='final',timed_cases=32,final_seed_unopened=202609203,
        final_freeze_path=str(out/'synthetic_freeze.json'),frozen_source_directory=str(source),smoke_fixture_only=True)
    manifest=json.loads((source/'assets/manifest.json').read_text())
    freeze=dict(configuration=configuration(cfg),final_fields_generated=False,assets=manifest,asset_manifest_sha256=file_hash(source/'assets/manifest.json'),
        selection_files={name:file_hash(source/name) for name in ('result.json','timing_rows.json','timed_fields.npz')},smoke_fixture_only=True)
    path=Path(cfg['final_freeze_path']);path.write_text(json.dumps(freeze,indent=2)+'\n');verify(cfg)
    changed=copy.deepcopy(cfg);changed['rom_budget']+=1
    try:verify(changed)
    except AssertionError:config_rejected=True
    else:raise AssertionError('configuration mutation was accepted')
    changed=copy.deepcopy(freeze);changed['assets']['files']['head.pkl']='0'*64;path.write_text(json.dumps(changed,indent=2)+'\n')
    try:verify(cfg)
    except AssertionError:hash_rejected=True
    else:raise AssertionError('checkpoint hash mutation was accepted')
    path.write_text(json.dumps(freeze,indent=2)+'\n')
    report=dict(passed=True,correct_fixture_accepted=True,config_mutation_rejected=config_rejected,checkpoint_hash_mutation_rejected=hash_rejected,
        final_parameter_generator_called=False,scientific_final_freeze=False)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
