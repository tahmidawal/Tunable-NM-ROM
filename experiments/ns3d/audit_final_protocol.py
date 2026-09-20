"""Independently bind final outputs to the precommitted model/config freeze."""
import argparse, hashlib, json, subprocess
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024**2): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--out',required=True);a=p.parse_args()
    run=Path(a.run);root=run/'collected';out=root/'output'
    raw=json.loads((out/'result.json').read_text());cfg=raw['config']
    freeze_path=root/cfg['final_freeze_path'];freeze=json.loads(freeze_path.read_text())
    commit=(root/'COMMIT.txt').read_text().strip();checks={}
    def gate(name,passed,**values): checks[name]=dict(passed=bool(passed),**values)
    git_blob=subprocess.check_output(['git','show',commit+':'+cfg['final_freeze_path']])
    gate('precommitted_complete_configuration',git_blob==freeze_path.read_bytes() and raw['freeze']==freeze and
         freeze['configuration']=={k:v for k,v in cfg.items() if k!='final_freeze_path'} and freeze['final_fields_generated'] is False)
    observed=json.loads((run/'replay_observation.json').read_text());proof=json.loads((out/'freeze_verification.json').read_text())
    replay=json.loads((out/'development_replay.json').read_text())
    gate('preobserved_real_replay',proof['development_replay']==replay==raw['development_replay'] and replay['passed'] and
         observed['input_sha256']['development_replay.json']==sha(out/'development_replay.json') and
         observed['input_sha256']['freeze_verification.json']==sha(out/'freeze_verification.json') and
         observed['final_access_utc']==raw['final_access_utc']==proof['final_access_utc'] and
         len(replay['checks'])==len(raw['method_names']) and
         all(x['passed'] and x['identical_stopping'] for x in replay['checks']),
         largest_field_relative_difference=max(x['field_relative_difference'] for x in replay['checks']),
         final_access_utc=proof['final_access_utc'])
    source=root/cfg['frozen_source_directory'];bad=[]
    for name,expected in freeze['selection_files'].items():
        if sha(source/name)!=expected: bad.append(name)
    assets=json.loads((source/'assets/manifest.json').read_text())
    gate('exact_development_selection_bytes',not bad and sha(source/'assets/manifest.json')==freeze['asset_manifest_sha256'] and assets==freeze['assets'],failures=bad)
    dev=json.loads((source/'result.json').read_text())
    eligible=[(name,row['gpu_seconds_median']) for name,row in dev['timing_summary'].items()
              if name.startswith('fom_') and row['nonfinite_invocations']==0 and row['worst_evolved_physical']<=cfg['physical_accuracy_target']]
    gate('development_only_control_selection',dev['complete'] and not dev['final_cohort_opened'] and
         dev['source_commit']==freeze['selection_source_commit'] and dev['job_id']==freeze['selection_job_id'] and
         min(eligible,key=lambda x:x[1])[0]==freeze['cheapest_passing_development_fom'])
    bad=[]
    for name,expected in freeze['accepted_development_audit_sha256'].items():
        blob=subprocess.check_output(['git','show',commit+':experiments/ns3d/runs/confirmation06b/'+name])
        value=json.loads(blob)
        if hashlib.sha256(blob).hexdigest()!=expected or not value['passed']: bad.append(name)
    gate('development_audits_committed_before_final',not bad,files=len(freeze['accepted_development_audit_sha256']),failures=bad)
    gate('reserved_complete_final_membership',raw['complete'] and raw['verified_before_final_parameter_generation'] and
         proof['verified_before_final_parameter_generation'] and raw['final_cohort_opened'] and raw['cohort_count']==32 and
         raw['cohort_seed']==202609203 and raw['source_commit']==commit and str(raw['job_id'])==str(observed['job_id']) and
         set(raw['method_names'])=={x['method'] for x in replay['checks']})
    result=dict(passed=all(c['passed'] for c in checks.values()),complete=True,scope=__doc__,checks=checks,
                source_commit=commit,job_id=raw['job_id'])
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__': main()
