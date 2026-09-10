"""Independent NumPy field/provenance audit; never imports solver kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    root=Path(args.directory);handoff=json.loads((root/'handoff.json').read_text())
    rows=handoff['invocations'];proxy=handoff.get('selection_timings',[])
    fieldchecks=[];errors=[]
    identities={(r['provenance']['job_id'],r['provenance']['gpu']) for r in rows+proxy}
    if len(identities)>1:errors.append('active timing rows mix job/GPU allocations')
    case0={(r['intervals'],r['configuration']):r for r in rows if r['split']=='validation' and r['case']==0 and r['rep']==0}
    for row in rows:
        if not np.isfinite(row['seconds']) or row['seconds']<=0:errors.append(f'invalid time {row["configuration"]}')
        if not row.get('field_artifact'):continue
        data=np.load(root/row['field_artifact']);u=data['u'];truth=data['truth_u']
        hashed=hashlib.sha256(u.tobytes()).hexdigest()
        if hashed!=row['field_sha256']:errors.append('field hash mismatch '+row['field_artifact'])
        measured=None;defect=None
        if np.isfinite(u).all():
            measured=float(np.max(np.linalg.norm((u-truth).reshape(len(u),-1),axis=1))/max(np.linalg.norm(truth[0]),1e-30))
            defect=abs(measured-row['errors']['displacement'])
            if defect>1e-10:errors.append('field error mismatch '+row['field_artifact'])
        elif row['finite']:errors.append('nonfinite output labeled finite '+row['field_artifact'])
        fieldchecks.append(dict(artifact=row['field_artifact'],error=measured,absolute_defect=defect,hash_verified=hashed==row['field_sha256']))
    mismatches=[]
    for row in proxy:
        ref=case0.get((row['intervals'],row['configuration']))
        if ref is not None and row['field_sha256']!=ref['field_sha256']:
            mismatches.append(dict(intervals=row['intervals'],configuration=row['configuration'],rep=row['rep']))
    if mismatches:errors.append('selection proxy differs from matching validation query')
    for selection in handoff.get('selections',[]):
        if not selection['validation_passed']:continue
        if not case0:continue # evaluation-only phase carries immutable validation choice
        chosen=[r for r in rows if r['split']=='validation' and r['configuration']==selection['configuration'] and r['intervals']==selection['intervals']]
        selected_proxy=[r for r in proxy if r['configuration']==selection['configuration'] and r['intervals']==selection['intervals']]
        target=selection['target']
        if not chosen or not all(r['finite'] and r['completed'] and r['errors']['displacement']<=target for r in chosen):errors.append('selected cohort failed target')
        if not selected_proxy or not all(r['finite'] and r['errors']['displacement']<=target for r in selected_proxy):errors.append('selected timing proxy failed target')
    models=[]
    for arm,expected in handoff['checkpoint_hashes'].items():
        p=root/'checkpoints'/f'{arm}.pkl'
        actual=sha(p)
        if actual!=expected:errors.append('changed checkpoint '+arm)
        with p.open('rb') as f:model=pickle.load(f)
        if not np.isfinite(model['Z']).all():errors.append('nonfinite training codes '+arm)
        models.append(dict(architecture=arm,sha256=actual,latent_codes_shape=list(model['Z'].shape),
                           recorded_stage_steps=model['extra'].get('steps'),recorded_total_updates=model['extra'].get('updates')))
    output=dict(passed=not errors,errors=errors,field_checks=fieldchecks,proxy_hash_mismatches=mismatches,
                invocation_count=len(rows),proxy_count=len(proxy),identities=[list(x) for x in identities],models=models,
                handoff_sha256=sha(root/'handoff.json'),audit_kind='independent NumPy field metrics and immutable model/proxy provenance; not PDE reference proof')
    (root/'independent_owner_audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['field_checks','models']},indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
