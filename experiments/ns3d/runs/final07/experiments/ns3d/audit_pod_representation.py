"""Independent development projection metrics of retained POD bases.

This verifies fixed-basis projection errors, not a universal optimality claim.
Basis isometry is checked with recorded global coefficient probes.
"""
import argparse,json
from pathlib import Path
import numpy as np
from audit_comparison import file_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.collected)/'output'
    raw=json.loads((out/'result.json').read_text());screen=json.loads((out/'capacity/screen.json').read_text())
    with np.load(out/'capacity/pod.npz') as f:basis=f['basis']
    with np.load(out/'dev_data.npz') as f:states=f['states']
    x=states.reshape(-1,basis.shape[0]);den=np.repeat(np.linalg.norm(states[:,0].reshape(len(states),-1),axis=1),states.shape[1]);coefficients=x@basis
    rng=np.random.default_rng(202609328);c=rng.normal(size=(basis.shape[1],32));v=basis@c
    isometry=float(np.linalg.norm(v.T@v-c.T@c)/np.linalg.norm(c.T@c));checks={}
    for key,record in screen['POD'].items():
        rank=int(key);error=np.linalg.norm(x-coefficients[:,:rank]@basis[:,:rank].T,axis=1)/den
        measured=dict(mean=float(np.mean(error)),median=float(np.median(error)),worst=float(np.max(error)),over_5pct=int(np.sum(error>.05)),count=len(error))
        disagreement=max(abs(measured[k]-record['development_initial_normalized'][k]) for k in measured)
        checks[key]=dict(passed=disagreement<1e-10,maximum_metric_disagreement=disagreement,development_initial_normalized=measured)
    result=dict(passed=isometry<1e-10 and all(x['passed'] for x in checks.values()),complete=True,source_commit=raw['source_commit'],job_id=raw['job_id'],
        scope=__doc__,cases=len(states),snapshots=len(x),global_isometry_probe_count=32,probe_seed=202609328,global_isometry_relative=isometry,checks=checks,
        input_sha256={name:file_sha(out/name) for name in ('capacity/pod.npz','capacity/screen.json','dev_data.npz')})
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
