"""Independent analytical POD initial-projection check for every invocation."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.collected);out=root/'output'
    raw=json.loads((out/'result.json').read_text());cfg=raw['config'];assets=out/'assets' if cfg['evaluation_cohort']=='development' else root/cfg['frozen_source_directory']/'assets'
    with np.load(assets/'pod.npz') as f:basis=f['basis']
    with np.load(out/'dev_data.npz') as f:states=f['states']
    target=states[:,0].reshape(len(states),-1)@basis;checks=[]
    with np.load(out/'latent_histories.npz') as history:
        for row in json.loads((out/'timing_rows.json').read_text()):
            if not row['method'].startswith('pod_weak_'):continue
            rank=int(row['method'].rsplit('_',1)[-1]);case=row['case'];name=f"{row['method']}__case{case}__rep{row['repetition']}"
            initial=history[name][0];relative=float(np.linalg.norm(initial-target[case,:rank])/max(np.linalg.norm(target[case,:rank]),1e-300))
            checks.append(dict(method=row['method'],case=case,repetition=row['repetition'],coefficient_relative=relative,passed=relative<1e-10 and row['cold'][1]==0 and row['cold'][2]==4))
    result=dict(passed=bool(checks) and all(x['passed'] for x in checks),complete=True,scope=__doc__,checks=checks)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),maximum_relative=max(x['coefficient_relative'] for x in checks))))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
