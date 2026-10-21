"""Independent NumPy head Jacobians for every retained development fit."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from audit_pilot import head,jacobian


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.collected)/'output';cap=out/'capacity';screen=json.loads((cap/'screen.json').read_text())
    with (cap/'frozen_bank.pkl').open('rb') as f:bank=pickle.load(f)
    with np.load(out/'dev_data.npz') as f:X=f['states'].reshape(-1,bank['extra']['bank'].shape[0])
    Rb=bank['extra']['qr_R'];targets=np.linalg.solve(Rb,(X@bank['extra']['qr_Q']).T).T;checks=[]
    for name,record in screen['heads'].items():
        with (cap/(name+'.pkl')).open('rb') as f:params=pickle.load(f)['params']
        for index,z in enumerate(np.asarray(record['fits']['z'])):
            residual=Rb@(head(params,z)-targets[index]);J=Rb@jacobian(params,z)
            gradient=float(np.linalg.norm(J.T@residual)/(np.linalg.norm(J)*np.linalg.norm(residual)+1e-300))
            recorded=record['fits']['normalized_gradient'][index];stationary=record['fits']['reasons'][index]==4
            checks.append(dict(head=name,snapshot=index,normalized_gradient=gradient,absolute_disagreement=abs(gradient-recorded),
                reported_stationary=stationary,stationarity_consistent=(not stationary or gradient<=1e-7+1e-10)))
    result=dict(passed=all(r['absolute_disagreement']<1e-10 and r['stationarity_consistent'] for r in checks),
        scope=__doc__,source_commit=json.loads((out/'result.json').read_text())['source_commit'],checks=checks)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),maximum_disagreement=max(r['absolute_disagreement'] for r in checks))))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
