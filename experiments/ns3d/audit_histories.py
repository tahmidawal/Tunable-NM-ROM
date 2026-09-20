"""Independent NumPy residual/Jacobian audit of retained NS3D latent histories.

Deterministic bounded scope: every development case and weak arm, first timing
repetition, cold state plus first/middle/last step. Every history hash is checked.
"""
import argparse,json
from pathlib import Path
import pickle
import numpy as np
from audit_pilot import head,jacobian,rel
from audit_comparison import array_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--reuse',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.collected)/'output';reuse=Path(a.reuse)
    report=json.loads((root/'result.json').read_text());cfg=report['config'];rows=json.loads((root/'timing_rows.json').read_text())
    with (reuse/'checkpoint.pkl').open('rb') as f:checkpoint=pickle.load(f)
    hp=checkpoint['params'];G=checkpoint['extra']['bank'];Q=checkpoint['extra']['qr_Q'];Rb=checkpoint['extra']['qr_R']
    with np.load(reuse/'weak_operators.npz') as f:wa=f['A'];wt=f['T'];lam=f['lam'];C=f['C']
    with np.load(root/'pod_weak_operators.npz') as f:pa=f['A'];pt=f['T']
    with np.load(reuse/'pod.npz') as f:P=f['basis']
    with np.load(root/'dev_data.npz') as f:U=f['states'];pars=f['parameters']
    checks=[];hashfail=[]
    def measure(res,J):return float(np.linalg.norm(res)),float(np.linalg.norm(J.T@res)/(np.linalg.norm(J)*np.linalg.norm(res)+1e-300))
    with np.load(root/'latent_histories.npz') as hist:
        for row in rows:
            if 'state_sha256' not in row:continue
            key=f"{row['method']}__case{row['case']}__rep{row['repetition']}";W=hist[key]
            if array_sha(W)!=row['state_sha256']:hashfail.append(key)
            if row['repetition']!=0:continue
            linear=row['method'].startswith('pod_weak');k=0 if linear else cfg['k'];q=int(row['method'].split('_')[-1]) if linear else int(row['method'].split('q')[-1])
            if linear:basis=P[:,:q];metric=np.eye(q);Qb=basis;A=pa[:,:q];T=pt[:,:q,:q];cor=np.empty((q,0))
            else:basis=G;metric=Rb;Qb=Q;A=wa;T=wt;cor=C[:,:q]
            def c(w):return w if linear else head(hp,w[:k])+cor@w[k:]
            def Jc(w):return np.eye(q) if linear else np.concatenate((jacobian(hp,w[:k]),cor),axis=1)
            case=row['case'];target=np.linalg.solve(metric,Qb.T@U[case,0].ravel());coldres=metric@(c(W[0])-target);coldJ=metric@Jc(W[0]);coldrn,coldgn=measure(coldres,coldJ)
            if not linear:checks.append(dict(method=row['method'],case=case,step='cold',residual_error=abs(coldrn-row['cold'][0]),gradient_error=abs(coldgn-row['cold'][3]),normalized_gradient=coldgn,reported_reason=row['cold'][2]))
            for step in sorted(set([0,(len(W)-2)//2,len(W)-2])):
                old=c(W[step]);new=c(W[step+1]);mid=.5*(old+new);dt=cfg['rom_dt'];nu=pars[case,-1];den=1+.5*dt*nu*lam
                contraction=np.einsum('mjk,j,k->m',T,mid,mid,optimize=True)
                deriv=np.einsum('mjk,k->mj',T,mid,optimize=True)+np.einsum('mjk,j->mk',T,mid,optimize=True)
                residual=(A@(new-old)-dt*(contraction-nu*lam*(A@mid)))/den
                J=((A-.5*dt*deriv+.5*dt*nu*lam[:,None]*A)/den[:,None])@Jc(W[step+1]);rn,gn=measure(residual,J)
                checks.append(dict(method=row['method'],case=case,step=step+1,residual_error=abs(rn-row['steps'][0][step]),gradient_error=abs(gn-row['steps'][3][step]),normalized_gradient=gn,reported_reason=row['steps'][2][step]))
    result=dict(passed=not hashfail and all(r['residual_error']<1e-8 and r['gradient_error']<1e-8 for r in checks),
        scope='all latent history hashes; every case/weak method, repetition0, cold and first/middle/last steps independently differentiated',hash_failures=hashfail,checks=checks)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}));raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
