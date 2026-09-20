"""Independent advective-form Jacobians for the dense weak NS3D evaluator.

Checks all history hashes; cold and first/middle/last solves for every case and
weak arm in repetition zero. No JAX or production residual code is imported.
"""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np
from scipy.fft import fftn,ifftn
from audit_pilot import head,jacobian
from audit_comparison import array_sha
import ns3d_independent as I


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=Path(a.collected) if a.smoke else Path(a.collected)/'output';raw=json.loads((out/'result.json').read_text());cfg=raw['config']
    if not raw['larger_rollout_gate']['eligible']:
        result=dict(passed=True,applicable=False,scope='no larger rollout passed the prospective representation gate')
        Path(a.out).write_text(json.dumps(result,indent=2)+'\n');return
    selected=raw['larger_rollout_gate']['selected_head'];record=raw['capacity']['heads'][selected];k=record['configuration']['k']
    with (out/'capacity/frozen_bank.pkl').open('rb') as f:bank=pickle.load(f)
    with (out/'capacity'/(selected+'.pkl')).open('rb') as f:hp=pickle.load(f)['params']
    with np.load(out/'capacity/pod.npz') as f:P=f['basis']
    with np.load(out/'dense_weak_operators.npz') as f:A=f['A'];lam=f['lam'];C=f['C']
    ids=json.loads((out/'dense_weak_test_modes.json').read_text());indices=np.asarray([r['wave'] for r in ids])%cfg['n'];pol=np.asarray([r['polarization'] for r in ids]);cosine=np.asarray([r['kind']=='cos' for r in ids])
    n=cfg['n'];spec=I.setup(n);wave=spec[0];axes=(-3,-2,-1)
    tf=lambda u:fftn(u,axes=axes,norm='forward')
    inv=lambda h:ifftn(h,axes=axes,norm='forward').real
    def extract(h):
        vals=h[...,indices[:,0],indices[:,1],indices[:,2]]
        scalar=np.einsum('...cm,mc->...m',vals,pol)
        return np.sqrt(2*n**3)*np.where(cosine,scalar.real,-scalar.imag)
    with np.load(out/'dev_data.npz') as f:states=f['states'];parameters=f['parameters']
    rows=json.loads((out/'timing_rows.json').read_text());checks=[];bad=[]
    metric=lambda r,J:(float(np.linalg.norm(r)),float(np.linalg.norm(J.T@r)/(np.linalg.norm(J)*np.linalg.norm(r)+1e-300)))
    with np.load(out/'latent_histories.npz') as histories:
        for row in rows:
            if 'state_sha256' not in row:continue
            name=row['method'];case=row['case'];key=f"{name}__case{case}__rep{row['repetition']}";W=histories[key]
            if array_sha(W)!=row['state_sha256']:bad.append(key)
            if row['repetition']!=0:continue
            linear=name.startswith('pod_weak_')
            if linear:
                q=int(name.split('_')[-1]);G=P[:,:q];Q=G;Rb=np.eye(q);cor=None
                hG=tf(G.T.reshape(q,3,n,n,n));AA=extract(hG).T
            else:
                q=int(name.rsplit('q',1)[-1]);G=bank['extra']['bank'];Q=bank['extra']['qr_Q'];Rb=bank['extra']['qr_R'];cor=C[:,:q];AA=A
            coefficient=lambda w:w if linear else head(hp,w[:k])+cor@w[k:]
            derivative=lambda w:np.eye(q) if linear else np.concatenate((jacobian(hp,w[:k]),cor),axis=1)
            target=np.linalg.solve(Rb,Q.T@states[case,0].ravel())
            if not linear:
                residual=Rb@(coefficient(W[0])-target);J=Rb@derivative(W[0]);rn,gn=metric(residual,J)
                checks.append(dict(method=name,case=case,step='cold',residual_error=abs(rn-row['cold'][0]),gradient_error=abs(gn-row['cold'][3]),normalized_gradient=gn))
            for step in sorted({0,(len(W)-2)//2,len(W)-2}):
                old=coefficient(W[step]);new=coefficient(W[step+1]);mid=.5*(old+new);Jc=derivative(W[step+1])
                u=(G@mid).reshape(3,n,n,n);uh=tf(u);advection=extract(I.advective(uh,spec))
                gradients=np.stack([inv(1j*wave[d]*uh) for d in range(3)])
                # Independent derivative of -(u dot grad)u, batched only over
                # tangent directions, rather than the production rotational form.
                tangents=(.5*G@Jc).T.reshape(Jc.shape[1],3,n,n,n);dadv=[]
                for start in range(0,len(tangents),8):
                    du=tangents[start:start+8];duh=tf(du);value=np.zeros_like(du)
                    for d in range(3):
                        value-=du[:,d,None]*gradients[d][None]+u[d][None,None]*inv(1j*wave[d]*duh)
                    dadv.append(extract(tf(value)))
                dN=np.concatenate(dadv).T;dt=cfg['rom_dt'];nu=parameters[case,-1];den=1+.5*dt*nu*lam
                residual=(AA@(new-old)-dt*(advection-nu*lam*(AA@mid)))/den
                dlinear=AA@Jc;J=(dlinear-dt*dN+.5*dt*nu*lam[:,None]*dlinear)/den[:,None]
                rn,gn=metric(residual,J)
                checks.append(dict(method=name,case=case,step=step+1,residual_error=abs(rn-row['steps'][0][step]),gradient_error=abs(gn-row['steps'][3][step]),normalized_gradient=gn))
    result=dict(passed=not bad and all(r['residual_error']<1e-8 and r['gradient_error']<1e-8 for r in checks),
        applicable=True,scope=__doc__,hash_failures=bad,checks=checks)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
