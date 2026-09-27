"""Independent NumPy analytic-Jacobian audit of the worst saved stopping records.

One maximal recorded step gradient and one maximal initial-fit gradient per
method are recomputed; this is a declared sample, not an all-state certificate.
"""
import argparse,json,pickle
from pathlib import Path
import numpy as np


def head_jac(p,z):
    x=z.copy();jac=np.eye(len(z))
    for w,b in p['h'][:-1]:
        x=x@w+b;jac=jac@w;s=1/(1+np.exp(-x));jac*=s+x*s*(1-s);x=x*s
    w,b=p['h'][-1]
    return x@w+b+z@p['h_lin'],(jac@w+p['h_lin']).T


def advection_jac(u,d,n):
    u=u.reshape((n-2,)*3);d=d.reshape((n-2,)*3+(d.shape[-1],))
    up=np.pad(u,1);dp=np.pad(d,[(1,1)]*3+[(0,0)]);a=np.zeros_like(u);j=np.zeros_like(d)
    for ax in range(3):
        left=[slice(1,-1)]*3;right=left.copy();left[ax]=slice(0,-2);right[ax]=slice(2,None)
        lo=up[tuple(left)];hi=up[tuple(right)];dl=dp[tuple(left)];dh=dp[tuple(right)]
        a+=u*np.where(u>0,u-lo,hi-u)*(n-1)
        j+=np.where(u[...,None]>0,(2*u-lo)[...,None]*d-u[...,None]*dl,
                      (hi-2*u)[...,None]*d+u[...,None]*dh)*(n-1)
    return a.ravel(),j.reshape(-1,d.shape[-1])


def normalized_gradient(res,jac):
    return float(np.linalg.norm(jac.T@res)/(np.linalg.norm(jac)*np.linalg.norm(res)+1e-300))


def main():
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--checkpoint',required=True);args=p.parse_args()
    root=Path(args.out);r=json.loads((root/'result.json').read_text());cfg=r['config'];n=cfg['nodes']
    ck=pickle.loads(Path(args.checkpoint).read_bytes());basis=np.load(root/'bases.npz');C=np.load(root/'directions.npz')['C']
    axis=np.arange(1,n-1);l1=4*(n-1)**2*np.sin(np.pi*axis/(2*(n-1)))**2
    spectrum=(l1[:,None,None]+l1[None,:,None]+l1[None,None,:]).ravel();order=np.argsort(spectrum,kind='stable')[:len(basis['lam'])]
    lam=spectrum[order];assert np.max(np.abs(lam-basis['lam']))<1e-10
    modes=np.stack(np.unravel_index(order,(n-2,)*3),axis=1);sine=np.sqrt(2/(n-1))*np.sin(np.pi*np.outer(axis,axis)/(n-1))
    phi=np.stack([np.einsum('i,j,k->ijk',sine[:,a],sine[:,b],sine[:,c]).ravel() for a,b,c in modes],axis=1)
    checks=[]
    for method in sorted({v['method'] for v in r['invocations'] if 'gradients' in v}):
        rows=[v for v in r['invocations'] if v['method']==method]
        nonlinear=method.startswith('rom_q');rank=int(method.split('_')[-1]) if method.startswith('pod_') else ck['cfg']['r']
        B=basis['pod'][:,:rank] if method.startswith('pod_') else basis['bank'];A=phi.T@B
        q=int(method[5:]) if nonlinear else rank;k=ck['cfg']['k'] if nonlinear else 0
        def decode(w):
            if not nonlinear:return w,np.eye(rank)
            h,jh=head_jac(ck['params'],w[:k]);return basis['bank_R']@h+C[:,:q]@w[k:],np.concatenate((basis['bank_R']@jh,C[:,:q]),axis=1)
        worst=max(rows,key=lambda v:max(v['gradients']));a=np.load(root/worst['artifact']);step=int(np.argmax(worst['gradients']))+1
        ref=np.load(root/f"reference_case{worst['case']}.npz");nu=float(ref['nu']);coef,D=decode(a['states'][step]);prev,_=decode(a['states'][step-1])
        adv,der=advection_jac(B@coef,B@D,n);den=1+cfg['dt']*nu*lam
        residual=(A@(coef-prev)+cfg['dt']*(phi.T@adv+nu*lam*(A@coef)))/den
        jac=(A@D+cfg['dt']*(phi.T@der+nu*lam[:,None]*(A@D)))/den[:,None]
        grad=normalized_gradient(residual,jac);recorded=worst['gradients'][step-1]
        assert abs(grad-recorded)<1e-10,(method,grad,recorded)
        assert abs(np.linalg.norm(residual)-a['residuals'][step-1])<1e-9
        sv=np.linalg.svd(D,compute_uv=False)
        checks.append(dict(method=method,case=worst['case'],phase='evolution',step=step,gradient=grad,recorded=recorded,
                           absolute_discrepancy=abs(grad-recorded),coefficient_jacobian_sigma_min=float(sv[-1]),coefficient_jacobian_sigma_max=float(sv[0])))
        if nonlinear:
            worst=max(rows,key=lambda v:v['initial_fit'][2]);a=np.load(root/worst['artifact']);ref=np.load(root/f"reference_case{worst['case']}.npz")
            coef,D=decode(a['states'][0]);target=B.T@ref['u0'];floor=np.linalg.norm(ref['u0']-B@target)**2
            residual=np.r_[coef-target,np.sqrt(floor)];jac=np.vstack((D,np.zeros((1,D.shape[1]))));grad=normalized_gradient(residual,jac);recorded=worst['initial_fit'][2]
            assert abs(grad-recorded)<1e-10,(method,grad,recorded)
            checks.append(dict(method=method,case=worst['case'],phase='initial',step=0,gradient=grad,recorded=recorded,absolute_discrepancy=abs(grad-recorded)))
        print('AUDIT STATIONARITY',method,flush=True)
    out=dict(passed=True,scope='largest recorded step gradient and initial-fit gradient per method; repeated records deduplicated by value',
             checked_states=len(checks),max_gradient_discrepancy=max(x['absolute_discrepancy'] for x in checks),checks=checks)
    (root/'audit-stationarity-local.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
