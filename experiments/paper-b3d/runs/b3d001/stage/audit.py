"""Independent NumPy saved-field audit; imports neither JAX nor the driver."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np


def stencil(u,n):
    u=np.asarray(u).reshape((n-2,)*3);p=np.pad(u,1)
    acc=np.zeros_like(u);lap=np.zeros_like(u)
    for axis in range(3):
        low=[slice(1,-1)]*3;high=[slice(1,-1)]*3
        low[axis]=slice(0,-2);high[axis]=slice(2,None)
        left=p[tuple(low)];right=p[tuple(high)]
        acc+=u*np.where(u>0,u-left,right-u)*(n-1)
        lap+=(left+right-2*u)*(n-1)**2
    return acc.ravel(),lap.ravel()


def head(p,z):
    x=z.copy()
    for w,b in p['h'][:-1]:
        x=x@w+b;x=x/(1+np.exp(-x))
    w,b=p['h'][-1]
    return x@w+b+z@p['h_lin']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('--checkpoint',default='inputs/refined_checkpoint.pkl')
    args=parser.parse_args();root=Path(args.run);r=json.loads((root/'result.json').read_text());cfg=r['config']
    assert r['backend']=='gpu' and r['x64'] and r['precision']=='highest'
    assert hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()==r['checkpoint_sha256']
    ck=pickle.loads(Path(args.checkpoint).read_bytes());K=ck['cfg']['k']
    bases=np.load(root/'bases.npz');dirs=np.load(root/'directions.npz')['C']
    n=cfg['nodes'];dt=cfg['dt'];refs={};max_defect=0.;max_metric=0.;max_decode=0.
    for case in range(len(cfg['validation_rows'])):
        a=np.load(root/f'reference_case{case}.npz');f=a['fields'];refs[case]=f
        assert np.array_equal(f[0],a['u0'])
        for step in range(1,len(f)):
            adv,lap=stencil(f[step],n)
            res=f[step]-f[step-1]+dt*(adv-float(a['nu'])*lap)
            max_defect=max(max_defect,float(np.linalg.norm(res)/np.linalg.norm(f[step-1])))
        assert np.isfinite(f).all()
    for row in r['invocations']:
        a=np.load(root/row['artifact']);f=a['fields'];ref=refs[row['case']]
        assert f.shape==ref.shape and np.isfinite(f).all()
        err=np.linalg.norm(f-ref,axis=1)/np.linalg.norm(ref[0])
        discrepancy=float(np.max(np.abs(err-np.asarray(row['error_fixed_initial']))))
        max_metric=max(max_metric,discrepancy)
        assert abs(err.max()-row['worst_all'])<1e-12
        assert abs(err[1:].max()-row['worst_evolved'])<1e-12
        method=row['method']
        if method.startswith('rom_q'):
            q=int(method[5:]);w=a['states']
            coef=head(ck['params'],w[:,:K])@bases['bank_R'].T+w[:,K:]@dirs[:,:q].T
            decoded=coef@bases['bank'].T
        elif method.startswith('free_R'):
            decoded=a['states']@bases['bank'].T
            assert a['states'].shape[1]==ck['cfg']['r']
        elif method.startswith('pod_'):
            rank=int(method[4:]);decoded=a['states']@bases['pod'][:,:rank].T
        else:
            assert method.startswith('fom_');decoded=f
        max_decode=max(max_decode,float(np.linalg.norm(decoded-f)/max(np.linalg.norm(f),1e-300)))
        if 'gradients' in row:
            assert np.array_equal(a['gradients'],row['gradients'])
            assert np.array_equal(a['iterations'],row['iterations'])
            station=bool(np.all(a['gradients']<=cfg['gradient_tolerance']) and a['initial_fit'][2]<=cfg['gradient_tolerance'])
            assert station==row['stationary']
        assert row['gpu_ms']>0 and np.isfinite(row['gpu_ms'])
    assert max_defect<2e-9 and max_metric<1e-12 and max_decode<1e-10
    expected=(len(cfg['q_ladder'])+1+len(cfg['pod_ranks'])+3)*len(refs)*cfg['repetitions']
    if r['complete']:
        assert len(r['invocations'])==expected,(len(r['invocations']),expected)
    output=dict(passed=True,complete=bool(r['complete']),invocations=len(r['invocations']),expected_invocations=expected,
                max_reference_defect=max_defect,max_metric_discrepancy=max_metric,max_decode_discrepancy=max_decode,
                final_cohort_unopened=r['final_cohort_unopened'],comparison_scope=r['comparison_scope'])
    (root/'audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
