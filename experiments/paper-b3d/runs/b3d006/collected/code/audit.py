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
    parser.add_argument('--audit-output',default='audit.json')
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
    checked_controls=set()
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
            if 'native_fields' in a.files and row['artifact'] not in checked_controls:
                checked_controls.add(row['artifact']);native=a['native_fields'];nn=row['nodes'];dd=row['dt']
                for step in range(1,len(native)):
                    adv,lap=stencil(native[step],nn)
                    residual=native[step]-native[step-1]+dd*(adv-float(np.load(root/f"reference_case{row['case']}.npz")['nu'])*lap)
                    relative=float(np.linalg.norm(residual)/np.linalg.norm(native[step-1]))
                    assert abs(relative-a['residuals'][step-1])<1e-11
                assert row['converged']==bool(np.max(a['residuals'])<=row['nonlinear_tolerance']*(1+1e-6))
                knots=native
                if nn!=n:
                    knots=np.pad(native.reshape((len(native),)+(nn-2,)*3),[(0,0),(1,1),(1,1),(1,1)])
                    positions=np.linspace(0,nn-1,n)[1:-1];low=np.floor(positions).astype(int);fraction=positions-low
                    for axis in (1,2,3):
                        shape=[1]*4;shape[axis]=len(positions);weight=fraction.reshape(shape)
                        knots=np.take(knots,low,axis=axis)*(1-weight)+np.take(knots,low+1,axis=axis)*weight
                    knots=knots.reshape(len(native),-1)
                weights=np.stack([np.interp(np.arange(cfg['steps']+1)*dt,np.arange(len(native))*dd,v)
                                  for v in np.eye(len(native))],axis=1)
                reconstructed=weights@knots;reconstructed[0]=ref[0]
                assert np.max(np.abs(reconstructed-f))<1e-11
        max_decode=max(max_decode,float(np.linalg.norm(decoded-f)/max(np.linalg.norm(f),1e-300)))
        if 'gradients' in row:
            assert np.array_equal(a['gradients'],row['gradients'])
            assert np.array_equal(a['iterations'],row['iterations'])
            station=bool(np.all(a['gradients']<=cfg['gradient_tolerance']) and a['initial_fit'][2]<=cfg['gradient_tolerance'])
            assert station==row['stationary']
        assert row['gpu_ms']>0 and np.isfinite(row['gpu_ms'])
    for row in r.get('refinement',[]):
        a=np.load(root/row['artifact']);f=a['fields'];nu=float(a['nu']);nn=row['nodes'];dd=row['dt']
        defect=0.
        for step in range(1,len(f)):
            adv,lap=stencil(f[step],nn)
            residual=f[step]-f[step-1]+dd*(adv-nu*lap)
            defect=max(defect,float(np.linalg.norm(residual)/np.linalg.norm(f[step-1])))
        assert defect<2e-9 and abs(defect-row['numpy_max_relative_residual'])<1e-12
        shaped=f.reshape((row['steps']+1,)+(nn-2,)*3)[::int(round(dt/dd))]
        reduced=shaped if nn==n else shaped[:,1::2,1::2,1::2]
        reduced=reduced.reshape(refs[row['case']].shape)
        assert np.array_equal(reduced,a['restricted'])
        err=np.linalg.norm(refs[row['case']]-reduced,axis=1)/np.linalg.norm(reduced[0])
        assert np.max(np.abs(err-np.asarray(row['error_fixed_initial'])))<1e-12
    assert max_defect<2e-9 and max_metric<1e-12 and max_decode<1e-10
    expected=(len(cfg['q_ladder'])+1+len(cfg['pod_ranks'])+len(cfg.get('fom_controls',[0,1,2]))+len(cfg.get('fom_variants',[])))*len(refs)*cfg['repetitions']
    if r['complete']:
        assert len(r['invocations'])==expected,(len(r['invocations']),expected)
    operator_count=0
    if 'operators' in cfg:
        observed=np.asarray(cfg['train_steps']);eye=np.eye(len(observed))
        weights=np.stack([np.interp(np.arange(cfg['steps']+1),observed,eye[:,j]) for j in range(len(observed))],axis=1)
        for row in r.get('operator_invocations',[])+r.get('interpolation_controls',[]):
            a=np.load(root/row['artifact']);f=a['fields'];ref=refs[row['case']]
            assert f.shape==ref.shape and np.isfinite(f).all()
            assert np.array_equal(a['knots'][0],ref[0])
            recomposed=weights@a['knots']
            assert np.linalg.norm(recomposed-f)/np.linalg.norm(f)<1e-12
            err=np.linalg.norm(f-ref,axis=1)/np.linalg.norm(ref[0])
            assert np.max(np.abs(err-np.asarray(row['error_fixed_initial'])))<1e-12
            assert abs(err[1:].max()-row['worst_evolved'])<1e-12
            if 'method' in row:
                assert row['gpu_ms']>0 and np.isfinite(row['gpu_ms']);operator_count+=1
            else:assert np.array_equal(a['knots'],ref[observed])
        if r.get('operator_complete'):
            assert operator_count==len(cfg['operators'])*len(refs)*cfg['repetitions']
            assert len(r['interpolation_controls'])==len(refs)
    output=dict(passed=True,complete=bool(r['complete']),invocations=len(r['invocations']),expected_invocations=expected,
                max_reference_defect=max_defect,max_metric_discrepancy=max_metric,max_decode_discrepancy=max_decode,
                final_cohort_unopened=r['final_cohort_unopened'],comparison_scope=r['comparison_scope'],
                operator_invocations=operator_count,operator_complete=r.get('operator_complete',False))
    (root/args.audit_output).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
