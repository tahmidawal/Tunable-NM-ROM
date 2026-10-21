"""Independent NumPy audit of collected NS3D complete comparison outputs.

Imports no production solver/model modules. A pass validates the recorded
measurements; failed accuracy/stationarity outcomes remain in the results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.signal import resample
import ns3d_independent as I
from audit_pilot import independently_evaluate_bank,head,jacobian,rel,stats


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def leaves(value):
    if isinstance(value,dict):
        for v in value.values():yield from leaves(v)
    elif isinstance(value,(list,tuple)):
        for v in value:yield from leaves(v)
    elif isinstance(value,np.ndarray):yield value


def smooth_tests(n,m):
    cut=int(np.ceil(n/3))-1
    candidates=sorted((a*a+b*b+c*c,a,b,c) for a in range(-cut,cut+1)
       for b in range(-cut,cut+1) for c in range(-cut,cut+1)
       if a>0 or (a==0 and b>0) or (a==0 and b==0 and c>0))
    xyz=np.array(np.meshgrid(*([np.arange(n)/n]*3),indexing='ij'))
    columns=[]
    for square,a,b,c in candidates:
        wave=np.array([a,b,c],dtype=float);anchor=np.eye(3)[np.argmin(abs(wave))]
        first=np.cross(wave,anchor);first/=np.linalg.norm(first)
        for pol in (first,np.cross(wave/np.sqrt(square),first)):
            angle=2*np.pi*np.einsum('d,dxyz->xyz',wave,xyz)
            for fn in (np.cos,np.sin):
                columns.append((pol[:,None,None,None]*fn(angle)*np.sqrt(2/n**3)).ravel())
                if len(columns)==m:return np.stack(columns,1)
    raise ValueError('test count')


def audit(root,destination):
    root=Path(root);out=root/'output';report=json.loads((out/'result.json').read_text());cfg=report['config']
    result=dict(passed=True,source_commit=report['source_commit'],job_id=report['job_id'],checks={},
        scope='independent saved-field metrics, coordinate bank, reference refinement and sampled operator parity; no independent retraining or global optimum proof',
        limitations=['comparison02 has no latent history; per-step reported stopping records are audited for internal consistency only',
                     'timing over_5pct is a generic-key error and is omitted from derived timing summaries',
                     'opened development cohort, not final generalization evidence'])
    def gate(name,passed,**extra):
        result['checks'][name]=dict(passed=bool(passed),**extra);result['passed'] &= bool(passed)
        print(name,json.dumps(result['checks'][name]),flush=True)
        Path(destination).write_text(json.dumps(result,indent=2)+'\n')
    mismatches=[];count=0
    for line in (root/'OUTPUTS.sha256').read_text().splitlines():
        want,name=line.split(None,1);name=name.strip();count+=1
        if file_sha(root/name)!=want:mismatches.append(name)
    gate('output_checksums',not mismatches,count=count,mismatches=mismatches)
    stdout=(root/'logs'/f"{report['job_id']}.out").read_text();stderr=(root/'logs'/f"{report['job_id']}.err").read_text()
    forbidden=['captured-large-constant','out of memory','resource_exhausted','no space left','cudnn_status_alloc_failed']
    gate('backend_complete','jax_backend=gpu' in stdout and 'PILOT_EXIT=0' in stdout and report['complete']
        and report['x64'] and report['precision']=='highest' and not any(x in (stdout+stderr).lower() for x in forbidden),
        stderr_bytes=len(stderr),stderr_preserved=True,delay_kernel_warning_during_training='Delay kernel timed out' in stderr)
    with (out/'checkpoint.pkl').open('rb') as stream:checkpoint=pickle.load(stream)
    p=checkpoint['params'];G=checkpoint['extra']['bank'];Q=checkpoint['extra']['qr_Q'];Rb=checkpoint['extra']['qr_R'];n=cfg['n']
    bad_dtype=[str(a.dtype) for a in leaves(checkpoint) if a.dtype.kind in 'fc' and a.dtype not in (np.float64,np.complex128)]
    gate('checkpoint_f64',not bad_dtype,bad_dtypes=bad_dtype)
    computed=independently_evaluate_bank(p,n)
    parity=rel(computed,G);gate('coordinate_bank',parity<1e-11,relative=parity)
    qr_parity=rel(Q@Rb,computed);orth=rel(Q.T@Q,np.eye(Q.shape[1]))
    gate('bank_QR',qr_parity<1e-11 and orth<1e-11,relative=qr_parity,orthogonality=orth)
    del computed
    with np.load(out/'representation_fields.npz') as f:
        truth=f['truth'].reshape(-1,G.shape[0]);prediction=f['prediction'].reshape(truth.shape);z=f['z'];he=f['head_error'];be=f['bank_error'];C=f['correction_directions'];grad=f['normalized_gradient'];reason=f['reasons']
    h=head(p,z);reconstructed=h@G.T;projected=(truth@Q)@Q.T
    he_np=np.linalg.norm(reconstructed-truth,axis=1)/np.linalg.norm(truth,axis=1)
    be_np=np.linalg.norm(projected-truth,axis=1)/np.linalg.norm(truth,axis=1)
    gate('representation_fields',rel(reconstructed,prediction)<1e-11 and np.max(abs(he_np-he))<1e-11 and np.max(abs(be_np-be))<1e-11,
        prediction_relative=rel(reconstructed,prediction),head=stats(he_np),bank=stats(be_np))
    gate('representation_summaries',all(abs(report['head_fit'][key]-value)<1e-11 for key,value in stats(he_np).items())
        and all(abs(report['bank_floor'][key]-value)<1e-11 for key,value in stats(be_np).items()))
    target=np.linalg.solve(Rb,(truth@Q).T).T;grad_np=[]
    for zi,ci in zip(z,target):
        residual=Rb@(head(p,zi)-ci);J=Rb@jacobian(p,zi)
        grad_np.append(np.linalg.norm(J.T@residual)/(np.linalg.norm(J)*np.linalg.norm(residual)+1e-300))
    grad_np=np.asarray(grad_np)
    gate('representation_stationarity_records',np.max(abs(grad_np-grad))<1e-9 and np.all((reason!=4)|(grad_np<=1.01e-7)),
        max_gradient_disagreement=float(np.max(abs(grad_np-grad))),stationary_count=int(np.sum(reason==4)),maximum_gradient=float(max(grad_np)))
    gate('correction_geometry',rel((Rb@C).T@(Rb@C),np.eye(C.shape[1]))<1e-9,
        orthogonality=rel((Rb@C).T@(Rb@C),np.eye(C.shape[1])))
    reference_rows=[]
    for grid in (cfg['n'],cfg['fine_reference_n']):
        with np.load(out/f'reference_N{grid}'/'reference_fields.npz') as f:
            for case in range(cfg['verification_cases']):
                coarse=f[f'case{case}_coarse'];fine=f[f'case{case}_fine_time_half'];lift=coarse
                for axis in (-3,-2,-1):lift=resample(lift,2*grid,axis=axis)
                error=np.linalg.norm((lift-fine).reshape(6,-1),axis=1)/np.linalg.norm(fine[0])
                removed=np.linalg.norm((coarse-f[f'case{case}_advection_removed']).reshape(6,-1),axis=1)/np.linalg.norm(coarse[0])
                reference_rows.append(dict(n=grid,case=case,initial_normalized_error=error.tolist(),advection_removed_error=removed.tolist()))
    gate('physical_reference_refinement',max(max(r['initial_normalized_error']) for r in reference_rows)<=.005,rows=reference_rows)
    with np.load(out/'weak_operators.npz') as f:A=f['A'];T=f['T'];lam=f['lam']
    Phi=smooth_tests(n,len(lam));spec=I.setup(n);operator_rows=[]
    for case in (0,len(z)//2,len(z)-1):
        coeff=h[case];field=(G@coeff).reshape(3,n,n,n)
        nonlinear=I.physical(I.advective(I.transform(field),spec)).ravel()
        got=np.einsum('mjk,j,k->m',T,coeff,coeff,optimize=True);want=Phi.T@nonlinear
        operator_rows.append(dict(case=case,relative=rel(got,want)))
    gate('independent_weak_tensor',rel(A,Phi.T@G)<1e-10 and max(r['relative'] for r in operator_rows)<1e-9,
        test_projection_relative=rel(A,Phi.T@G),probes=operator_rows)
    del T,Phi,A,truth,reconstructed,projected,prediction
    galrows=[]
    with np.load(out/'pod.npz') as f:P=f['basis']
    for filename,basis in [('pod_galerkin.npz',P),('bank_galerkin.npz',Q)]:
        with np.load(out/filename) as f:L=f['L'];T=f['T']
        rng=np.random.default_rng(91923)
        for i in range(2):
            coeff=rng.normal(size=basis.shape[1]);field=(basis@coeff).reshape(3,n,n,n);spectrum=I.transform(field)
            want=basis.T@I.physical(I.advective(spectrum,spec)).ravel()
            got=np.einsum('ijk,j,k->i',T,coeff,coeff,optimize=True)
            want_lap=basis.T@I.physical(-spec[1]*spectrum).ravel()
            galrows.append(dict(tensor=filename,probe=i,nonlinear_relative=rel(got,want),laplacian_relative=rel(L@coeff,want_lap)))
        del T,L
    gate('independent_Galerkin_tensors',all(max(r['nonlinear_relative'],r['laplacian_relative'])<1e-9 for r in galrows),probes=galrows)
    rows=json.loads((out/'timing_rows.json').read_text());fail=[];field_count=0;derived={};maxmetric=0.;hist={}
    with np.load(out/'timing_references.npz') as f:same=f['same_grid'];fine=f['fine_grid']
    with np.load(out/'timed_fields.npz') as fields:
        keys=list(fields.keys());cache={}
        for key in keys:
            field=fields[key];field_count+=1
            parts=key.split('__');method=parts[0];case=int(parts[1][4:]);digest=array_sha(field)
            e=np.linalg.norm((field-same[case]).reshape(6,-1),axis=1)/np.linalg.norm(same[case,0]);lift=field
            for axis in (-3,-2,-1):lift=resample(lift,cfg['fine_reference_n'],axis=axis)
            physical=np.linalg.norm((lift-fine[case]).reshape(6,-1),axis=1)/np.linalg.norm(fine[case,0])
            cache[(method,case,digest)]=(e,physical,bool(np.isfinite(field).all()))
        for row in rows:
            found=cache.get((row['method'],row['case'],row['field_sha256']))
            if found is None:fail.append('missing invocation field');continue
            e,physical,finite=found;delta=max(np.max(abs(e-row['same_grid_errors'])),np.max(abs(physical-row['fine_grid_errors'])));maxmetric=max(maxmetric,float(delta))
            if delta>1e-10 or finite!=row['finite']:fail.append('metric/finite mismatch')
            if row['with_host_seconds']<row['gpu_seconds'] or row['gpu_seconds']<=0:fail.append('time order')
            if 'steps' in row:
                rn,it,rs,gn=map(np.asarray,row['steps']);hist.setdefault(row['method'],dict(cold_nonstationary=0,step_nonstationary=0))
                hist[row['method']]['cold_nonstationary']+=int(row['cold'][2]!=4)
                hist[row['method']]['step_nonstationary']+=int(np.sum(rs!=4))
                if np.any((rs==4)&(gn>1.01e-7)) or (row['cold'][2]==4 and row['cold'][3]>1.01e-7):fail.append('stationarity record inconsistency')
        for method in sorted({r['method'] for r in rows}):
            arm=[r for r in rows if r['method']==method];times=np.array([r['gpu_seconds'] for r in arm]);med=np.median(times)
            derived[method]=dict(invocations=len(arm),device_ms_median=float(med*1000),timing_outliers_above_three_median=int(np.sum(times>3*med)),
                worst_evolved_same_grid=max(max(r['same_grid_errors'][1:]) for r in arm),worst_evolved_fine_grid=max(max(r['fine_grid_errors'][1:]) for r in arm),
                accuracy_failures_over_5pct=sum(max(r['fine_grid_errors'][1:])>.05 for r in arm),nonstationarity=hist.get(method))
    gate('timed_fields_and_records',not fail,checked_fields=field_count,checked_invocations=len(rows),maximum_metric_absolute_disagreement=maxmetric,failures=fail)
    result['derived_summary']=derived;Path(destination).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True);a=p.parse_args()
    raise SystemExit(0 if audit(a.collected,a.out)['passed'] else 2)
