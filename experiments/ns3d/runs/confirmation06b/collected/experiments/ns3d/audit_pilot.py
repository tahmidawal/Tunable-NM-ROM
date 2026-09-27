"""Independent NumPy field/model/reference audit of a checksum-collected pilot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.signal import resample
from scipy.special import expit
import ns3d_independent as I


def rel(a,b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b),1e-300))


def error(pred,true):
    a=(pred-true).reshape(len(true),-1)
    return np.linalg.norm(a,axis=1)/np.linalg.norm(true.reshape(len(true),-1),axis=1)


def stats(a):
    return dict(mean=float(np.mean(a)),median=float(np.median(a)),worst=float(np.max(a)))


def mlp(layers,x):
    for w,b in layers[:-1]:
        x=x@w+b;x=x*expit(x)
    w,b=layers[-1]
    return x@w+b


def head(params,z):
    return mlp(params['h'],z)+z@params['h_lin']


def jacobian(params,z):
    x=z.copy();J=np.eye(len(z))
    for w,b in params['h'][:-1]:
        x=x@w+b
        sig=expit(x)
        J=(J@w)*(sig+x*sig*(1-sig))[None,:]
        x=x*sig
    return (J@params['h'][-1][0]+params['h_lin']).T


def independently_evaluate_bank(params,n):
    x=np.arange(n)/n
    coordinates=np.stack(np.meshgrid(x,x,x,indexing='ij'),axis=-1).reshape(-1,3)
    blocks=[]
    for start in range(0,len(coordinates),4096):
        angles=2*np.pi*coordinates[start:start+4096]@params['B']
        features=np.concatenate((np.sin(angles),np.cos(angles)),axis=-1)
        blocks.append(params['out_scale']*mlp(params['g'],features))
    raw=np.concatenate(blocks)
    rank=raw.shape[1]//3
    fields=raw.reshape(n,n,n,3,rank).transpose(4,3,0,1,2)
    result=[]
    for field in fields:
        result.append(I.physical(I.solenoidal(I.transform(field),I.setup(n))).ravel())
    return np.stack(result,axis=1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('collected')
    args=parser.parse_args();root=Path(args.collected);out=root/'output'
    report=json.loads((out/'result.json').read_text());cfg=report['config']
    audit=dict(passed=True,job_id=report['job_id'],source_commit=report['source_commit'],checks={})
    def gate(name,passed,**values):
        audit['checks'][name]=dict(passed=bool(passed),**values)
        audit['passed'] &= bool(passed)
        print(name,json.dumps(audit['checks'][name]),flush=True)
    hashes=[]
    for line in (root/'OUTPUTS.sha256').read_text().splitlines():
        want,name=line.split(None,1);name=name.strip()
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        hashes.append(actual==want)
    gate('all_output_hashes',all(hashes),count=len(hashes))
    stdout=(root/'logs'/f"{report['job_id']}.out").read_text()
    stderr=(root/'logs'/f"{report['job_id']}.err").read_text()
    forbidden=['captured-large-constant','out of memory','resource_exhausted','no space left','cudnn_status_alloc_failed']
    gate('backend_and_completion','jax_backend=gpu' in stdout and 'PILOT_EXIT=0' in stdout
         and report['complete'] and report['x64'] and report['precision']=='highest'
         and not any(word in (stdout+stderr).lower() for word in forbidden),stderr_bytes=len(stderr))
    with (out/'checkpoint.pkl').open('rb') as stream:
        checkpoint=pickle.load(stream)
    params=checkpoint['params'];G=checkpoint['extra']['bank'];n=cfg['n']
    independent=independently_evaluate_bank(params,n)
    bank_parity=rel(independent,G)
    gate('independent_coordinate_bank',bank_parity<1e-11,relative=bank_parity)
    Q,R=np.linalg.qr(independent,mode='reduced')
    with np.load(out/'representation_fields.npz') as fields:
        truth=fields['truth'];pred=fields['prediction'];z=fields['z'];reasons=fields['reasons']
        recorded_head_error=fields['head_error'];recorded_bank_error=fields['bank_error']
        C=fields['correction_directions'];P=fields['pod_basis']
    x=truth.reshape(len(truth),-1)
    head_np=head(params,z)@independent.T
    h_error=error(head_np,x)
    bank_np=(x@Q)@Q.T
    b_error=error(bank_np,x)
    gate('independent_reconstruction',rel(head_np,pred.reshape(len(truth),-1))<1e-11
         and rel(h_error,recorded_head_error)<1e-11 and rel(b_error,recorded_bank_error)<1e-11,
         prediction_relative=rel(head_np,pred.reshape(len(truth),-1)),head=stats(h_error),bank=stats(b_error))
    gate('reported_summaries',all(abs(report['head_fit'][k]-v)<1e-11 for k,v in stats(h_error).items())
         and all(abs(report['bank_floor'][k]-v)<1e-11 for k,v in stats(b_error).items()))
    ctarget=np.linalg.solve(R,(x@Q).T).T
    gradients=[]
    for zi,ci in zip(z,ctarget):
        residual=R@(head(params,zi)-ci)
        J=R@jacobian(params,zi)
        gradients.append(float(np.linalg.norm(J.T@residual)/(np.linalg.norm(J)*np.linalg.norm(residual)+1e-300)))
    gate('independent_stationarity',max(gradients)<1.1e-7 and np.all(reasons==4),
         maximum_normalized_gradient=max(gradients))
    orth=rel((R@C).T@(R@C),np.eye(C.shape[1]))
    gate('correction_geometry',orth<1e-10,relative_orthogonality=orth)
    pod_errors={}
    for dim in cfg['pod_ranks']:
        basis=P[:,:dim]
        value=error((x@basis)@basis.T,x)
        pod_errors[str(dim)]=stats(value)
    gate('independent_POD_floors',all(abs(report['pod_floors'][d][k]-v)<1e-10
         for d,values in pod_errors.items() for k,v in values.items()),floors=pod_errors)
    with np.load(out/'train_data.npz') as train:
        training=train['states'].reshape(-1,G.shape[0])
    train_free=error((training@Q)@Q.T,training)
    train_pod=error((training@P)@P.T,training)
    audit['training_decomposition']=dict(bank_free=stats(train_free),pod_full_rank=stats(train_pod),
                                          head=report['training_reconstruction'])
    print('training_decomposition',json.dumps(audit['training_decomposition']),flush=True)
    reference=[]
    for index in range(cfg['verification_cases']):
        with np.load(out/f'reference_N{n}'/'reference_fields.npz') as fields:
            coarse=fields[f'case{index}_coarse'];fine=fields[f'case{index}_fine_time_half']
            linear=fields[f'case{index}_advection_removed']
        lifted=coarse
        for axis in (-3,-2,-1):
            lifted=resample(lifted,2*n,axis=axis)
        physical=np.linalg.norm((lifted-fine).reshape(6,-1),axis=1)/np.linalg.norm(fine[0])
        removed=np.linalg.norm((coarse-linear).reshape(6,-1),axis=1)/np.linalg.norm(coarse[0])
        reference.append(dict(index=index,full_refined_field_error=physical.tolist(),advection_removed=removed.tolist()))
    gate('full_refined_field_budget',max(max(c['full_refined_field_error']) for c in reference)<=.005,
         cases=reference,worst=max(max(c['full_refined_field_error']) for c in reference))
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if not audit['passed']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
