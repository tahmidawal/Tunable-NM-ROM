"""Independent SciPy weak moments and NumPy analytic cold/evolution gradients."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from scipy.fft import dstn
from scipy.linalg import lstsq
from audit_head import head
from audit_coverage import bank_at


def audit(out,destination):
    out=Path(out);record=json.loads((out/'result.json').read_text());cfg=record['config']
    assert record['complete'] and cfg['retain_solver_states']
    saved=pickle.loads((out/'bank.pkl').read_bytes());checks=[];maxgrad=0.;maxcoef=0.
    for mesh in record['meshes']:
        n=mesh['intervals'];bank=bank_at(saved,n)
        numbers=np.arange(1,n);triples=np.stack(np.meshgrid(numbers,numbers,numbers,indexing='ij'),-1).reshape(-1,3)
        triples=triples[np.argsort(np.sum(triples**2,axis=1),kind='stable')[:cfg['weak_tests']]]
        indices=tuple((triples-1).T);lam=np.sum(4*n*n*np.sin(np.pi*triples/(2*n))**2,axis=1)
        a=dstn(bank.reshape(n-1,n-1,n-1,-1),type=1,norm='ortho',axes=(0,1,2))[indices]/n**1.5
        for row in record['invocations']:
            if row['intervals']!=n or row['repetition']!=0 or 'initial_stats' not in row:continue
            meta=mesh['methods'][row['method']];k=meta['k'];q=meta['q'];model=pickle.loads((out/f'head_K{k}.pkl').read_bytes())
            directions=model['directions'][:,:q];l=a@directions
            qq=np.linalg.qr(l,mode='reduced')[0] if q else np.zeros((len(a),0));ap=a-qq@(qq.T@a)
            fields=np.load(out/row['field_file']);truth=np.load(out/'fields'/f'N{n}_case{row["case"]}_reference.npz')['same_grid']
            if meta['cold_start']=='NNLS sampled':
                quadrature=np.load(out/f'eq_N{n}_K{k}.npz');target=quadrature['weighted_tests'].T@truth[0].reshape(-1)[quadrature['indices']]
            else:target=dstn(truth[0],type=1,norm='ortho')[indices]/n**1.5
            def one(z,target,coefficient=None):
                value,jac=head(model['params'],z);projected=target-qq@(qq.T@target)
                scale=max(np.linalg.norm(projected),1e-14);res=(ap@value-projected)/scale;j=ap@jac/scale
                gradient=np.linalg.norm(j.T@res)/max(np.linalg.norm(j),1e-30)
                reconstructed=value+directions@lstsq(l,target-a@value)[0] if q else value
                if coefficient is not None:np.testing.assert_allclose(reconstructed,coefficient,rtol=1e-8,atol=2e-10)
                fullj=np.concatenate((a@jac,l),axis=1);fullres=a@reconstructed-target
                fullgrad=np.linalg.norm(fullj.T@fullres)/(max(np.linalg.norm(fullj),1e-30)*max(np.linalg.norm(target),1e-14))
                return gradient,fullgrad,reconstructed
            for z,info in zip(fields['initial_latents'],fields['initial_stats']):
                grad,_,_=one(z,target);maxgrad=max(maxgrad,abs(grad-info[4]))
                np.testing.assert_allclose(grad,info[4],rtol=1e-5,atol=3e-10)
                if info[2]==1:assert grad<=cfg['lm_tolerance']+3e-10
            chosen=int(np.argmin(fields['initial_stats'][:,3]));_,_,coef=one(fields['initial_latents'][chosen],target,fields['initial_coefficient'])
            previous=fields['initial_coefficient'];dt=meta.get('dt',cfg['dt']);factor=(1-dt*cfg['diffusivity']*lam/2)/(1+dt*cfg['diffusivity']*lam/2)
            for z,coefficient,info in zip(fields['step_latents'],fields['step_coefficients'],fields['step_stats']):
                target=factor*(a@previous);grad,fullgrad,reconstructed=one(z,target,coefficient)
                maxcoef=max(maxcoef,float(np.max(np.abs(coefficient-reconstructed))))
                maxgrad=max(maxgrad,abs(grad-info[4]),abs(fullgrad-info[5]))
                np.testing.assert_allclose([grad,fullgrad],info[4:6],rtol=1e-5,atol=3e-10)
                if info[2]==1:assert grad<=cfg['lm_tolerance']+3e-10
                previous=coefficient
            stride=round((cfg['times'][1]-cfg['times'][0])/dt)
            saved_coef=np.concatenate((fields['initial_coefficient'][None],fields['step_coefficients'][stride-1::stride]))
            reconstructed=(saved_coef@bank.T).reshape(fields['prediction'].shape)
            np.testing.assert_allclose(reconstructed,fields['prediction'],rtol=1e-8,atol=2e-10)
            checks.append(dict(intervals=n,method=row['method'],case=row['case'],cold_starts=len(fields['initial_latents']),evolved_states=len(fields['step_latents'])))
    result=dict(passed=True,source_commit=record['source_commit'],job_id=record['job_id'],checks=checks,
        maximum_gradient_absolute_difference=maxgrad,maximum_coefficient_absolute_difference=maxcoef,
        independent_weak_moments='SciPy orthonormal DST, exact smooth sine-test coefficient extraction',
        final_cohort_opened=record['final_cohort_opened'])
    Path(destination).write_text(json.dumps(result,indent=2)+'\n');print('STATE_AUDIT_PASS',len(checks),maxgrad,maxcoef)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--destination',required=True);a=p.parse_args();audit(a.out,a.destination)
