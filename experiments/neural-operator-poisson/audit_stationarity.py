"""Independent NumPy/analytic derivative replay of every recorded ROM answer."""
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.special import expit


def inspect(archive, diagnosis="diagnosis", validation="validation"):
    archive=Path(archive); out=archive/'out'
    result=json.loads((out/diagnosis/'result.json').read_text())
    cfg=result['config']; n=result['setup']['intervals']
    params=pickle.loads((archive/'source'/cfg['checkpoint']).read_bytes())['params']
    assert 'hB' not in params
    with np.load(archive/'source'/cfg['basis']) as a: C=a['coefficient_directions'][:,:32]
    x=np.arange(1,n,dtype=float)/n
    xx,yy=np.meshgrid(x,x,indexing='ij');xy=np.column_stack((xx.ravel(),yy.ravel()))
    def mlp(layers,a):
        for w,b in layers[:-1]:
            a=a@w+b; a=a*expit(a)
        w,b=layers[-1];return a@w+b
    bank=[]
    for start in range(0,len(xy),4096):
        loc=xy[start:start+4096]; ang=2*np.pi*(loc@params['B'])
        ff=np.concatenate((np.sin(ang),np.cos(ang)),axis=-1)
        bc=16*loc[:,0]*(1-loc[:,0])*loc[:,1]*(1-loc[:,1])
        bank.append((params['out_scale']*bc)[:,None]*mlp(params['g'],ff))
    G=np.concatenate(bank)
    modes=np.asarray(result['setup']['mode_indices'])-1
    I,J=modes.T
    maximum=int(modes.max())+1
    S=np.sqrt(2./n)*np.sin(np.pi*np.outer(np.arange(1,n),np.arange(1,maximum+1))/n)
    cubes=G.reshape(n-1,n-1,G.shape[-1])
    transformed=np.einsum('xa,xyr,yb->abr',S,cubes,S,optimize=True)
    B=transformed[I,J]
    eigen=4*n*n*(np.sin(np.pi*(I+1)/(2*n))**2+np.sin(np.pi*(J+1)/(2*n))**2)
    Q,R=np.linalg.qr(B@C,mode='reduced'); Bp=B-Q@(Q.T@B)
    def head(z):
        a=z.copy(); jac=np.eye(len(z))
        for w,b in params['h'][:-1]:
            a=a@w+b; jac=jac@w
            sigmoid=expit(a);jac=jac*(sigmoid*(1+a*(1-sigmoid)))[None,:]
            a=a*sigmoid
        w,b=params['h'][-1]
        return a@w+b+z@params['h_lin'],(jac@w+params['h_lin']).T
    index=json.loads((out/validation/'index.json').read_text()); cases={r['case_id']:r for r in index['records']}
    records=[]
    for row in result['rows']:
        if row['method']!='nmrom' or row['repetition']!=0:continue
        with np.load(out/validation/cases[row['case_id']]['path']) as a: source=a['input'][0]
        f=(S.T@source[1:-1,1:-1]@S)[I,J]/eigen
        z=np.asarray(row['latent']); y=np.asarray(row['correction_coefficients'])
        h,D=head(z); coefficients=h+C@y
        actual=np.pad((G@coefficients).reshape(n-1,n-1),1)
        with np.load(out/diagnosis/row['field_path']) as a: expected=a['field']
        parity=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        residual=B@coefficients-f; Jfull=B@D; Jlinear=B@C
        stationarity=float(np.linalg.norm(np.concatenate((Jfull.T@residual,Jlinear.T@residual)))/(np.sqrt(np.sum(Jfull*Jfull)+np.sum(Jlinear*Jlinear))*np.linalg.norm(residual)))
        rp=Bp@h-(f-Q@(Q.T@f)); jp=Bp@D
        reduced=float(np.linalg.norm(jp.T@rp)/(np.linalg.norm(jp)*np.linalg.norm(rp)))
        assert parity<1e-10
        np.testing.assert_allclose(stationarity,row['stationarity'],rtol=2e-4,atol=2e-10)
        np.testing.assert_allclose(reduced,row['reduced_stationarity'],rtol=2e-4,atol=2e-10)
        records.append(dict(case_id=row['case_id'],field_replay_relative_error=parity,
            full_stationarity=stationarity,reduced_stationarity=reduced,
            independently_stationary=bool(max(stationarity,reduced)<=cfg['stationarity_tolerance'])))
    assert len(records)==32
    return dict(independent_numpy_analytic_head_derivative_pass=True,records=records,
        invalid_stationarity_cases=sum(not r['independently_stationary'] for r in records),
        maximum_field_replay_relative_error=max(r['field_replay_relative_error'] for r in records))
