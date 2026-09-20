"""Independent NumPy head derivatives and SciPy weak-Poisson state audit."""
import pickle
import numpy as np
from scipy.fft import dstn
from scipy.special import expit


def head(params,z):
    value=z;derivative=np.eye(len(z))
    for w,b in params['net'][:-1]:
        value=value@w+b;derivative=w.T@derivative
        sigmoid=expit(value);derivative=(sigmoid+value*sigmoid*(1-sigmoid))[:,None]*derivative
        value=value*sigmoid
    w,b=params['net'][-1]
    return value@w+b+z@params['skip'],w.T@derivative+params['skip'].T


def bank_nodes(params,rotation,n,indices):
    positions=np.array(np.unravel_index(indices,(n-1,)*3)).T
    x=(positions+1)/n;angles=2*np.pi*x@params['freq']
    value=np.concatenate((np.sin(angles),np.cos(angles)),axis=1)
    for w,b in params['net'][:-1]:value=(value@w+b)*expit(value@w+b)
    w,b=params['net'][-1];value=value@w+b
    value*=np.asarray(params['scale'])*64*np.prod(x*(1-x),axis=1)[:,None]
    return value@rotation


def audit(out,record):
    import re
    saved_bank=pickle.loads((out/'bank.pkl').read_bytes());setups={};heads={}
    maximum=dict(bank_node_defect=0.,weak_operator_defect=0.,field_relative_defect=0.,
        coefficient_relative_defect=0.,full_gradient_absolute_defect=0.,eliminated_gradient_absolute_defect=0.)
    checked=0
    for mesh in record['meshes']:
        n=mesh['intervals'];z=np.load(out/f'weak_setup_N{n}.npz')
        bank=z['bank'];a=z['operator'];triples=z['triples'];lam=z['eigenvalues']
        indices=np.linspace(0,len(bank)-1,min(257,len(bank)),dtype=int)
        expected=bank_nodes(saved_bank['params'],saved_bank['rotation'],n,indices)
        defect=np.linalg.norm(expected-bank[indices])/np.linalg.norm(bank[indices]);maximum['bank_node_defect']=max(maximum['bank_node_defect'],float(defect));assert defect<1e-11
        independent=np.empty_like(a)
        for j in range(bank.shape[1]):
            transformed=dstn(bank[:,j].reshape((n-1,)*3),type=1,norm='ortho')/n**1.5
            independent[:,j]=transformed[tuple((triples-1).T)]
        defect=np.linalg.norm(a-independent)/np.linalg.norm(a);maximum['weak_operator_defect']=max(maximum['weak_operator_defect'],float(defect));assert defect<1e-11
        expected_lam=np.sum(4*n*n*np.sin(np.pi*triples/(2*n))**2,axis=1)
        assert np.allclose(lam,expected_lam,rtol=1e-13,atol=1e-13)
        setups[n]=(bank,independent,triples,lam)
    for row in record['invocations']:
        if not row['method'].startswith('nmrom_') or 'field_file' not in row:continue
        match=re.fullmatch(r'nmrom_K(\d+)_q(\d+)_dense',row['method']);assert match,'state audit currently covers dense panels only'
        k,q=map(int,match.groups());n=row['intervals'];bank,a,triples,lam=setups[n]
        if k not in heads:heads[k]=pickle.loads((out/f'head_K{k}.pkl').read_bytes())
        model=heads[k];saved=np.load(out/row['field_file']);coef=saved['coefficients'];latent=saved['latent']
        h,jac=head(model['params'],latent);directions=model['directions'][:,:q]
        forcing=np.load(out/'fields'/f'N{n}_case{row["case"]}_reference.npz')['forcing']
        target=dstn(forcing,type=1,norm='ortho')[tuple((triples-1).T)]/n**1.5/lam
        correction=a@directions
        if q:
            qq,rr=np.linalg.qr(correction,mode='reduced');y=np.linalg.solve(rr,qq.T@(target-a@h))
            ap=a-qq@(qq.T@a);tp=target-qq@(qq.T@target)
        else:y=np.zeros(0);ap=a;tp=target
        expected=h+directions@y
        defect=np.linalg.norm(coef-expected)/max(np.linalg.norm(expected),1e-300)
        maximum['coefficient_relative_defect']=max(maximum['coefficient_relative_defect'],float(defect));assert defect<1e-10
        defect=np.linalg.norm(bank@coef-saved['prediction'].reshape(-1))/max(np.linalg.norm(saved['prediction']),1e-300)
        maximum['field_relative_defect']=max(maximum['field_relative_defect'],float(defect));assert defect<1e-11
        residual=a@coef-target;full=np.concatenate((a@jac,correction),axis=1)
        full_gradient=np.linalg.norm(full.T@residual)/max(np.linalg.norm(full),1e-30)/max(np.linalg.norm(target),1e-14)
        eliminated=np.linalg.norm((ap@jac).T@(ap@h-tp))/max(np.linalg.norm(ap@jac),1e-30)/max(np.linalg.norm(tp),1e-14)
        for key,value,column in [('full_gradient_absolute_defect',full_gradient,5),('eliminated_gradient_absolute_defect',eliminated,4)]:
            defect=abs(value-row['selected_stats'][column]);maximum[key]=max(maximum[key],float(defect));assert defect<1e-10,(key,defect)
        assert row['stationary']==bool(int(row['selected_stats'][2])==1 and full_gradient<=record['config']['lm_tolerance']+1e-12)
        checked+=1
    return dict(passed=True,checked_states=checked,checked_meshes=len(setups),maximum=maximum,
        scope='Independent analytic NumPy head/Jacobian, full-field coefficient reconstruction, sampled bank-coordinate identity and SciPy sine-transform weak operator/gradient; no global optimality claim.')
