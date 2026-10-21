"""Three-dimensional Poisson, exact weak correction elimination and controls."""
from __future__ import annotations
import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import common as C
import shared_rom as S


def source(n,p):
    return C.initial(jnp.asarray(C.coords(n)),jnp.asarray(p)).reshape((n-1,)*3)


@jax.jit
def solve_dst(f,lam):
    return C.dst3(C.dst3(f)/lam)


def dataset(n,parameters,continuum=False):
    lam=C.eigenvalues(n,continuum)
    fields=[]
    for p in parameters:fields.append(np.asarray(solve_dst(source(n,p),lam)))
    return np.stack(fields)


def error(pred,truth):
    a,b=np.asarray(pred).reshape(-1),np.asarray(truth).reshape(-1)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))


def assemble(bank,n,count):
    triples=C.modes(count,n)
    test=np.asarray(C.phi(jnp.asarray(C.coords(n)),jnp.asarray(triples)))/n**3
    lam=np.asarray(C.mode_eigenvalues(n,triples))
    operator=test.T@bank
    projection=test/lam[None,:]
    return test,operator,projection,lam,triples


def engine(model,bank,operator,projection,indices,q,cfg):
    """Solve weak PDE from nodal forcing; exact elimination of linear corrections."""
    directions=np.asarray(model['directions'])[:,:q]
    if q:
        qq,rr=np.linalg.qr(operator@directions,mode='reduced')
        sv=np.linalg.svd(rr,compute_uv=False)
        assert sv[-1]>sv[0]*1e-12,('correction rank deficiency',q,sv)
        reduced=operator-qq@(qq.T@operator)
    else:qq=np.zeros((len(operator),0));rr=np.zeros((0,0));reduced=operator
    p=jax.device_put(model['params']);codes=jnp.asarray(model['codes'])
    library=C.head(p,codes)@jnp.asarray(reduced).T
    fit=S.lm(C.head,cfg['lm_budget'],cfg['lm_tolerance'])

    @jax.jit
    def query(f,p,bank,a,ap,qq,rr,directions,projection,indices,library,codes):
        target=projection.T@f.reshape(-1)[indices]
        tp=target-qq@(qq.T@target)
        order=jnp.argsort(jnp.sum((library-tp)**2,axis=1))[:cfg['initial_starts']]
        zs,stats=jax.vmap(lambda z:fit(p,ap,tp,z))(codes[order])
        which=jnp.argmin(stats[:,3]);z=zs[which];h=C.head(p,z)
        y=jsl.solve_triangular(rr,qq.T@(target-a@h),lower=False) if q else jnp.zeros(0)
        coef=h+directions@y
        residual=a@coef-target
        jac=jnp.concatenate((a@jax.jacfwd(C.head,argnums=1)(p,z),a@directions),axis=1)
        fullgrad=jnp.linalg.norm(jac.T@residual)/(jnp.maximum(jnp.linalg.norm(jac),1e-30)*jnp.maximum(jnp.linalg.norm(target),1e-14))
        info=jnp.concatenate((stats[which],fullgrad[None],which[None],jnp.sum(stats[:,0])[None]))
        values=(bank@coef,info,coef,stats)
        return (*values,z) if cfg.get('retain_solver_states',False) else values
    args=(p,jnp.asarray(bank),jnp.asarray(operator),jnp.asarray(reduced),jnp.asarray(qq),jnp.asarray(rr),
          jnp.asarray(directions),jnp.asarray(projection),jnp.asarray(indices),library,codes)
    jax.block_until_ready(args)
    return lambda f:query(f,*args)


def linear_weak(bank,operator,projection):
    qq,rr=np.linalg.qr(operator,mode='reduced');sv=np.linalg.svd(rr,compute_uv=False)
    assert sv[-1]>sv[0]*1e-12,('weak bank rank',sv)
    @jax.jit
    def query(f,bank,qq,rr,projection):
        y=jsl.solve_triangular(rr,qq.T@(projection.T@f.reshape(-1)),lower=False)
        return bank@y
    args=tuple(jnp.asarray(v) for v in (bank,qq,rr,projection))
    return lambda f:query(f,*args),dict(condition_number=float(sv[0]/sv[-1]),solver='thin QR plus triangular solve')


def galerkin(bank,n):
    basis,_=np.linalg.qr(bank,mode='reduced')
    lap=np.asarray(jax.vmap(lambda v:C.negative_laplacian(v.reshape((n-1,)*3),n).reshape(-1),in_axes=1,out_axes=1)(jnp.asarray(basis)))
    operator=basis.T@lap
    assert np.max(np.abs(operator-operator.T))<1e-8
    chol=np.linalg.cholesky(operator)
    @jax.jit
    def query(f,basis,chol):
        y=jsl.cho_solve((chol,True),basis.T@f.reshape(-1))
        return basis@y
    b=jnp.asarray(basis);l=jnp.asarray(chol)
    return lambda f:query(f,b,l)


def pod_basis(fields,maxrank):
    u=np.asarray(fields).reshape(len(fields),-1)
    values,vectors=np.linalg.eigh(u@u.T);order=np.argsort(values)[::-1]
    values=values[order];vectors=vectors[:,order]
    available=int(np.count_nonzero(values>values[0]*1e-12));rank=min(maxrank,available)
    basis=u.T@vectors[:,:rank]/np.sqrt(values[:rank])[None,:]
    defect=float(np.max(np.abs(basis.T@basis-np.eye(rank))))
    assert defect<1e-6,defect
    return basis,dict(available_rank=available,retained_rank=rank,orthogonality=defect,training_hash=C.sha(u))


def verify():
    from scipy.fft import dstn
    from scipy.sparse import diags,eye,kron
    from scipy.sparse.linalg import spsolve
    n=8;x=C.coords(n);rng=np.random.default_rng(920401)
    f=rng.normal(size=(n-1,)*3)
    one=diags([-np.ones(n-2),2*np.ones(n-1),-np.ones(n-2)],[-1,0,1])*n*n
    ident=eye(n-1);sparse=kron(kron(one,ident),ident)+kron(kron(ident,one),ident)+kron(kron(ident,ident),one)
    u=np.asarray(solve_dst(jnp.asarray(f),C.eigenvalues(n)))
    independent=spsolve(sparse.tocsr(),f.reshape(-1)).reshape(f.shape)
    checks=dict(dst_scipy=error(C.dst3(jnp.asarray(f)),dstn(f,type=1,norm='ortho')),
                sparse_direct=error(u,independent),sparse_stencil=error(sparse@u.reshape(-1),C.negative_laplacian(jnp.asarray(u),n)),
                residual=error(C.negative_laplacian(jnp.asarray(u),n),f))
    triple=np.array([[1,2,3]]);mode=np.asarray(C.phi(jnp.asarray(x),jnp.asarray(triple)))[:,0].reshape(f.shape)
    manufactured=np.asarray(C.mode_eigenvalues(n,triple))[0]*mode
    checks['manufactured_discrete']=error(solve_dst(jnp.asarray(manufactured),C.eigenvalues(n)),mode)
    test,a,projection,lam,_=assemble(np.eye((n-1)**3)[:,:8],n,32)
    moments=projection.T@f.reshape(-1);exact=test.T@u.reshape(-1)
    checks['weak_inverse_identity']=error(moments,exact)
    assert max(checks.values())<1e-11,checks
    continuum_errors=[]
    for m in (16,32,64):
        x=C.coords(m);truth=np.prod(np.sin(np.pi*x*np.array([1,2,3])),axis=1).reshape((m-1,)*3)
        result=solve_dst(jnp.asarray(14*np.pi**2*truth),C.eigenvalues(m))
        continuum_errors.append(error(result,truth))
    ratios=np.asarray(continuum_errors[:-1])/continuum_errors[1:]
    assert np.all((ratios>3.9)&(ratios<4.2)),ratios
    # Actual nonlinear engine with a linear head: compare joint QR elimination
    # against a completely independent least-squares solve in its exact span.
    p=C.init_head(jax.random.PRNGKey(3),1,4,4)
    p['net']=jax.tree_util.tree_map(jnp.zeros_like,p['net']);p['skip']=jnp.array([[1.,0.,0.,0.]])
    model=dict(params=p,codes=jnp.array([[0.],[1.],[-1.]]),directions=np.eye(4)[:,[1,2,3,0]])
    operator=rng.normal(size=(12,4));target=operator@np.array([.7,.8,0.,0.])
    cfg=dict(lm_budget=80,lm_tolerance=1e-10,initial_starts=2)
    fn=engine(model,np.eye(4),operator,np.eye(12),np.arange(12),1,cfg)
    field,info,coef,_=fn(jnp.asarray(target))
    wanted=np.zeros(4);wanted[:2]=np.linalg.lstsq(operator[:,:2],target,rcond=None)[0]
    checks['actual_augmented_engine']=error(field,wanted)
    assert checks['actual_augmented_engine']<1e-8 and int(info[2])==1 and float(info[5])<1e-9,info
    linear,_=linear_weak(np.eye(4),operator,np.eye(12))
    checks['direct_QR_endpoint']=error(linear(jnp.asarray(target)),np.linalg.lstsq(operator,target,rcond=None)[0])
    checks.update(continuum_errors=continuum_errors,refinement_ratios=ratios.tolist(),passed=True)
    return checks
