"""Fixed-test weak heat evolution, exact correction elimination and cold EQ."""
import time
import numpy as np
import scipy.linalg
import scipy.optimize
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import common as C


def assemble(bank,n,m):
    x=C.coords(n);triples=C.modes(m,n);test=np.asarray(C.phi(jnp.asarray(x),jnp.asarray(triples)))/n**3
    a=test.T@bank
    return test,a,np.asarray(C.mode_eigenvalues(n,triples)),triples


def fit_quadrature(bank,test,decoded,cfg):
    """NNLS on decoder output times smooth tests; no PDE-residual snapshots."""
    rng=np.random.default_rng(cfg['quadrature_seed']);count=min(cfg['quadrature_candidates'],len(bank))
    indices=np.sort(rng.choice(len(bank),count,replace=False))
    take=np.linspace(0,len(decoded)-1,min(cfg['quadrature_decoder_snapshots'],len(decoded)),dtype=int)
    fields=np.asarray(decoded)[take]@bank.T
    exact=fields@test
    mat=(fields[:,indices,None]*test[None,indices,:]).transpose(0,2,1).reshape(-1,count)
    rhs=exact.reshape(-1)
    # Common row scaling by each test's decoder-output RMS, with a global floor.
    scale=np.maximum(np.sqrt(np.mean(exact**2,axis=0)),np.max(np.abs(exact))*1e-5)
    scales=np.tile(scale,len(take));mat=mat/scales[:,None];rhs=rhs/scales
    rows=np.sort(rng.choice(len(rhs),min(cfg['quadrature_fit_rows'],len(rhs)),replace=False))
    begin=time.perf_counter()
    weights,rnorm=scipy.optimize.nnls(np.ascontiguousarray(mat[rows]),rhs[rows],maxiter=10*count)
    weighted=test[indices]*weights[:,None]
    recon=fields[:,indices]@weighted
    relative=np.linalg.norm(recon-exact,axis=1)/np.maximum(np.linalg.norm(exact,axis=1),1e-300)
    return indices,weighted,dict(candidate_count=count,positive_support=int(np.count_nonzero(weights>0)),
            source='decoder-output snapshots, NNLS positive weights, random candidate grid nodes',
            fit_rows=len(rows),seconds=time.perf_counter()-begin,scaled_fit_norm=float(rnorm),
            training_moment_error_max=float(max(relative)),weights_hash=C.sha(weights),indices_hash=C.sha(indices))


def lm(head_fn,budget,tol):
    def solve(params,matrix,target,z0):
        scale=jnp.maximum(jnp.linalg.norm(target),1e-14)
        def parts(z):
            r=(matrix@head_fn(params,z)-target)/scale
            d=jax.jacfwd(head_fn,argnums=1)(params,z);jac=matrix@d/scale
            return r,jac
        r,j=parts(z0)
        state=(z0,r,j,jnp.dot(r,r),jnp.float64(1e-4),jnp.int32(0),jnp.int32(0),jnp.int32(0))
        def condition(s):
            grad=jnp.linalg.norm(s[2].T@s[1])/jnp.maximum(jnp.linalg.norm(s[2]),1e-30)
            return (s[5]<budget)&(s[7]==0)&(grad>tol)
        def body(s):
            z,r,j,value,damping,attempts,accepted,_=s
            gram=j.T@j;system=gram+damping*jnp.diag(jnp.maximum(jnp.diag(gram),1e-12))
            dz=jnp.linalg.solve(system,-j.T@r);new=z+dz;rn,jn=parts(new);vn=jnp.dot(rn,rn)
            accept=jnp.isfinite(vn)&(vn<value)
            damping=jnp.where(accept,jnp.maximum(damping/3,1e-12),damping*10)
            reason=jnp.where(jnp.linalg.norm(dz)<1e-12*(1+jnp.linalg.norm(z)),2,jnp.where(damping>1e12,3,0)).astype(jnp.int32)
            return (jnp.where(accept,new,z),jnp.where(accept,rn,r),jnp.where(accept,jn,j),
                    jnp.where(accept,vn,value),damping,attempts+1,accepted+accept.astype(jnp.int32),reason)
        z,r,j,value,damping,attempts,accepted,reason=jax.lax.while_loop(condition,body,state)
        gradient=jnp.linalg.norm(j.T@r)/jnp.maximum(jnp.linalg.norm(j),1e-30)
        reason=jnp.where(gradient<=tol,1,reason)
        return z,jnp.array([attempts,accepted,reason,jnp.sqrt(value),gradient])
    return solve


def engine(model,bank,a,lam,projection,indices,q,cfg,dt=None):
    dt=cfg['dt'] if dt is None else dt
    directions=np.asarray(model['directions'])[:,:q]
    l=a@directions
    if q:
        qq,rr=np.linalg.qr(l,mode='reduced');sv=np.linalg.svd(rr,compute_uv=False)
        assert sv[-1]>sv[0]*1e-12,('correction rank',q,sv)
        ap=a-qq@(qq.T@a)
    else:qq=np.zeros((a.shape[0],0));rr=np.zeros((0,0));ap=a
    params=model['params'];codes=jnp.asarray(model['codes']);library=C.head(params,codes)@jnp.asarray(ap).T
    fit=lm(C.head,cfg['lm_budget'],cfg['lm_tolerance'])
    nsteps=round(cfg['times'][-1]/dt);stride=round((cfg['times'][1]-cfg['times'][0])/dt)
    assert np.allclose(np.arange(len(cfg['times']))*stride*dt,cfg['times'])

    @jax.jit
    def query(u0,params,bank,a,ap,qq,rr,directions,projection,indices,library,codes,lam):
        target=projection.T@u0.reshape(-1)[indices]
        projected=target-qq@(qq.T@target)
        order=jnp.argsort(jnp.sum((library-projected)**2,axis=1))[:cfg['initial_starts']]
        zs,stats=jax.vmap(lambda z:fit(params,ap,projected,z))(codes[order])
        which=jnp.argmin(stats[:,3]);z=zs[which]
        def recover(z,target):
            h=C.head(params,z)
            y=jsl.solve_triangular(rr,qq.T@(target-a@h),lower=False) if q else jnp.zeros(0)
            return h+directions@y
        coef=recover(z,target);initial_coef=coef
        factor=(1-dt*cfg['diffusivity']*lam/2)/(1+dt*cfg['diffusivity']*lam/2)
        def step(carry,_):
            zprev,previous_coef=carry
            target=factor*(a@previous_coef)
            tprojected=target-qq@(qq.T@target)
            znew,info=fit(params,ap,tprojected,zprev);coef=recover(znew,target)
            residual=a@coef-target;d=jax.jacfwd(C.head,argnums=1)(params,znew)
            fullj=jnp.concatenate((a@d,a@directions),axis=1)
            fullgrad=jnp.linalg.norm(fullj.T@residual)/(jnp.maximum(jnp.linalg.norm(fullj),1e-30)*jnp.maximum(jnp.linalg.norm(target),1e-14))
            return (znew,coef),(coef,jnp.concatenate((info,fullgrad[None])))
        _,(all_coef,infos)=jax.lax.scan(step,(z,coef),None,length=nsteps)
        saved=jnp.concatenate((initial_coef[None],all_coef[stride-1::stride]))
        return saved@bank.T,stats,infos,saved
    args=(params,jnp.asarray(bank),jnp.asarray(a),jnp.asarray(ap),jnp.asarray(qq),jnp.asarray(rr),jnp.asarray(directions),
          jnp.asarray(projection),jnp.asarray(indices),library,codes,jnp.asarray(lam))
    return lambda u0:query(u0,*args)


def linear_weak(bank,a,lam,test,cfg):
    left=np.linalg.pinv(a);factor=(1-cfg['dt']*cfg['diffusivity']*lam/2)/(1+cfg['dt']*cfg['diffusivity']*lam/2)
    step=left@(factor[:,None]*a)
    maps=np.stack([np.linalg.matrix_power(step,round(t/cfg['dt'])) for t in cfg['times']])
    generator=-cfg['diffusivity']*left@(lam[:,None]*a)
    eig=np.linalg.eigvals(generator)
    assert np.max(eig.real)<1e-8, ('unstable linear weak generator',eig)
    exact=np.stack([scipy.linalg.expm(t*generator) for t in cfg['times']])
    @jax.jit
    def query(u,bank,a_inv,projection,maps):
        coef=a_inv@(projection.T@u.reshape(-1))
        return (maps@coef)@bank.T
    base=(jnp.asarray(bank),jnp.asarray(left),jnp.asarray(test))
    return dict(linear_bank_cn=lambda u:query(u,*base,jnp.asarray(maps)),
                linear_bank_exact=lambda u:query(u,*base,jnp.asarray(exact)))


def pod_models(train_fields,n,ranks,cfg):
    u=np.asarray(train_fields).reshape(-1,(n-1)**3)
    # Snapshot Gram keeps the SVD on the smaller sample axis; no neural-bank substitution.
    gram=u@u.T;values,vectors=np.linalg.eigh(gram);order=np.argsort(values)[::-1]
    values=values[order];vectors=vectors[:,order]
    valid=values>values[0]*1e-12;limit=int(np.count_nonzero(valid))
    maxrank=min(max(ranks),limit)
    basis=u.T@vectors[:,:maxrank]/np.sqrt(values[:maxrank])[None,:]
    lap=np.asarray(jax.vmap(lambda v:C.negative_laplacian(v.reshape((n-1,)*3),n).reshape(-1),in_axes=1,out_axes=1)(jnp.asarray(basis)))
    operator=basis.T@lap
    @jax.jit
    def query(u,basis,maps):
        y=basis.T@u.reshape(-1)
        return (maps@y)@basis.T
    out={}
    for rank in ranks:
        r=min(rank,maxrank)
        if r!=rank:continue
        maps=np.stack([scipy.linalg.expm(-cfg['diffusivity']*t*operator[:r,:r]) for t in cfg['times']])
        b=jnp.asarray(basis[:,:r]);p=jnp.asarray(maps)
        out[f'pod{rank}_exact']=lambda u,b=b,p=p:query(u,b,p)
    return out,dict(available_rank=limit,training_hash=C.sha(u),orthogonality=float(np.max(np.abs(basis.T@basis-np.eye(maxrank)))))


def bank_galerkin(bank,n,cfg):
    basis,_=np.linalg.qr(bank,mode='reduced')
    lap=np.asarray(jax.vmap(lambda v:C.negative_laplacian(v.reshape((n-1,)*3),n).reshape(-1),in_axes=1,out_axes=1)(jnp.asarray(basis)))
    operator=basis.T@lap
    assert np.max(np.abs(operator-operator.T))<1e-8
    eigen=np.linalg.eigvalsh(operator);assert min(eigen)>0
    maps=np.stack([scipy.linalg.expm(-cfg['diffusivity']*t*operator) for t in cfg['times']])
    @jax.jit
    def query(u,basis,maps):return (maps@(basis.T@u.reshape(-1)))@basis.T
    b=jnp.asarray(basis);p=jnp.asarray(maps)
    return lambda u:query(u,b,p)


def best_found_fields(model,bank,fields,cfg):
    """Multistart projection diagnostic using truth; never an online initializer."""
    basis,triangular=np.linalg.qr(bank,mode='reduced')
    targets=np.asarray(fields).reshape(-1,len(bank))@basis
    p=model['params'];codes=jnp.asarray(model['codes']);mat=jnp.asarray(triangular)
    library=C.head(p,codes)@mat.T;solve=lm(C.head,max(160,cfg['lm_budget']),min(1e-9,cfg['lm_tolerance']))
    @jax.jit
    def fit(targets,p,codes,matrix,library):
        def single(target):
            order=jnp.argsort(jnp.sum((library-target)**2,axis=1))[:4]
            zs,infos=jax.vmap(lambda z:solve(p,matrix,target,z))(codes[order])
            best=jnp.argmin(infos[:,3]);return C.head(p,zs[best]),infos[best]
        return jax.vmap(single)(targets)
    coef,stats=fit(jnp.asarray(targets),p,codes,mat,library)
    prediction=np.asarray(coef)@bank.T
    return prediction.reshape(fields.shape),np.asarray(stats).reshape(*fields.shape[:2],5)


def verify():
    cfg=dict(dt=.025,times=[0.,.1,.2],diffusivity=.02,lm_budget=60,lm_tolerance=1e-9,initial_starts=1)
    # Nontrivial exactly representable nonlinear head with zero linear correction.
    h=lambda p,z:z+.1*z**3
    solve=jax.jit(lm(h,80,1e-10));matrix=jnp.array([[1.],[.5],[.25],[.125]])
    z,info=solve({},matrix,matrix[:,0]*.7,jnp.array([.1]))
    assert abs(float(h({},z)[0])-.7)<1e-8 and int(info[2])==1
    a=np.eye(4);bank=np.eye(4);lam=np.arange(1,5,dtype=float)
    methods=linear_weak(bank,a,lam,np.eye(4),cfg)
    result=np.asarray(methods['linear_bank_cn'](jnp.ones(4)))
    fac=(1-cfg['dt']*.02*lam/2)/(1+cfg['dt']*.02*lam/2)
    expected=np.stack([fac**round(t/cfg['dt']) for t in cfg['times']])
    error=float(np.max(np.abs(result-expected)))
    assert error<1e-12 and np.linalg.norm(result[-1]-result[1])>1e-3
    # Exercise the ACTUAL augmented engine: nonzero corrections must survive
    # beyond step one. All equal eigenvalues make the independent answer exact.
    cfg.update(lm_tolerance=1e-10)
    p=C.init_head(jax.random.PRNGKey(1),1,4,4)
    p['net']=jax.tree_util.tree_map(jnp.zeros_like,p['net'])
    p['skip']=jnp.array([[1.,0.,0.,0.]])
    model=dict(params=p,codes=jnp.array([[.1],[.5],[1.]]),directions=np.eye(4)[:,[1,2,3,0]])
    amat=np.random.default_rng(920302).normal(size=(12,4));lams=np.full(12,14*np.pi**2)
    query=engine(model,np.eye(4),amat,lams,amat.T,np.arange(4),1,cfg)
    u0=jnp.array([.7,.8,0.,0.]);actual,initial_stats,step_stats,_=query(u0)
    factor=(1-cfg['dt']*.02*lams[0]/2)/(1+cfg['dt']*.02*lams[0]/2)
    wanted=np.stack([np.asarray(u0)*factor**round(t/cfg['dt']) for t in cfg['times']])
    augmented_error=float(np.max(np.abs(np.asarray(actual)-wanted)))
    assert augmented_error<1e-7,augmented_error
    assert np.all(np.asarray(step_stats)[:,2]==1),step_stats
    coarse=factor**round(cfg['times'][-1]/cfg['dt'])
    half=(1-cfg['dt']*.02*lams[0]/4)/(1+cfg['dt']*.02*lams[0]/4)
    fine=half**round(2*cfg['times'][-1]/cfg['dt']);truth=np.exp(-.02*lams[0]*cfg['times'][-1])
    refinement=abs(coarse-truth)/abs(fine-truth)
    assert 3.9<refinement<4.1,refinement
    return dict(nonlinear_fit_error=abs(float(h({},z)[0])-.7),linear_cn_error=error,
                augmented_carry_error=augmented_error,cn_refinement_ratio=refinement,
                advances_after_first_output=True,passed=True)
