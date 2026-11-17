"""Matched implicit Burgers solvers with preserved diagnostic states."""
import numpy as np
import jax
import jax.numpy as jnp
import engines as e


def make_fom(L, dt=.005, preconditioner='fft', max_newton=20):
    k=jnp.arange(1,L,dtype=jnp.float64)
    S=jnp.sqrt(2./L)*jnp.sin(jnp.pi*k[:,None]*k[None,:]/L) if preconditioner=='dense' else jnp.empty((0,0))
    lam=4*L**2*jnp.sin(jnp.pi*k/(2*L))**2
    def query(u0,nu,ntol,ltol,S,lam):
        def pre(v):
            if preconditioner=='fft':return e.helmholtz(v,nu,dt,L)
            a=S.T@v.reshape(L-1,L-1)@S
            return (S@(a/(1+dt*nu*(lam[:,None]+lam[None,:])))@S.T).reshape(-1)
        def step(prev,_):
            scale=jnp.maximum(jnp.linalg.norm(prev),1e-300)
            def body(state):
                u,it,rn,lres=state;r=e.residual(u,prev,nu,dt,L)
                def jv(v):return jax.jvp(lambda q:e.residual(q,prev,nu,dt,L),(u,),(v,))[1]
                delta,_=jax.scipy.sparse.linalg.bicgstab(jv,-r,tol=ltol,maxiter=200,M=pre)
                lr=jnp.linalg.norm(jv(delta)+r)/jnp.maximum(jnp.linalg.norm(r),1e-300)
                cand=u+delta
                return cand,it+1,jnp.linalg.norm(e.residual(cand,prev,nu,dt,L)),lres.at[it].set(lr)
            state=(prev,jnp.int32(0),jnp.linalg.norm(e.residual(prev,prev,nu,dt,L)),jnp.zeros(max_newton))
            u,it,rn,lres=jax.lax.while_loop(lambda s:(s[2]>ntol*scale)&(s[1]<max_newton)&jnp.isfinite(s[2]),body,state)
            return u,(it,rn/scale,lres,prev)
        nsub=int(round(.05/dt))
        def block(u,_):
            u,(it,rn,lr,prev)=jax.lax.scan(step,u,None,length=nsub)
            return u,(e.output_field(u,L,L),it,rn,lr,e.output_field(prev[-1],L,L))
        initial=u0[1:-1,1:-1].reshape(-1)
        _,(f,it,rn,lr,p)=jax.lax.scan(block,initial,None,length=5)
        return jnp.concatenate((u0[None],f)),it.reshape(-1),rn.reshape(-1),lr.reshape(-1,max_newton),p
    return jax.jit(query),(S,lam)


def make_rom(params,L,dt,trust):
    """Same Gauss180 fit and original weak steps, adding all internal latents."""
    _,parts=e.make_gauss_rom(params,L,dt,trust,stall=.01,ic_budget=180,return_parts=True)
    K=params['h_lin'].shape[0]
    lm=e.make_lm(lambda z,p,nu,data:e.weak(z,p,nu,data,params,L,dt),K,30,trust,.01)
    def query(u0,nu,data,cold):
        z,icit,icreason,scale=parts['initialize'](u0,data,cold)
        def step(carry,_):
            z,zprev=carry;p=data[1]@e.sc.head(params,z);ze=z+(z-zprev)
            r0=jnp.linalg.norm(e.weak(z,p,nu,data,params,L,dt));re=jnp.linalg.norm(e.weak(ze,p,nu,data,params,L,dt))
            zi=jnp.where(jnp.isfinite(re)&(re<r0),ze,z)
            z2,rn,it,reason=lm(zi,(p,nu,data),1e-9*scale)
            return (z2,z),(z2,rn,it,reason)
        _,(zs,rn,it,reason)=jax.lax.scan(step,(z,z),None,length=int(round(.25/dt)))
        internal=jnp.concatenate((z[None],zs));Z=internal[::int(round(.05/dt))]
        return parts['decode'](Z,data),it,rn,reason,Z,icit,icreason,internal
    return jax.jit(query)


def verify():
    L=8;dt=.005;nu=.03;rng=np.random.default_rng(910701)
    x=rng.normal(size=(L+1,L+1))*.1;x[[0,-1]]=0;x[:,[0,-1]]=0
    outputs=[]
    for pre in ['fft','dense']:
        fun,a=make_fom(L,dt,pre);value=jax.tree_util.tree_map(np.asarray,fun(jnp.asarray(x),nu,1e-11,1e-9,*a))
        outputs.append(value);assert np.max(value[2])<=1e-11
        old,_=e.make_fom(L,dt);old=np.asarray(old(jnp.asarray(x),nu,1e-11,1e-9)[0])
        assert np.linalg.norm(value[0]-old)/np.linalg.norm(old)<1e-9
    parity=float(np.linalg.norm(outputs[0][0]-outputs[1][0])/np.linalg.norm(outputs[0][0]))
    assert parity<1e-10
    # Independent dense Jacobian Newton for the first backward-Euler step, mixed signs.
    n=L-1;u=x[1:-1,1:-1].ravel().copy();prev=u.copy()
    def residual(a):
        p=np.pad(a.reshape(n,n),1);c=p[1:-1,1:-1]
        adv=L*c*(np.where(c>0,c-p[:-2,1:-1],p[2:,1:-1]-c)+np.where(c>0,c-p[1:-1,:-2],p[1:-1,2:]-c))
        lap=L*L*(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*c)
        return a-prev+dt*(adv-nu*lap).ravel()
    for _ in range(7):
        eps=1e-6;J=np.stack([(residual(u+eps*np.eye(n*n)[j])-residual(u-eps*np.eye(n*n)[j]))/(2*eps) for j in range(n*n)],1)
        u-=np.linalg.solve(J,residual(u))
    _,step=e.make_fom(L,dt);truth=np.asarray(step(jnp.asarray(prev),nu,1e-12,1e-10)[0])
    dense_error=float(np.linalg.norm(truth-u)/np.linalg.norm(u));assert dense_error<1e-10
    return dict(fft_dense_trajectory_relative=parity,dense_fd_newton_relative=dense_error)
