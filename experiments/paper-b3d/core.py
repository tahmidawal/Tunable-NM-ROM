"""Current bank+head+correction decoder with an exact 3D weak residual.

Large arrays are explicit jit arguments. QR coordinates preserve the trained
spatial span; only the declared POD controls replace that learned span.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
import b3d_common as b3
from block_lm import make_block_lm


def operators_np(u, n, omit_z=False):
    v = np.asarray(u).reshape((n-2,)*3)
    p = np.pad(v, 1)
    adv = np.zeros_like(v)
    lap = np.zeros_like(v)
    for axis in range(2 if omit_z else 3):
        minus = [slice(1,-1)] * 3
        plus = [slice(1,-1)] * 3
        minus[axis] = slice(None,-2)
        plus[axis] = slice(2,None)
        lo, hi = p[tuple(minus)], p[tuple(plus)]
        adv += v * np.where(v > 0, v-lo, hi-v) * (n-1)
        lap += (hi - 2*v + lo) * (n-1)**2
    return adv.ravel(), lap.ravel()


def verify(params, codes, n=9):
    rng = np.random.default_rng(20260920)
    u = rng.uniform(.2,1.,(n-2)**3) * rng.choice([-1.,1.],(n-2)**3)
    d = rng.normal(size=u.size)
    a,l = operators_np(u,n)
    aj,lj = np.asarray(b3.upwind_adv_field_3d(jnp.asarray(u),n)),np.asarray(b3.lap_3d(jnp.asarray(u),n))
    err_a = float(np.linalg.norm(a-aj)/np.linalg.norm(a))
    err_l = float(np.linalg.norm(l-lj)/np.linalg.norm(l))
    ad = np.asarray(jax.jvp(lambda x:b3.upwind_adv_field_3d(x,n),(jnp.asarray(u),),(jnp.asarray(d),))[1])
    eps = 1e-6
    fd = (operators_np(u+eps*d,n)[0]-operators_np(u-eps*d,n)[0])/(2*eps)
    derivative = float(np.linalg.norm(ad-fd)/np.linalg.norm(fd))
    negative = float(np.linalg.norm(a-operators_np(u,n,True)[0])/np.linalg.norm(a))
    xyz=rng.uniform(0,1,(17,3))
    g=np.asarray(b3.features(params,jnp.asarray(xyz)))
    gn=b3.features_np(params,xyz)
    bank = float(np.linalg.norm(g-gn)/np.linalg.norm(gn))
    z=np.asarray(codes[:7])
    h=np.asarray(b3.head(params,jnp.asarray(z)))
    hn=b3.head_np(params,z)
    head = float(np.linalg.norm(h-hn)/np.linalg.norm(hn))
    _,_,_,phi,lam=b3.test_modes_3d(n,32)
    lphi=np.stack([operators_np(phi[:,i],n)[1] for i in range(phi.shape[1])],axis=1)
    sine=float(np.linalg.norm(lphi+phi*lam)/np.linalg.norm(lphi))
    walls=xyz.copy();walls[:,0]=0.
    boundary=float(np.max(np.abs(np.asarray(b3.features(params,jnp.asarray(walls))))))
    result=dict(upwind_relative=err_a,laplacian_relative=err_l,jvp_relative=derivative,
                axis_omission_relative=negative,bank_numpy_relative=bank,head_numpy_relative=head,
                sine_laplacian_relative=sine,boundary_max=boundary)
    assert err_a<1e-12 and err_l<1e-12 and derivative<1e-7 and negative>.01,result
    assert bank<1e-11 and head<1e-11 and sine<1e-12 and boundary==0,result
    return result


def make_fit(k,q,budget=400,gtol=1e-6):
    """Truth-assisted coefficient diagnostic; exact field orthogonal residual retained."""
    def res(z,y,target,floor,C,R,hp):
        c=R@b3.head(hp,z)+C@y
        return jnp.concatenate((c-target,jnp.sqrt(jnp.maximum(floor,0.))[None]))
    lm=make_block_lm(res,k,q,budget,gtol=gtol)
    def one(target,floor,starts,C,R,hp):
        def run(z):
            y=C.T@(target-R@b3.head(hp,z))
            zz,yy,rn,it,reason,grad=lm(z,y,(target,floor,C,R,hp),0.)
            return jnp.concatenate((zz,yy)),rn,it,reason,grad
        out=jax.vmap(run)(starts)
        best=jnp.argmin(out[1])
        return tuple(x[best] for x in out),out
    return jax.jit(jax.vmap(one,in_axes=(0,0,0,None,None,None)))


def make_query(n,k,q,linear=False,steps=50,dt=.005,fit_budget=400,step_budget=200,gtol=1e-6):
    """Supplied dense interior initial field -> full dense interior trajectory.

    Linear endpoints have k=0 and q equal to their basis rank. Dynamic data:
    B (orthonormal physical bank), Phi, A, lam, C, R, Z and H (training starts).
    """
    def coeff(z,y,data,hp):
        return y if linear else data['R']@b3.head(hp,z)+data['C']@y
    def weak(z,y,prev,nu,data,hp):
        c=coeff(z,y,data,hp)
        ac=data['A']@c
        adv=b3.upwind_adv_field_3d(data['B']@c,n)
        return (ac-prev+dt*(data['Phi'].T@adv+nu*data['lam']*ac))/(1.+dt*nu*data['lam'])
    lm=make_block_lm(weak,k,q,step_budget,gtol=gtol)
    def initial(z,y,target,floor,data,hp):
        return jnp.concatenate((coeff(z,y,data,hp)-target,jnp.sqrt(jnp.maximum(floor,0.))[None]))
    iclm=make_block_lm(initial,k,q,fit_budget,gtol=gtol)

    def query(u0,nu,data,hp):
        target=data['B'].T@u0
        floor=jnp.linalg.norm(u0-data['B']@target)**2
        if linear:
            z=jnp.empty((0,),dtype=jnp.float64);y=target
            ic=(jnp.int32(0),jnp.int32(4),jnp.asarray(0.),jnp.sqrt(floor))
        else:
            idx=jnp.argmin(jnp.sum((data['H']-target[None,:])**2,axis=1))
            z0=data['Z'][idx]
            y0=data['C'].T@(target-data['R']@b3.head(hp,z0))
            z,y,rn,it,reason,grad=iclm(z0,y0,(target,floor,data,hp),0.)
            ic=(it,reason,grad,rn)
        w0=jnp.concatenate((z,y))
        def body(w,_):
            prev=data['A']@coeff(w[:k],w[k:],data,hp)
            zz,yy,rn,it,reason,grad=lm(w[:k],w[k:],(prev,nu,data,hp),0.)
            ww=jnp.concatenate((zz,yy))
            return ww,(ww,it,reason,grad,rn)
        _,(ws,its,reasons,grads,rns)=jax.lax.scan(body,w0,None,length=steps)
        states=jnp.concatenate((w0[None,:],ws),axis=0)
        cs=jax.vmap(lambda w:coeff(w[:k],w[k:],data,hp))(states)
        fields=cs@data['B'].T
        return fields,states,its,reasons,grads,rns,jnp.stack(ic)
    return jax.jit(query)


def data_for_basis(B,Phi,lam,C,R,Z,H):
    return {key:jnp.asarray(value,dtype=jnp.float64) for key,value in
            dict(B=B,Phi=Phi,A=Phi.T@B,lam=lam,C=C,R=R,Z=Z,H=H).items()}


def metrics(fields,reference):
    f=np.asarray(fields);ref=np.asarray(reference)
    initial_norm=max(float(np.linalg.norm(ref[0])),1e-300)
    err=np.linalg.norm(f-ref,axis=1)/initial_norm
    current=np.linalg.norm(f-ref,axis=1)/np.maximum(np.linalg.norm(ref,axis=1),1e-300)
    return dict(error_fixed_initial=err.tolist(),error_current_relative=current.tolist(),
                worst_all=float(err.max()),worst_evolved=float(err[1:].max()),
                initial_error=float(err[0]),finite=bool(np.isfinite(f).all()))


def make_fom(n,dt=.005,steps=50):
    """Tolerance-terminated BE with explicit time step for reference refinement."""
    axis=jnp.arange(1,n-1,dtype=jnp.float64)
    lam=4*(n-1)**2*jnp.sin(jnp.pi*axis/(2*(n-1)))**2
    lam=lam[:,None,None]+lam[None,:,None]+lam[None,None,:]
    def dst(x,axis):
        x=jnp.moveaxis(x,axis,-1);zero=jnp.zeros(x.shape[:-1]+(1,),dtype=x.dtype)
        odd=jnp.concatenate((zero,x,zero,-x[...,::-1]),axis=-1)
        return jnp.moveaxis(-jnp.fft.rfft(odd,axis=-1).imag[...,1:n-1]/jnp.sqrt(2*(n-1)), -1,axis)
    def dst3(x):
        for axis in range(3):x=dst(x,axis)
        return x
    def residual(u,prev,nu):return u-prev+dt*(b3.upwind_adv_field_3d(u,n)-nu*b3.lap_3d(u,n))
    def roll(u0,nu,ntol,ltol):
        def pre(v):return dst3(dst3(v.reshape((n-2,)*3))/(1+dt*nu*lam)).ravel()
        def step(prev,_):
            scale=jnp.maximum(jnp.linalg.norm(prev),1e-300)
            def body(state):
                u,it,rn=state;r=residual(u,prev,nu)
                jv=lambda v:jax.jvp(lambda v:residual(v,prev,nu),(u,),(v,))[1]
                du,_=jax.scipy.sparse.linalg.bicgstab(jv,-r,tol=ltol,maxiter=b3.LIN_MAXITER,M=pre)
                u=u+du
                return u,it+1,jnp.linalg.norm(residual(u,prev,nu))
            u,it,rn=jax.lax.while_loop(lambda s:(s[2]>ntol*scale)&(s[1]<b3.MAX_NEWTON),body,
                (prev,jnp.int32(0),jnp.linalg.norm(residual(prev,prev,nu))))
            return u,(u,it,rn/scale)
        _,(fields,it,rn)=jax.lax.scan(step,u0,None,length=steps)
        return jnp.concatenate((u0[None],fields)),it,rn
    return jax.jit(roll)


def make_fom_control(input_nodes, nodes, dt, final_time, output_steps):
    """Supplied fine-grid initial field to the common dense output contract.

    Spatial restriction, zero-Dirichlet trilinear reconstruction and temporal
    interpolation are all inside the timed query. The exact supplied initial
    field is returned at time zero, matching the operator contract.
    """
    steps=int(round(final_time/dt))
    assert abs(steps*dt-final_time)<1e-12
    assert (input_nodes-1)%(nodes-1)==0
    factor=(input_nodes-1)//(nodes-1)
    fom=make_fom(nodes,dt,steps)
    eye=np.eye(steps+1)
    weight=jnp.asarray(np.stack([np.interp(np.linspace(0,final_time,output_steps+1),
        np.linspace(0,final_time,steps+1),eye[:,j]) for j in range(steps+1)],axis=1))
    if nodes!=input_nodes:
        axis=jnp.linspace(0,nodes-1,input_nodes)[1:-1]
        coordinates=jnp.stack(jnp.meshgrid(axis,axis,axis,indexing='ij'))
    def query(u0,nu,ntol,ltol):
        initial=u0.reshape((input_nodes-2,)*3)
        initial=initial if factor==1 else initial[factor-1::factor,factor-1::factor,factor-1::factor]
        native,iterations,residuals=fom(initial.ravel(),nu,ntol,ltol)
        if nodes!=input_nodes:
            def reconstruct(u):
                padded=jnp.pad(u.reshape((nodes-2,)*3),1)
                return jax.scipy.ndimage.map_coordinates(padded,coordinates,order=1,mode='constant').ravel()
            knots=jax.vmap(reconstruct)(native)
        else:
            knots=native
        fields=(weight@knots).at[0].set(u0)
        return fields,native,iterations,residuals
    return jax.jit(query)
