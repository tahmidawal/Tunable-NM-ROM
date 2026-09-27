"""Bounded GPU component checks; run with the repository jaxrun wrapper."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engines as e
import numpy as np
import jax
import jax.numpy as jnp


def run():
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    L=8;r=np.random.default_rng(39012);u=jnp.asarray(r.normal(size=(L-1)**2));nu=.03;dt=.005
    S=np.sqrt(2/L)*np.sin(np.pi*np.outer(np.arange(1,L),np.arange(1,L))/L)
    a=np.asarray(e.dst2(u.reshape(L-1,L-1)))
    dst_error=np.max(np.abs(a-S.T@np.asarray(u).reshape(L-1,L-1)@S))
    assert dst_error<1e-12
    v=e.helmholtz(u,nu,dt,L);_,lap=e.spatial(v,L)
    helm_error=float(jnp.linalg.norm(v-dt*nu*lap-u)/jnp.linalg.norm(u));assert helm_error<1e-12
    # Independent dense Newton with a materialized Jacobian, mixed-sign initial data.
    prev=u*.1;_,step=e.make_fom(L,dt,target=L)
    out,(it,rn)=step(prev,nu,1e-12,1e-10)
    dense=np.array(prev,copy=True)
    for _ in range(7):
        res=e.residual(jnp.asarray(dense),prev,nu,dt,L)
        J=jax.jacfwd(lambda x:e.residual(x,prev,nu,dt,L))(jnp.asarray(dense))
        dense-=np.linalg.solve(np.asarray(J),np.asarray(res))
    fom_error=np.linalg.norm(np.asarray(out)-dense)/np.linalg.norm(dense)
    assert fom_error<1e-10 and float(rn)<1e-11
    # Discrete sine weak diffusion identity and explicit mixed-sign sampled operator.
    Phi,lam,_=e.modes(L,16);adv,lap=e.spatial(u,L)
    weak_error=np.linalg.norm(Phi.T@np.asarray(lap)+lam*(Phi.T@np.asarray(u)))/np.linalg.norm(Phi.T@np.asarray(lap))
    assert weak_error<1e-12
    p=np.pad(np.asarray(u).reshape(L-1,L-1),1);ref=np.zeros_like(p)
    for i in range(1,L):
        for j in range(1,L):
            ux=(p[i,j]-p[i-1,j]) if p[i,j]>0 else (p[i+1,j]-p[i,j])
            uy=(p[i,j]-p[i,j-1]) if p[i,j]>0 else (p[i,j+1]-p[i,j])
            ref[i,j]=L*p[i,j]*(ux+uy)
    assert np.max(np.abs(np.asarray(adv)-ref[1:-1,1:-1].ravel()))<1e-12
    # Actual regularized normal equations at several condition numbers.
    gj=[]
    for kappa in [1.,1e4,1e8]:
        Q,_=np.linalg.qr(r.normal(size=(16,16)));A=Q@np.diag(np.geomspace(1,kappa,16))@Q.T;b=r.normal(size=16)
        x=np.asarray(e.gj_solve(jnp.asarray(A),jnp.asarray(b)));truth=np.linalg.solve(A,b)
        gj.append(float(np.linalg.norm(x-truth)/np.linalg.norm(truth)))
    assert max(gj)<1e-7
    lifted=np.asarray(e.output_field(u,L,2*L));assert np.array_equal(lifted[::2,::2],p)
    print(dict(jax_backend=jax.default_backend(),x64=True,dst_dense_abs=dst_error,
        helmholtz_relative=helm_error,fom_dense_newton_relative=fom_error,
        weak_diffusion_relative=weak_error,gj_relative=gj,checks_passed=7),flush=True)


if __name__=='__main__':run()
