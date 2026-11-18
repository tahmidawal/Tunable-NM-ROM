"""Small algebra fixture: overlap-aware Hessian, zero transfer, known linear optimum."""
from common import *
import shared_child_head as H,decoder

def main():
    key=jax.random.PRNGKey(199);P,R,k=3,6,2
    A=jax.random.normal(key,(30,P*R),dtype=jnp.float64);S=A.T@A+.1*jnp.eye(P*R)
    params=dict(linear=jnp.eye(R,k),net=decoder.init_mlp(key,[k+3,8,R],0.))
    g=jnp.zeros((P,3));h=jnp.ones(P);scale=jnp.ones(R);mask=jnp.ones(P)
    ztrue=jnp.arange(P*k,dtype=jnp.float64).reshape(P,k)*.2
    c=H.head(params,ztrue,g,h,scale).reshape(-1);b=S@c;norm2=c@S@c+1.;constant=c@S@c
    J=jax.jacfwd(lambda z:H.head(params,z.reshape(P,k),g,h,scale).reshape(-1))(jnp.zeros(P*k))
    block=jax.vmap(jax.jacfwd(lambda z,gg,hh:H.head(params,z,gg,hh,scale)))(ztrue,g,h)
    Hess=jnp.einsum('prk,prqs,qsl->pkql',block,S.reshape(P,R,P,R),block).reshape(P*k,P*k)
    assert float(jnp.max(jnp.abs(Hess-J.T@S@J)))<1e-10
    fit=H.make_project(25);z,e,s,n=fit(S,b,constant,norm2,jnp.zeros_like(ztrue),params,g,h,scale,mask)
    assert float(jnp.max(jnp.abs(z-ztrue)))<1e-6,(z,e,s)
    assert float(e)<1e-7 and float(s)<1e-8
    assert float(jnp.max(jnp.abs(H.head(params,jnp.zeros_like(z),g,h,scale))))==0
    zd,ed,sd,nd=H.make_project(25,linear_solver='dense')(S,b,constant,norm2,jnp.zeros_like(ztrue),params,g,h,scale,mask)
    assert float(jnp.max(jnp.abs(zd-z)))<1e-6 and float(sd)<1e-8
    zc,ec,sc,nc=H.make_project(25,linear_solver='cholesky')(S,b,constant,norm2,jnp.zeros_like(ztrue),params,g,h,scale,mask)
    assert float(jnp.max(jnp.abs(zc-zd)))<1e-9 and float(sc)<1e-8
    print('PASS shared child physical LM / Hessian / zero transfer',float(e),float(s),flush=True)

if __name__=='__main__':main()
