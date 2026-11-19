"""Sub-minute fixtures for joint nonlinear coordinates and truth-free cold fitting."""
import argparse
from common import *
import discrete_weak as W,weak_quadrature as QW,coupled_child as C
import geometry,tensor_metric as TM,shared_child_head as H,decoder as D,operators


def setup():
    L=32;root=4;k=2;cells=geometry.uniform(1)
    basis=jnp.zeros((root,16,16)).at[jnp.arange(root),jnp.array([0,1,0,1]),jnp.array([0,0,1,1])].set(2.)
    G,T,_,_=TM.make_geometry(6)(basis,jnp.asarray(cells),*TM.rule(N=32))
    ca,slices=W.prepare(basis,cells,T,jnp.eye(36),L,100,G)
    q,qs=QW.prepare(basis,cells,T,jnp.eye(36),L,L)
    gram=QW.make_gram(qs)(q);ca=dict(ca,quadrature=q,input_gram=gram,gram=gram)
    fields=QW.make_fields(qs,36);field=lambda c,cache:fields(c,cache)[0]
    linear=jnp.eye(36,k)
    head=dict(linear=linear,net=D.init_mlp(jax.random.PRNGKey(1),[k+3,8,36],.01))
    encoder=dict(linear=linear,net=D.init_mlp(jax.random.PRNGKey(2),[39,8,k],0.))
    g,h=H.geometry(cells)
    ca.update(child_head=head,child_encoder=encoder,child_geometry=g,child_width=h,child_scale=jnp.ones(36)*.01)
    z=jnp.concatenate([jnp.array([.2,.02,-.01,.005]),.1*jnp.cos(jnp.arange(len(cells)*k))])
    return L,ca,field,z


def main():
    ap=argparse.ArgumentParser();ap.add_argument('fixture',choices=['coefficients','solver']);a=ap.parse_args()
    L,ca,field,z=setup();c=C.coefficients(z,ca)
    if a.fixture=='coefficients':
        E=jax.jacfwd(lambda cc:field(cc,ca))(jnp.zeros(len(c))).reshape(-1,len(c))
        discrepancy=E.T@E/L**2-ca['input_gram']
        # Whitening the polynomial tensor factors magnifies roundoff; compare the
        # entire mass matrix and physical energy, not individual entries at 1e-12.
        gram_relative=float(jnp.linalg.norm(discrepancy)/jnp.linalg.norm(ca['input_gram']))
        assert gram_relative<1e-9,gram_relative
        assert float(jnp.abs(c@discrepancy@c)/(c@ca['input_gram']@c))<1e-10
        J=jax.jacfwd(lambda zz:C.coefficients(zz,ca))(z)
        assert float(jnp.max(jnp.abs(J-C.coefficient_jacobian(z,ca))))<1e-13
        assert float(jnp.max(jnp.abs(ca['A']@J-C.tested_mass_jacobian(z,ca))))<1e-13
        u=field(c,ca);phi,lam,_=operators.modes(L,100);adv,lap=operators.spatial(u,L)
        prior=.99*(ca['A']@c);dt=.0025;nu=.005;norm=.2
        _,res=W.make_solver(field,L,coefficient_map=C.coefficients,tested_mass_jacobian=C.tested_mass_jacobian)
        expected=(phi.T@u.reshape(-1)/L-prior+dt*phi.T@(adv-nu*lap)/L)/(1+dt*nu*lam)/norm
        assert float(jnp.max(jnp.abs(res(z,prior,ca,dt,nu,norm)-expected)))<1e-12
        direction=jnp.sin(jnp.arange(len(z)));eps=1e-6
        jv=jax.jvp(lambda zz:res(zz,prior,ca,dt,nu,norm),(z,),(direction,))[1]
        fd=(res(z+eps*direction,prior,ca,dt,nu,norm)-res(z-eps*direction,prior,ca,dt,nu,norm))/(2*eps)
        assert float(jnp.linalg.norm(jv-fd)/jnp.linalg.norm(fd))<1e-7
        fit=C.make_initial_fit(field,maxiter=60);zz,error,station,accepted=fit(u,ca)
        assert float(error)<1e-6 and float(station)<1e-7,(error,station)
        print('PASS physical Gram, block Jacobians, independent stencil/JVP, cold nonlinear fit',gram_relative,float(error),float(station),flush=True)
    else:
        # A zero MLP gives an independently constructed restricted linear space.
        ca=dict(ca,child_head=dict(ca['child_head'],net=jax.tree.map(jnp.zeros_like,ca['child_head']['net'])))
        B=C.coefficient_jacobian(z,ca);linear_cache=dict(ca,A=ca['A']@B,gram=B.T@ca['gram']@B,restriction=B)
        restricted=lambda zz,cache:field(cache['restriction']@zz,cache)
        direct,_=W.make_solver(restricted,L,theta=.5)
        joint,_=W.make_solver(field,L,theta=.5,coefficient_map=C.coefficients,tested_mass_jacobian=C.tested_mass_jacobian)
        d,ds=direct(z,linear_cache,.0025,.005,1e-8,1e-10)
        n,ns=joint(z,ca,.0025,.005,1e-8,1e-10)
        assert float(ds[1])<1e-8 and float(ns[1])<1e-8,(ds,ns)
        assert float(jnp.linalg.norm(field(B@(d-n),ca)))<1e-10
        # Then restore a genuinely nonlinear head and require a stationary step.
        _,nonlinear_cache,_,_=setup();n,ns=joint(z,nonlinear_cache,.0025,.005,1e-8,1e-10)
        assert bool(jnp.all(jnp.isfinite(n))) and float(ns[1])<1e-8,ns
        scaled,_=W.make_solver(field,L,theta=.5,coefficient_map=C.coefficients,tested_mass_jacobian=C.tested_mass_jacobian,scale_linear_coordinates=True)
        s,ss=scaled(z,nonlinear_cache,.0025,.005,1e-8,1e-10)
        dc=C.coefficients(s,nonlinear_cache)-C.coefficients(n,nonlinear_cache)
        assert float(jnp.linalg.norm(field(dc,nonlinear_cache))/jnp.linalg.norm(field(C.coefficients(n,nonlinear_cache),nonlinear_cache)))<1e-8
        assert float(ss[1])<1e-8,ss
        print('PASS restricted-linear CN equivalence and nonlinear stationary step',np.asarray(ns),flush=True)


if __name__=='__main__':main()
