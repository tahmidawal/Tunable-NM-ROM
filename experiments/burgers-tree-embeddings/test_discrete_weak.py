"""Small exact-grid arithmetic/Jacobian fixture for the fixed-tree solver."""
from common import *
import discrete_weak as W,operators,geometry,tensor_metric as TM,local_polynomial as LP
import weak_quadrature as QW

def main():
    L=32;root=4;rank=4;M=80;cells=geometry.uniform(1)
    base=jnp.zeros((root,16,16)).at[jnp.arange(root),jnp.array([0,1,0,1]),jnp.array([0,0,1,1])].set(2.)
    G,T,_,_=TM.make_geometry(6)(base,jnp.asarray(cells),*TM.rule(N=32))
    U=jnp.eye(36,rank)
    from build_local_cache import expand_matrix
    trans=expand_matrix(U,len(cells),root);G=trans.T@G@trans
    cache,slices=W.prepare(base,cells,T,U,L,M,G);field=W.make_field(slices,rank)
    c=jnp.zeros(root+len(cells)*rank).at[0].set(.2).at[root:].set(.001)
    u=field(c,cache);ref=LP.field_grid(base,c@trans.T,cells,T,6,np.arange(1,L)/L)[0]
    assert float(jnp.max(jnp.abs(u-ref)))<1e-12
    phi,lam,_=operators.modes(L,M);exact=phi.T@u.reshape(-1)/L
    assert float(jnp.max(jnp.abs(exact-cache['A']@c)))<1e-12
    assert float(jnp.max(jnp.abs(W.dst2(u)[cache['indices'][0],cache['indices'][1]]/L-exact)))<1e-12
    step,residual=W.make_solver(field,L);dt=.0025;nu=.005;prior=.99*exact;norm=.2
    r=residual(c,prior,cache,dt,nu,norm);adv,lap=operators.spatial(u,L)
    dense=(exact-prior+dt*(phi.T@(adv-nu*lap)/L))/(1+dt*nu*lam)/norm
    assert float(jnp.max(jnp.abs(r-dense)))<1e-12
    direction=jnp.cos(jnp.arange(len(c)));eps=1e-6
    jv=jax.jvp(lambda z:residual(z,prior,cache,dt,nu,norm),(c,),(direction,))[1]
    fd=(residual(c+eps*direction,prior,cache,dt,nu,norm)-residual(c-eps*direction,prior,cache,dt,nu,norm))/(2*eps)
    assert float(jnp.linalg.norm(jv-fd)/jnp.linalg.norm(fd))<1e-7
    ci,ei,li=W.make_initial_fit(field)(u,cache)
    ci_tight,_,li_tight=W.make_initial_fit(field,1e-11,1e-15)(u,cache)
    assert float(li_tight)<1e-10,li_tight
    assert float(jnp.linalg.norm(field(ci_tight,cache)-field(ci,cache))/jnp.linalg.norm(u))<1e-8
    assert float(ei)<1e-7 and float(li)<1e-7,(ei,li)
    assert float(jnp.linalg.norm(field(ci,cache)-u)/jnp.linalg.norm(u))<1e-7
    out,status=step(c,cache,dt,nu,1e-8,1e-10)
    assert jnp.all(jnp.isfinite(out)) and float(status[1])<1e-7,status
    # Explicit old/new dense stencils independently qualify the CN option.
    cnstep,cnres=W.make_solver(field,L,theta=.5)
    old=.98*c;oldu=field(old,cache);oldadv,oldlap=operators.spatial(oldu,L)
    oldtest=cache['A']@old
    rhs=oldtest*(1-.5*dt*nu*lam)-.5*dt*(phi.T@oldadv/L)
    densecn=(exact-oldtest+.5*dt*(phi.T@(adv+oldadv-nu*(lap+oldlap))/L))/(1+.5*dt*nu*lam)/norm
    assert float(jnp.max(jnp.abs(cnres(c,rhs,cache,dt,nu,norm)-densecn)))<1e-12
    cnout,cnstatus=cnstep(c,cache,dt,nu,1e-8,1e-10)
    assert jnp.all(jnp.isfinite(cnout)) and float(cnstatus[1])<1e-7,cnstatus
    tight,_=W.make_solver(field,L,theta=.5,cg_tolerance=1e-8,cg_absolute_tolerance=1e-14)
    tightout,tightstatus=tight(c,cache,dt,nu,1e-10,1e-12)
    assert float(tightstatus[1])<1e-10,tightstatus
    delta=tightout-cnout
    assert float(jnp.sqrt(jnp.maximum(delta@G@delta,0)/(cnout@G@cnout)))<1e-8
    for Q in (16,32):
        qc,qs=QW.prepare(base,cells,T,U,L,Q);ca=dict(cache,quadrature=qc)
        fields=QW.make_fields(qs,rank);values=fields(c,ca)
        qidx=jnp.arange(1,Q)*(L//Q);ii,jj=jnp.meshgrid(qidx,qidx,indexing='ij');padded=jnp.pad(u,1)
        expected=jnp.stack([padded[ii,jj],padded[ii+1,jj],padded[ii-1,jj],padded[ii,jj+1],padded[ii,jj-1]])
        assert float(jnp.max(jnp.abs(values-expected)))<1e-12
        qadv=QW.make_tested(fields,L,Q);exactadv=W.dst2(adv.reshape(L-1,L-1)[ii-1,jj-1])/Q
        assert float(jnp.max(jnp.abs(qadv(c,ca)-exactadv[cache['indices'][0],cache['indices'][1]])))<1e-12
        if Q==L:
            qstep,_=W.make_solver(field,L,theta=.5,advection_tests=qadv)
            qo,qs=qstep(c,ca,dt,nu,1e-8,1e-10)
            assert float(jnp.linalg.norm(qo-cnout))<1e-8 and float(qs[1])<1e-7
    LL=512;MM=5504;idx,lamfast=W.selected_modes(LL,MM)
    kk=np.arange(1,LL);vv=4*LL**2*np.sin(np.pi*kk/(2*LL))**2;dense=vv[:,None]+vv[None,:]
    selected=np.argsort(dense.ravel(),kind='stable')[:MM]
    assert np.array_equal(idx,np.asarray(np.unravel_index(selected,dense.shape)))
    assert np.array_equal(lamfast,dense.ravel()[selected])
    print('PASS exact dense weak-upwind / eigen-diffusion / field / Jacobian / step',np.asarray(status),flush=True)

if __name__=='__main__':main()
