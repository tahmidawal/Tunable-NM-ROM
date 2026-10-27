"""Lean table assembly with an explicit frozen test list; derived from original TB.build_tables.

Uses unchanged feature_rows, Gram accumulation and selected DST moments. The intentional
change is explicit kxyz instead of reselecting/shell-completing modes at the new mesh.
No bank transform or production equation is changed here.
"""
import numpy as np
import jax
import jax.numpy as jnp
import c3common as C3

def assemble(n,GTb,kxyz):
    kxyz=np.asarray(kxyz,np.int64).copy();assert kxyz.ndim==2 and kxyz.shape[1]==3
    assert np.all((kxyz>=1)&(kxyz<n-1))
    R=sum(g.shape[0] for g in GTb);N=(n-2)**3
    assert all(g.shape[1]==N for g in GTb)
    acc=jax.jit(lambda Gm,rows:Gm+rows@rows.T)
    Gm=jnp.zeros((R,R),jnp.float64)
    for s in range(0,N,1<<20):Gm=acc(Gm,jnp.concatenate([g[:,s:s+(1<<20)] for g in GTb],axis=0))
    assert bool(jnp.isfinite(Gm).all())
    L=jnp.linalg.cholesky(Gm);ev=jnp.linalg.eigvalsh(Gm)
    idx=tuple(jnp.asarray(kxyz[:,a]) for a in range(3))
    phi=jax.jit(lambda rows:C3.C.phiT(rows,n,idx))
    A=jnp.concatenate([phi(g[s:s+16]) for g in GTb for s in range(0,g.shape[0],16)],axis=0).T
    l1=C3.C.lam1(n);lam=np.sum(l1[kxyz-1],axis=1)
    return jax.tree_util.tree_map(jax.block_until_ready,dict(GTb=tuple(GTb),L=L,A=A,lam=jnp.asarray(lam))),dict(
        n=n,r=R,M=len(kxyz),mode_hash=C3.sha(kxyz),native_points=N,Gram_condition=float(ev[-1]/ev[0]))

def build(n,bank,r,kxyz,row_block=256):
    assert row_block<=256 and row_block>0 and r<=bank['T'].shape[1]
    x=C3.C.interior_coords(n);GTb=[]
    for a in range(0,r,row_block):
        GTb.append(jax.block_until_ready(C3.TB.feature_rows(bank['params'],bank['T'][:,a:min(a+row_block,r)],x)))
    return assemble(n,tuple(GTb),kxyz)

def direct_checks(n,data,kxyz):
    """Small selected DST/direct-sine and discrete sine-eigenvalue checks, avoiding full production-size matrices."""
    ni=n-2;k=np.asarray(kxyz,np.int64)
    # Only three frozen tests and two rows. Direct sums stream spatial slabs.
    chosen=np.unique(np.linspace(0,len(k)-1,min(3,len(k)),dtype=int));kk=k[chosen]
    rows=np.concatenate([np.asarray(g[:min(2,g.shape[0])]) for g in data['GTb'][:1]],axis=0)
    direct=np.zeros((len(kk),len(rows)))
    axis=np.arange(1,n-1);S=np.sqrt(2/(n-1))
    for ix in axis:
        y,z=np.meshgrid(axis,axis,indexing='ij');nodes=np.stack([np.full(y.size,ix),y.ravel(),z.ravel()],axis=1)
        Phi=C3.C.phi_explicit(n,kk,nodes)
        start=(ix-1)*ni**2;direct+=Phi.T@rows[:,start:start+ni**2].T
    observed=np.asarray(data['A'])[chosen,:len(rows)]
    moment=float(np.max(np.abs(direct-observed))/(1+np.max(np.abs(observed))))
    assert moment<=1e-12
    # Independent central difference eigenrelation on a small one-dimensional sine vector.
    xx=np.arange(1,n-1)/(n-1);p=int(kk[0,0]);v=np.sin(np.pi*p*xx);pad=np.pad(v,1)
    lhs=(2*v-pad[:-2]-pad[2:])*(n-1)**2;expected=C3.C.lam1(n)[p-1]*v
    eig=float(np.max(np.abs(lhs-expected))/(1+np.max(np.abs(expected))));assert eig<=1e-11
    return dict(selected_moment_relative=moment,sine_eigenrelation_relative=eig,selected_modes=kk.tolist())
