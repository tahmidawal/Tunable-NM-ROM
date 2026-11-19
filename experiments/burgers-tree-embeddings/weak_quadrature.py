"""Structured weak integration with the ORIGINAL fine-grid sign-upwind stencil.

Q controls integration locations only. Stencil spacing and diffusion remain 1/L
and the fine-L eigenvalues. Weights are uniform composite integration weights,
not residual-trained or NNLS weights. Dense comparisons qualify the rule.
"""
from common import *
import discrete_weak as W


def prepare(basis,cells,transforms,rotation,L,Q):
    assert Q<=L and L%Q==0
    axis=np.arange(1,Q)/Q;delta=1/L
    x=np.stack([axis,axis+delta,axis-delta,axis,axis])
    y=np.stack([axis,axis,axis,axis+delta,axis-delta])
    N=basis.shape[-1];k=np.arange(1,N+1)
    def sine(a):return np.sin(np.pi*a[...,None]*k)*((a>0)&(a<1))[...,None]
    axes=[];slices=[];T=[]
    for cell,transform in zip(cells,transforms):
        depth,ix,iy=cell;h=2.**(-depth);centers=(np.array([ix,iy])+.5)*h
        parts=[];locations=[]
        for allx,center in zip((x,y),centers):
            ids=np.where(np.any(abs(allx-center)<=.75*h,axis=0))[0]
            assert len(ids)>0,'Integration rule misses a patch'
            lo,hi=int(ids[0]),int(ids[-1])+1
            parts.append(jnp.asarray(W.raw_axis(allx[:,lo:hi],center,h)))
            locations.append((lo,hi))
        axes.append(tuple(parts));slices.append(tuple(locations));T.append(jnp.asarray(transform)@jnp.asarray(rotation))
    return dict(basis=jnp.asarray(basis),sine_x=jnp.asarray(sine(x)),sine_y=jnp.asarray(sine(y)),
                axes=tuple(axes),transform=tuple(T)),slices


def make_fields(slices,rank):
    def fields(c,cache):
        q=cache['quadrature'];root=len(q['basis']);B=jnp.einsum('k,kij->ij',c[:root],q['basis'])
        u=q['sine_x']@B@jnp.swapaxes(q['sine_y'],-1,-2)
        for i,((x0,x1),(y0,y1)) in enumerate(slices):
            raw=q['transform'][i]@c[root+i*rank:root+(i+1)*rank]
            X,Y=q['axes'][i];patch=X@raw.reshape(6,6)@jnp.swapaxes(Y,-1,-2)
            u=u.at[:,x0:x1,y0:y1].add(patch)
        return u
    return fields


def make_tested(fields,L,Q):
    def advection(c,cache):
        center,xp,xm,yp,ym=fields(c,cache)
        adv=center*L*(jnp.where(center>0,center-xm,xp-center)+jnp.where(center>0,center-ym,yp-center))
        coeff=W.dst2(adv)/Q
        return coeff[cache['indices'][0],cache['indices'][1]]
    return advection


def make_gram(slices):
    """Exact mass matrix for the chosen composite integration rule and frozen windows."""
    @jax.jit
    def gram(q):
        n=q['sine_x'].shape[1];Q=n+1;P=len(slices);assert Q>q['basis'].shape[-1]
        X=jnp.zeros((n,P,6));Y=jnp.zeros_like(X)
        for i,((x0,x1),(y0,y1)) in enumerate(slices):
            X=X.at[x0:x1,i,:].set(q['axes'][i][0][0]);Y=Y.at[y0:y1,i,:].set(q['axes'][i][1][0])
        gx=jnp.einsum('npa,nqb->paqb',X,X)/Q;gy=jnp.einsum('npa,nqb->paqb',Y,Y)/Q
        raw=jnp.einsum('paqb,pcqd->pacqbd',gx,gy).reshape(P,36,P,36);T=jnp.stack(q['transform'])
        local=jnp.einsum('par,paqb,qbs->prqs',T,raw,T).reshape(P*T.shape[-1],P*T.shape[-1])
        mx=jnp.einsum('ni,npa->pia',q['sine_x'][0],X)/Q;my=jnp.einsum('ni,npa->pia',q['sine_y'][0],Y)/Q
        cross=jnp.einsum('gij,pia,pjb->gpab',q['basis'],mx,my).reshape(len(q['basis']),P,36)
        cross=jnp.einsum('gpr,prk->gpk',cross,T).reshape(len(q['basis']),-1)
        root=.25*jnp.einsum('gij,hij->gh',q['basis'],q['basis'])
        G=jnp.block([[root,cross],[cross.T,local]])
        return (G+G.T)*.5
    return gram
