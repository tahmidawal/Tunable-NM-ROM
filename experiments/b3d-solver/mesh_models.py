"""Explicit-argument fixed-mode mesh queries around unchanged stepping kernels."""
from types import SimpleNamespace
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
import c3common as C3
import s3common as S
import qn3
import p1core as P1
import fs3
from work_counts import fixed_sweep,quasi_newton

def ready(x):return jax.tree_util.tree_map(jax.block_until_ready,x)
def relative(a,b):
    a,b=map(jnp.asarray,(a,b))
    return float(jnp.linalg.norm(a-b)/(1+jnp.linalg.norm(b)))

def row_equivalence(a,b):
    """Compare differently blocked copies without a full concatenation."""
    off=0;num=0.;den=0.
    for g in a:
        target=C3.TB.rows_prefix(b,off+g.shape[0]);parts=[];start=0
        for h in target:
            lo=max(off-start,0)
            if lo<h.shape[0]:parts.append(h[lo:])
            start+=h.shape[0]
        h=jnp.concatenate(parts,axis=0)
        num+=float(jnp.sum((g-h)**2));den+=float(jnp.sum(h*h));off+=g.shape[0]
    return np.sqrt(num)/(1+np.sqrt(den))

def data_for(lean,base,A=None,lam=None,P=None):
    """New A always receives a new predictor QR; B/D and quadrature projection are frozen."""
    A=lean['A'] if A is None else A;lam=lean['lam'] if lam is None else lam
    d=dict(lean,A=A,lam=lam,B=base['B'],D=base['D'],P=base['P'] if P is None else P,
           Bw=base['Bw'],Lq=base['Lq'],hmask=(lam>jnp.median(lam)).astype(jnp.float64),**qn3.qr_data(A))
    return ready(d)

@lru_cache(maxsize=None)
def kernels(r,a,trust,sp,path,n):
    """Cache only functions/static scalars; never cache any bank/table/input array here."""
    st=SimpleNamespace(Rp=r,a=a,trust=trust)
    source=S.make_query(sp,st,path='mf' if path=='mf' else 'mesh',decode=False,n=n)
    keep,steps=source.keep,source.steps
    def project(u,scale,d):
        return (u[None],jnp.asarray([scale])) if path=='held' else source.project(u[None],d)
    @jax.jit
    def query(u,nu,scale,d,decode_data):
        c0,s0=project(u,scale,d)
        o=source.evolve(c0,jnp.asarray([nu]),s0,d)
        o['W']=o['W_all'][::keep];o['w0']=c0
        o['fields']=S.decode_lattice(o['W'],decode_data)[0]
        return o
    @jax.jit
    def init(u,scale,d):return project(u,scale,d)
    @jax.jit
    def decode(o,decode_data):return S.decode_lattice(o['W_all'][::keep],decode_data)[0]
    anchors={}
    for label,gt in [('tight',1e-7),('tighter',1e-8)]:
        ev=P1.make_evolve(C3.nl3,C3.nlJ3,'LSPG',st.Rp,st.trust,nrec=steps+1)
        sch=P1.sched(sp[1],sp[2],gt,0.)
        def factory(ev):
            @jax.jit
            def anchor(u,nu,scale,d,sch):
                c0,s0=project(u,scale,d)
                W,stats=ev(c0[0],nu,s0[0],d,sch)
                return dict(W=W,stats=stats)
            return anchor
        anchors[label]=(factory(ev),sch,gt)
    return source,query,init,decode,anchors

def make(st,sp,path,n,data,output_data):
    """path=mesh/mf/held. Ordinary adapter closes over data; inner JIT takes every array explicitly."""
    source,query,init,decode,anchors=kernels(st.Rp,st.a,float(st.trust),tuple(sp),path,n)
    keep,steps=source.keep,source.steps
    def work(o):
        if sp[0]=='qn':
            return dict(quasi_newton(np.asarray(o['q'])[:,0],np.asarray(o['n_bt'])[:,0],np.asarray(o['reason'])[:,0]),
                        jacobian_contractions=int(np.asarray(o['n_exec']).sum()),raw_step_records=S.step_record(sp,o,0))
        a=fs3.a_effective(sp[1],sp[2],.01,st.a,steps)
        return dict(fixed_sweep(np.asarray(o['it'])[:,0],a,sp[3],sp[1]),raw_step_records=S.step_record(sp,o,0))
    return SimpleNamespace(run=lambda u,nu,scale=0.:query(u,nu,scale,data,output_data),
        initialize=lambda u,scale=0.:init(u,scale,data),
        evolve=lambda initial,nu:source.evolve(initial[0],jnp.asarray([nu]),initial[1],data),
        decode=lambda o:decode(o,output_data),anchors=anchors,data=data,output_data=output_data,
        certify=qn3.make_certify(sp[1],sp[2],steps),work=work,sp=sp,steps=steps,keep=keep,path=path,n=n,
        valid=lambda o:S.run_valid(sp,o,0))

def weighted_pair(c,cn,cm,nu,data,scheme,dt,step,p_scale=1.):
    a0,a1,a2,b0,b1,ns=qn3.SCHEMES[scheme]
    if step<ns:a0,a1,a2,b0,b1=qn3.BE_CO
    A,lam=data['A'],data['lam'];Srow=1/(a0+dt*b0*nu*lam)
    Ac=A@c;p=A@cn
    hist=a1*p+a2*(A@cm)+dt*b1*(p_scale*C3.nl3(cn,data)+nu*lam*p)
    r=Srow*(a0*Ac+hist+dt*b0*(p_scale*C3.nl3(c,data)+nu*lam*Ac))
    J=A+(dt*b0*Srow*p_scale)[:,None]*C3.OM.contract_offmesh(data,c)
    return r,J,Srow

def normalized_control(native,normalized,r):
    gen=np.random.default_rng(917);c,cn,cm=[jnp.asarray(gen.normal(size=r)*.01) for _ in range(3)]
    checks=[]
    for scheme,dt in [('CN',.01),('BDF2',.00625)]:
        for step in (0,3):
            # P129=sqrt(8)*P65 exactly by its explicit formula; factor the scalar
            # rather than allocate a second up-to17GB matrix only for this control.
            a,b,_=weighted_pair(c,cn,cm,.03,native,scheme,dt,step,p_scale=np.sqrt(8))
            x,y,_=weighted_pair(c,cn,cm,.03,normalized,scheme,dt,step)
            er,ej=relative(a/np.sqrt(8),x),relative(b/np.sqrt(8),y)
            assert max(er,ej)<=1e-12,(scheme,step,er,ej)
            checks.append(dict(scheme=scheme,dt=dt,step=step,residual_relative=er,Jacobian_relative=ej))
    return checks

def restrict(fields,n):
    F=np.asarray(fields)
    if n==65:return F.reshape(6,-1)
    assert n==129
    return F.reshape(6,127,127,127)[:,1::2,1::2,1::2].reshape(6,-1)

def metric_distance(a,b,L,n0):
    return float(jnp.linalg.norm((jnp.asarray(a)-jnp.asarray(b))@L,axis=1)[1:].max()/n0)
