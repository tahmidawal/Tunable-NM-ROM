"""Small synthetic wrapper/normalization controls, never scientific Burgers trials."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/speed3d'))
import jax
import jax.numpy as jnp
import c3common as C3
import s3common as S
import frozen_tables as FT
import mesh_models as MM

def sine_features(X,k):
    phase=np.pi*np.asarray(X)[:,None,:]*k[None,:,:];s=np.sin(phase)
    B=np.sqrt(8)*np.prod(s,axis=2);D=np.zeros_like(B)
    for a in range(3):
        other=[i for i in range(3) if i!=a]
        D+=np.sqrt(8)*np.pi*k[None,:,a]*np.cos(phase[:,:,a])*np.prod(s[:,:,other],axis=2)
    return B,D

def main():
    assert jax.config.x64_enabled
    k=np.array([[1,1,1],[2,1,1],[1,2,1]],np.int64);tests=np.concatenate([k,[[2,2,1]]])
    X=C3.C.interior_coords(5);G,_=sine_features(X,k)
    lean,meta=FT.assemble(5,(jnp.asarray(G.T[:1]),jnp.asarray(G.T[1:])),tests)
    z,w=np.polynomial.legendre.leggauss(5);z=(z+1)/2;w=w/2
    ix=np.stack(np.meshgrid(np.arange(5),np.arange(5),np.arange(5),indexing='ij'),axis=-1).reshape(-1,3)
    Q=z[ix];weights=np.prod(w[ix],axis=1);B,D=sine_features(Q,k)
    B,D=map(jnp.asarray,(B,D));Bw=B*jnp.asarray(weights)[:,None]
    base=dict(B=B,D=D,P=C3.OM.test_block(5,Q,weights,tests),Bw=Bw,Lq=jnp.linalg.cholesky(B.T@Bw))
    d=MM.data_for(lean,base);st=SimpleNamespace(Rp=3,a=3,trust=.2)
    u=jnp.asarray(G@np.array([.05,.01,-.005]));nu=.1;sp=('qn','BDF2',.05/3,4)
    source=S.make_query(sp,st,path='mesh',n=5);expected=MM.ready(source(u[None],jnp.asarray([nu]),d))
    q=MM.make(st,sp,'mesh',5,d,d);o=MM.ready(q.run(u,nu))
    c0,scale=MM.ready(q.initialize(u));held=MM.make(st,sp,'held',5,d,d)
    h=MM.ready(held.run(c0[0],nu,scale[0]));e=MM.ready(q.evolve((c0,scale),nu));F=MM.ready(q.decode(e))
    controls=dict(source_state_relative=MM.relative(o['W_all'],expected['W_all']),
        source_field_relative=MM.relative(o['fields'],expected['fields'][0]),
        held_state_relative=MM.relative(h['W_all'],o['W_all']),split_fused_relative=MM.relative(F,o['fields']))
    assert max(controls.values())<=1e-12,controls
    for label,(query,sch,gt) in q.anchors.items():
        a=MM.ready(query(u,nu,0.,d,sch));gamma=np.asarray(q.certify(a['stats']['states'],nu,d))
        controls[label+'_fresh_gamma_max']=float(gamma.max())
        assert int(a['stats']['nfail'])==0 and np.all(gamma<=gt)
    mf=S.make_query(sp,st,path='mf',n=5);m=MM.make(st,sp,'mf',5,d,d)
    uX=jnp.asarray(np.asarray(B)@np.array([.05,.01,-.005]))
    a=MM.ready(mf(uX[None],jnp.asarray([nu]),d));b=MM.ready(m.run(uX,nu))
    controls['mf_source_state_relative']=MM.relative(a['W_all'],b['W_all']);assert controls['mf_source_state_relative']<=1e-12
    native=dict(d,A=d['A']*np.sqrt(8));normalization=MM.normalized_control(native,d,3)
    changed=MM.data_for(lean,base,A=d['A']*1.01,lam=d['lam']*1.02)
    qr=MM.relative(changed['QA']@changed['RA'],changed['A']);assert qr<=1e-12
    c=jnp.asarray([.05,.01,-.005]);s0=MM.weighted_pair(c,c,c,nu,d,'BDF2',.05/3,3)[2]
    s1=MM.weighted_pair(c,c,c,nu,changed,'BDF2',.05/3,3)[2]
    assert not np.array_equal(s0,s1)
    flat=np.arange(127**3,dtype=np.float64);fields=np.broadcast_to(flat,(6,len(flat)))
    restricted=MM.restrict(fields,129);want=flat.reshape(127,127,127)[1::2,1::2,1::2].reshape(-1)
    assert restricted.shape==(6,63**3) and np.array_equal(restricted[0],want)
    out=dict(passed=True,synthetic_only=True,x64=True,wrapper_controls=controls,normalization_controls=normalization,
        changed_A_predictor_QR_relative=qr,lambda_also_changes_row_weights=True,native129_returned_field_restriction_pass=True)
    p=HERE/'audits/mesh-small-controls.json';p.write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
