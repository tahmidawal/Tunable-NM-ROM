"""Prescribed stable global completion and training-row-order sensitivity controls."""
import argparse
from common import *
from build_extended_span import read
from training_io import summary


@jax.jit
def complete(base,candidates):
    K,D=1376,base.shape[1]
    Q=jnp.zeros((K,D),dtype=jnp.float64).at[:base.shape[0]].set(base)
    accepted=jnp.full(K,-1,dtype=jnp.int32);sizes=jnp.zeros(K)
    def condition(state):
        Q,n,i,accepted,sizes=state
        return (n<K)&(i<len(candidates))
    def body(state):
        Q,n,i,accepted,sizes=state;index=candidates[i]
        v=jnp.zeros(D).at[index].set(1.)
        # Two passes; rows above n are zero. Avoid unstable near-duplicate additions.
        v=v-Q.T@(Q@v);v=v-Q.T@(Q@v);norm=jnp.linalg.norm(v)
        def add(args):
            Q,n,accepted,sizes=args
            return Q.at[n].set(v/norm),n+1,accepted.at[n].set(index),sizes.at[n].set(norm)
        Q,n,accepted,sizes=jax.lax.cond(norm>1e-3,add,lambda z:z,(Q,n,accepted,sizes))
        return Q,n,i+1,accepted,sizes
    return jax.lax.while_loop(condition,body,(Q,jnp.int32(len(base)),jnp.int32(0),accepted,sizes))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--train',required=True)
    ap.add_argument('--dev',required=True);ap.add_argument('--span',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    span=Path(a.span);sr=json.loads((span/'result.json').read_text())
    assert digest(span/'pod_basis.npz')==sr['basis_sha256']
    base=jnp.asarray(np.load(span/'pod_basis.npz')['coefficients'][:512].reshape(512,-1))*.5
    candidates=sorted((i*128+j for i in range(64) for j in range(64)),key=lambda p:((p//128+1)**2+(p%128+1)**2,p))
    Q,n,used,accepted,sizes=complete(base,jnp.asarray(candidates,dtype=jnp.int32))
    assert int(n)==1376
    defect=float(jnp.max(jnp.abs(Q@Q.T-jnp.eye(len(Q)))));assert defect<1e-9
    # Ordering and all basis decisions are frozen before development solution fields open.
    np.savez_compressed(out/'stable_global_basis.npz',coefficients=2*np.asarray(Q).reshape(len(Q),128,128),
                        accepted_fourier_indices=np.asarray(accepted),candidate_residual_norms=np.asarray(sizes))
    A,tn,tl,tp,ts=read(a.train);B,vn,vl,vp,vs=read(a.dev)
    tail=jnp.maximum(vn**2-jnp.sum(B*B,axis=1)*.25,0)
    def score(V):
        residual=B*.5-(B@V.T*.5)@V
        return summary((jnp.sqrt(jnp.sum(residual*residual,axis=1)+tail)/vn).reshape(-1,11))
    controls={str(k):score(Q[:k]) for k in (768,960,1024,1280,1376)}
    result=dict(meta=meta,status='stable_global_control_running',controls=controls,
        construction='frozen training POD512 plus ascending tensor-sine directions; twice orthogonalized; candidate norm >1e-3',
        orthogonality_defect=defect,candidates_examined=int(used),base_sha256=sr['basis_sha256'],
        basis_sha256=digest(out/'stable_global_basis.npz'),dev_labels=vl,training_sources=ts,dev_sources=vs,
        permutations=[],final_test_opened=False,reference_assisted=True,PDE_solve=False)
    save(out/'result.json',result)
    for seed in (2026111887,2026111888):
        order=np.random.default_rng(seed).permutation(len(A))
        _,s,vh=jax.jit(lambda x:jnp.linalg.svd(x,full_matrices=False))(A[order]/(2*tn[order,None]))
        result['permutations'].append(dict(seed=seed,controls={str(k):dict(projection=score(vh[:k]),
            relative_training_singular_value=float(s[k-1]/s[0])) for k in (1024,1280,1376)}))
        save(out/'result.json',result)
    result['status']='complete_stable_global_controls';save(out/'result.json',result)
    print('stable_global',{k:v['worst'] for k,v in controls.items()},flush=True)


if __name__=='__main__':main()
