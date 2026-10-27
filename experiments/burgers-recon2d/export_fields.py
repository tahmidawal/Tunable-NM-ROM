"""GPU-only exact single-query replay of saved coefficients, with optional field export."""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np
import jax
import jax.numpy as jnp
import reconstruction_core as C

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--record',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    assert jax.default_backend()=='gpu';assert jax.config.jax_enable_x64;assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    root=Path(a.run);r=json.loads((root/a.record).read_text());assert r['status']=='complete'
    def load(rel,want):
        path=root/rel;h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==want,(rel,h,want)
        return np.load(path)
    cf=load(r['coefficient_file'],r['coefficient_file_sha256'])['coefficient_repetitions']
    if r['geometry_file']:
        basis=jnp.asarray(load(r['geometry_file'],r['geometry_sha256'])['fine_basis']);decode=lambda c:C._dense_decode(c,basis)
    else:
        p=load(r['plan_arrays_file'],r['plan_arrays_sha256']);matrix=jnp.asarray(p['axis_matrix'] if 'axis_matrix' in p else p['synthesis_axis']);L=r['L'];nf=255**2
        decode=jax.jit(lambda c:C._tensor_apply(c.reshape((-1,L-1,L-1)),matrix,2).reshape((-1,nf)))
    fields=[];matches=[]
    for t in range(5):
        reps=[]
        for rep in range(3):
            f=C.host(C.ready(decode(jnp.asarray(cf[t,rep:rep+1]))));reps.append(f[0]);matches.append(hashlib.sha256(f.tobytes(order='C')).hexdigest()==r['timing'][t]['repetitions'][rep]['field_sha256'])
        fields.append(reps)
    np.savez(a.out,fields=np.asarray(fields),all_hashes_match=np.asarray(matches),times=np.arange(1,6)*.05)
    print(json.dumps(dict(all_hashes_match=all(matches),matches=matches,out=a.out)))
if __name__=='__main__':main()
