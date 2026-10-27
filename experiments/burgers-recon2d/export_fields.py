"""GPU-only exact single-query replay of saved coefficients, with optional field export."""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np
import jax
import jax.numpy as jnp
import reconstruction_core as C

def replay(root,record,out):
    root=Path(root);r=json.loads((root/record).read_text());assert r['status']=='complete'
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
    np.savez(out,fields=np.asarray(fields),all_hashes_match=np.asarray(matches),times=np.arange(1,6)*.05)
    return dict(record=record,all_hashes_match=all(matches),matches=matches,out=str(out))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--record');ap.add_argument('--out');ap.add_argument('--representatives',action='store_true');a=ap.parse_args()
    assert jax.default_backend()=='gpu';assert jax.config.jax_enable_x64;assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    if not a.representatives:
        if not a.record or not a.out:raise ValueError('record and out required')
        print(json.dumps(replay(a.run,a.record,a.out)));return
    root=Path(a.run);frozen=json.loads((root/'FROZEN.json').read_text());summary=json.loads((root/'summary.json').read_text())
    selected={(c['L'],c['source'],c['selected']['method']) for c in frozen['choices'] if c['selected']}
    for source in ['A','B']:
        for method in frozen['strict_coarse_control']['methods']:selected.add((128,source,method))
    dest=root/'replayed_fields';dest.mkdir();records=[]
    for row in summary:
        if row['case']!=0 or (row['L'],row['source'],row['method']) not in selected:continue
        if row['status']!='complete':records.append(dict(record=row['record'],status='failed_original',reason=row.get('reason')));continue
        result=replay(root,row['record'],dest/f"L{row['L']}_{row['source']}_{row['method']}.npz");records.append(result)
    audit=dict(records=records,completed_replays=sum('all_hashes_match' in r for r in records),failed_original=sum(r.get('status')=='failed_original' for r in records),all_completed_replays_match=all(r.get('all_hashes_match',True) for r in records))
    (dest/'index.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='records'}))
    if not audit['all_completed_replays_match']:raise RuntimeError('saved coefficient replay does not match timed field hashes')
if __name__=='__main__':main()
