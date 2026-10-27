"""Recompute historical CN quadrature differences with saved full-grid Gram metric; no JAX."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims')
E0=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/store')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    wp=OLD/'runs/m3b/slim/output/d3run_m3b/W_m3b.npz';rp=wp.parent/'result.json'
    metric=HERE/'results/sdiag01/raw/output/metric_W1024.npz'
    L=np.load(metric)['L'];z=np.load(wp,allow_pickle=False);raw=json.loads(rp.read_text())
    man={p.strip():h for line in (HERE/'vendor/e0/store/MANIFEST.sha256').read_text().splitlines() if line.strip() for h,p in [line.split(None,1)]}
    n0=[];metadata=[]
    for j in range(64):
        rel=f'3d/val64/ST/c{j:03d}.json';p=E0/rel
        assert sha(p)==man[rel];m=json.loads(p.read_text());assert m['cohort_sha256']==raw['cohort']['table_sha256']
        n0.append(m['judge']['n0']);metadata.append(dict(path=str(p),sha256=man[rel]))
    n0=np.asarray(n0)
    out=dict(scope='full64 validation, fixed W1024/default-M/CN step0.01, reference-free quadrature differences normalized by original initial norm',
             threshold=2.5e-4,sources=dict(coefficients=dict(path=str(wp),sha256=sha(wp)),run=dict(path=str(rp),sha256=sha(rp)),
                metric=dict(path=str(metric),sha256=sha(metric)),initial_norm_metadata=metadata),comparisons=[])
    for solver in ['fs|CN|0.01|K1','ad|LSPG|CN|0.01|tight']:
        for a,b in [('gl40','gl48'),('gl48','gl56'),('gl56','gl64')]:
            ka,kb=[f'W1024|{r}|{solver}' for r in (a,b)]
            ia,ib=[np.asarray(z['cases|'+k],int) for k in (ka,kb)]
            assert np.array_equal(ia,np.arange(64)) and np.array_equal(ib,ia)
            rows=[r for r in raw['rows'] if r['tag'] in ('W1024|'+a,'W1024|'+b) and r['spec']==solver]
            assert len(rows)==128 and all(r['valid'] for r in rows)
            if solver.startswith('ad'):
                assert all(r['steps']['nfail']==0 for r in rows)
            diff=(np.asarray(z[ka])-np.asarray(z[kb]))@L
            et=np.linalg.norm(diff,axis=-1)/n0[:,None];d=et[:,1:].max(1)
            out['comparisons'].append(dict(solver=solver,coarse=a,fine=b,keys=[ka,kb],ordered_cases=ia.tolist(),
                distances=d.tolist(),median=float(np.median(d)),maximum=float(d.max()),maximum_case=int(np.argmax(d)),
                cases_above_threshold=np.flatnonzero(d>out['threshold']).tolist(),maximum_output_times=(.05*(np.argmax(et[:,1:],axis=1)+1)).tolist()))
    dest=HERE/'results/historical-tight-quadrature.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps([{k:v for k,v in x.items() if k not in ('distances','ordered_cases','maximum_output_times')} for x in out['comparisons']]))

if __name__=='__main__':main()
