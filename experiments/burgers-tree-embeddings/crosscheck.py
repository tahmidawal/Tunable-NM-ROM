"""Independent central FD/CN references; same family, no changes to ROM physics."""
import argparse
from common import *
import data, fd_reference as F

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--reference',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ref=Path(a.reference)
    assert str(ref).startswith(os.environ['STUDY_NAMESPACE']+'/')
    source=json.loads((ref/'result.json').read_text())
    assert source['status']=='spectral_ladders_complete_pending_independent_checks'
    worst=sorted([r for r in source['cases'] if r['case'].startswith('dev')],key=lambda r:max(r['allowance']),reverse=True)[:4]
    cases=[r for r in source['cases'] if not r['case'].startswith('dev')]+worst
    result=dict(meta=preflight(),cases=[],source_result_sha256=digest(ref/'result.json'),status='running')
    solvers={L:F.make_solver(2,L) for L in (256,512)}
    for row in cases:
        label=row['case'];p=np.asarray(row['parameters']);rf=ref/row['levels'][-1]['file']
        assert digest(rf)==row['levels'][-1]['sha256'];b=np.load(rf)['coefficients']
        records=[]
        for L in (256,512):
            x=np.arange(1,L)/L;xy=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
            v,secs=timer(lambda:solvers[L](data.initial(xy,p),data.NU,.0003125,'cnr',1e-11,1e-11))
            state=np.asarray(v['out']).reshape(11,L-1,L-1)
            truth=np.asarray(data.synth_grid(jnp.asarray(b),x))
            err=np.linalg.norm((state-truth).reshape(11,-1),axis=1)/np.linalg.norm(truth.reshape(11,-1),axis=1)
            path=out/f'{label}_fd{L}.npz';np.savez_compressed(path,**v,parameters=p,times=data.TIMES)
            records.append(dict(L=L,dt=.0003125,relative_error=err,seconds=secs,sha256=digest(path),
                                file=path.name,max_solver_residual=float(np.nanmax(v['res']))))
            print(label,L,float(err.max()),secs,flush=True)
        result['cases'].append(dict(case=label,levels=records,passed=bool(max(records[-1]['relative_error'])<=.001)))
        save(out/'result.json',result)
    result['status']='complete';result['passed']=all(r['passed'] for r in result['cases']);save(out/'result.json',result)
if __name__=='__main__':main()
