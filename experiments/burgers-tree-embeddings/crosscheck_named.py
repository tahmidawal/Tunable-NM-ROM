"""Independent central FD/CN check of all twelve named development fixtures."""
import argparse
from common import *
import data,fd_reference as F
from neural.named_references import initial
from neural.build_data import resolve


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ns=Path(os.environ['STUDY_NAMESPACE'])
    src=resolve(ns/'b6namedrefs/output/result.json',ns);source=json.loads(src.read_text())
    assert source['status']=='complete_named_ladders_pending_independent_crosscheck'
    result=dict(meta=preflight(),source_sha256=digest(src),status='running',cases=[],final_test_opened=False)
    solvers={L:F.make_solver(2,L) for L in (256,512)}
    for row in source['cases']:
        assert row['provisional_ladder_pass'];rp=src.parent/row['levels'][-1]['file']
        assert digest(rp)==row['levels'][-1]['sha256'];b=np.load(rp)['coefficients'];records=[];states=[]
        for L,dt in [(256,.0003125),(512,.0003125),(512,.00015625)]:
            axis=np.arange(1,L)/L;xy=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
            v,seconds=timer(lambda:solvers[L](initial(jnp.asarray(xy),row['definition']),row['nu'],dt,'cnr',1e-11,1e-11))
            state=np.asarray(v['out']).reshape(11,L-1,L-1);assert np.isfinite(state).all()
            truth=np.asarray(data.synth_grid(jnp.asarray(b),jnp.asarray(axis)))
            norm=np.linalg.norm(truth.reshape(11,-1),axis=1)
            err=np.linalg.norm((state-truth).reshape(11,-1),axis=1)/norm
            used=int(v['stages']);residual=float(np.max(np.asarray(v['res'])[:used]));assert np.isfinite(residual)
            path=out/f"{row['case']}_L{L}_dt{dt:.8f}.npz"
            np.savez_compressed(path,**v,times=data.TIMES,nu=row['nu'])
            records.append(dict(L=L,dt=dt,file=path.name,sha256=digest(path),relative_error=err,
                max_solver_residual=residual,seconds=seconds));states.append(state)
            print('named_crosscheck',row['case'],L,dt,float(err.max()),residual,flush=True)
        timechange=np.linalg.norm((states[-1]-states[-2]).reshape(11,-1),axis=1)/norm
        passed=bool(max(records[-1]['relative_error'])<=.001 and max(timechange)<=.0002
                    and all(r['max_solver_residual']<=1e-11 for r in records))
        result['cases'].append(dict(case=row['case'],nu=row['nu'],levels=records,temporal_change=timechange,
            passed=passed,admitted_reference=passed,reference_allowance=row['allowance']))
        save(out/'result.json',result)
    result.update(status='complete',passed=all(r['passed'] for r in result['cases']))
    save(out/'result.json',result)


if __name__=='__main__':main()
