"""Read-only NumPy audit of every saved reference, empirical gate and data schema."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import data as d


def initial(L,seed):
    rng=np.random.default_rng(seed)
    cx,cy=rng.uniform(.15,.85),rng.uniform(.15,.85)
    width,amp=rng.uniform(.05,.20),rng.uniform(.5,2.)
    nu=np.exp(rng.uniform(np.log(.01),np.log(.1)))
    x=np.arange(L+1)/L
    u=amp*np.exp(-((x[:,None]-cx)**2+(x[None,:]-cy)**2)/(2*width*width))
    u[[0,-1],:]=0;u[:,[0,-1]]=0
    return u,nu


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('index',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--refined',action='store_true');a=p.parse_args()
    if a.refined:
        from refine import configure
        configure()
    x=json.loads(a.index.read_text());assert x['complete'] and x['count']==8
    parent=None
    if a.refined:
        parent_path=d.HERE/'artifacts/calibration01/calibration-index.json'
        parent=json.loads(parent_path.read_text())
        assert not parent['gate']['by_output']['256']['passing']
        parent_rows={(r['case_id'],r['intervals'],r['dt']):r for r in parent['solves']}
    assert x['pde']=='burgers' and x['provenance']['backend']=='gpu'
    assert x['provenance']['f64'] and x['provenance']['matmul_precision']=='highest'
    assert x['protocol_sha256']==d.sha(d.PROTOCOL_PATH)
    for name,value in x['provenance']['source_sha256'].items():
        assert d.sha(d.ROOT/name)==value
        stored=subprocess.check_output(['git','-C',str(d.ROOT),'show',x['provenance']['source_commit']+':'+name])
        import hashlib
        assert hashlib.sha256(stored).hexdigest()==value
    fields={};counts={};initial_discrepancies=[];initial_bitwise=[]
    for row in x['solves']:
        if row.get('cache_reused'):
            assert parent is not None
            assert row['cache_parent_index_sha256']==d.sha(parent_path)
            assert row['cache_parent_source_commit']==parent['provenance']['source_commit']
            assert row['cache_parent_protocol_sha256']==parent['protocol_sha256']
            previous=parent_rows[row['case_id'],row['intervals'],row['dt']]
            for key in ['sha256','path','seed','max_relative_residual','total_newton_iterations','wall_seconds_including_first_compile']:
                assert row[key]==previous[key]
        path=a.index.parent/row['path'];assert d.sha(path)==row['sha256']
        with np.load(path) as npz:
            f=npz['fields'];it=npz['iterations'];rn=npz['residuals']
        assert f.dtype==np.float64 and f.shape==(6,1025,1025)
        assert np.isfinite(f).all() and np.isfinite(rn).all()
        assert not np.any(f[:,[0,-1],:]) and not np.any(f[:,:,[0,-1]])
        expected,nu=initial(1024,row['seed'])
        discrepancy=float(np.linalg.norm(f[0]-expected)/np.linalg.norm(expected))
        assert discrepancy<=1e-12
        initial_discrepancies.append(discrepancy);initial_bitwise.append(bool(np.array_equal(f[0],expected)))
        assert len(it)==len(rn)==round(.25/row['dt'])
        assert rn.max()<=2e-11 and it.max()<=20
        assert int(it.sum())==row['total_newton_iterations']
        assert float(rn.max())==row['max_relative_residual']
        case=int(row['case_id'].split('-')[-1]);assert row['seed']==d.case_seed('calibration',case)
        fields[row['intervals'],row['dt'],case]=f
        counts[case]=counts.get(case,0)+1
    setting_count=len(set(d.anchor_settings()+[(r['intervals'],r['dt']) for r in d.PROTOCOL['candidates']]))
    assert counts=={i:setting_count for i in range(8)}
    gate=d.evaluate_gate(x,fields)
    def compare(actual,expected):
        if isinstance(actual,dict):
            assert actual.keys()==expected.keys()
            for key in actual:compare(actual[key],expected[key])
        elif isinstance(actual,list):
            assert len(actual)==len(expected)
            for aa,bb in zip(actual,expected):compare(aa,bb)
        elif isinstance(actual,float):
            assert np.isclose(actual,expected,rtol=1e-10,atol=1e-14)
        else:assert actual==expected
    compare(gate,x['gate'])
    for row in x['records']:
        path=a.index.parent/row['path'];assert d.sha(path)==row['sha256']
        with np.load(path) as npz:
            assert set(npz.files)=={'input','target','parameters','times'}
            for k in npz.files:assert npz[k].dtype==np.float64
            assert npz['input'].shape==(1,1025,1025) and npz['target'].shape==(6,1,1025,1025)
            assert np.array_equal(npz['input'][0],npz['target'][0,0])
            case=int(row['case_id'].split('-')[-1])
            anchor=d.PROTOCOL['anchor']
            assert np.array_equal(npz['target'][:,0],fields[anchor['intervals'],anchor['dt'],case])
            assert np.array_equal(npz['times'],d.TIMES)
            expected,nu=initial(1024,row['seed']);assert np.linalg.norm(npz['input'][0]-expected)/np.linalg.norm(expected)<=1e-12
            assert npz['parameters'].shape==(1,) and np.isclose(npz['parameters'][0],nu,rtol=1e-13,atol=0)
    result=dict(passed=True,reference_index_sha256=d.sha(a.index),solves_checked=len(x['solves']),cases_checked=8,
                source_hashes_checked=True,all_fields_finite_f64_zero_boundary=True,initial_fields_seed_regenerated_within_tolerance=True,initial_relative_tolerance=1e-12,
                initial_regeneration_worst_relative=max(initial_discrepancies),initial_regeneration_bitwise=all(initial_bitwise),
                all_step_residuals_below_declared_threshold=True,all_saved_empirical_gate_numbers_reproduced=True,
                cached_original_source_protocol_index_field_links_checked=bool(a.refined),
                model_facing_schema_exact=True,gate=gate,
                independence='NumPy independently regenerates Gaussian inputs; gate implementation shared with producer; root independently reviews refinement arithmetic')
    d.write_json(a.out,result);print(json.dumps({k:result[k] for k in ['passed','solves_checked','cases_checked']}))


if __name__=='__main__':main()
