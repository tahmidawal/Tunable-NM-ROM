"""Independent saved-array cost-gate audit; imports no PDE or neural model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def load(path):
    with np.load(path) as z: d={k:np.asarray(z[k]) for k in z.files}
    if 'coarse_fields_file' in d:
        with np.load(path.parent/str(d['coarse_fields_file'])) as z: d['coarse_fields']=np.asarray(z['fields'])
    return d


def cert(z, prefix='', full=True):
    g=z[prefix+'gamma_fresh']; recorded=z[prefix+'step_gn']; rn=z[prefix+'residual_norm_fresh']
    tol=float(z[prefix+'residual_tolerance'])
    count=50 if not prefix else int(z['coarse_valid_steps'])
    mask=np.arange(50)<count
    assert np.all(np.abs(g[mask]-recorded[mask])<=1e-7*np.abs(recorded[mask])+1e-12)
    valid=bool(np.isfinite(g[mask]).all() and np.all((g[mask]<=1e-3)|(rn[mask]<=tol)))
    return valid and (count==50 if full else True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('result'); ap.add_argument('--out',required=True); a=ap.parse_args()
    path=Path(a.result); x=json.loads(path.read_text()); assert x['complete']
    assert x['settings']['L']==1024 and x['settings']['M']==256 and x['settings']['rule']=='gauss128'
    assert x['backend']=='gpu' and x['x64'] and x['precision']=='highest'
    by={(t['arm'],t['case'],t['block']):t for t in x['timings']}
    assert len(by)==84 and len(x['timings'])==84
    records={(r['arm'],r['case']):r for r in x['records']}; assert len(records)==12
    cache={}
    for t in x['timings']:
        fn=t['array_witness']
        if fn not in cache: cache[fn]=load(path.parent/fn)
        z=cache[fn]
        assert t['seconds']>0 and sha(z['W'])==t['W_sha256'] and sha(z['fields'])==t['fields_sha256']
        assert int(z['it_sum'])==t['fine_corrections']
        if t['arm']=='replay':
            assert sha(z['coarse_fields'])==t['coarse_fields_sha256'] and int(z['coarse_it_sum'])==t['coarse_corrections']
    summary=dict(scope=x['scope'],job_id=x['job_id'],commit=x['commit'],gpu=x['gpu'],settings=x['settings'],
        source_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_manifest_sha256=x['model_manifest_sha256'],
        arms={},checks=x['checks'],parity=x['parity'],setup=x['setup'],compile_warm_seconds=x['compile_warm_seconds'],
        elapsed_program_seconds=x['elapsed_program_seconds'],strong_accuracy_checks=x['strong_accuracy_checks'])
    for arm in ('direct','replay'):
        rec=[records[arm,c] for c in range(6)]; arrays=[]; valid=[]
        for c in range(6):
            z=load(path.parent/f'{arm}_case{c}.npz'); arrays.append(z)
            assert np.array_equal(z['W'],z['states'][::10])
            assert int(z['it_sum'])==sum(z['step_it']) and int(z['it_max'])==max(z['step_it'])
            assert np.array_equal(z['exits'],np.bincount(z['step_reason'],minlength=5))
            stop=cert(z); valid.append(stop)
            assert stop or not records[arm,c]['valid']
            if records[arm,c]['valid']:
                assert np.isfinite(z['fields']).all() and int(z['nfail'])==0 and z['ic_reason'] in (1,4)
            if arm=='replay':
                direct=load(path.parent/f'direct_case{c}.npz')
                np.testing.assert_allclose(z['W'][0],direct['W'][0],rtol=1e-12,atol=1e-13)
                np.testing.assert_allclose(z['fields'][0],direct['fields'][0],rtol=1e-12,atol=1e-13)
                assert all(np.array_equal(z[key],direct[key]) for key in ('ic_it','ic_reason','ic_idx'))
                assert records[arm,c]['fine_initial_state_matches_direct']
                cstop=cert(z,'coarse_'); assert cstop or not records[arm,c]['valid']
                count=int(z['coarse_valid_steps']); attempted=int(z['coarse_steps'])
                assert attempted-count==int(z['coarse_nfail'])
                assert np.isnan(z['coarse_states'][count+1:]).all()
                assert np.array_equal(z['coarse_replay_valid'],np.arange(50)<count)
                info=z['replay_info']; accept=(info[:,1].astype(bool)&info[:,4].astype(bool)&
                    (info[:,3]<info[:,2])&(info[:,3]<=(1-1e-8)*info[:,2]))
                assert np.array_equal(info[:,0].astype(bool),accept)
                with np.load(path.parent/f'guard_case{c}.npz') as guard:
                    np.testing.assert_allclose(guard['objectives'],info[:,2:4],rtol=2e-11,atol=1e-22,equal_nan=True)
                    assert np.all(guard['promoted_coefficient_gap'][:count]<1e-12)
                assert not np.any(info[count:,0])
                assert records[arm,c]['accepted_candidates']==int(info[:,0].sum())
                assert records[arm,c]['coarse_physics_J']==attempted+int(z['coarse_it_sum'])
                if records[arm,c]['valid']:
                    assert count==50 and np.isfinite(z['coarse_fields']).all()
        ts=np.array([[by[arm,c,b]['seconds'] for b in range(7)] for c in range(6)])
        base=np.array([[by['direct',c,b]['seconds'] for b in range(7)] for c in range(6)])
        ratios=ts/base; med=np.median(ratios,axis=1)
        summary['arms'][arm]=dict(median_query_ms=float(np.median(np.median(ts,axis=1))*1e3),
            per_case_paired_median_ratios=med.tolist(),median_paired_ratio=float(np.median(med)),
            slower_pairs=int(np.sum(ratios>1)),pairs=42,production_stationarity=valid,
            fine_corrections=[int(z['it_sum']) for z in arrays],fine_physics_J=sum(50+int(z['it_sum']) for z in arrays),
            fine_ic_iterations=[int(z['ic_it']) for z in arrays],predictor_residual_calls=900,
            initial_fit_J_count='not exposed by unchanged initializer; both initial-fit costs included in timing')
        if arm=='replay':
            summary['arms'][arm].update(coarse_corrections=[int(z['coarse_it_sum']) for z in arrays],
                coarse_physics_J=sum(int(z['coarse_steps'])+int(z['coarse_it_sum']) for z in arrays),
                coarse_ic_iterations=[int(z['coarse_ic_it']) for z in arrays],guard_residual_calls=600,
                coarse_predictor_residual_calls=sum(3*int(z['coarse_steps']) for z in arrays),
                accepted_candidates=sum(int(z['replay_info'][:,0].sum()) for z in arrays),
                coarse_valid_steps=[int(z['coarse_valid_steps']) for z in arrays],
                unavailable_guesses=sum(int((~z['coarse_replay_valid']).sum()) for z in arrays))
            g=x['gate']; np.testing.assert_array_equal(med,g['per_case_paired_median_ratios'])
            quality=bool(x['checks']['forced_failure_controls_completed'] and all(r['valid'] for r in x['records']))
            identity=all(t['identity'] and t['online_valid'] for t in x['timings'])
            passed=bool(quality and identity and np.median(med)<=.85 and max(med)<=1.10)
            assert passed==g['passed']==x['continue_accuracy']
            summary.update(gate=g,continue_accuracy=passed)
    for control in x['parity']:
        assert control['passed']
        if control['check']=='forced_coarse_failure':
            c,k=control['case'],control['step']; z=load(path.parent/f'forced_failure_case{c}_step{k}.npz')
            assert int(z['coarse_steps'])==k+1 and int(z['coarse_valid_steps'])==k
            assert z['coarse_step_reason'][k]==3 and np.isnan(z['coarse_states'][k+1:]).all()
            assert not np.any(z['replay_info'][k:,0]) and not np.any(z['coarse_replay_valid'][k:])
            assert cert(z)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n'); print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
