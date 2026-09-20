"""Independent archive provenance, all-case references, coverage and table audit."""
from __future__ import annotations
import argparse,hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
from scipy.fft import dstn,idstn


def reference(n,p,cfg,continuum):
    x=np.arange(1,n,dtype=np.float64)/n
    factors=[x*(1-x)*np.exp(-((x-p[j])/p[3])**2/2) for j in range(3)]
    u=64*p[4]*factors[0][:,None,None]*factors[1][None,:,None]*factors[2][None,None,:]
    modes=np.arange(1,n,dtype=np.float64)
    eig=(np.pi*modes)**2 if continuum else 4*n*n*np.sin(np.pi*modes/(2*n))**2
    lam=eig[:,None,None]+eig[None,:,None]+eig[None,None,:]
    transformed=dstn(u,type=1,norm='ortho',workers=4)
    return np.stack([u]+[idstn(transformed*np.exp(-cfg['diffusivity']*t*lam),type=1,norm='ortho',workers=4) for t in cfg['times'][1:]])


def relative(a,b):
    return np.linalg.norm((a-b).reshape(len(a),-1),axis=1)/np.linalg.norm(b.reshape(len(b),-1),axis=1)


def audit(archive,destination):
    archive=Path(archive);out=archive/'out';d=json.loads((out/'result.json').read_text());cfg=d['config']
    assert d['complete'] and d['backend']=='gpu' and d['x64'] and d['matmul_precision']=='highest'
    assert not d['final_cohort_opened']
    log=(archive/'job.out').read_text();err=(archive/'job.err').read_text()
    assert 'jax_backend=gpu' in log and 'run_exit=0' in log
    assert not re.search(r'captur\w*.{0,30}large.{0,30}constant|RESOURCE_EXHAUSTED|out.of.memory|No space left',log+'\n'+err,re.I)
    provenance=json.loads((archive/'PROVENANCE.json').read_text());commit=provenance['source_commit']
    assert commit==d['source_commit'] and (archive/'COMMIT.txt').read_text().strip()==commit
    source_checks=0
    for row in provenance['files']:
        blob=subprocess.check_output(['git','show',f'{commit}:{row["path"]}'])
        assert hashlib.sha256(blob).hexdigest()==row['sha256']
        assert (archive/row['staged']).read_bytes()==blob
        source_checks+=1
    cohorts=json.loads((out/'cohorts.json').read_text());params=np.asarray(cohorts['validation_parameters'])
    drawn=np.random.default_rng(cfg['validation_seed']).random((cfg['validation_count'],5))*[.3,.3,.3,.05,.4]+[.35,.35,.35,.10,.8]
    np.testing.assert_array_equal(params,drawn)
    train=np.random.default_rng(cfg['train_seed']).random((cfg['train_count'],5))*[.3,.3,.3,.05,.4]+[.35,.35,.35,.10,.8]
    np.testing.assert_array_equal(cohorts['training_parameters'],train)
    refs=[];refinement=[];lo,hi=cfg['reference_intervals']
    for case,p in enumerate(params):
        fine=reference(hi,p,cfg,True);coarse=reference(lo,p,cfg,True);stride=hi//lo
        refinement.append(float(max(relative(coarse,fine[:,stride-1::stride,stride-1::stride,stride-1::stride]))))
        for n in cfg['evaluation_intervals']:
            saved=np.load(out/'fields'/f'N{n}_case{case}_reference.npz')
            physical_stride=hi//n
            physical=fine[:,physical_stride-1::physical_stride,physical_stride-1::physical_stride,physical_stride-1::physical_stride]
            same=reference(n,p,cfg,False)
            errors=[float(max(relative(saved['same_grid'],same))),float(max(relative(saved['physical'],physical)))]
            assert max(errors)<2e-12,(n,case,errors)
            refs.append(dict(intervals=n,case=case,same_grid_reference_relative=errors[0],physical_reference_relative=errors[1]))
        print('REFERENCE_AUDITED',case,flush=True)
    np.testing.assert_allclose(refinement,d['reference']['current_relative_refinement'],rtol=2e-7,atol=2e-13)
    assert d['reference']['passed']==bool(max(refinement)<cfg['reference_budget'])
    groups={};expected=set()
    for mesh in d['meshes']:
        for name in mesh['methods']:
            for case in range(cfg['validation_count']):
                for rep in range(cfg['repetitions']):expected.add((mesh['intervals'],name,case,rep))
    actual=[]
    for row in d['invocations']:
        key=(row['intervals'],row['method'],row['case'],row['repetition']);actual.append(key)
        assert row['device_ms']>0 and row['total_ms']>=row['device_ms']
        assert row['finite'],'nonfinite prediction requires explicit incomplete/failed panel disposition'
        groups.setdefault((row['intervals'],row['method']),[]).append(row)
        if 'initial_stats' in row:
            initial=np.asarray(row['initial_stats']);steps=np.asarray(row['step_stats']);choice=int(np.argmin(initial[:,3]))
            assert choice==row['selected_initial_start']
            count=int(initial[choice,2]!=1)+int(np.count_nonzero((steps[:,2]!=1)|(steps[:,5]>cfg['lm_tolerance'])))
            assert count==row['nonstationary_solves']
    assert len(actual)==len(set(actual)) and set(actual)==expected
    summaries={(r['intervals'],r['method']):r for r in json.loads((out/'summary.json').read_text())['rows']}
    for key,rows in groups.items():
        for case in range(cfg['validation_count']):
            repeated=[r for r in rows if r['case']==case]
            for row in repeated[1:]:
                for norm in ('same_grid','physical'):
                    for denominator in ('current_by_time','initial_by_time'):
                        np.testing.assert_allclose(row[norm][denominator],repeated[0][norm][denominator],rtol=2e-11,atol=2e-13)
        s=summaries[key];times=np.array([r['device_ms'] for r in rows]);med=float(np.median(times))
        cases={r['case']:r for r in rows};errors=[r['same_grid']['current_evolved'] for r in cases.values()]
        expected_values=dict(device_ms_median=med,total_ms_median=float(np.median([r['total_ms'] for r in rows])),
             same_grid_current_evolved_median=float(np.median(errors)),same_grid_current_evolved_worst=max(errors),
             physical_current_evolved_worst=max(r['physical']['current_evolved'] for r in cases.values()),
             cases_with_nonstationary_solves=sum(r['nonstationary_solves']>0 for r in cases.values()),
             time_outliers_above_1p5_median=int(np.count_nonzero(times>1.5*med)))
        for name,value in expected_values.items():np.testing.assert_allclose(s[name],value,rtol=1e-13,atol=1e-13)
        np.testing.assert_array_equal(s['device_ms_repetitions'],times)
    result=dict(passed=True,source_files_verified=source_checks,independent_scipy_reference_checks=refs,
       independent_refinement=refinement,paired_invocations=len(actual),summary_rows=len(summaries),
       complete_coverage=True,all_repetition_metrics_consistent=True,all_summary_aggregates_recomputed=True,
       final_cohort_opened=False,source_commit=commit,job_id=d['job_id'],
       scope='archived source vs git blobs, independent full numerical references, every paired invocation/summary; field-error audit retained separately')
    Path(destination).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='independent_scipy_reference_checks'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('destination');a=p.parse_args();audit(a.archive,a.destination)
