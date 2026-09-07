"""Independently audit every saved full/common-grid rollout error and repetition.

Only NumPy and committed artifacts are used; no new PDE solve or GPU work.
"""
import argparse,collections,functools,hashlib,itertools,json,subprocess
from pathlib import Path
import numpy as np


def fingerprint(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def errors(f,t):
    n0=np.linalg.norm(t[0]);norm=np.linalg.norm(t.reshape(len(t),-1),axis=1)
    delta=np.linalg.norm((f-t).reshape(len(f),-1),axis=1)
    return dict(fixed_initial_per_time=delta/n0,current_relative_per_time=delta/np.maximum(norm,1e-300))
def max_delta(found,saved):return max(float(np.max(np.abs(v-np.asarray(saved[k])))) for k,v in found.items())


def main():
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run);out=run/'out'
    d=json.loads((out/'pilot.json').read_text());root=Path(__file__).resolve().parents[3]
    assert d['complete'] and d['experiment'] in ['fixed_gauss_rollout','gauss_timestep_study']
    assert d['backend']=='gpu' and d['x64'] and d['matmul_precision']=='highest' and d['network_weights_frozen']
    assert d['checkpoint_sha256']==d['checkpoint_sha256_after']
    assert d['commit']==(run/'COMMIT.txt').read_text().strip() and str(d['job_id'])==(run/'JOB_ID.txt').read_text().strip()
    for r in json.loads((run/'PROVENANCE.json').read_text()):
        data=subprocess.check_output(['git','show',r['commit']+':'+r['source']],cwd=root)
        assert hashlib.sha256(data).hexdigest()==r['sha256'],r['source']
    for line in (run/'COLLECT.sha256').read_text().splitlines():
        h,file=line.split(maxsplit=1);assert fingerprint(run/file)==h,file
    stdout='\n'.join(x.read_text() for x in (run/'logs').glob('*.out'))
    stderr='\n'.join(x.read_text() for x in (run/'logs').glob('*.err'))
    assert not stderr.strip(),stderr
    assert 'jax_backend=gpu' in stdout and 'ALL-DONE' in stdout
    meshes=list(map(int,d['config']['meshes'].split(',')));cases=range(d['config']['cases']);reps=range(d['config']['reps'])
    declared={}
    for L in meshes:
        for arm in d['arm_grid']:
            rule,icb,dt,stall=arm['cold_rule'],arm['ic_budget'],arm['dt'],arm['stall']
            name=f'rom_L{L}_{rule}_ic{icb}_dt{dt}_stall{stall}_starts1';declared[name]=L
        for grid in [g for g in [128,256,512,1024] if g<=L]:
            for dt,ntol in [(.01,.01),(.01,.003),(.005,.01),(.005,.003),(.0025,.003)]:
                declared[f'fom_L{grid}_out{L}_dt{dt}_ntol{ntol}']=L
    assert len(d['declared_subjects'])==len(declared) and {r['name'] for r in d['declared_subjects']}==set(declared)
    expected=set(itertools.product(declared,cases,reps));actual=[(r['name'],r['case'],r['rep']) for r in d['invocations']]
    assert len(actual)==len(expected) and set(actual)==expected
    orders={(r['intervals'],r['case'],r['rep']):r['names'] for r in d['timing_order']}
    assert len(orders)==len(d['timing_order'])==len(meshes)*len(cases)*len(reps)
    for (L,case,rep),names in orders.items():
        assert len(names)==len(set(names)) and set(names)=={name for name,n in declared.items() if n==L}
        assert names==[r['name'] for r in d['invocations'] if r['output_intervals']==L and r['case']==case and r['rep']==rep]
    physical=np.asarray(d['physical_cases']);rng=np.random.default_rng(d['config']['seed']);n=len(cases)
    regenerated=np.stack([rng.uniform(.15,.85,n),rng.uniform(.15,.85,n),rng.uniform(.05,.2,n),rng.uniform(.5,2,n),np.exp(rng.uniform(np.log(.01),np.log(.1),n))],1)
    assert np.allclose(physical,regenerated,rtol=2e-15,atol=0)
    obs=d['observation_intervals'];dense_obs=d['dense_observation_intervals'];Rf=d['config']['reference_mesh'];dtr=d['config']['reference_dt']
    @functools.lru_cache(maxsize=6)
    def reference(L,dt,case):
        z=np.load(out/f'ref_L{L}_dt{dt}_case{case}.npz');f=z['dense_fields']
        assert f.shape==(6,dense_obs+1,dense_obs+1) and f.dtype==np.float64 and np.isfinite(f).all()
        assert np.max(z['residuals'])<2e-11
        assert np.array_equal(z['fields'],f[:,::dense_obs//obs,::dense_obs//obs])
        return f
    settings=[(Rf//2,dtr),(Rf,2*dtr),(Rf,dtr),(Rf//4,2*dtr),(Rf//2,2*dtr),(Rf,4*dtr)]
    assert len(d['reference'])==len(settings)*len(cases)
    assert {(r['intervals'],r['dt'],r['case']) for r in d['reference']}==set(itertools.chain.from_iterable(((L,dt,c) for c in cases) for L,dt in settings))
    ref_delta=0.
    for L,dt in settings:
        for case in cases:reference(L,dt,case)
    assert set(d['reference_metrics_by_output'])=={str(L) for L in meshes+[obs]}
    for target in sorted(set(meshes+[obs])):
        metrics=d['reference_metrics_by_output'][str(target)]
        for case in cases:
            def field(L,dt):return reference(L,dt,case)[:,::dense_obs//target,::dense_obs//target]
            def diff(L1,t1,L2,t2):return float(max(errors(field(L1,t1),field(L2,t2))['fixed_initial_per_time']))
            space=diff(Rf//2,dtr,Rf,dtr);temporal=diff(Rf,2*dtr,Rf,dtr)
            un=next(r for r in metrics['uncertainty'] if r['case']==case)
            ref_delta=max(ref_delta,abs(space+temporal-un['conservative_difference_sum']))
            sc=diff(Rf//4,2*dtr,Rf//2,2*dtr);sf=diff(Rf//2,2*dtr,Rf,2*dtr);tc=diff(Rf,4*dtr,Rf,2*dtr)
            ps=float(np.log2(sc/sf));pt=float(np.log2(tc/temporal));order=next(r for r in metrics['order_audit'] if r['case']==case)
            ref_delta=max(ref_delta,abs(ps-order['observed_spatial_order']),abs(pt-order['observed_temporal_order']))
            if ps>0 and pt>0:ref_delta=max(ref_delta,abs(sf/(2**ps-1)+temporal/(2**pt-1)-order['empirical_richardson_estimate']))
    assert ref_delta<1e-11
    dense_groups=collections.defaultdict(list)
    for r in d['invocations']:dense_groups[r['dense_artifact']].append(r)
    dense_delta=common_delta=same_delta=analytic_delta=0.;ic_counts=collections.Counter();lm_counts=collections.Counter();fom_failures=nonfinite=0
    for artifact,rows in dense_groups.items():
        r=rows[0];L=r['output_intervals'];case=r['case'];f=np.load(out/artifact)['fields']
        assert f.shape==(6,L+1,L+1) and f.dtype==np.float64
        sha=hashlib.sha256(f.tobytes()).hexdigest();finite=bool(np.isfinite(f).all())
        t=reference(Rf,dtr,case)[:,::dense_obs//L,::dense_obs//L]
        x=np.arange(L+1)/L;xx,yy=np.meshgrid(x,x,indexing='ij');cx,cy,w,amp,_=physical[case]
        analytic=amp*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*w*w));analytic[[0,-1],:]=0.;analytic[:,[0,-1]]=0.
        analytic_delta=max(analytic_delta,float(np.max(np.abs(t[0]-analytic))))
        assert np.max(np.abs(f[:,[0,-1],:]))==np.max(np.abs(f[:,:,[0,-1]]))==0.
        computed=errors(f,t) if finite else None
        for r in rows:
            assert r['output_intervals']==L and r['case']==case and r['field_sha256']==sha
            z=np.load(out/r['observation_artifact']);fc=z['fields'];assert np.array_equal(fc,f[:,::L//obs,::L//obs],equal_nan=True)
            assert np.array_equal(z['iterations'],r['iterations'])
            assert np.array_equal(z['residuals'],np.asarray(r['residuals'],dtype=float),equal_nan=True)
            assert len(r['iterations'])==len(r['residuals'])==round(d['output_times'][-1]/r['dt'])
            assert r['seconds']>0 and r['output_bytes']==f.nbytes
            nonfinite+=not r['finite']
            if r['finite']:
                assert finite and np.isfinite(z['residuals']).all()
                dense_delta=max(dense_delta,max_delta(computed,r['dense_physical_error']))
                common_delta=max(common_delta,max_delta(errors(fc,t[:,::L//obs,::L//obs]),r['physical_error']))
            if r['method']=='fom':
                valid=bool(r['finite'] and np.max(z['residuals'])<=r['newton_tolerance']*(1+1e-9))
                assert valid==r['nonlinear_tolerance_satisfied'];fom_failures+=not valid
            else:
                ic_counts[r['ic_reason']]+=1;lm_counts.update(r['stop_reasons'])
                assert r['ic_iterations']<=r['ic_budget']
                if r['ic_reason']==0:assert r['ic_iterations']==r['ic_budget']
                assert len(r['stop_reasons'])==len(r['iterations'])
                for reason,it in zip(r['stop_reasons'],r['iterations']):
                    assert reason in range(4) and 0<=it<=30
                    if reason==0:assert it==30
                st=np.load(out/f"same_L{L}_dt{r['dt']}_case{case}.npz")
                assert np.max(st['residuals'])<2e-11
                if r['finite']:
                    same_delta=max(same_delta,max_delta(errors(f,st['fields']),r['dense_same_grid_error']),max_delta(errors(fc,st['fields'][:,::L//obs,::L//obs]),r['same_grid_error']))
    assert max(dense_delta,common_delta,same_delta,analytic_delta)<1e-12
    for name,case in itertools.product(declared,cases):
        group=[r for r in d['invocations'] if r['name']==name and r['case']==case];first=next(r for r in group if r['rep']==0)
        for r in group:assert r['matches_first_dense_sha256']==(r['field_sha256']==first['field_sha256'])
    profiles=d.get('component_profile_configs',[dict(cold_rule='edge',ic_budget=60,dt=.005,stall=.01),dict(cold_rule='fixed_gauss',ic_budget=180,dt=.005,stall=.01)])
    keys=['cold_rule','ic_budget','dt','stall']
    expected_profiles={(L,c,*(p[k] for k in keys)) for L,c,p in itertools.product(meshes,cases,profiles)}
    actual_profiles=[(r['intervals'],r['case'],*(r.get(k,.005 if k=='dt' else .01) for k in keys)) for r in d['component_profiles']]
    assert len(actual_profiles)==len(expected_profiles) and set(actual_profiles)==expected_profiles
    assert max(r['fused_output_relative_difference'] for r in d['component_profiles'])<1e-8
    result=dict(source_sha256=fingerprint(out/'pilot.json'),output_checksums_verified=True,closed_logs_clean=True,source_hashes_against_commits_verified=True,
        checkpoint_unchanged=True,seed_and_family_verified=True,declared_configurations_verified=len(declared),invocations_verified=len(actual),
        full_grid_artifacts_verified=len(dense_groups),common_grid_error_recompute_max_difference=common_delta,dense_error_recompute_max_difference=dense_delta,
        same_grid_error_recompute_max_difference=same_delta,reference_metric_recompute_max_difference=ref_delta,analytic_initial_max_difference=analytic_delta,
        fom_nonlinear_tolerance_failures=int(fom_failures),nonfinite_invocations=int(nonfinite),ic_stop_reason_counts=dict(ic_counts),lm_stop_reason_counts=dict(lm_counts),
        full_output_repetitions_matching_first=sum(r['matches_first_dense_sha256'] for r in d['invocations']),
        note='All complete requested-grid norms are independently reconstructed from actual saved outputs; repeated outputs are linked by exact full-field hashes. Reference qualification remains empirical; configured stalls are not stationarity certificates.')
    (run/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
