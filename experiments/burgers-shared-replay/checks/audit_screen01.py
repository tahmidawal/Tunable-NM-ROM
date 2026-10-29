"""Post-run saved-evidence audit; NumPy only, with no decoder/PDE imports."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

LANE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LANE))
from analyze import load


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    run=LANE/'runs/screen01'; out=run/'output/dev6'
    result=json.loads((out/'result.json').read_text())
    summary=json.loads((LANE/'checks/screen01-summary.json').read_text())
    verified={}
    for name in ('MANIFEST.sha256','OUTPUTS.sha256'):
        lines=(run/name).read_text().splitlines()
        for line in lines:
            digest,path=line.split(maxsplit=1)
            assert sha(run/path)==digest,path
        verified[name]=dict(entries=len(lines),sha256=sha(run/name))
    assert result['complete'] and not result['continue_accuracy']
    assert result['settings']==dict(L=1024,K_fine=64,K_coarse=16,R=512,M=256,rule='gauss128',
        scheme='BE',dt=.005,steps=50,budget=600,gtol=1e-3,tolf=1e-9,repetitions=7,
        order_seed=20261028,coarse_preview_fields=6,fine_fields=6)
    assert result['model_manifest_sha256']==sha(run/'model/MODEL.json')
    assert result['commit']==(run/'COMMIT.txt').read_text().strip()
    assert result['backend']=='gpu' and result['x64'] and result['precision']=='highest'
    log=(run/'logs/6619863.out').read_text(); err=(run/'logs/6619863.err').read_text()
    assert not err and 'jax_backend=gpu x64=True precision=highest' in log and log.endswith('ALL-DONE\n')
    accounting=(run/'ACCOUNTING.txt').read_text().splitlines()
    row=next(line.split('|') for line in accounting if line.startswith(result['job_id']+'|'))
    assert row[1]=='COMPLETED' and row[-1]=='0:0'
    timing={(t['arm'],t['case'],t['block']):t for t in result['timings']}
    assert len(timing)==84 and all(t['identity'] and t['online_valid'] for t in timing.values())
    assert all(r['valid'] and r['timed_production_stationarity'] for r in result['records'])
    integers=('steps','it_sum','it_max','rej','exits','nfail','first_fail','step_reason','step_it','ic_it','ic_reason','ic_idx')
    gamma=[]; cases=[]; accepted=[]; attempts=0
    for c in range(6):
        records={arm:load(out/f'{arm}_case{c}.npz') for arm in ('direct','replay')}
        for arm,z in records.items():
            w=load(out/f'timed_identity_{arm}_case{c}.npz')
            for key in ('W','fields','w0'):
                np.testing.assert_allclose(w[key],z[key],rtol=1e-12,atol=1e-13)
            for prefix in ('','coarse_') if arm=='replay' else ('',):
                for key in integers:
                    assert np.array_equal(w[prefix+key],z[prefix+key],equal_nan=True),(arm,c,prefix,key)
                g=z[prefix+'gamma_fresh']; rn=z[prefix+'residual_norm_fresh']; tol=z[prefix+'residual_tolerance']
                np.testing.assert_allclose(w[prefix+'step_gn'],g,rtol=1e-7,atol=1e-12)
                np.testing.assert_allclose(w[prefix+'step_tolratio'],rn/(tol+1e-300),rtol=1e-7,atol=1e-12)
                assert np.isfinite(g).all() and np.isfinite(rn).all() and np.all((g<=1e-3)|(rn<=tol))
                gamma.extend(g.tolist())
                assert z[prefix+'ic_reason'] in (1,4) and np.isfinite(z[prefix+'ic_gn']) and np.isfinite(z[prefix+'ic_rn'])
                assert np.array_equal(z[prefix+'exits'],np.array([0,0,0,0,50]))
            if arm=='replay':
                for key in ('coarse_W','coarse_fields','coarse_w0','coarse_states','coarse_replay_valid'):
                    np.testing.assert_allclose(w[key],z[key],rtol=1e-12,atol=1e-13)
                assert np.array_equal(w['replay_info'][:,:2],z['replay_info'][:,:2])
                info=z['replay_info']; assert np.all(info[:,1]==1) and np.all(info[:,4]==1)
                accept=info[:,0].astype(bool); attempted=len(accept)
                assert np.all(info[accept,3] < info[accept,2])
                assert np.all(info[accept,3] <= (1-1e-8)*info[accept,2])
                accepted.append(int(accept.sum())); attempts+=attempted
        a=np.array([timing['direct',c,b]['seconds'] for b in range(7)])
        b=np.array([timing['replay',c,b]['seconds'] for b in range(7)])
        cases.append(dict(case=c,direct_median_ms=float(np.median(a)*1000),replay_median_ms=float(np.median(b)*1000),
            paired_median_ratio=float(np.median(b/a)),paired_ratio_min=float(min(b/a)),paired_ratio_max=float(max(b/a)),
            slower_pairs=int(sum(b>a)),pairs=7,accepted_candidates=accepted[-1]))
    forced=[]
    for item in result['parity']:
        assert item['passed']
        if item['check']!='forced_coarse_failure':continue
        c,k=item['case'],item['step']; z=load(out/f'forced_failure_case{c}_step{k}.npz')
        normal=load(out/f'replay_case{c}.npz')
        np.testing.assert_allclose(z['coarse_states'][:k+1],normal['coarse_states'][:k+1],rtol=1e-12,atol=1e-13)
        assert int(z['coarse_valid_steps'])==k and int(z['coarse_steps'])==k+1 and int(z['coarse_nfail'])==1
        assert int(z['coarse_step_reason'][k])==3 and np.isnan(z['coarse_states'][k+1:]).all()
        available=z['coarse_output_available']; assert np.array_equal(available,np.arange(6)*10<=k)
        # Decoder padding keeps boundary walls zero even when the interior is unavailable.
        assert np.isnan(z['coarse_fields'][~available,1:-1,1:-1]).all()
        assert np.isfinite(z['coarse_fields'][available]).all()
        assert np.all(z['coarse_fields'][:,[0,-1],:]==0) and np.all(z['coarse_fields'][:,:,[0,-1]]==0)
        assert not np.any(z['replay_info'][k:,0]) and not np.any(z['replay_info'][k:,1])
        if k==0:
            direct=load(out/f'direct_case{c}.npz')
            for key in ('states','W','fields'):np.testing.assert_allclose(z[key],direct[key],rtol=1e-12,atol=1e-13)
            for key in integers:assert np.array_equal(z[key],direct[key],equal_nan=True)
        forced.append(dict(case=c,failed_step=k,accepted_coarse_steps=k,later_guesses_disabled=True))
    assert len(gamma)==900 and len(forced)==2
    totals={arm:dict(fine_corrections=sum(summary['arms'][arm]['fine_corrections']),
        fine_physics_jacobians=summary['arms'][arm]['fine_physics_J'],
        fine_ic_iterations=sum(summary['arms'][arm]['fine_ic_iterations'])) for arm in ('direct','replay')}
    totals['replay'].update(coarse_corrections=sum(summary['arms']['replay']['coarse_corrections']),
        coarse_physics_jacobians=summary['arms']['replay']['coarse_physics_J'],
        coarse_ic_iterations=sum(summary['arms']['replay']['coarse_ic_iterations']),
        candidate_offers=attempts,accepted_candidates=sum(accepted),rejected_candidates=attempts-sum(accepted))
    audit=dict(disposition='ACCEPT cost failure and stop',source_result_sha256=sha(out/'result.json'),
        manifests=verified,job_id=result['job_id'],allocated_gpu_seconds=int(row[2]),gpu=result['gpu'],
        paired_timing_cases=cases,totals=totals,all_fresh_certificates=900,max_fresh_gamma=max(gamma),
        forced_failure_controls=forced,case_medians_above_1p10=sum(c['paired_median_ratio']>1.10 for c in cases),
        warnings='none; stderr empty; complete log',limitations=summary['strong_accuracy_checks'],
        interpretation='Complete coarse trajectory plus replay was slower. Valid production stationarity is not tight-baseline or physical-accuracy agreement.')
    (LANE/'checks/screen01-audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
