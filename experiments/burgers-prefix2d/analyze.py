"""Recompute the preregistered dev6 gate from preserved arrays and timing records."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/gate01'
DATA=RUN/'output/dev6'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ah(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    for manifest in ('OUTPUTS.sha256','MANIFEST.sha256'):
        for line in (RUN/manifest).read_text().splitlines():
            digest,name=line.split('  ',1); assert sha(RUN/name)==digest,name
    r=json.loads((DATA/'result.json').read_text()); assert r['complete'] and not r['continue_validation']
    assert r['backend']=='gpu' and r['precision']=='highest' and r['x64']
    arrays={}
    for arm in r['arms']:
        for ci in range(6):
            arrays[(arm,ci)]=dict(np.load(DATA/f'timed_identity_{arm}_case{ci}.npz'))
    for t in r['timings']:
        v=arrays[(t['arm'],t['case'])]
        assert t['identity'] and ah(v['W'])==t['W_sha256'] and ah(v['fields'])==t['fields_sha256']
        assert int(v['it_sum'])==t['fine_corrections']
    assert len(r['timings'])==126
    summary={'scope':'completed dev6 early cost screen; no validation or matched-physical-accuracy conclusion',
             'job_id':r['job_id'],'source_commit':r['commit'],'gpu':r['gpu'],
             'input_contract':'on-device sampled physical initial field; initial fit, evolution and six on-device fields included',
             'settings':r['settings'],'arms':{},'per_case':[],
             'all_fresh_step_checks_pass':all(x['valid'] for x in r['records']),
             'all_baseline_parity_checks_pass':all(x['pass'] for x in r['parity']),
             'transfer':r['transfer'],'timed_invocations':len(r['timings']),
             'strong_accuracy_checks':r['strong_accuracy_checks'],
             'continue_validation':False,'setup':r['setup'],'compile_warm_seconds':r['compile_warm_seconds'],
             'source_result_sha256':sha(DATA/'result.json'),'collected_archive_sha256':sha(RUN/'results.tgz')}
    times={(arm,ci):np.array([next(t['seconds'] for t in r['timings'] if t['arm']==arm and t['case']==ci and t['block']==b)
                             for b in range(7)]) for arm in r['arms'] for ci in range(6)}
    for arm in r['arms']:
        rows=[x for x in r['records'] if x['arm']==arm]
        ratios=[float(np.median(times[(arm,ci)]/times[('direct',ci)])) for ci in range(6)]
        vals={'median_query_ms':float(np.median([np.median(times[(arm,ci)]) for ci in range(6)])*1000),
              'paired_median_ratio':float(np.median(ratios)),
              'slowdown_percent':100*(float(np.median(ratios))-1),
              'minimum_case_ratio':min(ratios),'maximum_case_ratio':max(ratios),
              'slower_case_count':sum(x>1 for x in ratios),'case_count':6,
              'coarse_accepted_total':sum(x['coarse_accepted'] for x in rows),
              'all_production_step_checks':all(x['valid'] for x in rows)}
        for key in ('fine_corrections','fine_jacobian_pairs','coarse_corrections','coarse_jacobian_pairs'):
            vals['median_'+key]=float(np.median([x[key] for x in rows]))
        if arm!='direct':
            assert abs(vals['paired_median_ratio']-r['gate'][arm]['median_ratio'])<1e-14
            assert not r['gate'][arm]['pass'] and vals['paired_median_ratio']>.85
        summary['arms'][arm]=vals
    for ci in range(6):
        summary['per_case'].append({'case':ci,'median_query_ms':{arm:float(np.median(times[(arm,ci)])*1000) for arm in r['arms']},
             'paired_median_ratios':{arm:float(np.median(times[(arm,ci)]/times[('direct',ci)])) for arm in r['arms']}})
    accounting=(RUN/'sacct.txt').read_text().splitlines()
    header=accounting[0].split('|')
    records=[dict(zip(header,l.split('|'))) for l in accounting[1:] if l]
    job=next(x for x in records if x['JobIDRaw']==r['job_id'])
    assert job['State']=='COMPLETED' and job['ExitCode']=='0:0' and 'gres/gpu=1' in job['AllocTRES']
    sec=int(job['ElapsedRaw'])
    summary['resource_accounting']={'cluster_gpu_seconds':sec,'local_smoke_conservative_seconds':55,
          'total_charged_seconds':sec+55,'total_charged_gpu_hours':(sec+55)/3600,'cap_gpu_hours':2,
          'local_charge_is_upper_bound':True,'sacct':job}
    retention=[]
    for p in sorted(DATA.iterdir()):
        if p.is_file(): retention.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)})
    (HERE/'checks/RETENTION.json').write_text(json.dumps({'do_not_delete_unique_local_arrays':True,'files':retention,
        'archive':{'path':str((RUN/'results.tgz').relative_to(HERE)),'sha256':sha(RUN/'results.tgz')}},indent=2)+'\n')
    (HERE/'checks/SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'arms':summary['arms'],'resources':summary['resource_accounting']},indent=2))


if __name__=='__main__':main()
