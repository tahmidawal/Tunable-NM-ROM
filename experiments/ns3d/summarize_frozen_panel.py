"""Generate paper-facing rows only after completed independent panel audits."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024**2):h.update(block)
    return h.hexdigest()


def summarize(root):
    root=Path(root);out=root/'collected/output'
    raw=json.loads((out/'result.json').read_text());cfg=raw['config']
    assert raw['complete']
    names=['audit.json','source_audit.json','history_audit.json','pod_cold_audit.json']
    if raw['evaluation_cohort']=='development':names.append('teacher_audit.json')
    if raw['evaluation_cohort']=='final':names.append('protocol_audit.json')
    audits={name:json.loads((root/name).read_text()) for name in names}
    assert all(a['passed'] and a.get('complete',True) for a in audits.values()),'Every required audit must finish and pass'
    assert 'all_four_operator_families' in audits['audit.json']['checks'],'Field audit incomplete'
    records=json.loads((out/'timing_rows.json').read_text());rows=[]
    reference_passed=all(x['passed'] for x in raw['reference_refinement'])
    for name in raw['method_names']:
        arm=[r for r in records if r['method']==name]
        assert len(arm)==cfg['timed_cases']*cfg['repetitions']
        cases=[];failed_stationarity=set();reasons={}
        for case in range(cfg['timed_cases']):
            repeated=[r for r in arm if r['case']==case]
            finite=all(r['finite'] for r in repeated)
            row={'case':case,'finite':finite}
            for field in ('same_grid_errors','physical_errors'):
                prefix='same_grid' if field=='same_grid_errors' else 'physical'
                row[prefix+'_initial']=max(r[field][0] for r in repeated) if finite else None
                row[prefix+'_evolved']=max(max(r[field][1:]) for r in repeated) if finite else None
                row[prefix+'_all_times']=max(max(r[field]) for r in repeated) if finite else None
            cases.append(row)
            for invocation in repeated:
                if invocation['kind']=='weak':
                    codes=[invocation['cold'][2],*invocation['steps'][2]]
                    for code in codes:
                        key=str(int(code)) if code is not None else 'nonfinite'
                        reasons[key]=reasons.get(key,0)+1
                    if any(code!=4 for code in codes):failed_stationarity.add(case)
        complete=all(r['finite'] for r in cases)
        times=np.asarray([r['gpu_seconds'] for r in arm]);median=float(np.median(times))
        row=dict(method=name,kind=arm[0]['kind'],cases=len(cases),repetitions=cfg['repetitions'],
            gpu_ms=1000*median,with_host_ms=1000*float(np.median([r['with_host_seconds'] for r in arm])),
            input_transfer_ms=1000*float(np.median([r['input_transfer_seconds'] for r in arm])),
            timing_outlier_invocations_above_three_median=int(np.sum(times>3*median)),
            nonfinite_cases=sum(not c['finite'] for c in cases),
            physical_target_failing_cases=sum(not c['finite'] or c['physical_evolved']>cfg['physical_accuracy_target'] for c in cases),
            nonstationary_cases=len(failed_stationarity) if arm[0]['kind']=='weak' else None,
            stopping_reason_counts=reasons,reference_refinement_passed=reference_passed,case_metrics=cases)
        for field in ('same_grid','physical'):
            for scope in ('initial','evolved','all_times'):
                values=[c[field+'_'+scope] for c in cases]
                for stat,fn in (('mean',np.mean),('median',np.median),('worst',np.max)):
                    row[field+'_'+scope+'_'+stat+'_percent']=100*float(fn(values)) if complete else None
        rows.append(row)
    if raw['evaluation_cohort']=='final':
        baseline=raw['freeze']['cheapest_passing_development_fom']
    else:
        passing=[r for r in rows if r['kind']=='fom' and r['physical_target_failing_cases']==0]
        assert passing
        baseline=min(passing,key=lambda r:r['gpu_ms'])['method']
    control=next(r for r in rows if r['method']==baseline)
    for row in rows:
        row['development_selected_fom']=baseline
        row['fom_over_method_gpu']=control['gpu_ms']/row['gpu_ms']
        row['fom_over_method_with_host']=control['with_host_ms']/row['with_host_ms']
        row['matched_physical_target_and_stationarity']=bool(reference_passed and control['physical_target_failing_cases']==0 and
            row['physical_target_failing_cases']==0 and row['nonstationary_cases'] in (None,0))
    return dict(schema='ns3d-paper-panel-summary-v1',source_commit=raw['source_commit'],job_id=raw['job_id'],gpu=raw['gpu'],
        evaluation_cohort=raw['evaluation_cohort'],cohort_seed=raw['cohort_seed'],cohort_count=raw['cohort_count'],
        n=cfg['n'],k=cfg['k'],r=cfg['r'],q_values=cfg['q_values'],test_modes=cfg['test_modes'],
        target_percent=100*cfg['physical_accuracy_target'],reference_refinement_passed=reference_passed,
        timing_scope=raw['timing_scope'],timing_statistic='Median over all retained paired case/repetition invocations',
        accuracy_statistic='Per-case maximum across repetitions; evolved excludes the initial field, all-times includes it',
        ratio_scope='Same-job FOM/model timing ratio; matched-target claims require the separate accuracy, reference and stationarity flag',
        limitations=raw['limitations'],audits={name:digest(root/name) for name in names},
        raw_files={name:digest(out/name) for name in ('result.json','timing_rows.json','timed_fields.npz')},rows=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
    summary=summarize(a.run);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    columns=[k for k in summary['rows'][0] if k not in ('case_metrics','stopping_reason_counts')]
    with out.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(summary['rows'])
    print(json.dumps(dict(rows=len(summary['rows']),job_id=summary['job_id'],cohort=summary['evaluation_cohort'])))


if __name__=='__main__':main()
