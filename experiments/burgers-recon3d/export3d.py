"""Generate machine-readable case/aggregate exports from immutable screen JSON."""
import argparse
import json
from pathlib import Path
import numpy as np


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args()
    root=Path(a.input)
    raw=json.loads((root/'all_scores.json').read_text())
    execution=json.loads((root/'execution.json').read_text())
    phase=execution['phase']
    selected_file=root/'selection.json' if phase=='dev' else Path(__file__).parent/'configs/selection.json'
    selected=json.loads(selected_file.read_text())['selected']
    setup={lc:json.loads((root/f'L{lc}/plans/setup.json').read_text()) for lc in (16,32)}
    rows=[]
    for r in raw:
        lc=r['Lc'];key=f'{lc}/{r["source"]}/{r["descriptor"]["category"]}'
        times=[v for rep in r['repetitions'][1:] for v in rep]
        sens=[v for at in r['sensitivity'][1:] for v in at]
        alt=[]
        for level in range(2 if r['source']=='A' else 1):
            vals=[at[level]['error_against_finest'] for at in r['sensitivity'][1:] if len(at)>level]
            alt.append(max(vals) if vals else None)
        plan=setup[lc]['plans'].get(r['method'],{})
        row=dict(dimension=3,coarse_intervals=lc,fine_intervals=64,source=r['source'],cohort=r['cohort'],
            case=r['case'],config=r['method'],descriptor=r['descriptor'],status=r['status'],
            input_eligible=r['input_eligible'],error=r['e'],error_times=r['e_t'],reference_eps=r['eps_ref'],
            initial_error=r['initial_error'],selected=selected.get(key)==r['method'],
            physical_amplification=plan.get('physical_amplification'),
            input_sensitivity_max=max((v['prediction_gap'] for v in sens),default=None),
            error_vector_sensitivity_max=max((v['error_vector_gap'] for v in sens),default=None),
            alternative_input_errors=alt,
            approximate_pod_gap_max=max((v['gap'] for v in r['approximation_gap'] if v['time_index']>0),default=None),
            approximate_pod_gap_consequential=any(v['consequential'] for v in r['approximation_gap'] if v['time_index']>0),
            timing_policy=r['timing_policy'],timing_repetitions=len(times),
            timing_median={key:float(np.median([t[key] for t in times])) for key in
                ('fit_seconds','decode_seconds','device_query_seconds','host_query_seconds')} if times else None,
            raw_scores=f'L{lc}/{r["cohort"]}/c{r["case"]:03d}/{r["source"]}_scores.json',
            raw_coefficients=f'L{lc}/{r["cohort"]}/c{r["case"]:03d}/{r["source"]}_coefficients.npz')
        rows.append(row)
    groups={}
    for r in rows:
        key=f'{r["coarse_intervals"]}/{r["source"]}/{r["config"]}'
        groups.setdefault(key,[]).append(r)
    summaries={}
    for key,rr in groups.items():
        eligible=[r for r in rr if r['status']=='complete' and r['input_eligible']]
        vals=[r['error'] for r in eligible]
        summaries[key]=dict(cases=len(rr),eligible_cases=len(eligible),failures=len(rr)-len(eligible),
            median_error=float(np.median(vals)) if vals else None,p90_error=float(np.quantile(vals,.9)) if vals else None,
            worst_error=max(vals) if vals else None,selected=rr[0]['selected'],
            descriptor=rr[0]['descriptor'],unqualified_reference_bound_applies_only_to_full_field=True)
    comparisons=[]
    selected_rows=[r for r in rows if r['selected']]
    for lc in (16,32):
        for source in ('A','B'):
            ordinary={r['case']:r for r in selected_rows if r['coarse_intervals']==lc and r['source']==source
                      and r['descriptor']['category']=='ordinary'}
            for category in ('bank','pod_matched','pod_stronger'):
                method={r['case']:r for r in selected_rows if r['coarse_intervals']==lc and r['source']==source
                        and r['descriptor']['category']==category}
                pairs=[]
                for case,b in ordinary.items():
                    m=method[case]
                    ok=all(r['status']=='complete' and r['input_eligible'] for r in (b,m))
                    delta=None if not ok else b['error']-m['error']
                    eps=2*m['reference_eps']
                    win=bool(ok and delta>eps);loss=bool(ok and delta < -eps)
                    temporal_stable=None
                    if source=='B' and ok:
                        oldb=b['alternative_input_errors'][0];oldm=m['alternative_input_errors'][0]
                        if oldb is not None and oldm is not None:
                            temporal_stable=bool((oldb-oldm>eps)==win and (oldb-oldm < -eps)==loss)
                    pairs.append(dict(case=case,eligible=ok,ratio=m['error']/b['error'] if ok else None,
                        difference=delta,resolved_win=win,resolved_loss=loss,temporal_classification_stable=temporal_stable,
                        decoder_input_sensitivity=m['input_sensitivity_max'],ordinary_input_sensitivity=b['input_sensitivity_max']))
                good=[x for x in pairs if x['eligible']]
                comparisons.append(dict(coarse_intervals=lc,source=source,category=category,
                    median_ratio=float(np.median([x['ratio'] for x in good])) if good else None,
                    resolved_wins=sum(x['resolved_win'] for x in pairs),resolved_losses=sum(x['resolved_loss'] for x in pairs),
                    cases=len(pairs),eligible_cases=len(good),pairs=pairs,
                    interpretation='A reference-restricted observations are reference-dependent; input perturbation evidence is descriptive, not an inherited uncertainty bound.' if source=='A' else
                    'Reference resolution rule uses two eps_ref; separate previous-time-rung reconstruction checks input sensitivity.'))
    write(root/'EXPORT.json',dict(schema='burgers-reconstruction-case-export-v1',dimension=3,phase=phase,
        execution=execution,selected=selected,cases=rows,config_summary=summaries,selected_comparisons=comparisons,
        scope='No rollout/correction. No sealed/OOD. POD is approximate; matched-information extension and stronger-extra129-values comparator are distinct.'))


if __name__=='__main__':main()
