"""Generate descriptive screen summaries from persisted records, without refitting."""
from pathlib import Path
from collections import defaultdict
import argparse,json,statistics,math

def stats(values):
    v=sorted(float(x) for x in values if x is not None and math.isfinite(x))
    if not v:return dict(n=0)
    return dict(n=len(v),median=statistics.median(v),minimum=v[0],maximum=v[-1])
def read(path): return json.loads(Path(path).read_text())
def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def analyze(run,prep):
    run=Path(run); prep=Path(prep); summary=read(run/'summary.json'); frozen=read(run/'FROZEN.json')
    groups=defaultdict(list)
    for row in summary:groups[(row['L'],row['source'],row['method'])].append(row)
    aggregates=[]
    for (L,source,method),rows in sorted(groups.items()):
        valid=[r for r in rows if r['status']=='complete']; host=[]; device=[]; fit=[]; decode=[]; transfer=[]
        for row in valid:
            record=read(run/row['record'])
            for t in record['timing']:
                for rep in t['repetitions']:
                    host.append(rep['host_query_seconds']); device.append(rep['device_query_seconds']); fit.append(rep['fit_seconds']); decode.append(rep['decode_seconds']); transfer.append(rep['input_seconds']+rep['output_seconds'])
        aggregates.append(dict(L=L,source=source,method=method,group=rows[0]['group'],cases=len(rows),failed=len(rows)-len(valid),coarse_inputs_unqualified=sum(not r['coarse_input_admitted'] for r in rows),error_max_over_evolved_times=stats(r['error_max'] for r in valid),steady_warm_single_field_seconds=dict(host=stats(host),device=stats(device),fit=stats(fit),decode=stats(decode),transfer=stats(transfer))))
    paired=[]
    for choice in frozen['choices']:
        L=choice['L']; source=choice['source']; group=choice['group']; pick=choice['selected']
        base=next(c for c in frozen['choices'] if c['L']==L and c['source']==source and c['group']=='ordinary')['selected']
        if pick is None or base is None: continue
        a={r['case']:r for r in groups[(L,source,pick['method'])]}; b={r['case']:r for r in groups[(L,source,base['method'])]}
        pairs=[]
        for case,x in sorted(a.items()):
            y=b[case]
            if x['status']!='complete' or y['status']!='complete': pairs.append(dict(case=case,status='failed')); continue
            delta=y['error_max']-x['error_max']; margin=2*x['ref_eps'] if source=='B' else 0.
            pairs.append(dict(case=case,status='complete',method_error=x['error_max'],baseline_error=y['error_max'],gain=y['error_max']/x['error_max'] if x['error_max']>0 else None,relative_degradation=x['error_max']/y['error_max'] if y['error_max']>0 else None,improvement=delta,reference_comparison_margin=margin,resolved_improvement=bool(delta>margin),coarse_input_admitted=x['coarse_input_admitted']))
        ok=[x for x in pairs if x['status']=='complete']
        paired.append(dict(L=L,source=source,group=group,method=pick['method'],baseline=base['method'],gain=stats(x['gain'] for x in ok),worst_relative_degradation=max((x['relative_degradation'] for x in ok),default=None),wins20pct=sum(x['gain'] is not None and x['gain']>=1.25 for x in ok),losses10pct=sum(x['relative_degradation'] is not None and x['relative_degradation']>1.1 for x in ok),resolved_improvements=sum(x['resolved_improvement'] for x in ok),cases=pairs))
    qualification=[]
    for L in [32,64,128]:
        rr=[r for r in read(prep/'coarse/summary.json') if r['L']==L]; gaps=[]
        for row in rr:gaps.append(read(prep/'coarse'/f"{row['key']}.json")['levels'][-1]['relative_refinement_gap'])
        qualification.append(dict(L=L,total=len(rr),admitted=sum(r['accepted'] for r in rr),last_pair_gap=stats(gaps),failed_keys=[r['key'] for r in rr if not r['accepted']]))
    return dict(source_run=str(run.resolve()),phase=read(run/'COMPLETE.json')['phase'],aggregates=aggregates,paired_selected=paired,strict_coarse_control=frozen['strict_coarse_control'],input_qualification=qualification,measurement_notes=['Field errors use the fixed fine-reference initial-field norm and maximum over five evolved times per case.','Latency is steady-warm one-field service, all timings from the same invocation as stored errors.','Selection used exposed development cases; validation remains exposed, not sealed.','Training is regenerated from original recipe and checked fingerprints, not bitwise-identical to archived data.','POD is fixed-seed approximate uncentered POD; tail eigen residuals are reported separately.'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--prep',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();write(a.out,analyze(a.run,a.prep))
if __name__=='__main__':main()
