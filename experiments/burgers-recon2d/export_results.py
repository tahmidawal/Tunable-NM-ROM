"""Export a common cross-lane schema from immutable screen records; no selection/refit."""
from pathlib import Path
import argparse,hashlib,json,statistics
import analyze as N

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def export(run,prep):
    run=Path(run);prep=Path(prep); frozen=N.read(run/'FROZEN.json'); summary=N.read(run/'summary.json')
    winners={(c['L'],c['source'],c['group']):c['selected']['method'] if c['selected'] else None for c in frozen['choices']}
    plans={}
    for L in [32,64,128]:
        d=N.read(run/f'plans/L{L}/plans.json')
        for p in d['plans']+d['failed_plans']:plans[(L,p['name'])]=p
    cases={};rows=[]
    for brief in summary:
        L,cohort,case=brief['L'],brief['cohort'],brief['case'];key=f'{cohort}_c{case:03d}_L{L}'
        if key not in cases:cases[key]=N.read(prep/'coarse'/f'{key}.json')
        m=cases[key];r=N.read(run/brief['record']);p=plans[(L,r['method'])];timings=r.get('timing',[])
        t={k:[rep[k] for time in timings for rep in time['repetitions']] for k in ['host_query_seconds','device_query_seconds','fit_seconds','decode_seconds','input_seconds','output_seconds']}
        rows.append(dict(dimension=2,Lc=L,Lf=256,source=r['source'],cohort=cohort,case=case,config=r['method'],group=r['group'],desc=m['reference_meta']['desc'],status=r['status'],failure_reason=r.get('reason'),input_eligible=r['input_admitted'],actual_B_input_eligible=r['actual_B_input_admitted'],selected=winners[(L,r['source'],r['group'])]==r['method'],strict_coarse_control=r['strict_coarse_control'],max_error=r['error_max'],error_t=r['error_t'],epsilon_reference=r['ref_eps'],amplification=p.get('physical_amplification'),sampled_condition=p.get('sampled_condition'),sampled_effective_rank=p.get('sampled_effective_rank'),physical_rank=p.get('physical_rank'),timings=t,reference_level_sensitivity=r.get('sensitivity',[]) if r['source']=='A' else [],coarse_time_sensitivity=r.get('sensitivity',[]) if r['source']=='B' else [],raw=dict(record=brief['record'],record_sha256=sha(run/brief['record']),coefficients=r.get('coefficient_file'),coefficient_sha256=r.get('coefficient_file_sha256'),plan=p.get('array_file'),plan_sha256=p.get('array_sha256'),geometry=p.get('geometry_file'),geometry_sha256=p.get('geometry_sha256'),prepared_case=key,field_hashes=r.get('output_field_hashes'))))
    analysis=N.analyze(run,prep)
    paired=[]
    for choice in frozen['choices']:
        if not choice['selected']:continue
        group=choice['group']; L=choice['L'];source=choice['source']
        targets=['ordinary']+(['pod_'+group.split('_',1)[1]] if group.startswith('bank_') else [])
        for baseline_group in targets:
            base=next(c for c in frozen['choices'] if (c['L'],c['source'],c['group'])==(L,source,baseline_group))
            if not base['selected']:continue
            a={r['case']:r for r in rows if (r['Lc'],r['source'],r['config'])==(L,source,choice['selected']['method'])}
            b={r['case']:r for r in rows if (r['Lc'],r['source'],r['config'])==(L,source,base['selected']['method'])}
            ok=[(a[k],b[k]) for k in a if a[k]['status']=='complete' and b[k]['status']=='complete'];ae=[x['max_error'] for x,y in ok];be=[y['max_error'] for x,y in ok]
            paired.append(dict(Lc=L,source=source,group=group,method=choice['selected']['method'],baseline_group=baseline_group,baseline=base['selected']['method'],declared_cases=len(a),complete_pairs=len(ok),ratio_of_medians=statistics.median(be)/statistics.median(ae) if ae and statistics.median(ae)>0 else None,paired_ratio=N.stats(y['max_error']/x['max_error'] for x,y in ok if x['max_error']>0),cases=[dict(case=x['case'],method_error=x['max_error'],baseline_error=y['max_error'],input_eligible=x['input_eligible'],conditional_reference_resolved_improvement=(y['max_error']-x['max_error']>2*x['epsilon_reference']) if source=='B' else None) for x,y in ok]))
    return dict(schema_version=1,dimension=2,phase=N.read(run/'COMPLETE.json')['phase'],runtime=N.read(run/'runtime.json'),frozen_sha256=sha(run/'FROZEN.json'),source_hashes=N.read(run/'SOURCES.json'),rows=rows,config_summary=analysis['aggregates'],paired_selected=paired,input_qualification=analysis['input_qualification'],notes=analysis['measurement_notes']+['Source A comparisons describe completion of stored arrays; physical interpretation requires retained error-vector level sensitivity.','Primary ratio is ratio of case medians; paired ratios are secondary. Failed cases remain declared denominator entries.'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--prep',required=True);a=ap.parse_args();N.write(Path(a.run)/'EXPORT.json',export(a.run,a.prep))
if __name__=='__main__':main()
