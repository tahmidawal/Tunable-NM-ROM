"""Complete machine panel for the coordinator's source-generated tables."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tuning_summary import summarize,select


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve();d=json.loads((run/'result.json').read_text());audit=json.loads((run/'audit.json').read_text());assert audit['passed'];cfg=d['config']
    panel=dict(pde='poisson2d',status='audited development tuning; fixed checkpoint, no final cohort',provenance=d['provenance'],config=cfg,presets=cfg['presets'],source_commit=d['provenance']['commit'],checkpoint_sha256=cfg['checkpoint_sha256'],job_id=d['provenance']['job_id'],
        source_result_path=str(run/'result.json'),run_path=str(run),audit_path=str(run/'audit.json'),archive_path=str(run/'ARCHIVE.json'),cleanup_path=str(run/'cleanup.json'),
        cohort_count=len(d['cohort']['parameters']),timed_invocations=len(d['rows']),timing_statistic='median of all retained case/repetition samples; equal repetition counts',
        error_definition='worst current-relative full-field L2 against restricted 2048 FD-DST reference over every development case and repetition',adjusted_error_definition='(physical_error+empirical_reference_delta)/(1-empirical_reference_delta)',
        target=cfg['development_target'],stationarity_tolerance=cfg['stationarity_tolerance'],selection_rules=cfg['selection_rules'],frozen_screen_selection=d['selection'],
        method_exit_labels=d['contract']['stopping_reasons'],phases=[],limitations=['Selections are cohort-wide, never per-case truth chosen','Coarse-screen shortlist is frozen before confirmation timing; per-mesh winners are selections within that shortlist','More accurate refers to complete-cohort stationarity at existing tolerance; unrestricted physical-error minimum is a separate diagnostic','No globally optimal setting or final held-out claim','CG is an iterative comparator; direct DST is retained and may be faster','Post-query stationarity audits are excluded from both online timing measures; any enabled in-loop stationarity stop is charged'])
    ids=[x['id'] for x in cfg['presets']]
    for phase,meshes in [('screen',[cfg['screen_intervals']]),('confirmation',cfg['intervals'])]:
        for n in meshes:
            rows=[r for r in d['rows'] if r['phase']==phase and r['intervals']==n];methods=summarize(rows,cfg);presets=[x for x in ids if x in methods]
            for m in presets:
                sample=next(r for r in rows if r['method']==m);methods[m].update(preset=sample['preset'],requested_modes=sample['requested_modes'],retained_modes=sample['retained_modes'])
            selections=select(methods,presets);passing=[m for m in methods if m.startswith('cg_') and methods[m]['all_cases_pass_target']];cg=min(passing,key=lambda x:methods[x]['gpu_median_ms']) if passing else None
            for m in presets:
                item=methods[m];item.update(speedup_vs_tight_cg=methods['cg_1e-06']['gpu_median_ms']/item['gpu_median_ms'],speedup_vs_fastest_passing_cg=methods[cg]['gpu_median_ms']/item['gpu_median_ms'] if cg else None,speedup_vs_dst=methods['dst']['gpu_median_ms']/item['gpu_median_ms'])
            panel['phases'].append(dict(phase=phase,intervals=n,methods=methods,selections=selections,fastest_passing_cg=cg,selection_status='coarse development screen' if phase=='screen' else 'development extrema within frozen coarse shortlist'))
    (run/'panel.json').write_text(json.dumps(panel,indent=2)+'\n')
    print(json.dumps({f"{p['phase']}_{p['intervals']}":p['selections'] for p in panel['phases']},indent=2))


if __name__=='__main__':main()
