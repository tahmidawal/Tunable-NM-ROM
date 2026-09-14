"""Generate staged-training development panels only after complete source/field audit."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tuning_summary import summarize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve();d=json.loads((run/'result.json').read_text());audit=json.loads((run/'audit.json').read_text());assert audit['passed'] and d['complete'];cfg=d['config']
    panel=dict(pde='poisson2d',status='audited staged-training development comparison; no sealed final cases',source_commit=d['provenance']['commit'],job_id=d['provenance']['job_id'],provenance=d['provenance'],config=cfg,
        run_path=str(run),source_result_path=str(run/'result.json'),audit_path=str(run/'audit.json'),archive_path=str(run/'ARCHIVE.json'),cleanup_path=str(run/'cleanup.json'),
        training=d['training'],checkpoints=d['checkpoints'],cohort=d['cohort'],timed_invocations=len(d['rows']),timing_statistic='pooled median over all equal-count case/repetition samples',
        error_definition='worst current-relative full-field L2 versus restricted 2048 FD-DST truth; empirical 1024 reference allowance retained separately',
        groups=[],limitations=['Development cohorts only; fixed staged endpoints rather than development-selected checkpoints','Optimizer-loop time matched on one GPU within a declared block; not equal FLOPs or a convergence proof','SVD/POD and reference-only bank/head fits are diagnostics, never online models or initial guesses','All preassembled operators and checkpoint-specific setup are offline and reported separately','No universal FOM advantage; direct DST remains a named control'])
    for group in ['all','existing_development','new_development']:
        cases=[i for i,g in enumerate(d['cohort']['groups']) if group=='all' or g==group]
        if not cases:continue
        entry=dict(group=group,cases=len(cases),meshes=[])
        for n in cfg['intervals']:
            rows=[r for r in d['rows'] if r['case'] in cases and r['intervals']==n];methods=summarize(rows,cfg);bank={};head={}
            for model in cfg['trained_model_ids']:
                rs=[r for r in rows if r['method']==model];m=methods[model];m.update(checkpoint_sha256=rs[0]['checkpoint_sha256'],requested_modes=rs[0]['requested_modes'],retained_modes=rs[0]['retained_modes'],all_bank_ranks_valid=all(r['bank_rank_valid'] for r in rs));m['all_stationary']=m['all_stationary'] and m['all_bank_ranks_valid']
                m['gpu_speedup_vs_frozen']=methods['original_relative']['gpu_median_ms']/m['gpu_median_ms'];m['host_speedup_vs_frozen']=methods['original_relative']['host_median_ms']/m['host_median_ms']
                o=[o for o in d['oracles'] if o['model']==model and o['intervals']==n and o['case'] in cases];bank[model]=dict(worst_same_grid_relative_error=max(x['same_grid_relative_error'] for x in o),worst_physical_error=max(x['physical_error'] for x in o),all_ranks_valid=all(x['rank_valid'] for x in o))
                h=[h for h in d['head_oracles'] if h['model']==model and h['intervals']==n and h['case'] in cases]
                if h:
                    valid=[x for x in h if x['best_found_stationary_index'] is not None]
                    head[model]=dict(cases=len(h),cases_without_stationary_fit=len(h)-len(valid),worst_stationary_same_grid_error=max(x['candidates'][x['best_found_stationary_index']]['same_grid_relative_error'] for x in valid) if len(valid)==len(h) else None,
                        worst_best_any_same_grid_error=max(min(c['same_grid_relative_error'] for c in x['candidates']) for x in h),candidate_stop_reasons=[c['reason'] for x in h for c in x['candidates']])
            passing=[m for m in methods if m.startswith('cg_') and methods[m]['all_cases_pass_target']];cg=min(passing,key=lambda m:methods[m]['gpu_median_ms']) if passing else None
            for model in cfg['trained_model_ids']:
                m=methods[model];m['speedup_vs_tight_cg']=methods['cg_1e-06']['gpu_median_ms']/m['gpu_median_ms'];m['speedup_vs_fastest_passing_cg']=methods[cg]['gpu_median_ms']/m['gpu_median_ms'] if cg else None;m['speedup_vs_dst']=methods['dst']['gpu_median_ms']/m['gpu_median_ms']
            entry['meshes'].append(dict(intervals=n,methods=methods,bank_diagnostics=bank,head_diagnostics=head,fastest_passing_cg=cg))
        panel['groups'].append(entry)
    (run/'panel.json').write_text(json.dumps(panel,indent=2)+'\n');print(run/'panel.json')


if __name__=='__main__':main()
