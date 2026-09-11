"""Generate the coordinator panel from audited invocation records; no typed data."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve()
    d=json.loads((run/'result.json').read_text());audit=json.loads((run/'audit.json').read_text());assert audit['passed']
    cfg=d['config'];allrows=d['rows'];methods=['nmrom']+[f'cg_{t:.0e}' for t in cfg['cg_tolerances']]+['dst']
    panel=dict(pde='poisson2d',status='audited development comparison',provenance=d['provenance'],source_checkpoint_sha256=cfg['checkpoint_sha256'],
        source_commit=d['provenance']['commit'],job_id=d['provenance']['job_id'],run_path=str(run),audit_path=str(run/'audit.json'),config=cfg,
        cohort_count=len(d['cohort']['parameters']),parameters=d['cohort']['parameters'],timed_invocations=len(allrows),
        timing_statistic='median across cases of case median repetitions',paired_speedup_statistic='median across cases of FOM/ROM case median times',
        error_definition='Euclidean relative L2 on full requested nodal mesh, denominator finest 2048-interval FD-DST field restricted to that mesh; worst over every case and repetition',
        adjusted_error_definition='(physical_error + reference_delta)/(1-reference_delta); empirical finite-difference refinement, not a rigorous bound',
        primary_timing='GPU-resident supplied field through complete decoded/full-order GPU output, plus solver diagnostics',
        host_timing='same invocation including synchronized full source and solution transfers plus solver diagnostics',
        outlier_definition='a repetition above 1.5 times its own method/mesh/case median; all retained',
        meshes=[],limitations=['Already-open development sources, one training seed, no final cohort','ROM weights frozen across meshes; no per-mesh retraining',
            'Iterative CG is user-selected comparator; direct DST remains faster in the earlier study and is rerun as a diagnostic',
            'Classical tolerance ladder is finite and same-grid; no globally optimized FOM frontier','Same-grid and common64-observation error are separate diagnostics'])
    for n in cfg['intervals']:
        mesh=dict(intervals=n,nodes_per_axis=n+1,interior_unknowns=(n-1)**2,methods={});case_times={}
        for method in methods:
            rows=[r for r in allrows if r['intervals']==n and r['method']==method]
            groups={case:[r for r in rows if r['case']==case] for case in range(panel['cohort_count'])}
            gpu=np.array([np.median([r['fused_device_seconds'] for r in g]) for g in groups.values()])*1000
            host=np.array([np.median([r['total_seconds'] for r in g]) for g in groups.values()])*1000
            case_times[method]=(gpu,host)
            physical=np.array([max(r['physical_error'] for r in g) for g in groups.values()]);adjusted=np.array([max(r['conservative_physical_error'] for r in g) for g in groups.values()])
            eligible=[all(r['solver_valid'] for r in g) and max(r['conservative_physical_error'] for r in g)<=cfg['development_target'] and max(r['reference_delta'] for r in g)<=cfg['development_target']*cfg['reference_fraction'] for g in groups.values()]
            item=dict(gpu_median_ms=float(np.median(gpu)),host_median_ms=float(np.median(host)),case_gpu_medians_ms=gpu.tolist(),case_host_medians_ms=host.tolist(),
                median_relative_error=float(np.median(physical)),worst_relative_error=float(physical.max()),worst_error_case=int(np.argmax(physical)),
                worst_adjusted_relative_error=float(adjusted.max()),worst_same_grid_error=max(r['same_grid_error'] for r in rows),
                worst_common64_observation_error=max(r['common_observation_error'] for r in rows),maximum_reference_delta=max(r['reference_delta'] for r in rows),
                development_target=cfg['development_target'],all_cases_pass_target=all(eligible),target_failed_cases=sum(not x for x in eligible),
                invalid_invocations=sum(not r['solver_valid'] for r in rows),invalid_cases=sum(not all(r['solver_valid'] for r in g) for g in groups.values()),
                gpu_outlier_count=sum(r['fused_device_seconds']*1000>1.5*gpu[case] for case,g in groups.items() for r in g),
                host_outlier_count=sum(r['total_seconds']*1000>1.5*host[case] for case,g in groups.items() for r in g),
                stop_reason_counts=dict(Counter(str(r['reason']) for r in rows)),invocations=len(rows))
            if method=='nmrom':
                item.update(stationary_invocations=sum(r['stationary'] for r in rows),residual_target_invocations=sum(r['reason']==2 for r in rows),
                    nonstationary_residual_target_invocations=sum(not r['stationary'] and r['reason']==2 for r in rows),
                    nonstationary_other_invocations=sum(not r['stationary'] and r['reason']!=2 for r in rows),
                    median_attempts=float(np.median([r['attempts'] for r in rows])),max_attempts=max(r['attempts'] for r in rows),fallbacks=sum(r['fallback_count'] for r in rows))
            elif method.startswith('cg_'):
                item.update(tolerance=rows[0]['cg_tolerance'],cg_nonconverged_invocations=sum(not r['cg_converged'] for r in rows),
                    iterations_median=float(np.median([r['iterations'] for r in rows])),iterations_max=max(r['iterations'] for r in rows),
                    true_residual_max=max(r['true_relative_residual'] for r in rows))
            mesh['methods'][method]=item
        for method,(gpu,host) in case_times.items():
            if method=='nmrom':continue
            rom_gpu,rom_host=case_times['nmrom'];mesh['methods'][method].update(
                gpu_speedup_ratio_of_cohort_medians=float(np.median(gpu)/np.median(rom_gpu)),
                host_speedup_ratio_of_cohort_medians=float(np.median(host)/np.median(rom_host)),
                gpu_speedup_median_paired_ratio=float(np.median(gpu/rom_gpu)),host_speedup_median_paired_ratio=float(np.median(host/rom_host)))
        passing=[m for m in methods if m.startswith('cg_') and mesh['methods'][m]['all_cases_pass_target']]
        mesh['fastest_passing_tested_cg_gpu']=min(passing,key=lambda m:mesh['methods'][m]['gpu_median_ms']) if passing else None
        mesh['matched_target_nmrom_speedup_established']=mesh['methods']['nmrom']['all_cases_pass_target'] and bool(passing) and mesh['methods'][mesh['fastest_passing_tested_cg_gpu']]['gpu_speedup_ratio_of_cohort_medians']>1
        panel['meshes'].append(mesh)
    (run/'panel.json').write_text(json.dumps(panel,indent=2)+'\n')
    print(json.dumps({m['intervals']:{name:{k:item[k] for k in ['gpu_median_ms','host_median_ms','worst_relative_error','target_failed_cases','invalid_invocations']} for name,item in m['methods'].items()} for m in panel['meshes']},indent=2))


if __name__=='__main__':main()
