"""Generate every pilot finding from raw JSON, never manually typed values."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import statistics as stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    src=args.run/'result.json'
    if not src.exists(): src=args.run/'cluster/out/pilot/result.json'
    d=json.loads(src.read_text())
    assert d['complete']
    assert json.loads((args.run/'cleanup.json').read_text())['remote_deleted_and_absence_checked']
    rows=d['rows']; cfg=d['config']
    assert len(rows)==len(cfg['intervals'])*cfg['cohort_count']*cfg['repetitions']*(len(cfg['taus'])+2)
    groups=defaultdict(list)
    for row in rows:
        groups[(row['intervals'],row['arm'],row['tau'])].append(row)
        assert abs(row['total_seconds']-sum(row[k] for k in ('input_seconds','projection_init_seconds','solver_seconds','output_seconds')))<1e-10
    summaries=[]
    for (n,arm,tau),group in groups.items():
        cases=defaultdict(list)
        for row in group:cases[row['case']].append(row)
        assert len(cases)==cfg['cohort_count']
        assert all(len(x)==cfg['repetitions'] for x in cases.values())
        times=[row['total_seconds'] for row in group]
        middle=stats.median(times)
        per=[]
        for case,rs in cases.items():
            assert len({r['field_sha256'] for r in rs})==1
            per.append(dict(case=case,physical_error=rs[0]['physical_error'],same_grid_error=rs[0]['same_grid_error'],
                conservative_physical_error=rs[0]['conservative_physical_error'],uncertainty=rs[0]['uncertainty'],
                stationary=rs[0]['stationary'],solver_valid=rs[0]['solver_valid'],reason=rs[0]['reason'],
                stationarity=rs[0]['stationarity'],median_seconds=stats.median(r['total_seconds'] for r in rs),
                raw_seconds=[r['total_seconds'] for r in rs]))
        s=dict(intervals=n,arm=arm,tau=tau,median_seconds=middle,
            timing_outliers_above_three_medians=sum(t>3*middle for t in times),
            physical_median=stats.median(x['physical_error'] for x in per),physical_max=max(x['physical_error'] for x in per),
            same_grid_median=stats.median(x['same_grid_error'] for x in per),same_grid_max=max(x['same_grid_error'] for x in per),
            nonstationary_count=sum(not x['stationary'] for x in per),invalid_count=sum(not x['solver_valid'] for x in per),
            reasons=dict(Counter(str(x['reason']) for x in per)),
            median_attempts=stats.median(x['attempts'] for x in group),
            median_components_seconds={k:stats.median(x[k] for x in group) for k in ('input_seconds','projection_init_seconds','solver_seconds','output_seconds')},
            cases=per,targets={})
        for eps in cfg['targets']:
            fail=[x for x in per if not x['solver_valid'] or x['conservative_physical_error']>eps or x['uncertainty']>cfg['reference_fraction']*eps]
            s['targets'][str(eps)]=dict(qualified=len(fail)==0,failure_count=len(fail),failure_cases=[x['case'] for x in fail])
        summaries.append(s)
    envelope=[]
    for n in cfg['intervals']:
        for eps in cfg['targets']:
            candidates=[s for s in summaries if s['intervals']==n and s['targets'][str(eps)]['qualified']]
            rom=sorted((s for s in candidates if s['arm']=='rom'),key=lambda s:s['median_seconds'])
            fom=sorted((s for s in candidates if s['arm']!='rom'),key=lambda s:s['median_seconds'])
            envelope.append(dict(intervals=n,target=eps,
                rom_tau=rom[0]['tau'] if rom else None,fom_arm=fom[0]['arm'] if fom else None,
                rom_seconds=rom[0]['median_seconds'] if rom else None,fom_seconds=fom[0]['median_seconds'] if fom else None,
                speedup=fom[0]['median_seconds']/rom[0]['median_seconds'] if fom and rom else None))
    for e in envelope:
        if e['speedup'] is not None:
            rr=groups[(e['intervals'],'rom',e['rom_tau'])]
            ff=groups[(e['intervals'],e['fom_arm'],None)]
            paired={(x['case'],x['repetition']):x['total_seconds'] for x in ff}
            e['raw_paired_speedups']=[paired[(x['case'],x['repetition'])]/x['total_seconds'] for x in rr]
            e['median_paired_speedup']=stats.median(e['raw_paired_speedups'])
    summary=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),config=cfg,
        provenance=d['provenance'],checkpoint_sha256=d['checkpoint_sha256'],
        summaries=summaries,envelope=envelope,reference=d['references'],setup=d['setup'])
    (args.run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Frozen Poisson mesh-transfer development findings','',
        'These generated results are provisional development evidence from one checkpoint and one source cohort. '
        'They compare complete queries against direct transform solvers; no final-cohort claim is made.','',
        f"Source `{d['provenance']['commit']}`, job `{d['provenance']['job_id']}`, GPU `{d['provenance']['gpu']}`. "
        f"The checkpoint was trained on {d['checkpoint_config']['N']} nodes per axis; the new meshes use intervals. "
        f"The fixed development seed is {cfg['seed']} with {cfg['cohort_count']} sources and {cfg['repetitions']} timing repetitions.", '',
        '## Accuracy and complete query time','',
        '| Intervals | Arm | Tau | Median ms | Median physical error | Worst physical error | Worst same-grid error | Invalid sources | Nonstationary sources | Timing outliers |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['intervals']} | {s['arm']} | {s['tau'] if s['tau'] is not None else '—'} | {s['median_seconds']*1000:.4f} | {s['physical_median']:.6g} | {s['physical_max']:.6g} | {s['same_grid_max']:.6g} | {s['invalid_count']} | {s['nonstationary_count']} | {s['timing_outliers_above_three_medians']} |")
    lines+=['','### Complete-query components','',
        '| Intervals | Arm | Tau | Input ms | Projection/init ms | Solver ms | Output ms |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        c=s['median_components_seconds']
        parts=' | '.join(f'{c[k]*1000:.4f}' for k in ('input_seconds','projection_init_seconds','solver_seconds','output_seconds'))
        lines.append(f"| {s['intervals']} | {s['arm']} | {s['tau'] if s['tau'] is not None else '—'} | {parts} |")
    lines+=['','Component medians do not necessarily sum to the median total. Raw synchronized timestamps are retained for each invocation.',
        '', 'Physical errors use the same nested observation nodes and independently refined reference. '
        'Qualification below adds empirical reference uncertainty; every source must meet the target. '
        'Intentional tau stops may pass physical qualification before stationarity. Other terminal ROM solves require measured stationarity. '
        'The complete query includes synchronized input, source projection, solve and full output.','',
        '## Cheapest qualifying development configurations','',
        '| Requested intervals | Target | Selected ROM tau | Selected FOM | ROM ms | FOM ms | FOM/ROM speedup |',
        '|---:|---:|---:|---|---:|---:|---:|']
    f=lambda x:'unattained' if x is None else f'{x:.6g}'
    for e in envelope:
        lines.append(f"| {e['intervals']} | {e['target']} | {f(e['rom_tau'])} | {e['fom_arm'] or 'unattained'} | {f(None if e['rom_seconds'] is None else e['rom_seconds']*1000)} | {f(None if e['fom_seconds'] is None else e['fom_seconds']*1000)} | {f(e['speedup'])} |")
    lines+=['','Speedup above unity favors the reduced model. Unattained targets have no speedup. '
        'The classical envelope searches only the declared same-grid and coarse-grid options. '
        'Selection and evaluation use this development cohort; independent confirmation is still required.','',
        '## Reference and mesh setup','',
        '| Source | Refinement gap, first pair | Refinement gap, final pair | Ratio | Empirical uncertainty |',
        '|---:|---:|---:|---:|---:|']
    for r in d['references']:
        lines.append(f"| {r['case']} | {r['successive_relative_gaps'][0]:.6g} | {r['successive_relative_gaps'][1]:.6g} | {r['refinement_ratio']:.6g} | {r['uncertainty']:.6g} |")
    lines+=['',f"Reference intervals: {cfg['reference_intervals']}; common observation intervals: {cfg['observation_intervals']}. "
        'The last difference is retained without optimistic Richardson reduction. This empirical evidence is not a rigorous continuum error bound.','',
        '| Intervals | Nodes per axis | Interior unknowns | Retained modes | Retained bank rank | Bank bytes | Bank setup s | Weak assembly s |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in d['setup']:
        lines.append(f"| {s['intervals']} | {s['nodes_per_axis']} | {s['interior_unknowns']} | {s['retained_modes']} | {s['retained_bank_rank']} | {s['bank_bytes']} | {s['bank_build_seconds']:.6g} | {s['weak_assembly_seconds']:.6g} |")
    lines+=['','Both network weights remain identical across meshes. New-grid bank evaluation and weak operator assembly recur. '
        'Setup timings include their first compilation; warmup timings are retained in JSON. Training cost is inherited and is not amortized by this pilot. '
        'Dense projection and requested output depend on grid size, so no flat end-to-end complexity claim follows.','',
        '## Limits and open work','',
        'The result covers a compact inherited checkpoint and the fixed smooth single-Gaussian source family on a square, constant-coefficient, zero-Dirichlet problem. '
        'It does not establish performance on variable coefficients, new geometries, other checkpoints, broader input families or final cohorts. '
        'The tolerance ladder is a deployment study of frozen weights; no per-resolution retraining has run. '
        'Future performance improvements need separate paired validation and must preserve the input/output contract.','',
        '## Plain-language glossary','',
        '- **Intervals / nodes / interior unknowns:** cells along an axis / coordinates including boundary walls / values solved inside the walls.',
        '- **Arm / tau:** measured algorithm / requested reduction of its initial weak residual. A zero tau disables that early stop.',
        '- **ROM / FOM / DST:** reduced model / full discrete model / fast sine transform direct solver.',
        '- **Median / worst:** middle measured value / largest source error. Raw repeated times remain in JSON.',
        '- **Physical / same-grid error:** relative discrepancy from refined reference / from the full solver on the identical mesh.',
        '- **Invalid / nonstationary sources:** failed solver validity checks / normalized gradient above the configured stationary threshold. An intentional tau stop may remain valid.',
        '- **Timing outlier:** repetition taking more than three times its configuration median; retained, never discarded.',
        '- **Target / speedup:** required error ceiling / full-model cost divided by reduced-model cost after qualification.',
        '- **Coarse DST / prolongation:** source restriction to fewer grid points followed by direct solve / interpolation back to requested output; its cost is charged.',
        '- **Reference gap / ratio / uncertainty:** discrepancy between consecutive fine meshes / first gap divided by final gap / conservative empirical residual reference error.',
        '- **Mode / bank rank / setup:** smooth weak test function / independently retained spatial directions / one-time mesh preparation.',
        '- **Bytes / ms / s:** stored memory units / milliseconds / seconds.',
        '- **Checkpoint / seed / cohort / commit:** saved trained weights / reproducible random generator setting / fixed source group / immutable code revision.',
        '- **Development / qualification / amortization:** preliminary selection evidence / meeting every declared accuracy and validity condition / repaying setup cost with repeated-query savings.','']
    (args.run/'FINDINGS.md').write_text('\n'.join(lines))
    print(json.dumps(envelope,indent=2))

if __name__=='__main__':main()
