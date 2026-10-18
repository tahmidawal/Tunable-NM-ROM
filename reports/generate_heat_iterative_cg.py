"""Generate the user-selected iterative-FOM heat comparison from native evidence."""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/iterative_cg09'
NAMES = {'nmrom':'Original NMROM','nmrom_cholesky':'Cholesky NMROM',
         'fom_cg_cn':'CG, matched CN, historical tolerance',
         'fom_cg_cn_tol1e3':'CG, matched CN, moderately loose tolerance',
         'fom_cg_cn_tol1e2':'CG, matched CN, loosest tolerance',
         'fom_cg_be50':'CG, historical backward-Euler step count',
         'linear_weak_exact':'Free-coefficient linear bank diagnostic',
         'fom_same_grid':'Direct-transform FOM diagnostic'}


def plot_scaling(lookup,settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    ns=settings['requested_intervals']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),layout='constrained')
    curves=[(settings['primary_method'],'NMROM','#0072B2')]
    curves += [(name,f"CG: tolerance {cs['tolerance']:.0e}",color)
               for (name,cs),color in zip([(name,cs) for name,cs in settings['cg_methods'].items() if cs['theta']==.5],['#D55E00','#009E73','#CC79A7'])]
    for name,label,color in curves:
        axes[0].plot(ns,[lookup[n,name]['device_median_ms'] for n in ns],'-o',label=label,color=color)
        axes[1].plot(ns,[100*lookup[n,name]['worst_physical_error'] for n in ns],'-o',label=label,color=color)
    target_line=axes[1].axhline(100*settings['accuracy_target'],color='0.35',linestyle='--',label=f"{100*settings['accuracy_target']:.6g}% development target")
    for ax in axes:
        ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.set_xticks(ns)
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.set_xlabel('Intervals per spatial axis')
        ax.grid(True,which='major',alpha=.25)
    axes[0].set_ylabel('Median complete GPU query (ms)')
    axes[1].set_ylabel('Worst current-relative field error (%)')
    axes[0].legend(fontsize=8)
    axes[1].legend(handles=[target_line],loc='upper right',bbox_to_anchor=(1,.85),fontsize=8)
    fig.suptitle('Heat development comparison: frozen NMROM and iterative CG')
    paths=[ROOT/f'reports/2026-09-10-heat-iterative-cg-scaling.{ext}' for ext in ['png','pdf']]
    for path in paths: fig.savefig(path,dpi=180)
    plt.close(fig)
    return paths


def main():
    paths=[RECORD/'archive/outputs/results.json',RECORD/'analysis/audit.json',
           RECORD/'COLLECTION-CHECK.json',RECORD/'REMOTE-CLEANUP.json',RECORD/'smoke/SMOKE-CHECK.json']
    result,audit,collection,cleanup,smoke=[json.loads(p.read_text()) for p in paths]
    assert result['complete'] and audit['passed'] and collection['archive_verified'] and cleanup['remote_absent']
    assert audit['result_sha256']==hashlib.sha256(paths[0].read_bytes()).hexdigest()
    cfg=result['config']; settings=result['settings']; meta=result['metadata']
    lookup={(x['intervals'],x['method']):x for x in audit['summaries']}
    rows={(x['intervals'],x['case'],x['method']):x for x in result['rows']}
    primary=settings['primary_method']; baseline=settings['primary_fom']; spec=settings['cg_methods'][baseline]
    largest=max(settings['requested_intervals']); nsteps=round(cfg['times'][-1]/settings['dt'])
    def passes(row):
        return row['worst_physical_error']+audit['maximum_reference_refinement']<=settings['accuracy_target'] and row.get('cg_nonconverged_steps',0)==0 and row['nonstationary_fit_count']==0 and row['nonstationary_step_count']==0
    def case_ratios(n,name):
        ratios=[]
        for case in result['cases']:
            cid=case['case']
            a=median(r['phases']['device_seconds'] for r in rows[n,cid,primary]['repetitions'])
            b=median(r['phases']['device_seconds'] for r in rows[n,cid,name]['repetitions'])
            ratios.append(b/a)
        return ratios
    a=lookup[largest,primary]; b=lookup[largest,baseline]
    figures=plot_scaling(lookup,settings)
    lines=['# Heat NMROM versus the iterative CG FOM across resolution','',
        'Completed and independently audited development results using the iterative full-order algorithm selected by the user. These numbers remain provisional for paper claims: one frozen training checkpoint and existing development cases are used, with final confirmation cases unopened.','',
        f"The main comparison uses the same grid, supplied initial field, diffusivity, {nsteps} Crank–Nicolson time steps, and {len(cfg['times'])} requested full-field outputs. CG uses relative tolerance {spec['tolerance']:.6g}, the historical tolerance, and a cap of {spec['max_iterations']} iterations per step. The current Cholesky NMROM and its stopping rule are unchanged.",'',
        f"At {largest} intervals per axis, the Cholesky NMROM takes {a['device_median_ms']:.6f} ms and matched-step CG takes {b['device_median_ms']:.6f} ms: CG/NMROM = {b['device_median_ms']/a['device_median_ms']:.3f}. Their worst current-relative errors are {100*a['worst_physical_error']:.6f}% and {100*b['worst_physical_error']:.6f}%. Both methods {'pass' if passes(a) and passes(b) else 'do not pass'} the declared {100*settings['accuracy_target']:.6g}% development target and solver convergence checks.",'',
        '![Measured heat runtime and error across resolution](2026-09-10-heat-iterative-cg-scaling.png)','',
        'Each CG curve keeps its stated tolerance fixed across meshes. The figure uses all retained repetitions for timing medians and the full development cohort for worst errors.','',
        '## Main comparison: iterative CG at the historical tolerance','',
        '| Intervals/axis | NMROM GPU ms | CG GPU ms | CG/NMROM time ratio | NMROM worst error (%) | CG worst error (%) | NMROM faster cases | GPU outliers, ROM / CG |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in settings['requested_intervals']:
        rom=lookup[n,primary]; fom=lookup[n,baseline]; ratios=case_ratios(n,baseline)
        lines.append(f"| {n} | {rom['device_median_ms']:.6f} | {fom['device_median_ms']:.6f} | {fom['device_median_ms']/rom['device_median_ms']:.3f} | {100*rom['worst_physical_error']:.6f} | {100*fom['worst_physical_error']:.6f} | {sum(r>1 for r in ratios)} / {len(ratios)} | {rom['device_outliers']} / {fom['device_outliers']} |")
    lines += ['',f"All {len(result['cases'])} cases and {settings['timing_repetitions']} repetitions per case are retained. Ratios in the table divide cohort median times; faster-case counts compare within-case medians. GPU timing starts after the supplied initial field is on the device and ends when all requested device outputs and diagnostics are ready. Initialization is included for the ROM. Separate measured host times below include full input/output transfers from the same invocation.",'',
        '## Accuracy and CG tolerance','',
        'The historical tolerance can ask the full solver for substantially more accuracy than the ROM supplies. The following predeclared looser tolerances show that effect. All outcomes, including any physical-target failures, remain in the table. Each linear solve records its actual true residual and iteration count.','',
        '| Intervals/axis | CG relative tolerance | CG GPU ms | CG worst error (%) | Physical target and convergence pass | CG/NMROM GPU ratio |',
        '| ---: | ---: | ---: | ---: | --- | ---: |']
    for n in settings['requested_intervals']:
        for name in settings['cg_methods']:
            cs=settings['cg_methods'][name]
            if cs['theta']!=.5: continue
            row=lookup[n,name]
            lines.append(f"| {n} | {cs['tolerance']:.6g} | {row['device_median_ms']:.6f} | {100*row['worst_physical_error']:.6f} | {'yes' if passes(row) else 'no'} | {row['device_median_ms']/lookup[n,primary]['device_median_ms']:.3f} |")
    lines += ['', 'A time ratio does not establish an acceptable speedup when a method fails the physical target or its stopping rule. The fastest passing tolerance below is selected from these tested development settings; it is not an independent final-cohort result or a global classical-solver optimum.','',
        '| Intervals/axis | Fastest passing tested CN-CG tolerance | CG GPU ms | CG/NMROM GPU ratio | CG/NMROM host ratio |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for n in settings['requested_intervals']:
        candidates=[(name,lookup[n,name]) for name,cs in settings['cg_methods'].items() if cs['theta']==.5 and passes(lookup[n,name])]
        if not candidates:
            lines.append(f'| {n} | none | — | — | — |'); continue
        name,row=min(candidates,key=lambda item:item[1]['device_median_ms']); rom=lookup[n,primary]
        lines.append(f"| {n} | {settings['cg_methods'][name]['tolerance']:.6g} | {row['device_median_ms']:.6f} | {row['device_median_ms']/rom['device_median_ms']:.3f} | {row['host_median_ms']/rom['host_median_ms']:.3f} |")
    lines += ['', '## Historical stepping control','',
        'The primary pair matches the current ROM time discretization. This separate CG control retains the historical backward-Euler step count, applied to the current heat problem and output times. It is not a replay of the historical geometry, training checkpoint or input family.','',
        '| Intervals/axis | BE steps | CG relative tolerance | GPU ms | Worst error (%) | CG/NMROM GPU ratio |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in settings['requested_intervals']:
        name='fom_cg_be50'; row=lookup[n,name]; cs=settings['cg_methods'][name]
        lines.append(f"| {n} | {round(cfg['times'][-1]/cs['dt'])} | {cs['tolerance']:.6g} | {row['device_median_ms']:.6f} | {100*row['worst_physical_error']:.6f} | {row['device_median_ms']/lookup[n,primary]['device_median_ms']:.3f} |")
    lines += ['', '## Complete accounting and diagnostic controls','',
        'The direct-transform FOM remains a separately labeled diagnostic; the user-selected primary comparator is iterative CG. The free-coefficient bank diagnostic is a linear model with more freely evolving coefficients than the nonlinear latent model.','',
        '| Intervals/axis | Method | Median GPU ms | Median host ms | Worst error (%) | GPU / host outliers |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
    for n in settings['requested_intervals']:
        for name in settings['methods']:
            row=lookup[n,name]
            lines.append(f"| {n} | {NAMES[name]} | {row['device_median_ms']:.6f} | {row['host_median_ms']:.6f} | {100*row['worst_physical_error']:.6f} | {row['device_outliers']} / {row['host_outliers']} |")
    lines += ['', '## Solver and reference checks','',
        '| Intervals/axis | CG method | Total timed CG iterations | Nonconverged steps | Largest true relative residual | Worst error versus exact matching time discretization (%) |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
    for n in settings['requested_intervals']:
        for name in settings['cg_methods']:
            row=lookup[n,name]
            lines.append(f"| {n} | {NAMES[name]} | {row['cg_total_iterations']} | {row['cg_nonconverged_steps']} | {row['cg_max_true_relative_residual']:.12g} | {100*row['worst_exact_time_discrete_error']:.9f} |")
    lines += ['',f"Independent NumPy/SciPy checks cover {audit['timed_invocations_checked']} timed invocations, {audit['unique_fields_checked']} distinct full arrays and {audit['metric_entries_checked']} metric entries, with maximum metric difference {audit['maximum_metric_difference']:.12g}. Every CG field is compared with independent SciPy sine-transform propagation of its exact finite-difference time-step formula. Internal full-grid CG states are not archived, so per-step true-residual diagnostics are checked from their recorded values and source, not independently reconstructed for every step.",'',
        f"The CG fixture agrees with independent dense solves within {max(x['dense_time_step_relative_error'] for x in result['verification']['cg_independent_fixture'].values()):.12g} and with installed JAX CG within {max(x['jax_cg_field_parity'] for x in result['verification']['cg_independent_fixture'].values()):.12g}. The local smoke took {smoke['elapsed_seconds']:.6f} seconds; its timings are excluded from benchmark evidence.",'',
        f"The nonlinear trajectories retain the frozen decoder: sampled reconstruction mismatch is at most {audit['maximum_sampled_decoder_error']:.12g}. Independent nonlinear residual/gradient checks cover {audit['time_step_weak_checks']} time steps and {audit['initial_fit_weak_checks']} initial fits. Cholesky field and latent parity {'passes' if audit['algebra_parity_passed'] else 'fails'}, and iteration/acceptance/termination counters {'agree' if audit['all_counts_equal'] else 'differ'}. The original control reproduces earlier archived fields within {max(x['relative_difference'] for x in audit['prior_nmrom_field_parity']):.12g}.",'',
        f"Maximum continuum-spectral reference refinement is {audit['maximum_reference_refinement']:.12g}; this is empirical agreement, not a rigorous continuum bound. The target test adds this allowance to each method's worst physical error. Worst error always includes all cases, requested times and retained repetitions, normalized by the reference norm at that time.",'',
        '## Scope and reproducibility','',
        r"CG solves $[I+\theta\Delta t\nu L]u_{j+1}=[I-(1-\theta)\Delta t\nu L]u_j$ on the full requested grid, starting from the previous state. Here $L$ is the positive finite-difference Dirichlet Laplacian, $\nu$ is diffusivity, and $\theta$ selects the time discretization. Both iterations and rollout are compiled. The explicit final residual verification for each linear solve is included in its timing.",'',
        f"The NMROM retains {cfg['k']} latent variables, {cfg['r']} spatial-bank functions and {cfg['modes_per_axis']**2} weak test modes. No retraining or nonlinear tolerance change was made. This is a same-grid iterative-solver comparison on the current fixed-diffusivity single-bump family; it does not establish a speedup over every full-order algorithm, a global cost-to-accuracy optimum, or performance on other PDEs. The older [direct-transform comparison](2026-09-10-heat-cp-algebra-comparison.md) remains valid for its own baseline.",'',
        f"Job `{meta['job_id']}`, `{meta['gpu']}` on `{meta['node']}`, scientific source `{audit['source_commit']}`. GPU preflight, float64/highest precision, seed regeneration, source/checkpoint hashes, private job directory and complete logs were checked. GPU burn-in precedes timed blocks. Compilation and offline mesh assembly are excluded from every online time and retained separately. Outlier rule: {audit['outlier_rule']}",'',
        'The archive passed checksum collection and its exact completed remote directory was removed. No historical archive, other session job or merge was changed.','',
        f"[Native results](../{paths[0].relative_to(ROOT).as_posix()}) · [Independent audit](../{paths[1].relative_to(ROOT).as_posix()}) · [Configuration](../{(RECORD/'archive/experiments/mr-heat2d/config-iterative-cg.json').relative_to(ROOT).as_posix()}) · [Archive manifest](../{(RECORD/'ARCHIVE.json').relative_to(ROOT).as_posix()})",'',
        '## Glossary','',
        '- **Intervals/axis:** spatial cells along each direction; saved arrays contain interior nodes.',
        '- **NMROM / FOM:** nonlinear-manifold reduced model / full-grid numerical solver.',
        '- **CG / Cholesky:** conjugate-gradient iterative full-grid solve / factorization used for the small nonlinear update system.',
        '- **CN / BE:** Crank–Nicolson / backward Euler, the two stated implicit time discretizations.',
        '- **Relative tolerance / true residual:** stopping threshold relative to the linear-system right-hand side / equation discrepancy recomputed from the actual returned state.',
        '- **Median GPU / host ms:** middle retained blocked device-query time / same invocation including full CPU–GPU input/output transfers, in milliseconds.',
        '- **CG/NMROM ratio:** CG time divided by NMROM time; greater than unity means the NMROM is faster under that timing contract.',
        '- **Worst error:** largest full-field discrepancy divided by the current reference norm across all cases, requested times and repetitions.',
        '- **Faster cases / outliers:** cases whose paired median favors NMROM / repetitions above the stated within-case threshold, all retained.',
        '- **Physical target / development selection:** declared field-error threshold / choice made using the current tested cases, not separate confirmation data.',
        '- **Nonconverged steps / total CG iterations:** linear solves missing their stopping rule / sum of iterations over every timed repetition.',
        '- **Exact matching time discretization:** independent solution of the same grid and time-step formula with linear-solve error removed; it still has discretization error.',
        '- **Bank / latent / weak modes:** learned spatial functions / compressed nonlinear coordinates / smooth tests used for the reduced equation.',
        '- **Direct transform / free-coefficient bank:** full-grid propagation in sine modes / linear reduced model with freely evolving bank coefficients.',
        '- **Parity / checkpoint / refinement:** agreement between implementations / frozen trained parameters / comparison between reference resolutions.',
        '- **Invocation / source hash / checksum collection:** one solve producing both the timing and graded fields / proof of exact code content / verification that collected files match their remote originals.']
    target=ROOT/'reports/2026-09-10-heat-iterative-cg-comparison.md'
    target.write_text('\n'.join(lines)+'\n')
    paths.append(Path(__file__).resolve())
    manifest=dict(source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},report_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),figure_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in figures})
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(target)


if __name__=='__main__': main()
