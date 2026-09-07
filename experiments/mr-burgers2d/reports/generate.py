"""Generate every numerical claim/table from the saved pilot JSON."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
d=json.loads(Path(a.input).read_text());groups=defaultdict(list)
for row in d['invocations']:groups[row['name']].append(row)
summary=[]
for name,rows in groups.items():
    c=rows[0];cases=sorted(set(x['case'] for x in rows));errs=[];curr=[];same=[]
    for case in cases:
        r=[x for x in rows if x['case']==case]
        errs.append(max(x['physical_error']['fixed_initial_max'] if x['physical_error'] else np.inf for x in r))
        curr.append(max(x['physical_error']['current_relative_max'] if x['physical_error'] else np.inf for x in r))
        if c['method']=='rom':same.append(max(x['same_grid_error']['fixed_initial_max'] if x['same_grid_error'] else np.inf for x in r))
    summary.append(dict(name=name,method=c['method'],grid=c['solver_intervals'],output=c['output_intervals'],dt=c['dt'],
        solver=c.get('stall',c.get('newton_tolerance')),ms=float(np.median([r['seconds'] for r in rows])*1000),
        err=np.median(errs),worst=max(errs),current=max(curr),outliers=sum(e>.01 for e in errs),
        failures=sum(not r['finite'] or (c['method']=='fom' and not r['nonlinear_tolerance_satisfied']) for r in rows),
        same=max(same) if same else None,cases=len(cases),reps=len(rows)//len(cases)))
lines=['# Burgers 2D frozen-network resolution transfer and complete-query cost','',
    'Development pilot; all tables below are generated from the saved invocation records. Physical-error qualifications remain provisional wherever the measured reference-refinement estimate misses the target budget.','',
    f"Source commit `{d['commit']}`, job `{d['job_id']}`, GPU `{d['gpu']}`. Backend `{d['backend']}`, f64 `{d['x64']}`, matmul precision `{d['matmul_precision']}`. Checkpoint SHA-256 `{d['checkpoint_sha256']}`.",'',
    f"The frozen checkpoint was trained on {d['checkpoint_training_intervals']} intervals ({d['checkpoint_training_nodes']} nodes per axis). The new validation seed is {d['config']['seed']}, with {d['config']['cases']} physical cases; the final cohort is unopened.",'',
    'The query starts with a dense initial field in host memory and ends with all requested dense fields in host memory. Input handling, cold fitting, evolution, output reconstruction/interpolation and transfers are included. Compilation and reusable setup are separate. FOM candidates use sign-upwind backward Euler with adaptive Newton/BiCGStab and FFT sine-transform Helmholtz preconditioning. ROM quadrature retains sign-dependent upwinding on decoded undershoots.','',
    '## Reference refinement','',
    '| Case | Spatial difference | Time difference | Sum estimate |','|---|---:|---:|---:|']
for r in d['reference_uncertainty']:lines.append(f"| {r['case']} | {r['space_difference']['fixed_initial_max']:.6g} | {r['time_difference']['fixed_initial_max']:.6g} | {r['conservative_difference_sum']:.6g} |")
unc=max(r['conservative_difference_sum'] for r in d['reference_uncertainty'])
lines+=['','These are differences between independently converged refinement levels, not rigorous continuum-error bounds. The physical norm uses exact nested-node restriction onto the observation grid and the initial-state reference norm. Current-field normalization is reported separately.','',
'## Mesh setup','',
'| Intervals | Interior unknowns | K / R / M / m | Sampled bank rank | Setup s | Stored arrays MiB | Quadrature fit |','|---|---:|---|---:|---:|---:|---:|']
for r in d['mesh_setup']:lines.append(f"| {r['intervals']} | {r['interior_unknowns']} | {r['K']} / {r['R']} / {r['M']} / {r['m']} | {r['sampled_bank_rank_relative_1e12']} | {r['setup_seconds']:.3f} | {r['array_bytes']/2**20:.3f} | {r['eq_relative_fit']:.6g} |")
lines+=['','## Measured configurations','',
'Errors are maximum-in-observation-time values, aggregated over cases. Outliers count cases above relative error 0.01. Failed invocations are retained. Wall times are medians over all cases/repetitions in the same job.','',
'| Configuration | Query ms | Physical median | Physical worst | Current-relative worst | Same-grid worst | Outliers | Failed invocations |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in sorted(summary,key=lambda r:(r['output'],r['method'],r['ms'])):
    same=f"{r['same']:.6g}" if r['same'] is not None else '—'
    lines.append(f"| `{r['name']}` | {r['ms']:.3f} | {r['err']:.6g} | {r['worst']:.6g} | {r['current']:.6g} | {same} | {r['outliers']} / {r['cases']} | {r['failures']} |")
lines+=['','## Target-qualified validation selections','',
'Both methods may choose a configuration within the declared pilot search. The FOM envelope includes coarser solves with aligned interpolation to the same requested dense output. Solver choices are tuned per resolution; both neural networks remain frozen. This does not test retraining benefits.','',
'| Output intervals | Target | Qualification | Selected ROM | Selected FOM envelope | Envelope speedup |','|---|---:|---|---|---|---:|']
for L in sorted(set(r['output'] for r in summary)):
    for target in [.1,.05,.01,.001]:
        eligible=[r for r in summary if r['output']==L and r['worst']<=target and r['failures']==0]
        rom=min((r for r in eligible if r['method']=='rom'),key=lambda r:r['ms'],default=None)
        fom=min((r for r in eligible if r['method']=='fom'),key=lambda r:r['ms'],default=None)
        state='provisional refinement budget passed' if unc<=target/10 else 'unresolved reference uncertainty'
        if rom is None:state+='; ROM target unattained'
        if fom is None:state+='; FOM target unattained'
        speed=f"{fom['ms']/rom['ms']:.3f}x" if rom and fom and unc<=target/10 else '—'
        lines.append(f"| {L} | {target:g} | {state} | {rom['name'] if rom else '—'} | {fom['name'] if fom else '—'} | {speed} |")
lines+=['','No final-cohort result, unchanged-weight cross-PDE transfer, universal speed advantage, or fully optimized per-resolution training claim is established by this bounded pilot. Raw invocation arrays and fields remain the source of truth.','',
'## Plain-language glossary','',
'- **Intervals / interior unknowns:** grid cells along an axis / non-wall values solved for.',
'- **K / R / M / m:** latent coordinates / learned spatial features / smooth weak test functions / advection quadrature points.',
'- **Sampled bank rank:** independent feature directions on deterministic rows, with a relative singular-value cutoff of 1e-12.',
'- **Setup s / stored arrays MiB:** mesh-specific preparation seconds / stored numerical-array memory in binary megabytes.',
'- **Quadrature fit:** normalized residual of nonnegative fitting on decoder-output advection training rows; it is not a physical accuracy certificate.',
'- **Configuration / dt / stall / ntol:** solver setting name / timestep / ROM relative-improvement stopping threshold / FOM relative nonlinear-residual tolerance.',
'- **Query ms:** median complete input-to-output latency in milliseconds, including transfers and requested dense output.',
'- **Physical median / physical worst:** median / maximum over casewise time-maximum errors against the refined reference, normalized by initial reference norm.',
'- **Current-relative worst:** maximum error normalized by the current reference field norm, which can grow as the field decays.',
'- **Same-grid worst:** maximum ROM discrepancy from tightly converged FOM on the same grid and timestep, with initial-state normalization.',
'- **Outliers / failed invocations:** cases exceeding the stated error threshold / timed calls with nonfinite output or unmet FOM nonlinear tolerance.',
'- **Spatial difference / time difference / sum estimate:** observed reference changes after spatial / temporal refinement / their sum; these diagnose uncertainty without proving a bound.',
'- **Qualification / target / selected ROM / selected FOM envelope / envelope speedup:** development accuracy status / requested error limit / cheapest eligible reduced configuration / cheapest eligible full configuration including coarser solves / FOM query time divided by ROM query time.',
'- **Frozen / validation / final:** unchanged network weights / cases used for development selection / separate unopened confirmation cases.',
'- **FOM / ROM / sign-upwind / NNLS:** full model / reduced model / sign-selected spatial differences / nonnegative least squares.',
'- **Newton / BiCGStab / FFT sine transform / Helmholtz preconditioning:** nonlinear iteration / iterative linear solve / fast transform / exact inversion of the diffusion-plus-identity part to help that solve.',
'- **Commit / SHA-256 / backend / f64:** saved source revision / content fingerprint / execution device type / 64-bit floating-point arithmetic.','']
Path(a.output).write_text('\n'.join(lines))
print(a.output)
