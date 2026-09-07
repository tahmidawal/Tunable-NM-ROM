"""Generate matched-wave findings only from audited run JSON and fields."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

NAMES=('displacement','velocity','energy_state')
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def maximum(m):return max(m[k]['max_initial_normalized'] for k in NAMES)
def median(a):return float(np.median(a))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(record):
    native=record/'cluster/out/pilot';out=record/'analysis';data=json.loads((native/'result.json').read_text());cfg=data['config']
    audit=json.loads((out/'audit.json').read_text());assert audit['passed'] and audit['result_sha256']==sha(native/'result.json')
    grouped=defaultdict(list)
    for row in data['invocations']:grouped[row['boundary'],row['intervals'],row['method'],row['setting']].append(row)
    summary=dict(provenance=data['provenance'],config=cfg,result_sha256=sha(native/'result.json'),audit_sha256=sha(out/'audit.json'),
        groups=[],representation=[],training=data['training'],linear_exponential_checks=data['linear_exponential_checks'],time_refinement=[],reference_uncertainty=[],final_test_opened=False,
        interpretation='Two development cases per boundary. Raw paired ratios are not equal-accuracy certification. Linear controls live in the frozen learned bank and are not claims about unchanged MLP performance.')
    for (bc,n,method,setting),rows in grouped.items():
        cases=[]
        for ci in cfg['validation_indices']:
            rr=sorted((r for r in rows if r['case']==ci),key=lambda r:r['repetition'])
            cases.append(dict(case=ci,query_repetitions=[r['seconds']['complete_query'] for r in rr],query_median=median([r['seconds']['complete_query'] for r in rr]),
                phase_medians={key:median([r['seconds'][key] for r in rr]) for key in rr[0]['seconds']},
                worst_physical_error=max(maximum(r['physical_reference_error']) for r in rr),
                worst_energy_error=max(r['physical_reference_error']['energy_state']['max_initial_normalized'] for r in rr),
                worst_displacement_error=max(r['physical_reference_error']['displacement']['max_initial_normalized'] for r in rr),
                final_current_relative={key:rr[0]['physical_reference_error'][key]['current_relative'][-1] for key in NAMES},
                final_absolute={key:rr[0]['physical_reference_error'][key]['absolute'][-1] for key in NAMES},
                final_reference_norm={key:rr[0]['physical_reference_error'][key]['reference_norm'][-1] for key in NAMES},
                final_vanishing={key:rr[0]['physical_reference_error'][key]['reference_vanishing'][-1] for key in NAMES},
                completed=all(r['completed'] for r in rr),stationary=None if method!='rom' else all(r['fit_stationary'] for r in rr)))
        summary['groups'].append(dict(boundary=bc,intervals=n,method=method,setting=setting,cases=cases,
            query_median=median([c['query_median'] for c in cases]),physical_error_median=median([c['worst_physical_error'] for c in cases]),physical_error_worst=max(c['worst_physical_error'] for c in cases),
            errors_above_targets={str(t):sum(not c['completed'] or c['worst_physical_error']>t for c in cases) for t in cfg['accuracy_targets']}))
    for g in summary['groups']:
        baseline=next(x for x in summary['groups'] if x['boundary']==g['boundary'] and x['intervals']==g['intervals'] and x['method']==('dst' if g['boundary']=='dirichlet' else 'rk4') and x['setting']==(0. if g['boundary']=='dirichlet' else .45))
        g['paired_FOM_over_method_ratio']=median([next(b for b in baseline['cases'] if b['case']==c['case'])['query_median']/c['query_median'] for c in g['cases']])
    for bc in cfg['boundaries']:
        for n in cfg['meshes']:
            for ci in cfg['validation_indices']:
                coarse,fine=cfg['rom_dts'];matched=next(r for r in data['invocations'] if (r['boundary'],r['intervals'],r['case'],r['method'],r['setting'],r['repetition'])==(bc,n,ci,'rom',coarse,0))
                with np.load(native/f'{bc}_{n}_{ci}_0_rom_{coarse}.npz') as a,np.load(native/f'{bc}_{n}_{ci}_0_rom_{fine}.npz') as b,np.load(native/f'mesh_{bc}_{n}.npz') as m:
                    da,db=a['coefficients']-b['coefficients'],a['velocity_coefficients']-b['velocity_coefficients'];c=matched['parameters'][5];scales=matched['same_grid_discrepancy']
                    error=max(float(np.max(np.linalg.norm(da,axis=1)/scales['displacement']['initial_scale'])),float(np.max(np.linalg.norm(db,axis=1)/scales['velocity']['initial_scale'])),float(np.max(np.sqrt(np.sum(db*db,axis=1)+c*c*np.einsum('tr,rs,ts->t',da,m['stiffness'],da))/scales['energy_state']['initial_scale'])))
                    summary['time_refinement'].append(dict(boundary=bc,intervals=n,case=ci,maximum_required_difference=error,passed=error<.01))
                diag=next(d for d in data['diagnostics'] if (d['boundary'],d['intervals'],d['case'])==(bc,n,ci))
                for arm in diag['arms']:
                    summary['representation'].append(dict(boundary=bc,intervals=n,case=ci,arm=arm['arm'],
                        displacement=max(arm['snapshot_metrics']['displacement']['initial_normalized']),
                        tangent_velocity=max(arm['snapshot_metrics']['velocity']['initial_normalized']),
                        snapshot_energy_state=max(arm['snapshot_metrics']['energy_state']['initial_normalized']),
                        normal_force_max=max(arm['normal_force_absolute'][:-1]),force_scale=diag['force_scale'],
                        normal_force_fixed_scaled_max=max(arm['normal_force_fixed_scaled'][:-1]),
                        zero_mass_norm=arm['zero_field_mass_norm'],zero_initial_scaled=arm['zero_field_initial_displacement_scaled'],
                        selected_fit_nonstationary=sum(not b for b in diag['fitting'][-1]['selected_stationary']) if arm['arm']=='mlp16' else 0,
                        fit_budget_max_objective_change=diag['fit_budget_objective_max_change'] if arm['arm']=='mlp16' else None,
                        minimum_rank_ratio=min(arm['rank_ratios'])))
        for ci in cfg['validation_indices']:
            finest=next(r for r in data['references'] if r['boundary']==bc and r['case']==ci and 'fine_intervals' in r)
            if bc=='dirichlet':entry=dict(empirical_self_difference=maximum(finest['self_refinement']))
            else:
                nested=next(r for r in data['references'] if r['boundary']==bc and r['case']==ci and 'adjacent_coarse_to_this_mesh' in r)
                dc,df=maximum(nested['adjacent_coarse_to_this_mesh']),maximum(nested['same_grid_to_physical_reference']);ratio=df/dc;temporal=maximum(finest['temporal_refinement'])
                entry=dict(adjacent_coarse_difference=dc,adjacent_fine_difference=df,temporal_difference=temporal,observed_spatial_ratio=ratio,conditional_fine_reference_estimate=df*ratio/(1-ratio)+temporal if 0<=ratio<1 else None)
            summary['reference_uncertainty'].append(dict(boundary=bc,case=ci,proven_bound=None,**entry))
    write(out/'summary.json',summary)
    render(summary,audit,out/'FINDINGS.md')
    plot(summary,data,out)
    print(json.dumps(dict(summary=str(out/'summary.json'),invocations=audit['audited_invocations'],groups=len(summary['groups'])),indent=2))

def render(s,audit,path):
    p=s['provenance'];cfg=s['config'];lines=['# Matched-dimensional fresh-wave compression and dynamics','',
        'This generated report covers the completed bounded wave dynamics pilot. Numbers are provisional development evidence on the declared cases; the independent final cohort remains sealed.','',
        f"Source `{p['source_commit']}`, GPU job `{p['job_id']}`, device `{p['device_kind'][0]}`. Native result hash `{s['result_sha256']}`. All rows come from the audited native JSON.",'',
        'The frozen MLP16 and saved common-affine16 have equal displacement and phase dimensions. Affine32 and full64 are larger linear controls inside the same learned neural bank. No spatial or head weight is retrained, and the family and validation draws are unchanged. The full-bank control is not an equal-dimensional nonlinear comparison.','',
        'All queries consume full host displacement and velocity fields plus wave speed, and return both dense fields at every requested time. Input projection/fitting, speed-dependent augmented linear generator and exponential, evolution, decoding and host transfer are charged. Mesh-only bank assembly and training-only PCA construction are offline.','',
        '| Boundary | Intervals | Method | Step / CFL | Query median ms | Error median | Error worst | Raw FOM/method | Cases over 5% |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for g in s['groups']:
        lines.append(f"| {g['boundary']} | {g['intervals']} | {g['method']} | {g['setting']:.6g} | {g['query_median']*1000:.9g} | {g['physical_error_median']:.9g} | {g['physical_error_worst']:.9g} | {g['paired_FOM_over_method_ratio']:.9g} | {g['errors_above_targets']['0.05']} |")
    lines += ['', 'The matched comparison has boundary-dependent ordering; the larger linear ladder separates compression from the nonlinear head behavior.']
    for bc in cfg['boundaries']:
        chosen={g['method']:g for g in s['groups'] if g['boundary']==bc and g['intervals']==max(cfg['meshes']) and (g['method']!='rom' or g['setting']==cfg['rom_dts'][0])}
        lines += ['',f"At the finer mesh, `{bc}` worst required errors are MLP16 `{chosen['rom']['physical_error_worst']:.9g}`, affine16 `{chosen['affine16']['physical_error_worst']:.9g}`, affine32 `{chosen['affine32']['physical_error_worst']:.9g}` and full64 `{chosen['full64']['physical_error_worst']:.9g}`. The equal-dimensional control {'improves on' if chosen['affine16']['physical_error_worst']<chosen['rom']['physical_error_worst'] else 'is worse than'} the frozen nonlinear rollout for this boundary."]
    lines += ['', 'A better snapshot fit alone does not ensure accurate autonomous evolution. Conversely, failure of one linear model at a given dimension does not establish that every nonlinear manifold of that dimension must fail. This experiment identifies a strong larger-dimensional linear control, not a successful retrained nonlinear model.']
    lines += ['', 'Errors are common-observation-grid time maxima of the required displacement, velocity and phase-energy state norms. Displacement is divided by initial displacement L2; velocity and energy-state are divided by $\\sqrt{2E(0)}$. The table takes medians across case timing medians; paired ratios are the median of per-case FOM/method timing ratios. A ratio above one means faster than the listed FOM, but is not equal-accuracy certification. Reflective FOM is exact discrete DST evolution; absorbing FOM uses the declared primary CFL. Both solve at the requested mesh. Coarser FOM solves with charged output interpolation are not swept in this diagnostic, so these ratios do not establish the fastest FOM cost at a target accuracy.','',
        '| Boundary | Training field hashes identical | Saved PCA projector defect | Saved center defect | Training construction seconds |','|---|---|---:|---:|---:|']
    for t in s['training']:lines.append(f"| {t['boundary']} | {t['data_hashes_match']} | {t['saved_projector_defect']:.9g} | {t['saved_center_defect']:.9g} | {t['seconds_including_first_compile']:.9g} |")
    lines += ['', 'Original training-only displacement coefficients use their original centering and standard-deviation convention. The saved affine initialization is retained exactly; covariance, basis, singular values, coefficient arrays, training membership and hashes are archived. The added dimension ladder is only labeled the same construction when the saved projector and center agree. Training-generation seconds include compilation and are not paired GPU training-speed measurements.','',
        '| Boundary | Intervals | Case | Head | Snapshot displacement | Tangent velocity | Snapshot energy state | Normal force absolute | Zero-field norm | Nonstationary selected fits |','|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in s['representation']:
        lines.append(f"| {r['boundary']} | {r['intervals']} | {r['case']} | {r['arm']} | {r['displacement']:.9g} | {r['tangent_velocity']:.9g} | {r['snapshot_energy_state']:.9g} | {r['normal_force_max']:.9g} | {r['zero_mass_norm']:.9g} | {r['selected_fit_nonstationary']} |")
    lines += ['', f"Snapshot/tangent/normal-force diagnostics are restricted to times `{[i*cfg['observation_dt'] for i in cfg['diagnostic_indices']]}`. MLP fitting uses the same eight starts at budgets `{cfg['diagnostic_fit_budgets']}`; all endpoints, gradients, stationarity, damping and rank ratios are retained. The longer-budget result is selected by the declared uniform rule. Nonstationary fit counts include the separate zero target. These are best-recorded local fits, not global nonlinear optima.", '',
        'The unrestricted full64 snapshot row isolates the learned-bank projection floor. Normal-force residual is computed inside weak bank coordinates as $(I-QQ^T)(-Ka-Db-h^{\\prime\\prime}[w,w])$. Absolute norms are shown; fixed scaled norms and their denominator $\\|u(0)\\|/T^2$ are saved in the JSON. It measures local force compatibility, not accumulated trajectory error or proof of a causal mechanism. Zero-field norm is the best-recorded mass norm of a fitted zero displacement. A nonzero fit is evidence about that tested image and optimizer; it does not prove the source of absorbing late-time error.','',
        '| Boundary | Intervals | Method | Case | Final current-relative displacement | Final current-relative energy state | Final absolute energy error | Final truth energy-state norm |','|---|---:|---|---:|---:|---:|---:|---:|']
    for g in s['groups']:
        if g['boundary']!='absorbing' or (g['method']=='rom' and g['setting']!=cfg['rom_dts'][0]):continue
        for c in g['cases']:
            fmt=lambda value:'undefined' if value is None else f'{value:.9g}'
            lines.append(f"| {g['boundary']} | {g['intervals']} | {g['method']} | {c['case']} | {fmt(c['final_current_relative']['displacement'])} | {fmt(c['final_current_relative']['energy_state'])} | {c['final_absolute']['energy_state']:.9g} | {c['final_reference_norm']['energy_state']:.9g} |")
    lines += ['', 'Fixed initial-state normalization and current-relative accuracy answer different questions as the absorbing field decays. Absolute errors, truth norms and vanishing flags remain available for every observation. No small initial-normalized error is described as current-relative accuracy.','',
        f"All `{len(s['time_refinement'])}` nonlinear time-step comparisons pass: `{all(r['passed'] for r in s['time_refinement'])}`; maximum required difference `{max(r['maximum_required_difference'] for r in s['time_refinement']):.9g}`. Each first-repetition affine control is checked against an independent direct-time SciPy matrix exponential. Nonzero affine offset forcing is retained; the small-system controls also verify the zero-nonlinearity manifold equation and coordinate-gauge invariance.",'',
        f"Native CPU audit reconstructs common-grid errors for `{audit['audited_invocations']}` timed invocations, maximum metric difference `{audit['maximum_physical_metric_difference']:.9g}`. It also reconstructs full-grid ROM fields from saved banks/coefficients and scores them against retained full-grid references, maximum native-grid metric difference `{audit['maximum_ROM_native_grid_metric_difference']:.9g}`. It independently differentiates the MLP in NumPy and checks physical velocities, curvature and normal forces. The saved reference trajectories are not independently regenerated by this audit; finer-grid FOM metrics for nonreference CFLs cannot be reconstructed from their common fields and remain outside its fine-grid scope.",'',
        'Reference self-differences and absorbing contraction estimates are empirical. Proven continuum uncertainty bounds remain unspecified. This small development experiment does not open final data, isolate all training effects, compare retrained nonlinear dimensions or establish a paper-wide cost-to-tolerance conclusion.','',
        '![Wave dynamics accuracy and cost](dynamics-accuracy-cost.png)','',
        '![Wave error evolution](dynamics-error-evolution.png)','',
        'Error-evolution plots omit values at or below $10^{-12}$ on the logarithmic axis; all raw values remain in the JSON.', '',
        '## Plain-language glossary','',
        '- **Intervals / common grid:** spatial cells per axis / the shared observation mesh used to compare physical errors across meshes.',
        '- **Bank / head / frozen:** learned spatial features / map from reduced coordinates to feature coefficients / unchanged network weights.',
        '- **MLP / affine / full64:** nonlinear multilayer coefficient map / linear coefficient map plus an offset / all dimensions of the learned spatial span.',
        '- **Configuration / phase / snapshot / tangent:** displacement coordinates / displacement and velocity coordinates / a field at one time / locally representable velocity directions.',
        '- **PCA / covariance / projector:** principal training-coefficient directions / their second-order variation matrix / map into their span.',
        '- **Mass / stiffness / damping / offset forcing:** physical inner product / restoring operator / energy-loss operator / constant force required by a nonzero affine displacement offset.',
        '- **Weak / normal force / curvature:** projected spatial equations / acceleration outside the tangent span / acceleration caused by bending of the coefficient map.',
        '- **Stationary / rank / budget / multistart:** locally small fitting gradient / independent local directions / maximum fitting iterations / repeated fitting from predetermined starting points.',
        '- **DST / RK4 / CFL / matrix exponential:** sine transform / fourth-order time integration / time-step-to-grid ratio / direct linear-system propagation.',
        '- **L2 / energy state / initial-normalized / current-relative:** mass-weighted field norm / velocity plus gradient state norm / divided by a fixed initial scale / divided by the current truth norm.',
        '- **Complete query / repetition / median / outlier:** supplied input through requested host output / repeated timing of the same physical case / central sorted value / case exceeding a declared error target.',
        '- **Reference / empirical / final cohort:** comparison solution / supported by observed refinement without a rigorous bound / independent cases reserved for later confirmation.',
        '- **Audit / hash / provenance:** independent artifact check / content fingerprint / recorded source, parameters, device and job identity.','']
    path.write_text('\n'.join(lines))

def plot(summary,data,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'rom':'#b44d3b','affine16':'#dd9b36','affine32':'#217f8c','full64':'#4d529a','dst':'#333333','rk4':'#333333'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    for col,bc in enumerate(summary['config']['boundaries']):
        for method in ('rom','affine16','affine32','full64','dst' if bc=='dirichlet' else 'rk4'):
            rows=[r for r in summary['groups'] if r['boundary']==bc and r['method']==method and (method!='rom' or r['setting']==.0025) and (method!='rk4' or r['setting']==.45)]
            label='MLP16' if method=='rom' else method
            for ax,key,scale in ((axes[0,col],'physical_error_worst',100),(axes[1,col],'query_median',1000)):
                ax.plot([r['intervals'] for r in rows],[scale*r[key] for r in rows],'-o',color=colors[method],label=label)
                ax.set_yscale('log');ax.set_xticks(summary['config']['meshes']);ax.grid(alpha=.2)
        axes[0,col].set_title('Reflective' if bc=='dirichlet' else 'Absorbing');axes[0,col].set_ylabel('Worst required physical error (%)')
        axes[1,col].set_ylabel('Complete query median (ms)');axes[1,col].set_xlabel('Spatial intervals per axis');axes[0,col].legend(fontsize=8)
    fig.suptitle('Frozen learned bank: matched dimensions and linear ladder\nTwo development cases per boundary; common-grid initial-state normalization')
    fig.tight_layout();fig.savefig(out/'dynamics-accuracy-cost.png',dpi=180);fig.savefig(out/'dynamics-accuracy-cost.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    n=max(summary['config']['meshes']);case=summary['config']['validation_indices'][-1]
    times=np.arange(49)*summary['config']['observation_dt']
    for col,bc in enumerate(summary['config']['boundaries']):
        for method in ('rom','affine16','affine32','full64','dst' if bc=='dirichlet' else 'rk4'):
            setting=.0025 if method=='rom' else .45 if method=='rk4' else 0.
            row=next(r for r in data['invocations'] if (r['boundary'],r['intervals'],r['case'],r['method'],r['setting'],r['repetition'])==(bc,n,case,method,setting,0))
            label='MLP16' if method=='rom' else method
            for ax,name,key in ((axes[0,col],'displacement','initial_normalized'),(axes[1,col],'energy_state','current_relative')):
                values=np.array([np.nan if x is None else x for x in row['physical_reference_error'][name][key]])
                ax.plot(times,np.where(values>1e-12,values,np.nan),color=colors[method],label=label)
                ax.set_yscale('log');ax.grid(alpha=.2)
        axes[0,col].set_title('Reflective' if bc=='dirichlet' else 'Absorbing')
        axes[0,col].set_ylabel('Displacement error / initial norm');axes[0,col].legend(fontsize=8)
        axes[1,col].set_ylabel('Energy-state error / current norm');axes[1,col].set_xlabel('Physical time')
    fig.suptitle(f'Wave error evolving: development case {case}, {n} intervals\nInitial-normalized displacement and current-relative phase energy')
    fig.tight_layout();fig.savefig(out/'dynamics-error-evolution.png',dpi=180);fig.savefig(out/'dynamics-error-evolution.pdf');plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);main(ap.parse_args().record)
