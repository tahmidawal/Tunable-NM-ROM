"""Generated larger-wave-head results with accuracy-only timings kept ineligible."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

NAMES=('displacement','velocity','energy_state')
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def median(values):return float(np.median(values))
def max_error(metric):return max(metric[k]['max_initial_normalized'] for k in NAMES)

def main(record):
    native=record/'cluster/out/pilot';out=record/'analysis';data=json.loads((native/'result.json').read_text());cfg=data['config']
    audit=json.loads((out/'audit.json').read_text());assert audit['passed'] and audit['result_sha256']==sha(native/'result.json')
    reference_config=json.loads((record/'cluster/in/dirichlet/campaign-config.json').read_text())
    refinement_target=reference_config['rom_refinement_target']
    summary=dict(provenance=data['provenance'],config=cfg,result_sha256=sha(native/'result.json'),audit_sha256=sha(out/'audit.json'),
        data_regeneration=data['training'],head_training=data['head_training'],groups=[],time_refinement=[],representation=[],reference_uncertainty=[],accuracy_controls=[],development_selection=[],
        time_refinement_target=refinement_target,time_refinement_target_source='Unchanged original fresh campaign-config.json rom_refinement_target',final_test_opened=False,
        timing_note='Only repeated invocations with comparison_eligible=true enter timing summaries. Single fine controls are accuracy-only even though their complete-query latency is preserved in raw JSON.')
    allrows=data['invocations']+data['accuracy_controls']
    for bc in cfg['boundaries']:
        for n in cfg['meshes']:
            for ci in cfg['validation_indices']:
                with np.load(native/f'mesh_{bc}_{n}.npz') as f:stiffness=f['stiffness']
                for seed in cfg['training']['optimizer_seeds']:
                    method=f'new_mlp32_seed{seed}'
                    ladder=cfg['nonlinear_dts']+[cfg['accuracy_only_dt']]
                    for coarse,fine in zip(ladder[:-1],ladder[1:]):
                        rr=[next(r for r in allrows if (r['boundary'],r['intervals'],r['case'],r['method'],r['setting'],r['repetition'])==(bc,n,ci,method,dt,0)) for dt in (coarse,fine)]
                        with np.load(native/(rr[0]['invocation_id']+'.npz')) as a,np.load(native/(rr[1]['invocation_id']+'.npz')) as b:
                            da=a['coefficients']-b['coefficients'];db=a['velocity_coefficients']-b['velocity_coefficients']
                            m=rr[0]['same_grid_discrepancy'];c=rr[0]['parameters'][5]
                            norms=dict(displacement=float(np.max(np.linalg.norm(da,axis=1)/m['displacement']['initial_scale'])),velocity=float(np.max(np.linalg.norm(db,axis=1)/m['velocity']['initial_scale'])),energy_state=float(np.max(np.sqrt(np.maximum(0,np.sum(db*db,axis=1)+c*c*np.einsum('tr,rs,ts->t',da,stiffness,da)))/m['energy_state']['initial_scale'])))
                            complete=all(r['completed'] for r in rr)
                            summary['time_refinement'].append(dict(boundary=bc,intervals=n,case=ci,method=method,coarse_dt=coarse,fine_dt=fine,maxima=norms,maximum_required_difference=max(norms.values()),completed=complete,passed=complete and max(norms.values())<=refinement_target,fine_timing_comparison_eligible=rr[1]['comparison_eligible']))
        for ci in cfg['validation_indices']:
            ref=next(r for r in data['references'] if r['boundary']==bc and r['case']==ci and 'fine_intervals' in r)
            if bc=='dirichlet':entry=dict(empirical_self_difference=max_error(ref['self_refinement']))
            else:
                nested=next(r for r in data['references'] if r['boundary']==bc and r['case']==ci and 'adjacent_coarse_to_this_mesh' in r)
                dc,df=max_error(nested['adjacent_coarse_to_this_mesh']),max_error(nested['same_grid_to_physical_reference']);ratio=df/dc;temporal=max_error(ref['temporal_refinement'])
                entry=dict(adjacent_coarse_difference=dc,adjacent_fine_difference=df,observed_spatial_ratio=ratio,temporal_difference=temporal,conditional_fine_reference_estimate=df*ratio/(1-ratio)+temporal if 0<=ratio<1 else None)
            summary['reference_uncertainty'].append(dict(boundary=bc,case=ci,proven_bound=None,**entry))
    grouped=defaultdict(list)
    for row in data['invocations']:
        assert row['comparison_eligible'] and row['setting']!=cfg['accuracy_only_dt']
        grouped[row['boundary'],row['intervals'],row['method'],row['setting']].append(row)
    for (bc,n,method,setting),rows in grouped.items():
        cases=[]
        for ci in cfg['validation_indices']:
            rr=sorted((r for r in rows if r['case']==ci),key=lambda x:x['repetition']);assert len(rr)==cfg['repetitions']
            refinement=next((r for r in summary['time_refinement'] if (r['boundary'],r['intervals'],r['case'],r['method'],r['coarse_dt'])==(bc,n,ci,method,setting)),None)
            cases.append(dict(case=ci,query_repetitions=[r['seconds']['complete_query'] for r in rr],query_median=median([r['seconds']['complete_query'] for r in rr]),
                phase_medians={key:median([r['seconds'][key] for r in rr]) for key in rr[0]['seconds']},completed=all(r['completed'] for r in rr),
                stationary=None if 'mlp' not in method else all(r['fit_stationary'] for r in rr),
                time_refinement_passed=None if refinement is None else refinement['passed'],
                worst_physical_error=max(max_error(r['physical_reference_error']) for r in rr),
                final_current_relative={key:rr[0]['physical_reference_error'][key]['current_relative'][-1] for key in NAMES},
                final_absolute={key:rr[0]['physical_reference_error'][key]['absolute'][-1] for key in NAMES},
                final_reference_norm={key:rr[0]['physical_reference_error'][key]['reference_norm'][-1] for key in NAMES},
                final_vanishing={key:rr[0]['physical_reference_error'][key]['reference_vanishing'][-1] for key in NAMES}))
        summary['groups'].append(dict(boundary=bc,intervals=n,method=method,setting=setting,cases=cases,query_median=median([c['query_median'] for c in cases]),
            physical_error_median=median([c['worst_physical_error'] for c in cases]),physical_error_worst=max(c['worst_physical_error'] for c in cases),
            failed_cases=sum(not c['completed'] for c in cases),nonstationary_cases=sum(c['stationary'] is False for c in cases),
            time_refinement_failed_cases=sum(c['time_refinement_passed'] is False for c in cases),
            errors_above_targets={str(t):sum(not c['completed'] or c['worst_physical_error']>t for c in cases) for t in cfg['accuracy_targets']}))
    for g in summary['groups']:
        baseline=next(x for x in summary['groups'] if x['boundary']==g['boundary'] and x['intervals']==g['intervals'] and x['method']==('dst' if g['boundary']=='dirichlet' else 'rk4') and x['setting']==(0. if g['boundary']=='dirichlet' else .45))
        g['paired_FOM_over_method_ratio']=median([next(b for b in baseline['cases'] if b['case']==c['case'])['query_median']/c['query_median'] for c in g['cases']])
    for r in data['accuracy_controls']:
        assert not r['comparison_eligible']
        summary['accuracy_controls'].append(dict(boundary=r['boundary'],intervals=r['intervals'],case=r['case'],method=r['method'],setting=r['setting'],
            completed=r['completed'],stationary=r['fit_stationary'],worst_physical_error=max_error(r['physical_reference_error']),comparison_eligible=False,
            final_current_relative={key:r['physical_reference_error'][key]['current_relative'][-1] for key in NAMES},source_invocation_id=r['invocation_id']))
    for diag in data['diagnostics']:
        arm=next(a for a in diag['arms'] if a['arm']==diag['model'])
        summary['representation'].append(dict(boundary=diag['boundary'],intervals=diag['intervals'],case=diag['case'],model=diag['model'],
            displacement=arm['snapshot_metrics']['displacement']['max_initial_normalized'],tangent_velocity=arm['snapshot_metrics']['velocity']['max_initial_normalized'],
            snapshot_energy_state=arm['snapshot_metrics']['energy_state']['max_initial_normalized'],zero_mass_norm=arm['zero_field_mass_norm'],zero_initial_scaled=arm['zero_field_initial_displacement_scaled'],
            normal_force_absolute_max=max(arm['normal_force_absolute'][:-1]),normal_force_fixed_scaled_max=max(arm['normal_force_fixed_scaled'][:-1]),force_scale=diag['force_scale'],
            selected_fit_nonstationary=sum(not a for a in diag['fitting'][-1]['selected_stationary']),fit_budget_objective_change=diag['fit_budget_objective_max_change'],minimum_rank_ratio=min(arm['rank_ratios'])))
    # This is development selection among repeated K32 time steps only. No fine-control latency is present here.
    for bc in cfg['boundaries']:
        for n in cfg['meshes']:
            for seed in cfg['training']['optimizer_seeds']:
                method=f'new_mlp32_seed{seed}'
                for target in cfg['accuracy_targets']:
                    eligible=[g for g in summary['groups'] if (g['boundary'],g['intervals'],g['method'])==(bc,n,method) and g['physical_error_worst']<=target and not g['failed_cases'] and not g['nonstationary_cases'] and not g['time_refinement_failed_cases']]
                    selected=min(eligible,key=lambda g:g['query_median']) if eligible else None
                    summary['development_selection'].append(dict(boundary=bc,intervals=n,method=method,target=target,selected_dt=None if selected is None else selected['setting'],
                        query_median=None if selected is None else selected['query_median'],physical_error_worst=None if selected is None else selected['physical_error_worst'],
                        raw_same_grid_FOM_ratio=None if selected is None else selected['paired_FOM_over_method_ratio'],rigorous_reference_qualified=False,independent_confirmation=False,
                        note='Within-job development selection among repeated time-step rows. Uncertainty bounds remain empirical and a coarse-FOM envelope is absent.'))
    write(out/'summary.json',summary);render(summary,audit,out/'FINDINGS.md');plot(summary,data,out)
    print(json.dumps(dict(summary=str(out/'summary.json'),timed_comparison_invocations=len(data['invocations']),accuracy_only_invocations=len(data['accuracy_controls'])),indent=2))


def render(s,audit,path):
    cfg=s['config'];p=s['provenance'];lines=['# Larger nonlinear wave heads and time-step controls','',
        'This generated report covers the completed fixed-bank larger-head development pilot. Numbers are provisional on the declared original development inputs; the final cohort remains unopened.','',
        f"Scientific source `{p['source_commit']}`, job `{p['job_id']}`, device `{p['device_kind'][0]}`. Native result SHA-256 `{s['result_sha256']}`.",'',
        'Both new heads use the original reconstruction-only training protocol and cohort, with the bank frozen. Their common affine32 initializer is the matched linear control. The unchanged MLP16 is retained at its original primary time step. The fixed weak bank has more equations than latent coordinates, with the explicitly changed overdetermination ratio recorded in the configuration.','',
        '| Boundary | Endpoint | Head coordinates | Seed | Updates | Training seconds | Final recorded training objective |','|---|---|---:|---:|---:|---:|---:|']
    for t in s['head_training']:lines.append(f"| {t['boundary']} | {t['name']} | {t['configuration_dimension']} | {t['optimizer_seed']} | {t['training']['head_steps']} | {t['training_seconds_including_first_compile']:.9g} | {t['history'][-1][1]:.9g} |")
    fine=max(cfg['meshes'])
    for bc in cfg['boundaries']:
        groups=[g for g in s['groups'] if (g['boundary'],g['intervals'])==(bc,fine)]
        frozen=next(g for g in groups if g['method'].startswith('frozen_'))
        affine=next(g for g in groups if g['method']=='affine32')
        coarse=[g for g in groups if g['method'].startswith('new_') and g['setting']==cfg['nonlinear_dts'][0]]
        primary=[g for g in groups if g['method'].startswith('new_') and g['setting']==frozen['setting']]
        error_range=f"{100*min(g['physical_error_worst'] for g in coarse):.6g}%–{100*max(g['physical_error_worst'] for g in coarse):.6g}%"
        cost_range=f"{min(g['query_median'] for g in primary):.6g}–{max(g['query_median'] for g in primary):.6g}"
        lines += ['',f"At `{fine}` intervals for `{bc}`, the two larger heads at the coarsest repeated step have worst required errors `{error_range}`, versus `{100*frozen['physical_error_worst']:.6g}%` for the frozen smaller head and `{100*affine['physical_error_worst']:.6g}%` for affine32. At the unchanged primary step, the larger heads cost `{cost_range}` seconds versus `{frozen['query_median']:.6g}` seconds for the smaller head. The faster larger-head rows use a larger time step; increasing dimension alone does not improve speed."]
    lines += ['', 'No nonlinear configuration is faster than the same-job requested-mesh FOM. The two larger-head seeds also fail the declared all-case accuracy targets below their reported worst errors. These results support an accuracy improvement over the smaller head, not an established nonlinear advantage over the matched affine model or FOM.']
    lines += ['', 'Training times include compilation and host work and are not paired warm-GPU training-speed measurements. Both fixed endpoints are reported. No velocity, tangent, curvature, force, energy, rollout or validation objective is added. Original seed data are regenerated on the cluster; saved prior coefficients are lineage checks only.','',
        '| Boundary | Intervals | Method | dt / CFL | Query median ms | Time-max error median | Time-max error worst | Raw same-grid FOM/method | Failed / nonstationary / unresolved cases |','|---|---:|---|---:|---:|---:|---:|---:|---|']
    for g in s['groups']:
        lines.append(f"| {g['boundary']} | {g['intervals']} | {g['method']} | {g['setting']:.6g} | {1000*g['query_median']:.9g} | {g['physical_error_median']:.9g} | {g['physical_error_worst']:.9g} | {g['paired_FOM_over_method_ratio']:.9g} | {g['failed_cases']} / {g['nonstationary_cases']} / {g['time_refinement_failed_cases']} |")
    lines += ['', 'These are medians across case timing medians and medians of per-case FOM/method timing ratios. Physical errors are time maxima on the common observation mesh, taking the maximum of displacement, velocity and phase-energy norms. Displacement is divided by its initial L2 norm; velocity/energy state use $\\sqrt{2E(0)}$. The frozen control has no newly measured time-step pair in this panel, which is distinct from a failed refinement test.','',
        'All repeated query costs include full host inputs, every cold-fit start, speed-dependent work, evolution, dense requested outputs and host transfer. Linear offset forcing and propagation are charged. The listed FOMs solve at the requested mesh. No cheaper coarse-FOM/interpolation envelope is swept, so raw ratios are not a paper cost-to-tolerance comparison.','',
        '| Boundary | Intervals | New head | Adjacent dt pair | Maximum required difference over cases | Failed refinement cases |','|---|---:|---|---|---:|---:|']
    groups=defaultdict(list)
    for r in s['time_refinement']:groups[r['boundary'],r['intervals'],r['method'],r['coarse_dt'],r['fine_dt']].append(r)
    for (bc,n,method,coarse,fine),rows in groups.items():lines.append(f"| {bc} | {n} | {method} | {coarse:.6g} / {fine:.6g} | {max(r['maximum_required_difference'] for r in rows):.9g} | {sum(not r['passed'] for r in rows)} |")
    lines += ['',f"The refinement criterion `{s['time_refinement_target']}` is the unchanged original fresh-campaign criterion on fixed initial displacement/phase-energy scales. The fine step is an accuracy-only control with one complete query per endpoint/case. Its full latency is saved in raw JSON, may include uncached compilation, and never enters the timing table, speed selection or reported ratios. No extra repetition is inferred from it.",'',
        '| Boundary | Intervals | Case | Head | Snapshot displacement | Tangent velocity | Snapshot energy state | Zero-field norm | Selected nonstationary fits |','|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in s['representation']:lines.append(f"| {r['boundary']} | {r['intervals']} | {r['case']} | {r['model']} | {r['displacement']:.9g} | {r['tangent_velocity']:.9g} | {r['snapshot_energy_state']:.9g} | {r['zero_mass_norm']:.9g} | {r['selected_fit_nonstationary']} |")
    lines += ['',f"Snapshot diagnostics use only the declared indices `{cfg['diagnostic_indices']}`. Every head receives the same eight-start rule at budgets `{cfg['diagnostic_fit_budgets']}`, using its own training-code library. All endpoints, gradients, stationarity, rank and budget changes are retained. These are local best-recorded fits, not global optima. Truth-only fits never initialize a measured query. Weak normal force uses the fixed scale $\\|u(0)\\|/T^2$ and is diagnostic, not a trajectory-error or causal certificate.",'',
        '| Absorbing intervals | Method | Step | Final current-relative energy error range |','|---|---|---:|---:|']
    for g in s['groups']:
        if g['boundary']=='absorbing' and ('mlp' in g['method'] or g['method']=='affine32'):
            values=[c['final_current_relative']['energy_state'] for c in g['cases']]
            lines.append(f"| {g['intervals']} | {g['method']} | {g['setting']:.6g} | {min(values):.9g}–{max(values):.9g} |")
    lines += ['', 'Current-relative errors, absolute errors, current truth norms and vanishing flags remain visible as the absorber decays. A small error relative to the initial state does not establish accurate prediction relative to the remaining field.','',
        'A separate [archived absorbing-wave moment diagnostic](../../dynamics02/analysis/ABSORBING-MOMENT.md) distinguishes initial fitting error in the outgoing-wave invariant from subsequent drift. It establishes a missing discrete invariance property in the learned bank, without showing that this defect alone causes the field errors. Neither constant-mode nor initial-moment corrections are tested in this larger-head panel.','',
        f"Independent CPU audit checks `{audit['audited_comparison_invocations']}` repeated comparison calls and `{audit['audited_accuracy_only_invocations']}` ineligible fine controls, with maximum common-grid metric discrepancy `{audit['maximum_physical_metric_difference']:.9g}`. Full-grid ROM fields are reconstructed from bank/coefficient states and checked against retained full-grid references, maximum metric discrepancy `{audit['maximum_ROM_native_grid_metric_difference']:.9g}`. Checkpoint/source/cohort, training initialization/objective, coordinate-bank, physical-velocity, curvature, normal-force, fitting and affine-generator checks are included. Failed trajectories remain recorded and do not become eligible winners.",'',
        'The audit reuses preserved reference trajectories; fine-grid timed FOM fields at nonreference CFLs remain outside its full-grid reconstruction scope. Physical-reference self-differences are empirical and rigorous bounds remain unspecified. Any within-job time-step selection is exploratory development selection, with no independent-cohort confirmation.','',
        '![Larger wave head accuracy and query cost](heads32-accuracy-cost.png)','',
        '![Larger wave head error evolving](heads32-error-evolution.png)','',
        'Error-evolution plots omit values at or below $10^{-12}$ on logarithmic axes; the unmodified values remain in the raw JSON.','',
        '## Plain-language glossary','',
        '- **Bank / head / frozen:** learned spatial functions / reduced-coordinate coefficient map / unchanged model weights.',
        '- **Configuration / phase / weak equations:** displacement coordinates / displacement and velocity coordinates / spatially projected equations.',
        '- **MLP / affine / PCA / endpoint:** nonlinear multilayer head / linear map plus an offset / principal training-coefficient directions / fixed trained model.',
        '- **Reconstruction / tangent / normal force / curvature:** snapshot fit / representable local velocity / force outside its tangent span / acceleration due to decoder bending.',
        '- **Code / seed / stationary / budget / rank:** training coordinates / recorded random generator state / local fitting convergence / maximum iterations / independent local directions.',
        '- **dt / CFL / DST / RK4:** time step / time-step-to-grid ratio / sine-transform propagation / fourth-order integration.',
        '- **Common grid / initial-normalized / current-relative / energy state:** shared physical observation mesh / fixed initial scale / current truth norm / displacement-gradient plus velocity norm.',
        '- **Complete query / repetition / median / paired ratio:** supplied input to requested host output / timing repeat of one case / central value / same-case FOM time divided by model time.',
        '- **Accuracy-only / unresolved / eligible:** retained control excluded from speed comparison / failed time-refinement check / permitted to enter a declared comparison.',
        '- **Reference / empirical / final cohort / provenance:** comparison solution / supported by observed refinement / unopened confirmation inputs / source, data, device and job record.',
        '- **Archive / checksum / audit:** preserved complete output / content fingerprint / independent verification.','']
    path.write_text('\n'.join(lines))


def plot(summary,data,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'frozen_mlp16_seed691200':'#b44d3b','new_mlp32_seed691200':'#217f8c','new_mlp32_seed691201':'#5b58a2','affine32':'#d79d2c','dst':'#333333','rk4':'#333333'}
    labels={'frozen_mlp16_seed691200':'Frozen MLP16','new_mlp32_seed691200':'MLP32 seed691200','new_mlp32_seed691201':'MLP32 seed691201','affine32':'Affine32','dst':'DST FOM','rk4':'RK4 FOM'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));n=max(summary['config']['meshes'])
    for ax,bc in zip(axes,summary['config']['boundaries']):
        for method in labels:
            rows=[g for g in summary['groups'] if g['boundary']==bc and g['intervals']==n and g['method']==method and (method!='rk4' or g['setting']==.45)]
            if not rows:continue
            rows=sorted(rows,key=lambda g:g['query_median'])
            ax.plot([g['query_median']*1000 for g in rows],[g['physical_error_worst']*100 for g in rows],'-o',color=colors[method],label=labels[method])
        step_labels=', '.join(f'{dt:g}' for dt in summary['config']['nonlinear_dts'])
        ax.text(.97,.45,'K32 dt, left to right:\n'+step_labels,transform=ax.transAxes,ha='right',fontsize=8)
        ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.2);ax.set_title('Reflective' if bc=='dirichlet' else 'Absorbing')
        ax.set_xlabel('Complete query median (ms)');ax.set_ylabel('Worst required time-max error (%)');ax.legend(fontsize=7)
    fig.suptitle(f'Fixed learned bank, larger nonlinear heads and time-step controls\n{n} intervals; two development cases per boundary; fine controls excluded from speed plots')
    fig.tight_layout();fig.savefig(out/'heads32-accuracy-cost.png',dpi=180);fig.savefig(out/'heads32-accuracy-cost.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5));times=np.arange(49)*summary['config']['observation_dt'];ci=summary['config']['validation_indices'][-1]
    for col,bc in enumerate(summary['config']['boundaries']):
        for method in labels:
            setting=.0025 if method.startswith('frozen_') else .01 if method.startswith('new_') else .45 if method=='rk4' else 0.
            rows=[r for r in data['invocations'] if (r['boundary'],r['intervals'],r['case'],r['method'],r['setting'],r['repetition'])==(bc,n,ci,method,setting,0)]
            if not rows:continue
            row=rows[0]
            for ax,metric,key in ((axes[0,col],'displacement','initial_normalized'),(axes[1,col],'energy_state','current_relative')):
                values=np.asarray([np.nan if x is None else x for x in row['physical_reference_error'][metric][key]])
                ax.plot(times,np.where(values>1e-12,values,np.nan),color=colors[method],label=labels[method]);ax.set_yscale('log');ax.grid(alpha=.2)
        axes[0,col].set_title('Reflective' if bc=='dirichlet' else 'Absorbing');axes[0,col].set_ylabel('Displacement error / initial norm');axes[0,col].legend(fontsize=7)
        axes[1,col].set_ylabel('Energy-state error / current norm');axes[1,col].set_xlabel('Physical time')
    fig.suptitle(f'Wave error evolving: development case {ci}, {n} intervals\nNew K32 heads shown at predeclared coarse dt=.01; frozen16 retains dt=.0025')
    fig.tight_layout();fig.savefig(out/'heads32-error-evolution.png',dpi=180);fig.savefig(out/'heads32-error-evolution.pdf');plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);main(ap.parse_args().record)
