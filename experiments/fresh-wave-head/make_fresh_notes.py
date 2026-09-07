"""Generate the single fresh-wave question report strictly from saved run JSONs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def number(value,percent=False):
    if value is None:
        return "—"
    if percent:
        return f"{100*value:.4e}%" if value!=0 and abs(value)<1e-6 else f"{100*value:.4f}%"
    return f"{value:.8g}"


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+["| "+" | ".join(str(v) for v in row)+" |" for row in rows])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--result",action="append",required=True)
    ap.add_argument("--refinement",action="append",default=[])
    ap.add_argument("--verification-result")
    ap.add_argument("--out",required=True)
    ap.add_argument("--reviewed",action="store_true")
    args=ap.parse_args()
    loaded=[(Path(p),json.loads(Path(p).read_text())) for p in args.result]
    refined=[(Path(p),json.loads(Path(p).read_text())) for p in args.refinement]
    verification=None if args.verification_result is None else json.loads(Path(args.verification_result).read_text())
    boundaries={}
    for path,run in loaded:
        if run.get("final_test_opened") or run.get("provenance",{}).get("jax_backend")!="gpu":
            raise RuntimeError("Unexpected cohort/backend provenance")
        for bc,value in run["boundary_results"].items():
            if bc in boundaries:
                raise RuntimeError("Duplicate BC; select one declared campaign invocation per BC")
            boundaries[bc]=(path,run,value)
    status="independently reviewed" if args.reviewed else "provisional pending independent result review"
    lines=["# Fresh waves: learned spatial bank, coefficient heads and actual evolution",f"This report covers newly verified absorbing and fixed-wall scalar waves, learned-bank reconstruction and actual reduced trajectories. The numerical results below are {status}; all earlier wave experiments are excluded from this evidence.",""]
    for bc,(_,run,entry) in boundaries.items():
        accepted=sum(bool(arm['accuracy_passed']) for arm in entry['arms'])
        full=next((b for b in entry['linear_baselines'] if b['label']=='learned_bank_linear_r'),None)
        linear_pass=full is not None and all(full[key]['outliers']==0 and full[key]['nonfinite']==0 for key in ('displacement','velocity','energy_state'))
        lines.append(f"For `{bc}`, {accepted} of {len(entry['arms'])} head/repeat runs meet the predeclared engineering target. The unrestricted full-bank linear baseline {'meets' if linear_pass else 'does not meet'} the same physical-error ceiling; its time propagation is independent of the nonlinear head.")
    lines += ["", "The learned spatial span supports accurate linear evolution at its full dimension. The compressed nonlinear heads have larger representation and trajectory errors; this comparison leaves the smaller state dimension and nonlinear dynamics coupled. It does not isolate a defective spatial bank or establish that nonlinear heads cannot work."]
    for bc,(_,run,entry) in boundaries.items():
        capacities={a['kind']:a['parameter_count'] for a in entry['arms']} if all('kind' in a for a in entry['arms']) else {a['name'].removesuffix('_velocity'):a['parameter_count'] for a in entry['arms']}
        repeats=len({a['optimizer_seed'] for a in entry['arms']})
        selected_nonstationary=sum(a['latent_fit']['nonstationary'] for a in entry['arms'])
        pairs=[]
        for arm in entry['arms']:
            if arm['name'].endswith('_velocity'):
                base=next(a for a in entry['arms'] if a['name']==arm['name'].removesuffix('_velocity') and a['optimizer_seed']==arm['optimizer_seed'])
                primary=lambda a:next(s for s in a['rollout']['summaries'] if s['dt']==a['rollout']['primary_dt'])
                pairs.append((arm['validation']['tangent']['mean']<base['validation']['tangent']['mean'],primary(arm)['energy_state']['median']>primary(base)['energy_state']['median'],arm['name']))
        unresolved=sum(not a['rollout']['refinement_passed'] for a in entry['arms'])
        lines.append(f"In `{bc}`, the MLP has {capacities['mlp']} head parameters and the quadratic head has {capacities['quadratic']}; this is not a parameter-matched architecture comparison. The {repeats} optimizer repeats share the same data. The velocity penalty improves mean tangent fitting in {sum(p[0] for p in pairs)} of {len(pairs)} matched comparisons, while median trajectory energy-state error worsens in {sum(p[1] for p in pairs)} at the original primary step. The latter comparison is qualified by {unresolved} head/repeat runs failing the original timestep check; their temporal accuracy remains unresolved at that step. Selected latent fits include {selected_nonstationary} nonstationary states; these remain failures of the declared fitting gate.")
    if refined:
        fine_cases=[case for _,record in refined for case in record['fine_diagnostic']['finest_two_refinement']]
        fine_passes=sum(record['fine_diagnostic']['refinement_passed'] for _,record in refined)
        lines += ["",f"The frozen-checkpoint continuation passes the additional finest-pair timestep check in {fine_passes} of {len(refined)} selected head/repeat runs and {sum(case['passed'] for case in fine_cases)} of {len(fine_cases)} case/repeat comparisons. Remaining failures stay temporally unresolved. These checks preserve the original primary-step verdict; the converged follow-ups still have large physical trajectory errors."]
    lines += ["", "A proposed next controlled test is a latent-dimension ladder using the same frozen spatial bank and MLP training setup, alongside linear models at each matching dimension. That would separate compression from nonlinear evolution before adding more elaborate heads. This proposal has not been run."]
    lines += ["", "## Scope and reference",""]
    for bc,(path,run,entry) in boundaries.items():
        config=run["config"]
        ref=config["reference_evidence"]
        lines.append(f"The `{bc}` run uses {config['n']} intervals per axis, {config['rank']} learned bank functions, {config['latent']} latent coordinates, {config['train_count']} training trajectories and {config['validation_count']} validation trajectories through time {number(config['end_time'])}. Its final cohort remains closed. Reference job `{ref['job_id']}` and source `{ref['source_commit']}` precede training; reference uncertainty and the absorber's boundary-model error remain distinct from reduced-model error.")
        lines.append(f"The provisional engineering target requires every validation trajectory to complete with time-maximum displacement, velocity and error-state-energy errors at most {number(config['accuracy_target'],True)}. The finest two predeclared steps must agree within {number(config['rom_refinement_target'],True)} on all three physical scales; latent-fit stationarity, rank and budget-stability checks also remain mandatory.")
    if verification is not None:
        reference_rows=[]
        for bc in ('dirichlet','absorbing'):
            estimates=[r['conditional_256_error_estimate'] for r in verification['contraction_estimates'] if r['bc']==bc]
            reference_rows.append([bc,number(max(estimates),True),'Conditional observed-contraction estimate on the declared empirical sample'])
        sine=verification['independent_sine_all_validation']+verification['independent_sine_extremes']
        spatial=max(r['max_discrete_reference_error'] for r in sine)
        spectral=max(r['max_spectral_self_error'] for r in sine)
        temporal=max(s['semidiscrete_state_error'] for r in verification['smooth_family'] if r.get('bc')=='dirichlet' and r.get('n')==256 for s in r['independent_sine'])
        reference_rows.append(['dirichlet',number(spatial+spectral+temporal,True),'Independent continuum-sine discrepancy plus spectral-self and measured RK4 contributions'])
        lines += ['',table(['Boundary','Reference uncertainty measure','Scope'],reference_rows),'']
    lines += ["", r"Write $E(u,v)=\tfrac12(v^TMv+u^TKu)$, $U_0=\sqrt{u_0^TMu_0}$ and $V_0=\sqrt{2E(u_0,v_0)}$. The pointwise-in-time normalized errors are $e_u=\sqrt{\delta u^TM\delta u}/U_0$, $e_v=\sqrt{\delta v^TM\delta v}/V_0$, and $e_E=\sqrt{E(\delta u,\delta v)/E(u_0,v_0)}$. These fixed trajectory scales also normalize timestep differences. Error-state energy is the energy of the state difference, rather than a difference of solution energies.","", "Representation means and medians pool the stored validation states; rollout means and medians summarize each trajectory's maximum over stored times. They are not time-RMS errors or continuous-time supremum bounds. The first-order absorbing condition has physical/model reflection at oblique incidence. Heads receive only latent coordinates: no physical family descriptor or time enters the head.","","## Learned bank and fresh linear baselines",""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        bank=entry["bank"]
        for split,floor in bank["floors"].items():
            rows.append([bc,split,number(floor["displacement"]["mean"],True),number(floor["displacement"]["median"],True),number(floor["displacement"]["worst"],True),number(floor["velocity"]["mean"],True),number(bank["raw_rank_ratio"])])
    lines += [table(["Boundary","Split","Bank displacement mean","Median","Worst","Bank velocity mean","Raw bank rank ratio"],rows),"","Bank floors are unrestricted projections onto the trained neural span. QR only changes that span's coordinates. A fresh coefficient PCA initializes each head's affine map; it does not replace the learned spatial network with a POD bank.",""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        for baseline in entry["linear_baselines"]:
            for metric in ('displacement','velocity','energy_state'):
                summary=baseline[metric]
                rows.append([bc,baseline["label"],baseline["rank"],metric,number(summary["mean"],True),number(summary["median"],True),number(summary["worst"],True),summary["outliers"]])
    lines += [table(["Boundary","Fresh baseline","Dimension","Time-maximum metric","Mean","Median","Worst","Outliers"],rows),"","Randomized POD is an explicitly approximate linear comparator trained from the fresh training data. Its linear dynamics and the unrestricted learned-bank dynamics use independent matrix-exponential propagation.","","## Unseen-state representation",""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        for arm in entry["arms"]:
            val=arm["validation"]
            rows.append([bc,arm["name"],arm["optimizer_seed"],arm["parameter_count"],number(val["reconstruction"]["mean"],True),number(val["reconstruction"]["median"],True),number(val["reconstruction"]["worst"],True),val["reconstruction"]["outliers"],number(val["tangent"]["mean"],True),arm["latent_fit"]["nonstationary"],val["rank_failures"]])
    lines += [table(["Boundary","Head/objective","Repeat seed","Head parameters","Reconstruction mean","Median","Worst","Reconstruction outliers","Tangent mean","Nonstationary fits","Rank failures"],rows),"","Every validation state is fitted from eight declared starts at both independent budgets. Saved objectives, gradients, projected stationarity, ranks and stopping reasons retain failed fits. These local multistart results do not establish global minima.","","## Actual reduced trajectories",""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        for arm in entry["arms"]:
            rollout=arm["rollout"]
            primary=next(s for s in rollout["summaries"] if s["dt"]==rollout["primary_dt"])
            temporal='Pairwise check passed' if rollout['refinement_passed'] else 'Time unresolved'
            rows.append([bc,arm["name"],arm["optimizer_seed"],number(primary["dt"]),temporal,primary["completed"],primary["failed"],number(primary["energy_state"]["mean"],True),number(primary["energy_state"]["median"],True),number(primary["energy_state"]["worst"],True),primary["energy_state"]["outliers"],str(arm["accuracy_passed"])])
    lines += [table(["Boundary","Head/objective","Repeat seed","Primary step","Original temporal status","Complete","Failed","Energy-state mean","Median","Worst","Energy outliers","Engineering target passed"],rows),""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        for arm in entry['arms']:
            primary=next(s for s in arm['rollout']['summaries'] if s['dt']==arm['rollout']['primary_dt'])
            d,v=primary['displacement'],primary['velocity']
            temporal='Pairwise check passed' if arm['rollout']['refinement_passed'] else 'Time unresolved'
            rows.append([bc,arm['name'],arm['optimizer_seed'],temporal,number(d['mean'],True),number(d['median'],True),number(d['worst'],True),d['outliers'],number(v['mean'],True),number(v['median'],True),number(v['worst'],True),v['outliers']])
    lines += [table(['Boundary','Head/objective','Repeat seed','Original temporal status','Displacement mean','Median','Worst','Outliers','Velocity mean','Median','Worst','Outliers'],rows),"","These values summarize each trajectory's maximum error over time. Means, medians and worst values are conditional on finite cases; failed/nonfinite trajectories count as outliers and prevent acceptance. The primary step was fixed before training. Physical velocity is the decoder Jacobian applied to the latent velocity; no finite-difference replacement or phase alignment is used.","","## Time-step refinement and phase diagnostics",""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        for arm in entry["arms"]:
            rollout=arm["rollout"]
            refinement=rollout["finest_two_refinement"]
            valid=[x for x in refinement if x["both_completed"]]
            maxima=[max((x[key] for x in valid),default=None) for key in ('max_displacement_difference','max_velocity_difference','max_energy_state_difference')]
            primary=[x for x in rollout["cases"] if x["dt"]==rollout["primary_dt"] and x["completed"]]
            phase=max((x["max_defined_modal_phase_error"] for x in primary if x["max_defined_modal_phase_error"] is not None),default=None)
            vanished=sum(x["vanished_mode_observations"] for x in primary)
            rows.append([bc,arm["name"],arm["optimizer_seed"],*[number(x,True) for x in maxima],len(valid),sum(x['passed'] for x in refinement),','.join(str(x['case']) for x in refinement if not x['passed']) or '—',str(rollout["refinement_passed"]),number(phase),vanished])
    lines += [table(["Boundary","Head/objective","Repeat seed","Finest-two displacement difference","Velocity difference","Energy-state difference","Both-step completions","Cases passing refinement","Unresolved case indices","Refinement passed","Worst defined phase error (radians)","Vanished-mode observations"],rows),"","A failed timestep gate leaves that trajectory's temporal accuracy unresolved and prevents attributing its error entirely to the trained architecture. Case indices are zero-based. Reflective phases use semidiscrete standing-mode frequencies. Absorbing sine projections are diagnostic coordinates, not absorbing-system eigenmodes. Vanished predicted amplitudes have undefined phase and explicit flags. Raw files also preserve valid-segment unwrapped phase drift, wall-strip peak-time differences, absorbing means, physical boundary power and integrated energy balance.","","## Provenance and limits",""]
    diagnostic_rows=[]
    for bc,(_,run,entry) in boundaries.items():
        for arm in entry['arms']:
            primary=[r for r in arm['rollout']['cases'] if r['dt']==arm['rollout']['primary_dt'] and r['completed']]
            balance=max((r['max_energy_balance_relative'] for r in primary),default=None)
            initial=arm['validation']['initial_reconstruction']
            diagnostic_rows.append([bc,arm['name'],arm['optimizer_seed'],number(initial['mean'],True),number(initial['median'],True),number(initial['worst'],True),number(arm['zero_state']['mass_norm']),number(balance,True)])
    lines[-2:]=["## Initial fit, zero-state bias and energy balance","",table(['Boundary','Head/objective','Repeat seed','Initial reconstruction mean','Median','Worst','Zero-target fitted mass norm','Primary worst energy-balance defect'],diagnostic_rows),"","The zero-target quantity is an absolute mass norm, distinct from normalized trajectory errors. Energy balance uses each ROM's own initial energy and integrated physical boundary power; small balance defect alone does not establish accurate displacement or phase.","","## Provenance and limits",""]
    if refined:
        refinement_rows=[]
        for path,record in refined:
            if record['retraining_performed'] or record['initial_fitting_performed'] or record['final_test_opened']:
                raise RuntimeError('Unexpected checkpoint-refinement scope')
            fine_dt=record['diagnostic_config']['rom_dts'][-1]
            fine=next(s for s in record['fine_diagnostic']['summaries'] if s['dt']==fine_dt)
            parity=max((r.get('max_energy_state_difference',0.) for r in record['old_fine_physical_parity']),default=None)
            refinement_rows.append([record['boundary'],record['name'],record['optimizer_seed'],str(record['original_accuracy_passed']),str(record['old_fine_parity_passed']),number(parity,True),number(fine_dt),fine['completed'],fine['failed'],number(fine['energy_state']['mean'],True),number(fine['energy_state']['median'],True),number(fine['energy_state']['worst'],True),fine['energy_state']['outliers'],str(record['fine_diagnostic']['refinement_passed'])])
        lines[-2:]=['## Frozen-checkpoint time-step continuation','',table(['Boundary','Head/objective','Repeat seed','Original primary target passed','Old-fine parity passed','Old-fine energy-state discrepancy','New finest step','Complete','Failed','Finest energy-state mean','Median','Worst','Outliers','New finest-two refinement passed'],refinement_rows),'','These separate numerical follow-ups were selected only because the original time-step refinement failed. They retain the exact trained bank/head and stored initial latent position and velocity, repeat the original finest step for hardware parity, and then evaluate both additional predeclared steps on every validation case. They preserve the original primary-step verdict and cannot be presented as retrained architectural improvements.','', '## Provenance and limits','']
        adjacent_rows=[]
        normalization_rows=[]
        contraction_rows=[]
        for path,record in refined:
            steps=record['diagnostic_config']['rom_dts']
            for first,second in zip(steps[:-1],steps[1:]):
                cases=[r for r in record['adjacent_refinement'] if r['coarse_dt']==first and r['fine_dt']==second]
                complete=[r for r in cases if r['both_completed']]
                maxima=[max((r.get(key) for r in complete if r.get(key) is not None),default=None) for key in ('max_displacement_difference','max_velocity_difference','max_energy_state_difference')]
                adjacent_rows.append([record['name'],record['optimizer_seed'],number(first),number(second),len(complete),sum(r['passed'] for r in cases),','.join(str(r['case']) for r in cases if not r['passed']) or '—',*[number(x,True) for x in maxima],sum(not r['passed'] for r in cases)])
            normalization_rows.append([record['name'],record['optimizer_seed'],record['provenance']['source_commit'],number(record['truth_parity']['scales_relative_parity']),str(record['truth_parity']['bit_identical_truth'])])
            orders=[r['observed_order'] for r in record['observed_contraction'] if r.get('observed_order') is not None]
            ratios=[r['energy_difference_ratio'] for r in record['observed_contraction'] if r.get('energy_difference_ratio') is not None]
            contraction_rows.append([record['name'],record['optimizer_seed'],len(orders),number(statistics.median(ratios)) if ratios else '—',number(min(orders)) if orders else '—',number(statistics.median(orders)) if orders else '—',number(max(orders)) if orders else '—'])
        lines[-2:]=[table(['Frozen head/objective','Repeat seed','Coarser step','Finer step','Both-step completions','Cases passing refinement','Unresolved case indices','Worst displacement difference','Worst velocity difference','Worst energy-state difference','Refinement failures'],adjacent_rows),'',table(['Frozen head/objective','Repeat seed','Defined-order cases','Median difference contraction','Minimum observed order','Median observed order','Maximum observed order'],contraction_rows),'','Contraction divides each case\'s finer energy-state difference by its preceding difference; observed order is the negative base-two logarithm of this ratio. It need not be asymptotic, and small roundoff-scale differences can make the order uninformative.','',table(['Frozen head/objective','Repeat seed','Continuation source','Regenerated-vs-original relative scale discrepancy','Bit-identical regenerated truth'],normalization_rows),'','The first continuation wrapper used regenerated normalization values after checking their agreement; its successor restores the original recorded values exactly after the same check. This bookkeeping distinction is retained with source hashes and measured scale discrepancies. Neither version changes the trained coefficients or initial latent position/velocity; the original trajectory verdict remains fixed. Passing a pairwise threshold is empirical timestep agreement, not a rigorous integration-error bound. Cases still failing this bounded continuation remain temporally unresolved; no further refinement was used to select a preferred architecture.','', '## Provenance and limits','']
    for path,run in loaded:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"- Source result: `{path}`; SHA-256 `{digest}`; GPU job `{run['provenance'].get('job_id')}`; source commit `{run['provenance'].get('source_commit')}`; devices `{run['provenance'].get('device_kind')}`.")
    for path,run in refined:
        lines.append(f"- Frozen-checkpoint continuation: `{path}`; SHA-256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`; GPU job `{run['provenance'].get('job_id')}`; pinned mathematical source `{run['frozen_mathematics']['source_commit']}`.")
    lines += ["","The scope is the declared smooth Gaussian-core compact family in two dimensions. Head weights and learned banks are fresh per PDE/boundary configuration. Initial fitting uses full-field initial-state projections, so this first accuracy campaign makes no grid-independent cold-start or speed claim. Results do not establish three-dimensional wave transfer or performance outside this family.","","## Glossary", "",
              "- **Boundary / fixed wall / absorber:** the physical edge condition; zero wall displacement produces sign-reversing reflection, while the local radiation condition approximates outgoing waves.",
              "- **Bank / head / latent coordinates:** spatial neural features, the coefficient function multiplying them, and its internal coordinates.",
              "- **MLP / quadratic / velocity objective:** a multilayer SiLU neural coefficient map, an affine map plus unique quadratic latent products, and an added training penalty for physical velocities outside the decoder tangent space.",
              "- **Rank ratio / parameters / seed:** smallest-to-largest singular value, number of trainable head coefficients, and the recorded optimizer-repeat random seed.",
              "- **Training / validation / final cohort:** data used to fit models, unseen development trajectories, and reserved unopened trajectories.",
              "- **POD / PCA / QR:** an approximate linear data subspace, a statistical coordinate initialization, and an orthonormal change of basis.",
              "- **Mass norm / energy-state error:** a spatially weighted field norm, and the energy norm of the difference in displacement and physical velocity.",
              "- **Reconstruction / tangent / nonstationary:** fitted displacement accuracy, representable physical-velocity accuracy, and a latent fit that has not met its declared first-order optimality check.",
              "- **Mean / median / worst / outlier:** average, middle value, largest finite value, and a case exceeding the predeclared threshold or failing numerically.",
              "- **Primary step / refinement / completion / contraction / observed order:** the predetermined reported time step, comparison after reducing it, successful finite integration through every required step, reduction in successive timestep differences, and its measured power-law rate.",
              "- **Phase / vanished mode / wall-strip peak:** oscillation angle, an amplitude too small to define that angle, and a boundary-neighborhood signal peak used only as a timing proxy.",
              "- **Boundary power / energy balance / absorbing mean:** instantaneous dissipative power, energy plus integrated power relative to its initial value, and average residual displacement that energy alone cannot control.",
              "- **Engineering target:** the declared provisional accuracy/completion requirement; satisfying it is specific to this bounded family and reference budget.",""]
    Path(args.out).write_text("\n".join(lines))


if __name__=="__main__":
    main()
