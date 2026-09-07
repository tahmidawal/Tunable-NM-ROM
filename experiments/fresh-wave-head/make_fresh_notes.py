"""Generate the single fresh-wave question report strictly from saved run JSONs."""
import argparse
import hashlib
import json
from pathlib import Path


def number(value,percent=False):
    if value is None:
        return "—"
    return f"{100*value:.4f}%" if percent else f"{value:.8g}"


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
    lines += ["", "All tabulated relative errors use the initial displacement mass norm or initial state energy as fixed trajectory scales. A time-maximum is the largest stored-time error; error-state energy is not the difference between two solution energies. The first-order absorbing condition has physical/model reflection at oblique incidence. Heads receive only latent coordinates: no physical family descriptor or time enters the head.","","## Learned bank and fresh linear baselines",""]
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
            rows.append([bc,arm["name"],arm["optimizer_seed"],number(primary["dt"]),primary["completed"],primary["failed"],number(primary["energy_state"]["mean"],True),number(primary["energy_state"]["median"],True),number(primary["energy_state"]["worst"],True),primary["energy_state"]["outliers"],str(arm["accuracy_passed"])])
    lines += [table(["Boundary","Head/objective","Repeat seed","Primary step","Complete","Failed","Energy-state mean","Median","Worst","Energy outliers","Engineering target passed"],rows),""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        for arm in entry['arms']:
            primary=next(s for s in arm['rollout']['summaries'] if s['dt']==arm['rollout']['primary_dt'])
            d,v=primary['displacement'],primary['velocity']
            rows.append([bc,arm['name'],arm['optimizer_seed'],number(d['mean'],True),number(d['median'],True),number(d['worst'],True),d['outliers'],number(v['mean'],True),number(v['median'],True),number(v['worst'],True),v['outliers']])
    lines += [table(['Boundary','Head/objective','Repeat seed','Displacement mean','Median','Worst','Outliers','Velocity mean','Median','Worst','Outliers'],rows),"","These values summarize each trajectory's maximum error over time. Means, medians and worst values are conditional on finite cases; failed/nonfinite trajectories count as outliers and prevent acceptance. The primary step was fixed before training. Physical velocity is the decoder Jacobian applied to the latent velocity; no finite-difference replacement or phase alignment is used.","","## Time-step refinement and phase diagnostics",""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        for arm in entry["arms"]:
            rollout=arm["rollout"]
            refinement=rollout["finest_two_refinement"]
            valid=[x for x in refinement if x["both_completed"]]
            maximum=max((x["max_energy_state_difference"] for x in valid),default=None)
            primary=[x for x in rollout["cases"] if x["dt"]==rollout["primary_dt"] and x["completed"]]
            phase=max((x["max_defined_modal_phase_error"] for x in primary if x["max_defined_modal_phase_error"] is not None),default=None)
            vanished=sum(x["vanished_mode_observations"] for x in primary)
            rows.append([bc,arm["name"],arm["optimizer_seed"],number(maximum,True),len(valid),str(rollout["refinement_passed"]),number(phase),vanished])
    lines += [table(["Boundary","Head/objective","Repeat seed","Finest-two energy-state difference","Both-step completions","Refinement passed","Worst defined phase error (radians)","Vanished-mode observations"],rows),"","Reflective phases use semidiscrete standing-mode frequencies. Absorbing sine projections are diagnostic coordinates, not absorbing-system eigenmodes. Vanished predicted amplitudes have undefined phase and explicit flags. Raw files also preserve valid-segment unwrapped phase drift, wall-strip peak-time differences, absorbing means, physical boundary power and integrated energy balance.","","## Provenance and limits",""]
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
        for path,record in refined:
            steps=record['diagnostic_config']['rom_dts']
            for first,second in zip(steps[:-1],steps[1:]):
                cases=[r for r in record['adjacent_refinement'] if r['coarse_dt']==first and r['fine_dt']==second]
                complete=[r for r in cases if r['both_completed']]
                maxima=[max((r.get(key) for r in complete if r.get(key) is not None),default=None) for key in ('max_displacement_difference','max_velocity_difference','max_energy_state_difference')]
                adjacent_rows.append([record['name'],record['optimizer_seed'],number(first),number(second),len(complete),*[number(x,True) for x in maxima],sum(not r['passed'] for r in cases)])
            normalization_rows.append([record['name'],record['optimizer_seed'],record['provenance']['source_commit'],number(record['truth_parity']['scales_relative_parity']),str(record['truth_parity']['bit_identical_truth'])])
        lines[-2:]=[table(['Frozen head/objective','Repeat seed','Coarser step','Finer step','Both-step completions','Worst displacement difference','Worst velocity difference','Worst energy-state difference','Refinement failures'],adjacent_rows),'',table(['Frozen head/objective','Repeat seed','Continuation source','Regenerated-vs-original relative scale discrepancy','Bit-identical regenerated truth'],normalization_rows),'','The first continuation wrapper used regenerated normalization values after checking their agreement; its successor restores the original recorded values exactly after the same check. This bookkeeping distinction is retained with source hashes and measured scale discrepancies. Neither version changes the trained coefficients or initial latent position/velocity; the original trajectory verdict remains fixed.','', '## Provenance and limits','']
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
              "- **Primary step / refinement / completion:** the predetermined reported time step, comparison after reducing it, and successful finite integration through every required step.",
              "- **Phase / vanished mode / wall-strip peak:** oscillation angle, an amplitude too small to define that angle, and a boundary-neighborhood signal peak used only as a timing proxy.",
              "- **Boundary power / energy balance / absorbing mean:** instantaneous dissipative power, energy plus integrated power relative to its initial value, and average residual displacement that energy alone cannot control.",
              "- **Engineering target:** the declared provisional accuracy/completion requirement; satisfying it is specific to this bounded family and reference budget.",""]
    Path(args.out).write_text("\n".join(lines))


if __name__=="__main__":
    main()
