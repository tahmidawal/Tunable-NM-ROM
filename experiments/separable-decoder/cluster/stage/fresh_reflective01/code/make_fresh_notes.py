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
    ap.add_argument("--out",required=True)
    ap.add_argument("--reviewed",action="store_true")
    args=ap.parse_args()
    loaded=[(Path(p),json.loads(Path(p).read_text())) for p in args.result]
    boundaries={}
    for path,run in loaded:
        if run.get("final_test_opened") or run.get("provenance",{}).get("jax_backend")!="gpu":
            raise RuntimeError("Unexpected cohort/backend provenance")
        for bc,value in run["boundary_results"].items():
            if bc in boundaries:
                raise RuntimeError("Duplicate BC; select one declared campaign invocation per BC")
            boundaries[bc]=(path,run,value)
    status="independently reviewed" if args.reviewed else "provisional pending independent result review"
    lines=["# Fresh waves: learned spatial bank, coefficient heads and actual evolution",f"This report covers newly verified absorbing and fixed-wall scalar waves, learned-bank reconstruction and actual reduced trajectories. The numerical results below are {status}; all earlier wave experiments are excluded from this evidence.","", "## Scope and reference",""]
    for bc,(path,run,entry) in boundaries.items():
        config=run["config"]
        ref=config["reference_evidence"]
        lines.append(f"The `{bc}` run uses {config['n']} intervals per axis, {config['rank']} learned bank functions, {config['latent']} latent coordinates, {config['train_count']} training trajectories and {config['validation_count']} validation trajectories through time {number(config['end_time'])}. Its final cohort remains closed. Reference job `{ref['job_id']}` and source `{ref['source_commit']}` precede training; reference uncertainty and the absorber's boundary-model error remain distinct from reduced-model error.")
    lines += ["", "All tabulated relative errors use the initial displacement mass norm or initial state energy as fixed trajectory scales. A time-maximum is the largest stored-time error; error-state energy is not the difference between two solution energies. The first-order absorbing condition has physical/model reflection at oblique incidence. Every latent decoder is parameter-free: no wave-family descriptors or time enter the head.","","## Learned bank and fresh linear baselines",""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        bank=entry["bank"]
        for split,floor in bank["floors"].items():
            rows.append([bc,split,number(floor["displacement"]["mean"],True),number(floor["displacement"]["median"],True),number(floor["displacement"]["worst"],True),number(floor["velocity"]["mean"],True),number(bank["raw_rank_ratio"])])
    lines += [table(["Boundary","Split","Bank displacement mean","Median","Worst","Bank velocity mean","Raw bank rank ratio"],rows),"","Bank floors are unrestricted projections onto the trained neural span. QR only changes that span's coordinates. A fresh coefficient PCA initializes each head's affine map; it does not replace the learned spatial network with a POD bank.",""]
    rows=[]
    for bc,(_,run,entry) in boundaries.items():
        for baseline in entry["linear_baselines"]:
            rows.append([bc,baseline["label"],baseline["rank"],number(baseline["displacement"]["mean"],True),number(baseline["velocity"]["mean"],True),number(baseline["energy_state"]["mean"],True),number(baseline["energy_state"]["worst"],True),baseline["energy_state"]["outliers"]])
    lines += [table(["Boundary","Fresh baseline","Dimension","Max-over-time displacement mean","Velocity mean","Energy-state mean","Energy-state worst","Energy outliers"],rows),"","Randomized POD is an explicitly approximate linear comparator trained from the fresh training data. Its linear dynamics and the unrestricted learned-bank dynamics use independent matrix-exponential propagation.","","## Unseen-state representation",""]
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
            rows.append([bc,arm["name"],arm["optimizer_seed"],number(primary["dt"]),primary["completed"],primary["failed"],number(primary["displacement"]["mean"],True),number(primary["velocity"]["mean"],True),number(primary["energy_state"]["mean"],True),number(primary["energy_state"]["median"],True),number(primary["energy_state"]["worst"],True),primary["energy_state"]["outliers"],str(arm["accuracy_passed"])])
    lines += [table(["Boundary","Head/objective","Repeat seed","Primary step","Complete","Failed","Displacement mean","Velocity mean","Energy-state mean","Median","Worst","Energy outliers","Engineering target passed"],rows),"","These values summarize each trajectory's maximum error over time. Means, medians and worst values are conditional on finite cases; failed/nonfinite trajectories count as outliers and prevent acceptance. The primary step was fixed before training. Physical velocity is the decoder Jacobian applied to the latent velocity; no finite-difference replacement or phase alignment is used.","","## Time-step refinement and phase diagnostics",""]
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
    for path,run in loaded:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"- Source result: `{path}`; SHA-256 `{digest}`; GPU job `{run['provenance'].get('job_id')}`; source commit `{run['provenance'].get('source_commit')}`; devices `{run['provenance'].get('device_kind')}`.")
    lines += ["","The scope is the declared smooth Gaussian-core compact family in two dimensions. Head weights and learned banks are fresh per PDE/boundary configuration. Initial fitting uses full-field initial-state projections, so this first accuracy campaign makes no grid-independent cold-start or speed claim. Results do not establish three-dimensional wave transfer or performance outside this family.","","## Glossary", "",
              "- **Boundary / fixed wall / absorber:** the physical edge condition; zero wall displacement produces sign-reversing reflection, while the local radiation condition approximates outgoing waves.",
              "- **Bank / head / latent coordinates:** spatial neural features, the coefficient function multiplying them, and its internal coordinates.",
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
