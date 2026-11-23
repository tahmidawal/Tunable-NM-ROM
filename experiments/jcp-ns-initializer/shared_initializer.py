"""One objective-preserving, bounded initializer for the phase and query paths.

Version v2 evaluates the same objective using one fixed local anchor. Both
original implementations remain controls, and both ordinary-direct and
anchored stationarity must pass. No residual recurrence or gate relaxation.
Host orchestration makes every proposed polishing step inspectable. All model
evaluations and linear solves use f64 GPU kernels; this is not a cost claim.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import head_rom as H
import legacy_query as L
import legacy_phase as P

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

POLICY = {
    "version": "v2-fixed-anchor-qr-svd",
    "baseline_budget": 250, "qr_budget": 40,
    "svd_iterations": 12, "svd_halvings": 8,
    "svd_cutoff": "max(J.shape) * float64_epsilon * largest_scaled_singular_value",
    "step_cap_relative": 1e-6, "tie_eps_multiplier": 1024,
    "stationarity": "max(abs(J.T @ residual)/(column_norms(J)*norm(residual)))",
    "endpoint_guard": "final_cost <= baseline_cost + 1024*eps*max(baseline_cost,1e-300)",
    "rank_truncation_does_not_certify_stationarity": True,
    "controls_per_start": ["original_host_phase", "legacy_compiled_core"],
    "anchor": "minimum fresh ordinary-direct objective over controls; seed order then phase/core; fixed per fit and tolerance",
    "stable_increment_local_preactivation_limit": .5,
    "stable_increment_outside_local_range": "ordinary SiLU difference; no overflowing expm1 branch",
    "direct_and_anchored_stationarity_required": True,
    "optional_svd_rollback": False,
}
EPS = np.finfo(np.float64).eps
REASONS = {1: "stationarity", 2: "stagnation", 3: "nonfinite", 4: "budget"}


@jax.jit
def evaluate(z, target, params):
    return H.head_apply(params, z)-target, H.head_jacobian(params, z)


def silu_increment(a, d):
    """Exact real identity; bounded expm1 avoids saturation/overflow hazards."""
    local=jnp.abs(d)<=POLICY['stable_increment_local_preactivation_limit']
    bounded=jnp.where(local,d,0.)
    s=jax.nn.sigmoid(a)
    t=jnp.expm1(bounded)
    ds=s*(1-s)*t/(1+s*t)
    increment=bounded*jax.nn.sigmoid(a+bounded)+a*ds
    return jnp.where(local,increment,jax.nn.silu(a+d)-jax.nn.silu(a))


@jax.jit
def head_increment(params, anchor, dz):
    a1=anchor@params['W1']+params['b1']
    d1=dz@params['W1']
    dx1=silu_increment(a1,d1)
    a2=jax.nn.silu(a1)@params['W2']+params['b2']
    dx2=silu_increment(a2,dx1@params['W2'])
    return dx2@params['W3']+dz@params['Ws']


@jax.jit
def evaluate_anchored(z,target,params,anchor):
    residual=H.head_apply(params,anchor)-target+head_increment(params,anchor,z-anchor)
    return residual,H.head_jacobian(params,z)


@jax.jit
def statistics(residual, jacobian):
    return jnp.asarray([.5*jnp.dot(residual, residual),
                        L.stationarity(residual, jacobian),
                        jnp.linalg.norm(residual)])


@jax.jit
def qr_direction(residual, jacobian, damping):
    scales = jnp.maximum(jnp.linalg.norm(jacobian, axis=0), 1e-30)
    normalized = jacobian/scales
    augmented = jnp.concatenate((normalized, jnp.sqrt(damping)*jnp.eye(jacobian.shape[1])))
    rhs = jnp.concatenate((-residual, jnp.zeros(jacobian.shape[1])))
    q, r = jnp.linalg.qr(augmented, mode="reduced")
    return jsl.solve_triangular(r, q.T@rhs, lower=False)/scales


@jax.jit
def svd_direction(residual, jacobian):
    scales = jnp.maximum(jnp.linalg.norm(jacobian, axis=0), 1e-30)
    normalized = jacobian/scales
    u, singular, vh = jnp.linalg.svd(normalized, full_matrices=False)
    cutoff = max(normalized.shape)*jnp.finfo(jnp.float64).eps*singular[0]
    kept = singular > cutoff
    inverse = jnp.where(kept, 1/jnp.maximum(singular, 1e-300), 0.)
    step = -(vh.T@(inverse*(u.T@residual)))/scales
    return step, singular, cutoff, kept


@jax.jit
def original_gpu_seed_indices(head_candidates, target):
    distances=jnp.sum((head_candidates-target[None])**2,axis=1)
    return jnp.argsort(distances,stable=True)[:4]


_baseline = jax.jit(L.make_lm(evaluate, POLICY["baseline_budget"], True))


def snapshot(code, target, params, anchor=None):
    """Fresh direct residual and original normalized gradient at returned code."""
    residual, jacobian = (evaluate(jnp.asarray(code),target,params) if anchor is None else
                          evaluate_anchored(jnp.asarray(code),target,params,jnp.asarray(anchor)))
    cost, gradient, residual_norm = np.asarray(statistics(residual, jacobian))
    finite = bool(np.isfinite(cost+gradient) and np.all(np.isfinite(np.asarray(jacobian))))
    return residual, jacobian, {"cost": float(cost), "gradient": float(gradient),
        "residual_norm": float(residual_norm), "finite": finite,
        "code": np.asarray(code).copy()}


def endpoint_pass(cost, original_cost):
    return bool(cost <= original_cost+1024*EPS*max(original_cost, 1e-300))


def acceptance(current, trial, step_norm, code_norm, original_cost):
    band = 1024*EPS*max(current["cost"], trial["cost"], 1e-300)
    strict = trial["finite"] and trial["cost"] < current["cost"]
    repair = bool(trial["finite"] and current["gradient"] <= 1e-5
        and trial["cost"] <= current["cost"]+band
        and trial["gradient"] <= .5*current["gradient"]
        and step_norm <= 1e-6*(1+code_norm))
    guard = endpoint_pass(trial["cost"], original_cost)
    accepted = bool((strict or repair) and guard)
    reason = ("strict_decrease" if strict else "roundoff_gradient_improvement") if accepted else (
        "original_endpoint_objective_guard" if (strict or repair) and not guard else "rejected")
    return accepted, reason, band, guard


def select_fit_candidate(infos):
    """The exact old selector: stationarity is preferred only inside its old tie."""
    costs = np.asarray([i["cost"] for i in infos])
    raw = int(np.argmin(costs))
    band = 1024*EPS*max(float(costs[raw]), 1e-300)
    selected = raw
    if infos[raw]["reason"] != "stationarity":
        eligible = [i for i, x in enumerate(infos)
                    if x["reason"] == "stationarity" and costs[i] <= costs[raw]+band]
        if eligible:
            selected = min(eligible, key=lambda i: costs[i])
    return selected, raw, band


def _stage(code, target, params, gtol, original_cost, stage, anchor=None):
    code = np.asarray(code).copy()
    res, jac, state = snapshot(code, target, params,anchor)
    history = []
    reason = "budget"
    damping = 1e-4
    budget = POLICY["qr_budget"] if stage == "qr" else POLICY["svd_iterations"]
    for iteration in range(budget):
        if not state["finite"]:
            reason = "nonfinite"; break
        if state["gradient"] <= gtol:
            reason = "stationarity"; break
        if stage == "qr":
            direction = np.asarray(qr_direction(res, jac, jnp.float64(damping)))
            spectra = None
            halvings = 1
        else:
            direction, singular, cutoff, kept = svd_direction(res, jac)
            direction = np.array(direction,copy=True)
            spectra = {"scaled_singular_values": np.asarray(singular),
                       "cutoff": float(cutoff), "retained": np.asarray(kept),
                       "retained_rank": int(np.sum(np.asarray(kept)))}
            norm = np.linalg.norm(direction)
            cap = POLICY["step_cap_relative"]*(1+np.linalg.norm(code))
            direction *= min(1., cap/max(norm, 1e-300))
            halvings = POLICY["svd_halvings"]
        accepted = False
        for halving in range(halvings):
            step = direction/(2**halving)
            candidate = code+step
            tr, tj, trial = snapshot(candidate, target, params,anchor)
            step_norm = float(np.linalg.norm(step))
            accept, why, band, guard = acceptance(state, trial, step_norm,
                                                 float(np.linalg.norm(code)), original_cost)
            history.append({"iteration": iteration, "halving": halving,
                "cost": state["cost"], "gradient": state["gradient"],
                "trial_cost": trial["cost"], "trial_gradient": trial["gradient"],
                "step_norm": step_norm, "damping": damping if stage == "qr" else None,
                "accepted": accept, "acceptance_reason": why,
                "roundoff_band": band, "original_endpoint_guard": guard,
                "svd": spectra, "trial_code": candidate.copy()})
            if accept:
                code, res, jac, state = candidate, tr, tj, trial
                accepted = True
                break
        if stage == "qr":
            damping = max(damping/3, 1e-14) if accepted else min(damping*10, 1e18)
            if step_norm <= 1e-15*(1+np.linalg.norm(code)) or damping >= 1e18:
                reason = "stagnation"; break
        elif not accepted:
            reason = "stagnation"; break
    # A stage label is never inherited as a convergence certificate.
    _, _, final = snapshot(code, target, params,anchor)
    _, _, direct = snapshot(code,target,params)
    if not final["finite"]:
        reason = "nonfinite"
    elif final["gradient"] <= gtol:
        reason = "stationarity" if direct['finite'] and direct['gradient']<=gtol else 'direct_stationarity_failure'
    final.update(reason=reason, history=history, iterations=len(history),
                 accepted=sum(x["accepted"] for x in history),
                 original_endpoint_guard=endpoint_pass(final["cost"], original_cost),
                 direct_check=direct)
    return code, final


def fit_head(target, params, codes, head_candidates, starts=4, extra=None,
             gtol=1e-9, budget=250):
    if budget != POLICY["baseline_budget"] or starts != 4:
        raise ValueError("the registered primary initializer uses four starts and budget 250")
    target = jnp.asarray(target)
    params = {k: jnp.asarray(v) for k, v in params.items()}
    distances = np.sum((np.asarray(head_candidates)-np.asarray(target)[None])**2, axis=1)
    seed_indices = np.argsort(distances, kind="stable")[:starts]
    gpu_indices=np.asarray(original_gpu_seed_indices(jnp.asarray(head_candidates),target))
    phase_indices=np.argsort(distances)[:starts]
    if not np.array_equal(seed_indices,gpu_indices) or not np.array_equal(seed_indices,phase_indices):
        raise RuntimeError('four nearest training-code identities differ between original paths')
    seeds = [np.asarray(codes[i]) for i in seed_indices]
    if extra is not None:
        seeds.extend(np.atleast_2d(extra))
    controls=[]; old_infos=[]
    for si,seed in enumerate(seeds):
        phase_code,phase_info=P.solve(P.fit_eval,seed,(target,params),gtol,budget)
        core_code,core_info=_baseline(jnp.asarray(seed),(target,params),jnp.float64(gtol))
        core_code,core_info=np.asarray(core_code),np.asarray(core_info)
        old_infos.append({'cost':float(core_info[2]),'gradient':float(core_info[1]),
                          'reason':REASONS[int(core_info[0])]})
        for kind,code,original in [('phase',phase_code,phase_info),('core',core_code,core_info)]:
            residual,jac,direct=snapshot(code,target,params)
            controls.append({'seed_index':si,'kind':kind,'code':np.asarray(code).copy(),
                'direct':direct,'direct_residual':np.asarray(residual),
                'direct_jacobian':np.asarray(jac),'original_info':original})
    # The anchor is deterministic, per-fit/tolerance, and is never changed again.
    anchor_index=int(np.argmin([c['direct']['cost'] for c in controls]))
    anchor=controls[anchor_index]['code'].copy()
    for c in controls:
        res,jac,info=snapshot(c['code'],target,params,anchor)
        info['reason']='stationarity' if info['finite'] and info['gradient']<=gtol and c['direct']['gradient']<=gtol else 'stagnation'
        c.update(anchored=info,anchored_residual=np.asarray(res),anchored_jacobian=np.asarray(jac))
    global_original=min(c['anchored']['cost'] for c in controls)
    repaired=[];baselines=[];qr_only=[];chosen_controls=[]
    for si in range(len(seeds)):
        pair=controls[2*si:2*si+2]
        pick,_,_=select_fit_candidate([c['anchored'] for c in pair])
        original_cost=min(c['anchored']['cost'] for c in pair)
        selected_control=pair[pick]
        chosen_controls.append(2*si+pick)
        code=selected_control['code']
        ores,ojac,old=snapshot(code,target,params,anchor)
        _,singular,cutoff,kept=svd_direction(ores,ojac)
        old.update(reason=selected_control['anchored']['reason'],
            seed=seeds[si].copy(),control_index=2*si+pick,
            scaled_jacobian_spectrum={'singular_values':np.asarray(singular),
                'cutoff':float(cutoff),'retained':np.asarray(kept),
                'retained_rank':int(np.sum(np.asarray(kept)))})
        qcode,qinfo=_stage(code,target,params,gtol,original_cost,'qr',anchor)
        scode,sinfo=_stage(qcode,target,params,gtol,original_cost,'svd',anchor)
        if not sinfo['original_endpoint_guard']:
            raise RuntimeError('internal error: final anchored objective regressed')
        baselines.append(old);qr_only.append(qinfo);repaired.append(sinfo)
    selected,raw,band=select_fit_candidate(repaired)
    baseline_selected,baseline_raw,baseline_band=select_fit_candidate(baselines)
    original_selected,original_raw,original_band=select_fit_candidate(old_infos)
    qr_selected,qr_raw,qr_band=select_fit_candidate(qr_only)
    final_code=repaired[selected]['code']
    _,_,direct=snapshot(final_code,target,params)
    _,_,anchored=snapshot(final_code,target,params,anchor)
    global_quality=endpoint_pass(anchored['cost'],global_original)
    qualified=bool(direct['finite'] and anchored['finite'] and direct['gradient']<=gtol
                   and anchored['gradient']<=gtol and global_quality)
    gap=float(np.linalg.norm(np.asarray(H.head_apply(params,final_code))-
                             np.asarray(H.head_apply(params,repaired[raw]['code']))))
    return final_code,{'best_index':selected,'raw_minimum_cost_index':raw,
        'selection_tie_band':band,'selected_stationary_tie':selected!=raw,
        'all_final_codes':[x['code'] for x in repaired],'starts':repaired,
        'coefficient_gap_selected_vs_raw':gap,
        'raw_minimum_found_distance':float(np.sqrt(2*repaired[raw]['cost'])),
        'best_found_fixed_frame_distance':float(np.sqrt(2*anchored['cost'])),
        'seed_indices':seed_indices,'seed_codes':seeds,
        'baseline_starts':baselines,'baseline_selected':baseline_selected,
        'baseline_raw':baseline_raw,'baseline_band':baseline_band,
        'original_compiled_selected':original_selected,'original_compiled_raw':original_raw,
        'original_compiled_band':original_band,'original_compiled_infos':old_infos,
        'qr_only_starts':qr_only,'qr_only_selected':qr_selected,
        'qr_only_raw':qr_raw,'qr_only_band':qr_band,
        'direct_returned_check':direct,'anchored_returned_check':anchored,
        'global_original_objective_quality':global_quality,
        'global_original_anchored_minimum':global_original,
        'direct_original_objective_guard':endpoint_pass(direct['cost'],min(c['direct']['cost'] for c in controls)),
        'anchor_index':anchor_index,'anchor_code':anchor,'controls':controls,
        'chosen_controls':chosen_controls,'qualified':qualified,'gtol':gtol,'policy':POLICY}
