"""One objective-preserving, bounded initializer for the phase and query paths.

Version v1 changes linear algebra only. The original compiled fit is retained
as the first stage; retry/polishing uses its exact objective and acceptance
guard. No stationary start outside the original cost tie can be substituted.
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

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

POLICY = {
    "version": "v1-qr-svd-unchanged-acceptance",
    "baseline_budget": 250, "qr_budget": 40,
    "svd_iterations": 12, "svd_halvings": 8,
    "svd_cutoff": "max(J.shape) * float64_epsilon * largest_scaled_singular_value",
    "step_cap_relative": 1e-6, "tie_eps_multiplier": 1024,
    "stationarity": "max(abs(J.T @ residual)/(column_norms(J)*norm(residual)))",
    "endpoint_guard": "final_cost <= baseline_cost + 1024*eps*max(baseline_cost,1e-300)",
    "rank_truncation_does_not_certify_stationarity": True,
}
EPS = np.finfo(np.float64).eps
REASONS = {1: "stationarity", 2: "stagnation", 3: "nonfinite", 4: "budget"}


@jax.jit
def evaluate(z, target, params):
    return H.head_apply(params, z)-target, H.head_jacobian(params, z)


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


def snapshot(code, target, params):
    """Fresh direct residual and original normalized gradient at returned code."""
    residual, jacobian = evaluate(jnp.asarray(code), target, params)
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


def _stage(code, target, params, gtol, original_cost, stage):
    code = np.asarray(code).copy()
    res, jac, state = snapshot(code, target, params)
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
            tr, tj, trial = snapshot(candidate, target, params)
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
    _, _, final = snapshot(code, target, params)
    if not final["finite"]:
        reason = "nonfinite"
    elif final["gradient"] <= gtol:
        reason = "stationarity"
    final.update(reason=reason, history=history, iterations=len(history),
                 accepted=sum(x["accepted"] for x in history),
                 original_endpoint_guard=endpoint_pass(final["cost"], original_cost))
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
    repaired, baselines, qr_only, old_infos = [], [], [], []
    for seed in seeds:
        code, info = _baseline(jnp.asarray(seed), (target, params), jnp.float64(gtol))
        code, info = np.asarray(code), np.asarray(info)
        ores, ojac, old = snapshot(code, target, params)
        _,singular,cutoff,kept=svd_direction(ores,ojac)
        old['scaled_jacobian_spectrum']={'singular_values':np.asarray(singular),
            'cutoff':float(cutoff),'retained':np.asarray(kept),'retained_rank':int(np.sum(np.asarray(kept)))}
        old.update(reason="stationarity" if old["finite"] and old["gradient"] <= gtol else (
            REASONS[int(info[0])] if int(info[0]) != 1 else "fresh_stationarity_failure"),
            original_compiled_info=info, seed=seed.copy())
        old_infos.append({"cost": float(info[2]), "gradient": float(info[1]),
                          "reason": REASONS[int(info[0])]})
        qcode, qinfo = _stage(code, target, params, gtol, old["cost"], "qr")
        scode, sinfo = _stage(qcode, target, params, gtol, old["cost"], "svd")
        if not sinfo["original_endpoint_guard"]:
            raise RuntimeError("internal error: final initializer objective regressed")
        baselines.append(old); qr_only.append(qinfo); repaired.append(sinfo)
    selected, raw, band = select_fit_candidate(repaired)
    baseline_selected, baseline_raw, baseline_band = select_fit_candidate(baselines)
    original_selected, original_raw, original_band = select_fit_candidate(old_infos)
    qr_selected, qr_raw, qr_band = select_fit_candidate(qr_only)
    final_code = repaired[selected]["code"]
    _, _, direct = snapshot(final_code, target, params)
    global_quality = endpoint_pass(direct["cost"], baselines[baseline_raw]["cost"])
    qualified = bool(direct["finite"] and direct["gradient"] <= gtol and global_quality)
    gap = float(np.linalg.norm(np.asarray(H.head_apply(params, final_code))-
                               np.asarray(H.head_apply(params, repaired[raw]["code"]))))
    return final_code, {"best_index": selected, "raw_minimum_cost_index": raw,
        "selection_tie_band": band, "selected_stationary_tie": selected != raw,
        "all_final_codes": [x["code"] for x in repaired], "starts": repaired,
        "coefficient_gap_selected_vs_raw": gap,
        "raw_minimum_found_distance": float(np.sqrt(2*repaired[raw]["cost"])),
        "best_found_fixed_frame_distance": float(np.sqrt(2*direct["cost"])),
        "seed_indices": seed_indices, "seed_codes": seeds,
        "baseline_starts": baselines, "baseline_selected": baseline_selected,
        "baseline_raw": baseline_raw, "baseline_band": baseline_band,
        "original_compiled_selected": original_selected, "original_compiled_raw": original_raw,
        "original_compiled_band": original_band, "original_compiled_infos": old_infos,
        "qr_only_starts": qr_only, "qr_only_selected": qr_selected,
        "qr_only_raw": qr_raw, "qr_only_band": qr_band,
        "direct_returned_check": direct, "global_original_objective_quality": global_quality,
        "qualified": qualified, "gtol": gtol, "policy": POLICY}
