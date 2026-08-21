#!/usr/bin/env python
"""Independent negative-aware audit for Phase-10-D."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

import b10_audit_phase9_train as p9audit
import b10_common as c
import b10_phase10_d as p10
import b10_phase9_train as p9
import b10_phase9_terminal_recovery as recovery
import b10_spline_train as legacy

PORTABILITY_RTOL = 2e-13
PORTABILITY_ATOL = 2e-14
FLOAT_METRICS = ("trajectory_error_mean", "trajectory_error_worst",
                 "mean_snapshot_relative_l2_squared", "k3_cox_identity_worst")
FIELD_ARRAY_SUFFIXES = ("numerator", "denominator", "trajectory", "boundary_count", "identity")


def load_json(path):
    with open(path,encoding="utf-8") as handle: return json.load(handle)


def manifest(path):
    rows={}
    with open(path,encoding="utf-8") as handle:
        for line in handle:
            digest,name=line.rstrip().split("  ",1)
            if len(digest)!=64 or not name.startswith("./"): raise SystemExit("bad manifest row")
            rows[name[2:]]=digest
    return rows


def tree_count(tree):
    return int(sum(np.prod(np.asarray(x).shape) for x in jax.tree_util.tree_leaves(tree)))


def exact_arrays(left,right,prefix,meshes):
    return bool(all(np.array_equal(np.asarray(left[f"{prefix}_N{n}_{suffix}"]),
                                   np.asarray(right[f"{prefix}_N{n}_{suffix}"]))
                    for n in meshes for suffix in ("numerator","denominator","trajectory","boundary_count","identity")))


def objectives(arrays,prefix,train):
    return np.concatenate([np.asarray(arrays[f"{prefix}_N{x['N']}_numerator"])/
        np.maximum(np.asarray(arrays[f"{prefix}_N{x['N']}_denominator"]),1e-300) for x in train])


def independent_initial_binding(actual_metrics, reference_metrics, fresh_arrays, reference_arrays, meshes):
    """Independent audit implementation of the prospective f64 portability contract."""
    details={"rtol":PORTABILITY_RTOL,"atol":PORTABILITY_ATOL,
             "metric_max_abs":{},"metric_exact":{},"array_max_abs":{},"array_exact":{}}
    passed=set(actual_metrics)=={"meshes","pooled"}==set(reference_metrics)
    actual_meshes=actual_metrics.get("meshes",{}); reference_meshes=reference_metrics.get("meshes",{})
    passed &= set(actual_meshes)==set(reference_meshes)==set(str(n) for n in meshes)
    scopes=[("pooled",actual_metrics.get("pooled",{}),reference_metrics.get("pooled",{}))]
    scopes += [(f"N{n}",actual_meshes.get(str(n),{}),reference_meshes.get(str(n),{})) for n in meshes]
    required=set(FLOAT_METRICS)|{"all_finite","boundary_violation_count"}
    for label,left,right in scopes:
        passed &= set(left)==required==set(right)
        passed &= type(left.get("all_finite")) is bool and type(right.get("all_finite")) is bool
        passed &= left.get("all_finite")==right.get("all_finite") is True
        passed &= type(left.get("boundary_violation_count")) is int
        passed &= type(right.get("boundary_violation_count")) is int
        passed &= left.get("boundary_violation_count")==right.get("boundary_violation_count")==0
        for name in FLOAT_METRICS:
            one=left.get(name); two=right.get(name)
            typed=type(one) is float and type(two) is float
            finite=typed and np.isfinite(one) and np.isfinite(two)
            delta=float(abs(one-two)) if finite else float("inf")
            passed &= bool(finite and np.isclose(one,two,rtol=PORTABILITY_RTOL,atol=PORTABILITY_ATOL))
            details["metric_max_abs"][f"{label}.{name}"]=delta
            details["metric_exact"][f"{label}.{name}"]=bool(typed and one==two)
    for n in meshes:
        for suffix in FIELD_ARRAY_SUFFIXES:
            left=np.asarray(fresh_arrays[f"initial_train_N{n}_{suffix}"])
            right=np.asarray(reference_arrays[f"terminal_train_N{n}_{suffix}"])
            same_shape=left.shape==right.shape
            if suffix=="boundary_count":
                valid=bool(same_shape and np.issubdtype(left.dtype,np.integer)
                           and np.issubdtype(right.dtype,np.integer)
                           and np.array_equal(left,right) and np.all(left==0))
                delta=float(np.max(np.abs(left.astype(np.int64)-right.astype(np.int64)))) if same_shape and left.size else 0.0
            else:
                finite=bool(same_shape and np.issubdtype(left.dtype,np.floating)
                            and np.issubdtype(right.dtype,np.floating)
                            and np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
                valid=bool(finite and np.allclose(left,right,rtol=PORTABILITY_RTOL,atol=PORTABILITY_ATOL))
                delta=float(np.max(np.abs(left-right))) if finite and left.size else (0.0 if finite else float("inf"))
            label=f"N{n}.{suffix}"
            details["array_max_abs"][label]=delta
            details["array_exact"][label]=bool(same_shape and np.array_equal(left,right))
            passed &= valid
    details["metrics_max_abs"]=float(max(details["metric_max_abs"].values(),default=0.0))
    details["arrays_max_abs"]=float(max(details["array_max_abs"].values(),default=0.0))
    details["metrics_bitwise_exact"]=bool(all(details["metric_exact"].values()))
    details["arrays_bitwise_exact"]=bool(all(details["array_exact"].values()))
    details["pass"]=bool(passed)
    return details


def portability_negative_tests():
    row={"trajectory_error_mean":float(.01),"trajectory_error_worst":float(.02),
         "mean_snapshot_relative_l2_squared":float(1e-4),"k3_cox_identity_worst":float(2e-15),
         "all_finite":True,"boundary_violation_count":0}
    metrics={"meshes":{"16":copy.deepcopy(row)},"pooled":copy.deepcopy(row)}
    fresh={f"initial_train_N16_{name}":np.asarray(
        [0,0] if name=="boundary_count" else [0.,1.],
        np.int32 if name=="boundary_count" else np.float64)
        for name in FIELD_ARRAY_SUFFIXES}
    reference={key.replace("initial_train_","terminal_train_",1):value.copy() for key,value in fresh.items()}
    exact=independent_initial_binding(metrics,metrics,fresh,reference,[16])["pass"]
    portable_metrics=copy.deepcopy(metrics); portable_metrics["meshes"]["16"]["trajectory_error_mean"]+=1e-15
    portable=independent_initial_binding(portable_metrics,metrics,fresh,reference,[16])["pass"]
    corruptions=[]
    one=copy.deepcopy(metrics); one["pooled"]["trajectory_error_mean"]+=1e-8
    corruptions.append(not independent_initial_binding(one,metrics,fresh,reference,[16])["pass"])
    one=copy.deepcopy(metrics); one["meshes"]["16"]["all_finite"]=False
    corruptions.append(not independent_initial_binding(one,metrics,fresh,reference,[16])["pass"])
    one={key:value.copy() for key,value in fresh.items()}; one["initial_train_N16_boundary_count"][0]=1
    corruptions.append(not independent_initial_binding(metrics,metrics,one,reference,[16])["pass"])
    one={key:value.copy() for key,value in fresh.items()}; one["initial_train_N16_numerator"]=one["initial_train_N16_numerator"][:1]
    corruptions.append(not independent_initial_binding(metrics,metrics,one,reference,[16])["pass"])
    one={key:value.copy() for key,value in fresh.items()}; one["initial_train_N16_identity"][0]=np.nan
    corruptions.append(not independent_initial_binding(metrics,metrics,one,reference,[16])["pass"])
    return {"exact_positive":bool(exact),"portable_positive":bool(portable),
            "corruption_count":len(corruptions),"corruptions_rejected":bool(all(corruptions)),
            "pass":bool(exact and portable and all(corruptions))}


def trace_check(arrays):
    attempted=np.asarray(arrays["trust_attempted"],bool); pred=np.asarray(arrays["trust_predicted"])
    actual=np.asarray(arrays["trust_actual"]); rho=np.asarray(arrays["trust_rho"])
    defined=np.asarray(arrays["trust_rho_defined"],bool); accepted=np.asarray(arrays["trust_accepted"],bool)
    finite=np.asarray(arrays["trust_finite"],bool); breakdown=np.asarray(arrays["trust_cg_breakdown"],bool)
    q=np.asarray(arrays["trust_q"]); objective=np.asarray(arrays["trust_objective"])
    trial=np.asarray(arrays["trust_trial_objective"]); step=np.asarray(arrays["trust_step"])
    terminated=np.asarray(arrays["trust_terminated"],bool); delta=np.asarray(arrays["trust_delta"])
    damping=np.asarray(arrays["trust_damping"]); active=np.asarray(arrays["trust_active"],bool)
    iters=np.asarray(arrays["trust_cg_iterations"]); jvp=np.asarray(arrays["trust_jvp_count"])
    vjp=np.asarray(arrays["trust_vjp_count"]); positive=attempted&(pred>0); nonpositive=attempted&(pred<=0)
    bound=np.asarray(arrays["trust_bound_active_count"])
    expected_accept=attempted&finite&~breakdown&defined&(actual>0)&(rho>=p9.ACCEPT_RHO)
    expected_terminate=expected_accept&(((objective[:,:,:-1]-trial)/np.maximum(objective[:,:,:-1],1e-300)<=1e-12)
        | (np.linalg.norm(step,axis=-1)/(1+np.linalg.norm(q[:,:,:-1],axis=-1))<=1e-12))
    shrink=(~accepted)|(defined&(rho<.25)); expand=accepted&(rho>.75)&(np.linalg.norm(step,axis=-1)>=.9*delta[:,:,:-1]); improve=accepted&(rho>.75)
    update_delta=np.where(shrink,np.maximum(delta[:,:,:-1]/4,p9.DELTA_MIN),np.where(expand,np.minimum(2*delta[:,:,:-1],p9.DELTA_MAX),delta[:,:,:-1]))
    update_damping=np.where(shrink,np.minimum(10*damping[:,:,:-1],p9.LAMBDA_MAX),np.where(improve,np.maximum(damping[:,:,:-1]/3,p9.LAMBDA_MIN),damping[:,:,:-1]))
    expected_delta=np.where(attempted,update_delta,delta[:,:,:-1]); expected_damping=np.where(attempted,update_damping,damping[:,:,:-1])
    expected_q=np.where(accepted[...,None],q[:,:,:-1]+step,q[:,:,:-1]); expected_objective=np.where(accepted,trial,objective[:,:,:-1])
    exhaustion=active[:,:,-1]&(delta[:,:,-1]<=p9.DELTA_MIN)&(damping[:,:,-1]>=p9.LAMBDA_MAX)
    checks={"one_start_shape":q.shape[0]==1,"attempt_work_finite":bool(np.all(finite[attempted])
            and np.all(np.isfinite(pred[attempted])) and np.all(np.isfinite(actual[attempted])) and np.all(np.isfinite(rho[attempted]))),
        "rho_definition":bool(np.array_equal(defined[attempted],pred[attempted]>0)),
        "positive_rho_exact":bool(p9audit.close(rho[positive],actual[positive]/pred[positive])) if np.any(positive) else True,
        "nonpositive_sentinel":bool(np.all(rho[nonpositive]==0)&~np.any(accepted[nonpositive])),
        "no_breakdown":bool(~np.any(breakdown[attempted])),"attempt_active_exact":bool(np.array_equal(attempted,active[:,:,:-1])),
        "acceptance_exact":bool(np.array_equal(accepted,expected_accept)),"termination_exact":bool(np.array_equal(terminated,expected_terminate)),
        "q_transition":bool(p9audit.close(q[:,:,1:],expected_q)),"objective_transition":bool(p9audit.close(objective[:,:,1:],expected_objective)),
        "delta_transition":bool(p9audit.close(delta[:,:,1:],expected_delta)),"damping_transition":bool(p9audit.close(damping[:,:,1:],expected_damping)),
        "active_transition":bool(np.array_equal(active[:,:,1:],active[:,:,:-1]&~terminated)),
        "work_exact":bool(np.array_equal(jvp,np.where(attempted,iters+1,0)) and np.array_equal(vjp,np.where(attempted,iters+1,0))
            and np.all(iters[attempted]>=0) and np.all(iters[attempted]<=19)),
        "inactive_no_work":bool(np.all(jvp[~attempted]==0)&np.all(vjp[~attempted]==0)),
        "bound_count_exact":bool(np.array_equal(bound,np.sum(np.abs(q[:,:,1:])>=1.0,axis=-1))),
        "terminal_exhaustion_exact":bool(np.array_equal(exhaustion,np.asarray(arrays["trust_unhealthy_exhaustion"],bool))),
        "no_unhealthy_exhaustion":bool(not np.any(exhaustion)),
        "sole_choice_binding":bool(np.all(np.asarray(arrays["trust_chosen_start"])==0)
            and p9audit.close(np.asarray(arrays["trust_chosen_terminal_q"]),q[0,:,-1]))}
    checks["pass"]=bool(all(checks.values())); return checks


def decision_contract(value):
    return bool(value["architecture_increase_licensed"] is False and value["g2_licensed"] is False
        and value["selection_evaluated"] is False and value["optimizer_updates"]==0
        and value["scientific_promotion_allowed"] is False
        and value["corrected_g1_proposal_eligible"]==value["fixed_g1_train_representable"])


def negative_tests(decision,arrays):
    decision_cases=[]
    for key,value in (("architecture_increase_licensed",True),("g2_licensed",True),
                      ("selection_evaluated",True),("optimizer_updates",1),
                      ("scientific_promotion_allowed",True),
                      ("corrected_g1_proposal_eligible",not decision["fixed_g1_train_representable"])):
        one=copy.deepcopy(decision); one[key]=value; decision_cases.append(not decision_contract(one))
    trace_cases=[]
    for key in ("trust_rho_defined","trust_q","trust_jvp_count","trust_bound_active_count",
                "trust_unhealthy_exhaustion","trust_chosen_start"):
        one={name:np.asarray(value).copy() for name,value in arrays.items()}
        if key=="trust_rho_defined": one[key].flat[0]=~one[key].flat[0]
        elif key=="trust_q": one[key][0,0,1,0]+=1e-5
        elif key=="trust_jvp_count": one[key].flat[0]+=1
        elif key=="trust_bound_active_count": one[key].flat[0]+=1
        elif key=="trust_unhealthy_exhaustion": one[key].flat[0]=~one[key].flat[0]
        else: one[key].flat[0]=1
        trace_cases.append(not trace_check(one)["pass"])
    return {"positive_decision_contract":decision_contract(decision),
        "decision_corruption_count":len(decision_cases),"decision_corruptions_rejected":bool(all(decision_cases)),
        "trace_corruption_count":len(trace_cases),"trace_corruptions_rejected":bool(all(trace_cases)),
        "pass":bool(decision_contract(decision) and all(decision_cases) and all(trace_cases))}


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-json",required=True); parser.add_argument("--source-npz",required=True)
    parser.add_argument("--initial-control-npz",required=True)
    parser.add_argument("--work-checkpoint",required=True); parser.add_argument("--manifest",required=True)
    parser.add_argument("--prereg",required=True); parser.add_argument("--recovery-dir")
    parser.add_argument("--accepted-audit-dir"); parser.add_argument("--failed-t1-dir")
    parser.add_argument("--r1-failed-dir")
    parser.add_argument("--p5-json"); parser.add_argument("--p7-npz"); parser.add_argument("--target-dir")
    parser.add_argument("--expected-commit",required=True); parser.add_argument("--expected-job",required=True)
    parser.add_argument("--slurm-out",required=True); parser.add_argument("--slurm-err",required=True)
    parser.add_argument("--output",required=True); parser.add_argument("--smoke",action="store_true")
    return parser.parse_args()


def main():
    args=parse_args(); c.require_gpu_highest(); report=load_json(args.source_json)
    source=np.load(args.source_npz,allow_pickle=False); arrays={name:source[name] for name in source.files}
    work=pickle.load(open(args.work_checkpoint,"rb")); rows=manifest(args.manifest)
    stdout=open(args.slurm_out,encoding="utf-8").read(); stderr=open(args.slurm_err,encoding="utf-8").read()
    provenance=report["provenance"]
    provenance_check={"commit":provenance["commit"]==args.expected_commit,
        "job":str(provenance["slurm_job_id"])==str(args.expected_job),
        "gpu":provenance["jax_backend"]=="gpu" and (args.smoke or provenance["gpu_kind"]=="NVIDIA H200"),
        "precision":provenance["x64"] is True and provenance["matmul_precision"]=="highest",
        "logs":"jax_backend=gpu" in stdout and "ALL-DONE" in stdout
            and not re.search(r"(?i)(captured.*large.*constant|oom|out of memory|traceback|disk.*full)",stdout+stderr)}
    source_binding=bool(args.smoke or all(rows.get("code/"+name)==digest
        for name,digest in provenance["source_sha256"].items()))
    prereg_binding=bool(rows.get("code/PHASE-10-PRE-REGISTRATION.md")==c.sha256(args.prereg))
    if args.smoke:
        dependency_binding=True; accepted_bindings=recovery_bindings=failed_bindings=r1_failed_bindings=True
        accepted_terminal_field=None; recovery_reference_arrays=None; accepted_source_exact=True
    else:
        recovery_bindings=(p10.exact_bundle(args.recovery_dir,p10.EXPECTED_RECOVERY)==report["bindings"]["recovery"])
        accepted_bindings=(p10.exact_bundle(args.accepted_audit_dir,p10.EXPECTED_ACCEPTED)==report["bindings"]["accepted_audit"])
        r1_failed_bindings=(p10.validate_r1_failure(args.r1_failed_dir)==report["bindings"]["r1_failed"])
        failed_bindings=(recovery.validate_failed(args.failed_t1_dir)==report["bindings"]["failed_t1"])
        dependency_binding=bool(all(rows.get("code/deps/recovery/"+name)==digest for name,digest in p10.EXPECTED_RECOVERY.items())
            and all(rows.get("code/deps/accepted/"+name)==digest for name,digest in p10.EXPECTED_ACCEPTED.items())
            and all(rows.get("code/deps/p10_r1_failed/"+name)==digest for name,digest in p10.EXPECTED_R1_FAILED.items())
            and all(rows.get("code/deps/failed/"+name)==digest for name,digest in recovery.EXPECTED_FAILED.items()))
        recovery_report=load_json(os.path.join(args.recovery_dir,"out/phase9_terminal_recovery.json"))
        for phase,stem in {"p4":"phase4_d","p5":"phase5_d","p6":"phase6_d","p7":"phase7_train","p8":"phase8_d"}.items():
            kinds=("json","npz","checkpoint","audit","manifest") if phase=="p7" else ("json","npz","audit","manifest")
            for kind in kinds:
                basename="AUDIT.json" if kind=="audit" else "MANIFEST.sha256" if kind=="manifest" else "checkpoint.pkl" if kind=="checkpoint" else f"{stem}.{kind}"
                dependency_binding &= rows.get(f"code/deps/{phase}/{basename}")==recovery_report["bindings"][f"{phase}_{kind}"]["sha256"]
        p5_report=load_json(args.p5_json)
        dependency_binding &= len(p5_report["train_targets"]["chunks"])==16
        dependency_binding &= all(rows.get("code/deps/p5/targets/"+x["basename"])==x["sha256"] for x in p5_report["train_targets"]["chunks"])
        accepted_terminal_field=load_json(os.path.join(args.accepted_audit_dir,"out/AUDIT.json"))["terminal_train"]
        accepted_source_exact=bool(recovery_report["terminal_train"]==accepted_terminal_field)
        with np.load(os.path.join(args.recovery_dir,"out/phase9_terminal_recovery.npz"),allow_pickle=False) as control:
            recovery_reference_arrays={name:np.asarray(control[name]) for name in control.files
                                       if name.startswith("terminal_train_")}
    train,coefficients,_,targets,regeneration=recovery.load_train(args,args.smoke)
    features=legacy.concatenate(train,"features"); norm=p9.train_normalization(coefficients,features)
    if args.smoke:
        generator=p9.init_generator(p10.CONFIG); qraw=np.zeros((len(coefficients),19),np.float64); source_sha="synthetic"
    else:
        checkpoint_path=os.path.join(args.recovery_dir,"out/terminal_checkpoint.pkl")
        checkpoint=pickle.load(open(checkpoint_path,"rb")); generator=checkpoint["generator"]
        qraw=np.asarray(checkpoint["q_raw"]); source_sha=c.sha256(checkpoint_path)
    q0=np.tanh(qraw); affine=legacy.concatenate(train,"affine")
    data_normalization=bool(np.array_equal(arrays["training_affine"],affine)
        and np.array_equal(arrays["training_features"],features)
        and np.array_equal(arrays["coefficient_mean"],norm["mean"])
        and np.array_equal(arrays["head_scales"],norm["scales"])
        and np.array_equal(arrays["normalization_source_indices"],np.arange(len(coefficients)))
        and report["data"]["training"]==legacy.metadata(train) and report["data"]["target_chunks"]==targets
        and report["data"]["regeneration"]==regeneration)
    source_model=bool(report["source_model"]["checkpoint_sha256_before"]==source_sha
        and report["source_model"]["checkpoint_sha256_after"]==source_sha
        and report["source_model"]["generator_tree_sha256_before"]==recovery.tree_sha(generator)
        and report["source_model"]["generator_tree_sha256_after"]==recovery.tree_sha(generator)
        and report["source_model"]["q_raw_sha256_before"]==p9.array_sha(qraw)
        and report["source_model"]["q_raw_sha256_after"]==p9.array_sha(qraw)
        and report["source_model"]["immutable"] is True and tree_count(generator)==30594
        and np.array_equal(arrays["final_q_raw"],qraw) and np.array_equal(arrays["final_q_start"],q0)
        and np.array_equal(arrays["trust_q"][0,:,0],q0))
    fresh_initial={}; initial=p9.evaluate_full(train,generator,np.concatenate((affine,q0),axis=1),norm["mean"],norm["scales"],p10.CONFIG,fresh_initial,"initial_train",True)
    recovery.add_snapshot_losses(initial,fresh_initial,"initial_train",train)
    terminal_q=np.asarray(arrays["trust_q"])[0,:,-1]; fresh_terminal={}
    terminal=p9.evaluate_full(train,generator,np.concatenate((affine,terminal_q),axis=1),norm["mean"],norm["scales"],p10.CONFIG,fresh_terminal,"recovered_train",True)
    recovery.add_snapshot_losses(terminal,fresh_terminal,"recovered_train",train)
    meshes=[x["N"] for x in train]
    field_check=bool(exact_arrays(fresh_initial,arrays,"initial_train",meshes)
        and exact_arrays(fresh_terminal,arrays,"recovered_train",meshes)
        and initial==report["initial_train"] and terminal==report["recovered_train"])
    if args.smoke:
        accepted_terminal_field=initial
        recovery_reference_arrays={key.replace("initial_train_","terminal_train_",1):value
                                   for key,value in fresh_initial.items()}
    accepted_control=independent_initial_binding(initial,accepted_terminal_field,fresh_initial,
                                                  recovery_reference_arrays,meshes)
    reported_control=report["checks"]["accepted_initial_control"]
    expected_reported_control={
        "accepted_and_source_report_exact":accepted_source_exact,
        "metrics":{"rtol":PORTABILITY_RTOL,"atol":PORTABILITY_ATOL,
            "metric_max_abs":accepted_control["metric_max_abs"],
            "metric_exact":accepted_control["metric_exact"],
            "metrics_max_abs":accepted_control["metrics_max_abs"],
            "metrics_bitwise_exact":accepted_control["metrics_bitwise_exact"],
            "pass":bool(accepted_control["pass"])},
        "arrays":{"rtol":PORTABILITY_RTOL,"atol":PORTABILITY_ATOL,
            "array_max_abs":accepted_control["array_max_abs"],
            "array_exact":accepted_control["array_exact"],
            "arrays_max_abs":accepted_control["arrays_max_abs"],
            "arrays_bitwise_exact":accepted_control["arrays_bitwise_exact"],
            "pass":bool(accepted_control["pass"])},
        "initial_control_sha256":c.sha256(args.initial_control_npz),
        "pass":bool(accepted_source_exact and accepted_control["pass"])}
    # Metric and array contracts are separately true in the driver. The
    # independent combined audit has the same pass value because both are
    # required and uses separately recomputed values above.
    accepted_field_binding=bool(accepted_source_exact and accepted_control["pass"]
                                and reported_control==expected_reported_control)
    with np.load(args.initial_control_npz,allow_pickle=False) as saved:
        saved_keys=set(saved.files); expected_saved={key for key in arrays if key.startswith("initial_train_")}
        expected_saved |= {"final_q_start","training_affine"}
        initial_control_file=bool(saved_keys==expected_saved
            and all(np.array_equal(np.asarray(saved[key]),np.asarray(arrays[key])) for key in saved_keys)
            and np.array_equal(np.asarray(saved["final_q_start"]),q0)
            and np.array_equal(np.asarray(saved["training_affine"]),affine))
    trace=trace_check(arrays)
    objective_binding=bool(np.allclose(objectives(arrays,"initial_train",train),arrays["trust_objective"][0,:,0],rtol=2e-13,atol=2e-14)
        and np.allclose(objectives(arrays,"recovered_train",train),arrays["trust_objective"][0,:,-1],rtol=2e-13,atol=2e-14))
    work_binding=bool(work["status"]=="complete" and work["attempt_cap"]==(1 if args.smoke else 40)
        and np.array_equal(work["terminal_q"],terminal_q) and np.array_equal(work["terminal_objective"],arrays["trust_objective"][0,:,-1])
        and work["source_checkpoint_sha256"]==source_sha and work["generator_tree_sha256"]==recovery.tree_sha(generator)
        and work["optimizer_updates"]==0 and report["work_checkpoint"]["sha256"]==c.sha256(args.work_checkpoint))
    identity=bool(all(row["k3_cox_identity_worst"]<=p9.IDENTITY_TOL for row in
        [*initial["meshes"].values(),initial["pooled"],*terminal["meshes"].values(),terminal["pooled"]]))
    field_health=bool(all(row["all_finite"] and row["boundary_violation_count"]==0 for row in
        [*initial["meshes"].values(),initial["pooled"],*terminal["meshes"].values(),terminal["pooled"]]))
    health=bool(trace["pass"] and identity and field_health and field_check and accepted_field_binding
                and initial_control_file and objective_binding)
    train_pass=bool(health and p9.gate(terminal,2e-4,7e-4,True))
    expected_decision={"p10_d_valid":bool(not args.smoke and health and source_model),
        "fixed_g1_train_representable":bool(not args.smoke and health and source_model and train_pass),
        "corrected_g1_proposal_eligible":bool(not args.smoke and health and source_model and train_pass),
        "architecture_increase_licensed":False,"g2_licensed":False,"selection_evaluated":False,
        "optimizer_updates":0,"scientific_promotion_allowed":False,
        "next_action":("root audit of corrected-G1 proposal" if not args.smoke and train_pass
            else "separate architecture-justification proposal; no architecture license" if not args.smoke and health
            else "root audit; invalid diagnostic" if not args.smoke else "excluded smoke only")}
    decision=report["decision"]==expected_decision and decision_contract(report["decision"])
    information=report["information_boundary"]=={"train_only":True,"selection_touched":False,
        "model_validation_touched":False,"confirmation_touched":False,"weak_eq_touched":False,
        "scaling_touched":False,"capacity_used":False}
    config=report["config"]
    config_check=bool(config["sole_start"]=="tanh(final_q_raw)" and config["selection_touched"] is False
        and config["optimizer_updates"]==0 and config["attempts"]==(1 if args.smoke else 40)
        and config["cg_max"]==19 and config["cg_tolerance"]==1e-12 and config["delta0"]==.25
        and config["delta_min"]==2**-20 and config["delta_max"]==1 and config["lambda0"]==1e-6
        and config["lambda_min"]==1e-12 and config["lambda_max"]==1e12 and config["accept_rho"]==1e-4)
    artifacts=bool(report["npz"]["sha256"]==c.sha256(args.source_npz)
        and report["initial_control"]=={"basename":os.path.basename(args.initial_control_npz),
            "sha256":c.sha256(args.initial_control_npz)}
        and report["progress"]["sha256"]==c.sha256(os.path.join(os.path.dirname(args.source_json),report["progress"]["basename"]))
        and report["work_checkpoint"]["sha256"]==c.sha256(args.work_checkpoint))
    negative=negative_tests(expected_decision,arrays)
    portability_negative=portability_negative_tests()
    checks={"provenance":provenance_check,"source_manifest_binding":source_binding,
        "prereg_manifest_binding":prereg_binding,"dependency_binding":dependency_binding,
        "recovery_binding":recovery_bindings,"accepted_audit_binding":accepted_bindings,
        "r1_zero_attempt_binding":r1_failed_bindings,
        "failed_t1_binding":failed_bindings,"data_normalization":data_normalization,
        "source_model_immutable":source_model,"full_field_independent":field_check,
        "accepted_terminal_field_binding":accepted_field_binding,
        "accepted_terminal_field_portability":accepted_control,
        "initial_control_file_binding":initial_control_file,
        "trust_trace":trace,"objective_binding":objective_binding,"work_checkpoint_binding":work_binding,
        "identity":identity,"field_health":field_health,"health":health,"train_pass":train_pass,
        "config":config_check,"information_boundary":information,"artifacts":artifacts,
        "decision":decision,"negative_self_test":negative,
        "portability_negative_self_test":portability_negative}
    passed=bool(all(provenance_check.values()) and all(value if isinstance(value,bool) else value.get("pass",False)
        for key,value in checks.items() if key not in ("provenance","train_pass")) and decision)
    output={"status":"pass" if passed else "fail","negative_aware":True,"expected_commit":args.expected_commit,
        "expected_job":str(args.expected_job),"manifest_sha256":c.sha256(args.manifest),
        "source_json_sha256":c.sha256(args.source_json),"source_npz_sha256":c.sha256(args.source_npz),
        "initial_control_sha256":c.sha256(args.initial_control_npz),
        "work_checkpoint_sha256":c.sha256(args.work_checkpoint),"checks":checks,
        "initial_train":initial,"recovered_train":terminal,"decision":expected_decision,
        "capacity":{"accepted":False,"reproducible":False,"license_complete":False,"g2_licensed":False,"used":False}}
    p9.atomic_json(args.output,output)
    if not passed: raise SystemExit("Phase10-D independent audit failed")
    print(json.dumps({"status":"pass","decision":expected_decision},sort_keys=True),flush=True)


if __name__=="__main__": main()
