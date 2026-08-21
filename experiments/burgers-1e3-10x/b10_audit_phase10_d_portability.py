#!/usr/bin/env python
"""Audit-only portability closure for immutable Phase-10-D r2 artifacts."""
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

import b10_audit_phase10_d as old
import b10_audit_phase9_train as p9audit
import b10_common as c
import b10_phase10_d as p10
import b10_phase9_train as p9
import b10_phase9_terminal_recovery as recovery
import b10_spline_train as legacy


RTOL = 2e-13
ATOL = 2e-14
TERMINATION_BAND = 4e-16
TERMINATION_THRESHOLD = 1e-12
FLOAT_METRICS = ("trajectory_error_mean", "trajectory_error_worst",
                 "mean_snapshot_relative_l2_squared", "k3_cox_identity_worst")
FIELD_SUFFIXES = ("numerator", "denominator", "trajectory", "boundary_count", "identity")
SOURCE_COMMIT = "25bb3502b851a4eb56c51a25af3fc844e3583219"
SOURCE_JOB = "2739690"
EXPECTED_R2 = {
    "LOCAL.sha256": "36076277caf1372eb4d02735219bda76a50ace2f8d0e5d1a5ab6dd3ba88e6990",
    "MANIFEST.sha256": "60cd3e3f2d4946c8671f3cb43c3a852fd72a12b86735f3314bf7e78379edb45d",
    "REMOTE.sha256": "b00d582486c2a42e007600f96e188a16786aba44c6f68a7f9f5a3113ba574cd3",
    "SACCT.txt": "e820866cc0644bac1ba5e637aa201b9a246074db425eee1f7aeeebaad8bed99d",
    "logs/2739690.err": "e5fe48ff1ae7be8842bae484e6ae1b49247665ba51d4c8e4e8723db5cbaa1d4b",
    "logs/2739690.out": "eac0cb9090384c506ab767b6fe68e2a313201c27666a611b2114d86e0dda3f84",
    "out/AUDIT.json": "d4d593db619bb84e5ef659a79c0b3fe6e44e6533fd7cedaf8f988acc6a57abda",
    "out/PROGRESS.json": "4d522adecc615739303fa81fa71fdee8c9a904532df0e4da33a814d916d0ccba",
    "out/initial_control.npz": "03bcbd51fc375b87bb876953b910d8c6a11dcffc893048ed7d9a6f2c46f4290c",
    "out/phase10_d.json": "f0fab7583d1f1590efb58cd97d0128f34cc80cf4ab7ce8e3271066158f239523",
    "out/phase10_d.npz": "35d139fedcfff539aef25a335c556d47f2ec5e8581066a9e1f9b6ad8fcef4664",
    "out/work_checkpoint.pkl": "5fd879c32f2205d6427b272042c095624d824a8278e1eaf074c5295bcb574d68",
}
EXPECTED_AUDIT_R1 = {
    "LOCAL.sha256":"43b7d071a4301b9511da38cbbd2e74ac63a0bcba48abd4a5cad29a69ef98796d",
    "MANIFEST.sha256":"d462f02c9f4c6b00f43c3002e56c30eb88d62061894589b4b62e30fb3685f5cb",
    "REMOTE.sha256":"e7c9fae182d37060bb137d9d56b8ee743bacc915631c349aa9b77da4f47d628e",
    "SACCT.txt":"28aab3917b0dc99e93f3e6d762662d7a798c944fd8d21f23f43ceb2b3af59d4e",
    "logs/2747981.err":"8666e85d654007462211b1f558558eb59e7493fc5d4bce2840432caafe673598",
    "logs/2747981.out":"ca168fd48464dd72d82d26752cdbb62c2a0b617e29ac8775174ef834774790a7",
}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def exact_bundle(root, expected):
    result = {}
    for relative, digest in expected.items():
        path = os.path.join(root, relative)
        if not os.path.isfile(path) or c.sha256(path) != digest:
            raise SystemExit(f"immutable Phase10 r2 bundle mismatch: {relative}")
        result[relative] = digest
    recovery.verify_checksum_file(root, "LOCAL.sha256")
    return result


def float_comparison(actual, reference):
    left = np.asarray(actual); right = np.asarray(reference)
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype and np.issubdtype(left.dtype, np.floating)
    finite = bool(same_shape and same_dtype and np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
    if finite:
        delta = np.abs(left-right); bound = ATOL+RTOL*np.abs(right)
        relative = delta/np.maximum(np.abs(right), 1e-300)
        normalized = delta/np.maximum(bound, 1e-300)
        within = bool(np.all(delta <= bound))
        max_abs = float(np.max(delta)) if delta.size else 0.0
        max_rel = float(np.max(relative)) if relative.size else 0.0
        max_norm = float(np.max(normalized)) if normalized.size else 0.0
    else:
        within = False; max_abs = max_rel = max_norm = float("inf")
    return {"same_shape":bool(same_shape), "same_dtype":bool(same_dtype), "finite":finite,
            "max_absolute_difference":max_abs, "max_relative_difference":max_rel,
            "max_bound_normalized_difference":max_norm, "bitwise_exact":bool(same_shape and np.array_equal(left,right)),
            "pass":bool(finite and within)}


def metric_comparison(actual, reference, meshes):
    checks = {}; passed = set(actual)=={"meshes","pooled"}==set(reference)
    am=actual.get("meshes",{}); rm=reference.get("meshes",{})
    passed &= set(am)==set(rm)==set(str(n) for n in meshes)
    scopes=[("pooled",actual.get("pooled",{}),reference.get("pooled",{}))]
    scopes += [(f"N{n}",am.get(str(n),{}),rm.get(str(n),{})) for n in meshes]
    required=set(FLOAT_METRICS)|{"all_finite","boundary_violation_count"}
    for label,left,right in scopes:
        passed &= set(left)==required==set(right)
        categorical=bool(type(left.get("all_finite")) is bool and type(right.get("all_finite")) is bool
            and left.get("all_finite")==right.get("all_finite") is True
            and type(left.get("boundary_violation_count")) is int
            and type(right.get("boundary_violation_count")) is int
            and left.get("boundary_violation_count")==right.get("boundary_violation_count")==0)
        checks[f"{label}.categorical"]={"pass":categorical}
        passed &= categorical
        for name in FLOAT_METRICS:
            one=np.asarray(left.get(name),np.float64); two=np.asarray(right.get(name),np.float64)
            check=float_comparison(one,two)
            typed=type(left.get(name)) is float and type(right.get(name)) is float
            check["python_float_type"]=typed; check["pass"] &= typed
            checks[f"{label}.{name}"]=check; passed &= check["pass"]
    return {"rtol":RTOL,"atol":ATOL,"fields":checks,"pass":bool(passed)}


def field_array_comparison(actual, reference, actual_prefix, reference_prefix, meshes):
    checks={}; passed=True
    for n in meshes:
        for suffix in FIELD_SUFFIXES:
            left=np.asarray(actual[f"{actual_prefix}_N{n}_{suffix}"])
            right=np.asarray(reference[f"{reference_prefix}_N{n}_{suffix}"])
            label=f"N{n}.{suffix}"
            if suffix=="boundary_count":
                valid=bool(left.shape==right.shape and left.dtype==right.dtype
                    and np.issubdtype(left.dtype,np.integer) and np.array_equal(left,right) and np.all(left==0))
                check={"same_shape":left.shape==right.shape,"same_dtype":left.dtype==right.dtype,
                       "bitwise_exact":bool(np.array_equal(left,right)),"zero":bool(np.all(left==0)),"pass":valid}
            else:
                check=float_comparison(left,right)
            checks[label]=check; passed &= check["pass"]
    return {"rtol":RTOL,"atol":ATOL,"fields":checks,"pass":bool(passed)}


def named_array_comparison(actual, reference, names):
    fields={name:float_comparison(actual[name],reference[name]) for name in names}
    return {"rtol":RTOL,"atol":ATOL,"fields":fields,"pass":bool(all(x["pass"] for x in fields.values()))}


def structured_comparison(actual, reference):
    """Use the locked f64 bound for floats and exact equality for metadata."""
    details={}; passed=True
    def visit(left,right,path):
        nonlocal passed
        if isinstance(left,dict) and isinstance(right,dict):
            valid=set(left)==set(right); details[path+".__keys__"]={"pass":valid}; passed &= valid
            for key in sorted(set(left)&set(right)): visit(left[key],right[key],path+"."+key)
        elif isinstance(left,list) and isinstance(right,list):
            valid=len(left)==len(right); details[path+".__length__"]={"pass":valid}; passed &= valid
            for index,(one,two) in enumerate(zip(left,right)): visit(one,two,f"{path}[{index}]")
        elif type(left) is float and type(right) is float:
            check=float_comparison(np.asarray(left,np.float64),np.asarray(right,np.float64))
            details[path]=check; passed &= check["pass"]
        else:
            valid=type(left) is type(right) and left==right
            details[path]={"exact":bool(valid),"pass":bool(valid)}; passed &= valid
    visit(actual,reference,"root")
    return {"rtol":RTOL,"atol":ATOL,"fields":details,"pass":bool(passed)}


def normalization_binding(checkpoint, recovery_arrays, recovery_report, phase10_arrays):
    """Bind the real nested recovery-checkpoint normalization schema."""
    names={"mean":"coefficient_mean","scales":"head_scales",
           "feature_mean":"predictor_feature_mean","feature_scale":"predictor_feature_scale"}
    nested=checkpoint.get("normalization") if isinstance(checkpoint,dict) else None
    schema=bool(isinstance(nested,dict) and set(nested)==set(names))
    exact={}; hashes={}
    for nested_name,array_name in names.items():
        value=np.asarray(nested[nested_name]) if schema else np.asarray([],np.float64)
        recovery_value=np.asarray(recovery_arrays.get(array_name,[]))
        exact[nested_name]=bool(schema and value.dtype==recovery_value.dtype
            and np.issubdtype(value.dtype,np.floating) and np.all(np.isfinite(value))
            and np.array_equal(value,recovery_value))
        hashes[nested_name]=bool(schema and recovery_report["normalization"].get(
            ("mean_sha256" if nested_name=="mean" else "scales_sha256" if nested_name=="scales"
             else "feature_mean_sha256" if nested_name=="feature_mean" else "feature_scale_sha256"))==p9.array_sha(value))
    phase10_exact={"mean":bool(schema and np.array_equal(nested["mean"],phase10_arrays["coefficient_mean"])),
        "scales":bool(schema and np.array_equal(nested["scales"],phase10_arrays["head_scales"]))}
    return {"schema_exact":schema,"nested_recovery_npz_exact":exact,
            "nested_recovery_report_hashes":hashes,"phase10_mean_scales_exact":phase10_exact,
            "pass":bool(schema and all(exact.values()) and all(hashes.values()) and all(phase10_exact.values()))}


def termination_contract(recorded, accepted, relative_improvement, step_relative):
    host=accepted&((relative_improvement<=TERMINATION_THRESHOLD)|(step_relative<=TERMINATION_THRESHOLD))
    mismatch=recorded!=host
    distance=np.abs(relative_improvement-TERMINATION_THRESHOLD)
    allowed=mismatch&accepted&(step_relative>TERMINATION_THRESHOLD)&(distance<=TERMINATION_BAND)
    bad=mismatch&~allowed; indices=np.argwhere(mismatch)
    return {"threshold":TERMINATION_THRESHOLD,"roundoff_band":TERMINATION_BAND,
            "host_true_count":int(np.sum(host)),"recorded_true_count":int(np.sum(recorded)),
            "mismatch_count":int(np.sum(mismatch)),"allowed_roundoff_mismatch_count":int(np.sum(allowed)),
            "bad_mismatch_count":int(np.sum(bad)),"mismatch_indices":indices.tolist(),
            "mismatch_distance_from_threshold":[float(distance[tuple(x)]) for x in indices],
            "pass":bool(not np.any(bad))}


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
    relative=(objective[:,:,:-1]-trial)/np.maximum(objective[:,:,:-1],1e-300)
    step_relative=np.linalg.norm(step,axis=-1)/(1+np.linalg.norm(q[:,:,:-1],axis=-1))
    termination=termination_contract(terminated,expected_accept,relative,step_relative)
    shrink=(~accepted)|(defined&(rho<.25)); expand=accepted&(rho>.75)&(np.linalg.norm(step,axis=-1)>=.9*delta[:,:,:-1]); improve=accepted&(rho>.75)
    update_delta=np.where(shrink,np.maximum(delta[:,:,:-1]/4,p9.DELTA_MIN),np.where(expand,np.minimum(2*delta[:,:,:-1],p9.DELTA_MAX),delta[:,:,:-1]))
    update_damping=np.where(shrink,np.minimum(10*damping[:,:,:-1],p9.LAMBDA_MAX),np.where(improve,np.maximum(damping[:,:,:-1]/3,p9.LAMBDA_MIN),damping[:,:,:-1]))
    expected_delta=np.where(attempted,update_delta,delta[:,:,:-1]); expected_damping=np.where(attempted,update_damping,damping[:,:,:-1])
    expected_q=np.where(accepted[...,None],q[:,:,:-1]+step,q[:,:,:-1]); expected_objective=np.where(accepted,trial,objective[:,:,:-1])
    exhaustion=active[:,:,-1]&(delta[:,:,-1]<=p9.DELTA_MIN)&(damping[:,:,-1]>=p9.LAMBDA_MAX)
    checks={"one_start_shape":q.shape[0]==1,
        "attempt_work_finite":bool(np.all(finite[attempted]) and np.all(np.isfinite(pred[attempted]))
            and np.all(np.isfinite(actual[attempted])) and np.all(np.isfinite(rho[attempted]))),
        "rho_definition":bool(np.array_equal(defined[attempted],pred[attempted]>0)),
        "positive_rho_exact":bool(p9audit.close(rho[positive],actual[positive]/pred[positive])) if np.any(positive) else True,
        "nonpositive_sentinel":bool(np.all(rho[nonpositive]==0)&~np.any(accepted[nonpositive])),
        "no_breakdown":bool(~np.any(breakdown[attempted])),"attempt_active_exact":bool(np.array_equal(attempted,active[:,:,:-1])),
        "acceptance_exact":bool(np.array_equal(accepted,expected_accept)),"termination_portable":termination["pass"],
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
    return {**checks,"termination":termination,"pass":bool(all(checks.values()))}


def corruption_tests(decision):
    reference=np.asarray([0.,1.,10.],np.float64)
    portable=reference+np.asarray([1e-14,1e-13,1e-12])
    beyond=reference.copy(); beyond[1]+=1e-8
    floating_positive=float_comparison(portable,reference)["pass"]
    floating_negative=not float_comparison(beyond,reference)["pass"]
    accepted=np.asarray([True,True,True,False])
    relative=np.asarray([.9e-12,1e-12+2e-16,1e-12+8e-16,.9e-12])
    step=np.full(4,1e-6); recorded=np.asarray([True,True,False,False])
    exact_and_near=termination_contract(recorded,accepted,relative,step)["pass"]
    corrupted=recorded.copy(); corrupted[2]=True
    beyond_rejected=not termination_contract(corrupted,accepted,relative,step)["pass"]
    corrupted=recorded.copy(); corrupted[3]=True
    inactive_rejected=not termination_contract(corrupted,accepted,relative,step)["pass"]
    decision_cases=[]
    for key,value in (("architecture_increase_licensed",True),("g2_licensed",True),
                      ("selection_evaluated",True),("optimizer_updates",1),
                      ("scientific_promotion_allowed",True),
                      ("corrected_g1_proposal_eligible",not decision["fixed_g1_train_representable"])):
        one=copy.deepcopy(decision); one[key]=value; decision_cases.append(not old.decision_contract(one))
    values={"mean":np.asarray([1.,2.]),"scales":np.asarray([3.,4.]),
            "feature_mean":np.asarray([5.,6.]),"feature_scale":np.asarray([7.,8.])}
    recovery_arrays={"coefficient_mean":values["mean"],"head_scales":values["scales"],
        "predictor_feature_mean":values["feature_mean"],"predictor_feature_scale":values["feature_scale"]}
    recovery_report={"normalization":{"mean_sha256":p9.array_sha(values["mean"]),
        "scales_sha256":p9.array_sha(values["scales"]),"feature_mean_sha256":p9.array_sha(values["feature_mean"]),
        "feature_scale_sha256":p9.array_sha(values["feature_scale"])}}
    checkpoint={"normalization":copy.deepcopy(values)}
    normalization_positive=normalization_binding(checkpoint,recovery_arrays,recovery_report,recovery_arrays)["pass"]
    normalization_cases=[]
    one={"coefficient_mean":values["mean"]}; normalization_cases.append(
        not normalization_binding(one,recovery_arrays,recovery_report,recovery_arrays)["pass"])
    one=copy.deepcopy(checkpoint); one["normalization"]["mean"][0]+=1e-8; normalization_cases.append(
        not normalization_binding(one,recovery_arrays,recovery_report,recovery_arrays)["pass"])
    one=copy.deepcopy(recovery_report); one["normalization"]["feature_scale_sha256"]="bad"; normalization_cases.append(
        not normalization_binding(checkpoint,recovery_arrays,one,recovery_arrays)["pass"])
    one=copy.deepcopy(checkpoint); del one["normalization"]["feature_mean"]; normalization_cases.append(
        not normalization_binding(one,recovery_arrays,recovery_report,recovery_arrays)["pass"])
    return {"floating_portable_positive":bool(floating_positive),
            "floating_larger_perturbation_rejected":bool(floating_negative),
            "termination_exact_and_near_positive":bool(exact_and_near),
            "termination_beyond_band_rejected":bool(beyond_rejected),
            "termination_inactive_corruption_rejected":bool(inactive_rejected),
            "decision_corruption_count":len(decision_cases),
            "decision_corruptions_rejected":bool(all(decision_cases)),
            "normalization_schema_positive":bool(normalization_positive),
            "normalization_corruption_count":len(normalization_cases),
            "normalization_corruptions_rejected":bool(all(normalization_cases)),
            "pass":bool(floating_positive and floating_negative and exact_and_near
                         and beyond_rejected and inactive_rejected and all(decision_cases)
                         and normalization_positive and all(normalization_cases))}


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--r2-bundle"); parser.add_argument("--audit-r1-failed-dir"); parser.add_argument("--recovery-dir")
    parser.add_argument("--accepted-audit-dir"); parser.add_argument("--failed-t1-dir")
    parser.add_argument("--r1-failed-dir"); parser.add_argument("--p5-json")
    parser.add_argument("--p7-npz"); parser.add_argument("--target-dir")
    parser.add_argument("--manifest",required=True); parser.add_argument("--prereg",required=True)
    parser.add_argument("--expected-source-commit",required=True); parser.add_argument("--expected-source-job",required=True)
    parser.add_argument("--expected-audit-commit",required=True); parser.add_argument("--expected-audit-job",required=True)
    parser.add_argument("--slurm-out",required=True); parser.add_argument("--slurm-err",required=True)
    parser.add_argument("--output",required=True); parser.add_argument("--smoke",action="store_true")
    return parser.parse_args()


def main():
    args=parse_args(); c.require_gpu_highest(); rows=old.manifest(args.manifest)
    audit_provenance=c.provenance(); audit_stdout=open(args.slurm_out,encoding="utf-8").read()
    audit_stderr=open(args.slurm_err,encoding="utf-8").read()
    audit_provenance_check={"commit":audit_provenance["commit"]==args.expected_audit_commit,
        "job":str(audit_provenance["slurm_job_id"])==str(args.expected_audit_job),
        "gpu":audit_provenance["jax_backend"]=="gpu" and (args.smoke or audit_provenance["gpu_kind"]=="NVIDIA H200"),
        "precision":audit_provenance["x64"] is True and audit_provenance["matmul_precision"]=="highest",
        "logs":"jax_backend=gpu" in audit_stdout and not re.search(
            r"(?i)(captured.*large.*constant|oom|out of memory|traceback|disk.*full)",audit_stdout+audit_stderr)}
    audit_source_binding=bool(args.smoke or all(rows.get("code/"+name)==digest
        for name,digest in audit_provenance["source_sha256"].items()))
    prereg_binding=bool(rows.get("code/PHASE-10-PRE-REGISTRATION.md")==c.sha256(args.prereg))

    if args.smoke:
        r2_binding=audit_r1_binding=recovery_binding=accepted_binding=failed_binding=r1_binding=True
        source_json=os.path.join(args.r2_bundle,"phase10_d.json")
        source_npz=os.path.join(args.r2_bundle,"phase10_d.npz")
        initial_path=os.path.join(args.r2_bundle,"initial_control.npz")
        work_path=os.path.join(args.r2_bundle,"work_checkpoint.pkl")
        source_bundle_manifest=None
    else:
        required=(args.r2_bundle,args.audit_r1_failed_dir,args.recovery_dir,args.accepted_audit_dir,args.failed_t1_dir,
                  args.r1_failed_dir,args.p5_json,args.p7_npz,args.target_dir)
        if any(x is None for x in required): raise SystemExit("audit-only cell requires exact dependency paths")
        r2_binding=exact_bundle(args.r2_bundle,EXPECTED_R2)
        audit_r1_binding=exact_bundle(args.audit_r1_failed_dir,EXPECTED_AUDIT_R1)
        recovery_binding=p10.exact_bundle(args.recovery_dir,p10.EXPECTED_RECOVERY)
        accepted_binding=p10.exact_bundle(args.accepted_audit_dir,p10.EXPECTED_ACCEPTED)
        failed_binding=recovery.validate_failed(args.failed_t1_dir)
        r1_binding=p10.validate_r1_failure(args.r1_failed_dir)
        source_json=os.path.join(args.r2_bundle,"out/phase10_d.json")
        source_npz=os.path.join(args.r2_bundle,"out/phase10_d.npz")
        initial_path=os.path.join(args.r2_bundle,"out/initial_control.npz")
        work_path=os.path.join(args.r2_bundle,"out/work_checkpoint.pkl")
        source_bundle_manifest=old.manifest(os.path.join(args.r2_bundle,"MANIFEST.sha256"))
    report=load_json(source_json)
    with np.load(source_npz,allow_pickle=False) as data: arrays={name:np.asarray(data[name]) for name in data.files}
    with np.load(initial_path,allow_pickle=False) as data: initial_control={name:np.asarray(data[name]) for name in data.files}
    work=pickle.load(open(work_path,"rb")); provenance=report["provenance"]
    if args.smoke:
        source_logs=True; source_manifest_binding=True; original_audit_failure=True
    else:
        source_out=open(os.path.join(args.r2_bundle,"logs/2739690.out"),encoding="utf-8").read()
        source_err=open(os.path.join(args.r2_bundle,"logs/2739690.err"),encoding="utf-8").read()
        source_logs=bool("jax_backend=gpu" in source_out and "ALL-DONE" in source_out
            and source_err.strip()=="Phase10-D independent audit failed"
            and not re.search(r"(?i)(captured.*large.*constant|oom|out of memory|traceback|disk.*full)",source_out))
        source_manifest_binding=bool(all(source_bundle_manifest.get("code/"+name)==digest
            for name,digest in provenance["source_sha256"].items()))
        old_audit=load_json(os.path.join(args.r2_bundle,"out/AUDIT.json"))
        original_audit_failure=bool(old_audit["status"]=="fail"
            and old_audit["checks"]["trust_trace"]["termination_exact"] is False
            and old_audit["checks"]["accepted_terminal_field_binding"] is False)
    source_provenance={"commit":provenance["commit"]==args.expected_source_commit,
        "job":str(provenance["slurm_job_id"])==str(args.expected_source_job),
        "gpu":provenance["jax_backend"]=="gpu" and (args.smoke or provenance["gpu_kind"]=="NVIDIA H200"),
        "precision":provenance["x64"] is True and provenance["matmul_precision"]=="highest",
        "logs":source_logs,"manifest":source_manifest_binding}

    if args.smoke:
        train,coefficients,_,targets,regeneration=recovery.load_train(args,True)
        norm=p9.train_normalization(coefficients,legacy.concatenate(train,"features"))
        generator=p9.init_generator(p10.CONFIG); qraw=np.zeros((len(coefficients),19),np.float64)
        source_sha="synthetic"; accepted_metrics=report["initial_train"]
        accepted_arrays={key.replace("initial_train_","terminal_train_",1):value
                         for key,value in initial_control.items() if key.startswith("initial_train_")}
        checkpoint={"normalization":{"mean":norm["mean"],"scales":norm["scales"],
            "feature_mean":norm["feature_mean"],"feature_scale":norm["feature_scale"]}}
        recovery_norm_arrays={"coefficient_mean":norm["mean"],"head_scales":norm["scales"],
            "predictor_feature_mean":norm["feature_mean"],"predictor_feature_scale":norm["feature_scale"]}
        recovery_report={"normalization":{"mean_sha256":p9.array_sha(norm["mean"]),
            "scales_sha256":p9.array_sha(norm["scales"]),"feature_mean_sha256":p9.array_sha(norm["feature_mean"]),
            "feature_scale_sha256":p9.array_sha(norm["feature_scale"])}}
        accepted_source_exact=True; dependency_binding=True
    else:
        recovery_report=load_json(os.path.join(args.recovery_dir,"out/phase9_terminal_recovery.json"))
        accepted_report=load_json(os.path.join(args.accepted_audit_dir,"out/AUDIT.json"))
        accepted_metrics=accepted_report["terminal_train"]
        accepted_source_exact=bool(recovery_report["terminal_train"]==accepted_metrics)
        with np.load(os.path.join(args.recovery_dir,"out/phase9_terminal_recovery.npz"),allow_pickle=False) as data:
            accepted_arrays={name:np.asarray(data[name]) for name in data.files if name.startswith("terminal_train_")}
            recovery_norm_arrays={name:np.asarray(data[name]) for name in
                ("coefficient_mean","head_scales","predictor_feature_mean","predictor_feature_scale")}
        train,coefficients,_,targets,regeneration=recovery.load_train(args,False)
        norm=p9.train_normalization(coefficients,legacy.concatenate(train,"features"))
        checkpoint_path=os.path.join(args.recovery_dir,"out/terminal_checkpoint.pkl")
        checkpoint=pickle.load(open(checkpoint_path,"rb")); generator=checkpoint["generator"]
        qraw=np.asarray(checkpoint["q_raw"]); source_sha=c.sha256(checkpoint_path)
        dependency_binding=bool(all(rows.get("code/deps/r2_failed/"+name)==digest for name,digest in EXPECTED_R2.items())
            and all(rows.get("code/deps/audit_r1_failed/"+name)==digest for name,digest in EXPECTED_AUDIT_R1.items())
            and all(rows.get("code/deps/recovery/"+name)==digest for name,digest in p10.EXPECTED_RECOVERY.items())
            and all(rows.get("code/deps/accepted/"+name)==digest for name,digest in p10.EXPECTED_ACCEPTED.items())
            and all(rows.get("code/deps/failed/"+name)==digest for name,digest in recovery.EXPECTED_FAILED.items())
            and all(rows.get("code/deps/p10_r1_failed/"+name)==digest for name,digest in p10.EXPECTED_R1_FAILED.items()))
        for phase,stem in {"p4":"phase4_d","p5":"phase5_d","p6":"phase6_d","p7":"phase7_train","p8":"phase8_d"}.items():
            kinds=("json","npz","checkpoint","audit","manifest") if phase=="p7" else ("json","npz","audit","manifest")
            for kind in kinds:
                basename="AUDIT.json" if kind=="audit" else "MANIFEST.sha256" if kind=="manifest" else "checkpoint.pkl" if kind=="checkpoint" else f"{stem}.{kind}"
                dependency_binding &= rows.get(f"code/deps/{phase}/{basename}")==recovery_report["bindings"][f"{phase}_{kind}"]["sha256"]
        p5_report=load_json(args.p5_json)
        dependency_binding &= len(p5_report["train_targets"]["chunks"])==16
        dependency_binding &= all(rows.get("code/deps/p5/targets/"+x["basename"])==x["sha256"] for x in p5_report["train_targets"]["chunks"])

    features=legacy.concatenate(train,"features"); affine=legacy.concatenate(train,"affine")
    q0=np.tanh(qraw); meshes=[x["N"] for x in train]
    expected_initial_keys={key for key in arrays if key.startswith("initial_train_")}
    expected_initial_keys |= {"final_q_start","training_affine"}
    initial_control_exact=bool(set(initial_control)==expected_initial_keys
        and all(np.array_equal(initial_control[key],arrays[key]) for key in expected_initial_keys)
        and np.array_equal(initial_control["final_q_start"],q0)
        and np.array_equal(initial_control["training_affine"],affine)
        and report["initial_control"]=={"basename":os.path.basename(initial_path),"sha256":c.sha256(initial_path)})

    # Bind the driver's persisted control directly to the accepted immutable
    # source, independently of this process's repeated evaluation.
    accepted_control_arrays=field_array_comparison(
        initial_control,accepted_arrays,"initial_train","terminal_train",meshes)
    accepted_control_metrics=metric_comparison(report["initial_train"],accepted_metrics,meshes)
    accepted_control=bool(accepted_source_exact and accepted_control_arrays["pass"]
                          and accepted_control_metrics["pass"])

    if args.smoke:
        immutable_data=True
    else:
        with np.load(args.p7_npz,allow_pickle=False) as p7_source:
            immutable_data=bool(np.array_equal(arrays["training_affine"],np.asarray(p7_source["training_affine"]))
                and np.array_equal(arrays["training_features"],np.asarray(p7_source["training_features"])))
        immutable_data &= bool(np.array_equal(arrays["normalization_source_indices"],np.arange(len(coefficients))))
    recovered_normalization=normalization_binding(checkpoint,recovery_norm_arrays,recovery_report,arrays)
    immutable_data &= recovered_normalization["pass"]
    immutable_data &= bool(report["normalization"]=={
        "source":"train only","mean_sha256":p9.array_sha(arrays["coefficient_mean"]),
        "scales_sha256":p9.array_sha(arrays["head_scales"]),
        "normalized_coefficients_sha256":p9.array_sha(norm["normalized"])})
    repeated_data=named_array_comparison(
        {"training_affine":affine,"training_features":features,
         "coefficient_mean":norm["mean"],"head_scales":norm["scales"]},arrays,
        ("training_affine","training_features","coefficient_mean","head_scales"))
    metadata_repeat=structured_comparison(legacy.metadata(train),report["data"]["training"])
    targets_exact=bool(targets==report["data"]["target_chunks"])
    regeneration_repeat=structured_comparison(regeneration,report["data"]["regeneration"])

    source_model=bool(report["source_model"]["checkpoint_sha256_before"]==source_sha
        and report["source_model"]["checkpoint_sha256_after"]==source_sha
        and report["source_model"]["generator_tree_sha256_before"]==recovery.tree_sha(generator)
        and report["source_model"]["generator_tree_sha256_after"]==recovery.tree_sha(generator)
        and report["source_model"]["q_raw_sha256_before"]==p9.array_sha(qraw)
        and report["source_model"]["q_raw_sha256_after"]==p9.array_sha(qraw)
        and report["source_model"]["immutable"] is True and old.tree_count(generator)==30594
        and np.array_equal(arrays["final_q_raw"],qraw)
        and np.array_equal(arrays["final_q_start"],q0)
        and np.array_equal(arrays["trust_q"][0,:,0],q0))

    fresh_initial={}
    initial=p9.evaluate_full(train,generator,np.concatenate((affine,q0),axis=1),
        norm["mean"],norm["scales"],p10.CONFIG,fresh_initial,"initial_train",True)
    recovery.add_snapshot_losses(initial,fresh_initial,"initial_train",train)
    terminal_q=np.asarray(arrays["trust_q"])[0,:,-1]; fresh_terminal={}
    terminal=p9.evaluate_full(train,generator,np.concatenate((affine,terminal_q),axis=1),
        norm["mean"],norm["scales"],p10.CONFIG,fresh_terminal,"recovered_train",True)
    recovery.add_snapshot_losses(terminal,fresh_terminal,"recovered_train",train)
    repeated_initial_arrays=field_array_comparison(fresh_initial,arrays,"initial_train","initial_train",meshes)
    repeated_terminal_arrays=field_array_comparison(fresh_terminal,arrays,"recovered_train","recovered_train",meshes)
    repeated_initial_metrics=metric_comparison(initial,report["initial_train"],meshes)
    repeated_terminal_metrics=metric_comparison(terminal,report["recovered_train"],meshes)
    repeated_full_field=bool(repeated_initial_arrays["pass"] and repeated_terminal_arrays["pass"]
        and repeated_initial_metrics["pass"] and repeated_terminal_metrics["pass"])

    trace=trace_check(arrays)
    objective_binding=bool(np.allclose(old.objectives(arrays,"initial_train",train),
        arrays["trust_objective"][0,:,0],rtol=RTOL,atol=ATOL)
        and np.allclose(old.objectives(arrays,"recovered_train",train),
        arrays["trust_objective"][0,:,-1],rtol=RTOL,atol=ATOL))
    work_binding=bool(work["status"]=="complete" and work["attempt_cap"]==(1 if args.smoke else 40)
        and np.array_equal(work["terminal_q"],terminal_q)
        and np.array_equal(work["terminal_objective"],arrays["trust_objective"][0,:,-1])
        and work["source_checkpoint_sha256"]==source_sha
        and work["generator_tree_sha256"]==recovery.tree_sha(generator)
        and work["optimizer_updates"]==0
        and report["work_checkpoint"]["sha256"]==c.sha256(work_path))
    identity=bool(all(row["k3_cox_identity_worst"]<=p9.IDENTITY_TOL for row in
        [*initial["meshes"].values(),initial["pooled"],*terminal["meshes"].values(),terminal["pooled"]]))
    field_health=bool(all(row["all_finite"] and row["boundary_violation_count"]==0 for row in
        [*initial["meshes"].values(),initial["pooled"],*terminal["meshes"].values(),terminal["pooled"]]))
    health=bool(trace["pass"] and identity and field_health and repeated_full_field
        and accepted_control and initial_control_exact and objective_binding)
    train_pass=bool(health and p9.gate(terminal,2e-4,7e-4,True))
    expected_decision={"p10_d_valid":bool(not args.smoke and health and source_model),
        "fixed_g1_train_representable":bool(not args.smoke and health and source_model and train_pass),
        "corrected_g1_proposal_eligible":bool(not args.smoke and health and source_model and train_pass),
        "architecture_increase_licensed":False,"g2_licensed":False,"selection_evaluated":False,
        "optimizer_updates":0,"scientific_promotion_allowed":False,
        "next_action":("root audit of corrected-G1 proposal" if not args.smoke and train_pass
            else "separate architecture-justification proposal; no architecture license" if not args.smoke and health
            else "root audit; invalid diagnostic" if not args.smoke else "excluded smoke only")}
    decision=bool(report["decision"]==expected_decision and old.decision_contract(report["decision"]))
    information=report["information_boundary"]=={"train_only":True,"selection_touched":False,
        "model_validation_touched":False,"confirmation_touched":False,"weak_eq_touched":False,
        "scaling_touched":False,"capacity_used":False}
    config=report["config"]
    config_check=bool(config["sole_start"]=="tanh(final_q_raw)" and config["selection_touched"] is False
        and config["optimizer_updates"]==0 and config["attempts"]==(1 if args.smoke else 40)
        and config["cg_max"]==19 and config["cg_tolerance"]==1e-12 and config["delta0"]==.25
        and config["delta_min"]==2**-20 and config["delta_max"]==1 and config["lambda0"]==1e-6
        and config["lambda_min"]==1e-12 and config["lambda_max"]==1e12 and config["accept_rho"]==1e-4)
    artifacts=bool(report["npz"]=={"basename":os.path.basename(source_npz),"sha256":c.sha256(source_npz)}
        and report["progress"]["sha256"]==c.sha256(os.path.join(os.path.dirname(source_json),report["progress"]["basename"]))
        and report["work_checkpoint"]["sha256"]==c.sha256(work_path))
    corruptions=corruption_tests(expected_decision)
    checks={"audit_provenance":audit_provenance_check,"audit_source_binding":audit_source_binding,
        "prereg_manifest_binding":prereg_binding,"dependency_binding":dependency_binding,
        "immutable_r2_bundle_binding":bool(r2_binding),"audit_r1_failure_binding":bool(audit_r1_binding),
        "recovery_binding":bool(recovery_binding),
        "accepted_audit_binding":bool(accepted_binding),"failed_t1_binding":bool(failed_binding),
        "r1_failure_binding":bool(r1_binding),"source_provenance":source_provenance,
        "original_audit_failure_bound":original_audit_failure,"initial_control_file_exact":initial_control_exact,
        "accepted_initial_control_arrays":accepted_control_arrays,
        "accepted_initial_control_metrics":accepted_control_metrics,
        "accepted_initial_control":accepted_control,"immutable_data_binding":immutable_data,
        "recovered_normalization_binding":recovered_normalization,
        "repeated_data_arrays":repeated_data,"repeated_training_metadata":metadata_repeat,
        "target_chunks_exact":targets_exact,"repeated_regeneration":regeneration_repeat,
        "source_model_immutable":source_model,"repeated_initial_arrays":repeated_initial_arrays,
        "repeated_terminal_arrays":repeated_terminal_arrays,"repeated_initial_metrics":repeated_initial_metrics,
        "repeated_terminal_metrics":repeated_terminal_metrics,"repeated_full_field":repeated_full_field,
        "trust_trace":trace,"objective_binding":objective_binding,"work_checkpoint_binding":work_binding,
        "identity":identity,"field_health":field_health,"health":health,"config":config_check,
        "information_boundary":information,"artifacts":artifacts,"decision":decision,
        "negative_self_test":corruptions,"train_gate_descriptive":train_pass}
    passed=bool(all(audit_provenance_check.values()) and all(source_provenance.values())
        and all((value if isinstance(value,bool) else value.get("pass",False))
                for key,value in checks.items() if key not in ("audit_provenance","source_provenance","train_gate_descriptive")))
    output={"status":"pass" if passed else "fail","audit_only":True,"negative_aware":True,
        "expected_audit_commit":args.expected_audit_commit,"expected_audit_job":str(args.expected_audit_job),
        "manifest_sha256":c.sha256(args.manifest),"source_commit":args.expected_source_commit,
        "source_job":str(args.expected_source_job),"source_bundle_sha256":EXPECTED_R2,
        "audit_r1_failure_sha256":EXPECTED_AUDIT_R1,
        "source_json_sha256":c.sha256(source_json),"source_npz_sha256":c.sha256(source_npz),
        "initial_control_sha256":c.sha256(initial_path),"work_checkpoint_sha256":c.sha256(work_path),
        "portability_contract":{"relative_tolerance":RTOL,"absolute_tolerance":ATOL,
            "termination_roundoff_band":TERMINATION_BAND},"checks":checks,
        "initial_train":initial,"recovered_train":terminal,"decision":expected_decision,
        "capacity":{"accepted":False,"reproducible":False,"license_complete":False,
                    "g2_licensed":False,"used":False}}
    p9.atomic_json(args.output,output)
    if not passed: raise SystemExit("Phase10-D audit-only portability repair failed")
    print(json.dumps({"status":"pass","decision":expected_decision},sort_keys=True),flush=True)


if __name__=="__main__": main()
