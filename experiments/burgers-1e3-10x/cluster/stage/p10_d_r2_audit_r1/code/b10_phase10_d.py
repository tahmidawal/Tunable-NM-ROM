#!/usr/bin/env python
"""Phase-10 train-only fixed-G1 final-q globalization diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_phase9_train as p9
import b10_phase9_terminal_recovery as recovery
import b10_spline_train as legacy


CONFIG = p9.ARMS["T1"]
ATTEMPTS = 40
BATCH = 8
PORTABILITY_RTOL = 2e-13
PORTABILITY_ATOL = 2e-14
FIELD_ARRAY_SUFFIXES = ("numerator", "denominator", "trajectory", "boundary_count", "identity")
FLOAT_METRICS = ("trajectory_error_mean", "trajectory_error_worst",
                 "mean_snapshot_relative_l2_squared", "k3_cox_identity_worst")
EXPECTED_RECOVERY = {
    "out/phase9_terminal_recovery.json": "8d86ab90c1d293e4810c3d3a9391db48f635f8c77a7a8c30e71809a112bc7f8d",
    "out/phase9_terminal_recovery.npz": "d6e5001d8dcb5ba7fb269241b989492533379807bcbc4cb65a384f7d429c773e",
    "out/terminal_checkpoint.pkl": "90e9df6388bf3c05905d52c5ff073f331728ffd493a965e4376e4df285bb07d9",
    "out/AUDIT.json": "a1c6c0ee23e16da4aab1e4320aedcab4bd7039ec650429908feb15a471f99cec",
    "LOCAL.sha256": "2ea4a09470eb36d7b1ea5d45fc1ff2144be1a23dac55789281b05ce7b33f94fc",
}
EXPECTED_ACCEPTED = {
    "out/AUDIT.json": "b7bb908addaeb54c81293624c4433c61388c03faf9f514dbe5c8f3ec9d281377",
    "out/AUDIT-WORK.npz": "66ef97a0daeb2036e0aa6ccc3f31863e6861702baaa9711311d1839541cd556b",
    "LOCAL.sha256": "65e546fa0226fa0cc50a9d4f5953f53f1b51995a1741be656481621571a35a65",
    "MANIFEST.sha256": "9173714f5a03feef10a9f22f21b4c0c6466caad34e710ccb3131ebb655730edc",
}
EXPECTED_R1_FAILED = {
    "FAILURE.json": "511f0654d43b9c2f05f530bdd03fad54a3a1db0c9adeb4d083ef8c5ac2ecb943",
    "LOCAL.sha256": "ac033e5fcff3afced17886831307c3bb6913fbda7c73607c878c939086e8f3a1",
    "MANIFEST.sha256": "78caed8611e904bb53b961ade0c97be33a5b0dbc19f2c71a1d6ba30590e9de16",
    "REMOTE.sha256": "e05aad95bcd633fd666b8bd0eb77ab82ac97ba504b2e63b48a1d5fff367842dc",
    "SACCT.txt": "04a7edd0a6ca9f495df685d08925001716d69f1ca812188de46babbdf8bc5030",
    "logs/2738710.err": "28fcd4e8dfe32f33db42a6900870cd6f6cd3638dc704e6fce557572692721039",
    "logs/2738710.out": "78623084ca91d06849f503e31ac869f7fadf5a087a2645364d61038c3de8e780",
    "out/PROGRESS.json": "73ffeb145cca7d0c214811d471c841bcab4febe30cf9cffda456f9257850dfde",
}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def exact_bundle(root, expected):
    result = {}
    for relative, digest in expected.items():
        path = os.path.join(root, relative)
        if not os.path.isfile(path) or c.sha256(path) != digest:
            raise SystemExit(f"immutable bundle mismatch: {relative}")
        result[relative] = digest
    recovery.verify_checksum_file(root, "LOCAL.sha256")
    return result


def validate_r1_failure(root):
    result = exact_bundle(root, EXPECTED_R1_FAILED)
    failure = load_json(os.path.join(root, "FAILURE.json"))
    progress = load_json(os.path.join(root, "out/PROGRESS.json"))
    if not (failure["status"] == "excluded_pre_trust_initial_binding_failure"
            and failure["scientific_result"] is False
            and failure["trust_attempts_started"] == 0
            and failure["optimizer_updates"] == 0
            and failure["selection_touched"] is False
            and failure["capacity_used"] is False
            and failure["scientific_metrics_persisted"] is False
            and progress["stage"] == "initial_full_field"
            and progress["optimizer_updates"] == 0
            and progress["scientific_metrics_exposed"] is False):
        raise SystemExit("Phase10 r1 zero-attempt failure classification mismatch")
    return result


def metric_binding(actual, expected):
    """Audit-grade cross-node comparison of the fixed initial metric control."""
    details = {"rtol": PORTABILITY_RTOL, "atol": PORTABILITY_ATOL,
               "metric_max_abs": {}, "metric_exact": {}}
    passed = set(actual) == {"meshes", "pooled"} == set(expected)
    scopes = [("pooled", actual.get("pooled", {}), expected.get("pooled", {}))]
    actual_meshes = actual.get("meshes", {}); expected_meshes = expected.get("meshes", {})
    passed &= set(actual_meshes) == set(expected_meshes)
    scopes += [(f"N{n}", actual_meshes.get(n, {}), expected_meshes.get(n, {}))
               for n in sorted(expected_meshes)]
    required = set(FLOAT_METRICS) | {"all_finite", "boundary_violation_count"}
    for label, fresh, reference in scopes:
        passed &= set(fresh) == required == set(reference)
        passed &= type(fresh.get("all_finite")) is bool and type(reference.get("all_finite")) is bool
        passed &= fresh.get("all_finite") == reference.get("all_finite") is True
        passed &= type(fresh.get("boundary_violation_count")) is int
        passed &= type(reference.get("boundary_violation_count")) is int
        passed &= fresh.get("boundary_violation_count") == reference.get("boundary_violation_count") == 0
        for name in FLOAT_METRICS:
            left = fresh.get(name); right = reference.get(name)
            typed = type(left) is float and type(right) is float
            finite = typed and np.isfinite(left) and np.isfinite(right)
            delta = float(abs(left-right)) if finite else float("inf")
            close = bool(finite and np.isclose(left, right, rtol=PORTABILITY_RTOL, atol=PORTABILITY_ATOL))
            details["metric_max_abs"][f"{label}.{name}"] = delta
            details["metric_exact"][f"{label}.{name}"] = bool(typed and left == right)
            passed &= close
    details["metrics_max_abs"] = float(max(details["metric_max_abs"].values(), default=0.0))
    details["metrics_bitwise_exact"] = bool(all(details["metric_exact"].values()))
    details["pass"] = bool(passed)
    return details


def array_binding(fresh, reference, meshes):
    details = {"rtol": PORTABILITY_RTOL, "atol": PORTABILITY_ATOL,
               "array_max_abs": {}, "array_exact": {}}
    passed = True
    for n in meshes:
        for suffix in FIELD_ARRAY_SUFFIXES:
            fresh_key = f"initial_train_N{n}_{suffix}"
            reference_key = f"terminal_train_N{n}_{suffix}"
            left = np.asarray(fresh[fresh_key]); right = np.asarray(reference[reference_key])
            same_shape = left.shape == right.shape
            if suffix == "boundary_count":
                close = bool(same_shape and np.issubdtype(left.dtype, np.integer)
                             and np.issubdtype(right.dtype, np.integer)
                             and np.array_equal(left, right) and np.all(left == 0))
                delta = float(np.max(np.abs(left.astype(np.int64)-right.astype(np.int64)))) if same_shape and left.size else 0.0
            else:
                finite = bool(same_shape and np.issubdtype(left.dtype, np.floating)
                              and np.issubdtype(right.dtype, np.floating)
                              and np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
                close = bool(finite and np.allclose(left, right, rtol=PORTABILITY_RTOL, atol=PORTABILITY_ATOL))
                delta = float(np.max(np.abs(left-right))) if finite and left.size else (0.0 if finite else float("inf"))
            label = f"N{n}.{suffix}"
            details["array_max_abs"][label] = delta
            details["array_exact"][label] = bool(same_shape and np.array_equal(left, right))
            passed &= close
    details["arrays_max_abs"] = float(max(details["array_max_abs"].values(), default=0.0))
    details["arrays_bitwise_exact"] = bool(all(details["array_exact"].values()))
    details["pass"] = bool(passed)
    return details


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-dir"); parser.add_argument("--accepted-audit-dir")
    parser.add_argument("--r1-failed-dir")
    parser.add_argument("--failed-t1-dir"); parser.add_argument("--p5-json")
    parser.add_argument("--p7-npz"); parser.add_argument("--target-dir")
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--output-json", required=True); parser.add_argument("--output-npz", required=True)
    parser.add_argument("--progress-json", required=True); parser.add_argument("--work-checkpoint", required=True)
    parser.add_argument("--initial-control-npz", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def trace_storage(total, attempts):
    return {
        "q": np.empty((1,total,attempts+1,19),np.float64),
        "objective": np.empty((1,total,attempts+1),np.float64),
        "delta": np.empty((1,total,attempts+1),np.float64),
        "damping": np.empty((1,total,attempts+1),np.float64),
        "active": np.zeros((1,total,attempts+1),bool),
        "trial_objective": np.empty((1,total,attempts),np.float64),
        "predicted": np.empty((1,total,attempts),np.float64),
        "actual": np.empty((1,total,attempts),np.float64),
        "rho": np.empty((1,total,attempts),np.float64),
        "rho_defined": np.zeros((1,total,attempts),bool),
        "accepted": np.zeros((1,total,attempts),bool),
        "terminated": np.zeros((1,total,attempts),bool),
        "attempted": np.zeros((1,total,attempts),bool),
        "cg_iterations": np.zeros((1,total,attempts),np.int32),
        "cg_relative": np.empty((1,total,attempts),np.float64),
        "cg_breakdown": np.zeros((1,total,attempts),bool),
        "finite": np.zeros((1,total,attempts),bool),
        "step": np.empty((1,total,attempts,19),np.float64),
        "gradient": np.empty((1,total,attempts,19),np.float64),
        "jvp_count": np.zeros((1,total,attempts),np.int32),
        "vjp_count": np.zeros((1,total,attempts),np.int32),
        "bound_active_count": np.zeros((1,total,attempts),np.int8),
    }


def run_trust(datasets, generator, q0, objective0, mean, scales, attempts, args, started):
    total = len(q0); trace = trace_storage(total, attempts); fn = p9.make_trust_attempt(CONFIG)
    q=q0.copy(); objective=objective0.copy(); delta=np.full(total,p9.DELTA0)
    damping=np.full(total,p9.LAMBDA0); active=np.ones(total,bool)
    trace["q"][0,:,0]=q; trace["objective"][0,:,0]=objective
    trace["delta"][0,:,0]=delta; trace["damping"][0,:,0]=damping; trace["active"][0,:,0]=active
    names=("trial_objective","predicted","actual","rho","rho_defined","accepted","terminated",
           "cg_iterations","cg_relative","cg_breakdown","finite","step","gradient")
    for attempt in range(attempts):
        attempted=active.copy(); old_delta=delta.copy(); old_damping=damping.copy()
        for item in datasets:
            lo,hi=item["global_start"],item["global_stop"]
            for global_start in range(lo,hi,BATCH):
                global_stop=min(global_start+BATCH,hi); local_start=global_start-lo; local_stop=global_stop-lo
                sl=slice(global_start,global_stop); local=slice(local_start,local_stop)
                result=tuple(map(np.asarray,fn(generator,jnp.asarray(q[sl]),jnp.asarray(item["affine"][local]),
                    jnp.asarray(item["flat"][local]),jnp.asarray(item["coords"]),jnp.asarray(item["mask"]),
                    jnp.asarray(mean),jnp.asarray(scales),jnp.asarray(delta[sl]),jnp.asarray(damping[sl]),
                    jnp.asarray(active[sl]),jnp.asarray(objective[sl]))))
                nq,no,nd,nl,na,old,*values=result
                if not np.allclose(old,objective[sl],rtol=2e-13,atol=2e-14):
                    raise SystemExit("initial/current Cox objective mismatch")
                for name,value in zip(names,values): trace[name][0,sl,attempt]=value
                q[sl],objective[sl],delta[sl],damping[sl],active[sl]=nq,no,nd,nl,na
        trace["attempted"][0,:,attempt]=attempted
        trace["jvp_count"][0,:,attempt]=np.where(attempted,trace["cg_iterations"][0,:,attempt]+1,0)
        trace["vjp_count"][0,:,attempt]=np.where(attempted,trace["cg_iterations"][0,:,attempt]+1,0)
        trace["q"][0,:,attempt+1]=q; trace["objective"][0,:,attempt+1]=objective
        trace["delta"][0,:,attempt+1]=delta; trace["damping"][0,:,attempt+1]=damping; trace["active"][0,:,attempt+1]=active
        trace["bound_active_count"][0,:,attempt]=np.sum(np.abs(q)>=1.0,axis=1)
        if not (np.array_equal(old_delta,trace["delta"][0,:,attempt])
                and np.array_equal(old_damping,trace["damping"][0,:,attempt])):
            raise SystemExit("trust state persistence mismatch")
        p9.atomic_pickle(args.work_checkpoint,{"status":"in_progress","attempt_completed":attempt+1,
            "q":q,"objective":objective,"delta":delta,"damping":damping,"active":active,
            "optimizer_updates":0})
        p9.atomic_json(args.progress_json,{"status":"in_progress","stage":"trust",
            "attempt_completed":attempt+1,"attempt_cap":attempts,"active_count":int(np.sum(active)),
            "attempted_total":int(np.sum(trace["attempted"][:,:,:attempt+1])),
            "accepted_total":int(np.sum(trace["accepted"][:,:,:attempt+1])),
            "cg_breakdown_count":int(np.sum(trace["cg_breakdown"][:,:,:attempt+1]&trace["attempted"][:,:,:attempt+1])),
            "elapsed_s":float(time.perf_counter()-started),"scientific_metrics_exposed":False})
    exhaustion=(trace["active"][:,:,-1]&(trace["delta"][:,:,-1]<=p9.DELTA_MIN)
                &(trace["damping"][:,:,-1]>=p9.LAMBDA_MAX))
    return trace,exhaustion


def metric_gate(metrics):
    return p9.gate(metrics,2e-4,7e-4,True)


def main():
    args=parse_args(); c.require_gpu_highest(); started=time.perf_counter(); attempts=1 if args.smoke else ATTEMPTS
    legacy.SMOKE=args.smoke
    if args.smoke:
        accepted_bindings=recovery_bindings=failed_bindings=r1_failed_bindings={"mode":"synthetic"}
        accepted_report=None; train,coefficients,_,targets,regeneration=recovery.load_train(args,True)
        norm=p9.train_normalization(coefficients,legacy.concatenate(train,"features"))
        generator=p9.init_generator(CONFIG); qraw=np.zeros((len(coefficients),19),np.float64)
        source_checkpoint={"generator":generator,"q_raw":qraw,"normalization":{"mean":norm["mean"],"scales":norm["scales"]}}
        source_hash_before="synthetic"
    else:
        required=(args.recovery_dir,args.accepted_audit_dir,args.r1_failed_dir,args.failed_t1_dir,
                  args.p5_json,args.p7_npz,args.target_dir)
        if any(value is None for value in required): raise SystemExit("scientific P10-D requires exact dependency paths")
        recovery_bindings=exact_bundle(args.recovery_dir,EXPECTED_RECOVERY)
        accepted_bindings=exact_bundle(args.accepted_audit_dir,EXPECTED_ACCEPTED)
        r1_failed_bindings=validate_r1_failure(args.r1_failed_dir)
        failed_bindings=recovery.validate_failed(args.failed_t1_dir)
        accepted_report=load_json(os.path.join(args.accepted_audit_dir,"out/AUDIT.json"))
        if not (accepted_report["status"]=="pass" and accepted_report["decision"]["terminal_full_field_accepted"] is True
                and accepted_report["decision"]["capacity_retracted"] is True
                and accepted_report["decision"]["capacity_accepted"] is False
                and accepted_report["decision"]["g2_licensed"] is False):
            raise SystemExit("accepted terminal-field/capacity-retraction chain mismatch")
        train,coefficients,_,targets,regeneration=recovery.load_train(args,False)
        norm=p9.train_normalization(coefficients,legacy.concatenate(train,"features"))
        checkpoint_path=os.path.join(args.recovery_dir,"out/terminal_checkpoint.pkl")
        source_hash_before=c.sha256(checkpoint_path); source_checkpoint=pickle.load(open(checkpoint_path,"rb"))
        source_work=pickle.load(open(os.path.join(args.failed_t1_dir,"out/work_checkpoint.pkl"),"rb"))
        if not (recovery.tree_exact(source_checkpoint["generator"],source_work["generator"])
                and np.array_equal(source_checkpoint["q_raw"],source_work["q_raw"])
                and p9.tree_count(source_checkpoint["generator"])==30594
                and np.array_equal(source_checkpoint["normalization"]["mean"],norm["mean"])
                and np.array_equal(source_checkpoint["normalization"]["scales"],norm["scales"])):
            raise SystemExit("fixed accepted G1/q/normalization binding mismatch")
        generator=source_checkpoint["generator"]; qraw=np.asarray(source_checkpoint["q_raw"])
    generator_before=recovery.tree_sha(generator); qraw_before=p9.array_sha(qraw)
    q0=np.tanh(qraw); affine=legacy.concatenate(train,"affine"); states0=np.concatenate((affine,q0),axis=1)
    arrays={"training_affine":affine,"training_features":legacy.concatenate(train,"features"),
            "coefficient_mean":norm["mean"],"head_scales":norm["scales"],
            "normalization_source_indices":np.arange(len(coefficients),dtype=np.int32),
            "final_q_raw":qraw,"final_q_start":q0}
    p9.atomic_json(args.progress_json,{"status":"in_progress","stage":"initial_full_field",
        "optimizer_updates":0,"scientific_metrics_exposed":False})
    initial=p9.evaluate_full(train,generator,states0,norm["mean"],norm["scales"],CONFIG,arrays,"initial_train",True)
    recovery.add_snapshot_losses(initial,arrays,"initial_train",train)
    control_arrays={key:np.asarray(value) for key,value in arrays.items() if key.startswith("initial_train_")}
    control_arrays.update({"final_q_start":q0,"training_affine":affine})
    os.makedirs(os.path.dirname(os.path.abspath(args.initial_control_npz)),exist_ok=True)
    np.savez_compressed(args.initial_control_npz,**control_arrays)
    if args.smoke:
        accepted_terminal=initial
        reference_arrays={key.replace("initial_train_","terminal_train_",1):value
                          for key,value in control_arrays.items() if key.startswith("initial_train_")}
        accepted_source_exact=True
    else:
        recovery_report=load_json(os.path.join(args.recovery_dir,"out/phase9_terminal_recovery.json"))
        accepted_terminal=accepted_report["terminal_train"]
        accepted_source_exact=bool(recovery_report["terminal_train"]==accepted_terminal)
        with np.load(os.path.join(args.recovery_dir,"out/phase9_terminal_recovery.npz"),allow_pickle=False) as source:
            reference_arrays={name:np.asarray(source[name]) for name in source.files
                              if name.startswith("terminal_train_")}
    initial_metric_binding=metric_binding(initial,accepted_terminal)
    initial_array_binding=array_binding(arrays,reference_arrays,[x["N"] for x in train])
    initial_binding={"accepted_and_source_report_exact":accepted_source_exact,
        "metrics":initial_metric_binding,"arrays":initial_array_binding,
        "initial_control_sha256":c.sha256(args.initial_control_npz)}
    initial_binding["pass"]=bool(accepted_source_exact and initial_metric_binding["pass"]
                                  and initial_array_binding["pass"])
    if not initial_binding["pass"]:
        raise SystemExit("accepted terminal full-field failed portable initial-control binding")
    objective0=np.concatenate([arrays[f"initial_train_N{x['N']}_numerator"]/
        np.maximum(arrays[f"initial_train_N{x['N']}_denominator"],1e-300) for x in train])
    trace,exhaustion=run_trust(train,generator,q0,objective0,norm["mean"],norm["scales"],attempts,args,started)
    for name,value in trace.items(): arrays["trust_"+name]=value
    arrays["trust_unhealthy_exhaustion"]=exhaustion
    arrays["trust_chosen_start"]=np.zeros(len(q0),np.int8)
    arrays["trust_chosen_terminal_q"]=trace["q"][0,:,-1]
    terminal_q=arrays["trust_chosen_terminal_q"]; terminal_states=np.concatenate((affine,terminal_q),axis=1)
    terminal=p9.evaluate_full(train,generator,terminal_states,norm["mean"],norm["scales"],CONFIG,arrays,"recovered_train",True)
    recovery.add_snapshot_losses(terminal,arrays,"recovered_train",train)
    terminal_objective=np.concatenate([arrays[f"recovered_train_N{x['N']}_numerator"]/
        np.maximum(arrays[f"recovered_train_N{x['N']}_denominator"],1e-300) for x in train])
    objective_binding=bool(np.allclose(terminal_objective,trace["objective"][0,:,-1],rtol=2e-13,atol=2e-14))
    trace_health={"finite":bool(np.all(trace["finite"]|~trace["attempted"])),
        "no_breakdown":bool(not np.any(trace["cg_breakdown"]&trace["attempted"])),
        "breakdown_count":int(np.sum(trace["cg_breakdown"]&trace["attempted"])),
        "unhealthy_exhaustion_count":int(np.sum(exhaustion)),"attempted_total":int(np.sum(trace["attempted"])),
        "accepted_total":int(np.sum(trace["accepted"])),"terminated_total":int(np.sum(trace["terminated"])),
        "jvp_total":int(np.sum(trace["jvp_count"])),"vjp_total":int(np.sum(trace["vjp_count"])),
        "max_attempts":attempts}
    identity=bool(all(row["k3_cox_identity_worst"]<=p9.IDENTITY_TOL for row in
        [*initial["meshes"].values(),initial["pooled"],*terminal["meshes"].values(),terminal["pooled"]]))
    health=bool(initial_binding["pass"] and objective_binding and trace_health["finite"] and trace_health["no_breakdown"]
                and trace_health["unhealthy_exhaustion_count"]==0 and identity
                and all(row["boundary_violation_count"]==0 and row["all_finite"] for row in
                    [*initial["meshes"].values(),initial["pooled"],*terminal["meshes"].values(),terminal["pooled"]]))
    recovered_pass=bool(health and metric_gate(terminal))
    source_hash_after=("synthetic" if args.smoke else c.sha256(os.path.join(args.recovery_dir,"out/terminal_checkpoint.pkl")))
    immutable_model=bool(source_hash_before==source_hash_after and generator_before==recovery.tree_sha(generator)
                         and qraw_before==p9.array_sha(qraw))
    decision={"p10_d_valid":bool(not args.smoke and health and immutable_model),
        "fixed_g1_train_representable":bool(not args.smoke and health and immutable_model and recovered_pass),
        "corrected_g1_proposal_eligible":bool(not args.smoke and health and immutable_model and recovered_pass),
        "architecture_increase_licensed":False,"g2_licensed":False,"selection_evaluated":False,
        "optimizer_updates":0,"scientific_promotion_allowed":False,
        "next_action":("root audit of corrected-G1 proposal" if not args.smoke and recovered_pass
                       else "separate architecture-justification proposal; no architecture license" if not args.smoke and health
                       else "root audit; invalid diagnostic" if not args.smoke else "excluded smoke only")}
    arrays["optimizer_updates"]=np.asarray((0,),np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)),exist_ok=True); np.savez_compressed(args.output_npz,**arrays)
    p9.atomic_pickle(args.work_checkpoint,{"status":"complete","attempt_cap":attempts,
        "terminal_q":terminal_q,"terminal_objective":trace["objective"][0,:,-1],
        "source_checkpoint_sha256":source_hash_before,"generator_tree_sha256":generator_before,
        "optimizer_updates":0})
    p9.atomic_json(args.progress_json,{"status":"complete","stage":"complete","attempt_cap":attempts,
        "optimizer_updates":0,"scientific_metrics_exposed":False,"elapsed_s":float(time.perf_counter()-started)})
    report={"status":"excluded_execution_smoke" if args.smoke else "complete",
        "classification":"fixed accepted G1, final-q-only, full-train globalization diagnostic",
        "provenance":c.provenance(),"prereg_sha256":c.sha256(args.prereg),
        "bindings":{"recovery":recovery_bindings,"accepted_audit":accepted_bindings,
            "r1_failed":r1_failed_bindings,"failed_t1":failed_bindings},
        "config":{"generator":"fixed accepted G1","latent_dimension":19,"sole_start":"tanh(final_q_raw)",
            "objective":"exact discrete full-grid Cox FOM relative-L2-squared","train_mix":p9.TRAIN_MIX,
            "selection_touched":False,"optimizer_updates":0,"attempts":attempts,"cg_max":19,"cg_tolerance":1e-12,
            "delta0":p9.DELTA0,"delta_min":p9.DELTA_MIN,"delta_max":p9.DELTA_MAX,
            "lambda0":p9.LAMBDA0,"lambda_min":p9.LAMBDA_MIN,"lambda_max":p9.LAMBDA_MAX,
            "accept_rho":p9.ACCEPT_RHO,"K3_role":"initial/terminal identity only"},
        "data":{"training":legacy.metadata(train),"target_chunks":targets,"regeneration":regeneration},
        "normalization":{"mean_sha256":p9.array_sha(norm["mean"]),"scales_sha256":p9.array_sha(norm["scales"]),
            "normalized_coefficients_sha256":p9.array_sha(norm["normalized"]),"source":"train only"},
        "source_model":{"checkpoint_sha256_before":source_hash_before,"checkpoint_sha256_after":source_hash_after,
            "generator_tree_sha256_before":generator_before,"generator_tree_sha256_after":recovery.tree_sha(generator),
            "q_raw_sha256_before":qraw_before,"q_raw_sha256_after":p9.array_sha(qraw),"immutable":immutable_model,
            "generator_parameter_count":p9.tree_count(generator)},
        "initial_train":initial,"recovered_train":terminal,"trace_health":trace_health,
        "checks":{"accepted_initial_control":initial_binding,"terminal_objective_binding":objective_binding,
            "identity":identity,"health":health,"recovered_train_gate":recovered_pass},
        "information_boundary":{"train_only":True,"selection_touched":False,"model_validation_touched":False,
            "confirmation_touched":False,"weak_eq_touched":False,"scaling_touched":False,"capacity_used":False},
        "decision":decision,"npz":{"basename":os.path.basename(args.output_npz),"sha256":c.sha256(args.output_npz)},
        "initial_control":{"basename":os.path.basename(args.initial_control_npz),
            "sha256":c.sha256(args.initial_control_npz)},
        "progress":{"basename":os.path.basename(args.progress_json),"sha256":c.sha256(args.progress_json)},
        "work_checkpoint":{"basename":os.path.basename(args.work_checkpoint),"sha256":c.sha256(args.work_checkpoint)},
        "elapsed_s":float(time.perf_counter()-started)}
    p9.atomic_json(args.output_json,report)
    print(json.dumps({"status":report["status"],"trace_health":trace_health,"decision":decision},sort_keys=True),flush=True)
    print("ALL-DONE",flush=True)


if __name__=="__main__": main()
