#!/usr/bin/env python
"""Independent, negative-aware audit for one immutable Phase-9 training arm."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re

import numpy as np

import b10_common as c
import b10_phase9_train as p9
import b10_spline_train as legacy


def load_json(path):
    with open(path,encoding="utf-8") as handle: return json.load(handle)


def close(a,b,rtol=2e-13,atol=2e-14):
    return bool(np.allclose(np.asarray(a),np.asarray(b),rtol=rtol,atol=atol,equal_nan=False))


def manifest(path):
    rows={}
    with open(path,encoding="utf-8") as handle:
        for line in handle:
            digest,name=line.rstrip().split("  ",1); rows[name.removeprefix("./")]=digest
    return rows


def metric_from_arrays(arrays,prefix,n):
    num=np.asarray(arrays[f"{prefix}_N{n}_numerator"],np.float64)
    den=np.asarray(arrays[f"{prefix}_N{n}_denominator"],np.float64)
    trajectory=np.asarray(arrays[f"{prefix}_N{n}_trajectory"],np.float64)
    boundary=np.asarray(arrays[f"{prefix}_N{n}_boundary_count"],np.int64)
    identity=np.asarray(arrays[f"{prefix}_N{n}_identity"],np.float64)
    return {"mean":float(np.mean(trajectory)),"worst":float(np.max(trajectory)),
            "loss":float(np.mean(num/np.maximum(den,1e-300))),"boundary":int(np.sum(boundary)),
            "identity":float(np.max(identity)),"finite":bool(all(np.all(np.isfinite(x)) for x in (num,den,trajectory,identity)))}


def metric_match(report,arrays,prefix,terminal):
    rows=[]; pooled_traj=[]; pooled_num=[]; pooled_den=[]; pooled_boundary=[]; pooled_identity=[]
    for n in map(int,report["meshes"]):
        one=metric_from_arrays(arrays,prefix,n); row=report["meshes"][str(n)]
        rows.append(close(one["mean"],row["trajectory_error_mean"]) and close(one["worst"],row["trajectory_error_worst"])
                    and one["boundary"]==row["boundary_violation_count"] and one["finite"]==row["all_finite"]
                    and (not terminal or close(one["identity"],row["k3_cox_identity_worst"])))
        pooled_traj.extend(np.asarray(arrays[f"{prefix}_N{n}_trajectory"])); pooled_num.extend(np.asarray(arrays[f"{prefix}_N{n}_numerator"]));
        pooled_den.extend(np.asarray(arrays[f"{prefix}_N{n}_denominator"])); pooled_boundary.extend(np.asarray(arrays[f"{prefix}_N{n}_boundary_count"])); pooled_identity.extend(np.asarray(arrays[f"{prefix}_N{n}_identity"]))
    pooled=report["pooled"]
    rows += [close(np.mean(pooled_traj),pooled["trajectory_error_mean"]),close(np.max(pooled_traj),pooled["trajectory_error_worst"]),
             close(np.mean(np.asarray(pooled_num)/np.maximum(pooled_den,1e-300)),pooled["mean_snapshot_relative_l2_squared"]),
             int(np.sum(pooled_boundary))==pooled["boundary_violation_count"],
             (not terminal or close(np.max(pooled_identity),pooled["k3_cox_identity_worst"]))]
    return bool(all(rows))


def schedule_check(report,arrays,smoke):
    if smoke: return bool(report["schedule"]["total_update_count"]==3)
    sizes={64:26112,128:6528,256:3264}
    for phase,row in p9.PHASES.items():
        expected=p9.schedule_permutations(phase,row["epochs"],sizes)
        for name,value in expected.items():
            if name not in arrays or not np.array_equal(arrays[name],value): return False
            n=int(name.rsplit("N",1)[1]); observed=np.asarray(arrays[name])
            if not np.array_equal(np.sort(observed),np.arange(sizes[n])): return False
    expected_counts={"encoder":88128,"joint":264384,"predictor":176256}
    if report["schedule"]!={"phase_order":["encoder","joint","predictor"],"epochs":{"encoder":9,"joint":27,"predictor":18},
            "resolution_cycle":[64,128,256],"batches":{"64":8,"128":2,"256":1},"batches_per_N_epoch":3264,
            "phase_update_counts":expected_counts,"total_update_count":528768,"terminal_epochs":{"encoder":9,"joint":27,"predictor":18},"terminal_only":True}:
        return False
    if not report["decision"]["updates_started"]:
        return bool(not report["history"] and "update_resolution_order" not in arrays)
    expected_order=np.tile(np.asarray((64,128,256),np.int16),54*3264)
    if not np.array_equal(arrays["update_resolution_order"],expected_order): return False
    terminals={"encoder":88128,"joint":352512,"predictor":528768}
    for phase,epochs in (("encoder",9),("joint",27),("predictor",18)):
        rows=[report["history"].get(f"{phase}_epoch{epoch}") for epoch in range(1,epochs+1)]
        if any(row is None for row in rows) or rows[-1]["terminal_update"]!=terminals[phase]: return False
        base=0 if phase=="encoder" else 88128 if phase=="joint" else 352512
        if any(row["terminal_update"]!=base+epoch*9792 for epoch,row in enumerate(rows,1)): return False
    return True


def trace_check(arrays):
    if "trust_attempted" not in arrays: return {"present":False,"pass":True}
    attempted=np.asarray(arrays["trust_attempted"],bool); pred=np.asarray(arrays["trust_predicted"]); actual=np.asarray(arrays["trust_actual"])
    rho=np.asarray(arrays["trust_rho"]); defined=np.asarray(arrays["trust_rho_defined"],bool); accepted=np.asarray(arrays["trust_accepted"],bool)
    finite=np.asarray(arrays["trust_finite"],bool); breakdown=np.asarray(arrays["trust_cg_breakdown"],bool)
    q=np.asarray(arrays["trust_q"]); objective=np.asarray(arrays["trust_objective"]); trial=np.asarray(arrays["trust_trial_objective"])
    step=np.asarray(arrays["trust_step"]); terminated=np.asarray(arrays["trust_terminated"],bool)
    delta=np.asarray(arrays["trust_delta"]); damping=np.asarray(arrays["trust_damping"]); active=np.asarray(arrays["trust_active"],bool)
    iters=np.asarray(arrays["trust_cg_iterations"]); jvp=np.asarray(arrays["trust_jvp_count"]); vjp=np.asarray(arrays["trust_vjp_count"])
    positive=attempted&(pred>0); nonpositive=attempted&(pred<=0)
    expected_accept=attempted&finite&~breakdown&defined&(actual>0)&(rho>=p9.ACCEPT_RHO)
    expected_terminate=expected_accept&(((objective[:,:,:-1]-trial)/np.maximum(objective[:,:,:-1],1e-300)<=1e-12)
        | (np.linalg.norm(step,axis=-1)/(1+np.linalg.norm(q[:,:,:-1],axis=-1))<=1e-12))
    shrink=(~accepted)|(defined&(rho<.25)); expand=accepted&(rho>.75)&(np.linalg.norm(step,axis=-1)>=.9*delta[:,:,:-1]); improve=accepted&(rho>.75)
    update_delta=np.where(shrink,np.maximum(delta[:,:,:-1]/4,p9.DELTA_MIN),np.where(expand,np.minimum(2*delta[:,:,:-1],p9.DELTA_MAX),delta[:,:,:-1]))
    update_damping=np.where(shrink,np.minimum(10*damping[:,:,:-1],p9.LAMBDA_MAX),np.where(improve,np.maximum(damping[:,:,:-1]/3,p9.LAMBDA_MIN),damping[:,:,:-1]))
    expected_delta=np.where(attempted,update_delta,delta[:,:,:-1]); expected_damping=np.where(attempted,update_damping,damping[:,:,:-1])
    expected_q=np.where(accepted[...,None],q[:,:,:-1]+step,q[:,:,:-1]); expected_objective=np.where(accepted,trial,objective[:,:,:-1])
    exhaustion=active[:,:,-1]&(delta[:,:,-1]<=p9.DELTA_MIN)&(damping[:,:,-1]>=p9.LAMBDA_MAX)
    checks={"attempt_work_finite":bool(np.all(finite[attempted]) and np.all(np.isfinite(pred[attempted])) and np.all(np.isfinite(actual[attempted])) and np.all(np.isfinite(rho[attempted]))),
            "rho_definition":bool(np.array_equal(defined[attempted],pred[attempted]>0)),
            "positive_rho_exact":bool(close(rho[positive],actual[positive]/pred[positive])) if np.any(positive) else True,
            "nonpositive_sentinel":bool(np.all(rho[nonpositive]==0)&~np.any(accepted[nonpositive])),
            "no_breakdown":bool(~np.any(breakdown[attempted])),
            "attempt_active_exact":bool(np.array_equal(attempted,active[:,:,:-1])),
            "acceptance_exact":bool(np.array_equal(accepted,expected_accept)),"termination_exact":bool(np.array_equal(terminated,expected_terminate)),
            "q_transition":bool(close(q[:,:,1:],expected_q)),"objective_transition":bool(close(objective[:,:,1:],expected_objective)),
            "delta_transition":bool(close(delta[:,:,1:],expected_delta)),"damping_transition":bool(close(damping[:,:,1:],expected_damping)),
            "active_transition":bool(np.array_equal(active[:,:,1:],active[:,:,:-1]&~terminated)),
            "work_exact":bool(np.array_equal(jvp,np.where(attempted,iters+1,0)) and np.array_equal(vjp,np.where(attempted,iters+1,0))
                              and np.all(iters[attempted]>=0) and np.all(iters[attempted]<=q.shape[-1])),
            "inactive_no_work":bool(np.all(jvp[~attempted]==0)&np.all(vjp[~attempted]==0)),
            "terminal_exhaustion_exact":bool(np.array_equal(exhaustion,np.asarray(arrays["trust_unhealthy_exhaustion"],bool))),
            "no_unhealthy_exhaustion":bool(not np.any(exhaustion)),
            "terminal_choice_binding":bool(np.array_equal(np.asarray(arrays["trust_chosen_start"]),np.argmin(objective[:,:,-1],axis=0))
                 and close(np.asarray(arrays["trust_chosen_terminal_q"]),np.where(np.argmin(objective[:,:,-1],axis=0)[:,None]==0,q[0,:,-1],q[1,:,-1])))}
    checks["pass"]=bool(all(checks.values())); checks["present"]=True
    return checks


def capacity_check(report,arrays):
    if report.get("capacity") is None: return {"present":False,"pass":True,"license":False}
    summaries={"meshes":{}}; work_pass=True
    for n in (64,128,256):
        values=[np.asarray(arrays[f"capacity_N{n}_{name}"]) for name in ("gamma","eta","residual","tangent","bound_fraction","cg_breakdown")]
        summaries["meshes"][str(n)]=p9.capacity_summary(*values)
        delta=np.asarray(arrays[f"capacity_N{n}_delta"]); iters=np.asarray(arrays[f"capacity_N{n}_cg_iterations"]); relative=np.asarray(arrays[f"capacity_N{n}_cg_relative"])
        work_pass &= bool(delta.ndim==2 and delta.shape[1]==19 and np.all(np.isfinite(delta)) and np.all(np.isfinite(relative))
                          and np.all(relative>=0) and np.all(iters>=0) and np.all(iters<=38))
    pooled=[]
    for name in ("gamma","eta","residual","tangent","bound_fraction"):
        pooled.append(np.concatenate([np.asarray(arrays[f"capacity_N{n}_{name}"]) for n in (64,128,256)]))
    summaries["pooled"]=p9.capacity_summary(*pooled,np.asarray([],bool))
    summary_match=close(json.dumps(summaries,sort_keys=True),json.dumps(report["capacity"],sort_keys=True)) if False else summaries==report["capacity"]
    # Recompute the exact Boolean conjunction without trusting the driver's label.
    failing=report["capacity_license"]["failing_strata"]; applicable=sorted(set(failing+["pooled"]))
    improvements={}
    for key in ("64","128","256","pooled"):
        def loss(epoch):
            metric=report["history"][f"joint_epoch{epoch}"]["metrics"]
            return metric["pooled"]["mean_snapshot_relative_l2_squared"] if key=="pooled" else metric["meshes"][key]["mean_snapshot_relative_l2_squared"]
        improvements[key]=(loss(24)-loss(27))/max(loss(24),1e-300)
    criteria={}
    for key in ["64","128","256","pooled"]:
        row=summaries["pooled"] if key=="pooled" else summaries["meshes"][key]
        imp=improvements[key]
        criteria[key]={"late_improvement":bool(0<=imp<=.01),"gamma":bool(row["gamma_median"]<=1e-4 and row["gamma_p95"]<=1e-3),
                       "eta":bool(row["eta_median"]>=.9 and row["eta_p10"]>=.8 and row["eta_E"]>=.8),
                       "bound":bool(row["bound_component_fraction"]<=.01),"finite":bool(row["all_finite"] and row["cg_breakdown_count"]==0)}
    licensed=bool(failing and all(all(criteria[k].values()) for k in applicable))
    improvement_match=all(close(improvements[k],report["capacity_license"]["late_improvement_24_27"][k]) for k in improvements)
    return {"present":True,"summary_match":summary_match,"work_pass":work_pass,"criteria_match":criteria==report["capacity_license"]["criteria"],"improvement_match":improvement_match,
            "license":licensed,"license_match":licensed==report["decision"]["g2_licensed"],
            "pass":bool(summary_match and work_pass and improvement_match and criteria==report["capacity_license"]["criteria"] and licensed==report["decision"]["g2_licensed"])}


def negative_self_test():
    valid={"pred":np.asarray((1.,0.,-1.)),"actual":np.asarray((.5,.2,.1)),"rho":np.asarray((.5,0.,0.)),
           "defined":np.asarray((True,False,False)),"accepted":np.asarray((True,False,False))}
    def accepts(x):
        pos=x["pred"]>0
        return bool(np.array_equal(x["defined"],pos) and np.allclose(x["rho"][pos],x["actual"][pos]/x["pred"][pos])
                    and np.all(x["rho"][~pos]==0) and not np.any(x["accepted"][~pos])
                    and all(np.all(np.isfinite(x[k])) for k in ("pred","actual","rho")))
    corrupt=[]
    for key,value in (("rho",np.asarray((.5,1.,0.))),("defined",np.asarray((True,True,False))),
                      ("accepted",np.asarray((True,True,False))),("pred",np.asarray((1.,np.nan,-1.)))):
        row={k:v.copy() for k,v in valid.items()}; row[key]=value; corrupt.append(accepts(row))
    return {"valid_sentinel_accepted":accepts(valid),"corruptions_rejected":bool(not any(corrupt)),"pass":bool(accepts(valid) and not any(corrupt))}


def tree_exact(left,right):
    import jax
    a=jax.tree_util.tree_leaves(left); b=jax.tree_util.tree_leaves(right)
    return bool(len(a)==len(b) and all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(a,b)))


def optimizer_state_check(checkpoint):
    import collections
    import jax
    states=checkpoint.get("optimizer_states",{})
    if set(states)!={"encoder","joint","predictor"}: return False
    params={"encoder":{"generator":checkpoint["generator"],"encoder":checkpoint["encoder"]},
            "joint":{"generator":checkpoint["generator"],"q_raw":checkpoint["q_raw"]},
            "predictor":checkpoint["predictor"]}
    for phase in states:
        leaves=[np.asarray(x) for x in jax.tree_util.tree_leaves(states[phase])]
        if not leaves or not all(np.all(np.isfinite(x)) for x in leaves): return False
        have=collections.Counter(x.shape for x in leaves); need=collections.Counter(np.asarray(x).shape for x in jax.tree_util.tree_leaves(params[phase]))
        if any(have[shape] < 2*count for shape,count in need.items()): return False
    return True


def data_and_normalization_check(args,report,arrays,rows,checkpoint):
    if args.smoke:
        train=p9.p7train.smoke_datasets("train"); rng=np.random.default_rng(92011); coefficients=rng.normal(0,.05,(4,3328))
        features=legacy.concatenate(train,"features"); norm=p9.train_normalization(coefficients,features)
        handoff=p9.encode_batches(checkpoint["encoder"],norm["normalized"],report["arm"]); q=np.asarray(checkpoint["q_raw"])
        folded=legacy.fold_predictor_standardization(checkpoint["predictor"],norm["feature_mean"],norm["feature_scale"])
        state=bool(np.array_equal(handoff,checkpoint["encoder_handoff_q_raw"]) and np.array_equal(handoff,arrays["encoder_handoff_q_raw"])
            and np.array_equal(q,arrays["final_q_raw"]) and tree_exact(folded,checkpoint["folded_predictor"]) and optimizer_state_check(checkpoint))
        normalization=bool(np.array_equal(norm["mean"],arrays["coefficient_mean"]) and np.array_equal(norm["scales"],arrays["head_scales"])
            and np.array_equal(features,arrays["training_features"]) and np.array_equal(norm["feature_mean"],arrays["predictor_feature_mean"])
            and np.array_equal(norm["feature_scale"],arrays["predictor_feature_scale"]))
        return {"pass":bool(state and normalization),"mode":"synthetic","state_handoff":state,"normalization":normalization}
    p5=load_json(args.p5_json); coefficients=[]; next_index=0; target_ok=True
    for row in p5["train_targets"]["chunks"]:
        path=os.path.join(args.target_dir,row["basename"])
        target_ok &= (os.path.isfile(path) and c.sha256(path)==row["sha256"]
                      and rows.get("code/deps/p5/targets/"+row["basename"])==row["sha256"])
        with np.load(path,allow_pickle=False) as data:
            value=np.asarray(data["coefficients"],np.float64).reshape(-1,3328)
            indices=np.asarray(data["global_snapshot_index"],np.int64).reshape(-1)
            target_ok &= (len(value)==row["snapshot_count"] and np.array_equal(indices,np.arange(next_index,next_index+len(value)))
                          and np.all(data["healthy"]) and np.all(data["boundary"]) and np.max(data["normal"])<=1e-8
                          and np.max(data["pou"])==0 and np.max(data["support"])==32)
            coefficients.append(value); next_index += len(value)
    coefficients=np.concatenate(coefficients)
    legacy.SMOKE=False; regenerated_train=legacy.load_mix(p9.TRAIN_MIX,"train"); regenerated_selection=legacy.load_mix(p9.SELECTION_MIX,"selection")
    features=legacy.concatenate(regenerated_train,"features"); normalization_values=p9.train_normalization(coefficients,features)
    mean=normalization_values["mean"]; scales=normalization_values["scales"]
    source=np.asarray(arrays["normalization_source_indices"])
    normalization=bool(next_index==35904 and np.array_equal(source,np.arange(35904)) and np.array_equal(mean,arrays["coefficient_mean"])
                       and np.array_equal(scales,arrays["head_scales"]) and p9.array_sha(mean)==report["normalization"]["mean_sha256"]
                       and p9.array_sha(scales)==report["normalization"]["scales_sha256"]
                       and p9.array_sha(normalization_values["normalized"])==report["normalization"]["normalized_training_coefficients_sha256"]
                       and np.array_equal(features,arrays["training_features"]) and p9.array_sha(features)==report["normalization"]["training_features_sha256"]
                       and np.array_equal(normalization_values["feature_mean"],arrays["predictor_feature_mean"])
                       and np.array_equal(normalization_values["feature_scale"],arrays["predictor_feature_scale"])
                       and np.array_equal(normalization_values["feature_empirical"],arrays["predictor_feature_empirical_scale"])
                       and p9.array_sha(normalization_values["feature_mean"])==report["normalization"]["feature_mean_sha256"]
                       and p9.array_sha(normalization_values["feature_scale"])==report["normalization"]["feature_scale_sha256"]
                       and p9.array_sha(normalization_values["feature_empirical"])==report["normalization"]["feature_empirical_scale_sha256"])
    expected_train=[(64,0,512,512,51,0,26112),(128,0,128,128,51,26112,32640),(256,0,64,64,51,32640,35904)]
    expected_selection=[(64,512,576,64,51,0,3264),(128,512,544,32,51,3264,4896),(256,512,528,16,51,4896,5712)]
    def tuples(metadata): return [(x["N"],x["source_start"],x["source_stop"],x["case_count"],x["num_times"],x["global_start"],x["global_stop"]) for x in metadata]
    metadata=bool(tuples(report["data"]["training"])==expected_train and tuples(report["data"]["selection"])==expected_selection
                  and tuples(legacy.metadata(regenerated_train))==expected_train and tuples(legacy.metadata(regenerated_selection))==expected_selection
                  and all(np.isfinite(x["reference_health"]["reported_max_relative_residual"]) and x["reference_health"]["reported_max_relative_residual"]<=1e-8
                          and np.isfinite(x["reference_health"]["independent_max_relative_residual"]) and x["reference_health"]["independent_max_relative_residual"]<=1e-8
                          for x in report["data"]["training"]+report["data"]["selection"]))
    state=True
    if report["decision"]["updates_started"]:
        handoff=p9.encode_batches(checkpoint["encoder"],normalization_values["normalized"],report["arm"])
        q=np.asarray(checkpoint["q_raw"]); affine=legacy.concatenate(regenerated_train,"affine")
        folded=legacy.fold_predictor_standardization(checkpoint["predictor"],normalization_values["feature_mean"],normalization_values["feature_scale"])
        state=bool(np.array_equal(handoff,checkpoint["encoder_handoff_q_raw"]) and np.array_equal(handoff,arrays["encoder_handoff_q_raw"])
            and np.array_equal(q,arrays["final_q_raw"]) and np.array_equal(np.concatenate((affine,np.tanh(q)),axis=1),arrays["final_target_states"])
            and np.array_equal(arrays["final_target_states"],checkpoint["final_target_states"]) and tree_exact(folded,checkpoint["folded_predictor"])
            and optimizer_state_check(checkpoint))
    return {"target_files":bool(target_ok),"normalization":normalization,"metadata":metadata,"state_handoff":state,"pass":bool(target_ok and normalization and metadata and state)}


def full_field_check(report,arrays,checkpoint,smoke):
    legacy.SMOKE=smoke
    train=(p9.p7train.smoke_datasets("train") if smoke else legacy.load_mix(p9.TRAIN_MIX,"train"))
    config=report["arm"]; states=np.concatenate((legacy.concatenate(train,"affine"),np.tanh(np.asarray(checkpoint["q_raw"]))),axis=1)
    recomputed={}; p9.evaluate_full(train,checkpoint["generator"],states,np.asarray(checkpoint["normalization"]["mean"]),
                                    np.asarray(checkpoint["normalization"]["scales"]),config,recomputed,"terminal_train",not smoke)
    checks=[]
    for item in train:
        n=item["N"]
        for suffix in ("numerator","denominator","trajectory","boundary_count","identity"):
            checks.append(np.array_equal(np.asarray(recomputed[f"terminal_train_N{n}_{suffix}"]),np.asarray(arrays[f"terminal_train_N{n}_{suffix}"])))
    capacity=True
    if report.get("capacity") is not None:
        fresh={}; recomputed_capacity=p9.capacity_metrics(train,checkpoint["generator"],np.tanh(np.asarray(checkpoint["q_raw"])),
            np.asarray(checkpoint["normalization"]["mean"]),np.asarray(checkpoint["normalization"]["scales"]),fresh)
        for item in train:
            n=item["N"]
            for suffix in ("gamma","eta","residual","tangent","delta","cg_iterations","cg_relative","cg_breakdown","bound_fraction"):
                capacity &= close(fresh[f"capacity_N{n}_{suffix}"],arrays[f"capacity_N{n}_{suffix}"])
        capacity &= recomputed_capacity==report["capacity"]
    return {"terminal_train_exact_arrays":bool(all(checks)),"capacity_exact_arrays":bool(capacity),"pass":bool(all(checks) and capacity)}


def decision_check(report):
    decision=report["decision"]
    if not decision["updates_started"]:
        expected=bool(report["preflight"].get("weights_bitwise_unchanged") and report["preflight"].get("proceed_before_update1"))
        return {"preflight_stop_honest":not expected,"pass":not expected}
    train=p9.gate(report["terminal_train"],2e-4,7e-4,not report["status"].startswith("excluded_execution_smoke"))
    selection=False
    if train:
        row=report["selection"]
        selection=bool(p9.gate(row["direct"],3e-4,1e-3,True) and p9.gate(row["oracle"],2e-4,7e-4,True)
                       and all(x<=1.5 for x in row["direct_oracle_mean_ratio"].values()) and row["trust_gate"])
    checks={"train":train==decision["train_pass"],"selection_open_exact":decision["selection_evaluated"]==train,
            "phase9":decision["phase9_pass"]==selection,"promotion":decision["scientific_promotion_allowed"]==selection,
            "g2":decision["g2_licensed"]==bool((report.get("capacity_license") or {}).get("g2_licensed",False))}
    checks["pass"]=bool(all(checks.values())); return checks


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source-json",required=True); parser.add_argument("--source-npz",required=True)
    parser.add_argument("--checkpoint",required=True); parser.add_argument("--manifest",required=True); parser.add_argument("--prereg",required=True)
    parser.add_argument("--p5-json"); parser.add_argument("--target-dir")
    parser.add_argument("--expected-commit",required=True); parser.add_argument("--expected-job",required=True); parser.add_argument("--slurm-out",required=True); parser.add_argument("--slurm-err",required=True)
    parser.add_argument("--output",required=True); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    report=load_json(args.source_json); arrays=np.load(args.source_npz,allow_pickle=False); checkpoint=pickle.load(open(args.checkpoint,"rb"))
    rows=manifest(args.manifest); stdout=open(args.slurm_out,encoding="utf-8").read(); stderr=open(args.slurm_err,encoding="utf-8").read()
    arm=report["arm"]; leaves=lambda tree:int(sum(np.prod(x.shape) for x in __import__('jax').tree_util.tree_leaves(tree)))
    provenance={"commit":report["provenance"].get("commit")==args.expected_commit,"job":str(report["provenance"].get("slurm_job_id"))==str(args.expected_job),
                "gpu":report["provenance"].get("jax_backend")=="gpu" and (args.smoke or report["provenance"].get("gpu_kind")=="NVIDIA H200"),
                "precision":report["provenance"].get("x64") is True and report["provenance"].get("matmul_precision")=="highest",
                "logs":("jax_backend=gpu" in stdout and "ALL-DONE" in stdout and not re.search(r"(?i)(captured.*large.*constant|oom|out of memory|traceback|disk.*full)",stdout+stderr))}
    counts={"generator":leaves(checkpoint["generator"]),"encoder":leaves(checkpoint["encoder"]),"predictor":leaves(checkpoint["predictor"])}
    parameter_pass=counts==report["parameter_counts"]["expected"]==report["parameter_counts"]["reported"]
    work_binding=True
    if report.get("work_checkpoint") is not None:
        work_path=os.path.join(os.path.dirname(args.source_json),report["work_checkpoint"]["basename"])
        work=pickle.load(open(work_path,"rb")) if os.path.isfile(work_path) else {}
        work_binding=bool(os.path.isfile(work_path) and c.sha256(work_path)==report["work_checkpoint"]["sha256"]
                          and work.get("phase")=="predictor" and work.get("epoch")== (1 if args.smoke else 18)
                          and work.get("global_update")== (3 if args.smoke else 528768)
                          and all(np.all(np.isfinite(np.asarray(x))) for x in __import__('jax').tree_util.tree_leaves(work.get("optimizer_state",{}))))
    source_binding=bool(args.smoke or all(rows.get("code/"+name)==digest for name,digest in report["provenance"].get("source_sha256",{}).items()))
    dependency_binding=True
    phase_names={"p4":"phase4_d","p5":"phase5_d","p6":"phase6_d","p7":"phase7_train","p8":"phase8_d"}
    if not args.smoke:
        for phase,stem in phase_names.items():
            for kind in ("json","npz","audit","manifest"):
                key=f"{phase}_{kind}"; basename=("AUDIT.json" if kind=="audit" else "MANIFEST.sha256" if kind=="manifest" else f"{stem}.{kind}")
                dependency_binding &= rows.get(f"code/deps/{phase}/{basename}")==report["bindings"][key]["sha256"]
        dependency_binding &= rows.get("code/deps/p7/checkpoint.pkl")==report["bindings"]["p7_checkpoint"]["sha256"]
        if report["arm"]["q"]==32:
            for kind in ("json","npz","checkpoint","audit","manifest"):
                basename="AUDIT.json" if kind=="audit" else "MANIFEST.sha256" if kind=="manifest" else "checkpoint.pkl" if kind=="checkpoint" else f"phase9_train.{kind}"
                dependency_binding &= rows.get(f"code/deps/t1/{basename}")==report["bindings"][f"t1_{kind}"]["sha256"]
    file_binding=(source_binding and dependency_binding and rows.get("code/b10_phase9_train.py") is not None and rows.get("code/PHASE-9-PRE-REGISTRATION.md")==c.sha256(args.prereg)
                  and report["npz"]["sha256"]==c.sha256(args.source_npz) and report["checkpoint"]["sha256"]==c.sha256(args.checkpoint))
    schedule=schedule_check(report,arrays,args.smoke); metrics=True
    for key,row in report.get("history",{}).items(): metrics &= metric_match(row["metrics"],arrays,key,False)
    if report.get("terminal_train") is not None: metrics &= metric_match(report["terminal_train"],arrays,"terminal_train",not args.smoke)
    trust=trace_check(arrays); capacity=capacity_check(report,arrays); negative=negative_self_test(); data_check=data_and_normalization_check(args,report,arrays,rows,checkpoint)
    full_field=({"pass":True,"skipped":"pre-update infrastructure stop"} if report.get("terminal_train") is None
                else full_field_check(report,arrays,checkpoint,args.smoke))
    boundary=report["information_boundary"]=={"train_only_weights":True,"selection_target_coefficients_training_use":False,
        "model_validation_touched":False,"confirmation_touched":False,"weak_eq_touched":False,"scaling_touched":False}
    fold=bool(report.get("predictor_fold_identity") is None if not report["decision"]["updates_started"] else
              np.isfinite(report.get("predictor_fold_identity")) and report["predictor_fold_identity"]<=1e-12)
    preflight=report["preflight"]
    timing_shape=bool(args.smoke or (set(preflight["raw_seconds"])=={f"{phase}_N{n}" for phase in ("encoder","joint","predictor","evaluation","trust","capacity") for n in (64,128,256)}
                     and all(len(x)==10 and np.all(np.isfinite(x)) and np.all(np.asarray(x)>0) for x in preflight["raw_seconds"].values())))
    structural_stop=bool(report["arm"]["q"]==32 and not (report.get("t2_structural_preflight") or {}).get("pass"))
    if structural_stop:
        projection_exact=True
    elif args.smoke:
        projection_exact=True
    else:
        med=preflight["median_seconds"]
        updates=sum(med[f"{phase}_N{n}"]*3264*p9.PHASES[phase]["epochs"] for phase in p9.PHASES for n in (64,128,256))
        sizes={64:26112,128:6528,256:3264}; selects={64:3264,128:1632,256:816}
        evaluation=56*sum(int(np.ceil(sizes[n]/p9.BATCH_BY_N[n]))*med[f"evaluation_N{n}"] for n in sizes)
        trust_seconds=2*p9.TRUST_ATTEMPTS*sum(selects[n]*med[f"trust_N{n}"] for n in selects)
        capacity_seconds=sum(sizes[n]*med[f"capacity_N{n}"] for n in sizes)
        reserve=preflight["audit_regeneration_reserve_seconds"]
        projected=updates+evaluation+max(trust_seconds,2*capacity_seconds)+reserve
        remaining=max(0.,preflight["allocation_seconds"]-preflight["actual_elapsed_at_decision_seconds"])
        proceed=1.15*projected<=remaining
        projection_exact=bool(close(updates,preflight["projected_update_seconds"]) and close(evaluation,preflight["projected_evaluation_seconds"])
            and close(trust_seconds,preflight["projected_trust_seconds"]) and close(capacity_seconds,preflight["projected_capacity_seconds"])
            and close(projected,preflight["projected_terminal_seconds"]) and close(remaining,preflight["actual_remaining_at_decision_seconds"])
            and close(reserve,preflight["elapsed_before_preflight_seconds"]) and reserve>=0
            and close(1.15*projected,preflight["required_with_safety_seconds"]) and proceed==preflight["proceed_before_update1"])
    preflight_check=bool(preflight.get("weights_bitwise_unchanged") and projection_exact and (structural_stop or
                         (timing_shape and np.isfinite(preflight["projected_terminal_seconds"]))))
    structural=True
    if report["arm"]["q"]==32 and not args.smoke:
        panel=report.get("t2_structural_preflight") or {}; work=panel.get("work",{})
        records=panel.get("records",{}); per_case={method:[float(np.median([x["elapsed_s"] for x in records.get(method,[]) if x["case_index"]==case])) for case in range(4)] for method in ("fom","rom")}
        timing_finite=all(np.all(np.isfinite(x)) and np.all(np.asarray(x)>0) for x in per_case.values())
        speed=(float(np.median(per_case["fom"])/np.median(per_case["rom"])) if timing_finite else 0.0)
        ci=(p9.base.clustered_speedup_ci(per_case["fom"],per_case["rom"],20266100) if timing_finite else [0,0])
        summaries={method:p9.base.summarize_timing(rows,4) for method,rows in records.items()}
        positions={method:[sum(row["position"]==position for row in records[method])//4 for position in (0,1)] for method in ("fom","rom")}
        fom_eligible=bool(all(row["finite"] and row["breakdowns"]==0 and row["flags_nonzero"]==0 and row["max_returned_relative_residual"]<=p9.base.FOM_OUTER for row in records["fom"])
            and np.mean([row["trajectory_relative_l2"] for row in records["fom"]])<=1e-3 and np.max([row["trajectory_relative_l2"] for row in records["fom"]])<=3e-3)
        rom_identity=bool(all(row["identity_relative_l2"]<=2e-14 and row["exact_boundary"] and row["finite"]
            and row["cold_recovery_finite"] and row["cold_sample_count"]<=4096
            and row["cox_weak_evaluations"]==50 and row["k3_coefficient_grid_full_field_evaluations"]==51
            and row["weak_jacobian_evaluations"]==0 and row["trial_evaluations"]==0 and row["failures"]==0 for row in records["rom"]))
        structural_gate=bool(fom_eligible and panel.get("compiled_device_bytes",np.inf)<=20_000_000_000 and rom_identity
            and work=={"cox_weak_evaluations":50,"k3_coefficient_grid_full_field_evaluations":51,"weak_jacobian_evaluations":0,"trial_evaluations":0,"failures":0}
            and positions==panel.get("position_counts")=={"fom":[12,12],"rom":[12,12]} and panel.get("burn_count",0)>0
            and summaries==panel.get("summaries") and fom_eligible==panel.get("fom_eligible")
            and close(speed,panel.get("paired_median_speedup",np.nan)) and close(ci,panel.get("clustered_speedup_ci",[]))
            and speed>=10 and ci[0]>=8)
        structural=bool(timing_finite and panel.get("pass") is structural_gate and (structural_gate or not report["decision"]["updates_started"]))
    decision=decision_check(report)
    health=bool(all(provenance.values()) and parameter_pass and work_binding and file_binding and schedule and metrics and trust["pass"] and capacity["pass"] and negative["pass"] and data_check["pass"] and full_field["pass"] and boundary and fold and decision["pass"] and preflight_check and structural)
    result={"status":"pass" if health else "fail","negative_aware":True,"source_json_sha256":c.sha256(args.source_json),
            "source_npz_sha256":c.sha256(args.source_npz),"checkpoint_sha256":c.sha256(args.checkpoint),"manifest_sha256":c.sha256(args.manifest),
            "expected_commit":args.expected_commit,"expected_job":str(args.expected_job),"checks":{"provenance":provenance,"parameter_counts":parameter_pass,
            "file_binding":file_binding,"work_checkpoint_binding":work_binding,"schedule":schedule,"metrics":metrics,"trust":trust,"capacity":capacity,"negative_self_test":negative,"data_normalization":data_check,"full_field":full_field,"information_boundary":boundary,"predictor_fold":fold,"decision":decision,"preflight":preflight_check,"t2_structural":structural},
            "decision":report["decision"]}
    p9.atomic_json(args.output,result)
    if not health: raise SystemExit("Phase9 independent audit failed")
    print(json.dumps({"status":"pass","decision":report["decision"]},sort_keys=True))


if __name__=="__main__": main()
