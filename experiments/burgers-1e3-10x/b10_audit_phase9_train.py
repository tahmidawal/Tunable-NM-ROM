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


def schedule_check(arrays,smoke):
    if smoke: return True
    sizes={64:26112,128:6528,256:3264}
    for phase,row in p9.PHASES.items():
        expected=p9.schedule_permutations(phase,row["epochs"],sizes)
        for name,value in expected.items():
            if name not in arrays or not np.array_equal(arrays[name],value): return False
            n=int(name.rsplit("N",1)[1]); observed=np.asarray(arrays[name])
            if not np.array_equal(np.sort(observed),np.arange(sizes[n])): return False
    return True


def trace_check(arrays):
    if "trust_attempted" not in arrays: return {"present":False,"pass":True}
    attempted=np.asarray(arrays["trust_attempted"],bool); pred=np.asarray(arrays["trust_predicted"]); actual=np.asarray(arrays["trust_actual"])
    rho=np.asarray(arrays["trust_rho"]); defined=np.asarray(arrays["trust_rho_defined"],bool); accepted=np.asarray(arrays["trust_accepted"],bool)
    finite=np.asarray(arrays["trust_finite"],bool); breakdown=np.asarray(arrays["trust_cg_breakdown"],bool)
    positive=attempted&(pred>0); nonpositive=attempted&(pred<=0)
    checks={"attempt_work_finite":bool(np.all(finite[attempted]) and np.all(np.isfinite(pred[attempted])) and np.all(np.isfinite(actual[attempted])) and np.all(np.isfinite(rho[attempted]))),
            "rho_definition":bool(np.array_equal(defined[attempted],pred[attempted]>0)),
            "positive_rho_exact":bool(close(rho[positive],actual[positive]/pred[positive])) if np.any(positive) else True,
            "nonpositive_sentinel":bool(np.all(rho[nonpositive]==0)&~np.any(accepted[nonpositive])),
            "no_breakdown":bool(~np.any(breakdown[attempted])),
            "inactive_no_work":bool(np.all(np.asarray(arrays["trust_jvp_count"])[~attempted]==0)&np.all(np.asarray(arrays["trust_vjp_count"])[~attempted]==0))}
    checks["pass"]=bool(all(checks.values())); checks["present"]=True
    return checks


def capacity_check(report,arrays):
    if report.get("capacity") is None: return {"present":False,"pass":True,"license":False}
    summaries={"meshes":{}}
    for n in (64,128,256):
        values=[np.asarray(arrays[f"capacity_N{n}_{name}"]) for name in ("gamma","eta","residual","tangent","bound_fraction","cg_breakdown")]
        summaries["meshes"][str(n)]=p9.capacity_summary(*values)
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
    return {"present":True,"summary_match":summary_match,"criteria_match":criteria==report["capacity_license"]["criteria"],"improvement_match":improvement_match,
            "license":licensed,"license_match":licensed==report["decision"]["g2_licensed"],
            "pass":bool(summary_match and improvement_match and criteria==report["capacity_license"]["criteria"] and licensed==report["decision"]["g2_licensed"])}


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


def data_and_normalization_check(args,report,arrays,rows):
    if args.smoke: return {"pass":True,"mode":"synthetic"}
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
    coefficients=np.concatenate(coefficients); mean=np.mean(coefficients,axis=0,dtype=np.float64); centered=coefficients-mean
    scales=np.asarray((np.sqrt(np.mean(centered[:,:2304]**2,dtype=np.float64)),np.sqrt(np.mean(centered[:,2304:]**2,dtype=np.float64))))
    source=np.asarray(arrays["normalization_source_indices"])
    normalization=bool(next_index==35904 and np.array_equal(source,np.arange(35904)) and np.array_equal(mean,arrays["coefficient_mean"])
                       and np.array_equal(scales,arrays["head_scales"]) and p9.array_sha(mean)==report["normalization"]["mean_sha256"]
                       and p9.array_sha(scales)==report["normalization"]["scales_sha256"])
    expected_train=[(64,0,512,512,51,0,26112),(128,0,128,128,51,26112,32640),(256,0,64,64,51,32640,35904)]
    expected_selection=[(64,512,576,64,51,0,3264),(128,512,544,32,51,3264,4896),(256,512,528,16,51,4896,5712)]
    def tuples(metadata): return [(x["N"],x["source_start"],x["source_stop"],x["case_count"],x["num_times"],x["global_start"],x["global_stop"]) for x in metadata]
    metadata=bool(tuples(report["data"]["training"])==expected_train and tuples(report["data"]["selection"])==expected_selection
                  and all(np.isfinite(x["reference_health"]["reported_max_relative_residual"]) and x["reference_health"]["reported_max_relative_residual"]<=1e-8
                          and np.isfinite(x["reference_health"]["independent_max_relative_residual"]) and x["reference_health"]["independent_max_relative_residual"]<=1e-8
                          for x in report["data"]["training"]+report["data"]["selection"]))
    return {"target_files":bool(target_ok),"normalization":normalization,"metadata":metadata,"pass":bool(target_ok and normalization and metadata)}


def full_field_check(report,arrays,checkpoint,smoke):
    legacy.SMOKE=smoke
    train=(p9.p7train.smoke_datasets("train") if smoke else legacy.load_mix(p9.TRAIN_MIX,"train"))
    config=report["arm"]; states=np.concatenate((legacy.concatenate(train,"affine"),np.tanh(np.asarray(checkpoint["q_raw"]))),axis=1)
    recomputed={}; p9.evaluate_full(train,checkpoint["generator"],states,np.asarray(checkpoint["normalization"]["mean"]),
                                    np.asarray(checkpoint["normalization"]["scales"]),config,recomputed,"terminal_train",True)
    checks=[]
    for item in train:
        n=item["N"]
        for suffix in ("numerator","denominator","trajectory","boundary_count","identity"):
            checks.append(np.array_equal(np.asarray(recomputed[f"terminal_train_N{n}_{suffix}"]),np.asarray(arrays[f"terminal_train_N{n}_{suffix}"])))
    return {"terminal_train_exact_arrays":bool(all(checks)),"pass":bool(all(checks))}


def decision_check(report):
    decision=report["decision"]
    if not decision["updates_started"]:
        expected=bool(report["preflight"].get("weights_bitwise_unchanged") and
                      1.15*report["preflight"].get("projected_terminal_seconds",np.inf) <= 57600)
        return {"preflight_stop_honest":not expected,"pass":not expected}
    train=p9.gate(report["terminal_train"],2e-4,7e-4,True)
    selection=False
    if train:
        row=report["selection"]
        selection=bool(p9.gate(row["direct"],3e-4,1e-3,True) and p9.gate(row["oracle"],2e-4,7e-4,True)
                       and all(x<=1.5 for x in row["direct_oracle_mean_ratio"].values()) and all(row["trust_health"].values()))
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
    source_binding=bool(args.smoke or all(rows.get("code/"+name)==digest for name,digest in report["provenance"].get("source_sha256",{}).items()))
    dependency_binding=True
    phase_names={"p4":"phase4_d","p5":"phase5_d","p6":"phase6_d","p7":"phase7_train","p8":"phase8_d"}
    if not args.smoke:
        for phase,stem in phase_names.items():
            for kind in ("json","npz","audit","manifest"):
                key=f"{phase}_{kind}"; basename=("AUDIT.json" if kind=="audit" else "MANIFEST.sha256" if kind=="manifest" else f"{stem}.{kind}")
                dependency_binding &= rows.get(f"code/deps/{phase}/{basename}")==report["bindings"][key]["sha256"]
        dependency_binding &= rows.get("code/deps/p7/checkpoint.pkl")==report["bindings"]["p7_checkpoint"]["sha256"]
    file_binding=(source_binding and dependency_binding and rows.get("code/b10_phase9_train.py") is not None and rows.get("code/PHASE-9-PRE-REGISTRATION.md")==c.sha256(args.prereg)
                  and report["npz"]["sha256"]==c.sha256(args.source_npz) and report["checkpoint"]["sha256"]==c.sha256(args.checkpoint))
    schedule=schedule_check(arrays,args.smoke); metrics=True
    for key,row in report.get("history",{}).items(): metrics &= metric_match(row["metrics"],arrays,key,False)
    if report.get("terminal_train") is not None: metrics &= metric_match(report["terminal_train"],arrays,"terminal_train",True)
    trust=trace_check(arrays); capacity=capacity_check(report,arrays); negative=negative_self_test(); data_check=data_and_normalization_check(args,report,arrays,rows)
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
    preflight_check=bool(preflight.get("weights_bitwise_unchanged") and (structural_stop or
                         (timing_shape and np.isfinite(preflight["projected_terminal_seconds"]))))
    structural=True
    if report["arm"]["q"]==32:
        panel=report.get("t2_structural_preflight") or {}; work=panel.get("work",{})
        records=panel.get("records",{}); per_case={method:[float(np.median([x["elapsed_s"] for x in records.get(method,[]) if x["case"]==case])) for case in range(4)] for method in ("fom","rom")}
        timing_finite=all(np.all(np.isfinite(x)) and np.all(np.asarray(x)>0) for x in per_case.values())
        speed=(float(np.median(per_case["fom"])/np.median(per_case["rom"])) if timing_finite else 0.0)
        ci=(p9.base.clustered_speedup_ci(per_case["fom"],per_case["rom"],20266100) if timing_finite else [0,0])
        structural_gate=bool(panel.get("fom_eligible") and panel.get("compiled_device_bytes",np.inf)<=20_000_000_000
            and all(x.get("relative_l2",np.inf)<=2e-14 and x.get("boundary") and x.get("finite") for x in panel.get("identity",[]))
            and work=={"cox_weak_evaluations":50,"k3_coefficient_grid_full_field_evaluations":51,"weak_jacobian_evaluations":0,"trial_evaluations":0,"failures":0}
            and panel.get("position_counts")=={"fom":[12,12],"rom":[12,12]}
            and close(speed,panel.get("paired_median_speedup",np.nan)) and close(ci,panel.get("clustered_speedup_ci",[]))
            and speed>=10 and ci[0]>=8)
        structural=bool(timing_finite and panel.get("pass") is structural_gate and (structural_gate or not report["decision"]["updates_started"]))
    decision=decision_check(report)
    health=bool(all(provenance.values()) and parameter_pass and file_binding and schedule and metrics and trust["pass"] and capacity["pass"] and negative["pass"] and data_check["pass"] and full_field["pass"] and boundary and fold and decision["pass"] and preflight_check and structural)
    result={"status":"pass" if health else "fail","negative_aware":True,"source_json_sha256":c.sha256(args.source_json),
            "source_npz_sha256":c.sha256(args.source_npz),"checkpoint_sha256":c.sha256(args.checkpoint),"manifest_sha256":c.sha256(args.manifest),
            "expected_commit":args.expected_commit,"expected_job":str(args.expected_job),"checks":{"provenance":provenance,"parameter_counts":parameter_pass,
            "file_binding":file_binding,"schedule":schedule,"metrics":metrics,"trust":trust,"capacity":capacity,"negative_self_test":negative,"data_normalization":data_check,"full_field":full_field,"information_boundary":boundary,"predictor_fold":fold,"decision":decision,"preflight":preflight_check,"t2_structural":structural},
            "decision":report["decision"]}
    p9.atomic_json(args.output,result)
    if not health: raise SystemExit("Phase9 independent audit failed")
    print(json.dumps({"status":"pass","decision":report["decision"]},sort_keys=True))


if __name__=="__main__": main()
