"""Preserve failed trials, extend unchanged gates and certify declared coverage."""
import argparse
import json
from pathlib import Path
import numpy as np
from fresh_fom import Grid,localized_initial,provenance
from fresh_verify import plane_pulse
from fresh_verify_refined import family_check,sine_reference,fft_self_check
from fresh_learning import parameter_rows


def edge_energy(u,v,c,n):
    padded=np.pad(u,1)
    return .5*(np.sum(v*v)/n**2+c*c*(np.sum(np.diff(padded,axis=0)**2)+np.sum(np.diff(padded,axis=1)**2)))


def independent_sine_audit(parameters):
    rows=[]
    for pi,p in enumerate(parameters):
        print("independent continuum sine",pi,flush=True)
        initial={n:tuple(np.asarray(x) for x in localized_initial(Grid(n),p)) for n in (256,512,1024)}
        e0=edge_energy(*initial[256],p[5],256)
        errors,spectral_errors=[],[]
        for ti in range(49):
            time=ti*.05
            ud,vd=sine_reference(*initial[256],p[5],time,256,semidiscrete=True)
            u512,v512=sine_reference(*initial[512],p[5],time,512)
            u1024,v1024=sine_reference(*initial[1024],p[5],time,1024)
            ur,vr=u1024[3:-1:4,3:-1:4],v1024[3:-1:4,3:-1:4]
            uhalf,vhalf=u512[1:-1:2,1:-1:2],v512[1:-1:2,1:-1:2]
            errors.append(float(np.sqrt(edge_energy(ud-ur,vd-vr,p[5],256)/e0)))
            spectral_errors.append(float(np.sqrt(edge_energy(uhalf-ur,vhalf-vr,p[5],256)/e0)))
        rows.append({"parameter_index":pi,"parameters":list(p),"discrete256_vs_continuum1024_error":errors,"spectral512_vs1024_error":spectral_errors,"max_discrete_reference_error":max(errors),"max_spectral_self_error":max(spectral_errors)})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    meta=provenance()
    result={"provenance":meta,"previous_attempts":"verify01 and verify02 remain failed at their declared grids/gates; no thresholds changed."}
    if meta["jax_backend"]!="gpu" or not meta["x64"] or meta["matmul_precision"]!="highest":
        raise RuntimeError("GPU/f64/highest required")
    result["original_plane_reflective"]=plane_pulse("reflective",(512,1024))
    extremes=[(.5,.49,.36,.36,1.,1.15,.5,-.5,.12,.12),(.48,.51,.42,.42,.9,.85,0.,0.,.16,.16),
              (.385,.615,.36,.36,1.3,1.15,.5,.5,.16,.16),(.405,.445,.38,.42,1.3,1.15,.5,-.5,.12,.16),
              (.445,.405,.42,.38,1.3,1.15,-.5,.5,.16,.12)]
    train=parameter_rows(690601,64)
    validation=parameter_rows(690602,16)
    # Predetermined bounded sample: first four train and first four validation.
    coverage=extremes+[tuple(x) for x in train[:4]]+[tuple(x) for x in validation[:4]]
    result["coverage"]={"extreme_count":len(extremes),"train_indices":[0,1,2,3],"validation_indices":[0,1,2,3],"claim":"Direct nested absorber refinement covers this predetermined sample, not every parameter in the continuous family."}
    result["smooth_family"],gates=family_check(coverage,(128,256,512),"gaussian_compact_coverage",out)
    gates["original_plane_reflective_fine_order"]=result["original_plane_reflective"]["joint_orders"][-1]>=1.5
    result["contraction_estimates"]=[]
    for bc in ("dirichlet","absorbing"):
        for pi in range(len(coverage)):
            pairs=[row for row in result["smooth_family"] if row.get("bc")==bc and row.get("parameter_index")==pi and "coarse_n" in row]
            coarse=next(row for row in pairs if row["coarse_n"]==128)["max_energy_state_difference"]
            fine=next(row for row in pairs if row["coarse_n"]==256)["max_energy_state_difference"]
            ratio=fine/coarse
            estimate=fine/(1-ratio) if ratio<1 else float("inf")
            result["contraction_estimates"].append({"bc":bc,"parameter_index":pi,"coarse_difference":coarse,"fine_difference":fine,"observed_ratio":ratio,"observed_order":float(-np.log2(ratio)),"conditional_256_error_estimate":estimate,"interpretation":"Conditional on observed contraction continuing; not a rigorous bound."})
            gates[f"{bc}_{pi}_fine_difference"]=fine<=.015
            gates[f"{bc}_{pi}_contracting_reference"]=ratio<.5 and estimate<=.02
    result["independent_sine_all_validation"]=independent_sine_audit([tuple(p) for p in validation])
    result["independent_sine_extremes"]=independent_sine_audit(extremes)
    sine=result["independent_sine_all_validation"]+result["independent_sine_extremes"]
    gates["direct_reflective_reference"]=max(r["max_discrete_reference_error"] for r in sine)<=.02
    gates["spectral_reference_self_resolution"]=max(r["max_spectral_self_error"] for r in sine)<=1e-3
    result["fft_self_check"]=fft_self_check([extremes[0],extremes[2],extremes[3]])
    for field in ("resolution_difference","domain_difference","resolution_l2_difference","domain_l2_difference","resolution_mean_difference","domain_mean_difference"):
        gates["fft_"+field]=max(r[field] for r in result["fft_self_check"])<1e-3
    result["gates"],result["passed"]=gates,all(gates.values())
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"gates":gates,"passed":result["passed"]}),flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
