"""Preserve both failed refinement trials; extend their unchanged limiting gates."""
import argparse
import json
from pathlib import Path
from fresh_fom import provenance
from fresh_verify import plane_pulse
from fresh_verify_refined import family_check


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    result={"provenance":provenance(),"previous_attempts":"verify01 and verify02 remain failed at their declared grids/gates; no thresholds changed."}
    if result["provenance"]["jax_backend"]!="gpu":
        raise RuntimeError("GPU required")
    result["original_plane_reflective"]=plane_pulse("reflective",(512,1024))
    smooth=[(.5,.49,.36,.36,1.,1.15,.5,-.5,.12,.12),(.48,.51,.42,.42,.9,.85,0.,0.,.16,.16)]
    result["smooth_family"],gates=family_check(smooth,(256,512),"gaussian_compact_fine",out)
    gates["original_plane_reflective_fine_order"]=result["original_plane_reflective"]["joint_orders"][-1]>=1.5
    differences=[row for row in result["smooth_family"] if row.get("coarse_n")==256]
    gates["smooth_256_reference_difference"]=max(row["max_energy_state_difference"] for row in differences)<=.015
    result["richardson_reference_error_estimates"]=[{"bc":row["bc"],"parameter_index":row["parameter_index"],"coarse_n":256,"fine_n":512,"maximum_difference":row["max_energy_state_difference"],"estimated_coarse_error_if_second_order":4*row["max_energy_state_difference"]/3} for row in differences]
    result["gates"],result["passed"]=gates,all(gates.values())
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"gates":gates,"passed":result["passed"]}),flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
