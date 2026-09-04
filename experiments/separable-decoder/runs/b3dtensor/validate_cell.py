#!/usr/bin/env python3
"""Cross-artifact contract validator (design r4 amendment 22 / code review r2 item 15): reads every
artifact of the cell and decides which claims the report may make.  Run before gen_tables.py."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
out = dict(phase0={}, promotion=None, per_N={}, C1=None, M1=None, claims={})
for p in sorted(glob.glob(os.path.join(HERE, "gates*/out/b3d_fom_gates_n*.json"))):
    r = json.load(open(p)); out["phase0"][str(r["config"]["N"])] = bool(r.get("complete"))
prom = load(os.path.join(HERE, "promotion.json")); out["promotion"] = prom
for p in sorted(glob.glob(os.path.join(HERE, "n*/out/sep_b3d_tensor_n*.json"))):
    if "withdrawn" in p: continue
    r = json.load(open(p)); N = str(r["config"]["N"]); pre = r.get("preconditions", {})
    cmp_ = r.get("comparison", {}).get("tensor_vs_full", {})
    out["per_N"][N] = dict(complete=bool(r.get("complete")), rows=pre.get("result_rows_allowed"),
                           E1=cmp_.get("E1_pass"), TR_concern=r.get("gates", {}).get("TR_candidate_path", {}).get("concern"),
                           bracket={a: e.get("bracket") for a, e in r.get("matched", {}).get("arms", {}).items()},
                           speed_win={a: e.get("speed_win") for a, e in r.get("matched", {}).get("arms", {}).items()},
                           A1=r.get("A1", {}).get("arms", {}).get("tensor"))
k = load(os.path.join(HERE, "kernels/out/sep_b3d_kernels.json")); out["C1"] = None if k is None else k.get("C1")
m = load(os.path.join(HERE, "micro129/out/sep_b3d_micro_n129.json")); out["M1"] = None if m is None else m.get("M1")
ok_common = bool(out["phase0"].get("33")) and bool(prom and prom.get("promoted"))
t = out["per_N"]
def row_ok(N, arm="tensor"):
    v = t.get(N); return bool(v and v["complete"] and (v["rows"] or {}).get(arm))
out["claims"] = dict(
    oracle_equivalent_all_N=bool(ok_common and all(row_ok(N) and t[N]["E1"] for N in ("33", "65", "129") if N in t) and len(t) == 3),
    kernel_flat=bool(out["C1"] and out["C1"].get("passed")),
    speed_win_129=bool(ok_common and row_ok("129") and t["129"]["bracket"].get("tensor") and t["129"]["speed_win"].get("tensor")),
    useful_accuracy_129=bool(row_ok("129") and t["129"]["A1"] and t["129"]["A1"]["within_3x"] and t["129"]["A1"]["useful"]),
    M1_passed=bool(out["M1"] and out["M1"].get("passed")), TR_caveats={N: v["TR_concern"] for N, v in t.items()})
out["positive_row"] = bool(out["claims"]["oracle_equivalent_all_N"] and out["claims"]["kernel_flat"] and out["claims"]["speed_win_129"]
                           and out["claims"]["useful_accuracy_129"] and out["claims"]["M1_passed"])
json.dump(out, open(os.path.join(HERE, "validation.json"), "w"), indent=1)
print(json.dumps(out["claims"], indent=1)); print("positive_row:", out["positive_row"])
