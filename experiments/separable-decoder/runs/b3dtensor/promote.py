#!/usr/bin/env python3
"""Capacity-pilot promotion (design r3, validation only): read the two pilot JSONs,
apply the predeclared rule, write runs/b3dtensor/promotion.json.  Rule: a
configuration PASSES if D3 passed, its M-stability tail/head <= 0.05, and D4
passed (mean <= 5e-2, worst <= 1.5e-1, pool ratio <= 1.5, doubling < 1e-2,
optimality <= 1e-6, oracle/POD-K <= 0.5).  Promote the SMALLER if it passes and
its D4 mean is within 1.2x of the larger's; else the larger if it passes; else
STOP (the test table stays closed)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "pilot33a/out/sep_b3d_pilot_n33_k16.json")))
B = json.load(open(os.path.join(HERE, "pilot33b/out/sep_b3d_pilot_n33_k32.json")))
def ok(r):
    g = r["gates"]; d3 = g.get("D3_rank_of_A", {}); d4 = g.get("D4_heldout_oracle_validation", {})
    return bool(d3.get("passed") and d3.get("M_stability_pass") and d4.get("passed")), d4.get("mean")
pa, ma = ok(A); pb, mb = ok(B)
if pa and (not pb or ma <= 1.2 * mb):
    choice = "a"
elif pb:
    choice = "b"
else:
    choice = None
out = dict(a=dict(passed=pa, d4_mean=ma, cfg={k: A["config"][k] for k in ("k", "r", "M", "m_nnls")}),
           b=dict(passed=pb, d4_mean=mb, cfg={k: B["config"][k] for k in ("k", "r", "M", "m_nnls")}),
           promoted=choice, rule="smaller if it passes and its D4 mean <= 1.2x the larger's; else larger if it passes; else stop")
json.dump(out, open(os.path.join(HERE, "promotion.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(0 if choice else 4)
