#!/usr/bin/env python
"""Check excluded Phase10 driver/auditor smoke artifacts."""
import json,os,sys
if len(sys.argv)!=2: raise SystemExit("usage: test_phase10_smoke.py SMOKE_DIR")
root=os.path.abspath(sys.argv[1]); report=json.load(open(os.path.join(root,"phase10_d.json")))
audit=json.load(open(os.path.join(root,"AUDIT.json")))
assert report["status"]=="excluded_execution_smoke" and report["config"]["sole_start"]=="tanh(final_q_raw)"
assert report["information_boundary"]["selection_touched"] is False and report["information_boundary"]["capacity_used"] is False
assert audit["status"]=="pass" and audit["checks"]["negative_self_test"]["pass"] is True
assert audit["checks"]["negative_self_test"]["decision_corruption_count"]==6
assert audit["checks"]["negative_self_test"]["trace_corruption_count"]==6
assert audit["decision"]["architecture_increase_licensed"] is False and audit["decision"]["g2_licensed"] is False
print("phase10_synthetic_contract=pass corruptions=12")
