#!/usr/bin/env python
"""Excluded T2 structural execution smoke; synthetic N32/one step only."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if len(sys.argv)<6: sys.argv.extend(["synthetic"]*(6-len(sys.argv)))
import numpy as np
import b10_phase9_train as p9

result=p9.t2_structural_preflight(p9.ARMS["T2"],np.zeros(3328),np.ones(2),True)
identity=result["same_invocation_identity"]
assert result["scientific"] is False and result["pass"] is False
assert identity["finite"] and identity["boundary"] and identity["relative_l2"]<=p9.IDENTITY_TOL
assert 0 < result["compiled_device_bytes"] <= 20_000_000_000
print("phase9_t2_structural_smoke=pass",result["compiled_device_bytes"])
