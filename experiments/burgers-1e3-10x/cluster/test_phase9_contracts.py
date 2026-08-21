#!/usr/bin/env python
"""Fixed synthetic/static Phase-9 contract regressions; no locked data."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if len(sys.argv) < 6:
    sys.argv.extend(["synthetic"] * (6-len(sys.argv)))

import numpy as np
import b10_phase9_train as p9
import b10_audit_phase9_train as audit

for arm,config in p9.ARMS.items():
    assert p9.tree_count(p9.init_generator(config)) == config["generator_count"]
    assert p9.tree_count(p9.init_encoder(config)) == config["encoder_count"]
    assert p9.tree_count(p9.init_predictor(config)) == config["predictor_count"]

sizes={64:26112,128:6528,256:3264}
for phase,row in p9.PHASES.items():
    first=p9.schedule_permutations(phase,row["epochs"],sizes)
    second=p9.schedule_permutations(phase,row["epochs"],sizes)
    assert first.keys()==second.keys()
    for name,value in first.items():
        assert np.array_equal(value,second[name])
        n=int(name.rsplit("N",1)[1])
        assert np.array_equal(np.sort(value),np.arange(sizes[n]))

negative=audit.negative_self_test()
assert negative["pass"] and negative["valid_sentinel_accepted"] and negative["corruptions_rejected"]
print("phase9_contracts=pass")
