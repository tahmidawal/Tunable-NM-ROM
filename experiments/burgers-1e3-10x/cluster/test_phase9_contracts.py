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

clean_health={"finite":True,"breakdown_count":0,"no_breakdown":True,
              "unhealthy_exhaustion_count":0,"no_unhealthy_exhaustion":True,
              "defined_rho_match":True,"undefined_rho_zero":True,"undefined_never_accepted":True}
assert p9.trust_health_gate(clean_health)
for key in ("finite","no_breakdown","no_unhealthy_exhaustion","defined_rho_match","undefined_rho_zero","undefined_never_accepted"):
    corrupt=dict(clean_health); corrupt[key]=False; assert not p9.trust_health_gate(corrupt)

shape=(2,1,1); q=np.zeros((2,1,2,19)); q[:,:,1,0]=.1
fixture={"trust_attempted":np.ones(shape,bool),"trust_predicted":np.full(shape,.2),"trust_actual":np.full(shape,.1),
 "trust_rho":np.full(shape,.5),"trust_rho_defined":np.ones(shape,bool),"trust_accepted":np.ones(shape,bool),
 "trust_finite":np.ones(shape,bool),"trust_cg_breakdown":np.zeros(shape,bool),"trust_objective":np.asarray([[[1.,.8]],[[1.,.8]]]),
 "trust_trial_objective":np.full(shape,.8),"trust_q":q,"trust_step":np.pad(np.full((2,1,1,1),.1),((0,0),(0,0),(0,0),(0,18))),
 "trust_terminated":np.zeros(shape,bool),"trust_delta":np.full((2,1,2),.25),"trust_damping":np.full((2,1,2),1e-6),
 "trust_active":np.ones((2,1,2),bool),"trust_cg_iterations":np.ones(shape,np.int32),"trust_jvp_count":np.full(shape,2,np.int32),
 "trust_vjp_count":np.full(shape,2,np.int32),"trust_unhealthy_exhaustion":np.zeros((2,1),bool),
 "trust_chosen_start":np.zeros(1,np.int8),"trust_chosen_terminal_q":q[0,:,1]}
assert audit.trace_check(fixture)["pass"]
for key,index,value in (("trust_rho",(0,0,0),.6),("trust_q",(0,0,1,0),.2),("trust_jvp_count",(0,0,0),3),
                        ("trust_active",(0,0,1),False),("trust_cg_breakdown",(0,0,0),True)):
    corrupt={name:array.copy() for name,array in fixture.items()}; corrupt[key][index]=value; assert not audit.trace_check(corrupt)["pass"]
assert p9.capacity_summary(np.ones(2),np.ones(2),np.ones(2),np.ones(2),np.zeros(2),np.asarray((False,True)))["cg_breakdown_count"]==1
print("phase9_contracts=pass")
