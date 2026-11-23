"""Frozen new-cohort descriptor guard; no solver or truth generation here."""
import hashlib
import numpy as np
def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def regenerate(cfg,draw):
    assert cfg['seed_first_draw']==2026111902 and cfg['seed_first_count']==8
    assert cfg['case_ids']==list(range(8))
    fresh=draw(2026111902,8)
    development=np.concatenate((draw(7090702,4),draw(911702,2)))
    training=draw(0,576)
    overlaps={}
    for name,old in [('original_training_and_validation',training),('width_development',development)]:
        overlaps[name]=int(sum(np.any(np.all(np.isclose(old,row,rtol=0,atol=1e-14),axis=1)) for row in fresh))
    record=dict(cohort='fresh8',seed=2026111902,count=8,physical_parameters=fresh.tolist(),
        selected_case_ids=cfg['case_ids'],parameter_sha256=sha(fresh),
        nonoverlap_counts=overlaps,training_descriptor_recipe='params_draw(0,576); original sample_params sequential draw',
        training_descriptor_sha256=sha(training),development_descriptor_sha256=sha(development),
        guard_passed=all(v==0 for v in overlaps.values()))
    return fresh,record
