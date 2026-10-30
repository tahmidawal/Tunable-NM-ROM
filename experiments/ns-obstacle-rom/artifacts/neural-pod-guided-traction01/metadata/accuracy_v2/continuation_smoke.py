"""Pure-array optimizer-state round trip, no training/admission/native work."""
from pathlib import Path
import tempfile,json,hashlib
import numpy as np
from accuracy_v2.continuation import freeze,load,sha
with tempfile.TemporaryDirectory() as directory:
 p=Path(directory);bindings={'unit_test':'optimizer state only'};(p/'CONFIG.json').write_text(json.dumps(bindings))
 rng=np.random.default_rng(123);rng.normal(size=12);rstate=rng.bit_generator.state
 for i in range(2):
  x=np.array([1.,2.])+i;y=2*x;np.savez_compressed(p/f'c{i}.npz',theta=x,gradient=y)
  (p/f'c{i}.json').write_text(json.dumps({'array_sha256':sha(p/f'c{i}.npz')}))
 a={'theta':x,'gradient':y,'adam_m':np.array([.1,.2]),'adam_v':np.array([.3,.4])}
 state={'phase':'lbfgs','curvature_state_pairs':[['c0','c1']],'best':{'objective':1.,'checkpoint':'c1'},'last_measured_checkpoint':'c1','rng_state':rstate,'adam_step':9,'completed_lbfgs_iterations':1,'elapsed_seconds':123.5,'serial':2}
 freeze(p,state,a);restored,arrays,pairs=load(p,bindings,sha(p/'CONTINUATION.json'))
 assert restored['adam_step']==9 and restored['completed_lbfgs_iterations']==1 and restored['elapsed_seconds']>=123.5
 for k in a:assert np.array_equal(a[k],arrays[k])
 assert np.array_equal(pairs[0][0],np.ones(2)) and np.array_equal(pairs[0][1],2*np.ones(2)) and pairs[0][2]==.25
 r=np.random.default_rng();r.bit_generator.state=restored['rng_state'];assert np.array_equal(r.integers(0,100,50),rng.integers(0,100,50))
 try:load(p,{'changed':True},sha(p/'CONTINUATION.json'))
 except ValueError:pass
 else:raise AssertionError('changed recipe accepted')
 (p/'c0.npz').write_bytes(b'corrupt')
 try:load(p,bindings,sha(p/'CONTINUATION.json'))
 except ValueError:pass
 else:raise AssertionError('changed history accepted')
 print(json.dumps({'passed':True,'restored_fields':['theta','gradient','Adam_m','Adam_v','RNG','phase','cumulative_time','step_counts','curvature_history','best_state'],'changed_recipe_and_history_rejected':True,'genuine_data_opened':False}))
