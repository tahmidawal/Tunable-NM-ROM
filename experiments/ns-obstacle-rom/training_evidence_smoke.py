"""Array-only checkpoint serialization; no JAX execution, fitting or admission."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,tempfile
import numpy as np
from training_evidence import TrainingEvidence
class Device:
 def memory_stats(self):return {'bytes_in_use':1234,'peak_bytes_in_use':2345}
 def __str__(self):return 'synthetic serialization fixture; no GPU execution'
fake=SimpleNamespace(__version__='synthetic no JAX execution',devices=lambda:[Device()],
 tree_util=SimpleNamespace(tree_flatten=lambda state:(list(state),'synthetic optimizer tuple')))
with tempfile.TemporaryDirectory() as tmp:
 e=TrainingEvidence(Path(tmp)/'evidence',{'synthetic_only':True,'step_cap':3,'time_cap_seconds':2.})
 params=[(np.arange(6,dtype=np.float64).reshape(2,3),np.zeros(3))]
 state=(np.array(2,dtype=np.int32),np.ones((2,3)))
 e.step({'step':2,'pre_update_relative_squared_loss':.5,'elapsed_seconds':1.2})
 r=e.checkpoint(params,state,fake,1.3,'failed');e.finish('failed',1.4,'fixture error')
 assert r['cumulative_completed_steps']==2 and r['cumulative_training_seconds']>=1.3 and r['checkpoint_start_training_seconds']==1.3
 assert not r['resume_authorized'] and r['step_cap']==3 and r['time_cap_seconds']==2.
 archive=e.path/r['arrays'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['arrays_sha256']
 with np.load(archive,allow_pickle=False) as d:
  assert np.array_equal(d['W0'],params[0][0]) and np.array_equal(d['optimizer_0001'],state[1])
  assert d['optimizer_0000'].dtype==np.int32
 assert json.loads((e.path/'FINAL.json').read_text())['status']=='failed'
 try:TrainingEvidence(e.path,{'step_cap':3,'time_cap_seconds':2.})
 except FileExistsError:pass
 else:raise AssertionError('existing evidence must not restart/reset a fit')
 failed=TrainingEvidence(Path(tmp)/'failed_setup',{'synthetic_only':True,'step_cap':3,'time_cap_seconds':2.})
 def setup_failure():raise RuntimeError('original setup failure')
 try:failed.guarded(setup_failure)
 except RuntimeError as exc:assert str(exc)=='original setup failure'
 else:raise AssertionError('original failure must propagate')
 final=json.loads((failed.path/'FINAL.json').read_text());assert final['status']=='setup_failed'
 assert 'original setup failure' in final['error'] and final['cumulative_completed_steps']==0
 failed.finish=lambda *args,**kwargs:(_ for _ in ()).throw(OSError('retention failure'))
 try:failed.guarded(setup_failure)
 except RuntimeError as exc:assert str(exc)=='original setup failure'
 else:raise AssertionError('retention failure must not mask original error')
 print('PASS exact parameter/optimizer retention, cumulative budgets, failure state; no training or resume')
