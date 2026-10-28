"""Small exact training-state retention. This module never resumes or runs a fit."""
from pathlib import Path
import hashlib,json,resource,time
import numpy as np

class TrainingEvidence:
 def __init__(self,directory,configuration):
  self.path=Path(directory);self.path.mkdir(parents=True,exist_ok=False)
  self.configuration=dict(configuration,resume_implemented=False,resume_authorized=False,
   budget_policy='Any future resume must consume the remaining cumulative original step/time budget; retention authorizes no retry.',
   stochastic_state={'initialization_seed':27102027,'random_operations_after_initialization':False,
    'order':'all original points and nine training snapshots in immutable input order'})
  (self.path/'CONFIG.json').write_text(json.dumps(self.configuration,indent=2,allow_nan=False)+'\n')
  self.started=time.perf_counter();self.completed_steps=0;self.last_checkpoint=None
 def memory(self,jax):
  device=jax.devices()[0]
  stats=device.memory_stats()
  return {'host_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
   'device':str(device),'device_memory_stats':None if stats is None else
    {str(k):v.item() if isinstance(v,np.generic) else v for k,v in stats.items()}}
 def step(self,row):
  self.completed_steps=int(row['step'])
  with (self.path/'history.jsonl').open('a') as f:
   f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
 def checkpoint(self,params,optstate,jax,elapsed,reason):
  arrays={}
  for i,(w,b) in enumerate(params):arrays[f'W{i}']=np.asarray(w);arrays[f'b{i}']=np.asarray(b)
  leaves,structure=jax.tree_util.tree_flatten(optstate)
  for i,value in enumerate(leaves):arrays[f'optimizer_{i:04d}']=np.asarray(value)
  name=f'checkpoint-{self.completed_steps:06d}-{reason}';target=self.path/(name+'.npz')
  if target.exists():raise FileExistsError('unique optimizer checkpoint already exists')
  with target.open('xb') as f:np.savez_compressed(f,**arrays)
  with target.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  parameter_hash=hashlib.sha256()
  for key in sorted(k for k in arrays if not k.startswith('optimizer_')):
   array=np.ascontiguousarray(arrays[key]);parameter_hash.update(key.encode())
   parameter_hash.update(str((array.dtype.str,array.shape)).encode());parameter_hash.update(array.tobytes())
  record={'schema':'ns-obstacle-training-checkpoint-v1','arrays':target.name,'arrays_sha256':digest,
   'cumulative_completed_steps':self.completed_steps,'cumulative_training_seconds':float(elapsed),
   'step_cap':self.configuration['step_cap'],'time_cap_seconds':self.configuration['time_cap_seconds'],
   'optimizer_tree_structure':str(structure),'optimizer_leaf_count':len(leaves),
   'configuration_sha256':hashlib.sha256((self.path/'CONFIG.json').read_bytes()).hexdigest(),
   'precision':'float64','matmul_precision':'highest','reason':reason,'memory':self.memory(jax),
   'parameter_sha256':parameter_hash.hexdigest(),'runtime':{'jax':jax.__version__,'numpy':np.__version__},
   'resume_authorized':False,'state_semantics':'parameters and Adam state after completed updates; history losses precede their updates'}
  (self.path/(name+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
  self.last_checkpoint=record
  return record
 def finish(self,status,elapsed,error=None):
  record={'status':status,'cumulative_completed_steps':self.completed_steps,
   'cumulative_training_seconds':float(elapsed),'last_checkpoint':self.last_checkpoint,
   'error':error,'resume_authorized':False}
  (self.path/'FINAL.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
