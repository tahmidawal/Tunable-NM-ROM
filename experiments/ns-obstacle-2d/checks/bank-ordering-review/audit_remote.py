"""Bounded, read-only saved-bank ordering replay. Never trains or rewrites data."""
import os,sys,json,hashlib,time,signal,resource,traceback
from pathlib import Path
started=time.monotonic();report={'schema':'ns-obstacle-bank-ordering-replay-v1','job_id':'6634647','passed':False,'wall_limit_seconds':60,'address_space_limit_bytes':4*2**30,'scope':'read-only training-only saved-bank ordering; no optimizer/native FE/model rewrite'}
def timed_out(*_):raise TimeoutError('60-second ordering audit bound reached')
signal.signal(signal.SIGALRM,timed_out);signal.alarm(60)
resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
try:
 import numpy as np
 base=Path(sys.argv[1]);sys.path.insert(0,str(base))
 from bank import FrozenBank
 from spaces import ordered_transform,RANK_CUTOFF
 from train_bank import array_hash
 model=base/'results/fit/bank.npz';training=base/'inputs/prepared/training.npz'
 report.update(bank_sha256=sha(model),training_archive_sha256=sha(training),source_sha256={n:sha(base/n) for n in ('bank.py','geometry.py','spaces.py','train_bank.py','qualification.py')},source_ledger_sha256=sha(base/'SOURCE-SHA256SUMS'))
 assert report['bank_sha256']=='da32607494d7ddf8c1d7c231be7289488ff8cfb3a071a2ec58b369a9d936eacd'
 bank=FrozenBank.load(model)
 with np.load(training,allow_pickle=False) as z:x=z['points_physical'];w=z['weights_dimensionless'];y=z['velocity_corrections'];re=z['Re']
 assert x.shape==(1036497,2) and y.shape==(9,len(x),2) and w.shape==(len(x),)
 assert np.array_equal(re,np.arange(15,25.01,1.25)) and all(a.dtype==np.float64 and np.isfinite(a).all() for a in (x,w,y)) and np.all(w>0)
 h=array_hash(x,w,y);assert h=='6a65cbcd387c32b9ab70fc3956335a975e40b62646370fdf3413c7df4bfd8f30'==bank.metadata['training_array_sha256']
 assert bank.geometry.as_dict()=={'kind':'channel2d','inlet':0.0,'outlet':2.2,'center':(0.2,0.2),'obstacle_radius':0.05,'height':0.41,'pipe_radius':None}
 assert [(a.shape,b.shape) for a,b in bank.layers]==[((2,64),(64,)),((64,64),(64,)),((64,16),(16,))] and bank.transform.shape==(8,8)
 assert bank.geometry.fluid_mask(x).all()
 report.update(training_array_sha256=h,points=len(x),geometry=bank.geometry.as_dict(),layer_shapes=[list(a.shape) for a,b in bank.layers],original_ordering_metadata=bank.metadata['ordering'])
 raw=np.empty((len(x),2,8));offset,scale=bank.geometry.coordinate_scaling()
 for lo in range(0,len(x),4096):
  hi=min(len(x),lo+4096);v=(x[lo:hi]-offset)/scale
  for i,(a,b) in enumerate(bank.layers):
   v=v@a+b
   if i<len(bank.layers)-1:v=np.tanh(v)
  raw[lo:hi]=v.reshape(-1,2,8)*bank.geometry.envelope(x[lo:hi])[0][:,None,None]
 # Exact source path spot-check: same first complete block, including source transform.
 check=bank.evaluate(x[:4096])[0];source_gap=float(np.max(abs(check-raw[:4096]@bank.transform)));assert source_gap==0
 transform,diagnostics=ordered_transform(raw,y,w)
 sw=np.repeat(np.sqrt(w),2);a=raw.reshape(-1,8)*sw[:,None]
 saved=a@bank.transform;replayed=a@transform;gram=saved.T@saved;cross=saved.T@replayed
 delta=saved-replayed
 report.update(recovered_ordering=diagnostics,raw_rank_ratio=diagnostics['basis_singular_values'][-1]/diagnostics['basis_singular_values'][0],saved_transform=bank.transform.tolist(),replayed_transform=transform.tolist(),transform_relative_max_gap=float(np.max(abs(transform-bank.transform))/np.max(abs(bank.transform))),weighted_gram=gram.tolist(),weighted_gram_max_absolute_error=float(np.max(abs(gram-np.eye(8)))),weighted_cross_gram=cross.tolist(),ordered_field_relative_frobenius_gap=float(np.linalg.norm(delta)/np.linalg.norm(saved)),ordered_field_max_absolute_gap=float(np.max(abs(delta))),weighted_subspace_cosines=np.linalg.svd(cross,compute_uv=False).tolist(),source_first_block_max_absolute_gap=source_gap)
 # Recover SVD rotation/sign rule using QR of the same weighted full-rule bank.
 q,rf=np.linalg.qr(a,mode='reduced');rotation=rf@transform;pivots=np.argmax(abs(rotation),axis=0)
 importance=np.array(diagnostics['training_importance']);gaps=importance[:-1]-importance[1:]
 report.update(rotation=rotation.tolist(),sign_pivots=pivots.tolist(),sign_pivot_values=rotation[pivots,np.arange(8)].tolist(),relative_importance_gaps=(gaps/importance[0]).tolist(),exact_importance_ties=np.flatnonzero(gaps==0).tolist(),tie_policy='Existing ordered_transform fixes largest-coordinate sign; no added rotation rule for exactly tied singular values.')
 assert report['raw_rank_ratio']>RANK_CUTOFF
 assert np.all(rotation[pivots,np.arange(8)]>0)
 # Report numerical discrepancies without inventing a scientific accuracy threshold.
 report['passed']=True
except BaseException as e:
 report.update(exception=repr(e),traceback=traceback.format_exc())
finally:
 signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 print(json.dumps(report,indent=2,allow_nan=False));sys.exit(0 if report['passed'] else 1)
