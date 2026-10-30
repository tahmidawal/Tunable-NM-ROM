from pathlib import Path
import io,json,hashlib,tarfile,time,sys
import numpy as np
B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B))
from accuracy_v2.spatial_bank import SpatialBank
from accuracy_v2.fast_spatial_bank import FastSpatialBank
out=Path(__file__).parent;ret=json.loads((B/'artifacts/neural-pod-guided-trace01/RETENTION.json').read_text());started=time.monotonic()
raw=b''.join((B/'artifacts/neural-pod-guided-trace01'/x['path']).read_bytes() for x in ret['chunks']);assert hashlib.sha256(raw).hexdigest()==ret['archive_sha256'];model_bytes=None
with tarfile.open(fileobj=io.BytesIO(raw),mode='r|gz') as tf:
 for m in tf:
  if m.name=='neural-pod-guided-trace01/results/fit/spatial-bank-raw.npz':model_bytes=tf.extractfile(m).read();break
assert hashlib.sha256(model_bytes).hexdigest()=='aad67694e5a2c11b7a736e8e013723e811f6093c2b553c097a13f2e4bc48d2d8'
model_path=out/'input-model.npz';model_path.write_bytes(model_bytes);del raw
old=SpatialBank.load(model_path);new=FastSpatialBank.load(model_path);rng=np.random.default_rng(271029);points=rng.uniform([0,0],[2.2,.41],size=(512,2));points=points[np.linalg.norm(points-[.2,.2],axis=1)>=.05][:256]
t=np.linspace(0,1,17);angles=np.arange(32)*2*np.pi/32
boundary=np.concatenate([np.c_[np.zeros(len(t)),.41*t],np.c_[2.2*t,np.zeros(len(t))],np.c_[2.2*t,np.full(len(t),.41)],np.c_[.2+.05*np.cos(angles),.2+.05*np.sin(angles)]])
points=np.concatenate([points,boundary]);rows=[];arrays={'points':points}
for rank in [1,2,4]:
 a,da=old.evaluate(points,rank);b,db=new.evaluate(points,rank)
 rows.append({'rank':rank,'value_max_absolute_gap':float(np.max(np.abs(a-b))),'gradient_max_absolute_gap':float(np.max(np.abs(da-db))),'passed':bool(np.allclose(a,b,rtol=1e-12,atol=1e-12) and np.allclose(da,db,rtol=1e-12,atol=1e-12))})
 arrays.update({f'old_values_{rank}':a,f'new_values_{rank}':b,f'old_gradient_{rank}':da,f'new_gradient_{rank}':db})
assert all(x['passed'] for x in rows);np.savez_compressed(out/'arrays.npz',**arrays)
result={'passed':True,'scope':'bounded actual-bank NumPy contraction equivalence only; no PDE solve, fit, native FE or performance claim','model_sha256':hashlib.sha256(model_bytes).hexdigest(),'archive_sha256':ret['archive_sha256'],'source_sha256':hashlib.sha256((B/'accuracy_v2/fast_spatial_bank.py').read_bytes()).hexdigest(),'original_source_sha256':hashlib.sha256((B/'accuracy_v2/spatial_bank.py').read_bytes()).hexdigest(),'rows':rows,'points':len(points),'boundary_points':len(boundary),'array_sha256':hashlib.sha256((out/'arrays.npz').read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-started,'dtype':'float64','JAX_training_path_modified':False}
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
