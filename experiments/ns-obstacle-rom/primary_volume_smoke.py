"""Mocked native API: schema/hash/coverage checks only, never native evidence."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,tempfile
import numpy as np
from primary_volume import write_primary,PrimaryVolumeStore,materialize,FIELDS,VECTORS
from point_store import sha,write_volume
base=Path(__file__).parent/'artifacts/native-block02'
with np.load(base/'inputcase/full_rule.npz',allow_pickle=False) as z:
 volume={k:z[k] for k in ('points','weights','jacobians')}
data={}
for vector,label in ((True,'vector_volume'),(False,'scalar_volume')):
 with np.load(base/('results/interface/'+label+'-dense.npz'),allow_pickle=False) as z:data[vector]={k:z[k] for k in z.files}
mesh=SimpleNamespace(dim=2,ne=1,GetCurveOrder=lambda:2)
ng=SimpleNamespace(__version__='mock-only',Mesh=lambda path:mesh,VectorH1=lambda *a,**k:SimpleNamespace(ndof=2),H1=lambda *a,**k:SimpleNamespace(ndof=3))
a=SimpleNamespace(volume=volume,D=.1,U=.2,ng=ng,layout_sha256='a'*64,
 coefficients={name:np.zeros(2 if name in VECTORS else 3) for name in FIELDS})
def blocks(adapter,c,vector,boundary,block_points):
 for start in range(0,len(volume['points']),block_points):
  end=min(start+block_points,len(volume['points']))
  yield start,end,data[vector]['values'][start:end],data[vector]['gradients'][start:end]
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);case=root/'case';case.mkdir();(case/'mesh.vol.bin').write_bytes(b'mocked mesh, not native input')
 (case/'config.json').write_text(json.dumps({'D':.1,'U':.2,'geometry_order':2,'velocity_order':2,'pressure_order':1,'diagnostic_order':12}))
 (case/'fe_layout.json').write_text(json.dumps({'layout':{'mock':True},'sha256':'a'*64}))
 with patch('primary_volume.runtime_identity',return_value={'mock':True}),patch('fe_blocks.evaluated_blocks_batched',side_effect=blocks),patch('fe_adapter.layout_record',return_value=({'mock':True},'a'*64)),patch('fe_adapter.TrainingFEAdapter.mapped_rule',return_value=volume):
  original=root/'volume';write_volume(a,original)
  target=root/'primary';manifest=write_primary(a,target,case,original)
  store=PrimaryVolumeStore(target,sha(target/'manifest.json'),ng)
  count=0
  for start,end,arrays in store.iter_blocks():
   assert start==count
   for name in FIELDS:
    assert np.array_equal(arrays[name],data[name in VECTORS]['values'][start:end])
    assert np.array_equal(arrays[name+'_gradient_dimensionless'],.1*data[name in VECTORS]['gradients'][start:end])
   count=end
  assert count==len(volume['points']) and store.replay_accounting['iterations'][-1]['completed']
  restored=materialize(store,root/'restored')
  assert restored['manifest_sha256']==sha(original/'manifest.json')
  manifest['fields']['lift_u']['blocks'][0]['array_sha256']='0'*64
  (target/'manifest.json').write_text(json.dumps(manifest))
  bad=None
  try:
   bad=PrimaryVolumeStore(target,sha(target/'manifest.json'),ng)
   list(bad.iter_blocks())
  except ValueError:pass
  else:raise AssertionError('modified block fingerprint accepted')
  if bad is not None:assert not bad.replay_accounting['iterations'][-1]['completed']
 print(json.dumps({'passed':True,'scope':'mocked native API and retained-array serialization only','points':count,'native_replay_executed':False,'physical_admission':False},indent=2))
