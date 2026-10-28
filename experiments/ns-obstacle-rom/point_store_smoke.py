"""Serialization replay of retained native block-control arrays; no FE/training."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,tempfile
import numpy as np
from point_store import write_volume,PointVolumeStore,sha

base=Path(__file__).parent/'artifacts/native-block02'
with np.load(base/'inputcase/full_rule.npz',allow_pickle=False) as d:
 volume={k:d[k].copy() for k in ('points','weights','jacobians')}
expected={}
for name,label in [('lift_u','vector_volume'),('pressure','scalar_volume')]:
 with np.load(base/('results/interface/'+label+'-dense.npz'),allow_pickle=False) as d:
  expected[name]={k:d[k].copy() for k in d.files}
adapter=SimpleNamespace(volume=volume,D=.1,coefficients={'lift_u':np.zeros(2), 'pressure':np.zeros(3)})
def replay(adapter,coefficients,vector,boundary,block_points):
 data=expected['lift_u' if vector else 'pressure']
 for start in range(0,len(volume['points']),block_points):
  end=min(len(volume['points']),start+block_points)
  yield start,end,data['values'][start:end],data['gradients'][start:end]
with tempfile.TemporaryDirectory() as root:
 path=Path(root)/'volume'
 with patch('fe_blocks.evaluated_blocks_batched',side_effect=replay):
  manifest=write_volume(adapter,path,block_points=4096)
 assert all(field['columns']==1 for field in manifest['fields'].values())
 store=PointVolumeStore(path,sha(path/'manifest.json'))
 count=0
 for start,end,block in store.iter_blocks():
  assert start==count
  for name in expected:
   assert np.array_equal(block[name],expected[name]['values'][start:end])
   assert np.array_equal(block[name+'_gradient_dimensionless'],expected[name]['gradients'][start:end]*.1)
  count=end
 assert count==len(volume['points'])
 print(json.dumps({'scope':'retained native arrays serialized through point-block schema',
  'all_original_points':count,'exact_fields_and_scaled_gradients':True,
  'compressed_bytes':manifest['actual_compressed_bytes'],'expanded_bytes':manifest['projected_expanded_bytes'],
  'physical_admission':False},indent=2))
