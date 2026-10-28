"""Serialization replay of already retained native arrays; no native solve."""
from pathlib import Path
import json,tempfile
import numpy as np
from scipy.sparse import load_npz
from fe_adapter import TrainingFEAdapter,sparse_digest
root=Path(__file__).parent/'artifacts/native-block02/results/interface'
a=TrainingFEAdapter.__new__(TrainingFEAdapter)
a.metadata={'scope':'retained native serialization only','physical_admission':False}
a.replay_matrices={};a.replay_arrays={}
for name in ('velocity_scalar','pressure_scalar'):
 for metric in ('h1','mass'):a.replay_matrices[name+'_'+metric]=load_npz(root/(name+'_'+metric+'.npz'))
 with np.load(root/(name+'_eigen.npz'),allow_pickle=False) as z:a.replay_arrays[name+'_modes']=z['coefficients']
with tempfile.TemporaryDirectory() as directory:
 out=Path(directory);files,retention=a.export_replay(out)
 for name,matrix in a.replay_matrices.items():
  assert sparse_digest(matrix)==sparse_digest(load_npz(out/('replay_metric_'+name+'.npz')))
 for name,array in a.replay_arrays.items():
  with np.load(out/('replay_'+name+'.npz'),allow_pickle=False) as z:assert np.array_equal(array,z['values'])
 assert json.loads((out/'replay_build.json').read_text())==a.metadata
 print(json.dumps({'passed':True,'scope':a.metadata['scope'],'file_count':len(files),'retention':retention},indent=2))
