"""Synthetic rank/metric equivalence for full candidate TSQR, no native execution."""
import json
import numpy as np
from spaces import select_test_columns
from fe_blocks import select_test_columns_blocks
rng=np.random.default_rng(28102026);n=257
values=rng.normal(size=(n,2,8));gradients=rng.normal(size=(n,2,2,8));w=rng.uniform(.1,1.,n)
values[...,2]=values[...,0];gradients[...,2]=gradients[...,0]
_,_,dense=select_test_columns(values,w,6,gradients)
rows=[]
for size in (17,64,257):
 def blocks():
  for start in range(0,n,size):
   end=min(n,start+size);yield start,end,values[start:end],w[start:end],gradients[start:end]
 result=select_test_columns_blocks(blocks(),6,n,True)
 assert result['candidate_indices']==dense['candidate_indices']
 assert result['rejected_indices']==dense['rejected_indices']
 indices=result['candidate_indices'];tr=np.array(result['transform']);td=np.array(dense['transform'])
 gram=np.einsum('n,niv,niw->vw',w,values[...,indices],values[...,indices])+np.einsum('n,nijv,nijw->vw',w,gradients[...,indices],gradients[...,indices])
 gap=float(np.max(np.abs(tr.T@gram@tr-np.eye(6))))
 assert gap<1e-12
 # Both bases span the same selected space; cross Gram is orthogonal.
 cross=td.T@gram@tr;subspace=float(np.max(np.abs(cross.T@cross-np.eye(6))))
 assert subspace<1e-12
 rows.append({'block_points':size,'same_selected_and_rejected':True,'orthonormality_max_gap':gap,'subspace_max_gap':subspace})
scalar=rng.normal(size=(n,7));scalar[:,1]=scalar[:,0]
_,_,dense_scalar=select_test_columns(scalar,w,5)
scalar_rows=[]
for size in (13,257):
 def scalar_blocks():
  for start in range(0,n,size):
   end=min(n,start+size);yield start,end,scalar[start:end],w[start:end],None
 result=select_test_columns_blocks(scalar_blocks(),5,n,False)
 assert result['candidate_indices']==dense_scalar['candidate_indices']
 assert result['rejected_indices']==dense_scalar['rejected_indices']
 selected=scalar[:,result['candidate_indices']];tr=np.array(result['transform'])
 gram=selected.T@(w[:,None]*selected)
 gap=float(np.max(np.abs(tr.T@gram@tr-np.eye(5))))
 assert gap<1e-12
 scalar_rows.append({'block_points':size,'same_selected_and_rejected':True,'orthonormality_max_gap':gap})
try:
 select_test_columns_blocks(iter([(0,10,scalar[:10],w[:10],None)]),5,n,False)
except ValueError: pass
else: raise AssertionError('partial point set accepted')
print(json.dumps({'scope':'synthetic TSQR candidate reduction','native_executed':False,'physical_admission':False,'vector_H1_cases':rows,'scalar_L2_cases':scalar_rows,'incomplete_points_rejected':True},indent=2))
