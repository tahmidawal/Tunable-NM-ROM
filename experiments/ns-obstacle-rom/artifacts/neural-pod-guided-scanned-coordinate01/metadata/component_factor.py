"""Two exact scalar solves in certified vector FE component coordinates."""
import numpy as np
from fe_algebra import array_digest

class ComponentFactorSolve:
 def __init__(self,scalar_solve,component_maps,vector_free,scalar_free):
  maps=np.asarray(component_maps);vf=np.asarray(vector_free);sf=np.asarray(scalar_free)
  if maps.dtype.kind not in 'iu' or maps.shape!=(2,len(sf)) or vf.dtype!=bool or sf.dtype!=bool:
   raise ValueError('actual integer component maps and Boolean masks required')
  if len(vf)!=2*len(sf) or not np.array_equal(np.sort(maps.ravel()),np.arange(len(vf))):
   raise ValueError('component maps must partition the vector coefficient layout')
  if any(not np.array_equal(vf[row],sf) for row in maps):raise ValueError('component essential masks differ')
  self.vector_ids=np.flatnonzero(vf);scalar_ids=np.flatnonzero(sf)
  position=np.full(len(vf),-1,dtype=np.int64);position[self.vector_ids]=np.arange(len(self.vector_ids))
  self.positions=position[maps[:,scalar_ids]]
  if not np.array_equal(np.sort(self.positions.ravel()),np.arange(len(self.vector_ids))):
   raise ValueError('free component positions do not cover vector free space')
  self.scalar_solve=scalar_solve
  self.evidence={'component_maps_sha256':array_digest(maps),'free_positions_sha256':array_digest(self.positions),
   'vector_free_dofs':len(self.vector_ids),'scalar_free_dofs':len(scalar_ids),
   'scalar_solve_calls_per_vector_rhs_block':2,
   'precondition':'caller must independently certify actual vector/scalar H1 block equality'}
 def solve(self,rhs):
  rhs=np.asarray(rhs)
  if rhs.dtype!=np.dtype('float64') or rhs.ndim not in (1,2) or rhs.shape[0]!=len(self.vector_ids) or not np.isfinite(rhs).all():
   raise ValueError('finite float64 vector free-space RHS required')
  out=np.empty_like(rhs)
  for positions in self.positions:
   values=np.asarray(self.scalar_solve(rhs[positions]))
   if values.dtype!=rhs.dtype or values.shape!=rhs[positions].shape or not np.isfinite(values).all():
    raise ValueError('scalar factor returned incompatible values')
   out[positions]=values
  return out
