"""Sub-minute permutation/mask algebra control; native equivalence remains separate."""
import numpy as np
from component_factor import ComponentFactorSolve
rng=np.random.default_rng(7128);a=rng.normal(size=(4,4));h=a.T@a+np.eye(4)
maps=np.array([[4,1,7,0],[6,3,5,2]],dtype=np.int64);sf=np.array([True,False,True,True]);vf=np.zeros(8,dtype=bool)
full=np.zeros((8,8))
for row in maps:full[np.ix_(row,row)]=h;vf[row]=sf
hs=h[np.ix_(sf,sf)];rhs=rng.normal(size=(vf.sum(),5))
solver=ComponentFactorSolve(lambda b:np.linalg.solve(hs,b),maps,vf,sf)
expected=np.linalg.solve(full[np.ix_(vf,vf)],rhs);actual=solver.solve(rhs)
assert np.max(abs(actual-expected))<1e-14
assert np.max(abs(solver.solve(rhs[:,0])-expected[:,0]))<1e-14
try:ComponentFactorSolve(lambda b:b,maps,np.ones(8,dtype=bool),sf)
except ValueError:pass
else:raise AssertionError('incompatible essential masks accepted')
print({'synthetic_only':True,'max_abs_gap':float(np.max(abs(actual-expected))),'physical_admission':False})
