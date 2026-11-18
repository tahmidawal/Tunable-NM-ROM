from common import *
import data, references as R, fd_reference as F
from vendor import pscore as P

assert np.array_equal(data.parameters('train',8),data.parameters('train',64)[:8])
try:
    data.parameters('final_test')
    raise AssertionError('test seal failed')
except PermissionError: pass
xy,w=data.grid(32,True)
assert abs(np.sum(w)-1)<1e-14
assert abs(np.sum(w*xy[:,0]**2*xy[:,1]**3)-1/12)<1e-14
p=data.fixtures()[1];b=data.initial_coefficients(p,32)
ic=np.asarray(data.synth(jnp.asarray(b),jnp.asarray(xy)))
icerr=float(np.linalg.norm(ic-data.initial(xy,p))/np.linalg.norm(ic))
u=jnp.asarray(data.initial(data.grid(17)[0],p).reshape(17,17)[1:-1,1:-1])
np.testing.assert_allclose(F.dstn(F.dstn(u)),u,atol=1e-14)
spectral=R.trajectory(32,.00125)(jnp.asarray(b))
fd=F.make_solver(2,16)(u,data.NU,.0025,'cnr',1e-11,1e-11)
assert np.asarray(fd['out']).shape==(11,225)
assert int(fd['units'])==200
assert np.isfinite(spectral).all() and np.isfinite(fd['out']).all()
save(HERE/'smoke/reference.json',dict(meta=meta(),prefix_stable=True,final_seal=True,
     integration=True,dst_roundtrip=True,ic_relative=icerr,fd_snapshots=11,
     maximum_fd_residual=float(np.nanmax(fd['res'])),status='passed_interface_only'))
print('reference smoke passed',flush=True)
