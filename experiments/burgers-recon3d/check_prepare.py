"""Small algebra/data-layout fixtures, intended inside reviewed GPU smoke."""
import json
import numpy as np
import prepare3d as P


def run():
    P.guard()
    lc,lf=4,64
    xc=P.C.interior_coords(lc+1);xf=P.C.interior_coords(lf+1)
    f=lambda x:np.prod(np.sin(np.pi*x*np.array([1,2,1])[None,:]),axis=1)
    u=P.sine_prolong(f(xc)[None,:],lc,lf)[0]
    assert np.max(np.abs(u-f(xf)))<1e-12
    ix,w=P.idw_map(xf)
    assert np.max(np.abs(w[:,0]-1))==0
    test=np.stack((np.arange(len(xf),dtype=float),np.sin(np.arange(len(xf)))))
    assert np.array_equal(P.apply_idw(test,ix,w),test)
    a=np.arange(36).reshape(9,4);b=-np.ones((3,4))
    v=P.SnapshotView(a,b)
    assert np.array_equal(v.rows(4,8),np.r_[a[4:6],b[:2]])
    print(json.dumps(dict(status='pass',sine_physical_amplitude=True,idw_coincident_exact=True,
                          snapshot_view_switch=True)),flush=True)

if __name__=='__main__':run()
