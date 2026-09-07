"""CPU finite-difference control for the independent MLP geometry audit."""
import unittest
import numpy as np
from audit_dynamics import geometry

class AuditGeometryTests(unittest.TestCase):
    def test_numpy_chain_rule_with_independent_finite_differences(self):
        rng=np.random.default_rng(7090722)
        p={'p/linear':rng.normal(size=(7,3))*.2,'p/bias':rng.normal(size=7)*.1,'frozen/output_scale':np.array(.3)}
        for name,nin,nout in [('l1',3,8),('l2',8,8),('out',8,7)]:
            p[f'p/{name}/w']=rng.normal(size=(nin,nout))*.2;p[f'p/{name}/b']=rng.normal(size=nout)*.1
        transform=rng.normal(size=(7,7));z=rng.normal(size=3);w=rng.normal(size=3)
        def forward(zz):
            h=zz
            for layer in ('l1','l2'):
                h=h@p[f'p/{layer}/w']+p[f'p/{layer}/b'];h=h/(1+np.exp(-h))
            return transform@(p['p/bias']+p['p/linear']@zz+.3*(h@p['p/out/w']+p['p/out/b']))
        a,b,j,curve=geometry(p,transform,z,w)
        np.testing.assert_allclose(a,forward(z),atol=1e-14)
        eps=1e-5
        fd=np.column_stack([(forward(z+eps*e)-forward(z-eps*e))/(2*eps) for e in np.eye(3)])
        np.testing.assert_allclose(j,fd,atol=1e-10)
        eps=1e-3
        second=(forward(z+eps*w)-2*forward(z)+forward(z-eps*w))/(eps*eps)
        np.testing.assert_allclose(curve,second,atol=1e-8)
        np.testing.assert_allclose(b,j@w,atol=1e-14)

if __name__=='__main__':unittest.main()
