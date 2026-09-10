"""Exact-objective QR NNLS parity on full rank and nearly dependent supports."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.quadrature import solve_nnls,nnls_diagnostics,fit_quadrature


def main():
    rng=np.random.default_rng(260910)
    A=rng.normal(size=(180,24)).astype(np.float64)
    matrices=[('full_rank',A),('near_dependent',np.column_stack((A[:,:20],A[:,:4]+1e-9*rng.normal(size=(180,4))))),
              ('rank_deficient',np.column_stack((A[:,:20],A[:,:4])))]
    for name,A in matrices:
        truth=rng.uniform(0.,1.,A.shape[1]);truth[::3]=0.
        b=A@truth+.02*rng.normal(size=A.shape[0])
        direct,_=solve_nnls(A,b,method='direct',maxiter=10000)
        qr,rn=solve_nnls(A,b,method='qr',maxiter=10000)
        dd=nnls_diagnostics(A,b,direct);dq=nnls_diagnostics(A,b,qr)
        assert np.min(qr)>=0 and np.isfinite(qr).all()
        np.testing.assert_allclose(A@direct,A@qr,atol=1e-7,rtol=1e-7)
        np.testing.assert_allclose(dd['objective'],dq['objective'],atol=1e-10,rtol=1e-8)
        np.testing.assert_allclose(rn,np.linalg.norm(A@qr-b),atol=1e-14)
        assert dq['scaled_kkt']<1e-10
        if name=='full_rank':np.testing.assert_allclose(direct,qr,atol=1e-10,rtol=1e-9)
        print(name,dd['objective'],dq['objective'],dq['scaled_kkt'])
    for action in (lambda:solve_nnls(np.ones((3,4)),np.ones(3),method='qr'),
                   lambda:fit_quadrature(np.ones((3,8)),np.ones(3),4,nnls_method='qr')):
        try:action()
        except ValueError:pass
        else:raise AssertionError('wide QR support was not rejected')
    print('PASS original objective, fitted values, nonnegativity and KKT; no support truncation or Gram system')


if __name__=='__main__':main()
