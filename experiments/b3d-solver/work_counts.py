"""Exact logical call counts from frozen FS step records; counts precede compiler fusion/CSE."""
import numpy as np

def fixed_sweep(iterations,adaptive_steps,K,scheme):
    it=np.asarray(iterations,int);a=int(adaptive_steps);n=len(it)
    assert 0<=a<=n and np.all(it>=0) and np.all(it[a:]==K)
    ad=int(it[:a].sum());fixed=int(it[a:].sum())
    pairs=a+ad+(n-a)+fixed
    return dict(nonlinear_iterations=ad+fixed,dense_GN_matrix_builds=ad+fixed,linear_solves=ad+fixed,
        jacobian_contractions=1+ad+2*a+fixed,residual_jacobian_pairs=pairs,gradient_evaluations=pairs,
        predictor_residual_evaluations=3*n,residual_evaluations=3*n+pairs,
        history_F_components=n if scheme=='CN' else 0,
        count_kind='exact logical source-level calls before compiler fusion/CSE; certification excluded',
        derivation='FS initial contraction1; adaptive LM(it+1) contracts plus final carry contraction1 per adaptive step; fixedK contracts. Predictor3 residual calls/step. Full r/J and gradient pairs: adaptive(it+1), fixed(K+1).')

def quasi_newton(q,backtracks,reasons,batch_size=1):
    q=np.asarray(q,int);b=np.asarray(backtracks,int);reason=np.asarray(reasons,int)
    assert q.shape==b.shape==reason.shape
    if batch_size!=1 or np.any(reason==3):
        return dict(residual_evaluations=None,gradient_evaluations=None,advection_value_evaluations=None,
                    missing_counter_reason='invalid direction bodies or batched global while/branch counts are not recoverable from per-case q')
    n=len(q);Q=int(q.sum());B=int(b.sum());residuals=6*n+Q+B
    return dict(residual_evaluations=residuals,gradient_evaluations=n+Q,advection_value_evaluations=residuals+n+2,
                linear_solves=Q,count_kind='exact logical singleton source-level calls before compiler optimization; fresh certification excluded')
