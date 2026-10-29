"""Mandatory training-only, zero-PDE-solve PCHIP coefficient baseline."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator


class CoefficientPCHIP:
    def __init__(self,training_re,training_coefficients,metadata):
        self.re=np.asarray(training_re,dtype=np.float64)
        self.coefficients=np.asarray(training_coefficients,dtype=np.float64)
        if self.re.shape!=(9,) or not np.array_equal(self.re,15+1.25*np.arange(9)):
            raise ValueError("only frozen nine-state training family is admitted")
        if self.coefficients.ndim!=2 or len(self.coefficients)!=len(self.re) or not np.isfinite(self.coefficients).all():
            raise ValueError("finite training coefficient matrix required")
        if metadata.get("split")!="training":
            raise ValueError("interpolator sees training states only")
        self.metadata=dict(metadata,method="componentwise PCHIP in Re",extrapolation=False,
                           coefficient_scope="POD velocity/pressure correction; shared Stokes lift added at output",
                           pde_solve=False,physical_admission=False)
        self.interpolator=PchipInterpolator(self.re,self.coefficients,axis=0,extrapolate=False)

    def evaluate(self,reynolds):
        if not self.re[0]<=reynolds<=self.re[-1]:
            raise ValueError("extrapolation is not released")
        return np.asarray(self.interpolator(reynolds),dtype=np.float64)
