"""Common physical metrics from retained fields; no reference pressure shift."""
from __future__ import annotations
import numpy as np
from spaces import positive_weights


def field_errors(velocity,pressure,reference_velocity,reference_pressure,weights,U=1.):
    u,p,ur,pr=[np.asarray(x,dtype=np.float64) for x in (velocity,pressure,reference_velocity,reference_pressure)]
    w=positive_weights(weights,len(u))
    if u.shape!=ur.shape or p.shape!=pr.shape or p.shape!=(len(u),) or U<=0:
        raise ValueError("incompatible field metric arrays")
    if not all(np.isfinite(x).all() for x in (u,p,ur,pr)):
        raise ValueError("nonfinite field metric input")
    volume=float(w.sum())
    un=float(np.sqrt(np.einsum("n,ni,ni->",w,u-ur,u-ur)))
    uden=max(float(np.sqrt(np.einsum("n,ni,ni->",w,ur,ur))),1e-12*U*np.sqrt(volume))
    pn=float(np.sqrt(w@((p-pr)**2)))
    pden=U**2*np.sqrt(volume)
    return {"velocity_numerator":un,"velocity_denominator":uden,"velocity_relative":un/uden,
            "pressure_numerator":pn,"pressure_denominator":pden,"pressure_scaled":pn/pden,
            "volume":volume,"U":U,"pressure_shift_applied":False}


def representation_admission(development_rows):
    if len(development_rows)!=2:
        raise ValueError("both frozen development cases required")
    ok=all(row["velocity_relative"]<=.002 and row["pressure_scaled"]<=.004 for row in development_rows)
    return {"passed":ok,"velocity_limit":.002,"pressure_limit":.004,
            "scope":"achieved development projection fields before online PDE solve",
            "force_or_divergence_admission":False,"rows":development_rows}
