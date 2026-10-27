"""Bounded numerical identity for archived POD replay; no arrays are changed."""
import numpy as np

FIELD_TOL=1e-12
ERROR_TOL=1e-11

def field_distance(a,b,gram,n0):
    diff=np.asarray(a)-np.asarray(b)
    assert diff.ndim==2 and diff.shape[1]==len(gram) and np.isfinite(n0) and n0>0
    values=np.sqrt(np.maximum(np.sum((diff@gram)*diff,axis=1),0))/n0
    return float(values.max()) if np.isfinite(values).all() else float('inf')

def original_policy(current_C,old_C,current_fields,old_fields,gram,n0,current_error,old_error):
    assert np.asarray(current_C).shape==np.asarray(old_C).shape
    active=field_distance(current_C,old_C,gram,n0)
    c0=field_distance(np.asarray(current_C)[:1],np.asarray(old_C)[:1],gram,n0)
    output=np.linalg.norm(np.asarray(current_fields)-np.asarray(old_fields),axis=1)/n0
    error=abs(current_error-old_error)
    return dict(active_state_distance=active,C0_distance=c0,output_distance=float(output.max()),E0_error_difference=float(error),
        passed=bool(active<=FIELD_TOL and c0<=FIELD_TOL and np.isfinite(output).all() and output.max()<=FIELD_TOL and error<=ERROR_TOL),
        field_tolerance=FIELD_TOL,error_tolerance=ERROR_TOL,
        interpretation='operational identity much tighter than unchanged scientific gates; actual states and hashes retained, byte equality descriptive')
