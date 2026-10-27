"""Bounded restoration of approved derived normalisers; the complete cohort hash remains strict."""
import hashlib
import numpy as np

FIELDS=('B','c','w','rho','A','nu','s_star')

def table_hash(table):
    h=hashlib.sha256()
    for key in FIELDS:h.update(np.ascontiguousarray(table[key]).tobytes())
    return h.hexdigest()

def restore(table,approved_normalisers,expected_hash):
    """Regenerated raw draws are untouched; only <=one-ULP derived normalisers can be restored.

    `approved_normalisers` must come from manifest-verified E0 metadata at the caller.
    Requiring the final full SHA proves all active AND inactive raw draw bytes agree.
    """
    raw=table_hash(table)
    assert raw==str(table['sha256']),'generated table fingerprint is inconsistent'
    a=np.asarray(table['s_star']);b=np.asarray(approved_normalisers,dtype=np.float64)
    assert a.dtype==np.float64 and a.shape==b.shape==(64,)
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.all(a>0) and np.all(b>0)
    delta=a-b
    # Positive finite float64 bit patterns are monotone; integer distance also handles exponent boundaries exactly.
    ulp=np.abs(a.view(np.int64)-b.view(np.int64))
    assert np.all(ulp<=1),('normaliser difference exceeds one ULP',np.flatnonzero(ulp>1).tolist())
    canonical=dict(table,s_star=b.copy())
    digest=table_hash(canonical)
    assert digest==expected_hash,'raw seed draws or approved normalisers differ from the original cohort'
    canonical['sha256']=digest
    audit=dict(raw_sha256=raw,canonical_sha256=digest,changed_cases=np.flatnonzero(delta!=0).tolist(),
               delta=delta.tolist(),ulp_difference=ulp.tolist(),maximum_ulp=int(ulp.max()),
               final_strict_hash_passed=True)
    return canonical,audit
