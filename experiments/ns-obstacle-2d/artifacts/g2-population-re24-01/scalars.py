"""Strict conversion of scalar NGSolve results, with bounded shape evidence."""
import json
import math
from numbers import Real

import numpy as np

_observations = {}


def scalar_observations():
    return list(_observations.values())


def scalar_value(value, label):
    """Accept a finite real scalar or shape-(1,) numeric ndarray/tuple only.

    Never squeeze, flatten, sum, select a component from a larger vector, or
    discard an imaginary part. Each distinct type/shape outcome is logged once.
    """
    record = {'label': label, 'python_type': type(value).__module__+'.'+type(value).__qualname__,
              'shape': None, 'dtype': None, 'finite': None, 'accepted': False}
    result = None
    if isinstance(value, (Real, np.ndarray, tuple)) and not np.ma.isMaskedArray(value):
        try:
            array = np.asarray(value)
            record.update(shape=list(array.shape), dtype=str(array.dtype))
            if array.shape in ((), (1,)) and array.dtype.kind in 'fiu':
                record['finite'] = bool(np.isfinite(array).all())
                if record['finite']:
                    result = float(array.item() if array.shape == () else array[0])
                    record['accepted'] = math.isfinite(result)
        except (TypeError, ValueError, OverflowError) as exc:
            record['conversion_error'] = repr(exc)
    key = json.dumps(record, sort_keys=True)
    if key not in _observations:
        _observations[key] = record
        print(json.dumps({'scalar_return_contract': record}, allow_nan=False), flush=True)
    if not record['accepted']:
        raise ValueError('expected finite scalar shape () or (1,): '+key)
    return result
