"""poisson2d-test: the parent bank builder with the truth fields kept on the HOST.

`build_banks_host` is `pbk_core.build_banks` (poisson-bank-knob @06546331) with exactly three textual
edits, applied to the parent's own source at import time and asserted to match once each:
the truth stack stays in host memory and only one row slice at a time is copied to the device.
Reason: 32 test fields at 4096^2 are 4.3e9 bytes, and the parent's 4096^2 peak was already 77.7e9 of an
80.8e9-byte A100 limit (12 fields). The bank, its contraction and R_G are untouched; only the floor
diagnostic reads the truth, and the host sum of squares replaces the device one.
"""
import inspect
import textwrap

import pbk_core as P

EDITS = [
    ("Uj = jnp.asarray(truth)", "Uj = truth"),
    ("g = Uj[:, s:e].reshape(Uj.shape[0], -1) @ G",
     "g = jnp.asarray(np.ascontiguousarray(Uj[:, s:e]).reshape(Uj.shape[0], -1)) @ G"),
    ("nu2 = np.asarray(jnp.sum(Uj * Uj, axis=(1, 2)))", "nu2 = np.sum(Uj * Uj, axis=(1, 2))"),
]

_src = textwrap.dedent(inspect.getsource(P.build_banks))
for old, new in EDITS:
    assert _src.count(old) == 1, old
    _src = _src.replace(old, new)
_src = _src.replace('def build_banks(', 'def build_banks_host(', 1)
_ns = dict(vars(P))
exec(compile(_src, '<p2t_core.build_banks_host>', 'exec'), _ns)
build_banks_host = _ns['build_banks_host']
SOURCE = _src
