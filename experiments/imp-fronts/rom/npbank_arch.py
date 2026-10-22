"""imp-fronts arm (c): architecture dispatch for the vendored NumPy evaluator `npbank` (hand gradient), DESIGN
Amendments 19-20. NumPy only.

The vendored npbank (byte-identical, used by rom/npcheck.py and by jcp-regimes' npcheck that fcommon / floors_b /
score_b import) knows only the incumbent formula. install(nb) patches a loaded npbank MODULE in place:
  nb.load(ckpt)              -> the incumbent dict, plus 'Kc' when the checkpoint's params carry it; the checkpoint cfg
                                ('arch') must agree with the presence of 'Kc' (else ValueError)
  nb.features_grad(P, X)     -> deps/sine_even_ref.features_grad when 'Kc' in P, else the original (unchanged)
Dispatch rule shared with the training side and the ROM-path sep_common: "Kc" in params <=> cfg['arch'] == 'sine_even'.
Call it before any bank is loaded, e.g. `import npcheck as NC; import npbank_arch; npbank_arch.install(NC.NB)`.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE / 'deps') not in sys.path:
    sys.path.insert(0, str(HERE / 'deps'))
import sine_even_ref as SE  # noqa: E402


def arch_of(ck):
    """'sine_even' or 'bc16poly' from a loaded checkpoint dict; raises on a cfg / params disagreement."""
    has = 'Kc' in ck['params']
    a = (ck.get('cfg') or {}).get('arch') if isinstance(ck.get('cfg'), dict) else None
    if a not in (None, 'bc16poly', 'sine_even'):
        raise ValueError(f'unknown arch {a!r}')
    if has != (a == 'sine_even'):
        raise ValueError(f"checkpoint cfg arch {a!r} disagrees with 'Kc' in params ({has})")
    return 'sine_even' if has else 'bc16poly'


def install(nb):
    if getattr(nb, '_arch_patched', False):
        return nb
    load0, fg0 = nb.load, nb.features_grad

    def load(ckpt):
        ck = pickle.load(open(ckpt, 'rb'))
        arch = arch_of(ck)
        P = load0(ckpt)
        if arch == 'sine_even':
            P['Kc'] = np.asarray(ck['params']['Kc'], float)
        P['arch'] = arch
        return P

    def features_grad(P, X):
        if 'Kc' in P:
            return SE.features_grad(P, X)
        return fg0(P, X)

    nb.load, nb.features_grad, nb._arch_patched = load, features_grad, True
    return nb
