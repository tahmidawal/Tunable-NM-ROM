"""SHIM (not a copy): refcore.py imports `cohorts` only for `cohorts.E2` (2D engines) and `cohorts.C3` (3D common).
This lane provides exactly those two names from its own vendored modules, so refcore.py stays byte-identical
(see INHERITED.json). No cohort, sealed or OOD definition is provided here."""
import sys
from pathlib import Path

_L = Path(__file__).resolve().parents[1]
for _p in (_L.parents[1] / 'experiments' / 'mr-burgers2d', _L / 'vendor' / 'quad3d' / 'vendor' / 'burgers3d-span'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import engines as E2   # noqa: E402,F401
import common as C3    # noqa: E402,F401
