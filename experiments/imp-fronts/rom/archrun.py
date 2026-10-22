"""imp-fronts arm (c): run a ROM script (r2acc.py, ...) with the arch-dispatching sep_common pinned.

engines.py (mr-burgers2d, shared, unchanged) inserts experiments/separable-decoder at sys.path[0] and imports sep_common,
which would shadow rom/deps/sep_common.py (the training side's copy with the 'sine_even' dispatch; no 'Kc' = the
incumbent code path). Importing rom/deps/sep_common first puts it in sys.modules, so every later `import sep_common`
(engines, qcore, arms, hfast) gets it.

    python archrun.py r2acc.py --config configs/e2sc.json --out <dir>
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pin():
    sys.path.insert(0, str(HERE / 'deps'))
    import sep_common
    assert Path(sep_common.__file__).resolve().parent == HERE / 'deps', sep_common.__file__
    return sep_common


if __name__ == '__main__':
    pin()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name='__main__')
