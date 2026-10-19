"""jcp-robust: E0 scoring of an E2 3D job on its OWN bank (code audit 1 #8). Wraps score3.py (copied unchanged from
jcp-dims/jcp-speed3d): before scoring, score3's setting -> (bank, width) map is rebuilt from the job's settings and bank
registry (c3common.BANKS gains the retrained bank with its sha256, verified at load), and score3's decode cache is
cleared, so no incumbent row matrix can be used for a retrained bank. CPU only.

    python e2score3.py --run <output dir> --out <scores json>
"""
import offline  # noqa: F401  (CPU, highest, x64)

import json
import sys
from pathlib import Path

import c3common as C3
import score3 as SC

HERE = Path(__file__).resolve().parent


def install_union(cfgs):
    """Settings of several jobs of ONE bank (e.g. e2q<b>a and e2q<b>b): the union map, every name bound to one
    (bank, width); a conflicting name raises (code audit 3 #28)."""
    m = {}
    for c in cfgs:
        install(c)
        for s_ in c['settings']:
            v = (s_['bank'], int(s_['Rp']))
            assert m.setdefault(s_['name'], v) == v, ('setting bound to two banks', s_['name'])
    SC.BANKS = m
    SC._G.clear()


def install(cfg):
    import hashlib
    for name, b in (cfg.get('bank_registry') or {}).items():
        want = b['sha256']
        if want == 'SMOKE-RUNTIME':
            assert str(cfg.get('attempt', '')).startswith('smk')
            want = hashlib.sha256((HERE / b['path']).read_bytes()).hexdigest()
        prev = C3.BANKS.get(name)
        assert prev is None or (Path(prev[0]).resolve() == (HERE / b['path']).resolve() and prev[1] == want), \
            ('bank alias bound to two different checkpoints', name)                 # code audit 4 #35
        C3.BANKS[name] = (HERE / b['path'], want)
    SC.BANKS = {s['name']: (s['bank'], int(s['Rp'])) for s in cfg['settings']}
    SC._G.clear()


if __name__ == '__main__':
    run = Path(sys.argv[sys.argv.index('--run') + 1])
    cfg = json.loads((run / 'result.json').read_text())['config']
    install(cfg)
    SC.main()
