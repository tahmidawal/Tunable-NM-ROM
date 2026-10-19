"""jcp-regimes: slim copies of the references a job wrote (COMMON-BRIEF §16: only small files are pulled). CPU/NumPy.

    python refslim.py --config configs/<x>.json --out <dir>
config: {"sources": ["<output dir of a refjob step, relative to the job's output/>", ...]}.
For every <source>/2d/<cohort>/ST/c<j>.npz|json: writes <out>/2d/<cohort>/ST/c<j>.npz with exactly the arrays the scorer
reads (the final triple's finest field under its own key, and t0) and copies the json unchanged; MANIFEST.json lists
sha256 of source and slim files. The full npz stays on the cluster (archived).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    r = str(out.resolve())                            # output-root guard (code audit 1, fix 1): never the E0 lane / store
    if '/jcp-ref/' in r + '/' or 'jcp-references' in r:
        raise SystemExit(f'refusing to write into the E0 lane / store: {r}')
    if os.environ.get('SLURM_JOB_ID'):
        assert r.startswith('/cluster/tufts/paralab/tawal01/jcp-regimes/'), r
    else:
        assert r.startswith('/tmp/'), r
    rows = []
    pairs = [(Path(a.out).parent / s_, js) for s_ in cfg['sources'] for js in sorted((Path(a.out).parent / s_).glob('2d/*/ST/c*.json'))]
    for src, js in pairs:
        meta = json.loads(js.read_text())
        z = np.load(js.with_suffix('.npz'))
        keep = {meta['final'][2]: z[meta['final'][2]], 't0': z['t0']}
        d = out / js.relative_to(src).parent
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / js.with_suffix('.npz').name, **keep)
        (d / js.name).write_bytes(js.read_bytes())
        rows.append(dict(case=str(js.relative_to(src)), source_npz=sha(js.with_suffix('.npz')), source_json=sha(js),
                         slim_npz=sha(d / js.with_suffix('.npz').name), finest=meta['final'][2], outcome=meta['outcome']))
    (out / 'MANIFEST.json').write_text(json.dumps(rows, indent=1) + '\n')
    print('REFSLIM', len(rows), 'cases', flush=True)
    assert rows or cfg.get('allow_empty'), 'no reference files found'
    os.sync()


if __name__ == '__main__':
    main()
