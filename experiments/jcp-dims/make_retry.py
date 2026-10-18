"""jcp-dims retry configs for a timed-out 2D accuracy job (DESIGN A7.1): configs/<job>r<k>.json keeps only the settings
whose arms lack an archived npz in every collected attempt of <job> (partial collections included), unchanged
otherwise, so analyze2d.load_all can merge per arm. Job-level blocks are kept only if no complete attempt exists.

    python make_retry.py <job> <k>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import analyze2d as A2

HERE = Path(__file__).resolve().parent


def main():
    job, k = sys.argv[1], int(sys.argv[2])
    cfg = json.loads((HERE / 'configs' / f'{job}.json').read_text())
    have, complete = set(), False
    for d in A2.attempt_dirs(job):
        res = json.loads((d / 'result.json').read_text())
        complete = complete or bool(res.get('complete'))
        for s in res['config']['settings']:
            for a in s['arms']:
                if (d / f"R_{s['name']}_{a['form']}_{a['scheme']}_{float(a['factor']):g}.npz").exists():
                    have.add((s['name'], a['form'], a['scheme']))
    keep = [s for s in cfg['settings'] if not all((s['name'], a['form'], a['scheme']) in have for a in s['arms'])]
    out = dict(cfg, attempt=f'{job}r{k}', settings=keep, blocks=[] if complete else cfg.get('blocks', []),
               retry_of=job, note='settings without archived evidence in any collected attempt of ' + job)
    (HERE / 'configs' / f'{job}r{k}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(job, 'retry', k, 'settings', len(keep), 'of', len(cfg['settings']))


if __name__ == '__main__':
    main()
