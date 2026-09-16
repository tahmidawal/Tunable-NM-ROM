"""Build `checks/comparators.json`: both reported metrics for every arm of every source job.

The audit's cross-job fidelity gates, and the DESIGN §1.1 table that located the regression
in the first place, both need the **same-grid** metric (against each job's own converged
`fft_tight`) for arms of jobs this lane did not run. Two of the three sources already
carry it in their committed audit JSONs; `cclad01` does not, so it is recomputed here from
that job's own retained fields, which are Git-tracked in this worktree as bounded chunks.

    python make_comparators.py [--work <dir>]

Reassembles each needed archive, verifies its whole-archive SHA256 against `archive.json`,
extracts only `output/`, recomputes, and writes the table. Nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDITS = {
    'btq101': ROOT / 'experiments/b-ladder-top/checks/btq101-audit.json',
    'btq102': ROOT / 'experiments/b-ladder-top/checks/btq102-audit.json',
    'btq201': ROOT / 'experiments/b-ladder-top/checks/btq201-audit.json',
}
ARCHIVES = {'cclad01': ROOT / 'experiments/cheap-corrections/artifacts/cclad01'}
KEEP = ('arm', 'q', 'k', 'M', 'm', 'rule', 'quadrature', 'fix', 'variant', 'gtol',
        'converged', 'total_budget_exits', 'max_joint_stationarity', 'median_gpu_ms',
        'worst_reference_percent', 'worst_all_times_percent', 'worst_evolved_percent',
        'worst_t0_compression_percent', 'quadrature_fit_relative', 'quadrature_truncated')


def restore(src: Path, work: Path):
    meta = json.loads((src / 'archive.json').read_text())
    tar = work / 'collection.tar.gz'
    h = hashlib.sha256()
    with tar.open('wb') as out:
        for chunk in meta['chunks']:
            b = (src / chunk['path']).read_bytes()
            assert hashlib.sha256(b).hexdigest() == chunk['sha256'], chunk['path']
            h.update(b)
            out.write(b)
    assert h.hexdigest() == meta['sha256'], src
    subprocess.run(['tar', '-xzf', str(tar), 'output'], cwd=work, check=True)
    tar.unlink()
    return work / 'output', meta['sha256']


def from_fields(out: Path):
    r = json.loads((out / 'result.json').read_text())
    cache = {}

    def F(n):
        if n not in cache:
            cache[n] = np.load(out / n)['fields']
        return cache[n]

    refs = {x['case']: F(x['artifact']) for x in r['reference']}
    base = {x['case']: x['artifact'] for x in r['invocations'] if x['name'] == 'fft_tight'}
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    t = {}
    for x in r['invocations']:
        truth = refs[x['case']]
        n0 = np.linalg.norm(truth[0])
        f = F(x['artifact'])
        sg = np.linalg.norm((f - F(base[x['case']])).reshape(len(f), -1), axis=1) / n0
        ref = np.linalg.norm((f - truth).reshape(len(f), -1), axis=1) / n0
        d = t.setdefault(x['name'], dict(arm=x['name'], q=x.get('q'), M=x.get('M'),
                                         m=x.get('m'), rule=x.get('rule'),
                                         quadrature=x.get('quadrature'),
                                         variant=x.get('variant'), fix=x.get('fix'),
                                         all={}, ev={}, t0={}, ref={}, ms=[], conv=[]))
        d['all'][x['case']] = float(sg.max())
        d['ev'][x['case']] = float(sg[1:].max())
        d['t0'][x['case']] = float(sg[0])
        d['ref'][x['case']] = float(ref.max())
        d['ms'].append(x['gpu_seconds'] * 1e3)
        if x['kind'] == 'rom':
            d['conv'].append(bool(x.get('converged')))
    rows = {}
    for name, d in t.items():
        s = setup.get(name, {})
        rows[name] = dict(
            arm=name, q=d['q'], M=d['M'], m=d['m'], rule=d['rule'], quadrature=d['quadrature'],
            variant=d['variant'], fix=d['fix'],
            worst_all_times_percent=100 * max(d['all'].values()),
            worst_evolved_percent=100 * max(d['ev'].values()),
            worst_t0_compression_percent=100 * max(d['t0'].values()),
            worst_reference_percent=100 * max(d['ref'].values()),
            median_gpu_ms=float(np.median(d['ms'])),
            converged=(bool(all(d['conv'])) if d['conv'] else None),
            quadrature_fit_relative=s.get('eq_relative_fit'))
    return rows, r.get('job_id')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', default=None, help='scratch directory (a temp dir by default)')
    a = p.parse_args()
    table = {}
    for name, path in AUDITS.items():
        au = json.loads(path.read_text())
        table[name] = dict(job_id=au.get('job_id'), gpu=au.get('gpu'), source=str(path),
                           recomputed_here=False,
                           arms={x['arm']: {k: x.get(k) for k in KEEP} for x in au['arms']})
    work_root = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix='qridge-cmp-'))
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        for name, src in ARCHIVES.items():
            w = work_root / name
            w.mkdir(parents=True, exist_ok=True)
            out, sha = restore(src, w)
            rows, job = from_fields(out)
            table[name] = dict(job_id=job, source=str(src), archive_sha256=sha,
                               recomputed_here=True,
                               note=('same-grid metrics recomputed from this job\'s own '
                                     'retained fields; its committed audit reports only the '
                                     'all-times figure'),
                               arms={k: {kk: v.get(kk) for kk in KEEP} for k, v in rows.items()})
            shutil.rmtree(w)
    finally:
        if not a.work:
            shutil.rmtree(work_root, ignore_errors=True)
    (HERE / 'checks/comparators.json').write_text(json.dumps(table, indent=2) + '\n')
    for k, v in table.items():
        print(k, 'arms', len(v['arms']), 'recomputed' if v['recomputed_here'] else 'from audit')
    print('WROTE', HERE / 'checks/comparators.json')


if __name__ == '__main__':
    main()
