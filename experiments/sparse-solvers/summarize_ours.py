"""Task B (amendment 3): our reduced model re-timed in this lane on the same GPU type as the cuDSS runs, using the
paper's own timing drivers and frozen checkpoints unchanged.  Reads the small result files pulled into
results/ours/<gpu>/<label>.json, compares with the paper's values (ours_burgers.json for Burgers; ours.json headline
for the rest), writes results/ours/same_gpu.json and prints RESULTS-ours-same-gpu.md.

  python summarize_ours.py > RESULTS-ours-same-gpu.md

A timing is marked usable only if the worst-case and median errors reproduce the paper's values (relative
difference <= 1e-6; the paths are deterministic).
"""
import json
from pathlib import Path

import statistics as st

HERE = Path(__file__).resolve().parent
D = HERE / 'results' / 'ours'
B2 = {'t256': 256, 't512': 512, 't1024': 1024, 't2048': 2048, 't4096': 4096}
B3 = {'ho65': 64, 'ho129': 128}
P2 = {'p2t256': 256, 'p2t1024': 1024, 'p2t2048': 2048, 'p2t4096': 4096}


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)


def b2d(r, paper):
    """Per role: errors from the quick pass, time from all timed invocations of that arm (the paper's rule)."""
    out = {}
    for role in ('accurate', 'fast'):
        arm = paper[role]['arm']
        q = [x for x in r['quick'] if x['name'] == arm]
        inv = [x['gpu_seconds'] for x in r['invocations'] if x['name'] == arm]
        if not q or not inv:
            out[role] = dict(arm=arm, status='arm not in this result')
            continue
        errs = [100 * x['same_grid_evolved'] for x in q]
        out[role] = dict(arm=arm, worst_percent=max(errs), median_percent=st.median(errs), ms=1e3 * st.median(inv),
                         ms_min=1e3 * min(inv), ms_max=1e3 * max(inv), invocations=len(inv), cases=len(q))
    foms = {}
    for name in sorted({x['name'] for x in r['invocations'] if x['family'] == 'fom'}):
        inv = [x['gpu_seconds'] for x in r['invocations'] if x['name'] == name]
        q = [x for x in r['quick'] if x['name'] == name]
        if q:
            foms[name] = dict(ms=1e3 * st.median(inv), worst_percent=100 * max(x['same_grid_evolved'] for x in q))
    return out, foms


def b3d(r, paper):
    out = {}
    for role in ('accurate', 'fast'):
        arm = paper[role]['arm']
        a, t = r['arms'].get(arm), r['timing']['summary'].get(arm)
        if not a or not t:
            out[role] = dict(arm=arm, status='arm not in this result')
            continue
        inv = [x for x in r['timing'].get('invocations', []) if x.get('name', x.get('arm')) == arm]
        secs = [x.get('gpu_seconds', x.get('seconds')) for x in inv]
        secs = [s for s in secs if s is not None]
        out[role] = dict(arm=arm, worst_percent=100 * a['worst_evolved'], median_percent=100 * a['median_worst_evolved'],
                         ms=t['median_ms'], invocations=t['n'])
        if secs:
            out[role].update(ms_min=1e3 * min(secs), ms_max=1e3 * max(secs))
    foms = {k: dict(ms=v['median_ms']) for k, v in r['timing']['summary'].items() if k.startswith('fom_')}
    return out, foms


def p2d(r, paper):
    """Poisson 2D (the logic of poisson2d-test/make_report.py one()): per arm, error = same_grid_error per case, time =
    median fused_device_seconds over all timed invocations including the slow ones."""
    inv = r['invocations'] + r['slow_invocations']
    def arm(nm):
        rows = [x for x in inv if x['name'] == nm]
        if not rows:
            return None
        errs = list({x['case']: x['same_grid_error'] for x in rows}.values())
        secs = [x['fused_device_seconds'] for x in rows]
        return dict(arm=nm, worst_percent=100 * max(errs), median_percent=100 * st.median(errs), ms=1e3 * st.median(secs),
                    ms_min=1e3 * min(secs), ms_max=1e3 * max(secs), invocations=len(rows), cases=len(errs))
    out = {role: arm(paper[role]['arm']) or dict(arm=paper[role]['arm'], status='arm not in this result')
           for role in ('accurate', 'fast')}
    foms = {nm: arm(nm) for nm in paper['fom'] if arm(nm)}
    return out, foms


def main():
    paper = json.loads((HERE / 'ours_burgers.json').read_text())
    paper.update({k: v for k, v in json.loads((HERE / 'ours_poisson2d.json').read_text()).items() if k != 'source'})
    same, rows, fom_rows = {}, [], []
    for gdir in sorted(p for p in D.iterdir() if p.is_dir()) if D.exists() else []:
        for f in sorted(gdir.glob('*.json')):
            label = f.stem
            r = json.loads(f.read_text())
            if label in B2:
                prob, L = 'burgers2d', B2[label]
                got, foms = b2d(r, paper[prob][str(L)])
            elif label in B3:
                prob, L = 'burgers3d', B3[label]
                got, foms = b3d(r, paper[prob][str(L)])
            elif label in P2:
                prob, L = 'poisson2d', P2[label]
                got, foms = p2d(r, paper[prob][str(L)])
            else:
                continue
            gpu = r.get('gpu', gdir.name)
            for role, g in got.items():
                p = paper[prob][str(L)][role]
                if 'ms' not in g:
                    rows.append((prob, L, role, gpu, r.get('job_id'), g.get('status'), '', '', '', '', '', '', ''))
                    continue
                ok = close(g['worst_percent'], p['worst_percent']) and close(g['median_percent'], p['median_percent'])
                g.update(gpu=gpu, job_id=r.get('job_id'), host=r.get('host'), complete=r.get('complete'), errors_reproduce=ok,
                         paper_ms=p['ms'], paper_gpu=paper[prob][str(L)]['gpu'], paper_worst_percent=p['worst_percent'])
                if ok and gdir.name == 'h200':
                    same.setdefault(prob, {}).setdefault(str(L), {})[role] = g
                same.setdefault('_all', []).append(dict(g, problem=prob, L=L, role=role, gpu_dir=gdir.name))
                spread = f"{g['ms_min']:.4g}–{g['ms_max']:.4g}" if 'ms_min' in g else ''
                rows.append((prob, L, role, gpu, r.get('job_id'), 'yes' if ok else 'NO',
                             f"{g['worst_percent']:.4f}", f"{p['worst_percent']:.4f}", f"{g['median_percent']:.4f}",
                             f"{p['median_percent']:.4f}", f"{g['ms']:.4g}", spread,
                             f"{p['ms']:.4g} ({paper[prob][str(L)]['gpu']})"))
            pf = paper[prob][str(L)]['fom']
            for name, v in foms.items():
                if name in pf:
                    fom_rows.append((prob, L, name, gpu, f"{v['ms']:.4g}", f"{pf[name]['ms']:.4g}",
                                     '' if 'worst_percent' not in v else f"{v['worst_percent']:.4g}"))
            same.setdefault('_fom', {}).setdefault(gdir.name, {}).setdefault(prob, {})[str(L)] = foms
    if D.exists():
        (D / 'same_gpu.json').write_text(json.dumps(same, indent=1))
    P = print
    P('# Our reduced model re-timed in this lane (amendment 3, Task B)')
    P()
    P('Generated by `summarize_ours.py` from the result files of the paper\'s own timing drivers, re-run unchanged (same '
      'code trees, verified against the original run manifests; frozen checkpoints; only the job root path differs) in '
      'this lane\'s jobs, on the GPU type used for the cuDSS runs. A timing is used only if the errors reproduce the '
      'paper\'s values. Time = median over all timed invocations (the paper\'s timing contract: initial field on the GPU '
      'to six output fields on the GPU).')
    P()
    P('| problem | mesh | setting | GPU | job | errors reproduce | worst % (this run) | worst % (paper) | median % (this run) | '
      'median % (paper) | ms median (this run) | ms min–max | ms (paper, GPU) |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for row in sorted(rows, key=lambda x: (x[0], x[1], x[2], str(x[3]))):
        P('| ' + ' | '.join(str(x) for x in row) + ' |')
    P()
    P('Rows of the paper not listed here were not re-measured in this lane.')
    P()
    P('## The paper\'s own matrix-free full-order solver, re-timed by the same drivers in the same jobs')
    P()
    P('| problem | mesh | solver setting | GPU | ms median (this run) | ms (paper) | worst % |')
    P('|---|---|---|---|---|---|---|')
    for row in sorted(fom_rows, key=lambda x: (x[0], x[1], x[2])):
        P('| ' + ' | '.join(str(x) for x in row) + ' |')


if __name__ == '__main__':
    main()
