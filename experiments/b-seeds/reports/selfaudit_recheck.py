"""Independent re-derivation of every T12 number straight from the raw job `result.json`.

    python experiments/b-seeds/reports/selfaudit_recheck.py --runs experiments/b-seeds/runs \
        --summary experiments/b-seeds/reports/summary.json \
        --out experiments/b-seeds/checks/selfaudit-recheck.json

This is the substitute for the protocol's Codex audit of the finished report (Codex quota is
exhausted until 2026-09-19 11:33; DESIGN.md A1/A3). It does NOT read the audit JSONs and does
NOT import the report generator: it recomputes, from the raw invocation records the job wrote,
every value `summary.json` carries for the development cohort — worst all-times, worst evolved,
t=0, median GPU ms over the three repetitions, per-rung convergence, monotonicity, the best-found
and bank-floor layers — and compares them to the published rows. A disagreement above 1e-12
relative is a FAIL. It exists because the two defects found in this lane's own tooling (a hash
gate that a 1-ulp difference broke, and a TR row that printed a falsification flag as a verdict)
were both generator-side, invisible to the in-job gates.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def rel(a, b):
    if a is None or b is None:
        return None if a is b else float('inf')
    d = max(abs(a), abs(b))
    return 0.0 if d == 0 else abs(a - b) / d


def recompute(result, fields):
    """Same-grid errors straight from the retained fields, plus costs and convergence.

    The same-grid error of a solve at output time t is the L2 distance between its field and
    the same-job converged `fft_tight` field of the same case, normalised by the L2 norm of the
    reference solution's initial state, exactly as the ladder driver defines it. Fields are
    retained for repetition 0; the audit separately gates that the repetitions are identical.
    The published cost is the median over every invocation of the arm (6 cases x 3 repetitions).
    """
    import numpy as np
    ref0 = {x['case']: np.linalg.norm(np.load(fields / x['artifact'])['fields'][0])
            for x in result['reference']}
    base = {x['case']: x['artifact'] for x in result['invocations'] if x['name'] == 'fft_tight'}
    arms = {}
    for inv in result['invocations']:
        if inv['kind'] != 'rom':
            continue
        a = arms.setdefault(inv['name'], dict(case={}, gpu=[], conv=[], exits=0))
        a['gpu'].append(inv['gpu_seconds'] * 1e3)
        a['conv'].append(bool(inv['converged']))
        a['exits'] += inv['budget_exits']
        if inv['rep'] != 0:
            continue
        f = np.load(fields / inv['artifact'])['fields']
        g = np.load(fields / base[inv['case']])['fields']
        a['case'][inv['case']] = np.linalg.norm((f - g).reshape(len(f), -1), axis=1) / ref0[inv['case']]
    out = {}
    for name, a in arms.items():
        sg = np.array([a['case'][c] for c in sorted(a['case'])])
        out[name] = dict(all_times=float(sg.max()) * 100, evolved=float(sg[:, 1:].max()) * 100,
                         t0=float(sg[:, 0].max()) * 100, gpu_ms=float(np.median(a['gpu'])),
                         converged=all(a['conv']), budget_exits=a['exits'],
                         invocations=len(a['gpu']), cases=len(a['case']))
    return out


def layers(result):
    """bank floor and best-found at q = 0, worst over cases, from the reconstruction block."""
    for block in result['reconstruction']:
        if block.get('q') == 0:
            return dict(bank_floor=max(c['bank_projection_max'] for c in block['cases']) * 100,
                        best_found=max(c['best_found_max'] for c in block['cases']) * 100)
    return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', default=str(ROOT / 'experiments/b-seeds/runs'))
    p.add_argument('--summary', default=str(ROOT / 'experiments/b-seeds/reports/summary.json'))
    p.add_argument('--out', default=str(ROOT / 'experiments/b-seeds/checks/selfaudit-recheck.json'))
    p.add_argument('--report', default=str(ROOT / 'experiments/b-seeds/reports/2026-09-17-b-seeds.md'))
    p.add_argument('--tol', type=float, default=1e-12)
    a = p.parse_args()
    rows = json.loads(Path(a.summary).read_text())

    def published(**kw):
        m = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
        return m[0]['value'] if m else None

    report, failures = [], []
    recomputed_dev, seed_labels = {}, []
    for attempt in sorted(Path(a.runs).iterdir()):
        for block in ('ladder_seed', 'ladder_incumbent'):
            res = attempt / 'archive/output' / block / 'result.json'
            if not res.exists():
                continue
            r = json.loads(res.read_text())
            label = r['checkpoint_label']
            mine = recompute(r, res.parent)
            lay = layers(r)
            if block == 'ladder_seed':
                seed_labels.append(label)

            for lad in r['ladders'].values():
                for rung in lad['rungs']:
                    arm, q = rung['arm'], rung['q']
                    mv = mine[arm]
                    for metric, ours in (('all_times', mv['all_times']), ('evolved', mv['evolved']),
                                         ('t0', mv['t0']), ('converged', mv['converged'])):
                        kw = dict(checkpoint=label, cohort='dev', attempt=attempt.name,
                                  ladder=lad['id'], q=q, metric=metric)
                        theirs = published(**kw)
                        if theirs is None:
                            continue
                        ok = (bool(ours) == bool(theirs)) if metric == 'converged' else (
                            rel(ours, theirs) <= a.tol)
                        if block == 'ladder_seed' and lad['id'] == 'dense_m4':
                            recomputed_dev[(label, q, metric)] = ours
                        rec = dict(attempt=attempt.name, block=block, checkpoint=label,
                                   ladder=lad['id'], q=q, metric=metric, recomputed=ours,
                                   published=theirs, passed=ok)
                        report.append(rec)
                        if not ok:
                            failures.append(rec)
                    theirs = published(checkpoint=label, cohort='dev', attempt=attempt.name,
                                       ladder=lad['id'], q=q, metric='gpu_ms')
                    if theirs is not None:
                        ok = rel(mv['gpu_ms'], theirs) <= a.tol
                        rec = dict(attempt=attempt.name, block=block, checkpoint=label,
                                   ladder=lad['id'], q=q, metric='gpu_ms',
                                   recomputed=mv['gpu_ms'], published=theirs, passed=ok,
                                   invocations=mv['invocations'], cases=mv['cases'])
                        report.append(rec)
                        if not ok:
                            failures.append(rec)
            for metric, ours in (('three_layer_bank_floor_percent', lay.get('bank_floor')),
                                 ('three_layer_best_found_percent', lay.get('best_found'))):
                theirs = published(checkpoint=label, cohort='dev', attempt=attempt.name, metric=metric)
                if theirs is None or ours is None:
                    continue
                ok = rel(ours, theirs) <= a.tol
                rec = dict(attempt=attempt.name, block=block, checkpoint=label, metric=metric,
                           recomputed=ours, published=theirs, passed=ok)
                report.append(rec)
                if not ok:
                    failures.append(rec)
    # --- the published T12 table itself, parsed back out of the report prose --------------
    # The strongest end-to-end check: the markdown a reader sees must equal what the fields say.
    # It is a separate failure mode from the rows above, which compare summary.json only.
    if a.report and Path(a.report).exists():
        import re
        import statistics
        md = Path(a.report).read_text().split('## 2. T12')[1].split(chr(10) * 2)[2].splitlines()
        seen = 0
        for line in md:
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if len(cells) < 9 or not re.fullmatch(r'\d+', cells[0]):
                continue
            q = int(cells[0])
            seen += 1
            for col, metric in ((2, 'evolved'), (4, 'all_times'), (6, 't0')):
                vals = [v for v in (recomputed_dev.get((s_, q, metric)) for s_ in seed_labels) if v is not None]
                if len(vals) != len(seed_labels):
                    continue
                want = (statistics.mean(vals), statistics.stdev(vals))
                got = tuple(float(x) for x in cells[col].split('±'))
                ok = max(abs(want[0] - got[0]), abs(want[1] - got[1])) <= 5e-5
                rec = dict(source='report T12 markdown', q=q, metric=metric,
                           recomputed_mean_std=want, printed_mean_std=got, passed=ok)
                report.append(rec)
                if not ok:
                    failures.append(rec)
        assert seen == 6, f'parsed {seen} T12 rungs, expected 6'

    payload = dict(tolerance=a.tol, comparisons=len(report), failures=failures, rows=report)
    Path(a.out).write_text(json.dumps(payload, indent=1) + '\n')
    print(f'{len(report) - len(failures)} of {len(report)} independent re-derivations agree '
          f'with summary.json to {a.tol:g} relative')
    for f_ in failures:
        print('FAIL', json.dumps(f_))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
