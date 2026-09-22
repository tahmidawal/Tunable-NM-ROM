"""Generate the ops-timing-panel report and its summary.json from the audit JSON alone.

    python reports/generate_ops_panel.py <audit.json> --out reports/2026-09-22-ops-timing-panel.md

No number in the report is typed by hand: every cell is read from the audit, which itself
recomputed every error in NumPy from the saved fields. The script also:

* applies DESIGN.md section 6's full-order rule per row and per timing scope;
* refuses to print an operator row whose own gates failed (DESIGN section 7);
* refuses to run at all if the audit's `failed` list is non-empty (DESIGN section 9);
* compares this job's `fno-large` ERROR row with b-panel `bpn301`'s, which is the same cohort,
  metric and code -- errors only, never the timings, which are from another allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FAMILY_LABEL = {'rom': 'NM-ROM', 'fast': 'NM-ROM (fast kernel)', 'pod': 'POD-LSPG', 'free': 'free bank',
                'fno': 'FNO', 'unet': 'U-Net', 'transolver': 'Transolver', 'fom': 'FOM'}
ROLE = {'unet-refine': 'validation-selected', 'tsol-refine': 'validation-selected',
        'fno-large': 'validation-selected', 'unet-medium': 'best worst-case (not selected)',
        'tsol-large': 'best worst-case (not selected)'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fom_rule(rows, arm, scope):
    """DESIGN section 6: fastest tested FOM whose worst evolved error <= this arm's, on `scope`."""
    key = 'median_gpu_ms' if scope == 'gpu' else 'median_host_ms'
    cands = [f for f in rows if f['family'] == 'fom'
             and f['worst_evolved_percent'] <= arm['worst_evolved_percent'] + 1e-12]
    if not cands:
        return None, None
    best = min(cands, key=lambda f: f[key])
    return best['arm'], best[key] / arm[key]


def operator_gates(checks, arm):
    """Every gate this lane added for `arm`, and whether all of them passed."""
    got = {k: v for k, v in checks.items() if k.endswith(f'_{arm}') and v.get('blocking') is not False}
    return got, all(v['passed'] for v in got.values()) if got else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit')
    p.add_argument('--out', required=True)
    p.add_argument('--bpn301', default=None, help="b-panel bpn301 audit.json, for the FNO error cross-check")
    a = p.parse_args()
    d = json.loads(Path(a.audit).read_text())
    if d['failed']:
        raise SystemExit(f"audit has failed gates, the job is not accepted: {d['failed']}")
    rows = d['arms']
    by = {r['arm']: r for r in rows}

    printed, suppressed = [], []
    for r in rows:
        if r['kind'] == 'fno':
            got, ok = operator_gates(d['checks'], r['arm'])
            r['gates'] = {k: v['passed'] for k, v in got.items()}
            if ok is not True:
                suppressed.append(dict(arm=r['arm'], gates=r['gates']))
                continue
        for scope in ('gpu', 'host'):
            comp, ratio = fom_rule(rows, r, scope)
            r[f'fom_{scope}'] = comp
            r[f'speedup_{scope}'] = ratio
        printed.append(r)

    def cell(v, n=3):
        return '—' if v is None else f'{v:.{n}f}'

    lines = ['| arm | family | role | worst evolved % | median evolved % | worst all-times % | '
             'GPU-query ms | complete-query ms | FOM by the rule (GPU) | speedup (GPU) | speedup (complete) |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in printed:
        lines.append('| `{arm}` | {fam} | {role} | {we} | {me} | {wa} | {g} | {h} | {c} | {sg} | {sh} |'.format(
            arm=r['arm'], fam=FAMILY_LABEL.get(r['family'], r['family']), role=ROLE.get(r['arm'], '—'),
            we=cell(r['worst_evolved_percent'], 4), me=cell(r['median_evolved_percent'], 4),
            wa=cell(r['worst_all_times_percent'], 4), g=cell(r['median_gpu_ms']), h=cell(r['median_host_ms']),
            c=f"`{r['fom_gpu']}`" if r['fom_gpu'] else 'none at least as accurate',
            sg=cell(r['speedup_gpu'], 3) + ('×' if r['speedup_gpu'] is not None else ''),
            sh=cell(r['speedup_host'], 3) + ('×' if r['speedup_host'] is not None else '')))
    table = '\n'.join(lines)

    cross = None
    if a.bpn301:
        b = json.loads(Path(a.bpn301).read_text())
        bb = {r['arm']: r for r in b['arms']}
        cross = []
        for arm in ('fno-large',):
            if arm in by and arm in bb:
                cross.append(dict(
                    arm=arm, job=d['job_id'], bpn301_job=b['job_id'],
                    worst_evolved_percent=by[arm]['worst_evolved_percent'],
                    bpn301_worst_evolved_percent=bb[arm]['worst_evolved_percent'],
                    absolute_difference=abs(by[arm]['worst_evolved_percent'] - bb[arm]['worst_evolved_percent']),
                    per_case_max_difference=max(
                        abs(by[arm]['per_case_evolved_percent'][k] - bb[arm]['per_case_evolved_percent'][k])
                        for k in by[arm]['per_case_evolved_percent']),
                    note='errors only; the two jobs are different allocations and no timing ratio is formed'))

    summary = dict(
        lane='ops-timing-panel', attempt=d['attempt'], job_id=d['job_id'], commit=d['commit'], gpu=d['gpu'],
        intervals=d['intervals'], dt=d['dt'], K=d['K'], R=d['R'], elapsed_seconds=d['elapsed_seconds'],
        audit=str(Path(a.audit).name), audit_sha256=sha(a.audit), result_sha256=d['result_sha256'],
        failed_gates=d['failed'], operators=d.get('operators'),
        fom_rule='fastest tested FOM setting whose worst evolved same-grid error <= the row, per timing scope',
        rows=[{k: r[k] for k in (
            'arm', 'kind', 'family', 'worst_evolved_percent', 'median_evolved_percent',
            'worst_all_times_percent', 'median_all_times_percent', 'worst_reference_percent',
            'median_gpu_ms', 'median_host_ms', 'per_case_evolved_percent', 'admissible',
            'fom_gpu', 'speedup_gpu', 'fom_host', 'speedup_host') if k in r} | (
            {'gates': r['gates']} if 'gates' in r else {}) for r in printed],
        suppressed_rows=suppressed, fno_error_cross_check=cross,
        fom_discretisation_error_percent=d['fom_discretisation_error_percent'])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (out.parent / 'table-256.md').write_text(table + '\n')
    print(table)
    print()
    print('suppressed (gates failed):', suppressed or 'none')
    print('fno error cross-check vs bpn301:', json.dumps(cross))
    print('wrote', out.parent / 'summary.json', 'and', out.parent / 'table-256.md')


if __name__ == '__main__':
    main()
