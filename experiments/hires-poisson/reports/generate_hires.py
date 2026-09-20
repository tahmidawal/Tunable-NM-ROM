"""Generate summary.json and the lane report from the collected audit.json files.

    python experiments/hires-poisson/reports/generate_hires.py

Every number in the report comes from `runs/<attempt>/archive/output/audit.json`, which the
independent NumPy audit wrote on the cluster from the saved fields. Nothing is typed by hand.
"""
import hashlib
import json
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
OUT = LANE / 'reports'
ATTEMPTS_2D = ['hp2048', 'hp4096', 'hp4096b', 'hp2048b']
ATTEMPTS_3D = ['hp3d128']
ATTEMPTS_L = ['hpl1024', 'hpl2048']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(x):
    return f'{100 * x:.3f}'


def rows_for(table, selections, headline_variant, mesh_label, job, gpu, status):
    out = []
    sel = {s['rom']: s for s in selections}
    for name, r in table.items():
        s = sel.get(name)
        out.append(dict(mesh=mesh_label, job_id=job, gpu=gpu, status=status, subject=name,
                        family=r['family'], q=r.get('q'), tolerance=r.get('tolerance'),
                        coarse_intervals=r.get('coarse_intervals'), cases=r['cases'],
                        repetitions_per_case=r['repetitions_per_case'],
                        worst_same_grid=r['worst_same_grid'], worst_physical=r['worst_physical'],
                        median_total_ms=r['median_total_ms'], median_device_ms=r['median_device_ms'],
                        median_input_ms=r['median_input_ms'], median_output_ms=r['median_output_ms'],
                        median_iterations=r.get('median_iterations'),
                        median_lm_attempts=r.get('median_lm_attempts'),
                        speedups=None if s is None else {
                            k: (None if s[k] is None else dict(comparator=s[k]['comparator'],
                                                               total=s[k]['speedup_total'],
                                                               device=s[k]['speedup_device']))
                            for k in ('named_cg_1e-2', 'fastest_cg_matched',
                                      'fastest_coarse_matched', 'dst_direct')}))
    return out


def table_md(rows, keep):
    head = ('| subject | worst same-grid % | worst physical % | median total ms | median device ms '
            '| iterations | × vs CG 1e-2 (total / device) | × vs fastest matched CG | × vs matched coarse grid | × vs DST |\n'
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    body = ''
    for r in rows:
        if not keep(r):
            continue
        s = r['speedups']

        def f(key, dev=False):
            if not s or s.get(key) is None:
                return '—'
            txt = f"{s[key]['total']:.2f}"
            if dev:
                txt += f" / {s[key]['device']:.1f}"
            if key in ('fastest_cg_matched', 'fastest_coarse_matched'):
                txt += f" (`{s[key]['comparator']}`)"
            return txt
        it = r['median_iterations'] if r['median_iterations'] is not None else r['median_lm_attempts']
        body += (f"| `{r['subject']}` | {pct(r['worst_same_grid'])} | {pct(r['worst_physical'])} | "
                 f"{r['median_total_ms']:.2f} | {r['median_device_ms']:.2f} | "
                 f"{'' if it is None else f'{it:g}'} | {f('named_cg_1e-2', True)} | {f('fastest_cg_matched')} | "
                 f"{f('fastest_coarse_matched')} | {f('dst_direct')} |\n")
    return head + body


def main():
    summary = dict(sources=[], verdicts=[], rows=[], parity=[], bank_floor=[])
    md = []
    for attempt in ATTEMPTS_2D:
        p = LANE / 'runs' / attempt / 'archive' / 'output' / 'audit.json'
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        n = a['intervals']
        label = f'square {n}²'
        status = 'audited (development sources)' if a['passed'] else 'AUDIT GATES FAILED'
        summary['sources'].append(dict(attempt=attempt, path=str(p.relative_to(LANE)), sha256=sha(p),
                                       job_id=a['job_id'], commit=a['commit'], gpu=a['gpu'],
                                       gpu_uuid=a['gpu_uuid'], audit_passed=a['passed'], gates=a['gates'],
                                       error_checks=a['error_checks'],
                                       peak_device_bytes=(a.get('device_memory') or {}).get('peak_bytes_in_use')))
        summary['verdicts'].append(dict(attempt=attempt, mesh=label, **(a['verdict'] or {})))
        summary['bank_floor'].append(dict(attempt=attempt, mesh=label,
                                          worst=a['bank']['projection_floor']['worst'],
                                          median=a['bank']['projection_floor']['median']))
        summary['parity'] += [dict(attempt=attempt, **x) for x in a['parity']]
        rows = rows_for(a['table'], a['selections'], a['headline_variant'], label, a['job_id'], a['gpu'], status)
        summary['rows'] += [dict(attempt=attempt, **r) for r in rows]
        v = a['verdict']
        md.append(f"## {label} — `{attempt}`, job `{a['job_id']}`, {a['gpu']}, source `{a['commit'][:12]}`\n")
        md.append(f"Audit: **{'passed' if a['passed'] else 'FAILED'}** ({a['error_checks']} recomputed errors, "
                  f"worst difference {a['worst_error_difference']:.1e}). Bank projection floor: worst "
                  f"{pct(a['bank']['projection_floor']['worst'])} %. Bar (accurate arm `{v['arm']}` vs `cg_0.01`): "
                  f"worst same-grid {pct(v['worst_same_grid'])} % "
                  f"({'≤' if v['accuracy_bar_1pct'] else '>'} 1 %, stretch 0.5 % "
                  f"{'met' if v['stretch_bar_0p5pct'] else 'missed'}), speedup {v['speedup_total']:.2f}× "
                  f"({'≥' if v['speed_bar_5x'] else '<'} 5) → **{'BAR MET' if v['bar_met'] else 'BAR MISSED'}**.\n")
        md.append(table_md(rows, lambda r: r['family'] in ('nm-rom', 'linear-rom')))
        md.append('\nComparators and controls in the same job:\n')
        md.append(table_md(rows, lambda r: r['family'] not in ('nm-rom', 'linear-rom')))
    for attempt in ATTEMPTS_3D:
        p = LANE / 'runs' / attempt / 'archive' / 'output' / 'audit.json'
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        summary['sources'].append(dict(attempt=attempt, path=str(p.relative_to(LANE)), sha256=sha(p),
                                       job_id=a['job_id'], commit=a['commit'], gpu=a['gpu'],
                                       gpu_uuid=a['gpu_uuid'], audit_passed=a['passed'], gates=a['gates'],
                                       error_checks=a['error_checks'],
                                       peak_device_bytes=(a.get('device_memory') or {}).get('peak_bytes_in_use')))
        summary['parity'] += [dict(attempt=attempt, **x) for x in a['parity']]
        for n, m in a['meshes'].items():
            label = f'cube {n}³'
            status = 'audited (development sources)' if a['passed'] else 'AUDIT GATES FAILED'
            v = m['verdict']
            summary['verdicts'].append(dict(attempt=attempt, mesh=label, **v))
            summary['bank_floor'].append(dict(attempt=attempt, mesh=label, worst=m['mesh']['bank_floor_worst']))
            rows = rows_for(m['table'], m['selections'], a['headline_variant'], label, a['job_id'], a['gpu'], status)
            summary['rows'] += [dict(attempt=attempt, **r) for r in rows]
            md.append(f"## {label} — `{attempt}`, job `{a['job_id']}`, {a['gpu']}, source `{a['commit'][:12]}`\n")
            md.append(f"Audit: **{'passed' if a['passed'] else 'FAILED'}**. Bank floor worst "
                      f"{pct(m['mesh']['bank_floor_worst'])} %. Bar (accurate arm `{v['arm']}` vs `cg_0.01`): "
                      f"worst same-grid {pct(v['worst_same_grid'])} %, speedup {v['speedup_total']:.2f}× → "
                      f"**{'BAR MET' if v['bar_met'] else 'BAR MISSED'}**; fast arm `{v['fast_arm']['arm']}` "
                      f"{pct(v['fast_arm']['worst_same_grid'])} % / {v['fast_arm']['speedup_total']:.2f}×.\n")
            md.append(table_md(rows, lambda r: r['family'] in ('nm-rom', 'linear-rom')))
            md.append('\nComparators and controls in the same job:\n')
            md.append(table_md(rows, lambda r: r['family'] not in ('nm-rom', 'linear-rom')))
    OUT.mkdir(exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (OUT / 'tables.generated.md').write_text('\n'.join(md) + '\n')
    for v in summary['verdicts']:
        print(v)


if __name__ == '__main__':
    main()
