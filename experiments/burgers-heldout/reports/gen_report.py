"""Generate the lane report tables and the lane summary.json from the audited per-job summaries (checks/*-summary.json).

    python reports/gen_report.py

No number in the report is typed by hand: every table row below is read from an audited summary.
Writes reports/tables.generated.md and reports/summary.json (with the SHA256 of every source file).
"""
import hashlib
import json
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
CH = LANE / 'checks'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name):
    p = CH / f'{name}-summary.json'
    return (json.loads(p.read_text()), p) if p.exists() else (None, p)


def row(s, group, arm, role):
    g = s['groups'][group]
    t = g['table'].get(arm)
    if t is None:
        return None
    f = t.get('fastest_fom_at_least_as_accurate')
    fom = g['table'].get(f) if f else None
    return dict(job=s['attempt'], job_id=s['job_id'], mesh=s['intervals'], cohort=group, role=role, arm=arm,
                q=t.get('q'), M=t.get('M'), rule=t.get('rule'), gtol=t.get('gtol'),
                worst_evolved_percent=t['worst_evolved_percent'], median_evolved_percent=t['median_evolved_percent'],
                floor_percent=g.get('worst_floor_evolved_percent'), rom_ms=t['median_gpu_ms'],
                rom_host_ms=t['median_host_ms'], certified=t.get('certified_primary'),
                fom_setting=f, fom_ms=(fom or {}).get('median_gpu_ms'), fom_error_percent=(fom or {}).get('worst_evolved_percent'),
                speedup=t.get('speedup_vs_fastest_fom_at_least_as_accurate'),
                speedup_host=(t['speedup_host'].get(f) if f else None), stalled=t.get('stalled_exits'),
                reps=t.get('reps'), cases=g['cases'])


def fmt(x, d=3):
    return '—' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def main():
    out, sources, rows = {}, {}, []
    for name in ('bh2', 'bh2b', 'bh2c', 'bh3', 'bh4', 'bh5'):
        s, p = load(name)
        if s is None:
            continue
        sources[p.name] = sha(p)
        out[name] = dict(job_id=s['job_id'], commit=s['commit'], gpu=s['gpu'], mesh=s['intervals'],
                         failed_gates=s['failed_gates'], groups=list(s['groups']))
        cfg_heads = [a for a in (s['groups'].get('all', next(iter(s['groups'].values()))).get('headline') or {},) if a]
        for gname, g in s['groups'].items():
            if gname == 'all' and len(s['groups']) > 1:
                continue
            for arm, t in g['table'].items():
                if t['family'] != 'rom':
                    continue
                role = ('headline (pre-registered)' if (g.get('headline') or {}).get('arm') == arm else
                        'fast q=0' if (g.get('fast') or {}).get('arm') == arm else 'ladder')
                r = row(s, gname, arm, role)
                if r:
                    rows.append(r)
    lines = ['| job | mesh | cohort | role | arm | q | M | rule | cert. | worst evolved % | floor % | ROM ms | FOM (paper rule) | FOM ms | FOM % | speedup |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['job']} | {r['mesh']}² | {r['cohort']} | {r['role']} | `{r['arm']}` | {r['q']} | {r['M']} | {r['rule']} | "
                     f"{fmt(r['certified'])} | {fmt(r['worst_evolved_percent'])} | {fmt(r['floor_percent'])} | {fmt(r['rom_ms'], 1)} | "
                     f"`{r['fom_setting']}` | {fmt(r['fom_ms'], 1)} | {fmt(r['fom_error_percent'])} | {fmt(r['speedup'], 2)}× |")
    (LANE / 'reports/tables.generated.md').write_text('\n'.join(lines) + '\n')
    for name in ('compress.json', 'hfit_bh.json', 'directions.json'):
        p = CH / 'bh1' / name
        if p.exists():
            sources[f'bh1/{name}'] = sha(p)
    c = json.loads((CH / 'bh1/compress.json').read_text())
    summary = dict(lane='burgers-heldout', branch='exp/2026-09-21-burgers-heldout', jobs=out,
                   bank_selection=c['selection'], bank_floors_256_sel32={k: v['sel32']['worst_evolved_metric']
                                                                         for k, v in c['floors'].items()},
                   rows=rows, sources_sha256=sources)
    (LANE / 'reports/summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
