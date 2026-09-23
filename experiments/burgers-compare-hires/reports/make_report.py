"""Generate the lane's report and its pinned summary from the audited per-job summaries. No number is typed.

    python reports/make_report.py checks/p1024-summary.json [checks/p2048-summary.json ...] \
        --out-md reports/<date>-burgers-compare-hires.md --out-json reports/summary.json [--status provisional]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FAMILY = dict(nmrom='NM-ROM (ours)', bankspan='NM-ROM bank-span (ours)', pod='POD-LSPG', qman='quadratic manifold', fno='FNO', unet='U-Net',
              transolver='Transolver', deeponet='DeepONet', fom='FOM (Newton–BiCGStab)')
ORDER = ['nmrom', 'bankspan', 'pod', 'qman', 'fno', 'unet', 'transolver', 'deeponet']


def f4(x):
    return '—' if x is None else f'{x:.4f}'


def ms(x):
    return '—' if x is None else (f'{x:,.1f}' if x >= 100 else f'{x:.2f}')


def sp(x):
    return '—' if x is None else (f'{x:.2f}×' if x >= .1 else f'{x:.3f}×')


def label(a, roles):
    inv = {v: k for k, v in roles.items()}
    r = inv.get(a['name'])
    if r == 'nmrom_fast':
        return 'NM-ROM fast (q=0, M=64)'
    if r == 'nmrom_accurate':
        return 'reference only: NM-ROM with corrections (q=256, M=1088)'
    if r == 'nmrom_accurate_robust':
        return 'reference only: NM-ROM with corrections (q=256), robust rule (1 exact step)'
    if a['family'] == 'bankspan':
        return f"NM-ROM bank-span R'={a['k']} (M={a['M']}, `{a['rule']}` m={a['m']})"
    if a['family'] == 'pod':
        return f"POD-LSPG k={a['k']} (M={a['M']})"
    if a['family'] == 'qman':
        return f"quadratic manifold r={a['k']} (M={a['M']})"
    return FAMILY.get(a['family'], a['family']) + (f" (`{a['name']}`)" if a['family'] in ('fno', 'unet', 'transolver', 'deeponet') else '')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summaries', nargs='+')
    p.add_argument('--out-md', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--status', default='final')
    a = p.parse_args()
    meshes, lines = [], []
    for path in a.summaries:
        raw = Path(path).read_bytes()
        s = json.loads(raw)
        roles = s.get('roles') or {}
        bsr = {b['arm']: b for b in ((s.get('bank_span') or {}).get('arms') or [])}
        rows = []
        for fam in ORDER:
            for x in sorted((x for x in s['arms'] if x['family'] == fam and x['phase'] != 'P'),
                            key=lambda x: (x.get('k') or 0, x['name'])):
                rows.append(dict(method=label(x, roles), arm=x['name'], family=x['family'], unknowns=x.get('unknowns'),
                                 worst_evolved_percent=x['worst_evolved_percent'],
                                 median_evolved_percent=x['median_evolved_percent'], gpu_ms=x['gpu_ms'],
                                 complete_ms=x['complete_ms'], fom=x.get('fom_gpu_full'), fom_ms=x.get('fom_ms_gpu_full'),
                                 fom_worst_evolved_percent=x.get('fom_worst_gpu_full'), speedup=x.get('speedup_gpu_full'),
                                 speedup_complete=x.get('speedup_complete_full'), fom_complete=x.get('fom_complete_full'),
                                 epochs=x.get('epochs'), stop_reason=x.get('stop_reason'), trained_at=x.get('trained_at'),
                                 parameters=x.get('parameters'),
                                 rule_status=(('held-out rho_max %.4f over all states (k>=0, the house convention) -- %s the %.3f bar; '
                                               '%.4f over time-stepped states (k>=1); %d states' % (
                                                   bsr[x['name']]['rho_max'], 'EXCEEDS' if bsr[x['name']]['exceeds_bar'] else 'within',
                                                   bsr[x['name']]['bar'], bsr[x['name']]['rho_max_k_ge_1'], bsr[x['name']]['states']))
                                              if x['name'] in bsr else x.get('rule_status')),
                                 stalled_exits=x.get('stalled', 0) + x.get('timed_stalled', 0),
                                 timing_withheld=x.get('timing_withheld')))
        foms = sorted((x for x in s['arms'] if x['family'] == 'fom'), key=lambda x: x['gpu_ms'])
        meshes.append(dict(intervals=s['intervals'], attempt=s['attempt'], job_id=s['job_id'], commit=s['commit'],
                           gpu=s['gpu'], summary_file=path, summary_sha256=hashlib.sha256(raw).hexdigest(),
                           failed_gates=s['failed_gates'], rows=rows,
                           fom_grid=[dict(arm=x['name'], worst_evolved_percent=x['worst_evolved_percent'],
                                          median_evolved_percent=x['median_evolved_percent'], gpu_ms=x['gpu_ms'],
                                          complete_ms=x['complete_ms']) for x in foms],
                           dropped=s['dropped'], notes=s.get('notes', []), gates={k: v.get('passed') for k, v in s['gates'].items()},
                           quadratic_manifold_fits=s.get('quadratic_manifold_fits')))
    Path(a.out_json).write_text(json.dumps(dict(status=a.status, meshes=meshes,
                                                 error_convention=s['error_convention'], fom_rule=s['fom_rule']),
                                            indent=1) + '\n')

    L = lines.append
    L('# Burgers 2D comparison with other methods at higher resolution')
    L('')
    L(f"Status: **{a.status}**. Generated by `reports/make_report.py` from the independently audited summaries "
      f"({', '.join('`' + Path(m['summary_file']).name + '`' for m in meshes)}); no number below is typed by hand. "
      "Every row of a mesh comes from ONE cluster job (one allocation, one GPU), and every speedup divides two times "
      "from that job.")
    L('')
    L('**Error**: same-grid evolved — the largest, over the five evolved output times, of '
      '$\\lVert u - u_{\\rm ref}\\rVert_2/\\lVert u_0\\rVert_2$, with $u_{\\rm ref}$ the tight Newton–BiCGStab solve '
      '(`fft_tight`) computed in the same job at the same mesh; worst and median over the six development cases (dev6). '
      '**Speedup**: the GPU time of the fastest tested FOM setting whose worst error is no larger than the row\'s, '
      'divided by the row\'s GPU time. Below 1× the row is slower than that FOM setting.')
    for m in meshes:
        n = m['intervals']
        L('')
        L(f"## ${n}^2$ — attempt `{m['attempt']}`, job {m['job_id']}, {m['gpu']}, commit `{m['commit'][:10] if m['commit'] else '?'}`")
        L('')
        L(f"Summary `{m['summary_file']}` (SHA256 `{m['summary_sha256']}`). Failed gates: "
          f"{', '.join('`' + g + '`' for g in m['failed_gates']) if m['failed_gates'] else 'none'}.")
        L('')
        if m['failed_gates']:
            L(f"> **This mesh FAILED gates {', '.join(m['failed_gates'])}; its rows are reported as measured but the "
              "mesh is not accepted until each failure is dispositioned below.**")
            L('')
        L('| method | unknowns | worst % | median % | GPU ms | FOM chosen (its GPU ms, worst %) | speedup | status / notes |')
        L('|---|---|---|---|---|---|---|---|')
        for r in m['rows']:
            fom = (f"`{r['fom']}` ({ms(r['fom_ms'])}, {f4(r['fom_worst_evolved_percent'])})" if r['fom'] else
                   'none at least as accurate')
            unk = f"{r['unknowns']:,}" if isinstance(r['unknowns'], int) else '— (no online solve)'
            note = []
            if r['rule_status']:
                note.append(r['rule_status'])
            if r['trained_at']:
                note.append(f"trained at {r['trained_at']}" + (f", {r['epochs']} epochs ({r['stop_reason']})" if r['epochs'] else ''))
            if r['timing_withheld']:
                note.append('timing withheld: ' + r['timing_withheld'])
                fom, r['speedup'] = '—', None
            if r['stalled_exits']:
                note.append(f"{r['stalled_exits']} stalled exits")
            L(f"| {r['method']} | {unk} | {f4(r['worst_evolved_percent'])} | {f4(r['median_evolved_percent'])} | "
              f"{ms(r['gpu_ms'])} | {fom} | {sp(r['speedup'])} | {'; '.join(note) if note else ''} |")
        L('')
        # generated findings: never typed. "ours" = the head-only fast arm and the bank-span arms whose rule is within
        # the bar (the q=256 rows are reference only by the coordinator's direction).
        ours = [r for r in m['rows'] if r['gpu_ms'] is not None and (
            r['method'].startswith('NM-ROM fast') or (r['family'] == 'bankspan' and 'within' in (r['rule_status'] or '')))]
        others = [r for r in m['rows'] if r['gpu_ms'] is not None and r['family'] not in ('nmrom', 'bankspan')]
        L('**Generated findings at this mesh** (from the table above):')
        L('')
        for r in ours:
            L(f"- {r['method']}: worst {f4(r['worst_evolved_percent'])} % at {ms(r['gpu_ms'])} ms; "
              + (f"{sp(r['speedup'])} against `{r['fom']}`" if r['fom'] else 'no FOM setting at least as accurate'))
            dom = [o for o in others if o['worst_evolved_percent'] <= r['worst_evolved_percent'] and o['gpu_ms'] <= r['gpu_ms']]
            L('  - baselines both at least as accurate AND at least as fast: ' +
              (', '.join(f"{o['method']} ({f4(o['worst_evolved_percent'])} %, {ms(o['gpu_ms'])} ms)" for o in dom) if dom else 'none'))
        beat = [o for o in others if o['speedup'] is not None and o['speedup'] > 1]
        L('- baselines faster than the FOM setting the rule assigns them: ' +
          (', '.join(f"{o['method']} {sp(o['speedup'])}" for o in beat) if beat else 'none'))
        L('')
        L(f"FOM candidate grid (the rule chooses among all of these; ${n-1}^2$ = {(n-1)**2:,} unknowns):")
        L('')
        L('| FOM setting | worst % | median % | GPU ms | complete ms |')
        L('|---|---|---|---|---|')
        for f_ in m['fom_grid']:
            L(f"| `{f_['arm']}` | {f4(f_['worst_evolved_percent'])} | {f4(f_['median_evolved_percent'])} | "
              f"{ms(f_['gpu_ms'])} | {ms(f_['complete_ms'])} |")
        ops = [r for r in m['rows'] if r['family'] in ('fno', 'unet', 'transolver', 'deeponet')]
        if ops:
            L('')
            L('Operator training record (equal 3000 s wall budget per arm, the $256^2$ protocol):')
            L('')
            L('| arm | trained at | epochs | stop reason | parameters |')
            L('|---|---|---|---|---|')
            for r in ops:
                L(f"| `{r['arm']}` | {r['trained_at'] or '—'} | {r['epochs'] if r['epochs'] is not None else '—'} | "
                  f"{r['stop_reason'] or '—'} | {r['parameters'] if r['parameters'] else '—'} |")
        if m['notes']:
            L('')
            L('Notes: ' + '; '.join(m['notes']))
        if m['dropped']:
            L('')
            L('Dropped (not reported as rows): ' + '; '.join(f"`{d['name']}` ({d['phase']}: {d['reason'][:120]})"
                                                            for d in m['dropped']))
        L('')
        L('Gates: ' + ', '.join(f"`{k}`={v}" for k, v in m['gates'].items()) + '.')
    L('')
    L('## Glossary')
    L('')
    L('- **unknowns** — the number of values solved for online per time step: $K+q$ for the NM-ROM, $k$ for POD-LSPG, '
      '$r$ for the quadratic manifold, $(L-1)^2$ grid values for the FOM. Operators solve nothing online.')
    L('- **worst % / median %** — the same-grid evolved error defined above, worst and median over the six cases.')
    L('- **GPU ms** — median over 6 cases × 5 repetitions of one query (initial field already on the GPU to all six '
      'output fields on the GPU).')
    L('- **FOM chosen** — the fastest tested full-order setting at least as accurate as the row, in the same job.')
    L('- **q, M** — correction rank and number of weak test functions of the NM-ROM. **k / r** — POD / quadratic-manifold dimension.')
    L('- **dev6** — the six development cases `params_draw(7090702,4)` + `params_draw(911702,2)` used by every existing panel.')
    L('- **FOM settings** — `nt*` = audited Newton–BiCGStab (ntol, dt), `lean_*` = the same solver without a per-Newton '
      'diagnostic; `fft_tight`/`lean_tight` = the tight reference settings.')
    Path(a.out_md).write_text('\n'.join(lines) + '\n')
    print('written', a.out_md, a.out_json)


if __name__ == '__main__':
    main()
