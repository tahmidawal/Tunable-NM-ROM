"""Generate reports/summary.json and reports/2026-09-25-t2-burgers-test.md from the audited summaries (no typed numbers).

    python reports/make_report.py --t256 checks/t256-summary.json --t1024 checks/t1024-summary.json \
        --t2048 checks/t2048-summary.json [--d256 checks/d256-summary.json] [--status final|provisional]

Test side: this lane's audited summaries (t2audit.py). Development side (read-only records, DESIGN.md section 7):
256^2 ops-timing-panel opt201 summary, 1024^2 / 2048^2 burgers-compare-hires p1024 / p2048e summaries. Both sides use
the Table 2 rule: one FOM per cell = the fastest FOM setting (median GPU ms, same job) whose worst evolved error is
<= the accurate NM-ROM setting's worst error (256^2 development: <= 0.17 %, inferred, see DESIGN.md section 7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
WT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees')
DEV = {256: WT / '2026-09-22-ops-timing-panel/experiments/ops-timing-panel/reports/summary.json',
       1024: WT / '2026-09-23-burgers-compare-hires/experiments/burgers-compare-hires/checks/p1024-summary.json',
       2048: WT / '2026-09-23-burgers-compare-hires/experiments/burgers-compare-hires/checks/p2048e-summary.json'}
B2T = {256: WT / '2026-09-25-burgers2d-test/experiments/burgers2d-test/checks/t256-summary.json',
       1024: WT / '2026-09-25-burgers2d-test/experiments/burgers2d-test/checks/t1024-summary.json',
       2048: WT / '2026-09-25-burgers2d-test/experiments/burgers2d-test/checks/t2048-summary.json'}
HEAD = 'q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2'
ACC = 'bank384_M1536_lat64_g1em06'
HEAD256_DEV = 'q0_M64_eqcert_g1em06_fastL4'          # the arm printed in Table 2 at 256^2 (opt201)
DEV256_THRESHOLD = 0.17                              # inferred, DESIGN.md section 7
FAM = {'fno': 'fno', 'unet': 'unet', 'transolver': 'tsol', 'deeponet': 'don'}
KEYS = ['accurate', 'head', 'fno', 'unet', 'tsol', 'don']
LABEL = {'accurate': 'NM-ROM accurate (span R\'=384)', 'head': 'NM-ROM second setting (head only, k=16)',
         'fno': 'FNO', 'unet': 'U-Net', 'tsol': 'Transolver', 'don': 'DeepONet'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt(x, nd=3):
    if x is None:
        return '—'
    if abs(x) >= 100:
        return f'{x:,.0f}'
    return f'{x:.{nd}g}' if abs(x) < 1 else f'{x:.{max(nd, 3)}g}'


def pick_fom(foms, thr):
    cand = [f for f in foms if f['gpu_ms'] is not None and f['worst'] <= thr]
    return min(cand, key=lambda f: f['gpu_ms']) if cand else None


def cell_cmp(path, table2_names, subset='full'):
    """A cell from a burgers-compare-hires style audited summary (this lane's t2audit.py or the source audit_cmp.py)."""
    s = json.loads(Path(path).read_text())
    arms = {a['name']: a for a in s['arms']}
    foms = [dict(name=a['name'], worst=a['worst_evolved_percent'], median=a['median_evolved_percent'], gpu_ms=a['gpu_ms'],
                 stalled=a.get('stalled', 0))
            for a in s['arms'] if a['family'] == 'fom' and (subset == 'full' or not a['name'].startswith('lean_'))]
    thr = arms[ACC]['worst_evolved_percent']
    fom = pick_fom(foms, thr)
    rows = {}
    names = dict(accurate=ACC, head=HEAD, **table2_names)
    for k, n in names.items():
        a = arms.get(n)
        if a is None:
            rows[k] = dict(arm=n, missing=True)
            continue
        rows[k] = dict(arm=n, family=a['family'], worst_percent=a['worst_evolved_percent'],
                       median_percent=a['median_evolved_percent'], gpu_ms=a['gpu_ms'],
                       speedup=(fom['gpu_ms'] / a['gpu_ms']) if (fom and a['gpu_ms']) else None,
                       own_fom=a.get('fom_gpu_full' if subset == 'full' else 'fom_gpu_b_panel'),
                       own_speedup=a.get('speedup_gpu_full' if subset == 'full' else 'speedup_gpu_b_panel'),
                       checkpoint_sha256=a.get('checkpoint_sha256'), trained_at=a.get('trained_at'),
                       epochs=a.get('epochs'))
    extra = []
    for a in s['arms']:
        if a['family'] in FAM and a['name'] not in table2_names.values():
            extra.append(dict(arm=a['name'], family=a['family'], worst_percent=a['worst_evolved_percent'],
                              median_percent=a['median_evolved_percent'], gpu_ms=a['gpu_ms'],
                              speedup=(fom['gpu_ms'] / a['gpu_ms']) if (fom and a['gpu_ms']) else None,
                              checkpoint_sha256=a.get('checkpoint_sha256')))
    return dict(source=str(path), sha256=sha(path), attempt=s.get('attempt'), job_id=s.get('job_id'), gpu=s.get('gpu'),
                commit=s.get('commit'), cohort=s.get('cohort'), cohort_sha256=s.get('cohort_sha256'),
                cases=len(arms[ACC]['evolved']), fom_subset=subset, fom_threshold_percent=thr,
                fom=fom, rows=rows, extra_operators=sorted(extra, key=lambda x: (x['family'], x['arm'])),
                fom_grid=sorted(foms, key=lambda f: f['gpu_ms'] or 1e99),
                failed_gates=s.get('failed_gates'), dropped=s.get('dropped'),
                bank_span=[dict(arm=b['arm'], rho_max=b['rho_max'], rho_max_k_ge_1=b['rho_max_k_ge_1'],
                                exceeds_bar=b['exceeds_bar']) for b in (s.get('bank_span') or {}).get('arms', [])])


def cell_opt201(path, table2_names):
    s = json.loads(Path(path).read_text())
    rows_ = {r['arm']: r for r in s['rows']}
    foms = [dict(name=r['arm'], worst=r['worst_evolved_percent'], median=r['median_evolved_percent'],
                 gpu_ms=r['median_gpu_ms']) for r in s['rows'] if r['family'] == 'fom']
    fom = pick_fom(foms, DEV256_THRESHOLD)
    rows = dict(accurate=dict(arm=None, missing=True, note='Table 2 prints "—" at 256^2'))
    for k, n in dict(head=HEAD256_DEV, **table2_names).items():
        r = rows_[n]
        rows[k] = dict(arm=n, family=r['family'], worst_percent=r['worst_evolved_percent'],
                       median_percent=r['median_evolved_percent'], gpu_ms=r['median_gpu_ms'],
                       speedup=fom['gpu_ms'] / r['median_gpu_ms'])
    return dict(source=str(path), sha256=sha(path), attempt=s.get('attempt'), job_id=s.get('job_id'), gpu=s.get('gpu'),
                cases=6, fom_subset='opt201 grid (8 non-lean settings)',
                fom_threshold_percent=DEV256_THRESHOLD, fom_threshold_note='inferred (Table 1 dev accurate at 256^2)',
                fom=fom, rows=rows, failed_gates=s.get('failed_gates'))


def paper_check(dev, paper):
    out = []
    for k in KEYS:
        pv = paper.get(k)
        r = dev['rows'].get(k, {})
        if pv is None or r.get('missing'):
            out.append(dict(row=k, paper=pv, recomputed=None, matches=(pv is None and r.get('missing', False))))
            continue
        e, x = r['worst_percent'], r['speedup']
        ok = (float(f'{e:.3g}') == pv[0] or round(e, 2) == pv[0]) and round(x, 2) == pv[1]
        out.append(dict(row=k, paper=pv, recomputed=[e, x], matches=bool(ok)))
    return out


def fom_reproduction(test_cell, L):
    """FOM settings' worst test errors against burgers2d-test's same-named settings (default compile mode)."""
    if not B2T[L].exists():
        return None
    t = json.loads(B2T[L].read_text())['table']
    per = []
    for f in test_cell['fom_grid']:
        if f['name'] in t:
            o = t[f['name']]['worst_evolved_percent']
            per.append(dict(setting=f['name'], this=f['worst'], burgers2d_test=o,
                            rel_diff=abs(f['worst'] - o) / max(abs(o), 1e-300) if o else abs(f['worst'])))
    return dict(source=str(B2T[L]), sha256=sha(B2T[L]), per_setting=per,
                max_rel_diff=max((p['rel_diff'] for p in per), default=None))


def dev_reproduction(d256, dev256):
    """d256 (this driver, dev6) against opt201 (the dev cell's driver, dev6): errors of the shared subjects."""
    o = {r['arm']: r for r in json.loads(Path(dev256['source']).read_text())['rows']}
    s = json.loads(Path(d256['source']).read_text())
    per = []
    for a in s['arms']:
        if a['name'] in o:
            per.append(dict(arm=a['name'], family=a['family'], this=a['worst_evolved_percent'],
                            opt201=o[a['name']]['worst_evolved_percent'],
                            rel_diff=abs(a['worst_evolved_percent'] - o[a['name']]['worst_evolved_percent']) /
                            max(o[a['name']]['worst_evolved_percent'], 1e-12)))
    return dict(per_arm=per, max_rel_diff=max((p['rel_diff'] for p in per), default=None))


def table2_names(opsfile):
    ops = json.loads((LANE / opsfile).read_text())['operators']
    return {FAM[o['family']]: o['name'] for o in ops if o['table2_row']}


def md_cell(L, c, lines):
    t, d = c['test'], c['dev']
    lines.append(f'## ${L}^2$\n')
    lines.append(f"Test: attempt `{t['attempt']}`, job {t['job_id']}, {t['gpu']}, commit `{(t['commit'] or '')[:10]}`, "
                 f"{t['cases']} test cases, summary `{Path(t['source']).name}` sha256 `{t['sha256']}`, failed gates: "
                 f"{', '.join(t['failed_gates']) or 'none'}.  ")
    lines.append(f"Development: job {d['job_id']}, {d['gpu']}, {d['cases']} cases, `{Path(d['source']).name}` sha256 "
                 f"`{d['sha256']}`.\n")
    ft, fd = t['fom'], d['fom']
    lines.append(f"FOM of the cell (Table 2 rule): **test** `{ft['name']}` — worst {fmt(ft['worst'], 4)} %, "
                 f"{fmt(ft['gpu_ms'], 4)} ms (threshold = accurate worst {fmt(t['fom_threshold_percent'], 4)} %); "
                 f"**development** `{fd['name']}` — worst {fmt(fd['worst'], 4)} %, {fmt(fd['gpu_ms'], 4)} ms "
                 f"(threshold {fmt(d['fom_threshold_percent'], 4)} %{', inferred' if L == 256 else ''}).\n")
    lines.append('| method | checkpoint / arm | test worst % | test median % | test ms | test × (cell FOM) | test own comparator (×) | dev worst % | dev × | paper dev (worst %, ×) |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for k in KEYS:
        r, q = t['rows'][k], d['rows'].get(k, {})
        pv = c['paper'].get(k)
        own = f"`{r.get('own_fom')}` ({fmt(r.get('own_speedup'))})" if r.get('own_fom') else '—'
        lines.append(f"| {LABEL[k]} | `{r['arm']}` | {fmt(r.get('worst_percent'), 4)} | {fmt(r.get('median_percent'), 4)} | "
                     f"{fmt(r.get('gpu_ms'), 4)} | {fmt(r.get('speedup'))}× | {own} | "
                     f"{fmt(q.get('worst_percent'), 4) if not q.get('missing') else '—'} | "
                     f"{(fmt(q.get('speedup')) + '×') if not q.get('missing') else '—'} | "
                     f"{('%s, %s×' % tuple(pv)) if pv else '—'} |")
    lines.append('')
    if L == 256 and c.get('dev_same_driver'):
        s = c['dev_same_driver']
        lines.append(f"Same-driver development context (`d256`, job {s['job_id']}, {s['gpu']}, dev6, sha256 `{s['sha256']}`; "
                     f"FOM `{s['fom']['name']}` {fmt(s['fom']['worst'], 4)} %, {fmt(s['fom']['gpu_ms'], 4)} ms; failed gates: "
                     f"{', '.join(s['failed_gates']) or 'none'}):\n")
        lines.append('| method | arm | dev worst % | dev ms | dev × |')
        lines.append('|---|---|---|---|---|')
        for k in KEYS:
            r = s['rows'][k]
            lines.append(f"| {LABEL[k]} | `{r['arm']}` | {fmt(r.get('worst_percent'), 4)} | {fmt(r.get('gpu_ms'), 4)} | {fmt(r.get('speedup'))}× |")
        lines.append('')
    if c.get('test_b_panel'):
        b = c['test_b_panel']
        lines.append(f"Secondary rule on the dev cell's non-lean grid (test): FOM `{b['fom']['name']}` "
                     f"{fmt(b['fom']['worst'], 4)} %, {fmt(b['fom']['gpu_ms'], 4)} ms → " +
                     ', '.join(f"{LABEL[k]} {fmt(b['rows'][k]['speedup'])}×" for k in KEYS) + '.\n')
    if t['extra_operators']:
        lines.append('Other operator checkpoints timed in the same test job (not Table 2 rows):\n')
        lines.append('| checkpoint | family | test worst % | test median % | test ms | test × (cell FOM) |')
        lines.append('|---|---|---|---|---|---|')
        for r in t['extra_operators']:
            lines.append(f"| `{r['arm']}` | {r['family']} | {fmt(r['worst_percent'], 4)} | {fmt(r['median_percent'], 4)} | "
                         f"{fmt(r['gpu_ms'], 4)} | {fmt(r['speedup'])}× |")
        lines.append('')
    for b in t['bank_span']:
        lines.append(f"Span arm held-out rho (in-job): `{b['arm']}` rho_max {fmt(b['rho_max'], 4)} (k≥1: "
                     f"{fmt(b['rho_max_k_ge_1'], 4)}), {'EXCEEDS' if b['exceeds_bar'] else 'within'} the 0.116 bar.\n")
    fr = c.get('fom_reproduction')
    if fr:
        lines.append(f"FOM reproduction vs burgers2d-test (`{Path(fr['source']).name}` sha256 `{fr['sha256'][:12]}…`): "
                     f"{len(fr['per_setting'])} settings, max relative difference of the worst test error "
                     f"{fmt(fr['max_rel_diff'], 3)}.\n")
    pc = c['paper_check']
    bad = [p['row'] for p in pc if not p['matches']]
    lines.append(f"Paper Table 2 development values recomputed from the records: "
                 f"{'all match' if not bad else 'MISMATCH in ' + ', '.join(bad)}.\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t256', required=True)
    p.add_argument('--t1024', required=True)
    p.add_argument('--t2048', required=True)
    p.add_argument('--d256', default=None)
    p.add_argument('--status', default='provisional')
    a = p.parse_args()
    paper = json.loads((HERE / 'paper_table2_dev.json').read_text())['cells']
    cells = {}
    for L, tp, opsf in ((256, a.t256, 'operators-t256.json'), (1024, a.t1024, 'operators-t1024.json'),
                        (2048, a.t2048, 'operators-t2048.json')):
        names = table2_names(opsf)
        test = cell_cmp(LANE / tp, names)
        dev = cell_opt201(DEV[L], names) if L == 256 else cell_cmp(DEV[L], names)
        c = dict(test=test, dev=dev, paper=paper[str(L)], paper_check=paper_check(dev, paper[str(L)]),
                 fom_reproduction=fom_reproduction(test, L))
        if L == 256:
            c['test_b_panel'] = cell_cmp(LANE / tp, names, subset='b_panel')
            if a.d256:
                c['dev_same_driver'] = cell_cmp(LANE / a.d256, names)
                c['dev_same_driver_reproduction'] = dev_reproduction(c['dev_same_driver'], dev)
        cells[str(L)] = c
    out = dict(status=a.status, lane='t2-burgers-test', design='experiments/t2-burgers-test/DESIGN.md', cells=cells,
               rule='one FOM per cell: fastest FOM setting (median GPU ms, same job) with worst evolved error <= the '
                    'accurate NM-ROM setting\'s worst; speedup = FOM ms / row ms, same job')
    (HERE / 'summary.json').write_text(json.dumps(out, indent=1, default=float) + '\n')

    L_ = []
    L_.append('# Table 2 (Burgers 2D, NM-ROM vs neural operators) on the 64 held-out test cases\n')
    L_.append(f"Status: **{a.status}**. Every number below is generated by `reports/make_report.py` from the audited "
              "job summaries (test) and the read-only development records; nothing is typed by hand. Test = the 64 "
              "held-out cases `test64`; development = the six cases the paper's Table 2 uses today. Same frozen "
              "NM-ROM settings, same frozen operator checkpoints, no retraining; speedups are same-job ratios against "
              "the cell's one FOM.\n")
    L_.append('| mesh | method | test worst % | test × | dev worst % | dev × |')
    L_.append('|---|---|---|---|---|---|')
    for L in ('256', '1024', '2048'):
        c = cells[L]
        for k in KEYS:
            r, q = c['test']['rows'][k], c['dev']['rows'].get(k, {})
            L_.append(f"| {L}² | {LABEL[k]} | {fmt(r.get('worst_percent'))} | {fmt(r.get('speedup'))}× | "
                      f"{fmt(q.get('worst_percent')) if not q.get('missing') else '—'} | "
                      f"{(fmt(q.get('speedup')) + '×') if not q.get('missing') else '—'} |")
        L_.append(f"| {L}² | FOM of the cell | {fmt(c['test']['fom']['worst'])} (`{c['test']['fom']['name']}`) | 1× | "
                  f"{fmt(c['dev']['fom']['worst'])} (`{c['dev']['fom']['name']}`) | 1× |")
    L_.append('')
    for L in ('256', '1024', '2048'):
        md_cell(int(L), cells[L], L_)
    L_.append('## Glossary\n')
    L_.append('- **test64 / test cases**: 64 Burgers initial conditions drawn with seed 20260916, never used to train, '
              'select or tune anything; the paper\'s Table 1 already uses them.')
    L_.append('- **dev6 / development cases**: the six cases the paper\'s Table 2 currently uses; they were looked at '
              'while choosing settings.')
    L_.append('- **worst % / median %**: over the cases, the largest (median) value of the relative error '
              '$\\max_t \\lVert u-u_{\\rm ref}\\rVert/\\lVert u_0\\rVert$ over the five evolved output times, against a '
              'tightly converged full-order solve on the same mesh in the same job.')
    L_.append('- **ms**: median GPU time of one query (initial field on the GPU → six output fields on the GPU).')
    L_.append('- **×, cell FOM**: speedup = the cell FOM\'s ms / the row\'s ms, both from the same job. The cell FOM is the '
              'fastest of the 15 Newton–BiCGStab settings whose worst error is at most the accurate NM-ROM setting\'s.')
    L_.append('- **own comparator**: the fastest FOM setting at least as accurate as that row itself.')
    L_.append('- **NM-ROM accurate (span R\'=384)**: 384 unknowns, the first 384 columns of the rotated spatial bank, '
              'solved by Levenberg–Marquardt through an empirical quadrature (`lat64`).')
    L_.append('- **second setting (head only, k=16)**: 16 latent unknowns through the trained two-layer head, EQ rule, '
              'no correction weights.')
    L_.append('- **rho / 0.116 bar**: the quadrature-rule check on held-out states; above the bar the rule is flagged.')
    L_.append('- **FNO, U-Net, Transolver, DeepONet**: neural operators trained once (3000 s budget per family) and '
              'frozen; at 1024²/2048² trained at that mesh, at 256² the opt201 checkpoints.')
    L_.append('- **d256**: the 256² job run on the development cases with this lane\'s driver, for a like-for-like '
              'development value (the paper\'s 256² cell came from a different driver).')
    (HERE / '2026-09-25-t2-burgers-test.md').write_text('\n'.join(L_) + '\n')
    print('report written', {L: (cells[L]['test']['fom']['name'], [p['matches'] for p in cells[L]['paper_check']]) for L in cells})


if __name__ == '__main__':
    main()
