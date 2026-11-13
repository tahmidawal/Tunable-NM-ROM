"""iso-error: every reported number, from the saved result.json files (PROTOCOL.md section 8).
    python summarise.py   ->  results/summary.json, reports/tables.md
Reads results/repro256/result.json, results/sweep256/result.json (if present), vendor/ref/t256-result.json and
inputs/operators.json (operator numbers transcribed from the two status files with file/line citations).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEVELS = [0.2, 0.1, 0.05, 0.026]


def arm_stats(r, name):
    q = [x for x in r['quick'] if x['name'] == name]
    assert len(q) == 64 and sorted(x['case'] for x in q) == list(range(64)), (name, len(q))
    err = 100 * np.array([x['same_grid_evolved'] for x in q])
    inv = [x for x in r['invocations'] if x['name'] == name and x['phase'] in ('romA1', 'romA2', 'fomB')]
    t = 1e3 * np.array([x['gpu_seconds'] for x in inv])
    out = dict(name=name, cases=len(q), median=float(np.median(err)), mean=float(err.mean()),
               p90=float(np.quantile(err, .9)), worst=float(err.max()), worst_case=int(err.argmax()),
               invocations=len(t), ms_median=float(np.median(t)), ms_min=float(t.min()), ms_max=float(t.max()),
               ms_p05=float(np.quantile(t, .05)), ms_p95=float(np.quantile(t, .95)),
               ms_phase={ph: float(np.median([1e3 * x['gpu_seconds'] for x in inv if x['phase'] == ph]))
                         for ph in sorted({x['phase'] for x in inv})},
               identical_to_quick=bool(all(x['identical_to_quick'] for x in inv)))
    if 'total_iterations' in q[0]:
        it = np.array([x['total_iterations'] for x in q])
        out.update(lm_iterations_median=float(np.median(it)), lm_iterations_max=int(it.max()),
                   stalled_exits=int(sum(x['stalled_exits'] for x in q)),
                   budget_exits=int(sum(x['budget_exits'] for x in q)))
    rc = [x for x in r.get('recon_invocations', []) if x['name'] == name]
    if rc:
        tr = 1e3 * np.array([x['gpu_seconds'] for x in rc])
        out.update(recon_invocations=len(tr), recon_ms_median=float(np.median(tr)), recon_ms_min=float(tr.min()),
                   recon_ms_max=float(tr.max()), solve_ms_derived=float(np.median(t) - np.median(tr)),
                   recon_identical_to_query=int(sum(x['identical_to_quick'] for x in rc)))
    return out


def job(r):
    g = r['gates']
    return dict(job_id=r['job_id'], host=r['host'], gpu=r['gpu'], backend=r['backend'], commit=r['commit'],
                complete=r['complete'], elapsed_seconds=r.get('elapsed_seconds'), cohort_sha256=r['physical_sha256'],
                gates={k: (v['passed'] if isinstance(v, dict) else all(x['passed'] for x in v)) for k, v in g.items()},
                drift_ratio_range=[min(d['ratio'] for d in g['drift']['rows']), max(d['ratio'] for d in g['drift']['rows'])],
                neighbour_ratio_max=max([x['ratio'] for x in g['neighbour']['rows'] if x['evaluable']] or [None]))


def sig4(x):
    return f'{x:.4g}'


def main():
    S = dict(levels=LEVELS)
    ref = json.loads((HERE / 'vendor/ref/t256-result.json').read_text())
    nm = lambda Rp, g=True: f'R{Rp}_lin_M{4 * Rp}_lat64_g0p01_fast_chol_clip_lamcarry_pred2__eng' + ('_graphs' if g else '')
    FOM = 'lean_nt3e-3_l3e-3_dt005__graphs'
    S['stored'] = dict(job=job(ref), arms={str(Rp): arm_stats(ref, nm(Rp)) for Rp in (384, 96, 128)}, fom=arm_stats(ref, FOM))
    lines = []
    runs = {}
    for a in ('repro256', 'sweep256'):
        p = HERE / 'results' / a / 'result.json'
        if p.exists():
            runs[a] = json.loads(p.read_text())
            assert runs[a]['complete'] and runs[a]['backend'] == 'gpu', a
            S[a] = dict(job=job(runs[a]))
    # ---- reproduction check
    if 'repro256' in runs:
        r = runs['repro256']
        S['repro256']['arms'] = {str(Rp): arm_stats(r, nm(Rp)) for Rp in (384, 96, 128)}
        S['repro256']['fom'] = arm_stats(r, FOM)
        chk = []
        for Rp in (384, 96, 128):
            a, b = S['stored']['arms'][str(Rp)], S['repro256']['arms'][str(Rp)]
            chk.append(dict(width=Rp, median_stored=a['median'], median_repro=b['median'], worst_stored=a['worst'],
                            worst_repro=b['worst'], errors_bit_identical=bool(a['median'] == b['median'] and a['worst'] == b['worst']),
                            errors_match_4sig=bool(sig4(a['median']) == sig4(b['median']) and sig4(a['worst']) == sig4(b['worst'])),
                            ms_stored=a['ms_median'], ms_repro=b['ms_median'], ms_ratio=b['ms_median'] / a['ms_median'],
                            time_within_10pct=bool(abs(b['ms_median'] / a['ms_median'] - 1) <= .1)))
        S['reproduction_check'] = dict(rows=chk, passed=bool(all(c['errors_match_4sig'] and c['time_within_10pct'] for c in chk)
                                                             and all(S['repro256']['job']['gates'].values())))
        lines += ['## Reproduction of the existing row', '',
                  f"Stored: job {S['stored']['job']['job_id']} ({S['stored']['job']['host']}, {S['stored']['job']['gpu']}). "
                  f"Reproduction: job {S['repro256']['job']['job_id']} ({S['repro256']['job']['host']}, {S['repro256']['job']['gpu']}), "
                  f"same driver/config/checkpoint.", '',
                  '| width | median % stored | median % repro | worst % stored | worst % repro | errors bit-identical | ms stored | ms repro | ratio |',
                  '|---|---|---|---|---|---|---|---|---|']
        for c in chk:
            lines.append(f"| {c['width']} | {c['median_stored']:.6f} | {c['median_repro']:.6f} | {c['worst_stored']:.6f} | "
                         f"{c['worst_repro']:.6f} | {'yes' if c['errors_bit_identical'] else 'no'} | {c['ms_stored']:.3f} | "
                         f"{c['ms_repro']:.3f} | {c['ms_ratio']:.4f} |")
        a, b = S['stored']['fom'], S['repro256']['fom']
        lines += ['', f"Full-order solver of Table 1 (`{FOM}`): stored {a['ms_median']:.3f} ms, reproduction {b['ms_median']:.3f} ms.",
                  f"Gates of the reproduction job: {S['repro256']['job']['gates']}. Check passed: {S['reproduction_check']['passed']}.", '']
    # ---- sweep
    if 'sweep256' in runs:
        r = runs['sweep256']
        setup = {x['arm']: x for x in r['arm_setup']}
        widths = sorted({x['R_prime'] for x in r['arm_setup']})
        rows = []
        for Rp in widths:
            pri = [n for n, x in setup.items() if x['R_prime'] == Rp and x['role'] == 'primary-mode']
            oth = [n for n, x in setup.items() if x['R_prime'] == Rp and x['role'] == 'other-mode']
            assert len(pri) == 1 and len(oth) == 1
            a, o = arm_stats(r, pri[0]), arm_stats(r, oth[0])
            rows.append(dict(width=Rp, M=setup[pri[0]]['M'], primary_mode='graphs' if setup[pri[0]]['graphs'] else 'default',
                             primary=a, other=o,
                             modes_error_identical_4sig=bool(sig4(a['median']) == sig4(o['median']) and sig4(a['worst']) == sig4(o['worst']))))
        S['sweep256']['widths'] = rows
        S['sweep256']['fom'] = arm_stats(r, FOM)
        cons = []
        for Rp in (384, 96, 128):
            a = S['stored']['arms'][str(Rp)]
            b = next(x for x in rows if x['width'] == Rp)['primary']
            c = S.get('repro256', {}).get('arms', {}).get(str(Rp))
            cons.append(dict(width=Rp, errors_match_4sig=bool(sig4(a['median']) == sig4(b['median']) and sig4(a['worst']) == sig4(b['worst'])),
                             errors_bit_identical=bool(a['median'] == b['median'] and a['worst'] == b['worst']),
                             ms_stored=a['ms_median'], ms_repro=c['ms_median'] if c else None, ms_sweep=b['ms_median']))
        S['sweep_consistency'] = cons
        j = S['sweep256']['job']
        lines += ['## Per-width results (test64, 256², primary compile mode)', '',
                  f"Sweep job {j['job_id']} ({j['host']}, {j['gpu']}); gates {j['gates']}; drift ratio A2/A1 in "
                  f"[{j['drift_ratio_range'][0]:.4f}, {j['drift_ratio_range'][1]:.4f}]; neighbour ratio max {j['neighbour_ratio_max']:.4f}.", '',
                  '| width R\' | M | median % | mean % | p90 % | worst % | time ms (median) | min–max ms | recon ms | solve ms (derived) | LM iters/query (median, max) | stalled exits | mode |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for x in rows:
            a = x['primary']
            lines.append(f"| {x['width']} | {x['M']} | {a['median']:.4f} | {a['mean']:.4f} | {a['p90']:.4f} | {a['worst']:.4f} | "
                         f"{a['ms_median']:.2f} | {a['ms_min']:.2f}–{a['ms_max']:.2f} | {a.get('recon_ms_median', float('nan')):.2f} | "
                         f"{a.get('solve_ms_derived', float('nan')):.2f} | {a['lm_iterations_median']:.0f}, {a['lm_iterations_max']} | "
                         f"{a['stalled_exits']} | {x['primary_mode']} |")
        lines += ['', '### Other compile mode (secondary; not used in any table)', '',
                  '| width R\' | primary mode | primary ms | other-mode ms | other/primary | errors equal to 4 sig. digits |', '|---|---|---|---|---|---|']
        for x in rows:
            lines.append(f"| {x['width']} | {x['primary_mode']} | {x['primary']['ms_median']:.2f} | {x['other']['ms_median']:.2f} | "
                         f"{x['other']['ms_median'] / x['primary']['ms_median']:.3f} | {'yes' if x['modes_error_identical_4sig'] else 'no'} |")
        lines += ['', '### Consistency of widths 384 / 96 / 128 across jobs', '',
                  '| width | errors equal to stored (4 sig.) | bit-identical | ms stored (4319328) | ms reproduction | ms sweep |', '|---|---|---|---|---|---|']
        for c in cons:
            lines.append(f"| {c['width']} | {'yes' if c['errors_match_4sig'] else 'NO'} | {'yes' if c['errors_bit_identical'] else 'no'} | "
                         f"{c['ms_stored']:.3f} | {c['ms_repro'] if c['ms_repro'] is None else round(c['ms_repro'], 3)} | {c['ms_sweep']:.3f} |")
        f = S['sweep256']['fom']
        lines += ['', f"Full-order solver of Table 1 in the sweep job: {f['ms_median']:.3f} ms (stored {S['stored']['fom']['ms_median']:.3f} ms).", '']
        # ---- iso-error
        opf = HERE / 'inputs/operators.json'
        ops = json.loads(opf.read_text()) if opf.exists() else dict(test_evaluated=[], best_validation_only=None)
        iso = []
        for lv in LEVELS:
            thr = 0.0265 if lv == 0.026 else lv
            ok = [x for x in rows if (x['primary']['median'] < thr if lv == 0.026 else x['primary']['median'] <= thr)]
            ours = min(ok, key=lambda x: x['width']) if ok else None
            fastest_ours = min(ok, key=lambda x: x['primary']['ms_median']) if ok else None
            oo = [o for o in ops['test_evaluated'] if (o['median'] < thr if lv == 0.026 else o['median'] <= thr)]
            op = min(oo, key=lambda o: o['ms']) if oo else None
            iso.append(dict(level=lv, ours=None if ours is None else dict(width=ours['width'], median=ours['primary']['median'],
                                                                         worst=ours['primary']['worst'], ms=ours['primary']['ms_median']),
                            ours_fastest_passing_width=None if fastest_ours is None else fastest_ours['width'],
                            operator=op, operator_note=None if op else ops.get('best_validation_only')))
        S['iso_error'] = iso
        lines += ['## Iso-error table (median relative error on test64)', '',
                  '| level (median ≤) | ours: narrowest passing width | our median % | our worst % | our time ms | fastest operator meeting the level | operator median % | operator worst % | operator time ms |',
                  '|---|---|---|---|---|---|---|---|---|']
        for x in iso:
            o, p = x['ours'], x['operator']
            left = f"{o['width']} | {o['median']:.4f} | {o['worst']:.4f} | {o['ms']:.2f}" if o else 'none in sweep | — | — | —'
            right = (f"{p['name']} [{p['cite']}] | {p['median']:.3f} | {p['worst']:.3f} | {p['ms']:.2f}" if p else
                     f"not reached yet ({x['operator_note']}) | — | — | —")
            lines.append(f"| {x['level']} % | {left} | {right} |")
        lines.append('')
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'reports').mkdir(exist_ok=True)
    (HERE / 'results/summary.json').write_text(json.dumps(S, indent=1) + '\n')
    (HERE / 'reports/tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
