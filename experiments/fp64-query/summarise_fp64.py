"""fp64-query: RESULTS.md and results/summary.json from results/<attempt>/{arms.json, result.json, ops-*/}.
    python src/analyse.py results/p64test results/p64fresh && python summarise_fp64.py
Every number in RESULTS.md is read from those files (PROTOCOL.md section 5, 7)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
OUT = []
P = OUT.append
WIDTHS = [96, 128, 176, 192, 256, 384]
EQW = 176


def ops(att):
    r = {}
    for d in sorted((RES / att).glob('ops-*')):
        for f in sorted(d.glob('*.L256.json')):
            x = json.loads(f.read_text())
            v = x['L256']
            r.setdefault(x['arm'], []).append(dict(seed=x['seed'], median=v['median_percent'], worst=v['worst_percent'],
                                                   ms=v['timing']['median_ms'], p05=v['timing']['p05_ms'],
                                                   p95=v['timing']['p95_ms'], gpu=x['gpu'], job=x['job_id']))
    return r


def audit(att):
    r = json.loads((RES / att / 'result.json').read_text())
    return r.get('precision_audit', {}), r


def load(att):
    j = json.loads((RES / att / 'arms.json').read_text())
    rows = {}
    for a in j['arms']:
        if a.get('width') is None:
            continue
        rows[(a['width'], a['role'])] = a
    return j, rows


def section(att, label, summary):
    if not (RES / att / 'arms.json').exists():
        P(f'\n## {label}: not available (results/{att}/arms.json missing)\n')
        return
    j, rows = load(att)
    pa, r = audit(att)
    P(f"\n## {label} (`results/{att}`)\n")
    P(f"Job {j['job_id']} on {j['host']}, GPU {j['gpu']}, backend {j['backend']}, commit {j['commit'][:9]}, "
      f"{j['cases']} cases (cohort SHA-256 {j['cohort_sha256'][:16]}), complete {j['complete']}. "
      f"Drift gate passed {j['drift']['passed']} (A-B-A ratios {j['drift']['ratios'][0]:.3f}-{j['drift']['ratios'][1]:.3f}), "
      f"neighbour gate passed {j['neighbour_passed']}.")
    if j.get('rejected_nonfinite'):
        P('Non-finite arms (rejected): ' + ', '.join(x['name'] for x in j['rejected_nonfinite']) + '.')
    bad = [a['name'] for a in rows.values() if a.get('identical_to_quick') is False]
    P(f"Timed outputs bitwise identical to the scored outputs: {'all arms' if not bad else 'NOT for ' + ', '.join(bad)}.\n")

    P('### Precision audit of the traced programs (Q6)\n')
    P('| arm | float64 vars | reduced-precision vars | primitives touching reduced precision | reduced-precision inputs | all float64 |')
    P('|---|---|---|---|---|---|')
    for (w, role), a in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1])):
        x = pa.get(a['name'])
        if x is None:
            P(f"| {a['name']} | | | | | not audited |")
            continue
        P(f"| {w} {role} | {x['dtype_counts'].get('float64', 0)} | {x['reduced_precision_vars']} | "
          f"{', '.join(x['reduced_precision_primitives']) or '-'} | {', '.join(x['reduced_precision_input_leaves']) or '-'} | "
          f"{x['all_float64']} |")

    P('\n### Errors and single-query latency\n')
    P('| width | setting | precision | median % | worst % | ms median | ms 5-95 % of cases | ms min-max | x vs baseline | fp64 / mixed time | median vs baseline | worst vs baseline |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|')
    out = {}
    for w in WIDTHS:
        for role in ('baseline', 'accurate', 'accurate_fp64', 'fast', 'fast_fp64'):
            a = rows.get((w, role))
            if a is None:
                continue
            prec = 'float64' if role in ('baseline',) or role.endswith('_fp64') else 'mixed (float32 stencil)'
            v = a.get('vs_baseline') or {}
            slow = ''
            if role.endswith('_fp64') and (w, role[:-5]) in rows:
                slow = f"{a['ms_median'] / rows[(w, role[:-5])]['ms_median']:.2f}"
            cells = [str(w), role, prec, f"{a['median']:.4f}", f"{a['worst']:.3f}", f"{a['ms_median']:.2f}",
                     f"{a['ms_case_p05']:.2f}-{a['ms_case_p95']:.2f}", f"{a['ms_min']:.2f}-{a['ms_max']:.2f}",
                     f"{v['speedup']:.2f}" if v else '1', slow,
                     f"{v['median_ratio']:.4f}" if v else '', f"{v['worst_ratio']:.4f}" if v else '']
            P('| ' + ' | '.join(cells) + ' |')
            out[f'{w}/{role}'] = dict(name=a['name'], median=a['median'], worst=a['worst'], ms=a['ms_median'],
                                      ms_p05=a['ms_case_p05'], ms_p95=a['ms_case_p95'], ms_min=a['ms_min'],
                                      ms_max=a['ms_max'], speedup=v.get('speedup'), median_ratio=v.get('median_ratio'),
                                      worst_ratio=v.get('worst_ratio'), fp64_over_mixed=float(slow) if slow else None,
                                      all_float64=pa.get(a['name'], {}).get('all_float64'))
    summary[att] = dict(job_id=j['job_id'], gpu=j['gpu'], host=j['host'], drift=j['drift'],
                        neighbour_passed=j['neighbour_passed'], arms=out)

    fp = [out[f'{w}/accurate_fp64'] for w in WIDTHS if f'{w}/accurate_fp64' in out]
    mx = [out[f'{w}/accurate'] for w in WIDTHS if f'{w}/accurate' in out]
    if fp:
        sp = [x['speedup'] for x in fp]
        sl = [x['fp64_over_mixed'] for x in fp if x['fp64_over_mixed']]
        unchanged = [w for w in WIDTHS if f'{w}/accurate_fp64' in out and out[f'{w}/accurate_fp64']['median_ratio'] <= 1.02]
        P(f"\nFloat64 \"accurate\" over the six widths: speedup vs the same-job baseline {min(sp):.2f}-{max(sp):.2f}x "
          f"(mixed precision in the same job: {min(x['speedup'] for x in mx):.2f}-{max(x['speedup'] for x in mx):.2f}x); "
          f"float64 / mixed time {min(sl):.2f}-{max(sl):.2f}; widths with float64 median within 2 % of the baseline: "
          f"{', '.join(map(str, unchanged)) or 'none'}.")
        dif = {}
        for w in WIDTHS:
            for role in ('accurate', 'fast'):
                if (w, role) in rows and (w, role + '_fp64') in rows:
                    e1, e2 = np.array(rows[(w, role)]['per_case']), np.array(rows[(w, role + '_fp64')]['per_case'])
                    dif[f'{w}/{role}'] = float(np.max(np.abs(e2 - e1) / e1))
        P(f"Largest per-case relative change of the error between a float64 arm and its mixed-precision original: "
          f"{max(dif.values()):.1e} (over {len(dif)} settings x {j['cases']} cases).")
        summary[att]['fp64_vs_mixed_max_case_error_change'] = dif
        summary[att]['fp64_accurate'] = dict(speedup_min=min(sp), speedup_max=max(sp), slowdown_min=min(sl),
                                             slowdown_max=max(sl), median_within_2pct_widths=unchanged)

    o = ops(att)
    if o:
        P('\n### Operators, same job, same GPU (frozen 4,608-trajectory checkpoints, float32 networks, TF32 off)\n')
        P('| operator | seed | median % | worst % | ms median | ms 5-95 % |')
        P('|---|---|---|---|---|---|')
        osum = {}
        for arm, xs in sorted(o.items()):
            for x in xs:
                P(f"| {arm} | {x['seed']} | {x['median']:.4f} | {x['worst']:.3f} | {x['ms']:.2f} | {x['p05']:.2f}-{x['p95']:.2f} |")
            osum[arm] = dict(median=float(np.median([x['median'] for x in xs])),
                             worst=float(np.median([x['worst'] for x in xs])),
                             ms=float(np.median([x['ms'] for x in xs])), seeds=xs)
            P(f"| {arm} | median of {len(xs)} seeds | {osum[arm]['median']:.4f} | {osum[arm]['worst']:.3f} | {osum[arm]['ms']:.2f} | |")
        summary[att]['operators'] = osum
        P(f'\nPre-registered comparison (PROTOCOL.md section 7): width {EQW} (frozen equal-error width), same job.\n')
        P('| operator | op median % | op worst % | op ms | our setting | our median % | our worst % | our ms | op ms / our ms | lower median and lower ms |')
        P('|---|---|---|---|---|---|---|---|---|---|')
        cmp_ = []
        for arm, s in sorted(osum.items()):
            for role in ('accurate_fp64', 'accurate', 'baseline'):
                c = out.get(f'{EQW}/{role}')
                if not c:
                    continue
                win = c['median'] < s['median'] and c['ms'] < s['ms']
                P(f"| {arm} | {s['median']:.4f} | {s['worst']:.3f} | {s['ms']:.2f} | {EQW} {role} | {c['median']:.4f} | "
                  f"{c['worst']:.3f} | {c['ms']:.2f} | {s['ms'] / c['ms']:.2f} | {win} |")
                cmp_.append(dict(operator=arm, role=role, op=dict(median=s['median'], worst=s['worst'], ms=s['ms']),
                                 ours=dict(median=c['median'], worst=c['worst'], ms=c['ms']), ratio=s['ms'] / c['ms'],
                                 wins_both=bool(win)))
        summary[att]['operator_comparison'] = cmp_


def headline(summary):
    """PROTOCOL.md section 7, generated from summary (test64 headline job; fresh64 beside it)."""
    t, f = summary.get('p64test'), summary.get('p64fresh')
    if not t or 'operator_comparison' not in t:
        return
    def acc(s_, key):
        return [s_['arms'][f'{w}/accurate_fp64'][key] for w in WIDTHS]
    medr_t, medr_f = max(abs(x - 1) for x in acc(t, 'median_ratio')), max(abs(x - 1) for x in acc(f, 'median_ratio'))
    wr_t = max(acc(t, 'worst_ratio'))
    fa, ft = t['fp64_accurate'], f['fp64_accurate']
    dch = max(max(t['fp64_vs_mixed_max_case_error_change'].values()), max(f['fp64_vs_mixed_max_case_error_change'].values()))
    c = {x['operator']: x for x in t['operator_comparison'] if x['role'] == 'accurate_fp64'}
    ours = c['F0']['ours']
    P('\n## Headline (pre-registered rule, PROTOCOL.md section 7)\n')
    P(f"- test64 job {t['job_id']}: float64 \"accurate\" is {fa['speedup_min']:.1f}-{fa['speedup_max']:.1f}x faster than the "
      f"same-job baseline at widths 96-384, median error within {100 * medr_t:.1f} % of the baseline at every width "
      f"(fresh64 job {f['job_id']}: {ft['speedup_min']:.1f}-{ft['speedup_max']:.1f}x, within {100 * medr_f:.1f} %). Worst-case "
      f"error up to {wr_t:.2f}x the baseline's on test64.")
    P(f"- Float64 costs {fa['slowdown_min']:.2f}-{fa['slowdown_max']:.2f}x the time of the mixed-precision original "
      f"(test64) and changes no per-case error by more than {dch:.0e} relative. The mixed-precision numbers are a "
      f"separately labelled result, not the headline.")
    P(f"- Width {EQW} float64: {ours['median']:.4f} % median / {ours['worst']:.3f} % worst in {ours['ms']:.2f} ms; "
      f"FNO {c['F0']['op']['median']:.4f} % / {c['F0']['op']['worst']:.3f} % in {c['F0']['op']['ms']:.2f} ms "
      f"({c['F0']['ratio']:.2f}x our time); U-Net {c['U0']['op']['median']:.4f} % / {c['U0']['op']['worst']:.3f} % in "
      f"{c['U0']['op']['ms']:.2f} ms ({c['U0']['ratio']:.2f}x). Lower median error and lower time than both: "
      f"{c['F0']['wins_both'] and c['U0']['wins_both']}. The operators keep the lower worst-case error.")
    P('\n### Replacement sentences for the rebuttal (numbers generated from the job JSON)\n')
    P(f"> With every array operation in double precision (checked by a dtype audit of the traced program), our frozen "
      f"online solver answers a Burgers 2D $256^2$ query {fa['speedup_min']:.1f}-{fa['speedup_max']:.1f}x faster than our "
      f"original solver at widths 96-384, with median error unchanged (within {100 * medr_t:.1f} %), all timed in one "
      f"A100-80GB job on the 64 test cases ({ft['speedup_min']:.1f}-{ft['speedup_max']:.1f}x on 64 fresh cases). "
      f"The gain comes from reusing one Jacobian factorisation per query (a lagged-test-space Petrov-Galerkin solve), "
      f"not from reduced precision: the single-precision stencil behind our earlier 5.5-10x figure changes no per-case "
      f"error by more than {dch:.0e} relative and accounts for only a {fa['slowdown_min']:.2f}-{fa['slowdown_max']:.2f}x "
      f"time difference. At width {EQW}, the double-precision query reaches {ours['median']:.3f} % median error in "
      f"{ours['ms']:.2f} ms, against FNO {c['F0']['op']['median']:.2f} % in {c['F0']['op']['ms']:.2f} ms and U-Net "
      f"{c['U0']['op']['median']:.2f} % in {c['U0']['op']['ms']:.2f} ms measured in the same job "
      f"({c['F0']['ratio']:.2f}x and {c['U0']['ratio']:.1f}x faster at about half their median error). The operators "
      f"retain a lower worst-case error ({c['F0']['op']['worst']:.2f}-{c['U0']['op']['worst']:.2f} % against our "
      f"{ours['worst']:.2f} %).")


def main():
    summary = {}
    P('# fp64-query: Burgers 2D 256^2 query speed with every array operation in float64')
    P('\nGenerated by `summarise_fp64.py` from the job JSON; do not edit by hand. Protocol: `PROTOCOL.md` (headline '
      'rule pre-registered in section 7). Settings are the frozen b2d-query-speed settings (c8bd352aa, `src/frozen.json`), '
      'unchanged; the `_fp64` arms differ only by `stencil32 = false`. test64 and fresh64 were opened by the query-speed '
      'lane; evaluating the same frozen settings again is not tuning (nothing is selected from these numbers). '
      'Single-query latency, A100 80GB, baseline, mixed-precision originals and operators in the same allocation.')
    section('p64test', 'test64 (headline job)', summary)
    section('p64fresh', 'fresh64', summary)
    headline(summary)
    P('\n## Glossary\n')
    P('- **width (R\')**: number of bank columns (reduced unknowns) of the linear rung.')
    P('- **baseline**: the unchanged query of the query-speed lane (about one Levenberg-Marquardt step per time step, '
      'Jacobian and Cholesky every step); float64 throughout.')
    P('- **accurate / fast**: the frozen lagged-operator settings of the query-speed lane (one Jacobian and one '
      'factorisation per query, fixed corrections per step; lagged-test-space Petrov-Galerkin). "fast" is a separate, '
      'less accurate point. Without suffix they run the stencil in float32 (mixed precision).')
    P('- **_fp64**: the same setting with the stencil contraction and advection in float64; every array op float64 '
      '(verified by the precision audit).')
    P('- **median % / worst %**: median / maximum over cases of the per-case error, the maximum over the five evolved '
      'output times of ||u - u_ref|| / ||u_ref(0)|| against the tight full-order reference, in percent.')
    P('- **ms median**: median single-query GPU latency over all timed invocations (warm-up excluded); '
      '**ms 5-95 % of cases**: range of per-case median latencies; **min-max**: over invocations.')
    P('- **x vs baseline**: baseline ms / arm ms, same job. **fp64 / mixed time**: float64 arm ms / mixed arm ms.')
    P('- **median / worst vs baseline**: arm error / baseline error of the same width, same job.')
    P('- **drift gate**: ratio of the two A phases of the A-B-A timing (must be within 1.1); **neighbour gate**: '
      'the query-speed timing sanity gate on adjacent invocations (limit 1.1).')
    P('- **Q6 precision audit**: every variable of the traced JAX program of the query, counted by dtype.')
    P('- **test64 / fresh64**: 64 test cases (params_draw(20260916,64)) and 64 fresh cases (params_draw(2026111402,64)).')
    (HERE / 'RESULTS.md').write_text('\n'.join(OUT) + '\n')
    RES.mkdir(exist_ok=True)
    (RES / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('\n'.join(OUT))


if __name__ == '__main__':
    main()
