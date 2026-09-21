"""Pre-registered verdicts, computed from summary.json. Shared by the report and the lab entry.

Lane success (DESIGN): a P1-promoted bank whose full-bank solved worst error is <= 0.5 % (stretch
<= 0.2 %) on BOTH the primary cohort and the confirmation cohort, at online cost <= 4x the
incumbent's same-rung (full-bank) cost. The protocol bar (<= 1 % / 0.5 % error AND >= 5x faster
than the named FOM in the same allocation) is evaluated beside it.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEARNED = ('ft512', 'cat1024', 'cat2048')
ALL = ('ft512', 'pod512', 'cat1024', 'pod1024', 'cat2048', 'pod2048')


def pct(x):
    return f'{100 * x:.4f} %'


def rows(S):
    out = []
    for pde, rk, sk, suf, prim, conf in (
            ('Poisson', 'rep_poisson', 'solve_poisson', '_weak', 'dev12', 'confirm256'),
            ('Burgers', 'rep_burgers', 'solve_burgers', '_full', 'dev6', 'confirm')):
        rep, sol = S.get(rk), S.get(sk)
        if not (rep and sol):
            continue
        inc_ms = sol['inc512' + suf]['median_total_ms']
        err = (lambda s, c: s['worst'][c]) if pde == 'Poisson' else (lambda s, c: s['worst_all'][c])
        foms = {k: dict(ms=v['median_total_ms'], worst=max(err(v, prim), err(v, conf)), kind=v['kind'])
                for k, v in sol.items() if v['kind'].startswith('fom')}
        for t in ALL:
            s = sol[t + suf]
            e_p, e_c = err(s, prim), err(s, conf)
            worst = max(e_p, e_c)
            cost = s['median_total_ms'] / inc_ms
            # comparators at matched-or-better accuracy on the same two cohorts (protocol): the cheapest
            # NAMED full-order setting (CG / Newton) and the cheapest full-order solver of ANY kind
            ok_named = {k: v for k, v in foms.items() if v['kind'] in ('fom_named', 'fom') and v['worst'] <= worst}
            ok_any = {k: v for k, v in foms.items() if v['worst'] <= worst}
            named = min(ok_named, key=lambda k: ok_named[k]['ms'])
            fastest = min(ok_any, key=lambda k: ok_any[k]['ms'])
            out.append(dict(
                pde=pde, bank=t, learned=t in LEARNED, R=s['R'], promoted=rep[t]['p1_promoted'],
                floor_improvement=rep[t]['improvement_vs_inc512'], solved_primary=e_p, solved_confirm=e_c,
                ms=s['median_total_ms'], cost_vs_incumbent=cost,
                sub_05=worst <= 0.005, sub_02=worst <= 0.002,
                lane_success=bool(rep[t]['p1_promoted'] and worst <= 0.005 and cost <= 4),
                lane_stretch=bool(rep[t]['p1_promoted'] and worst <= 0.002 and cost <= 4),
                speedup_vs_named=foms[named]['ms'] / s['median_total_ms'], named=named,
                named_error=foms[named]['worst'],
                speedup_vs_fastest=foms[fastest]['ms'] / s['median_total_ms'], fastest=fastest))
    return out


def markdown(S):
    R = rows(S)
    md = ['## Verdicts (computed by `reports/verdicts.py`)', '',
          'Worst solved error is the larger of the primary and the confirmation cohort. "cost × inc" is the full-bank '
          'query time relative to the incumbent bank\'s full-bank query in the same job. Pre-registered lane success: '
          'P1-promoted, worst solved ≤ 0.5 % (stretch ≤ 0.2 %), cost ≤ 4× inc. Every FOM comparator is chosen per row as the '
          'cheapest full-order setting whose worst error on the same two cohorts is no larger than the ROM\'s. Speedup < 1 means '
          'the ROM is slower.', '',
          '| PDE | bank | learned (mesh-free) | R | P1 | floor gain vs inc512 | solved primary | solved confirm | cost × inc | '
          'sub-0.5 % | sub-0.2 % | lane success | stretch | speedup vs cheapest named FOM at ≤ ROM error | that FOM\'s worst error | speedup vs cheapest FOM of any kind at ≤ ROM error |',
          '|---|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|']
    for r in R:
        md.append(f"| {r['pde']} | `{r['bank']}` | {'yes' if r['learned'] else 'no (grid-bound POD)'} | {r['R']} | "
                  f"{'promoted' if r['promoted'] else '—'} | {r['floor_improvement']:.2f}× | {pct(r['solved_primary'])} | "
                  f"{pct(r['solved_confirm'])} | {r['cost_vs_incumbent']:.2f}× | {'yes' if r['sub_05'] else 'no'} | "
                  f"{'yes' if r['sub_02'] else 'no'} | {'**PASS**' if r['lane_success'] else 'fail'} | "
                  f"{'**PASS**' if r['lane_stretch'] else 'fail'} | {r['speedup_vs_named']:.3g}× vs `{r['named']}` | "
                  f"{pct(r['named_error'])} | {r['speedup_vs_fastest']:.3g}× vs `{r['fastest']}` |")
    md.append('')
    for pde in ('Poisson', 'Burgers'):
        P = [r for r in R if r['pde'] == pde]
        if not P:
            continue
        ok = [r for r in P if r['lane_success']]
        okl = [r for r in ok if r['learned']]
        best_l = min((r for r in P if r['learned'] and r['sub_05']), key=lambda r: r['ms'], default=None)
        best_02 = min((r for r in P if r['sub_02']), key=lambda r: r['ms'], default=None)
        names = ', '.join('`' + r['bank'] + '`' for r in ok) or 'none'
        md.append(f"- **{pde}.** Lane success: {names}"
                  f" ({len(okl)} of them learned/mesh-free). "
                  + (f"Cheapest learned bank under 0.5 % on both cohorts: `{best_l['bank']}` at {best_l['ms']:.3f} ms "
                     f"({best_l['cost_vs_incumbent']:.2f}× the incumbent's full-bank query; speedup "
                     f"{best_l['speedup_vs_named']:.3g}× vs `{best_l['named']}`, {best_l['speedup_vs_fastest']:.3g}× vs `{best_l['fastest']}`). " if best_l else
                     'No learned bank is under 0.5 % on both cohorts. ')
                  + (f"Cheapest bank under 0.2 % on both cohorts: `{best_02['bank']}` at {best_02['ms']:.3f} ms "
                     f"(speedup {best_02['speedup_vs_named']:.3g}× vs `{best_02['named']}`, {best_02['speedup_vs_fastest']:.3g}× vs "
                     f"`{best_02['fastest']}`)." if best_02 else 'No bank is under 0.2 % on both cohorts.'))
    md.append('')
    return '\n'.join(md), R


if __name__ == '__main__':
    print(markdown(json.loads((HERE / 'summary.json').read_text()))[0])
