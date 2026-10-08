"""Sections of report.md for the fair full-order comparison (fom1k, DESIGN section 9, A9-A11) and the 3D job (b3d65,
A10.2, A13). Imported by make_report.py; every number is read from the pulled job results and their audits.

Eligibility: a section makes claims (matched speed-ups, hypotheses, order claims) only if its audit passed on the exact
result bytes (the audit's recorded result sha256, job id, commit and attempt must match), the cohort is the registered one, and every configuration used is
verified on every case with finite timings. Otherwise the numbers are printed as diagnostics and every claim reads
"unavailable".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
COLORS = dict(BE='#2a78d6', CN='#eb6834', CNR='#1baf7a', BDF2='#eda100', TH06='#e87ba4')
NAMES = dict(BE='BE', CN='CN', CNR='CN-R', BDF2='BDF2', TH06='TH06')
SCHEMES = ['BE', 'CN', 'CNR', 'BDF2']
FIN = lambda x: x is not None and np.isfinite(x)
pct = lambda x: '—' if not FIN(x) else f'{100 * x:.3f}'
num = lambda x, f='.1f': '—' if not FIN(x) else format(x, f)


def _timing(inv):
    """per-subject median paired ratio / IQR / median candidate ms; None for a subject with any non-finite record."""
    by = {}
    for i in inv:
        by.setdefault(i['B'], []).append(i)
    out = {}
    for b, L in by.items():
        if not all(FIN(i.get('ratio')) and FIN(i.get('tB')) and i['tB'] > 0 for i in L):
            out[b] = None
            continue
        r = np.array([i['ratio'] for i in L])
        out[b] = dict(n=len(r), ratio=float(np.median(r)), iqr=[float(np.quantile(r, .25)), float(np.quantile(r, .75))],
                      ms=1e3 * float(np.median([i['tB'] for i in L])))
    tA = [x for i in inv for x in (i.get('tA1'), i.get('tA2'))]
    A_ms = 1e3 * float(np.median(tA)) if tA and all(FIN(x) and x > 0 for x in tA) else None
    return out, A_ms


def _audit_ok(arc, att):
    res_path = arc / 'output/result.json'
    res = json.loads(res_path.read_text())
    ap = HERE / 'checks' / f'audit-{att}.json'
    if not ap.exists():
        return res, False, 'no audit'
    aud = json.loads(ap.read_text())
    ok = bool(aud['all_pass'] and aud['job_id'] == res['job_id'] and aud['commit'] == res['commit'] and aud.get('attempt') == att
              and aud.get('result_sha256') ==
              hashlib.sha256(res_path.read_bytes()).hexdigest())
    why = 'pass' if ok else 'FAIL: ' + ', '.join(k for k, v in aud['checks'].items() if not v)
    return res, ok, why


# ------------------------------------------------------------------------------------------------ FOM ----
def fom_section(W, att='fom1k'):
    arc = HERE / 'runs' / att / 'archive'
    if not (arc / 'output/result.json').exists():
        return None
    res, aud_ok, why = _audit_ok(arc, att)
    n_cases = len({(r['cohort'], r['case']) for r in res['rows']})
    eligible = aud_ok and n_cases == 38 and res.get('complete')
    tim, A_ms = _timing(res['timing'].get('invocations', []))
    A_ = res['timing'].get('A')
    if A_ms is not None:
        tim[A_] = dict(n=None, ratio=1., iqr=None, ms=A_ms)
    agg = {}
    for r in res['rows'] + res['rom_rows']:
        agg.setdefault(r['run'], []).append(r)
    T = {}
    for k, L in agg.items():
        st = np.array([x['e_ST'] for x in L], float)
        t_ = tim.get(k)
        cal_ok = True
        if k.startswith('fom'):
            cal = res['calibration'].get('|'.join(k.split('|')[1:]), {})
            cal_ok = cal.get('status') in ('calibrated', 'fallback_tight')
        T[k] = dict(n=len(L), verified=sum(x['verified'] for x in L), worst=float(np.max(st)), median=float(np.median(st)),
                    ms=t_['ms'] if t_ else None, ratio=t_['ratio'] if t_ else None,
                    ok=bool(len(L) == 38 and all(x['verified'] for x in L) and cal_ok and t_ is not None))
    W(f'## Fair full-order comparison (`{att}`, $L={res["mesh"]}$; DESIGN section 9, A9–A11)')
    W('')
    W(f"Job {res['job_id']} on {res['gpu']}; independent audit: {why}. "
      + ('' if eligible else '**Not eligible for claims (audit, completeness or cohort): every claim below reads "unavailable".** ') +
      'The full-order model uses the same two-step methods (sign-upwind advection, 5-point Laplacian, Newton–BiCGStab with the '
      'DST Helmholtz preconditioner), each at its calibrated Newton tolerance. Accuracy over the 38 dev6 ∪ val32 cases; timing '
      'over six dev6 cases × 3 repetitions on one GPU, ratios relative to the full-order BE at $\\Delta t_0$. All errors are '
      'PROVISIONAL (first-order references); the full-order model is first order (upwind) in space at this mesh.')
    W('')
    W('FOM order check ($L=256$, dev6 cases 0 and 2; orders from the pairs starting at $2\\Delta t_0$, $\\Delta t_0$, $\\Delta t_0/2$):')
    W('')
    W('| scheme | case | orders | status |')
    W('|---|---|---|---|')
    for o in res['order']:
        p = o['orders'].get('0.5')
        inb = FIN(p) and (.8 <= p <= 1.25 if o['scheme'] == 'BE' else 1.7 <= p <= 2.3)
        if o['scheme'] == 'CN':
            stat = 'in band (reported only, A11.1)' if inb else 'pre-asymptotic (consistent with a stiff-mode transient); not gated (A11.1)'
        else:
            stat = 'in band (gated)' if inb else 'OUT OF BAND (gated)'
        W(f"| {NAMES[o['scheme']]} | {o['cohort']}{o['case']} | {', '.join(f'{v:.2f}' for v in o['orders'].values())} | {stat} |")
    W('')
    W('Accuracy (PROVISIONAL, % of $\\lVert u_0\\rVert$) and cost; rows not verified on all 38 cases or without a complete timing are marked and excluded from the matching:')
    W('')
    W('| solver | scheme | $\\Delta t/\\Delta t_0$ | tolerance | ST worst | ST median | verified | ms | ratio to FOM BE $\\Delta t_0$ | eligible |')
    W('|---|---|---|---|---|---|---|---|---|---|')
    keys = sorted(T, key=lambda k: (not k.startswith('fom'), k))
    for k in keys:
        e = T[k]
        parts = k.split('|')
        if parts[0] == 'fom':
            cal = res['calibration'][f'{parts[1]}|{parts[2]}']
            tol = f"ntol {cal['chosen'][0]:g} ({cal['status']})" if cal.get('chosen') else 'unresolved'
            W(f"| FOM | {NAMES[parts[1]]} | {parts[2]} | {tol} | {pct(e['worst'])} | {pct(e['median'])} | {e['verified']}/{e['n']} | "
              f"{num(e['ms'])} | {num(e['ratio'], '.3f')} | {'yes' if e['ok'] else 'no'} |")
        else:
            W(f"| ROM {parts[1]} {parts[3]} | {NAMES[parts[4]]} | {parts[5]} | production | {pct(e['worst'])} | {pct(e['median'])} | "
              f"{e['verified']}/{e['n']} | {num(e['ms'])} | {num(e['ratio'], '.3f')} | {'yes' if e['ok'] else 'no'} |")
    W('')
    foms = [k for k in T if k.startswith('fom') and T[k]['ok']]
    W('Matched comparison (A9.4; "best among the listed configurations" only): for each eligible ROM arm, the cheapest '
      'eligible FOM configuration whose cohort-worst ST error is no larger; speed-up = FOM time / ROM time (same job, same GPU). '
      'PROVISIONAL (both errors are against first-order references).')
    W('')
    W('| ROM arm | ROM ST worst | ROM ms | matched FOM | FOM ST worst | FOM ms | speed-up |')
    W('|---|---|---|---|---|---|---|')
    match = []
    for k in [k for k in keys if k.startswith('rom') and T[k]['ok']]:
        c = [f for f in foms if T[f]['worst'] <= T[k]['worst']]
        if not eligible:
            W(f"| `{k}` | {pct(T[k]['worst'])} | {num(T[k]['ms'])} | unavailable | — | — | — |")
            continue
        if not c:
            W(f"| `{k}` | {pct(T[k]['worst'])} | {num(T[k]['ms'])} | none at this accuracy | — | — | — |")
            continue
        fbest = min(c, key=lambda f: T[f]['ms'])
        sp = T[fbest]['ms'] / T[k]['ms']
        match.append(dict(rom=k, fom=fbest, speedup=sp))
        W(f"| `{k}` | {pct(T[k]['worst'])} | {num(T[k]['ms'])} | `{fbest}` | {pct(T[fbest]['worst'])} | {num(T[fbest]['ms'])} | {sp:.1f}× |")
    W('')
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    for fam, mk in (('fom', 's'), ('rom', 'o')):
        for sc in SCHEMES:
            for setting in (['-'] if fam == 'fom' else ['fast', 'acc']):
                kk = [k for k in T if T[k]['ok'] and k.startswith(fam) and
                      ((fam == 'fom' and k.split('|')[1] == sc) or
                       (fam == 'rom' and k.split('|')[4] == sc and k.split('|')[3] == 'LSPG' and k.split('|')[1] == setting))]
                kk = sorted(kk, key=lambda k: float(k.split('|')[-1]))
                if not kk:
                    continue
                lab = f'FOM {NAMES[sc]}' if fam == 'fom' else f'ROM {setting} LSPG {NAMES[sc]}'
                ax.plot([T[k]['ms'] for k in kk], [100 * T[k]['worst'] for k in kk], marker=mk, color=COLORS[sc], lw=1.6,
                        ms=6, ls='-' if fam == 'rom' else '--', label=lab, alpha=.55 if setting == 'fast' else .95)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('end-to-end time per case (ms; paired A–B–A, six dev6 cases, one GPU)')
    ax.set_ylabel('worst ST error over 38 cases (%), PROVISIONAL')
    ax.set_title(f'Full-order vs reduced, same time schemes, $L={res["mesh"]}$ (eligible configurations only)', fontsize=10)
    ax.grid(True, which='major', color='#e4e4e0', lw=.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / 'plots' / 'fom_vs_rom.png', dpi=150)
    plt.close(fig)
    W('![fom_vs_rom.png](plots/fom_vs_rom.png)')
    W('')
    (HERE / 'checks' / f'analysis-{att}.json').write_text(json.dumps(dict(eligible=eligible, table=T, matched=match), indent=1) + '\n')
    return dict(T=T, matched=match, eligible=eligible)


# ------------------------------------------------------------------------------------------------- 3D ----
def d3_section(W, att='b3d65'):
    arc = HERE / 'runs' / att / 'archive'
    if not (arc / 'output/result.json').exists():
        return None
    res, aud_ok, why = _audit_ok(arc, att)
    n = res['config']['cohort_count']
    eligible = aud_ok and n == 16 and res['config']['cohort_seed'] == 923801 and res.get('complete')
    need, half = int(np.ceil(.8 * n)), n / 2
    DT0 = .01
    RL = 'reference-limited (BE reference at $\\Delta t_0/4$), PROVISIONAL'
    W(f'## Burgers 3D (`{att}`, $65^3$, validation cohort 923801, first {n} cases; A10.2, A13)')
    W('')
    W(f"Job {res['job_id']} on {res['gpu']}; audit: {why}. "
      + ('' if eligible else '**Not eligible for claims: every claim below reads "unavailable".** ') +
      'Reference: the 3D lane\'s 513-node backward-Euler solution at $\\Delta t=\\Delta t_0/4$ on the 65-node lattice. '
      '**Errors against it are ' + RL + ': by A13 they cannot rank first- against second-order schemes.** The time-stepping '
      'evidence in 3D is the anchor discrepancy (distance to GAL-BDF2 at $\\Delta t_0/16$, on the $63^3$ lattice), the observed '
      'orders and the cost. Comparator: the deployed fixed-sweep BE at $\\Delta t_0=0.01$; timing: first four cases × 3 '
      'repetitions, paired A–B–A against it.')
    W('')
    out = dict(eligible=eligible, arms={})
    for arm in sorted({r['arm'] for r in res['rows']}):
        rows = [r for r in res['rows'] if r['arm'] == arm]
        by = {(r['run'], r['case']): r for r in rows}
        vend = [v for v in res['vendor_rows'] if v['arm'] == arm]
        tim, A_ms = _timing(res['timing'].get(arm, {}).get('invocations', []))
        ve = np.array([v['e_ref'] for v in vend], float) if len(vend) == n else None
        va = np.array([v['anchor'] for v in vend], float) if len(vend) == n else None

        def unc(j):
            parts = [by.get((f'GAL|BDF2|{x:.8g}|{lv}', j)) for x in (DT0 / 8, DT0 / 16) for lv in ('tight', 'tighter')]
            if None in parts or not all(x['verified'] for x in parts):
                return None
            a8, a16 = parts[0], parts[2]
            comp = [a8.get('d_half'), a8.get('s_h'), a16.get('s_h')]
            if not all(FIN(c) for c in comp):
                return None
            return float(sum(comp))

        def triple(form, sc, h, j):
            hs = [h, h / 2, h / 4]
            rt = [by.get((f'{form}|{sc}|{x:.8g}|tight', j)) for x in hs]
            rr = [by.get((f'{form}|{sc}|{x:.8g}|tighter', j)) for x in hs]
            if any(x is None for x in rt + rr) or not all(x['verified'] for x in rt + rr):
                return None, False
            d1, d2 = rt[0].get('d_half'), rt[1].get('d_half')
            s = [x.get('s_h') for x in rt]
            if not (FIN(d1) and FIN(d2) and all(FIN(x) for x in s)) or d1 <= 0 or d2 <= 0:
                return None, False
            return float(np.log2(d1 / d2)), bool(d1 > 10 * (s[0] + s[1]) and d2 > 10 * (s[1] + s[2]))
        W(f'### `{arm}`')
        W('')
        if ve is not None:
            W(f"Deployed fixed-sweep BE at $\\Delta t_0$: error worst / median {pct(ve.max())} / {pct(float(np.median(ve)))} % "
              f"({RL}); anchor discrepancy median {pct(float(np.median(va)))} %; {num(A_ms)} ms median; distance from the generic "
              f"adaptive BE at $\\Delta t_0$: median {pct(float(np.median([v['vs_generic_BE'] for v in vend])))} %.")
        W('')
        W(f'| form | scheme | $\\Delta t/\\Delta t_0$ | error worst ({RL}) | error median | anchor disc. median, resolved cases only (%) | resolved / eligible | verified | ratio to deployed | ms |')
        W('|---|---|---|---|---|---|---|---|---|---|')
        tab = []
        for form in ('LSPG', 'GAL'):
            for sc in SCHEMES:
                for fdt in res['config']['grid']['dt_factors']:
                    k = f'{form}|{sc}|{DT0 * fdt:.8g}|prod'
                    L = [by[(k, j)] for j in range(n) if (k, j) in by]
                    if len(L) != n:
                        continue
                    e = np.array([x['e_ref'] for x in L], float)
                    an = []
                    for x in L:
                        u = unc(x['case'])
                        if x['verified'] and u is not None and FIN(x.get('anchor')):
                            an.append((x['anchor'], u))
                    resolved = [a_ for a_, u_ in an if a_ >= 3 * u_]
                    t_ = tim.get(k)
                    ok = all(x['verified'] for x in L) and t_ is not None
                    ent = dict(key=k, form=form, scheme=sc, f=fdt, worst=float(e.max()), median=float(np.median(e)),
                               anc=float(np.median(resolved)) if len(resolved) >= half else None, resolved=len(resolved),
                               n_anc=len(an), verified=sum(x['verified'] for x in L), n=n, ok=ok,
                               ratio=t_['ratio'] if t_ else None, ms=t_['ms'] if t_ else None)
                    tab.append(ent)
                    W(f"| {form} | {NAMES[sc]} | {fdt:g} | {pct(ent['worst'])} | {pct(ent['median'])} | "
                      f"{pct(ent['anc']) if ent['anc'] is not None else 'unresolved in most cases'} | {len(resolved)}/{len(an)} | "
                      f"{ent['verified']}/{n} | {num(ent['ratio'], '.3f')} | {num(ent['ms'])} |")
        W('')
        W(f'Observed order ({n} cases: a claim needs ≥ {need} valid primary triples; the adjacent check is unresolved below {half:g} cases with both triples valid):')
        W('')
        W('| form | scheme | claim | primary median $p$ | valid | adjacent median $p$ | both valid |')
        W('|---|---|---|---|---|---|---|')
        orders = {}
        for form in ('LSPG', 'GAL'):
            for sc in SCHEMES + ['TH06']:
                prim = [triple(form, sc, DT0 / 2, j) for j in range(n)]
                adj = [triple(form, sc, DT0, j) for j in range(n)]
                pv = [p for p, v in prim if v]
                both = [(p, q) for (p, v), (q, w) in zip(prim, adj) if v and w]
                claim = 'not established'
                for lab, (lo, hi) in (('order 2', (1.7, 2.3)), ('order 1', (.8, 1.25))):
                    if len(pv) >= need and np.mean([lo <= p <= hi for p in pv]) >= .8:
                        if len(both) < half:
                            claim = f'{lab} (adjacent check unresolved)'
                        elif np.mean([lo <= p <= hi and lo <= q <= hi for p, q in both]) >= .8:
                            claim = lab
                        break
                if not eligible:
                    claim = f'unavailable (would read: {claim})'
                av = [p for p, v in adj if v]
                o = dict(claim=claim, primary=float(np.median(pv)) if pv else None, valid=len(pv),
                         adjacent=float(np.median(av)) if av else None, both=len(both))
                orders[f'{form}|{sc}'] = o
                W(f"| {form} | {NAMES[sc]} | {claim} | {num(o['primary'], '.2f')} | {len(pv)}/{n} | {num(o['adjacent'], '.2f')} | {len(both)} |")
        W('')
        h1 = h2 = []
        if eligible and ve is not None:
            cw, cm = float(ve.max()), float(np.median(ve))
            so2 = [e for e in tab if e['scheme'] != 'BE' and e['ok']]
            h1 = [e['key'] for e in so2 if abs(e['f'] - 1) < 1e-12 and e['median'] <= .7 * cm and e['worst'] <= cw]
            h2 = [e['key'] for e in so2 if e['f'] >= 2 and e['worst'] <= cw and e['median'] <= cm and FIN(e['ratio']) and e['ratio'] <= .75]
            v1, v2 = ('passed' if h1 else 'failed'), ('passed' if h2 else 'failed')
        else:
            v1 = v2 = 'unavailable'
        W(f"- H1-3D (**reference-dependent, PROVISIONAL**; CN/BDF2-family at $\\Delta t_0$, median ≤ 0.7× and worst ≤ the deployed "
          f"fixed-sweep BE against the {RL.split(',')[0]}): **{v1}** ({', '.join(h1) or 'none'}).")
        W(f"- H2-3D (**reference-dependent, PROVISIONAL**; CN/BDF2-family at $\\Delta t\\ge2\\Delta t_0$, worst and median ≤ deployed, "
          f"paired ratio ≤ 0.75): **{v2}** ({', '.join(h2) or 'none'}).")
        W('')
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        for form, ls, mk in (('LSPG', '-', 'o'), ('GAL', '--', 's')):
            for sc in SCHEMES:
                E = sorted([e for e in tab if e['form'] == form and e['scheme'] == sc and e['ok']], key=lambda e: e['f'])
                if not E:
                    continue
                ax[0].plot([DT0 * e['f'] for e in E], [100 * e['worst'] for e in E], ls=ls, marker=mk, color=COLORS[sc], label=f'{form} {NAMES[sc]}')
                ax[1].plot([DT0 * e['f'] for e in E], [100 * e['anc'] if e['anc'] is not None else np.nan for e in E], ls=ls,
                           marker=mk, color=COLORS[sc], label=f"{form} {NAMES[sc]} ({orders[f'{form}|{sc}']['claim']})")
        if ve is not None:
            ax[0].axhline(100 * float(ve.max()), color='#52514e', ls=':', label='deployed fixed-sweep BE, $\\Delta t_0$')
        for a_ in ax:
            a_.set_xscale('log')
            a_.set_xlabel('time step $\\Delta t$')
            a_.grid(True, color='#e4e4e0', lw=.6)
            a_.spines[['top', 'right']].set_visible(False)
        ax[1].set_yscale('log')
        ax[0].set_ylabel('worst error vs BE reference at $\\Delta t_0/4$ (%)\nREFERENCE-LIMITED, PROVISIONAL')
        ax[1].set_ylabel('median anchor discrepancy, resolved cases (%)')
        ax[0].set_title(f'3D {arm}: error vs the BE ($\\Delta t_0/4$) reference', fontsize=10)
        ax[1].set_title(f'3D {arm}: anchor discrepancy (GAL-BDF2, $\\Delta t_0/16$)', fontsize=10)
        ax[0].legend(fontsize=6.5, frameon=False)
        ax[1].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(HERE / 'plots' / f'error_vs_dt_3d_{arm}.png', dpi=150)
        plt.close(fig)
        W(f'![error_vs_dt_3d_{arm}.png](plots/error_vs_dt_3d_{arm}.png)')
        W('')
        out['arms'][arm] = dict(table=tab, orders=orders, H1=h1, H2=h2,
                                comparator=dict(worst=float(ve.max()) if ve is not None else None, ms=A_ms))
    W('3D definitions: anchor discrepancies are on the $63^3$ lattice; a value is shown only when at least half the eligible '
      'cases are resolved (≥ 3× the per-case anchor uncertainty indicator), and the median is over the resolved cases; '
      'timing is over four cases × 3 repetitions.')
    W('')
    (HERE / 'checks' / f'analysis-{att}.json').write_text(json.dumps(out, indent=1) + '\n')
    return out
