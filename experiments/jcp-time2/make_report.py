"""Generate report.md and plots/*.png of the jcp-time2 lane from the pulled job results (no hand-typed numbers).

    /home/tahmid/Dev/.venv/bin/python experiments/jcp-time2/make_report.py [attempt ...]   (default: a1kfast a1kacc)

Order claims (A2.12, A3.3, A4.1), anchor resolution (A2.8), timing summaries (A2.15), H1/H2 and selection
(A1.8, A2.18) are computed here from runs/<attempt>/archive/output/result.json; audit verdicts are read from
checks/audit-<attempt>.json; the manufactured tests from checks/test_lmm.json; the motivating evidence from
checks/evidence-be-gap.json. Writes checks/analysis-<attempt>.json as well.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DT0 = .005
SCHEMES = ['BE', 'CN', 'CNR', 'BDF2']
NAMES = dict(BE='BE', CN='CN', CNR='CN-R', BDF2='BDF2', TH06='TH06')
COLORS = dict(BE='#2a78d6', CN='#eb6834', CNR='#1baf7a', BDF2='#eda100', TH06='#e87ba4')
FORMS = ['LSPG', 'GAL']
STYLE = dict(LSPG=dict(ls='-', marker='o'), GAL=dict(ls='--', marker='s'))
RULE_NAME = dict(gauss96='Gauss $96^2$', fib1597='Fibonacci 1597', gauss192='Gauss $192^2$', fib6765='Fibonacci 6765',
                 lat64='mesh lattice $63^2$')


def k(role, form, sc, dt, lev):
    return f'{role}|{form}|{sc}|{dt:.8g}|{lev}'


def pct(x):
    return '—' if x is None or not np.isfinite(x) else f'{100 * x:.3f}'


def sci(x):
    return '—' if x is None or not np.isfinite(x) else f'{x:.1e}'


def dtf(dt):
    f = dt / DT0
    return f'{f:g}' if f >= 1 else f'1/{1 / f:g}'


class Setting:
    def __init__(self, res, s):
        self.s, self.res = s, res
        self.rows = [r for r in res['rows'] if r['setting'] == s]
        self.cases = sorted({(r['cohort'], r['case']) for r in self.rows})
        self.by = {(r['cohort'], r['case'], r['run']): r for r in self.rows}
        self.runs = res['setup'][s]['runs']

    def get(self, case, key):
        return self.by.get((case[0], case[1], key))

    def agg(self, key, metric='e_ST'):
        v = [self.get(c, key) for c in self.cases]
        v = [r for r in v if r is not None]
        if not v:
            return None
        x = np.array([r.get(metric, np.nan) if r.get(metric) is not None else np.nan for r in v], float)
        ver = sum(r['verified'] for r in v)
        alt = sum((r['stats']['alt_index'] or 0) > .5 for r in v)
        return dict(n=len(v), worst=float(np.nanmax(x)), median=float(np.nanmedian(x)), verified=ver, alternating=alt,
                    it_median=float(np.median([r['stats']['it_sum'] for r in v])))

    # ---- A2.11 / A2.12 / A3.3 / A4.1 ----
    def triple(self, case, form, sc, h):
        """order at triple (h, h/2, h/4) and its validity."""
        hs = [h, h / 2, h / 4]
        rt = [self.get(case, k('main', form, sc, x, 'tight')) for x in hs]
        rr = [self.get(case, k('main', form, sc, x, 'tighter')) for x in hs]
        if any(r is None for r in rt + rr):
            return None, False
        if not all(r['verified'] for r in rt + rr):
            return None, False
        d1, d2 = rt[0].get('d_half'), rt[1].get('d_half')
        s = [r.get('s_h') for r in rt]
        if d1 is None or d2 is None or any(x is None for x in s):
            return None, False
        valid = d1 > 10 * (s[0] + s[1]) and d2 > 10 * (s[1] + s[2]) and d1 > 0 and d2 > 0
        return (float(np.log2(d1 / d2)) if d1 > 0 and d2 > 0 else None), bool(valid)

    def order_claim(self, form, sc):
        prim = [self.triple(c, form, sc, DT0 / 2) for c in self.cases]
        adj = [self.triple(c, form, sc, DT0) for c in self.cases]
        n = len(self.cases)
        pv = [p for p, v in prim if v]
        out = dict(cases=n, primary_valid=len(pv), primary_median=float(np.median(pv)) if pv else None,
                   adjacent_median=float(np.median([p for p, v in adj if v])) if any(v for _, v in adj) else None,
                   primary_orders=[p for p, _ in prim])
        both = [(p, q) for (p, v), (q, w) in zip(prim, adj) if v and w]
        out['n_both'] = len(both)
        claim = 'not established'
        for label, (lo, hi) in (('order 2', (1.7, 2.3)), ('order 1', (.8, 1.25))):
            if len(pv) >= .8 * n and pv and np.mean([lo <= p <= hi for p in pv]) >= .8:
                if len(both) < 19 and n >= 19 or (n < 19 and len(both) < n / 2):
                    claim = f'{label} (adjacent check unresolved)'
                elif np.mean([lo <= p <= hi and lo <= q <= hi for p, q in both]) >= .8:
                    claim = label
                else:
                    claim = 'not established'
                break
        out['claim'] = claim
        out['outliers_2'] = int(sum(not (1.7 <= p <= 2.3) for p in pv))
        out['outliers_1'] = int(sum(not (.8 <= p <= 1.25) for p in pv))
        co = [self.triple(c, form, sc, 2 * DT0) for c in self.cases]
        cv = [p for p, v in co if v]
        out['coarse_median'], out['coarse_valid'] = (float(np.median(cv)) if cv else None), len(cv)
        if form == 'GAL' and sc in ('BDF2', 'CN'):
            fin = [self.triple(c, form, sc, DT0 / 4) for c in self.cases]
            fv = [p for p, v in fin if v]
            out['finest_extra_median'] = float(np.median(fv)) if fv else None
            out['finest_extra_valid'] = len(fv)
        return out

    def anchor_unc(self, case):
        a8 = self.get(case, k('main', 'GAL', 'BDF2', DT0 / 8, 'tight'))
        a16 = self.get(case, k('main', 'GAL', 'BDF2', DT0 / 16, 'tight'))
        b8 = self.get(case, k('main', 'GAL', 'BDF2', DT0 / 8, 'tighter'))
        b16 = self.get(case, k('main', 'GAL', 'BDF2', DT0 / 16, 'tighter'))
        if None in (a8, a16, b8, b16) or not all(r['verified'] for r in (a8, a16, b8, b16)):
            return None
        if a8.get('d_half') is None or a8.get('s_h') is None or a16.get('s_h') is None:
            return None
        return a8['d_half'] + a8['s_h'] + a16['s_h']

    def time_error(self, key):
        """median / worst anchor discrepancy over cases, and the count of cases where it is resolved (>= 3 U_anc)."""
        v, res_ = [], 0
        for c in self.cases:
            r = self.get(c, key)
            u = self.anchor_unc(c)
            if r is None or r.get('anchor_257') is None or u is None or not r['verified']:
                continue
            v.append(r['anchor_257'])
            res_ += r['anchor_257'] >= 3 * u
        if not v:
            return None
        return dict(median=float(np.median(v)), worst=float(np.max(v)), resolved=res_, n=len(v))

    def failures(self):
        out = []
        for r in self.rows:
            if not r['verified']:
                out.append(dict(case=f"{r['cohort']}{r['case']}", run=r['run'], first_failed_step=int(r['stats']['first_fail']),
                                failed_steps=int(r['stats']['nfail']), exits=r['stats']['exits']))
        return out

    def timing(self):
        t = self.res['timing'].get(self.s)
        if not t:
            return {}, None
        by = {}
        for i in t['invocations']:
            by.setdefault(i['B'], []).append(i)
        out = {}
        tA = [x for i in t['invocations'] for x in (i['tA1'], i['tA2'])]
        for b, L in by.items():
            r = np.array([i['ratio'] for i in L])
            med = float(np.median(r))
            mad = float(np.median(np.abs(r - med)))
            dr = np.array([i['drift'] for i in L])
            out[b] = dict(n=len(r), median=med, iqr=[float(np.quantile(r, .25)), float(np.quantile(r, .75))],
                          outliers=int(np.sum(np.abs(r - med) > 3 * mad)),
                          ms=1e3 * float(np.median([i['tB'] for i in L])),
                          drift_median=float(np.median(dr)), drift_iqr=[float(np.quantile(dr, .25)), float(np.quantile(dr, .75))],
                          drift_outliers=int(np.sum(np.abs(dr - np.median(dr)) > 3 * np.median(np.abs(dr - np.median(dr))))))
        return out, 1e3 * float(np.median(tA))


def analyze(st, aud, att):
    s = st.s
    an = dict(setting=s, cases=len(st.cases), table=[], orders={}, timing={}, A_ms=None)
    want = sorted([('dev6', c) for c in range(6)] + [('val32', c) for c in range(32)])
    an['claims_eligible'] = sorted(st.cases) == want
    an['audit_pass'] = bool(aud['all_pass'] and aud['job_id'] == st.res['job_id'] and aud['commit'] == st.res['commit']
                            and aud['attempt'] == att)
    an['failures'] = st.failures()
    tim, A_ms = st.timing()
    an['timing'], an['A_ms'] = tim, A_ms
    comp = k('main', 'LSPG', 'BE', DT0, 'prod')
    dts = sorted({float(x.split('|')[3]) for x in st.runs if x.startswith('main|') and x.endswith('|prod')})
    for form in FORMS:
        for sc in SCHEMES:
            for dt in dts:
                key = k('main', form, sc, dt, 'prod')
                a = st.agg(key)
                if a is None:
                    continue
                ent = dict(key=key, form=form, scheme=sc, dt=dt, **{f'ST_{x}': a[x] for x in ('worst', 'median')},
                           verified=a['verified'], n=a['n'], alternating=a['alternating'], it_median=a['it_median'])
                for m in ('e_S', 'e_TX'):
                    b = st.agg(key, m)
                    ent[f'{m[2:]}_worst'], ent[f'{m[2:]}_median'] = b['worst'], b['median']
                rat, vm, pvt = [], [], []
                for c in st.cases:
                    r = st.get(c, key)
                    if r is None or not r['verified']:
                        continue
                    if r.get('anchor_full') and r.get('anchor_257') and st.anchor_unc(c) is not None:
                        rat.append(r['anchor_full'] / r['anchor_257'])
                    h = st.get(c, k('hq', form, sc, dt, 'prod'))
                    if h is not None and h['verified'] and h.get('vs_main') is not None:
                        vm.append(h['vs_main'])
                    t_ = st.get(c, k('main', form, sc, dt, 'tight'))
                    if t_ is not None and t_['verified'] and r.get('prod_vs_tight') is not None:
                        pvt.append(r['prod_vs_tight'])
                ent['full_over_257_median'] = float(np.median(rat)) if rat else None
                ent['full_over_257_n'] = len(rat)
                ent['hq_dist_worst'] = float(max(vm)) if vm else None
                ent['hq_pairs'] = len(vm)
                ent['prod_vs_tight_worst'] = float(max(pvt)) if pvt else None
                ent['pvt_pairs'] = len(pvt)
                te = st.time_error(key)
                ent['time_err'] = te
                ent['ratio'] = (tim.get(key) or {}).get('median') if key != comp else 1.
                ent['ratio_iqr'] = (tim.get(key) or {}).get('iqr')
                ent['ms'] = (tim.get(key) or {}).get('ms') if key != comp else A_ms
                hq = st.agg(k('hq', form, sc, dt, 'prod'))
                ent['hq_ST_worst'] = hq['worst'] if hq else None
                an['table'].append(ent)
    for form in FORMS:
        for sc in SCHEMES + ['TH06']:
            o = st.order_claim(form, sc)
            if not an['claims_eligible']:
                o['claim'] = f"diagnostic only ({len(st.cases)} cases; would read: {o['claim']})"
            elif not an['audit_pass']:
                o['claim'] = f"unavailable (audit failed; would read: {o['claim']})"
            an['orders'][f'{form}|{sc}'] = o
    old = []
    for sc in ('BE', 'CN', 'BDF2'):
        for f in (.5, 1, 2, 5):
            a = st.agg(k('old', 'LSPG', sc, DT0 * f, 'prod'))
            if a:
                old.append(dict(scheme=sc, dt=DT0 * f, ST_worst=a['worst'], ST_median=a['median'], verified=a['verified'], n=a['n']))
    an['old'] = old
    an['vendor_ms'] = (tim.get('vendor|LSPG|BE|0.005|prod') or {}).get('ms')
    an['vendor_ratio'] = (tim.get('vendor|LSPG|BE|0.005|prod') or {}).get('median')
    # ---- H1 / H2 / selection (A1.8, A2.18) ----
    T = {e['key']: e for e in an['table']}
    C = T.get(comp)
    n = len(st.cases)
    elig = [e for e in an['table'] if e['verified'] == n and e['dt'] >= DT0 / 2 - 1e-15 and e['ratio'] is not None]
    an['eligible'] = [e['key'] for e in elig]
    an['comparator'] = comp
    an['comparator_verified'] = bool(C and C['verified'] == n)
    prereq = an['audit_pass'] and an['claims_eligible']
    for ref in ('ST', 'TX'):
        if not prereq:
            an[f'select_{ref}'] = dict(fast='unavailable (audit failed or not the full cohort)',
                                       accurate='unavailable (audit failed or not the full cohort)')
            continue
        if not an['comparator_verified']:
            an[f'select_{ref}'] = dict(fast='unavailable (comparator unverified)', accurate='unavailable (comparator unverified)')
            continue
        cw, cm = C[f'{ref}_worst'], C[f'{ref}_median']
        so = lambda e: (SCHEMES.index(e['scheme']), )
        tie = lambda e: (e[f'{ref}_worst'], -e['dt'], FORMS.index(e['form']), SCHEMES.index(e['scheme']))
        fa = [e for e in elig if e[f'{ref}_worst'] <= cw and e[f'{ref}_median'] <= cm]
        fa.sort(key=lambda e: (e['ratio'],) + tie(e))
        ac = [e for e in elig if e['ratio'] <= 1.]
        ac.sort(key=lambda e: (e[f'{ref}_median'],) + tie(e))
        pick = lambda L: (L[0]['key'] if L[0]['key'] != comp else 'baseline retained (no improvement)') if L else 'none'
        an[f'select_{ref}'] = dict(fast=pick(fa), accurate=pick(ac))
        del so
    so2 = [e for e in elig if e['scheme'] != 'BE']
    h1 = [e for e in so2 if abs(e['dt'] - DT0) < 1e-15 and e['ST_median'] <= .7 * C['ST_median'] and e['ST_worst'] <= C['ST_worst']] if C else []
    h2 = [e for e in so2 if e['dt'] >= 2 * DT0 - 1e-15 and e['ST_worst'] <= C['ST_worst'] and e['ST_median'] <= C['ST_median']
          and e['ratio'] <= .75] if C else []
    ok = prereq and an['comparator_verified']
    an['H1'] = dict(passed=bool(h1) if ok else None, arms=[e['key'] for e in h1])
    an['H2'] = dict(passed=bool(h2) if ok else None, arms=[e['key'] for e in h2])
    an['G1a'] = st.res['g1a']
    return an


def plots(st, an, rule):
    s = st.s
    T = [e for e in an['table']]
    comp = next(e for e in T if e['key'] == an['comparator'])
    # 1. error against ST and anchor discrepancy vs dt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for form in FORMS:
        for sc in SCHEMES:
            E = sorted([e for e in T if e['form'] == form and e['scheme'] == sc], key=lambda e: e['dt'])
            if not E:
                continue
            o = an['orders'][f'{form}|{sc}']
            lab = f"{form} {NAMES[sc]} (order: {o['claim']}, median p={o['primary_median']:.2f})" if o['primary_median'] is not None else f'{form} {NAMES[sc]}'
            x = [e['dt'] for e in E]
            ax[0].plot(x, [100 * e['ST_worst'] for e in E], color=COLORS[sc], lw=2, ms=7, label=f'{form} {NAMES[sc]}', **STYLE[form])
            y = [100 * e['time_err']['median'] if e['time_err'] else np.nan for e in E]
            ax[1].plot(x, y, color=COLORS[sc], lw=2, ms=7, label=lab, **STYLE[form])
            un = [(xx_, yy_) for e, xx_, yy_ in zip(E, x, y) if e['time_err'] and e['time_err']['resolved'] < e['time_err']['n'] / 2]
            if un:
                ax[1].scatter(*zip(*un), s=70, facecolors='white', edgecolors=COLORS[sc], zorder=4)
            bad = [(xx_, 100 * e['ST_worst']) for e, xx_ in zip(E, x) if e['verified'] < e['n']]
            if bad:
                ax[0].scatter(*zip(*bad), marker='x', s=80, color='#0b0b0b', zorder=5)
    ax[0].axhline(100 * comp['ST_worst'], color='#52514e', ls=':', lw=1.5, label='LSPG BE at $\\Delta t_0$ (deployed)')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('time step $\\Delta t$')
    ax[0].set_ylabel('worst error vs ST reference (% of $\\|u_0\\|$), PROVISIONAL')
    ax[0].set_title(f'{s}: worst error over dev6 ∪ val32 vs $\\Delta t$')
    ax[1].set_xscale('log')
    if any(e['time_err'] for e in T):
        ax[1].set_yscale('log')
    ax[1].set_xlabel('time step $\\Delta t$')
    ax[1].set_ylabel('median anchor discrepancy (% of $\\|u_0\\|$)')
    ax[1].set_title(f'{s}: anchor discrepancy (GAL-BDF2, $\\Delta t_0/16$); hollow = unresolved', fontsize=10)
    xx = np.array([DT0 / 8, 10 * DT0])
    for p_, ls in ((1, (0, (1, 3))), (2, (0, (4, 3)))):
        y0 = np.nanmedian([e['time_err']['median'] for e in T if e['time_err'] and abs(e['dt'] - DT0) < 1e-15 and e['scheme'] == ('BE' if p_ == 1 else 'BDF2')] or [np.nan])
        ax[1].plot(xx, 100 * y0 * (xx / DT0) ** p_, color='#9a9a96', lw=1, ls=ls, label=f'slope {p_} guide')
    for a_ in ax:
        a_.grid(True, which='major', color='#e4e4e0', lw=.6)
        a_.spines[['top', 'right']].set_visible(False)
    ax[0].legend(fontsize=7.5, frameon=False)
    ax[1].legend(fontsize=6.8, frameon=False)
    fig.suptitle(f'Burgers 2D, {RULE_NAME.get(rule, rule)}, $L={st.res["mesh"]}$, {len(st.cases)} cases — errors are provisional '
                 '(first-order references); x = unverified', fontsize=10)
    fig.tight_layout()
    p1 = HERE / 'plots' / f'error_vs_dt_{s}.png'
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    # 2. error vs cost
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    for form in FORMS:
        for sc in SCHEMES:
            E = sorted([e for e in T if e['form'] == form and e['scheme'] == sc and e['ms'] is not None], key=lambda e: e['dt'])
            if not E:
                continue
            ax.plot([e['ms'] for e in E], [100 * e['ST_median'] for e in E], color=COLORS[sc], lw=2, ms=7,
                    label=f'{form} {NAMES[sc]}', **STYLE[form])
            bad = [(e['ms'], 100 * e['ST_median']) for e in E if e['verified'] < e['n']]
            if bad:
                ax.scatter(*zip(*bad), marker='x', s=80, color='#0b0b0b', zorder=5)
            if form == 'LSPG' and sc in ('BE', 'BDF2'):
                for e in E:
                    ax.annotate(f"{dtf(e['dt'])}$\\Delta t_0$", (e['ms'], 100 * e['ST_median']), fontsize=7, color='#52514e',
                                xytext=(4, 4), textcoords='offset points')
    ax.scatter([comp['ms']], [100 * comp['ST_median']], s=160, facecolors='none', edgecolors='#0b0b0b', lw=1.5,
               label='deployed: LSPG BE $\\Delta t_0$', zorder=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('end-to-end query time per case (ms; median of paired A–B–A on one GPU, six dev6 timing cases)')
    ax.set_ylabel('median error vs ST over dev6 ∪ val32 (%), PROVISIONAL')
    ax.set_title(f'{s}: accuracy vs cost per time scheme ({RULE_NAME.get(rule, rule)}); x = unverified')
    ax.grid(True, which='major', color='#e4e4e0', lw=.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    p2 = HERE / 'plots' / f'error_vs_cost_{s}.png'
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return [p1.name, p2.name]


def main():
    atts = sys.argv[1:] or ['a1kfast', 'a1kacc']
    (HERE / 'plots').mkdir(exist_ok=True)
    md = []
    W = md.append
    tl = json.loads((HERE / 'checks/test_lmm.json').read_text())
    ev = json.loads((HERE / 'checks/evidence-be-gap.json').read_text())
    sections, runs_tbl = [], []
    for att in atts:
        arc = HERE / 'runs' / att / 'archive'
        res = json.loads((arc / 'output/result.json').read_text())
        aud = json.loads((HERE / 'checks' / f'audit-{att}.json').read_text())
        runs_tbl.append((att, res, aud))
        for s in res['config']['settings']:
            st = Setting(res, s)
            an = analyze(st, aud, att)
            rule = res['config']['rules'][s]['main']['rule']
            an['plots'] = plots(st, an, rule)
            (HERE / 'checks' / f'analysis-{att}-{s}.json').write_text(json.dumps(an, indent=1, default=float) + '\n')
            sections.append((att, s, rule, res, an, aud))

    W('# Second-order time stepping for the off-mesh Burgers 2D ROM (jcp-time2)')
    W('')
    smoke = any(not x[4]['claims_eligible'] for x in sections)
    W('Crank–Nicolson, Crank–Nicolson with a damped start, and BDF2 replace backward Euler in the off-mesh coordinate-bank '
      'ROM of the 2D quadrature lane. Within a solve form (LSPG, the deployed one, or GAL) only the time discretisation '
      'changes; GAL also changes the step equation. Status: '
      + ('**MACHINERY DIAGNOSTICS ONLY (smoke / partial cohort): no claim, order or selection below is a result**. '
         if smoke else '**provisional 2D development/validation results (dev6 ∪ val32, 38 cases)**. ') +
      f"Meshes: {', '.join(sorted({f'${x[3][chr(109)+chr(101)+chr(115)+chr(104)]}^2$' for x in sections}))}." + ' '
      ' Every accuracy number is PROVISIONAL because the references are first '
      'order in time and upwind in space (the second-order references lane has not run). Generated by `make_report.py` from '
      'the pulled job results; the pre-registration is `DESIGN.md` (amendments A1–A8 apply to these jobs; A9 to the separate FOM comparison).')
    W('')
    W('## What ran')
    W('')
    W('| job | settings | mesh | cases | GPU | job id | elapsed (h) | commit | audit |')
    W('|---|---|---|---|---|---|---|---|---|')
    for att, res, aud in runs_tbl:
        n = sum(1 for _ in {(r['cohort'], r['case']) for r in res['rows']})
        W(f"| `{att}` | {', '.join(res['config']['settings'])} | ${res['mesh']}^2$ | {n} | {res['gpu']} | {res['job_id']} | "
          f"{res.get('elapsed_seconds', 0) / 3600:.2f} | `{(res['commit'] or '')[:9]}` | {'pass' if aud['all_pass'] else 'FAIL: ' + ', '.join(k for k, v in aud['checks'].items() if not v)} |")
    W('')
    W('## The reduced step')
    W('')
    W('A two-step linear multistep method with coefficients $(\\alpha_0,\\alpha_1,\\alpha_2;\\beta_0,\\beta_1)$ on the tested residual')
    W('')
    W('$$ \\mathcal R_n(c)=A(\\alpha_0c+\\alpha_1c_n+\\alpha_2c_{n-1})+\\Delta t\\,(\\beta_0F(c)+\\beta_1F(c_n)),\\qquad F(c)=N(c)+\\nu\\Lambda Ac, $$')
    W('')
    W('solved either as **LSPG** (minimise $\\lVert D\\mathcal R_n\\rVert$, $D=(\\alpha_0+\\beta_0\\Delta t\\nu\\Lambda)^{-1}$; for backward Euler exactly the '
      'deployed step) or as **GAL** ($Q^{\\mathsf T}\\mathcal R_n=0$, $A=QR$: the same method applied to the Galerkin ODE $A^{\\mathsf T}A\\dot c=-A^{\\mathsf T}F(c)$).')
    W('')
    W('```mermaid')
    W('flowchart LR')
    W('  B[(frozen bank, rotation, rule)]:::frozen --> OPS[A, Lambda, tested advection N]:::frozen')
    W('  OPS --> STEP{{LMM step: LSPG or GAL, Levenberg-Marquardt}}:::solved')
    W('  H[history c_n, c_n-1, F of c_n]:::solved --> STEP')
    W('  STEP --> H')
    W('  STEP --> OUT[six output states]:::solved --> ERR[errors vs ST, S, TX; anchor; self-convergence]:::metric')
    W('  classDef frozen fill:#dbeafe,stroke:#1d4ed8;')
    W('  classDef solved fill:#dcfce7,stroke:#15803d;')
    W('  classDef metric fill:#fef3c7,stroke:#b45309;')
    W('```')
    W('')
    W('## Motivation, re-checked from the 2D lane data')
    W('')
    r1 = next(r for r in ev['rows'] if r['mesh'] == 1024 and r['setting'] == 'acc' and r['arm'] == 'gauss96')
    r2 = next(r for r in ev['rows'] if r['mesh'] == 1024 and r['setting'] == 'fast' and r['arm'] == 'fib1597')
    sst = ev['S_minus_ST']
    W(f"Backward-Euler rollouts of the 2D lane at $1024^2$ (38 cases): accurate setting (Gauss $96^2$) worst/median error "
      f"{r1['ST_worst']:.3f}/{r1['ST_median']:.3f} % against ST and {r1['S_worst']:.3f}/{r1['S_median']:.3f} % against S; "
      f"fast setting (Fibonacci 1597) {r2['ST_worst']:.3f}/{r2['ST_median']:.3f} % vs {r2['S_worst']:.3f}/{r2['S_median']:.3f} %. "
      f"The per-case gap ST − S has median {r1['gap_median']:.2f} pp and maximum {r1['gap_max']:.2f} pp (accurate), "
      f"{r2['gap_median']:.2f}/{r2['gap_max']:.2f} pp (fast). These are reference-sensitive gaps, not norms of the time error. "
      f"The full-order model's own S-to-ST distance (its backward-Euler time-error proxy at $8192^2$) is {sst['worst']:.2f} % "
      f"worst, {sst['median']:.2f} % median — the same size as the accurate ROM's error against ST. All of these numbers are "
      "PROVISIONAL (first-order references).")
    W('')
    W('## Machinery checks')
    W('')
    W(f"Manufactured tests (`test_lmm.py`, CPU, independent SciPy oracle): {'all gates pass' if tl['all_pass'] else 'FAILED'}. "
      'Observed orders against the oracle on the finest pair:')
    W('')
    W('| scheme | GAL order | LSPG order |')
    W('|---|---|---|')
    for sc in ('BE', 'TH06', 'CN', 'CNR', 'BDF2'):
        W(f"| {NAMES[sc]} | {tl['info'][f'order_GAL_{sc}']['orders'][-1]:.3f} | {tl['info'][f'order_LSPG_{sc}']['orders'][-1]:.3f} |")
    W('')
    W(f"Mutations detected: {sum(v['detected'] for v in tl['mutations'].values())} of {len(tl['mutations'])}. On the manufactured "
      'system the LSPG versions of CN and BDF2 converge at first order to the Galerkin trajectory, as predicted in DESIGN A1.2: '
      'the LSPG test space depends on $\\Delta t$.')
    W('')
    for att, s, rule, res, an, aud in sections:
        g = an['G1a']
        W(f'## Setting `{s}` ({RULE_NAME.get(rule, rule)}, $R\'={res["setup"][s]["R_prime"]}$, $M={res["setup"][s]["M"]}$, '
          f'$L={res["mesh"]}$, {an["cases"]} cases)')
        W('')
        if s == 'wide':
            W('**Secondary setting (DESIGN A7).** Its rule (Gauss $128^2$) was not selected by any study at $R\'=512$; the '
              'quadrature-sensitivity column (Gauss $192^2$) is an indicator, not a certificate, and exists only for the '
              'production arms with $\\Delta t\\ge\\Delta t_0/2$.')
            W('')
        W(f"Gates: G1a (generic LSPG-BE = vendor BE query, same job) max coefficient difference "
          f"{max(x['W_rel'] for x in g):.1e}, field {max(x['field_rel'] for x in g):.1e} "
          f"({'pass' if aud['checks'].get(f'G1a_{s}') else 'FAIL'}); G1b vs the 2D lane: "
          f"{aud['info'].get(f'G1b_{s}', {}).get('max_abs_diff')} (tol 1e-6, reported); metrics reconstructed independently: "
          f"{'pass' if aud['checks'].get(f'metrics_reconstructed_{s}') else 'FAIL'}; timed outputs match accuracy outputs: "
          f"{'pass' if aud['checks'].get(f'G4_timed_outputs_match_{s}') else 'FAIL'}; $A$ rank {res['setup'][s]['A']['rank']}, "
          f"condition {res['setup'][s]['A']['cond']:.2f}.")
        W('')
        W(f"### Accuracy and cost per scheme and step (production tolerance; errors PROVISIONAL, % of $\\lVert u_0\\rVert$)")
        W('')
        W('| form | scheme | $\\Delta t/\\Delta t_0$ | ST worst | ST median | TX worst | TX median | S worst | S median | anchor disc. median (%) | resolved | prod vs tight worst (%) | full/257 | paired ratio (IQR) | ms | verified | alternating | quad. sens. worst (%) | HQ ST worst |')
        W('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for e in an['table']:
            te = e['time_err']
            res_ = f"{te['resolved']}/{te['n']}" if te else '—'
            rat = '—' if e['ratio'] is None else f"{e['ratio']:.3f}"
            if e['ratio_iqr']:
                rat += f" ({e['ratio_iqr'][0]:.2f}–{e['ratio_iqr'][1]:.2f})"
            ms = '—' if e['ms'] is None else f"{e['ms']:.1f}"
            fo = '—' if e['full_over_257_median'] is None else f"{e['full_over_257_median']:.2f} ({e['full_over_257_n']})"
            W(f"| {e['form']} | {NAMES[e['scheme']]} | {dtf(e['dt'])} | {pct(e['ST_worst'])} | {pct(e['ST_median'])} | "
              f"{pct(e['TX_worst'])} | {pct(e['TX_median'])} | {pct(e['S_worst'])} | {pct(e['S_median'])} | "
              f"{pct(te['median']) if te else '—'} | {res_} | {pct(e['prod_vs_tight_worst'])} ({e['pvt_pairs']}) | {fo} | {rat} | {ms} | "
              f"{e['verified']}/{e['n']} | {e['alternating']} | {pct(e['hq_dist_worst'])} ({e['hq_pairs']}) | {pct(e['hq_ST_worst'])} |")
        W('')
        W('Timing detail (all subjects; ratio = candidate / mean of the flanking deployed-BE runs):')
        W('')
        W('| subject | samples | ratio median | ratio IQR | ratio outliers | drift median | drift IQR | drift outliers |')
        W('|---|---|---|---|---|---|---|---|')
        for b_, t_ in sorted(an['timing'].items()):
            W(f"| `{b_}` | {t_['n']} | {t_['median']:.3f} | {t_['iqr'][0]:.3f}–{t_['iqr'][1]:.3f} | {t_['outliers']} | "
              f"{t_['drift_median']:.3f} | {t_['drift_iqr'][0]:.3f}–{t_['drift_iqr'][1]:.3f} | {t_['drift_outliers']} |")
        W('')
        if an['failures']:
            W(f"Unverified trajectories ({len(an['failures'])}; excluded from orders, anchor and selection):")
            W('')
            W('| case | run | first failed step | failed steps | exits [budget, tol, tiny, damping, stationary] |')
            W('|---|---|---|---|---|')
            for f_ in an['failures'][:60]:
                W(f"| {f_['case']} | `{f_['run']}` | {f_['first_failed_step']} | {f_['failed_steps']} | {f_['exits']} |")
            if len(an['failures']) > 60:
                W(f"| … | {len(an['failures']) - 60} more in `checks/analysis-{att}-{s}.json` | | | |")
            W('')
        else:
            W('Every trajectory of this setting was verified (every step met its acceptance test).')
            W('')
        W(f"Deployed comparator: `{an['comparator']}`, {an['A_ms']:.1f} ms median; the vendor backward-Euler query (G1a "
          f"agreement within tolerance) paired ratio {an['vendor_ratio']:.3f} (generic-implementation overhead check).")
        W('')
        W('### Observed temporal order (self-convergence, tight tolerance)')
        W('')
        W('| form | scheme | claim | primary median $p$ | valid / cases | adjacent median $p$ | cases with both triples | outside [1.7, 2.3] | outside [0.8, 1.25] | coarse triple median (valid) | finer GAL triple median |')
        W('|---|---|---|---|---|---|---|---|---|---|---|')
        for kk, o in an['orders'].items():
            f_, sc = kk.split('|')
            fmt = lambda x: '—' if x is None else f'{x:.2f}'
            W(f"| {f_} | {NAMES[sc]} | {o['claim']} | {fmt(o['primary_median'])} | {o['primary_valid']}/{o['cases']} | "
              f"{fmt(o['adjacent_median'])} | {o['n_both']} | {o['outliers_2']} | {o['outliers_1']} | "
              f"{fmt(o['coarse_median'])} ({o['coarse_valid']}) | {fmt(o.get('finest_extra_median'))} |")
        W('')
        W('### Hypotheses and selection (dev6 ∪ val32; PROVISIONAL references)')
        W('')
        W(f"- H1 (a CN/BDF2-family arm at $\\Delta t_0$ with median ST error ≤ 0.7× and worst ≤ the deployed BE): "
          f"**{ {True: 'passed', False: 'failed', None: 'unavailable'}[an['H1']['passed']] }** ({', '.join(an['H1']['arms']) or 'none'}).")
        W(f"- H2 (a CN/BDF2-family arm with $\\Delta t\\ge2\\Delta t_0$, worst and median ST ≤ BE at $\\Delta t_0$, paired time ratio ≤ 0.75): "
          f"**{ {True: 'passed', False: 'failed', None: 'unavailable'}[an['H2']['passed']] }** ({', '.join(an['H2']['arms']) or 'none'}).")
        W(f"- Selection against ST (fast-equal-accuracy: cheapest arm with worst and median ST ≤ the comparator's; accurate-equal-cost: smallest median ST among arms with paired ratio ≤ 1, no worst-error constraint): fast-equal-accuracy **{an['select_ST']['fast']}**; accurate-equal-cost **{an['select_ST']['accurate']}**. "
          f"Sensitivity with TX: {an['select_TX']['fast']} / {an['select_TX']['accurate']}"
          f"{' (reference-dependent)' if an['select_TX'] != an['select_ST'] else ''}.")
        W('')
        if an['old']:
            W('Old method (deployed mesh lattice $63^2$, LSPG), worst / median ST error (%, PROVISIONAL):')
            W('')
            W('| scheme | $\\Delta t/\\Delta t_0$ | ST worst | ST median | verified |')
            W('|---|---|---|---|---|')
            for e in an['old']:
                W(f"| {NAMES[e['scheme']]} | {dtf(e['dt'])} | {pct(e['ST_worst'])} | {pct(e['ST_median'])} | {e['verified']}/{e['n']} |")
            W('')
        for p_ in an['plots']:
            W(f'![{p_}](plots/{p_})')
            W('')
    import report_extra as RX
    RX.fom_section(W)
    RX.d3_section(W)
    W('## What is provisional, and why')
    W('')
    W('- Every error against ST, S or TX: the references are backward Euler in time and sign-upwind in space at $8192^2$. '
      'TX removes only the leading backward-Euler term and assumes the asymptotic regime. A second-order ROM can be closer to '
      'the true solution than ST is; its ST error then partly measures ST\'s own time error.')
    W('- Anchor discrepancies concern the fixed reduced model and quadrature rule: they say nothing about the total PDE error or the spatial, representation and quadrature errors.')
    W('- Anchor discrepancies are distances to the GAL-BDF2 $\\Delta t_0/16$ rollout. They estimate the time-step error of a rollout only where GAL-BDF2 is shown to be second order on these data (its claim above) and the distance is resolved (≥ 3× the anchor uncertainty indicator); otherwise read them only as distances to that rollout.')
    W('- Timings are for the six dev6 timing cases on one GPU.')
    W('')
    W('## Glossary')
    W('')
    W('- **BE / CN / CN-R / BDF2 / TH06**: backward Euler (deployed); Crank–Nicolson (trapezoidal in advection and diffusion); '
      'CN whose first two steps are BE; second-order backward differentiation (first step BE); the θ = 0.6 method (a first-order control).')
    W('- **LSPG**: the deployed least-squares solve of each step over the $M$ sine tests. **GAL**: the fixed-test Galerkin version (residual projected onto the range of $A$ and set to zero).')
    W('- **ST / S / TX**: full-order references at $8192^2$ with $\\Delta t_0/16$ / with $\\Delta t_0$ / their Richardson combination $(16\\,\\mathrm{ST}-\\mathrm{S})/15$.')
    W('- **ST worst / median**: the largest / median over the 38 cases of the worst-over-time ($t=0.05$–$0.25$) error on the $257^2$ shared nodes, in % of $\\lVert u_0\\rVert$.')
    W('- **anchor disc.**: distance of a rollout from the anchor (GAL-BDF2 at $\\Delta t_0/16$, tight tolerance), worst over time, median over cases, % of $\\lVert u_0\\rVert$ on the $257^2$ nodes; a time-step error estimate only under the conditions stated above.')
    W('- **anchor uncertainty indicator**: per case, the anchor\'s own change from $\\Delta t_0/8$ to $\\Delta t_0/16$ plus the two solve-sensitivity indicators; **resolved**: cases whose anchor discrepancy exceeds 3× it.')
    W('- **prod vs tight**: distance between the production-tolerance rollout and the tight-tolerance rollout of the same arm (a production-to-tight solve-sensitivity indicator, not an established solver error).')
    W('- **full/257**: median ratio of the anchor discrepancy measured on the full $1024^2$ mesh (normalised by the full-mesh $\\lVert u_0\\rVert$) to that on the $257^2$ nodes; near 1 means the shared nodes do not hide fine-scale differences.')
    W('- **adjacent check**: the primary order must also hold on the coarser triple; "(adjacent check unresolved)" when fewer than half the cases have both triples valid; coarse triple = $(2\\Delta t_0,\\Delta t_0,\\Delta t_0/2)$, reported only.')
    W('- **solve-sensitivity indicator**: distance between the tight and the tighter rollout of the same arm.')
    W('- **ratio / drift outliers**: samples further than 3 median absolute deviations from the median; drift = second baseline time / first.')
    W('- **HQ ST worst**: worst ST error with the finer rule; **quad. sens.**: worst distance between the finer-rule and the main-rule rollouts.')
    W('- **observed order $p$**: $\\log_2$ of the ratio of successive self-differences $\\lVert u_h-u_{h/2}\\rVert$ when $h$ is halved; primary triple $(\\Delta t_0/2,\\Delta t_0/4,\\Delta t_0/8)$, adjacent $(\\Delta t_0,\\Delta t_0/2,\\Delta t_0/4)$; a triple is valid when each difference exceeds 10× the sum of the solve-sensitivity indicators of its two rollouts, and all its rollouts (tight and tighter) are verified.')
    W('- **claim**: "order 2" / "order 1" when ≥ 80 % of cases have a valid primary order, ≥ 80 % of those lie in the band, and, among cases with both triples valid, ≥ 80 % have both orders in the band (DESIGN A2.12, A3.3, A4.1); "not established" otherwise. "Not established" for an LSPG arm does not mean order 1.')
    W('- **paired time ratio**: candidate time divided by the mean of the deployed BE times run just before and after it (A–B–A), median over 6 cases × 3 repetitions; IQR is the interquartile range.')
    W('- **ms**: median end-to-end query time (initial fit, stepping, decoding six fields).')
    W('- **verified**: cases in which every time step met its solver acceptance test.')
    W('- **alternating**: cases whose stiff-mode increments alternate in sign (index > 0.5; a diagnostic only).')
    W('- **G1a / G1b**: the new code\'s backward Euler equals the deployed code in the same job / equals the 2D lane\'s stored numbers.')
    W('- **dev6, val32**: development (6) and validation (32) cohorts of initial bumps; selection uses only these.')
    W('- **pp**: percentage points of $\\lVert u_0\\rVert$.')
    (HERE / 'report.md').write_text('\n'.join(md) + '\n')
    print('wrote report.md', [x[4]['plots'] for x in sections])


if __name__ == '__main__':
    main()
