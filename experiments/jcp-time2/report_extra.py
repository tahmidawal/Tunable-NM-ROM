"""Sections of report.md for the fair full-order comparison (fom1k, DESIGN section 9, A9-A11) and the 3D job (b3d65,
A10.2). Imported by make_report.py; every number is read from the pulled job results and audits."""
from __future__ import annotations

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
pct = lambda x: '—' if x is None or not np.isfinite(x) else f'{100 * x:.3f}'


def _timing(inv):
    by = {}
    for i in inv:
        by.setdefault(i['B'], []).append(i)
    out = {}
    for b, L in by.items():
        r = np.array([i['ratio'] for i in L])
        out[b] = dict(n=len(r), ratio=float(np.median(r)), iqr=[float(np.quantile(r, .25)), float(np.quantile(r, .75))],
                      ms=1e3 * float(np.median([i['tB'] for i in L])))
    tA = [x for i in inv for x in (i['tA1'], i['tA2'])]
    return out, (1e3 * float(np.median(tA)) if tA else None)


# ------------------------------------------------------------------------------------------------ FOM ----
def fom_section(W, att='fom1k'):
    arc = HERE / 'runs' / att / 'archive'
    f = arc / 'output/result.json'
    if not f.exists():
        return None
    res = json.loads(f.read_text())
    aud = json.loads((HERE / 'checks' / f'audit-{att}.json').read_text())
    ok = aud['all_pass'] and aud['job_id'] == res['job_id'] and aud['commit'] == res['commit']
    tim, A_ms = _timing(res['timing']['invocations'])
    tim[res['timing']['A']] = dict(n=None, ratio=1., iqr=None, ms=A_ms)
    agg = {}
    for r in res['rows']:
        agg.setdefault(r['run'], []).append(r)
    for r in res['rom_rows']:
        agg.setdefault(r['run'], []).append(r)
    T = {}
    for k, L in agg.items():
        st = np.array([x['e_ST'] for x in L])
        T[k] = dict(n=len(L), verified=sum(x['verified'] for x in L), worst=float(st.max()), median=float(np.median(st)),
                    ms=tim.get(k, {}).get('ms'), ratio=tim.get(k, {}).get('ratio'))
    W(f'## Fair full-order comparison (`{att}`, $L={res["mesh"]}$; DESIGN section 9, A9–A11)')
    W('')
    W(f"Job {res['job_id']} on {res['gpu']}; independent audit: {'pass' if ok else 'FAIL: ' + ', '.join(k for k, v in aud['checks'].items() if not v)}. "
      'The full-order model uses the same two-step method (sign-upwind advection, 5-point Laplacian, Newton–BiCGStab with the '
      'DST Helmholtz preconditioner). Accuracy over the 38 cases; timing over six dev6 cases, three repetitions, one GPU, '
      'ratios relative to the full-order BE at $\\Delta t_0$. Errors PROVISIONAL (first-order references).')
    W('')
    W('FOM order check ($L=256$, dev6 cases 0 and 2, finest pair $\\Delta t_0/2\\to\\Delta t_0/8$):')
    W('')
    W('| scheme | case | orders at $2\\Delta t_0$, $\\Delta t_0$, $\\Delta t_0/2$ | gated |')
    W('|---|---|---|---|')
    for o in res['order']:
        W(f"| {NAMES[o['scheme']]} | {o['cohort']}{o['case']} | {', '.join(f'{v:.2f}' for v in o['orders'].values())} | "
          f"{'no (A11.1: reported only)' if o['scheme'] == 'CN' else 'yes'} |")
    W('')
    W('| solver | scheme | $\\Delta t/\\Delta t_0$ | tolerance | ST worst | ST median | verified | ms | ratio to FOM BE $\\Delta t_0$ |')
    W('|---|---|---|---|---|---|---|---|---|')
    keys = sorted(T, key=lambda k: (not k.startswith('fom'), k))
    for k in keys:
        e = T[k]
        parts = k.split('|')
        if parts[0] == 'fom':
            cal = res['calibration'][f'{parts[1]}|{parts[2]}']
            W(f"| FOM | {NAMES[parts[1]]} | {parts[2]} | ntol {cal['chosen'][0]:g} ({cal['status']}) | {pct(e['worst'])} | "
              f"{pct(e['median'])} | {e['verified']}/{e['n']} | {e['ms']:.1f} | {e['ratio']:.3f} |")
        else:
            W(f"| ROM {parts[1]} {parts[3]} | {NAMES[parts[4]]} | {parts[5]} | production | {pct(e['worst'])} | {pct(e['median'])} | "
              f"{e['verified']}/{e['n']} | {e['ms']:.1f} | {e['ratio']:.3f} |")
    W('')
    # matched comparison (A9.4)
    foms = [k for k in T if k.startswith('fom') and T[k]['verified'] == T[k]['n'] and T[k]['ms'] is not None]
    W('Matched comparison (A9.4): for each verified ROM arm, the cheapest verified FOM configuration whose cohort-worst ST '
      'error is no larger; speed-up = FOM time / ROM time (same job, same GPU). "Best among the listed configurations" only.')
    W('')
    W('| ROM arm | ROM ST worst | ROM ms | matched FOM | FOM ST worst | FOM ms | speed-up |')
    W('|---|---|---|---|---|---|---|')
    match = []
    for k in [k for k in keys if k.startswith('rom') and T[k]['verified'] == T[k]['n']]:
        c = [f for f in foms if T[f]['worst'] <= T[k]['worst']]
        if not c:
            W(f"| `{k}` | {pct(T[k]['worst'])} | {T[k]['ms']:.1f} | none at this accuracy | — | — | — |")
            continue
        fbest = min(c, key=lambda f: T[f]['ms'])
        sp = T[fbest]['ms'] / T[k]['ms']
        match.append((k, fbest, sp))
        W(f"| `{k}` | {pct(T[k]['worst'])} | {T[k]['ms']:.1f} | `{fbest}` | {pct(T[fbest]['worst'])} | {T[fbest]['ms']:.1f} | {sp:.1f}× |")
    W('')
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    for fam, mk in (('fom', 's'), ('rom', 'o')):
        for sc in SCHEMES:
            ks = [k for k in T if k.startswith(fam) and (k.split('|')[1] if fam == 'fom' else k.split('|')[4]) == sc
                  and (fam == 'fom' or k.split('|')[3] == 'LSPG')]
            for setting in (['-'] if fam == 'fom' else sorted({k.split('|')[1] for k in ks})):
                kk = [k for k in ks if fam == 'fom' or k.split('|')[1] == setting]
                kk = sorted(kk, key=lambda k: float(k.split('|')[-1]))
                if not kk:
                    continue
                lab = f"FOM {NAMES[sc]}" if fam == 'fom' else f"ROM {setting} LSPG {NAMES[sc]}"
                ax.plot([T[k]['ms'] for k in kk], [100 * T[k]['worst'] for k in kk], marker=mk, color=COLORS[sc], lw=1.6,
                        ms=6, ls='-' if fam == 'rom' else '--', label=lab, alpha=.9 if setting != 'fast' else .55)
    ax.set_xscale('log')
    ax.set_xlabel('end-to-end time per case (ms, paired A–B–A, six dev6 cases)')
    ax.set_ylabel('worst ST error over 38 cases (%), PROVISIONAL')
    ax.set_title(f'Full-order vs reduced, same time schemes, $L={res["mesh"]}$', fontsize=10)
    ax.grid(True, which='major', color='#e4e4e0', lw=.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / 'plots' / 'fom_vs_rom.png', dpi=150)
    plt.close(fig)
    W('![fom_vs_rom.png](plots/fom_vs_rom.png)')
    W('')
    (HERE / 'checks' / f'analysis-{att}.json').write_text(json.dumps(dict(table=T, matched=match, audit_pass=ok), indent=1) + '\n')
    return dict(T=T, matched=match, audit_pass=ok)


# ------------------------------------------------------------------------------------------------- 3D ----
def d3_section(W, att='b3d65'):
    arc = HERE / 'runs' / att / 'archive'
    f = arc / 'output/result.json'
    if not f.exists():
        return None
    res = json.loads(f.read_text())
    aud = json.loads((HERE / 'checks' / f'audit-{att}.json').read_text())
    ok = aud['all_pass'] and aud['job_id'] == res['job_id'] and aud['commit'] == res['commit']
    n = res['config']['cohort_count']
    need, half = int(np.ceil(.8 * n)), n / 2
    DT0 = .01
    W(f'## Burgers 3D (`{att}`, $65^3$, validation cohort 923801, first {n} cases; A10.2)')
    W('')
    W(f"Job {res['job_id']} on {res['gpu']}; audit: {'pass' if ok else 'FAIL: ' + ', '.join(k for k, v in aud['checks'].items() if not v)}. "
      'Reference: the 3D lane\'s 513-node backward-Euler solution at $\\Delta t=\\Delta t_0/4$ on the 65-node lattice (an ST-type '
      'reference; PROVISIONAL). Comparator: the deployed fixed-sweep BE at $\\Delta t_0=0.01$. Timing: first four cases × 3.')
    W('')
    out = {}
    for arm in sorted({r['arm'] for r in res['rows']}):
        rows = [r for r in res['rows'] if r['arm'] == arm]
        by = {(r['run'], r['case']): r for r in rows}
        vend = [v for v in res['vendor_rows'] if v['arm'] == arm]
        tim, A_ms = _timing(res['timing'].get(arm, {}).get('invocations', []))
        ve = np.array([v['e_ref'] for v in vend])

        def unc(j):
            a8 = by.get((f'GAL|BDF2|{DT0 / 8:.8g}|tight', j))
            r8 = by.get((f'GAL|BDF2|{DT0 / 8:.8g}|tighter', j))
            a16 = by.get((f'GAL|BDF2|{DT0 / 16:.8g}|tight', j))
            r16 = by.get((f'GAL|BDF2|{DT0 / 16:.8g}|tighter', j))
            if None in (a8, r8, a16, r16) or not all(x['verified'] for x in (a8, r8, a16, r16)):
                return None
            return a8['d_half'] + a8['s_h'] + a16['s_h']

        def triple(form, sc, h, j):
            hs = [h, h / 2, h / 4]
            rt = [by.get((f'{form}|{sc}|{x:.8g}|tight', j)) for x in hs]
            rr = [by.get((f'{form}|{sc}|{x:.8g}|tighter', j)) for x in hs]
            if any(x is None for x in rt + rr) or not all(x['verified'] for x in rt + rr):
                return None, False
            d1, d2 = rt[0].get('d_half'), rt[1].get('d_half')
            s = [x.get('s_h') for x in rt]
            if None in (d1, d2) or None in s or d1 <= 0 or d2 <= 0:
                return None, False
            return float(np.log2(d1 / d2)), bool(d1 > 10 * (s[0] + s[1]) and d2 > 10 * (s[1] + s[2]))
        W(f'### `{arm}`')
        W('')
        W(f"Deployed fixed-sweep BE at $\\Delta t_0$: error worst / median {pct(ve.max())} / {pct(float(np.median(ve)))} %, "
          f"{A_ms:.1f} ms median; its distance from the generic adaptive BE at $\\Delta t_0$: median "
          f"{pct(float(np.median([v['vs_generic_BE'] for v in vend])))} %.")
        W('')
        W('| form | scheme | $\\Delta t/\\Delta t_0$ | error worst | error median | anchor disc. median (%) | resolved | verified | ratio to deployed | ms |')
        W('|---|---|---|---|---|---|---|---|---|---|')
        tab = []
        for form in ('LSPG', 'GAL'):
            for sc in SCHEMES:
                for fdt in res['config']['grid']['dt_factors']:
                    k = f'{form}|{sc}|{DT0 * fdt:.8g}|prod'
                    L = [by[(k, j)] for j in range(n) if (k, j) in by]
                    if not L:
                        continue
                    e = np.array([x['e_ref'] for x in L])
                    an = [(x['anchor'], unc(j)) for j, x in enumerate(L) if x['verified'] and unc(j) is not None]
                    res_ = sum(a_ >= 3 * u_ for a_, u_ in an)
                    t_ = tim.get(k, {})
                    ent = dict(key=k, form=form, scheme=sc, f=fdt, worst=float(e.max()), median=float(np.median(e)),
                               anc=float(np.median([a_ for a_, _ in an])) if an else None, resolved=res_, n_anc=len(an),
                               verified=sum(x['verified'] for x in L), n=len(L), ratio=t_.get('ratio'), ms=t_.get('ms'))
                    tab.append(ent)
                    W(f"| {form} | {NAMES[sc]} | {fdt:g} | {pct(ent['worst'])} | {pct(ent['median'])} | {pct(ent['anc'])} | "
                      f"{res_}/{len(an)} | {ent['verified']}/{n} | {'—' if ent['ratio'] is None else f'{ent[chr(114)+chr(97)+chr(116)+chr(105)+chr(111)]:.3f}'} | "
                      f"{'—' if ent['ms'] is None else f'{ent[chr(109)+chr(115)]:.1f}'} |")
        W('')
        W(f'Observed order (thresholds for {n} cases: claim needs ≥ {need} valid cases; adjacent check unresolved below {half:g}):')
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
                if not ok:
                    claim = f'unavailable (audit failed; would read: {claim})'
                av = [p for p, v in adj if v]
                orders[f'{form}|{sc}'] = dict(claim=claim, primary=float(np.median(pv)) if pv else None, valid=len(pv),
                                              adjacent=float(np.median(av)) if av else None, both=len(both))
                o = orders[f'{form}|{sc}']
                W(f"| {form} | {NAMES[sc]} | {claim} | {'—' if o['primary'] is None else f'{o[chr(112)+chr(114)+chr(105)+chr(109)+chr(97)+chr(114)+chr(121)]:.2f}'} | "
                  f"{len(pv)}/{n} | {'—' if o['adjacent'] is None else f'{o[chr(97)+chr(100)+chr(106)+chr(97)+chr(99)+chr(101)+chr(110)+chr(116)]:.2f}'} | {len(both)} |")
        W('')
        cw, cm = float(ve.max()), float(np.median(ve))
        so2 = [e for e in tab if e['scheme'] != 'BE' and e['verified'] == n and e['ratio'] is not None]
        h1 = [e['key'] for e in so2 if abs(e['f'] - 1) < 1e-12 and e['median'] <= .7 * cm and e['worst'] <= cw]
        h2 = [e['key'] for e in so2 if e['f'] >= 2 and e['worst'] <= cw and e['median'] <= cm and e['ratio'] <= .75]
        W(f"- H1-3D (CN/BDF2-family at $\\Delta t_0$, median ≤ 0.7× and worst ≤ the deployed fixed-sweep BE): "
          f"**{'unavailable' if not ok else ('passed' if h1 else 'failed')}** ({', '.join(h1) or 'none'}).")
        W(f"- H2-3D (CN/BDF2-family at $\\Delta t\\ge2\\Delta t_0$, worst and median ≤ deployed, paired ratio ≤ 0.75): "
          f"**{'unavailable' if not ok else ('passed' if h2 else 'failed')}** ({', '.join(h2) or 'none'}).")
        W('')
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        for form, ls, mk in (('LSPG', '-', 'o'), ('GAL', '--', 's')):
            for sc in SCHEMES:
                E = sorted([e for e in tab if e['form'] == form and e['scheme'] == sc], key=lambda e: e['f'])
                ax[0].plot([DT0 * e['f'] for e in E], [100 * e['worst'] for e in E], ls=ls, marker=mk, color=COLORS[sc], label=f'{form} {NAMES[sc]}')
                ax[1].plot([DT0 * e['f'] for e in E], [100 * e['anc'] if e['anc'] else np.nan for e in E], ls=ls, marker=mk,
                           color=COLORS[sc], label=f"{form} {NAMES[sc]} ({orders[f'{form}|{sc}']['claim']})")
        ax[0].axhline(100 * cw, color='#52514e', ls=':', label='deployed fixed-sweep BE, $\\Delta t_0$')
        for a_ in ax:
            a_.set_xscale('log')
            a_.set_xlabel('time step $\\Delta t$')
            a_.grid(True, color='#e4e4e0', lw=.6)
            a_.spines[['top', 'right']].set_visible(False)
        ax[1].set_yscale('log')
        ax[0].set_ylabel('worst error vs refined reference (%), PROVISIONAL')
        ax[1].set_ylabel('median anchor discrepancy (%)')
        ax[0].set_title(f'3D {arm}: worst error vs $\\Delta t$', fontsize=10)
        ax[1].set_title(f'3D {arm}: anchor discrepancy (GAL-BDF2, $\\Delta t_0/16$)', fontsize=10)
        ax[0].legend(fontsize=6.5, frameon=False)
        ax[1].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(HERE / 'plots' / f'error_vs_dt_3d_{arm}.png', dpi=150)
        plt.close(fig)
        W(f'![error_vs_dt_3d_{arm}.png](plots/error_vs_dt_3d_{arm}.png)')
        W('')
        out[arm] = dict(table=tab, orders=orders, H1=h1, H2=h2, comparator=dict(worst=cw, median=cm, ms=A_ms))
    (HERE / 'checks' / f'analysis-{att}.json').write_text(json.dumps(out, indent=1) + '\n')
    return out
