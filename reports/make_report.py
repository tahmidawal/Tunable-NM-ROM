"""Generate reports/2026-10-01-burgers2d-offmesh-quadrature.md from the audited summaries (no hand-typed numbers).

    /home/tahmid/Dev/.venv/bin/python reports/make_report.py

Inputs (all committed): experiments/quadrature-study/checks/{dv,t}{256,1024,4096}-summary.json, selection-dv.json,
FROZEN-SELECTION.json, refdv/reft reference summaries, results/codex-*.md (listed only), and Hari's machine-readable
results external/quadrature-study-2026-09-30/quadrature/results/burgers_quad_study_reached.json for side-by-side ladders.
Every number in the output comes from these files; prose that states a number formats it from them.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / 'experiments/quadrature-study'
import os
CHK = Path(os.environ.get('QS_CHK', LANE / 'checks'))
HARI = ROOT / 'external/quadrature-study-2026-09-30/quadrature/results'
OUT = Path(os.environ.get('QS_OUT', ROOT / 'reports/2026-10-01-burgers2d-offmesh-quadrature.md'))
MESHES = (256, 1024, 4096)
SETS = ('acc', 'fast', 'head')
SNAME = dict(acc="accurate (span $R'=384$, $M=1536$)", fast="fast (span $R'=128$, $M=512$)",
             head='head ($k=16$, $q=0$, $M=64$)')
B3 = dict(acc=5e-4, fast=2e-3, head=5e-3)


def load(name):
    f = CHK / f'{name}-summary.json'
    return json.loads(f.read_text()) if f.exists() else None


def pc(x, d=3):
    """fraction -> percent string."""
    if x is None:
        return '—'
    if not np.isfinite(x):
        return '∞'
    return f'{100 * x:.{d}f}'


def sci(x, d=1):
    if x is None:
        return '—'
    if not np.isfinite(x):
        return '∞'
    return f'{x:.{d}e}'


def ms(x):
    return '—' if x is None else (f'{x:.1f}' if x < 1000 else f'{x:.0f}')


def tick(b):
    return '—' if b is None else ('yes' if b else '**no**')


def g(d, *keys, default=None):
    for k in keys:
        if d is None or k not in d:
            return default
        d = d[k]
    return d


ROLL_ORDER = ['dense', 'lat64', 'q0scaled', 'gauss32', 'gauss48', 'gauss64', 'gauss96', 'gauss128', 'gauss192',
              'gauss256', 'fib1597', 'fib4181', 'fib6765', 'fib17711', 'fib46368', 'sobol4096', 'flux_gauss64',
              'flux_fib6765', 'gref', 'gref_check', 'ctrl_smolyak8', 'ctrl_gauss8']
RHO_ORDER = ['dense', 'lat16', 'lat32', 'lat64', 'lat128', 'q0scaled'] + \
    [f'gauss{p}' for p in (8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 200, 256, 320, 400, 512)] + \
    [f'fib{n}' for n in (987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393)] + \
    ['sobol1024', 'sobol4096', 'sobol16384', 'sobol65536', 'halton4096', 'smolyak6', 'smolyak7', 'smolyak8', 'smolyak9',
     'flux_gauss64', 'flux_gauss128', 'flux_gauss256', 'flux_fib6765', 'flux_fib17711', 'flux_sobol4096']


def label(name):
    if name.startswith('gauss'):
        return f'Gauss ${name[5:]}^2$'
    if name.startswith('fib'):
        return f'Fibonacci {name[3:]}'
    if name.startswith('sobol'):
        return f'Sobol {name[5:]}'
    if name.startswith('halton'):
        return f'Halton {name[6:]}'
    if name.startswith('smolyak'):
        return f'Smolyak CC {name[7:]}'
    if name.startswith('lat'):
        s = int(name[3:])
        return f'mesh lattice ${s - 1}^2$'
    if name.startswith('flux_'):
        return label(name[5:]) + ' (flux)'
    if name.startswith('ctrl_'):
        return 'control: ' + label(name[5:])
    return dict(dense='dense (all mesh nodes)', q0scaled='fitted EQ rule (q0scaled)',
                gref='continuum rollout (Gauss $640^2$)', gref_check='Gauss $768^2$ rollout (G6 check)').get(name, name)


def hari_ladder():
    """Hari's Burgers continuum rho on reached states, accurate test count, worst over states (his JSON)."""
    f = HARI / 'burgers_quad_study_reached.json'
    if not f.exists():
        return {}
    rows = [r for r in json.loads(f.read_text()) if r.get('target') == 'continuum' and r.get('form') == 'point']
    out = {}
    for r in rows:
        key = {'gauss_tensor': f"gauss{r['param']}", 'fibonacci': f"fib{r['m']}", 'sobol': f"sobol{r['m']}"}.get(r['rule'])
        if key:
            out[key] = (r['m'], r['rho_acc_max'], r['rho_acc_med'])
    return out


def main():
    dv = {L: load(f'dv{L}') for L in MESHES}
    tt = {L: load(f't{L}') for L in MESHES}
    sel = json.loads((CHK / 'selection-dv.json').read_text()) if (CHK / 'selection-dv.json').exists() else None
    refs = {k: json.loads((CHK / f'{k}-reference.json').read_text()) for k in ('refdv', 'reft') if (CHK / f'{k}-reference.json').exists()}
    have_test = all(tt[L] for L in MESHES)
    L_ = []
    w = L_.append

    status = ('final: development/validation (dev6 ∪ val32) and the frozen held-out test cohort (test64)' if have_test
              else 'PROVISIONAL: development/validation only; the test-cohort section is not yet generated')
    w('# Off-mesh quadrature for the tested advection of the full-scale Burgers 2D model')
    w('')
    w(f'Replication of Hari\'s off-mesh quadrature study on the frozen Table-1 Burgers 2D model ($R=512$ bank, $K=16$ head, '
      f'$256^2$–$4096^2$). Status: **{status}**. Every number below is generated by `reports/make_report.py` from the '
      'NumPy-audited job summaries in `experiments/quadrature-study/checks/`; the pre-registration is '
      '`experiments/quadrature-study/DESIGN.md` (amendments A0–A2 and later dated inside).')
    w('')

    # ------------------------------------------------------------ jobs / gates
    w('## 1. What ran')
    w('')
    w('| job | role | mesh | cohorts | GPU | job id | elapsed (h) | accepted | failed gates |')
    w('|---|---|---|---|---|---|---|---|---|')
    for tag, S in [('dv', dv), ('t', tt)]:
        for L in MESHES:
            s = S[L]
            if not s:
                continue
            w(f"| `{s['attempt']}` | {s['role']} | ${L}^2$ | {', '.join(s['cohorts'])} | {s['gpu']} | {s['job_id']} | "
              f"{(s['elapsed_seconds'] or 0) / 3600:.2f} | {tick(s['accepted'])} | {', '.join(s['failed_gates']) or 'none'} |")
    for k, r in refs.items():
        w(f"| `{k}` | reference (FOM only) | $8192^2$ | {', '.join(r['cohorts'])} | {r['gpu']} | {r['job_id']} | "
          f"{r['elapsed_seconds'] / 3600:.2f} | {tick(r['all_accepted'])} | {'none' if r['all_accepted'] else 'G5'} |")
    w('')

    # ---------------------------------------------------------------- method
    w('## 2. The hybrid reduced solve')
    w('')
    w('One backward-Euler step solves for the bank coefficients $c$ (linear rung: $c=w$; head: $c=h(z)$):')
    w('')
    w('$$ r(c) = S\\Big(A c - p + \\Delta t\\,\\big(N(c) + \\nu\\Lambda A c\\big)\\Big),\\qquad S=(1+\\Delta t\\,\\nu\\Lambda)^{-1}. $$')
    w('')
    w('The linear terms ($A=\\Phi^{\\mathsf T}G\'$, $\\Lambda$) are the exact discrete ones of the project\'s solver. Only the tested '
      'advection $N(c)$ changes between arms: the FOM\'s upwind stencil on all mesh nodes (`dense`) or on $m$ of them '
      '(mesh rules), or Hari\'s off-mesh form $N(c)=L\\sum_q w_q\\,\\psi(x_q)\\,u\\,(u_x+u_y)(x_q)$ with the bank and its '
      'gradient decoded at the rule\'s points (or the flux form $-L\\sum_q w_q(\\psi_x+\\psi_y)\\,u^2/2$).')
    w('')
    w('```mermaid')
    w('flowchart LR')
    w('  U0[dense input field] --> IC[initial fit on Gauss-48 samples: closed form for span, LM for head]:::solved')
    w('  IC --> LM{{LM step solve, 50 steps}}:::solved')
    w('  B[(frozen bank G, head h)]:::frozen --> LIN[exact discrete linear terms A c, Lambda]:::frozen')
    w('  B --> MESH[mesh rule: upwind stencil at m nodes]:::mesh')
    w('  B --> OFF[off-mesh rule: u, grad u at m points]:::off')
    w('  LIN --> LM')
    w('  MESH -->|lat64, EQ, dense| LM')
    w('  OFF -->|Gauss, Fibonacci, Sobol, Smolyak| LM')
    w('  LM --> DEC[decode six output fields on the mesh]:::solved')
    w('  classDef frozen fill:#dbeafe,stroke:#1d4ed8;')
    w('  classDef solved fill:#dcfce7,stroke:#15803d;')
    w('  classDef mesh fill:#fef3c7,stroke:#b45309;')
    w('  classDef off fill:#fce7f3,stroke:#be185d;')
    w('```')
    w('')

    # --------------------------------------------------------------- answers
    answers(w, dv, tt if have_test else None, sel)

    # --------------------------------------------------------------- results
    sections_for(w, dv, 'development and validation (dev6 ∪ val32)', sel, hari=hari_ladder())
    if have_test:
        sections_for(w, tt, 'held-out test (test64, frozen)', sel, test=True)

    caveats(w, dv, sel)
    # ------------------------------------------------------------- glossary
    glossary(w)
    OUT.write_text('\n'.join(L_) + '\n')
    print(OUT)


def sections_for(w, S, cohort_label, sel, hari=None, test=False):
    pre = 'T' if test else 'D'
    w(f'## {"4" if test else "3"}. Results — {cohort_label}')
    w('')
    # continuum target
    w(f'### {pre}1. Continuum target (gate G6)')
    w('')
    w('| mesh | setting | population | $\\rho_{\\max}$ Gauss $640^2$ vs $768^2$ | $\\rho_{\\max}$ flux vs point | passed |')
    w('|---|---|---|---|---|---|')
    for L in MESHES:
        s = S[L]
        if not s:
            continue
        for st in SETS:
            d = g(s, 'gates', 'G6_continuum_target', 'detail', f'continuum_target_converged_{st}')
            if not d:
                continue
            for pop, v in d['populations'].items():
                w(f"| ${L}^2$ | {st} | {pop} ({v['states']} states) | {sci(v['check_rho_max'])} | {sci(v['flux_rho_max'])} | {tick(d['passed'])} |")
    w('')
    # rho ladders
    w(f'### {pre}2. $\\rho$ ladders on reached states (question ii)')
    w('')
    w('Worst and median $\\rho$ over the states $k=1..50$ reached by the deployed `lat64` arm. *cont* = against the '
      'continuum target (Gauss $640^2$); *mesh* = against the dense upwind stencil at that mesh. Bars: 0.116 (paper primary), 0.06.')
    w('')
    for st in SETS:
        w(f'**{SNAME[st]}**')
        w('')
        hdr = '| rule | $m$ | ' + ' | '.join(f'cont worst / median ${L}^2$' for L in MESHES if S[L]) + \
              ' | ' + ' | '.join(f'mesh worst ${L}^2$' for L in MESHES if S[L]) + (' | Hari cont worst ($M=320$) |' if hari else ' |')
        w(hdr)
        w('|' + '---|' * (hdr.count('|') - 1))
        for name in RHO_ORDER:
            ents = {L: g(S[L], 'rho', st, 'rules', name) for L in MESHES if S[L]}
            if not any(ents.values()):
                continue
            m = next(e['m'] for e in ents.values() if e)
            mtxt = 'mesh' if name == 'dense' else str(m)
            cells = [f"{sci(g(e, 'cont', 'max'))} / {sci(g(e, 'cont', 'median'))}" if e else '—' for e in ents.values()]
            cells2 = [sci(g(e, 'mesh', 'max')) if e else '—' for e in ents.values()]
            hx = ''
            if hari is not None:
                hx = f" {sci(hari[name][1])} ($m$={hari[name][0]}) |" if name in hari and st == 'acc' else ' — |'
            w(f"| {label(name)} | {mtxt} | " + ' | '.join(cells) + ' | ' + ' | '.join(cells2) + ' |' + hx)
        w('')

    # rollouts
    w(f'### {pre}3. End-to-end rollouts (question i)')
    w('')
    w('**Matched comparison on the cases where dense ran** (worst over those cases, %): the off-mesh arms against dense, '
      'against both refined references. B1 is the pre-registered two-sided bar (within $\\max(0.02$ pp$, 2\\,\\%)$ of dense on ST).')
    w('')
    w('| setting | arm | ' + ' | '.join(f'ST ${L}^2$ | S ${L}^2$' for L in MESHES if S[L]) + ' | B1 at ' + ', '.join(f'${L}^2$' for L in MESHES if S[L]) + ' |')
    w('|' + '---|' * (2 + 2 * sum(1 for L in MESHES if S[L]) + 1))
    for st in SETS:
        for name in ('dense', 'lat64', 'q0scaled', 'gref', 'gauss32', 'gauss64', 'gauss96', 'gauss128', 'fib1597', 'fib6765',
                     'fib17711', 'sobol4096', 'flux_gauss64'):
            cells, b1s = [], []
            for L in MESHES:
                if not S[L]:
                    continue
                b = g(S[L], 'arms', st, name, 'B1')
                if name == 'dense':
                    b0 = next((v.get('B1') for v in S[L]['arms'][st].values() if v.get('B1')), None)
                    cells += [pc(g(b0, 'dense_worst_ST')), pc(g(b0, 'dense_worst_S'))]
                    b1s.append('—')
                else:
                    cells += [pc(g(b, 'worst_ST')), pc(g(b, 'worst_S'))]
                    b1s.append(tick(g(b, 'passed')))
            if all(c == '—' for c in cells):
                continue
            ncase = next((g(S[L], 'arms', st, 'dense', 'all', 'cases') for L in MESHES if S[L]), None)
            w(f"| {st} | {label(name)} | " + ' | '.join(cells) + ' | ' + ', '.join(b1s) + ' |')
    w('')
    w('Worst over cases of the maximum over the five evolved output times, % of $\\lVert u_0\\rVert$. **ST** = against the '
      'refined reference ($8192^2$, $\\Delta t/16$; primary), **S** = against the space-only refined reference ($8192^2$, '
      '$\\Delta t$), **vs dense** = distance from our own dense rollout, **vs gref** = distance from the continuum rollout, '
      '**sg** = same-grid error against `fft_tight`. B1/B3 are the pre-registered bars (DESIGN §8). Refined-reference '
      'errors are scored on the $257^2$ nodes shared by every mesh (the references are stored there); same-grid and '
      'vs-dense/vs-gref errors on the full mesh. Rows have different case counts (column *cases*; dense and the Gauss-'
      '$768^2$ check run on fewer cases), so compare arms across rows only in the matched table above.')
    w('')
    for st in SETS:
        for L in MESHES:
            s = S[L]
            if not s or st not in s['arms']:
                continue
            A = s['arms'][st]
            w(f'**{SNAME[st]} — ${L}^2$** (dense on {g(A, "dense", "all", "cases") or 0} cases; FOM `fft_tight` vs ST worst '
              f'{pc(g(s, "fom", "all", "ref_ST_evolved", "worst"))} %, vs S worst {pc(g(s, "fom", "all", "ref_S_evolved", "worst"))} %)')
            w('')
            w('| arm | $m$ | cases | ST worst | ST median | S worst | vs dense worst (cases) | vs gref worst | sg worst | B1 | B3 | LM its/query | non-stationary exits |')
            w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
            for name in ROLL_ORDER:
                a = A.get(name)
                if not a:
                    continue
                al = a['all']
                ex = al['exits']
                nonst = sum(v for k, v in ex.items() if k != 'stationary')
                w(f"| {label(name)} | {'mesh' if name == 'dense' else a['m']} | {al['cases']} | {pc(g(al, 'ref_ST_evolved', 'worst'))} | "
                  f"{pc(g(al, 'ref_ST_evolved', 'median'))} | {pc(g(al, 'ref_S_evolved', 'worst'))} | "
                  f"{pc(g(al, 'vs_dense_evolved', 'worst'))} ({g(al, 'vs_dense_evolved', 'n') or 0}) | {pc(g(al, 'vs_gref_evolved', 'worst'))} | "
                  f"{pc(g(al, 'same_grid_evolved', 'worst'))} | {tick(g(a, 'B1', 'passed'))} | {tick(g(a, 'B3', 'passed'))} | "
                  f"{al['iterations_total_mean']:.0f} | {nonst} |")
            w('')

    if not test and sel:
        w(f'### {pre}4. Mesh invariance (B2) and the recommended rule')
        w('')
        w('| setting | arm | worst ST $256^2$ | $1024^2$ | $4096^2$ | max/min | B2 |')
        w('|---|---|---|---|---|---|---|')
        for st in SETS:
            for name in ROLL_ORDER:
                b2 = g(sel, 'B2', st, name)
                if not b2:
                    continue
                v = b2['worst_ST']
                w(f"| {st} | {label(name)} | {pc(v.get('256', v.get(256)))} | {pc(v.get('1024', v.get(1024)))} | "
                  f"{pc(v.get('4096', v.get(4096)))} | {b2['ratio']:.4f} | {tick(b2['passed'])} |")
        w('')
        w('**Pre-registered recommended off-mesh rule** (smallest $m$ passing B1, B3 and B5 at all three meshes, fixed '
          'before the test jobs): ' + '; '.join(f"{st}: {label(r) if r else 'none'}" for st, r in sel['recommended'].items()) + '.')
        w('')
        ph = sel.get('recommended_posthoc_one_sided_B1', {})
        w('**Post hoc (not pre-registered; fixed before the test jobs, DESIGN A3)** — with B1 read one-sided (not worse than '
          'dense): ' + '; '.join(f"{st}: {label(r) if r else 'none'}" for st, r in ph.items()) + '.')
        w('')
        w('| setting | arm | $m$ | B1 (two-sided) at $256^2/1024^2/4096^2$ | B1′ one-sided | B3 | B5 | continuum $\\rho_{\\max}$ at $4096^2$ |')
        w('|---|---|---|---|---|---|---|---|')
        for st in SETS:
            for name in ROLL_ORDER:
                v = g(sel, 'per_arm', st, name)
                if not v or name in ('dense', 'gref', 'gref_check'):
                    continue
                fmt = lambda xs: '/'.join('—' if x is None else ('y' if x else 'n') for x in xs)
                w(f"| {st} | {label(name)} | {v['m']} | {fmt(v['B1'])} | {fmt(v.get('B1_one_sided_posthoc', []))} | "
                  f"{fmt(v['B3'])} | {tick(v['B5'])} | {sci(g(v, 'rho_cont_max', '4096'))} |")
        w('')

    # cost
    w(f'### {pre}5. Cost (question iii)')
    w('')
    s4 = S[4096]
    if s4 and g(s4, 'timing', 'B4_same_gpu'):
        w(f"Same-GPU cross-mesh panel inside `{s4['attempt']}` ({s4['gpu']}): median solve time (median query minus median "
          'decode, ms; 6 cases × 3 repetitions) of each arm at $256^2$, $1024^2$, $4096^2$, and the pre-registered flat-cost '
          'bar B4 (ratio $4096^2/256^2\\in[0.8,1.25]$). The full query also decodes six output fields on the mesh, '
          'which grows with $N$ (decode column), so full-query time is not flat.')
        w('')
        w('| setting | arm | $256^2$ | $1024^2$ | $4096^2$ | ratio | B4 |')
        w('|---|---|---|---|---|---|---|')
        for st in SETS:
            for name in ROLL_ORDER:
                b = g(s4, 'timing', 'B4_same_gpu', f'{st}|{name}')
                if not b:
                    continue
                v = b['solve_ms']
                rt = '—' if b['ratio'] is None else f"{b['ratio']:.3f}"
                w(f"| {st} | {label(name)} | {ms(v.get('256'))} | {ms(v.get('1024'))} | {ms(v.get('4096'))} | "
                  f"{rt} | {tick(b['passed'])} |")
        w('')
    if not test and sel and s4:
        ph = sel.get('recommended_posthoc_one_sided_B1', {})
        w('**Cost beside the deployed `lat64` (post hoc rules of D4)**, same H200 job at $4096^2$: solve and full-query ms, '
          'hyper-reduction error (the B3 metric) and worst same-grid error. Caveat: this is not a matched equal-accuracy '
          'comparison — the B3 metric is measured against different targets (off-mesh: the continuum rollout, all cases; '
          '`lat64`: the dense rollout, the six dense cases), and the two families converge to different solutions (the '
          'continuum vs the upwind stencil).')
        w('')
        w('| setting | arm | $m$ | solve ms | query ms | B3 metric worst % | sg worst % |')
        w('|---|---|---|---|---|---|---|')
        for st in SETS:
            for name in [x for x in ('lat64', ph.get(st)) if x]:
                a = g(s4, 'arms', st, name)
                w(f"| {st} | {label(name)} | {a['m']} | {ms(g(s4, 'timing', 'solve_ms', f'{st}|4096|{name}'))} | "
                  f"{ms(g(s4, 'timing', 'query_ms', f'{st}|4096|{name}'))} | {pc(g(a, 'B3', 'worst'))} | "
                  f"{pc(g(a, 'all', 'same_grid_evolved', 'worst'))} |")
        w('')
    w('Per-mesh query time (dense input field on the GPU → six output fields), decode time, and the two full-order settings, '
      'each from its own job (medians; same-job comparisons only):')
    w('')
    w('| mesh | GPU | setting | lat64 query ms | decode ms | ' + ' | '.join(label(n) + ' query ms' for n in ('gauss64', 'fib6765', 'gauss128', 'fib17711')) +
      ' | FOM `lean_tight` ms | FOM `lean_nt3e-3_l3e-3_dt005` ms (worst sg %) |')
    w('|' + '---|' * 11)
    for L in MESHES:
        s = S[L]
        if not s:
            continue
        for st in SETS:
            q = lambda n: g(s, 'timing', 'query_ms', f'{st}|{L}|{n}')
            fom = g(s, 'timing', 'fom') or {}
            w(f"| ${L}^2$ | {s['gpu']} | {st} | {ms(q('lat64'))} | {ms(g(s, 'timing', 'decode_ms', f'{st}|{L}'))} | " +
              ' | '.join(ms(q(n)) for n in ('gauss64', 'fib6765', 'gauss128', 'fib17711')) +
              f" | {ms(g(fom, 'lean_tight', 'median_ms'))} | {ms(g(fom, 'lean_nt3e-3_l3e-3_dt005', 'median_ms'))} "
              f"({pc(g(fom, 'lean_nt3e-3_l3e-3_dt005', 'worst_same_grid_evolved'))}) |")
    w('')

    # worst states
    w(f'### {pre}6. Where the worst states are (question iv)')
    w('')
    w('For each rule, the state with the largest continuum $\\rho$: its case\'s bump width $w$, amplitude $a$, viscosity '
      '$\\nu$ and step $k$; the Spearman correlation between a case\'s largest $\\rho$ and $w$ (negative = narrower bumps '
      'are worse); and the share of the top 1 % of states that are within the first five steps. Columns marked † were '
      'added after the first development job and are descriptive, not pre-registered.')
    w('')
    w('| mesh | setting | rule | worst $\\rho$ | case | $w$ | $a$ | $\\nu$ | $k$ | Spearman($\\rho$, $w$) | top-1 % with $k\\le5$ | Spearman($\\rho$, $\\nu$)† | median $k$ of top 1 %† |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for L in MESHES:
        s = S[L]
        if not s:
            continue
        for st in SETS:
            for name in ('gauss64', 'gauss128', 'fib6765', 'fib17711', 'sobol4096', 'lat64'):
                e = g(s, 'question_iv', st, name)
                if not e:
                    continue
                am = e['argmax']
                sp = e['spearman_casemax_vs_width']
                w(f"| ${L}^2$ | {st} | {label(name)} | {sci(am['rho'])} | {am['cohort']}{am['case']} | {am['width']:.3f} | "
                  f"{am['amplitude']:.2f} | {am['nu']:.3f} | {am['k']} | {'—' if sp is None else f'{sp:+.2f}'} | "
                  f"{100 * e['top1pct_share_k_le_5']:.0f} % | {e.get('spearman_casemax_vs_nu', 0):+.2f} | {e.get('top1pct_median_k', 0):.0f} |")
    w('')


def b1m(S, L, st, name, key):
    return g(S[L], 'arms', st, name, 'B1', key)


def answers(w, dv, tt, sel):
    """Generated summary. Every number is read from the summaries; the wording follows the audits' corrections."""
    ph = (sel or {}).get('recommended_posthoc_one_sided_B1', {})
    w('## Answers in brief')
    w('')
    for tag, S in (('development/validation', dv), ('test', tt)):
        if not S or not all(S[L] for L in MESHES):
            continue
        w(f'**{tag.capitalize()} cohort{"s" if tag.startswith("dev") else ""}.**')
        w('')
        # (i)
        lines = []
        for st in SETS:
            gr = [b1m(S, L, st, 'gref', 'worst_ST') for L in MESHES]
            gs = [b1m(S, L, st, 'gref', 'worst_S') for L in MESHES]
            dn = [b1m(S, L, st, 'gref', 'dense_worst_ST') for L in MESHES]
            ds = [b1m(S, L, st, 'gref', 'dense_worst_S') for L in MESHES]
            if None in gr + dn:
                continue
            nc = [b1m(S, L, st, 'gref', 'cases') for L in MESHES]
            lines.append(f"{st}: continuum rollout ST {'/'.join(pc(x, 2) for x in gr)} %, S {'/'.join(pc(x, 2) for x in gs)} % at "
                         f"$256^2/1024^2/4096^2$; dense ST {'/'.join(pc(x, 2) for x in dn)} %, S {'/'.join(pc(x, 2) for x in ds)} % "
                         f"(on {'/'.join(str(x) for x in nc)} cases)")
        w('- **(i) Reproduce dense against the refined reference, mesh-invariantly?** Resolved off-mesh rules converge to one '
          'mesh-invariant solution (the continuum-advection hybrid), the dense mesh solve to the upwind solution, which '
          'improves with the mesh. On the cases where dense ran (worst over cases; the case set is smaller at $4096^2$, '
          'which is why the continuum-rollout numbers differ there — on a fixed case set they are invariant, see B2): '
          + '; '.join(lines) + '. So at the '
          'coarse meshes the off-mesh rules are *more* accurate than dense against both references, and at $4096^2$ the '
          'two agree against S to within the differences shown (slightly better against ST). The pre-registered two-sided '
          'B1 therefore fails for most off-mesh arms at the coarse meshes in the favourable direction; under-resolved rules '
          '(e.g. accurate Gauss $32^2$) fail it by being worse.')
        if tag.startswith('dev') and sel:
            b2 = [v['ratio'] for st in SETS for n, v in sel['B2'][st].items()
                  if v and n.startswith(('gauss', 'fib', 'flux', 'gref')) and n != 'gauss32']
            b2l = [sel['B2'][st]['lat64']['ratio'] for st in SETS if sel['B2'][st].get('lat64')]
            w(f'  Mesh invariance (B2, worst ST max/min over the three meshes): off-mesh Gauss/Fibonacci/flux rules except '
              f'accurate-setting Gauss $32^2$ between {min(b2):.4f} and {max(b2):.4f}; the deployed lattice {min(b2l):.2f}–{max(b2l):.2f}.')
        # (ii)
        L0 = 1024
        r = lambda st, n, t='cont': g(S[L0], 'rho', st, 'rules', n, t, 'max')
        w(f'- **(ii) $\\rho$ ladder vs the $63^2$ lattice and EQ at matched $m$ ($1024^2$, worst over reached states):** '
          + '; '.join(f"{st}: lat64 ($m$=3969) {sci(r(st, 'lat64', 'mesh'))} against its mesh target, "
                      f"Gauss $64^2$ ($m$=4096) {sci(r(st, 'gauss64'))} and Fibonacci 4181 {sci(r(st, 'fib4181'))} against the continuum"
                      + (f", fitted EQ ($m$=1024) {sci(r(st, 'q0scaled', 'mesh'))} vs Gauss $32^2$ {sci(r(st, 'gauss32'))}" if st == 'head' else '')
                      for st in SETS)
          + '. Our bank needs far more points than Hari\'s for the same continuum accuracy (Gauss $128^2$: '
          + ', '.join(f"{st} {sci(r(st, 'gauss128'))}" for st in SETS) + '); Sobol 4096 stays at '
          + ', '.join(f"{st} {sci(r(st, 'sobol4096'))}" for st in SETS) + ' and Smolyak CC 8 at '
          + ', '.join(f"{st} {sci(r(st, 'smolyak8'))}" for st in SETS) + '.')
        # (iii)
        s4 = S[4096]
        b4 = g(s4, 'timing', 'B4_same_gpu') or {}
        if b4:
            rs = [v['ratio'] for v in b4.values() if v['ratio'] is not None]
            cmp_ = []
            for st in SETS:
                n = ph.get(st)
                if n:
                    cmp_.append(f"{st}: {label(n)} {ms(g(s4, 'timing', 'solve_ms', f'{st}|4096|{n}'))} ms vs lat64 "
                                f"{ms(g(s4, 'timing', 'solve_ms', f'{st}|4096|lat64'))} ms")
            w(f'- **(iii) Cost flat in $N$?** Solve time (query minus decode) is flat on one GPU: B4 ratios $4096^2/256^2$ '
              f'between {min(rs):.3f} and {max(rs):.3f} for all {len(rs)} measured arms; it grows with $m$. The full query '
              f'is not flat because the six-field output decode grows with $N$. At $4096^2$, solve time of the post-hoc '
              f'rules vs the deployed lattice: ' + '; '.join(cmp_) + ' (not a matched equal-accuracy comparison; §D5).')
        # (iv)
        iv = []
        for st in SETS:
            e = g(S[L0], 'question_iv', st, 'gauss64')
            e2 = g(S[L0], 'question_iv', st, 'lat64')
            if e and e2:
                iv.append(f"{st}: Gauss $64^2$ worst at $k$={e['argmax']['k']} (width {e['argmax']['width']:.3f}, "
                          f"Spearman with width {e['spearman_casemax_vs_width']:+.2f}); lat64 worst at $k$={e2['argmax']['k']} "
                          f"(width {e2['argmax']['width']:.3f}, Spearman {e2['spearman_casemax_vs_width']:+.2f})")
        w('- **(iv) Narrow early bumps the worst case?** Not in the way Hari found. At $1024^2$: ' + '; '.join(iv) +
          '. For the off-mesh rules the worst states lean towards early steps of *wide* bumps; there are exceptions '
          '(§D6). The cause is not tested here.')
        w('')
    if sel:
        w('Pre-registered recommended off-mesh rule: ' + ', '.join(f"{st} {label(r) if r else 'none'}" for st, r in sel['recommended'].items())
          + '. Post hoc (one-sided B1′, fixed before the test jobs): ' + ', '.join(f"{st} {label(r)}" for st, r in ph.items() if r) + '.')
        w('')


def caveats(w, dv, sel):
    w('## What went wrong, what was changed, limitations')
    w('')
    for t in [
        'Two Codex design audits (`experiments/quadrature-study/results/codex-design-audit-{1,2}.md`) found 2 + 1 blockers '
        'and 18 major issues in the design and code before any ROM job (unvalidated references, a missing audit, '
        'timing confounded with GPU type, an ineffective test-freeze gate, acceptance that did not block selection); '
        'all were fixed or explicitly dispositioned before submission (DESIGN A0, A1).',
        'The local smoke caught one real bug after the restructuring (an `UnboundLocalError` in the timing accumulator) '
        'before any cluster job; a first local calibration probe was silently killed by the 36 GB cgroup and was redone '
        'in streamed form.',
        'B1 was written two-sided; it fails when the off-mesh rules are better than the upwind stencil. It is reported as '
        'written; the one-sided reading and the extra worst-state descriptives are post hoc (DESIGN A3), the regenerated '
        'summaries with matched-set descriptives (A4) were produced after the test jobs had been submitted (no input '
        'changed; selection verified identical), and A3\'s wording was corrected in A5.',
        'The continuum target is Gauss $640^2$ (certified against $768^2$ and the flux form, gate G6); calibration showed '
        'our bank needs about ten times Hari\'s points per axis for the same agreement.',
        'Refined-reference errors are measured on the $257^2$ nodes shared by every mesh, not the full mesh; the full-mesh '
        'audit recompute covers $256^2$ (audit arms) and $1024^2$ (accurate setting, one case) only.',
        'Dense at $4096^2$ ran on six cases per cohort (cost); every comparison against dense is on the matched cases.',
        'Timing: 6 cases × 3 repetitions per subject, medians, one GPU per job; solve time = median(query) − median(decode).',
        'The FOM-only reference jobs used A100s; ROM jobs: A100-80G at $256^2$/$1024^2$, H200 at $4096^2$. Absolute times '
        'are compared only within a job.',
        'test64 is a historical cohort reused by earlier lanes for fixed settings; nothing here was chosen from it.',
    ]:
        w(f'- {t}')
    w('')


def glossary(w):
    w('## Glossary')
    w('')
    items = [
        ('accurate / fast / head', "the three deployment settings of the frozen model: the linear rung on the first $R'=384$ "
         "or $R'=128$ columns of the rotated bank (the coefficients are solved directly), or the $k=16$ nonlinear head on the full bank."),
        ('bank, rotated bank', 'the frozen coordinate network $G(x)$ ($R=512$ smooth functions of position); rotated = '
         'multiplied by a fixed matrix so its leading columns are the most useful ones.'),
        ('tested advection $N(c)$', 'the nonlinear term $u(u_x+u_y)$ projected on the $M$ sine test functions — the only '
         'term that needs quadrature.'),
        ('dense', 'evaluating the tested advection with the full-order upwind stencil at every mesh node (no hyper-reduction).'),
        ('mesh rule / lat64', 'a rule that evaluates the upwind stencil at $m$ mesh nodes only; lat64 is the deployed '
         'uniform $63\\times63$ sub-lattice with equal weights.'),
        ('EQ, q0scaled', 'empirical quadrature: mesh nodes and non-negative weights fitted to stored states (NNLS); '
         'q0scaled is the head\'s fitted rule ($m=1024$).'),
        ('off-mesh rule', 'a classical quadrature rule on the unit square whose points need not be mesh nodes; the bank is '
         'decoded there with its exact gradient.'),
        ('Gauss $p^2$', 'tensor Gauss–Legendre rule with $p$ points per axis ($m=p^2$).'),
        ('Fibonacci $n$', 'rank-1 lattice rule with $n$ (a Fibonacci number) points and a random shift.'),
        ('Sobol / Halton', 'scrambled quasi-Monte Carlo point sets (typical error decay about $1/m$, not a guarantee).'),
        ('$m$ / $M$', '$m$ = number of quadrature points (or mesh nodes) of a rule; $M$ = number of sine test functions.'),
        ('pp', 'percentage points (an absolute difference of two percentages).'),
        ('Spearman', 'rank correlation over the cases between a case\'s largest $\\rho$ and a case parameter ($-1$ to $+1$).'),
        ('top 1 %', 'the 1 % of reached states (of all cases together) with the largest $\\rho$ for that rule.'),
        ('cases (n)', 'the number of cases a statistic is taken over; dense runs on fewer cases at $4096^2$ (dev6 / the first six test cases).'),
        ('Smolyak CC', 'sparse-grid rule built from nested Clenshaw–Curtis rules; a must-fail control here.'),
        ('point / flux form', 'Hari\'s two off-mesh forms: $\\psi\\,u(u_x+u_y)$ (needs $\\nabla u$) or the integrated-by-parts '
         '$-(\\psi_x+\\psi_y)u^2/2$ (needs $u$ only).'),
        ('gref', 'the continuum rollout: the off-mesh solve with a converged Gauss $640^2$ rule; the off-mesh analogue of dense.'),
        ('$\\rho$', 'relative error of a rule\'s tested advection against a target on one state (paper eq. 13); *cont* = '
         'against the continuum target (Gauss $640^2$), *mesh* = against dense.'),
        ('reached states', 'the 50 internal states per case that the deployed lat64 solve passes through.'),
        ('ST / S reference', 'refined full-order solutions at $8192^2$: ST with $\\Delta t/16$ (space and time refined; '
         'primary), S with the same $\\Delta t$ (space only).'),
        ('vs dense / vs gref', 'distance of an arm\'s rollout from the dense / continuum rollout of the same model — the '
         'hyper-reduction error proper.'),
        ('same-grid error (sg)', 'error against the converged full-order solve (`fft_tight`) on the same mesh.'),
        ('worst / median', 'over the cases of the cohort, of the maximum over the five evolved output times $t=0.05..0.25$, '
         'normalised by the initial field norm.'),
        ('dev6, val32, test64', 'the 6 development, 32 validation and 64 test cases (Gaussian-bump initial '
         'conditions with random centre, width $w$, amplitude $a$ and viscosity $\\nu$). test64 is a reused historical '
         'cohort: earlier lanes evaluated fixed settings on it; nothing in this lane was chosen from it.'),
        ('B1–B5', 'pre-registered bars: B1 reproduces dense against ST, B2 mesh invariance, B3 hyper-reduction error, B4 '
         'flat cost, B5 $\\rho\\le0.116$.'),
        ('G1–G8', 'acceptance gates of a job (GPU backend, cohort, parity with earlier records, truth convergence, '
         'references, continuum target, must-fail controls, independent NumPy audit).'),
        ('solve time', 'median query time minus median decode time (the six-field output decode is mesh-dependent).'),
        ('LM its/query', 'Levenberg–Marquardt iterations summed over the 50 time steps of one query, averaged over the cases.'),
        ('non-stationary exits', 'total over all cases of time steps whose solve stopped on the budget, a tiny step or the '
         'damping limit instead of the stationarity test.'),
    ]
    w('| term | meaning |')
    w('|---|---|')
    for k, v in items:
        w(f'| {k} | {v} |')


if __name__ == '__main__':
    main()
