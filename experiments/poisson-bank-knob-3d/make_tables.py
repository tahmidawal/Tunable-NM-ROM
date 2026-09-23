"""Generate every number of this lane from the collected, audited result JSONs.

    python make_tables.py                       -> reports/summary.json, reports/tables.generated.md
    python make_tables.py --freeze <attempt>    -> frozen-N<n>.json (cube development -> final settings)

Rule (DESIGN.md, fixed before any job):
  accurate = lowest worst same-grid error of any ROM arm (ties within 1e-9 relative -> cheaper);
  fast     = cheapest ROM arm with worst error <= the worst error of the paper's fast setting (orig_q0);
  Table-1 FOM = fastest CG with worst error <= the accurate arm's worst error; speedup = FOM / arm;
  every arm is also compared with the fastest CG at least as accurate as itself.
Time scope: L-shape complete query (total_seconds), cube GPU query (fused_device_seconds).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
ROM = ('nm-rom', 'linear-rung', 'nm-rom-parent', 'nm-rom-parent-engine')
SCOPE = dict(lshape='total_seconds', cube='fused_device_seconds')
OTHER = dict(total_seconds='fused_device_seconds', fused_device_seconds='total_seconds')
TIE = 1e-9

# attempt -> role; filled as jobs are collected (only audited, all-gates-passed runs are used)
ATTEMPTS = json.loads((HERE / 'attempts.json').read_text()) if (HERE / 'attempts.json').exists() else {}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(attempt):
    base = RUNS / attempt / 'archive' / 'output'
    R = json.loads((base / 'result.json').read_text())
    A = json.loads((base / 'audit.json').read_text())
    return R, A, base


def subjects(R):
    """name -> summary over cases (main phase for ROM + fast CG, slow phase for slow CG)."""
    scope = SCOPE[R['problem']]
    arms = {a['name']: a for a in R['arms']}
    by = {}
    for x in R['invocations'] + R['slow_invocations']:
        by.setdefault(x['name'], []).append(x)
    out = {}
    for name, rows in by.items():
        errs = {}
        for x in rows:
            errs.setdefault(x['case'], []).append(x['same_grid_error'])
        worst_case = [max(v) for v in errs.values()]
        fam = rows[0]['family']
        s = dict(name=name, family=fam, Rp=rows[0]['Rp'], q=rows[0]['q'], tolerance=rows[0]['tolerance'],
                 cases=len(errs), invocations=len(rows),
                 worst_pct=100 * float(max(worst_case)), median_pct=100 * float(np.median(worst_case)),
                 ms=1000 * float(np.median([x[scope] for x in rows])),
                 ms_other=1000 * float(np.median([x[OTHER[scope]] for x in rows])),
                 phase=rows[0]['phase'])
        if fam == 'cg':
            s['converged_all'] = all(x['cg_converged'] for x in rows)
            s['iterations_median'] = float(np.median([x['iterations'] for x in rows]))
        else:
            s['stationary'] = R['stationarity_by_arm'].get(name)
            for k in ('qr_condition_number', 'correction_condition_number'):
                if k in arms.get(name, {}):
                    s[k] = arms[name][k]
        out[name] = s
    return out


def fastest_cg(S, bar_pct, key='ms'):
    ok = [s for s in S.values() if s['family'] == 'cg' and s['worst_pct'] <= bar_pct * (1 + TIE)]
    return min(ok, key=lambda s: s[key]) if ok else None


def select(S, R):
    roms = [s for s in S.values() if s['family'] in ROM]
    best = min(s['worst_pct'] for s in roms)
    acc = min((s for s in roms if s['worst_pct'] <= best * (1 + TIE)), key=lambda s: s['ms'])
    bar = S['orig_q0']['worst_pct']
    fast = min((s for s in roms if s['worst_pct'] <= bar * (1 + TIE)), key=lambda s: s['ms'])
    return acc, fast, bar


def row_for(S, acc, fast):
    fom = fastest_cg(S, acc['worst_pct'])
    fom_o = fastest_cg(S, acc['worst_pct'], 'ms_other')
    r = dict(accurate=acc['name'], fast=fast['name'], accurate_worst_pct=acc['worst_pct'],
             fast_worst_pct=fast['worst_pct'], accurate_ms=acc['ms'], fast_ms=fast['ms'],
             fom=fom['name'] if fom else None, fom_worst_pct=fom['worst_pct'] if fom else None,
             fom_ms=fom['ms'] if fom else None,
             accurate_speedup=fom['ms'] / acc['ms'] if fom else None,
             fast_speedup=fom['ms'] / fast['ms'] if fom else None,
             other_scope=dict(fom=fom_o['name'] if fom_o else None,
                              accurate_speedup=fom_o['ms_other'] / acc['ms_other'] if fom_o else None,
                              fast_speedup=fom_o['ms_other'] / fast['ms_other'] if fom_o else None))
    return r


def per_arm(S):
    out = []
    for s in sorted((s for s in S.values() if s['family'] in ROM), key=lambda s: (-(s['Rp'] or 0), s['q'] or 0, s['name'])):
        own = fastest_cg(S, s['worst_pct'])
        named = S.get('cg_0.01')
        out.append(dict(**s, own_fom=own['name'] if own else None,
                        own_speedup=own['ms'] / s['ms'] if own else None,
                        speedup_vs_cg_1e2=named['ms'] / s['ms'] if named else None))
    return out


def profile(R):
    by = {}
    for x in R['profile']:
        by.setdefault(x['name'], []).append(x)
    out = {}
    for name, rows in by.items():
        st = {k: 1000 * float(np.median([x[k] for x in rows]))
              for k in ('project_and_start', 'lm_solve', 'elimination_and_map', 'reconstruction')}
        st['sum'] = sum(st.values())
        st['reconstruction_share'] = st['reconstruction'] / st['sum']
        out[name] = st
    return out


def fmt(x, d=3):
    return '' if x is None else (f'{x:.{d}f}' if abs(x) >= 0.01 else f'{x:.2e}')


def freeze(attempt):
    R, A, _ = load(attempt)
    assert R['problem'] == 'cube' and R['cohort']['role'] == 'development' and A['verdict'] == 'PASS'
    assert all(R['gates'].values()), R['gates']
    S = subjects(R)
    acc, fast, bar = select(S, R)
    fz = dict(intervals=R['intervals'], selected_on='development', source_attempt=attempt, source_job=R['job_id'],
              source_result_sha256=sha_file(RUNS / attempt / 'archive' / 'output' / 'result.json'),
              accurate=acc['name'], fast=fast['name'], paper_fast_bar_pct=bar,
              development_accurate_worst_pct=acc['worst_pct'], development_fast_worst_pct=fast['worst_pct'],
              rule='DESIGN.md: accurate = lowest worst error; fast = cheapest arm with worst <= orig_q0 worst')
    p = HERE / f"frozen-N{R['intervals']}.json"
    p.write_text(json.dumps(fz, indent=1) + '\n')
    print(p, fz)


def main():
    summary = dict(generated_by='make_tables.py', meshes=[])
    md = ['# poisson-bank-knob-3d — generated tables', '',
          'Generated by `make_tables.py` from the audited result JSONs; do not edit by hand. Errors: worst / median '
          'same-grid relative L2 over the cohort (%). Time: median over all retained repetitions x cases, in the '
          'series scope (L-shape: complete query; cube: GPU query). `own FOM` = fastest tested CG at least as accurate '
          'as the arm; `S own` = its time / arm time; `S cg1e-2` = named CG rtol 1e-2 / arm.', '']
    for attempt, role in ATTEMPTS.items():
        R, A, base = load(attempt)
        S = subjects(R)
        prob, n = R['problem'], R['intervals']
        entry = dict(attempt=attempt, role=role, problem=prob, intervals=n, job_id=R['job_id'], gpu=R['gpu'],
                     commit=R['commit'], cohort=R['cohort'].get('role', 'development'),
                     cases=len(R['cohort']['parameters']), scope=SCOPE[prob],
                     result_sha256=sha_file(base / 'result.json'), audit_sha256=sha_file(base / 'audit.json'),
                     audit=A['verdict'], audit_summary=A['summary'], gates=R['gates'],
                     neighbour_max_ratio=max(r['ratio'] for r in R['neighbour_gate']['rows']
                                             if r['scope'] == R['neighbour_gate']['gate_scope']
                                             and r.get('variant', 'after_cg') == R['neighbour_gate'].get('gate_variant', 'after_cg')),
                     parity=R['parity'], floors={k: 100 * v['worst'] for k, v in R['floors'].items()},
                     not_constructible=R.get('not_constructible', []))
        usable = A['verdict'] == 'PASS' and all(R['gates'].values())
        entry['usable'] = usable
        if R['cohort'].get('role') == 'final':
            fz = R['frozen_settings']
            acc, fast = S[fz['accurate']], S[fz['fast']]
            entry['selection'] = 'frozen on development (' + fz['source_attempt'] + ')'
            bar = S['orig_q0']['worst_pct']
            racc, rfast, _ = select(S, R)
            entry['rule_on_final_descriptive_only'] = dict(accurate=racc['name'], fast=rfast['name'])
        else:
            acc, fast, bar = select(S, R)
            entry['selection'] = 'rule on this cohort'
        entry['paper_fast_bar_pct'] = bar
        entry['row'] = row_for(S, acc, fast)
        entry['arms'] = per_arm(S)
        entry['cg'] = sorted((s for s in S.values() if s['family'] == 'cg'), key=lambda s: -s['tolerance'])
        entry['profile'] = profile(R)
        summary['meshes'].append(entry)
        # ---- markdown
        unit = f'{n}²' if prob == 'lshape' else f'{n}³'
        par = '; '.join(f"{p['candidate']} vs {p['baseline']} {p['worst_field_relative']:.1e}" for p in R['parity'])
        md += [f"## {'L-shape' if prob == 'lshape' else 'Poisson 3D'} {unit} — {attempt} ({role})", '',
               f"Job {R['job_id']}, {R['gpu']}, commit `{(R['commit'] or '')[:10]}`, cohort {entry['cohort']} "
               f"({entry['cases']} cases), scope {SCOPE[prob]}. Audit: **{A['verdict']}** — {A['summary']}. "
               f"Gates: {', '.join(k + ('=ok' if v else '=FAIL') for k, v in R['gates'].items())}; "
               f"neighbour max ratio {entry['neighbour_max_ratio']:.3f}. "
               f"Parity: {par}.", '']
        r = entry['row']
        md += [f"**Row ({entry['selection']}).** accurate `{r['accurate']}` {fmt(r['accurate_worst_pct'])} % @ "
               f"{fmt(r['accurate_ms'])} ms; fast `{r['fast']}` {fmt(r['fast_worst_pct'])} % @ {fmt(r['fast_ms'])} ms "
               f"(bar = orig_q0 {fmt(bar)} %). FOM `{r['fom']}` {fmt(r['fom_worst_pct'])} % @ {fmt(r['fom_ms'])} ms "
               f"→ speedups accurate **{fmt(r['accurate_speedup'], 2)}×**, fast **{fmt(r['fast_speedup'], 2)}×** "
               f"(other scope: {fmt(r['other_scope']['accurate_speedup'], 2)}× / {fmt(r['other_scope']['fast_speedup'], 2)}× "
               f"vs `{r['other_scope']['fom']}`).", '']
        if 'rule_on_final_descriptive_only' in entry:
            md += [f"Rule re-applied on the final cohort (descriptive only, not used): accurate "
                   f"`{entry['rule_on_final_descriptive_only']['accurate']}`, fast `{entry['rule_on_final_descriptive_only']['fast']}`.", '']
        md += ["| arm | R' | q | worst % | median % | floor(R') % | ms | other-scope ms | stationary | own FOM | S own | S cg1e-2 |",
               '|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|']
        for s in entry['arms']:
            st = s.get('stationary')
            md.append(f"| {s['name']} | {s['Rp']} | {s['q']} | {fmt(s['worst_pct'])} | {fmt(s['median_pct'])} | "
                      f"{fmt(entry['floors'].get(str(s['Rp'])))} | {fmt(s['ms'])} | {fmt(s['ms_other'])} | "
                      f"{'' if st is None else str(st['stationary']) + '/' + str(st['invocations'])} | "
                      f"{s['own_fom'] or '—'} | {fmt(s['own_speedup'], 2)} | {fmt(s['speedup_vs_cg_1e2'], 2)} |")
        md += ['', '| CG | worst % | median % | ms | other-scope ms | iterations (median) | converged |',
               '|---|---:|---:|---:|---:|---:|---|']
        for s in entry['cg']:
            md.append(f"| {s['name']} | {fmt(s['worst_pct'])} | {fmt(s['median_pct'])} | {fmt(s['ms'])} | "
                      f"{fmt(s['ms_other'])} | {s['iterations_median']:.0f} | {s['converged_all']} |")
        if entry['not_constructible']:
            md += ['', 'Not constructible: ' + '; '.join(f"{x['name']} ({x['reason']})" for x in entry['not_constructible'])]
        md += ['', '**Stage profile (ms, medians; separately jitted and synchronised stages).**', '',
               '| arm | project + start | LM | y elim + map | reconstruction | sum | reconstruction share |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for name, st in sorted(entry['profile'].items()):
            md.append(f"| {name} | {fmt(st['project_and_start'])} | {fmt(st['lm_solve'])} | "
                      f"{fmt(st['elimination_and_map'])} | {fmt(st['reconstruction'])} | {fmt(st['sum'])} | "
                      f"{st['reconstruction_share']:.2f} |")
        md.append('')
    (HERE / 'reports').mkdir(exist_ok=True)
    (HERE / 'reports' / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (HERE / 'reports' / 'tables.generated.md').write_text('\n'.join(md) + '\n')
    print('wrote reports/summary.json', sha_file(HERE / 'reports' / 'summary.json'))


if __name__ == '__main__':
    if '--freeze' in sys.argv:
        freeze(sys.argv[sys.argv.index('--freeze') + 1])
    else:
        main()
