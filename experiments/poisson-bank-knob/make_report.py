"""Generate summary.json + tables.generated.md from the collected cluster runs (no hand-typed numbers).

Inputs: runs/<attempt>/archive/<output|output2>/{result.json,audit.json}.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = [('pbkH', 'output'), ('pbkH', 'output2'), ('pbkH', 'output3'), ('pbkI', 'output')]
STAGES = ('project_and_start', 'lm_solve', 'y_elimination_and_map', 'reconstruction')


def med(xs):
    return float(np.median(xs)) if len(xs) else None


def one(attempt, sub):
    d = HERE / 'runs' / attempt / 'archive' / sub
    if not (d / 'result.json').exists():
        return None
    R = json.loads((d / 'result.json').read_text())
    A = json.loads((d / 'audit.json').read_text())
    n = R['intervals']
    inv = R['invocations'] + R['slow_invocations']
    names = list(dict.fromkeys(x['name'] for x in inv))
    subj = {}
    for nm in names:
        rows = [x for x in inv if x['name'] == nm]
        errs = {}
        for x in rows:
            errs[x['case']] = x['same_grid_error']
        subj[nm] = dict(name=nm, family=rows[0]['family'], Rp=rows[0].get('Rp'), q=rows[0].get('q'),
                        tolerance=rows[0].get('tolerance'),
                        worst_error=max(errs.values()), median_error=med(list(errs.values())),
                        gpu_ms=med([x['fused_device_seconds'] for x in rows]) * 1e3,
                        total_ms=med([x['total_seconds'] for x in rows]) * 1e3,
                        gpu_ms_A1=(med([x['fused_device_seconds'] for x in rows if x.get('phase') == 'romA1']) or 0) * 1e3,
                        gpu_ms_A2=(med([x['fused_device_seconds'] for x in rows if x.get('phase') == 'romA2']) or 0) * 1e3,
                        samples=len(rows),
                        lm_attempts=med([x['attempts'] for x in rows if x.get('attempts') is not None]),
                        cg_iterations=med([x['iterations'] for x in rows if x.get('iterations') is not None]))
    rom = [s for s in subj.values() if s['family'] != 'cg']
    cgs = sorted([s for s in subj.values() if s['family'] == 'cg'], key=lambda s: s['gpu_ms'])
    best = min([s for s in rom if s['family'] in ('nm-rom', 'linear-rung') and not (s['family'] == 'nm-rom' and s['q'] > 0)],
               key=lambda s: s['worst_error'])
    ref = next((c for c in cgs if c['worst_error'] <= best['worst_error']), None)
    named = subj.get('cg_0.01')
    for s in rom:
        s['floor'] = R['floors'][str(s['Rp'])]['worst']
        s['speedup_vs_matched_cg'] = ref['gpu_ms'] / s['gpu_ms'] if ref else None
        s['speedup_vs_cg_0.01'] = named['gpu_ms'] / s['gpu_ms'] if named else None
        s['speedup_vs_matched_cg_total'] = ref['total_ms'] / s['total_ms'] if ref else None
        own = next((c for c in cgs if c['worst_error'] <= s['worst_error']), None)
        s['own_matched_cg'] = own['name'] if own else None
        s['speedup_vs_own_matched_cg'] = own['gpu_ms'] / s['gpu_ms'] if own else None
    # PRE-REGISTERED Table-1 setting rule (coordinator, 2026-09-23, fixed before any cluster result):
    # accurate = most accurate arm; fast = cheapest arm with worst error <= the paper's fast setting
    # (q=0, R=512) at this mesh; FOM = fastest tested CG at least as accurate as the accurate arm.
    # primary arms only (A5): head-only q=0 and the bank-span linear rung q=R'; C_q arms are references
    cand = [s for s in rom if s['family'] in ('nm-rom', 'linear-rung') and not (s['family'] == 'nm-rom' and s['q'] > 0)]
    acc = min(cand, key=lambda s: (s['worst_error'], s['gpu_ms']))
    paper_fast = subj[f'R512_q0']
    fastarm = min([s for s in cand if s['worst_error'] <= paper_fast['worst_error']], key=lambda s: s['gpu_ms'])
    table1 = dict(rule='accurate = most accurate arm; fast = cheapest arm with worst error <= (q=0, R=512) worst error; '
                       'FOM = fastest tested CG with worst error <= accurate arm worst error; same job',
                  accurate=acc['name'], fast=fastarm['name'], paper_fast_reference='R512_q0',
                  paper_fast_reference_worst_error=paper_fast['worst_error'], fom=(ref['name'] if ref else None),
                  fom_gpu_ms=(ref['gpu_ms'] if ref else None), fom_worst_error=(ref['worst_error'] if ref else None),
                  accurate_worst_error=acc['worst_error'], accurate_gpu_ms=acc['gpu_ms'],
                  accurate_speedup=(ref['gpu_ms'] / acc['gpu_ms'] if ref else None),
                  fast_worst_error=fastarm['worst_error'], fast_gpu_ms=fastarm['gpu_ms'],
                  fast_speedup=(ref['gpu_ms'] / fastarm['gpu_ms'] if ref else None),
                  accurate_total_ms=acc['total_ms'], fast_total_ms=fastarm['total_ms'],
                  fom_total_ms=(ref['total_ms'] if ref else None))
    # conservative variant: arm time = its median right after a long CG neighbour (neighbour phase, cases 0-2)
    after = {}
    for r in R['neighbour_gate']['rows']:
        after[r['name']] = max(after.get(r['name'], 0), r['after_long_median'] * 1e3)
    table1['accurate_gpu_ms_after_long'] = after[acc['name']]
    table1['fast_gpu_ms_after_long'] = after[fastarm['name']]
    table1['accurate_speedup_after_long'] = ref['gpu_ms'] / after[acc['name']] if ref else None
    table1['fast_speedup_after_long'] = ref['gpu_ms'] / after[fastarm['name']] if ref else None
    primary = {x['name'] for x in cand}
    lim = R['neighbour_gate']['limit']
    gate_breakdown = dict(
        neighbour_primary_worst=max([r['ratio'] for r in R['neighbour_gate']['rows'] if r['name'] in primary] or [None]),
        neighbour_primary_failing=[f"{r['name']}@{r.get('phase', '')} {r['ratio']:.3f}" for r in R['neighbour_gate']['rows'] if r['name'] in primary and r['ratio'] > lim],
        neighbour_other_failing=[f"{r['name']}@{r.get('phase', '')} {r['ratio']:.3f}" for r in R['neighbour_gate']['rows'] if r['name'] not in primary and r['ratio'] > lim],
        drift_primary_range=([min(r['ratio'] for r in R['drift_gate']['rows'] if r['name'] in primary),
                              max(r['ratio'] for r in R['drift_gate']['rows'] if r['name'] in primary)] if R.get('drift_gate') else None),
        drift_failing=([f"{r['name']} {r['ratio']:.3f}" for r in R['drift_gate']['rows'] if not (1 / lim <= r['ratio'] <= lim)] if R.get('drift_gate') else None))
    prof = {}
    for nm in [s['name'] for s in rom]:
        rows = [x for x in R['profile'] if x['name'] == nm]
        prof[nm] = {k: med([x[k] for x in rows]) * 1e3 for k in STAGES}
        prof[nm]['sum'] = sum(prof[nm][k] for k in STAGES)
        prof[nm]['fused_gpu_ms'] = subj[nm]['gpu_ms']
        prof[nm]['host_copies_ms'] = subj[nm]['total_ms'] - subj[nm]['gpu_ms']
        prof[nm]['dominant'] = max(STAGES, key=lambda k: prof[nm][k])
    return dict(attempt=attempt, output=sub, intervals=n, job_id=R['job_id'], gpu=R['gpu'], gpu_uuid=R['gpu_uuid'],
                commit=R['commit'], gates=R['gates'], parity=R['parity'], neighbour_gate=R['neighbour_gate'],
                drift_gate=R.get('drift_gate'), design=R['config'].get('design', 'legacy'),
                audit=dict(verdict=A['verdict'], summary=A['summary']),
                floors={k: v['worst'] for k, v in R['floors'].items()},
                result_sha256=hashlib.sha256((d / 'result.json').read_bytes()).hexdigest(),
                gate_breakdown=gate_breakdown, most_accurate_arm=best['name'], matched_cg=(ref['name'] if ref else None), table1=table1,
                subjects=subj, profile=prof, device_memory=R.get('device_memory'))


def verdict(meshes):
    out = {}
    fam = {'C_q reference q=max': lambda s: (s['family'] in ('nm-rom', 'correction-reference') and (s['q'] > 0 or s['Rp'] <= 32)),
           'nm-rom q=0': lambda s: s['family'] == 'nm-rom' and s['q'] == 0,
           'linear rung q=R\'': lambda s: s['family'] == 'linear-rung'}
    for m in meshes:
        v = {}
        for label, sel in fam.items():
            arms = sorted([s for s in m['subjects'].values() if sel(s)], key=lambda s: -s['Rp'])
            errs = [s['worst_error'] for s in arms]
            mono = all(b >= a - 1e-12 for a, b in zip(errs, errs[1:]))
            v[label] = dict(arms=[s['name'] for s in arms], monotone_error=mono,
                            cost_range=arms[0]['gpu_ms'] / arms[-1]['gpu_ms'],
                            cost_monotone=all(b <= a for a, b in zip([s['gpu_ms'] for s in arms], [s['gpu_ms'] for s in arms][1:])))
        out[m['intervals']] = v
    return out


def main():
    meshes = [x for x in (one(a, s) for a, s in RUNS) if x]
    V = verdict(meshes)
    S = dict(meshes=meshes, verdict=V)
    (HERE / 'reports').mkdir(exist_ok=True)
    (HERE / 'reports' / 'summary.json').write_text(json.dumps(S, indent=1) + '\n')
    L = ['# poisson-bank-knob — generated tables', '',
         'Generated by `make_report.py` from the collected cluster JSONs; do not edit by hand.', '']
    for m in meshes:
        n = m['intervals']
        L += [f"## {n}² — job {m['job_id']}, {m['gpu']}, commit `{(m['commit'] or '')[:8]}`", '',
              f"Gates: {m['gates']}. Audit: {m['audit']['verdict']} ({m['audit']['summary']}).",
              f"Parity: {'; '.join(f"{p['candidate']} {p['worst_field_relative']:.1e}" for p in m['parity']) or 'not run at this mesh (original bank not kept)'}. "
              f"Neighbour gate: {'PASS' if m['neighbour_gate']['passed'] else 'FAIL'}, worst ratio "
              f"{max(r['ratio'] for r in m['neighbour_gate']['rows']):.3f} (limit {m['neighbour_gate']['limit']}).",
              (f"Design {m['design']}. Drift gate (romA2/romA1 per arm): {'PASS' if m['drift_gate']['passed'] else 'FAIL'}, range "
               f"{min(r['ratio'] for r in m['drift_gate']['rows']):.3f}–{max(r['ratio'] for r in m['drift_gate']['rows']):.3f}." if m['drift_gate'] else f"Design {m['design']}."),
              f"Gate breakdown: neighbour worst over primary arms {m['gate_breakdown']['neighbour_primary_worst']:.3f}; failing primary: "
              f"{', '.join(m['gate_breakdown']['neighbour_primary_failing']) or 'none'}; failing other: {', '.join(m['gate_breakdown']['neighbour_other_failing']) or 'none'}; "
              f"drift failing: {', '.join(m['gate_breakdown']['drift_failing'] or []) or 'none'}"
              + (f"; drift range over primary arms {m['gate_breakdown']['drift_primary_range'][0]:.3f}–{m['gate_breakdown']['drift_primary_range'][1]:.3f}." if m['gate_breakdown']['drift_primary_range'] else '.'),
              f"Most accurate ROM arm: `{m['most_accurate_arm']}`; matched CG (fastest with worst error ≤ it): `{m['matched_cg']}`.", '',
              f"**Table-1 settings (pre-registered rule):** accurate `{m['table1']['accurate']}` "
              f"{m['table1']['accurate_worst_error']*100:.3f} % at {m['table1']['accurate_gpu_ms']:.2f} ms = "
              f"{m['table1']['accurate_speedup']:.1f}× vs `{m['table1']['fom']}` ({m['table1']['fom_gpu_ms']:.1f} ms, "
              f"{m['table1']['fom_worst_error']*100:.3f} %); fast `{m['table1']['fast']}` {m['table1']['fast_worst_error']*100:.3f} % "
              f"at {m['table1']['fast_gpu_ms']:.2f} ms = {m['table1']['fast_speedup']:.1f}× (fast bar: `R512_q0` "
              f"{m['table1']['paper_fast_reference_worst_error']*100:.3f} %). Conservative (arm timed right after a long CG "
              f"neighbour): accurate {m['table1']['accurate_gpu_ms_after_long']:.2f} ms = {m['table1']['accurate_speedup_after_long']:.1f}×, "
              f"fast {m['table1']['fast_gpu_ms_after_long']:.2f} ms = {m['table1']['fast_speedup_after_long']:.1f}×.", '',
              "| R' | q | arm | worst err % | median err % | floor % | GPU ms | GPU ms A1 / A2 | total ms | × vs matched CG (GPU) | × vs cg_0.01 (GPU) | own matched CG | × vs own matched CG | LM attempts |",
              '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
        rom = sorted([s for s in m['subjects'].values() if s['family'] != 'cg'],
                     key=lambda s: ({'linear-rung': 0, 'nm-rom': 1, 'correction-reference': 2, 'nm-rom-parent': 3}[s['family']], -s['Rp'], s['q']))
        for s in rom:
            L.append(f"| {s['Rp']} | {s['q']} | {s['family']} | {s['worst_error']*100:.3f} | {s['median_error']*100:.3f} | "
                     f"{s['floor']*100:.3f} | {s['gpu_ms']:.2f} | {s['gpu_ms_A1']:.2f} / {s['gpu_ms_A2']:.2f} | {s['total_ms']:.2f} | "
                     f"{s['speedup_vs_matched_cg']:.1f} | {s['speedup_vs_cg_0.01']:.1f} | "
                     f"{s['own_matched_cg']} | {'' if s['speedup_vs_own_matched_cg'] is None else f"{s['speedup_vs_own_matched_cg']:.1f}"} | "
                     f"{'' if s['lm_attempts'] is None else f"{s['lm_attempts']:.0f}"} |")
        L += ['', '| CG rtol | worst err % | GPU ms | total ms | iterations |', '|---:|---:|---:|---:|---:|']
        for s in sorted([s for s in m['subjects'].values() if s['family'] == 'cg'], key=lambda s: -s['tolerance']):
            L.append(f"| {s['tolerance']:g} | {s['worst_error']*100:.3f} | {s['gpu_ms']:.1f} | {s['total_ms']:.1f} | {s['cg_iterations']:.0f} |")
        L += ['', '**Stage profile (median ms, each stage its own synchronised jit):**', '',
              '| arm | project + start | LM solve | y elim + map | reconstruction u=Gc | stage sum | fused GPU | host copies | dominant |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for nm, p in m['profile'].items():
            L.append(f"| {nm} | {p['project_and_start']:.2f} | {p['lm_solve']:.2f} | {p['y_elimination_and_map']:.2f} | "
                     f"{p['reconstruction']:.2f} | {p['sum']:.2f} | {p['fused_gpu_ms']:.2f} | {p['host_copies_ms']:.2f} | {p['dominant']} |")
        L += ['', '**Knob verdict inputs:**', '']
        for fam, v in V[n].items():
            L.append(f"- {fam}: error monotone as R' falls: {v['monotone_error']}; GPU cost monotone: {v['cost_monotone']}; "
                     f"cost range (most/least expensive end of the family): {v['cost_range']:.2f}×")
        L.append('')
    (HERE / 'reports' / 'tables.generated.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
