"""Generate the ns2d lane report (Markdown) and summary.json from the run result JSONs.

    python experiments/ns2d/reports/generate_ns2d.py \
        --phase1 experiments/ns2d/artifacts/ns101/result.json[:JOBID] \
        [--phase2 <result.json>[:JOBID] ...] [--phase3 <result.json>[:JOBID] ...] \
        [--audit <audit.json> ...] --out experiments/ns2d/reports/2026-09-1x-ns2d

Every number in the report is read from the JSONs; nothing is typed by hand.  summary.json
holds one object per table row: phase, mesh, subject, metric, value, gate, passed, job_id,
source_sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sci(x, d=3):
    return '—' if x is None else (f'{x:.{d}e}' if isinstance(x, float) else str(x))


def fx(x, d=3):
    return '—' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def yn(x):
    return {True: 'yes', False: 'no'}.get(x, '—')


def load(spec):
    path, _, job = spec.partition(':')
    d = json.loads(Path(path).read_text())
    return d, (job or d.get('job_id') or '—'), hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Doc:
    def __init__(self):
        self.md = []
        self.rows = []

    def h(self, level, t):
        self.md.append('#' * level + ' ' + t + '\n')

    def p(self, t):
        self.md.append(t + '\n')

    def table(self, header, rows):
        self.md.append('| ' + ' | '.join(header) + ' |')
        self.md.append('|' + '|'.join(['---'] * len(header)) + '|')
        for r in rows:
            self.md.append('| ' + ' | '.join(str(c) for c in r) + ' |')
        self.md.append('')

    def row(self, **kw):
        self.rows.append(kw)


def phase1(doc, d, job, sha):
    G = d['gates']
    doc.h(2, f'Phase 1 — full-order model gates and the dataset (job {job}, {d["gpu"] if "gpu" in d else d.get("device")}, commit `{str(d["commit"])[:12]}`, backend `{d["backend"]}`, complete={yn(d.get("complete"))})')
    rows = []
    for k, g in G.items():
        main = {kk: v for kk, v in g.items() if isinstance(v, float)}
        key = next(iter(main), None)
        rows.append([k, yn(g['passed']), key or '—', sci(main.get(key)) if key else '—'])
        for kk, v in main.items():
            doc.row(phase=1, mesh=g.get('N', g.get('meshes')), subject='fom', metric=f'{k}.{kk}', value=v,
                    gate=k, passed=g['passed'], job_id=job, source_sha256=sha)
    doc.table(['gate', 'passed', 'headline metric', 'value'], rows)
    doc.h(3, 'Taylor–Green')
    tg = [[g['N'], sci(g['exact_rel']), sci(g['semi_rel']), sci(g['cont_rel']), sci(g['cont_pred']),
           sci(g['wrong_lambda_rel']), sci(g['be_rel']), yn(g['passed'])]
          for k, g in G.items() if k.startswith('F-TG_N')]
    doc.table(['$N$', 'vs closed-form discrete', 'vs semi-discrete', 'vs continuum', 'predicted continuum',
               'wrong-$\\lambda$ control', 'backward-Euler control', 'passed'], tg)
    if 'F-TG-ORDER' in G:
        doc.p(f'Continuum TG observed orders over the ladder: {", ".join(fx(o) for o in G["F-TG-ORDER"]["orders"])} (pass = {yn(G["F-TG-ORDER"]["passed"])}).')
    if 'F-MMS' in G:
        m = G['F-MMS']
        doc.h(3, 'Manufactured solution with $J\\neq0$')
        doc.table(['spatial errors', 'spatial orders', 'temporal $\\Delta t$', 'temporal order', 'flipped-sign control', 'passed'],
                  [[', '.join(sci(e) for e in m['spatial_errors']), ', '.join(fx(o) for o in m['spatial_orders']),
                    ', '.join(sci(t, 1) for t in m['temporal_dts']), fx(m['temporal_order']), sci(m['flipped_sign_control']), yn(m['passed'])]])
    bud = [[g['N'], sci(g['identity_enstrophy']), sci(g['identity_energy']), sci(g['conservation_enstrophy']),
            sci(g['conservation_energy']), sci(g['be_control_enstrophy']), yn(g['passed'])]
           for k, g in G.items() if k.startswith('F-BUDGET_N')]
    if bud:
        doc.h(3, 'Budgets')
        doc.table(['$N$', 'enstrophy identity', 'energy identity', '$\\nu=0$ enstrophy drift', '$\\nu=0$ energy drift', 'backward-Euler control', 'passed'], bud)
    mesh = [[k[7:], sci(g['nu'], 1), ', '.join(sci(e) for e in g['errors_vs_finest']), ', '.join(fx(o) for o in g['orders']), yn(g['passed'])]
            for k, g in G.items() if k.startswith('F-MESH_')]
    if mesh:
        doc.h(3, 'Mesh refinement of the family')
        doc.table(['case', '$\\nu$', 'errors vs finest', 'observed orders (successive)', 'passed'], mesh)
    if 'F-INDEP' in G:
        g = G['F-INDEP']
        doc.p(f'Independent NumPy implementation at $N={g["N"]}$ over {g["steps"]} steps: worst relative difference {sci(g["worst_rel"])} (pass = {yn(g["passed"])}).')
    data = [[k, v['N'], v['trajectories'], v['states'], sci(v['worst_rel_residual']), fx(v['max_cfl'], 2), v['newton_max_it'], fx(v['seconds'], 0), v['sha256'][:16]]
            for k, v in d['data'].items() if isinstance(v, dict) and 'sha256' in v]
    doc.h(3, 'Dataset')
    doc.table(['cohort', '$N$', 'trajectories', 'states', 'worst Newton residual', 'max CFL', 'max Newton iters', 'seconds', 'SHA256 (prefix)'], data)
    for k, v in d['data'].items():
        if isinstance(v, dict) and 'sha256' in v:
            doc.row(phase=1, mesh=v['N'], subject=f'data_{v["cohort"]}', metric='max_cfl', value=v['max_cfl'], gate=f'F-DATA_{v["cohort"]}_N{v["N"]}',
                    passed=G.get(f'F-DATA_{v["cohort"]}_N{v["N"]}', {}).get('passed'), job_id=job, source_sha256=sha)


def phase2(doc, d, job, sha):
    G = d['gates']
    c = d['config']
    doc.h(2, f'Phase 2 — bank and head $K={c["K"]}$, $R={c["R"]}$ (job {job}, commit `{str(d["commit"])[:12]}`, complete={yn(d.get("complete"))})')
    t = d.get('training', {})
    doc.p(f'Training: {t.get("n_snapshots")} snapshots on ${c["TRAIN_N"]}^2$, {t.get("steps")} steps, reconstruction rel-$L_2$ mean {sci(t.get("recon_rel_l2_mean"))}, median {sci(t.get("recon_rel_l2_median"))}, max {sci(t.get("recon_rel_l2_max"))}, {fx(t.get("seconds"), 0)} s.')
    rows = []
    for N, fl in d.get('floors', {}).items():
        for kind, ks in fl.items():
            for k, v in ks.items():
                rows.append([N, kind, k, sci(v['worst_evolved_fixed']), sci(v['median_case_worst_fixed']), sci(v['t0_fixed_median']), sci(v['worst_all_times_fixed'])])
                for m in ('worst_evolved_fixed', 'median_case_worst_fixed', 't0_fixed_median'):
                    doc.row(phase=2, mesh=int(N), subject=f'{kind}_{k}', metric=f'floor.{m}', value=v[m], gate=f'B-FLOOR_N{N}',
                            passed=G.get(f'B-FLOOR_N{N}', {}).get('passed'), job_id=job, source_sha256=sha)
    doc.h(3, 'Projection floors on the held-out development trajectories')
    doc.table(['$N$', 'span', 'dimension', 'worst evolved', 'median case (worst over $t$)', '$t=0$ median', 'worst all times'], rows)
    rows = []
    for N, o in d.get('oracle', {}).items():
        rows.append([N, o['states'], sci(o['oracle_median']), sci(o['oracle_worst']), sci(o['single_start_median']), sci(o['podK_median']),
                     sci(o['bank_floor_median']), fx(o['podK_median'] / o['oracle_median'], 2), yn(G.get(f'H-ORACLE_N{N}', {}).get('passed')), yn(G.get(f'H-SOLVED_N{N}', {}).get('passed'))])
        for m in ('oracle_median', 'oracle_worst', 'single_start_median', 'podK_median', 'bank_floor_median'):
            doc.row(phase=2, mesh=int(N), subject=f'head_K{c["K"]}_R{c["R"]}', metric=f'oracle.{m}', value=o[m], gate=f'H-ORACLE_N{N}',
                    passed=G.get(f'H-ORACLE_N{N}', {}).get('passed'), job_id=job, source_sha256=sha)
    doc.h(3, 'Head oracle versus the linear POD-$K$ floor (same held-out states)')
    doc.table(['$N$', 'states', 'oracle median', 'oracle worst', 'single-start median', 'POD-$K$ median', 'bank floor median', 'POD-$K$ / oracle', 'H-ORACLE', 'H-SOLVED'], rows)
    doc.table(['gate', 'passed'], [[k, yn(g['passed'])] for k, g in G.items()])


def phase3(doc, d, job, sha):
    G = d['gates']
    c = d['config']
    ck = d['checkpoint']
    doc.h(2, f'Phase 3 — ROM ladder, head $K={ck["K"]}$, $R={ck["R"]}$, $N={c["N"]}$ (job {job}, {d.get("gpu")}, commit `{str(d["commit"])[:12]}`, complete={yn(d.get("complete"))})')
    agg = d.get('aggregates', {})
    rows = []
    for name, a in agg.items():
        rows.append([name, sci(a['worst_evolved']), sci(a['median_evolved']), sci(a['worst_all']), sci(a['worst_t0']), fx(a['median_seconds'], 3), a['budget_exits'], yn(a['finite'])])
        for m in ('worst_evolved', 'median_evolved', 'worst_all', 'worst_t0', 'median_seconds'):
            doc.row(phase=3, mesh=c['N'], subject=name, metric=m, value=a[m], gate='R-LADDER' if name.startswith('neural') else None,
                    passed=G.get('R-LADDER', {}).get('passed') if name.startswith('neural') else None, job_id=job, source_sha256=sha)
    doc.table(['subject', 'worst evolved', 'median evolved', 'worst all times', 'worst $t=0$', 'median s', 'budget exits', 'finite'], rows)
    if 'R-LADDER' in G:
        g = G['R-LADDER']
        doc.p(f'Pre-registered verdict: monotone = {yn(g["monotone"])}, gain top/q0 = {fx(g["gain_top_over_q0"], 2)}×, cost top/q0 = {fx(g["cost_ratio_top_over_q0"], 2)}×, **{"PASS" if g["passed"] else "FAIL"}**.')
    doc.table(['gate', 'passed'], [[k, yn(g['passed'])] for k, g in G.items()])


GLOSSARY = '''## Glossary

- **FOM / ROM**: the full grid solve / the reduced model.
- **Closed-form discrete TG**: the exact solution of the *discrete* equations for the Taylor–Green mode, $\\omega^n=((1-a)/(1+a))^n\\omega_0$.
- **Semi-discrete / continuum**: the same mode with exact time integration but the discrete Laplacian eigenvalue / the true PDE solution.
- **Predicted continuum error**: what a second-order scheme must produce for that mode; the gate checks the measured error matches it.
- **Enstrophy / energy identity**: the per-step discrete budget the implicit-midpoint scheme satisfies exactly (roundoff-level residual).
- **$\\nu=0$ drift**: change of the conserved quantities in the inviscid case; backward Euler is the control that must drift.
- **Observed order (successive)**: $\\log_2$ of the ratio of successive-level differences; the finest level is not used as a reference because that biases a second-order error to $\\log_2 5$.
- **Bank / head / latent code**: learned spatial features / the neural map from the code to bank coefficients / the numbers solved online.
- **Floor**: error of the best projection onto a span (bank or POD); no ROM on that span can do better.
- **Oracle / single-start**: best latent fit by multi-start LM / the query-time single-start policy.
- **Worst evolved / worst all times / $t=0$**: error normalised by the initial reference norm, maximised over $t>0$ / over all outputs / at $t=0$ only.
- **Correction rank $q$**: extra linear directions solved jointly with the code; POD-LSPG is the classical linear control.
- **Budget exits**: LM steps that hit the iteration cap (an unconverged rung is flagged).
- **CFL**: $\\max|\\mathbf u|\\Delta t/h$; recorded, not a stability limit for the implicit scheme.
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase1', nargs='*', default=[])
    p.add_argument('--phase2', nargs='*', default=[])
    p.add_argument('--phase3', nargs='*', default=[])
    p.add_argument('--audit', nargs='*', default=[])
    p.add_argument('--title', default='ns2d — 2D incompressible Navier–Stokes: certified FOM, dataset, bank/head floors, ROM ladder')
    p.add_argument('--status', default='provisional')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    doc = Doc()
    doc.h(1, a.title)
    doc.p(f'Numbers are **{a.status}**. Generated by `generate_ns2d.py` from the run JSONs listed below; nothing is typed by hand. Design and pre-registration: `../DESIGN.md`.')
    srcs = []
    for ph, specs in ((1, a.phase1), (2, a.phase2), (3, a.phase3)):
        for s in specs:
            d, job, sha = load(s)
            srcs.append([ph, s.split(':')[0], job, sha[:16], yn(d.get('complete')), yn(d.get('all_passed'))])
            {1: phase1, 2: phase2, 3: phase3}[ph](doc, d, job, sha)
    doc.md.insert(2, '')
    hdr = ['phase', 'source', 'job id', 'sha256 (prefix)', 'complete', 'all gates passed']
    doc.md.insert(3, '| ' + ' | '.join(hdr) + ' |\n|' + '|'.join(['---'] * len(hdr)) + '|\n' + '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in srcs) + '\n')
    for s in a.audit:
        d = json.loads(Path(s).read_text())
        doc.p(f'Independent NumPy audit `{s}`: {d.get("n_checks")} checks, all match = {yn(d.get("all_match"))}.')
    doc.md.append(GLOSSARY)
    out = Path(a.out)
    out.with_suffix('.md').write_text('\n'.join(doc.md))
    (out.parent / 'summary.json').write_text(json.dumps(doc.rows, indent=1, allow_nan=False) + '\n')
    print(out.with_suffix('.md'), len(doc.rows), 'rows')


if __name__ == '__main__':
    main()
