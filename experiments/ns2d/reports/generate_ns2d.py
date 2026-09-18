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
    arch = c.get('ARCH', {})
    doc.p(f'Training: {t.get("n_snapshots")} snapshots on ${c["TRAIN_N"]}^2$, {t.get("steps")} steps, reconstruction rel-$L_2$ mean {sci(t.get("recon_rel_l2_mean"))}, median {sci(t.get("recon_rel_l2_median"))}, max {sci(t.get("recon_rel_l2_max"))}, {fx(t.get("seconds"), 0)} s. '
          f'Architecture: bank MLP {arch.get("g_hidden")}×{arch.get("g_layers")} (structural rank bound $\\min(R,\\texttt{{g\\_hidden}})={min(c["R"], arch.get("g_hidden", c["R"]))}$, DESIGN §A4), head MLP {arch.get("h_hidden")}×{arch.get("h_layers")} + linear skip.')
    for m in ('recon_rel_l2_mean', 'recon_rel_l2_median', 'recon_rel_l2_max'):
        if m in t:
            doc.row(phase=2, mesh=c['TRAIN_N'], subject=f'head_K{c["K"]}_R{c["R"]}', metric=f'training.{m}', value=t[m], gate='H-TRAIN',
                    passed=G.get('H-TRAIN', {}).get('passed'), job_id=job, source_sha256=sha)
    orth = [[k[7:], g.get('rank'), sci(g.get('cond_Rb')), yn(g['passed'])] for k, g in G.items() if k.startswith('B-ORTH_N')]
    if orth:
        doc.h(3, 'Bank rank on each evaluation grid (thin QR of $G$)')
        doc.table(['$N$', 'numerical rank', '$\\kappa(R_b)$', 'B-ORTH passed'], orth)
        for k, g in G.items():
            if k.startswith('B-ORTH_N'):
                doc.row(phase=2, mesh=int(k[8:]), subject=f'bank_R{c["R"]}', metric='rank', value=g.get('rank'), gate=k, passed=g['passed'], job_id=job, source_sha256=sha)
    dat = [[k[7:], g.get('mode', 'hash'), yn(g.get('hash_mismatch', g.get('expected') is not None and g.get('expected') != g.get('got'))), sci(g.get('value_worst_rel')), yn(g['passed'])] for k, g in G.items() if k.startswith('B-DATA_')]
    if dat:
        doc.h(3, 'Cohort identity against Phase 1 (hash, or value on the archived 8 trajectories — DESIGN §A4)')
        doc.table(['cohort', 'mode', 'hash mismatch', 'value worst rel.', 'passed'], dat)
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
                     sci(o['bank_floor_median']), fx(o['podK_median'] / o['oracle_median'], 2), sci(o.get('formula_vs_field_worst_rel')),
                     f"{int(o.get('oracle_reasons', {}).get('0', 0))}/{o['states']} (median {fx(o.get('oracle_iters_median'), 0)} it)",
                     yn(G.get(f'H-ORACLE_N{N}', {}).get('passed')), yn(G.get(f'H-SOLVED_N{N}', {}).get('passed'))])
        doc.row(phase=2, mesh=int(N), subject=f'head_K{c["K"]}_R{c["R"]}', metric='oracle.budget_exits_of_states', value=int(o.get('oracle_reasons', {}).get('0', 0)), gate=f'H-ORACLE_N{N}',
                passed=G.get(f'H-ORACLE_N{N}', {}).get('passed'), job_id=job, source_sha256=sha)
        for m in ('oracle_median', 'oracle_worst', 'single_start_median', 'podK_median', 'bank_floor_median'):
            doc.row(phase=2, mesh=int(N), subject=f'head_K{c["K"]}_R{c["R"]}', metric=f'oracle.{m}', value=o[m], gate=f'H-ORACLE_N{N}',
                    passed=G.get(f'H-ORACLE_N{N}', {}).get('passed'), job_id=job, source_sha256=sha)
    doc.h(3, 'Head oracle versus the linear POD-$K$ floor (same held-out states)')
    doc.table(['$N$', 'states', 'oracle median', 'oracle worst', 'single-start median', 'POD-$K$ median', 'bank floor median', 'POD-$K$ / oracle', 'formula vs field', 'LM budget exits', 'H-ORACLE', 'H-SOLVED'], rows)
    doc.p('The H-ORACLE bar is oracle median $\\le \\tfrac12$ POD-$K$ median (ratio $\\ge 2$). "formula vs field" is the worst relative difference between the whitened-metric formula and the direct field-space evaluation of the oracle error (— for runs before DESIGN §A4, whose reported oracle numbers were computed through $R_b^{-1}$; the audit measured the contamination). "LM budget exits" counts held-out states whose best start stopped at the 300-iteration LM cap (reason code 0) rather than at the gradient tolerance; the oracle is an upper bound on the true best fit for those states.')
    # Per-output-time breakdown from the stored per-state arrays (state order is case-major:
    # cases x times, as the driver writes them; the audit checks json == npz element-wise).
    rows = []
    for N, o in d.get('oracle', {}).items():
        times = o.get('times')
        if not times or 'per_state_oracle' not in o:
            continue
        nt = len(times)
        nc = len(o['per_state_oracle']) // nt
        for key, label in (('per_state_oracle', 'oracle'), ('per_state_single', 'single-start'), ('per_state_podK', f'POD-{c["K"]}'), ('per_state_bank', f'bank-{c["R"]} floor')):
            E = np.asarray(o[key], dtype=float).reshape(nc, nt)
            med = np.median(E, 0)
            rows.append([N, label] + [sci(float(v)) for v in med] + [sci(float(E.max()))])
            for ti, tv in enumerate(times):
                doc.row(phase=2, mesh=int(N), subject=f'head_K{c["K"]}_R{c["R"]}', metric=f'oracle_by_time.{label}.median_t{tv * float(c.get("DT", 0.0)) * float(c.get("OUT_EVERY", 0)):g}', value=float(med[ti]), gate=f'H-ORACLE_N{N}',
                        passed=G.get(f'H-ORACLE_N{N}', {}).get('passed'), job_id=job, source_sha256=sha)
        Eo = np.asarray(o['per_state_oracle'], dtype=float).reshape(nc, nt)
        Ep = np.asarray(o['per_state_podK'], dtype=float).reshape(nc, nt)
        r0 = float(np.median(Ep[:, 0]) / np.median(Eo[:, 0]))
        rev = float(np.median(Ep[:, 1:]) / np.median(Eo[:, 1:]))
        rows.append([N, f'POD-{c["K"]} / oracle (median)', fx(r0, 2)] + ['—'] * (nt - 1) + [f'evolved: {fx(rev, 2)}'])
        doc.row(phase=2, mesh=int(N), subject=f'head_K{c["K"]}_R{c["R"]}', metric='oracle_by_time.podK_over_oracle.t0', value=r0, gate=f'H-ORACLE_N{N}', passed=G.get(f'H-ORACLE_N{N}', {}).get('passed'), job_id=job, source_sha256=sha)
        doc.row(phase=2, mesh=int(N), subject=f'head_K{c["K"]}_R{c["R"]}', metric='oracle_by_time.podK_over_oracle.evolved', value=rev, gate=f'H-ORACLE_N{N}', passed=G.get(f'H-ORACLE_N{N}', {}).get('passed'), job_id=job, source_sha256=sha)
    if rows:
        times = next(iter(d['oracle'].values()))['times']
        snap_dt = float(c.get('DT', 0.0)) * float(c.get('OUT_EVERY', 0))  # oracle 'times' are snapshot indices
        doc.h(3, f'Held-out error by output time (median over the {next(iter(d["oracle"].values()))["states"] // len(times)} dev cases; last column = worst state)')
        doc.table(['$N$', 'quantity'] + [f'$t={tv * snap_dt:g}$' for tv in times] + ['worst'], rows)
        tr = d.get('training', {})
        o256 = d['oracle'].get(str(c['TRAIN_N'])) or next(iter(d['oracle'].values()))
        gap = (o256['oracle_median'] / tr['recon_rel_l2_median']) if tr.get('recon_rel_l2_median') else None
        doc.p(f'Training reconstruction median {sci(tr.get("recon_rel_l2_median"))} (trained codes, training snapshots) versus held-out oracle median {sci(o256["oracle_median"])} at the training mesh: held-out / training ratio {fx(gap, 2)}. '
              'A ratio near 1 means the head is capacity-limited; a large ratio means the trained manifold does not cover held-out states (generalisation, not capacity).')
        doc.row(phase=2, mesh=c['TRAIN_N'], subject=f'head_K{c["K"]}_R{c["R"]}', metric='heldout_oracle_over_training_recon_median', value=gap, gate='H-ORACLE', passed=G.get(f'H-ORACLE_N{c["TRAIN_N"]}', {}).get('passed'), job_id=job, source_sha256=sha)
    doc.table(['gate', 'passed'], [[k, yn(g['passed'])] for k, g in G.items()])


def headfit(doc, d, job, sha):
    c = d['config']
    doc.h(2, f'Phase 2 diagnosis (DESIGN §A9, ns301) — head-only on the frozen $K={c["K"]}$, $R={c["R"]}$ bank of checkpoint `{Path(d["checkpoint"]["path"]).name}` (job {job}, commit `{str(d["commit"])[:12]}`, complete={yn(d.get("complete"))})')
    doc.p(f'Nested training subsets (the first $n$ trajectories of the gated {c["N_TRAIN"]}-trajectory cohort) × regimes; {c["STEPS"]} steps; '
          f'selection (reg regimes) on dev cases {c["REPORT_CASES"]}–{c["REPORT_CASES"] + c["SEL_CASES"] - 1}, every reported number on dev cases 0–{c["REPORT_CASES"] - 1} '
          f'({c["REPORT_CASES"] * len(c["ORACLE_TIMES"])} states); the train-oracle is the same fit on the first {c["TRAIN_ORACLE_CASES"]} training trajectories. '
          'Caveat (pre-registered): the frozen bank was trained on all trajectories, which biases the small-$n$ arms optimistically.')
    rows = []
    for tag, a in d['arms'].items():
        ev, tr = a['eval']['dev_report'], a['eval']['train']
        rows.append([a['n_traj'], a['regime'], f"{a['arch']['hidden']}×{a['arch']['layers']}", sci(ev['oracle_median']), sci(ev['podK_median']),
                     fx(ev['ratio_podK_over_oracle'], 3), sci(tr['oracle_median']), sci(a['training']['recon_rel_l2_median']),
                     fx(a['heldout_over_train_oracle_median'], 1), a['training']['best_step'], f"{ev['budget_exits']}/{ev['states']}"])
        for m, v in (('dev_report.oracle_median', ev['oracle_median']), ('dev_report.podK_median', ev['podK_median']),
                     ('dev_report.ratio_podK_over_oracle', ev['ratio_podK_over_oracle']), ('train.oracle_median', tr['oracle_median']),
                     ('training.recon_rel_l2_median', a['training']['recon_rel_l2_median']),
                     ('heldout_over_train_oracle_median', a['heldout_over_train_oracle_median']),
                     ('dev_select.oracle_median', a['eval']['dev_select']['oracle_median']), ('best_step', a['training']['best_step'])):
            doc.row(phase='2-headfit', mesh=c['TRAIN_N'], subject=tag, metric=m, value=v, gate='A9-DIAGNOSIS', passed=None, job_id=job, source_sha256=sha)
    doc.h(3, 'Held-out (dev-report) oracle versus training-subset size, with the train-oracle beside it')
    doc.table(['$n$ traj', 'regime', 'head', 'dev oracle median', 'POD-$K$ median (same subset)', 'POD-$K$ / oracle', 'train oracle median', 'train recon median', 'dev / train oracle', 'best step', 'LM budget exits'], rows)
    rows = []
    for name, sm in d['summary'].items():
        e512 = d['arms'][f'n{max(sm["n_traj"])}_{name}']['eval']['dev_report']['oracle_median']
        plain512 = d['arms'][f'n{max(sm["n_traj"])}_plain']['eval']['dev_report']['oracle_median']
        moved = (plain512 - e512) / plain512
        pair = [np.log(sm['dev_report_oracle_median'][i + 1] / sm['dev_report_oracle_median'][i]) / np.log(sm['n_traj'][i + 1] / sm['n_traj'][i]) for i in range(len(sm['n_traj']) - 1)]
        rows.append([name, ', '.join(sci(v) for v in sm['dev_report_oracle_median']), fx(sm['loglog_slope'], 3), ', '.join(fx(float(v), 2) for v in pair), f'{100 * moved:+.1f} %', sm['verdict']])
        doc.row(phase='2-headfit', mesh=c['TRAIN_N'], subject=name, metric='loglog_slope', value=sm['loglog_slope'], gate='A9-DIAGNOSIS', passed=None, job_id=job, source_sha256=sha)
        doc.row(phase='2-headfit', mesh=c['TRAIN_N'], subject=name, metric='verdict', value=sm['verdict'], gate='A9-DIAGNOSIS', passed=None, job_id=job, source_sha256=sha)
        doc.row(phase='2-headfit', mesh=c['TRAIN_N'], subject=name, metric='reg_moves_n512_vs_plain_fraction', value=moved, gate='A9-DIAGNOSIS', passed=None, job_id=job, source_sha256=sha)
    doc.h(3, 'Pre-registered reading (DESIGN §A9): slope of the dev oracle in $n$')
    doc.table(['regime', 'dev oracle median at $n$ = ' + ', '.join(str(v) for v in next(iter(d['summary'].values()))['n_traj']), 'log-log slope (LS fit)', 'successive slopes', 'vs plain at largest $n$', 'verdict (§A9 rule)'], rows)
    doc.p('Rule: slope $\\le -0.25$ = needs-data; slope $\\ge -0.10$ under every regime = head-limited; otherwise ambiguous. Regularisation "moves it" if the reg regime is $\\ge 10$ % below plain at the largest $n$. Successive slopes show whether the improvement is decelerating.')


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
- **Held-out / training ratio**: the held-out oracle median divided by the training-reconstruction median; near 1 = the head cannot represent the data (capacity), large = it represents training states but not new ones (generalisation).
- **Train-oracle / dev-oracle**: the same best-found fit on states of training trajectories / of held-out trajectories; their ratio is the generalisation gap of an arm.
- **Successive slopes**: log-log slope between consecutive subset sizes; a slope that shrinks toward 0 means more data is helping less and less.
- **Evolved**: output times $t>0$; the $t=0$ state is the band-limited initial condition and is reported separately because it is much easier to fit.
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
    p.add_argument('--headfit', nargs='*', default=[])
    p.add_argument('--audit', nargs='*', default=[])
    p.add_argument('--title', default='ns2d — 2D incompressible Navier–Stokes: certified FOM, dataset, bank/head floors, ROM ladder')
    p.add_argument('--status', default='provisional')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    doc = Doc()
    doc.h(1, a.title)
    doc.p(f'Numbers are **{a.status}**. Generated by `generate_ns2d.py` from the run JSONs listed below; nothing is typed by hand. Design and pre-registration: `../DESIGN.md`.')
    srcs = []
    for ph, specs in ((1, a.phase1), (2, a.phase2), ('2-headfit', a.headfit), (3, a.phase3)):
        for s in specs:
            d, job, sha = load(s)
            srcs.append([ph, s.split(':')[0], job, sha[:16], yn(d.get('complete')), yn(d.get('all_passed'))])
            {1: phase1, 2: phase2, '2-headfit': headfit, 3: phase3}[ph](doc, d, job, sha)
    doc.md.insert(2, '')
    hdr = ['phase', 'source', 'job id', 'sha256 (prefix)', 'complete', 'all gates passed']
    doc.md.insert(3, '| ' + ' | '.join(hdr) + ' |\n|' + '|'.join(['---'] * len(hdr)) + '|\n' + '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in srcs) + '\n')
    for s in a.audit:
        d = json.loads(Path(s).read_text())
        bad = [ch['name'] for ch in d.get('checks', []) if not ch.get('match', True)]
        extra = ''
        if bad:
            worst = max((ch for ch in d['checks'] if not ch.get('match', True)), key=lambda ch: abs(ch.get('recomputed', 0) - ch.get('reported', 0)))
            extra = f' Mismatched: {", ".join(f"`{b}`" for b in bad)} (largest: recomputed {sci(worst["recomputed"])} vs reported {sci(worst["reported"])}).'
        doc.p(f'Independent NumPy audit `{s}`: {d.get("n_checks")} checks, all match = {yn(d.get("all_match"))}.{extra}')
        for ch in d.get('checks', []):
            if not ch.get('match', True):
                doc.row(phase='audit', mesh=None, subject=Path(s).parent.name, metric=ch['name'], value=ch.get('recomputed'), gate='audit', passed=False, job_id='—', source_sha256=hashlib.sha256(Path(s).read_bytes()).hexdigest())
    doc.md.append(GLOSSARY)
    out = Path(a.out)
    out.with_suffix('.md').write_text('\n'.join(doc.md))
    (out.parent / 'summary.json').write_text(json.dumps(doc.rows, indent=1, allow_nan=False) + '\n')
    print(out.with_suffix('.md'), len(doc.rows), 'rows')


if __name__ == '__main__':
    main()
