"""Generate the lane summary (checks/summary.json) and the report (reports/2026-09-24-burgers-eq-tol-knobs.md) from the
audited per-mesh summaries checks/<attempt>-summary.json. No number in the report is typed by hand.

    python reports/generate_report.py [--attempts e1024 e4096c]
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPORT = HERE / 'reports' / '2026-09-24-burgers-eq-tol-knobs.md'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f(x, d=3):
    if x is None:
        return '—'
    if isinstance(x, float):
        return f'{x:.{d}f}'
    return str(x)


def g(x):
    return f'{x:g}' if x is not None else '—'


def row_order(lad):
    seen, out = set(), []
    for n in ([lad['dense']] if lad['dense'] else []) + lad['eq_ladder'][::-1] + lad['tol_ladder']:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempts', nargs='+', default=['e1024', 'e4096c'])
    a = p.parse_args()
    meshes, missing = {}, []
    for att in a.attempts:
        sp = HERE / 'checks' / f'{att}-summary.json'
        if not sp.exists():
            missing.append(att)
            continue
        s = json.loads(sp.read_text())
        lad = s['ladder']
        mesh = dict(attempt=att, job_id=s['job_id'], gpu=s['gpu'], host=s['host'], commit=s['commit'],
                    intervals=s['intervals'], accepted=s['accepted'], failed_gates=s['failed_gates'],
                    audit_summary=f'checks/{att}-summary.json', audit_summary_sha256=sha(sp),
                    result_json_sha256=s['sources']['result_json_sha256'], elapsed_seconds=s['elapsed_seconds'],
                    table1_fom=lad['table1_fom'], table1_fom_ms=lad['table1_fom_ms'],
                    table1_fom_worst_percent=lad['table1_fom_worst_percent'],
                    table1_parity=s['gates']['table1_parity'],
                    failed_gate_details={k: {kk: vv for kk, vv in s['gates'][k].items() if not isinstance(vv, (list, dict))}
                                         for k in s['failed_gates']},
                    gates={k: v['passed'] for k, v in s['gates'].items()},
                    gate_details={k: {kk: (len(vv) if kk == 'not_evaluable' else vv) for kk, vv in v.items()
                                      if kk in ('worst', 'limit', 'worst_relative_diff', 'worst_abs_diff', 'ratio_min',
                                                'ratio_max', 'not_evaluable', 'evaluable')}
                                  for k, v in s['gates'].items() if k in ('drift_ABA', 'neighbour', 'rho_recomputed_in_numpy',
                                                                          'full_grid_errors_recomputed',
                                                                          'restricted_recomputation_tracks_job')},
                    ladders={})
        for Rp, L_ in lad['ladders'].items():
            rows = [lad['rows'][n] for n in row_order(L_)]
            cur = lad['rows'][L_['current']]
            eqr = [lad['rows'][n] for n in L_['eq_ladder']]
            tlr = [lad['rows'][n] for n in L_['tol_ladder']]
            mesh['ladders'][Rp] = dict(
                current=cur['arm'], rows=rows,
                dense_over_current_ms=L_['dense_over_current_ms'],
                dense_worst_over_current=(lad['rows'][L_['dense']]['worst_over_current'] if L_['dense'] else None),
                eq_ms_range_over_current=[min(x['ms_over_current'] for x in eqr), max(x['ms_over_current'] for x in eqr)],
                tol_time_saving_vs_current={g(x['gtol']): x['time_saving_vs_current'] for x in tlr},
                tol_worst_over_current={g(x['gtol']): x['worst_over_current'] for x in tlr})
        meshes[str(s['intervals'])] = mesh
    summary = dict(lane='burgers-eq-tol-knobs', branch='exp/2026-09-24-burgers-eq-tol-knobs',
                   design='experiments/burgers-eq-tol-knobs/DESIGN.md', meshes=meshes, not_done=missing,
                   rule_used_by_table1='lat64: hops.lattice_rule(L, 64), uniform 63x63 interior sub-lattice, equal weights '
                                       '(L/64)^2, no fit (config rules.lat64 = {"lattice": 64})')
    sp = HERE / 'checks' / 'summary.json'
    sp.write_text(json.dumps(summary, indent=1) + '\n')

    L = []
    w = L.append
    w('# Burgers 2D: empirical-quadrature node count and stopping tolerance at the Table-1 settings')
    w('')
    state = ('final' if not missing and all(m['accepted'] for m in meshes.values()) else
             'final, with the failed audit gates stated per mesh' if not missing else 'partial')
    w(f'Same-job measurements of the two solver-side cost knobs of the frozen Burgers 2D NM-ROM (dev6 cases) at the '
      f'Table-1 accurate and fast settings, against the Table-1 Newton–BiCGStab FOM in the same allocation. '
      f'Numbers are **{state}**' + (f' ({", ".join(missing)} not done)' if missing else '') +
      '; every number below is generated by `reports/generate_report.py` from the audited summaries.')
    w('')
    w('## Which quadrature rule Table 1 uses')
    w('')
    w('Every Table-1 Burgers 2D arm at $1024^2$ and $4096^2$ carries rule `lat64` (arm names `…_lat64_…`), defined in '
      'the configs as `{"lattice": 64}` and built by `hops.lattice_rule(L, 64)`: the uniform $63\\times63$ interior '
      'sub-lattice with equal weights $(L/64)^2$ — **a fixed lattice, not an NNLS fit** (no training states, no draw). '
      'The ladder therefore varies the lattice spacing (anisotropic lattices for the 0.5× and 2× rungs, DESIGN §2).')
    w('')
    for mk, m in meshes.items():
        w(f'## ${mk}^2$ — attempt `{m["attempt"]}`, job {m["job_id"]}, {m["gpu"]} ({m["host"]}), commit `{m["commit"][:10]}`')
        w('')
        w(f'Gates: **{"all pass" if m["accepted"] else "FAILED: " + ", ".join(m["failed_gates"])}**. '
          f'Table-1 parity (current settings vs the recorded Table-1 error, bar $10^{{-6}}$ relative): ' +
          '; '.join(f"R'={x['R_prime']} {x['this_job_percent']:.10f} % vs {x['recorded_percent']:.10f} % "
                    f"(rel {x['relative']:.1e}, job {x['recorded_job']})" for x in m['table1_parity']['rows']) + '.')
        w('')
        for k, v in m['failed_gate_details'].items():
            w(f'Failed gate `{k}`: `{json.dumps(v)}`.' + (
                ' The perturbed-error control runs through the full-grid predicate, and no full fields are saved at this '
                'mesh (DESIGN §5, config `audit_cases: []`), so that control cannot fire; the swapped-case and '
                'perturbed-time controls are detected. The gate is reported as failed, not re-defined after the fact.'
                if k == 'controls_detected' and v.get('swapped_case_rejected') and v.get('perturbed_A2_time_x1p2_rejected')
                and not v.get('perturbed_error_1em6_rejected') else ''))
            w('')
        w(f'Table-1 FOM `{m["table1_fom"]}`: {m["table1_fom_ms"]:.2f} ms, worst {m["table1_fom_worst_percent"]:.4f} %.')
        w('')
        for Rp, lad in m['ladders'].items():
            w(f"### $R'={Rp}$ (current arm `{lad['current']}`)")
            w('')
            w('| residual | $N_{\\rm eq}$ (× current) | gtol | worst % | median % | ms | × current ms | speedup vs Table-1 FOM | '
              'own-comparator speedup | LM iterations per case | max it/step | certificate ($\\rho_{\\max}$ cert / conf) |')
            w('|---|---|---|---|---|---|---|---|---|---|---|---|')
            for x in lad['rows']:
                neq = '—' if x['N_eq'] is None else f"{x['N_eq']} ({x['N_eq_ratio']:.3f})"
                cert = x['certificate'] if x['rho_max_cert'] is None else (
                    f"{x['certificate']} ({x['rho_max_cert']:.3f} / {x['rho_max_confirmation']:.3f})")
                cur = ' **(current)**' if x['arm'] == lad['current'] else ''
                w(f"| {x['rule']}{cur} | {neq} | {g(x['gtol'])} | {x['worst_percent']:.4f} | {x['median_percent']:.4f} | "
                  f"{x['median_gpu_ms']:.2f} ({x['mode']}) | {x['ms_over_current']:.3f} | {x['speedup_vs_table1_fom']:.2f}× | "
                  f"{f(x['own_speedup'], 2)}{'×' if x['own_speedup'] else ''} ({x['own_fom'] or 'none'}) | "
                  f"{x['iterations_per_case']} | {x['max_iterations_per_step']} | {cert} |")
            w('')
            if lad['dense_over_current_ms'] is not None:
                w(f"Dense residual / current rule: time ×{lad['dense_over_current_ms']:.2f}, worst error "
                  f"×{lad['dense_worst_over_current']:.4f}.")
            w('Tolerance ladder (vs current gtol $10^{-3}$): ' + '; '.join(
                f"gtol {k}: time saving {100 * v:+.1f} %, worst error ×{lad['tol_worst_over_current'][k]:.4f}"
                for k, v in lad['tol_time_saving_vs_current'].items()) + '.')
            w('')
        dd = m['gate_details']
        w('Timing/audit gate values: ' + '; '.join(f'{k} {json.dumps(v)}' for k, v in dd.items()) + '.')
        w('')
    if missing:
        w('## Not done')
        w('')
        w(', '.join(missing) + ': no audited summary; no number is reported for it.')
        w('')
    w('## Caveats')
    w('')
    w('- Six development cases only (no held-out cohort in this lane); worst-case errors over 6 cases.')
    w('- The 0.5× and 2× quadrature rungs are anisotropic lattices ($31\\times63$, $63\\times127$), one orientation only.')
    w('- Certificates are on reached states of the eqcert populations with the lane\'s primary state set $k\\ge1$, '
      'bar 0.116; they cover stored endpoint states, not every trial state.')
    w('- $4096^2$ ran on an A100 80 GB (the H200 queue was ~2 days); the Table-1 row was timed on an H200 (bk4096b), so '
      'absolute ms and the speedup differ from the paper even though errors reproduce; ratios here are same-job.')
    w('- ms are medians of ≥10 invocations per case pooled over 6 cases and both ROM phases; no dispersion reported.')
    w('')
    w('## Glossary')
    w('')
    for k, v in [
        ('dev6', 'the six development parameter cases used for Table 1 (params_draw(7090702,4) + params_draw(911702,2)).'),
        ("$R'$", 'bank width: how many of the ordered bank functions the reduced solution uses (the unknowns of the solve).'),
        ('$M$', "number of fixed sine test functions in the weak residual ($M=4R'$)."),
        ('empirical quadrature (EQ) / rule', 'evaluating the nonlinear advection term only at a subset of $N_{\\rm eq}$ '
         'grid nodes with weights instead of at all $(L-1)^2$ nodes.'),
        ('lat64 / lat32 / lat32x64 / lat64x128 / lat128', 'tensor sub-lattices with $s_x\\times s_y$ intervals '
         '(lat64 = 64×64 → $63\\times63$ nodes), equal weights; lat64 is the Table-1 rule.'),
        ('$N_{\\rm eq}$', 'number of quadrature nodes; “× current” is relative to lat64 (3969).'),
        ('dense', 'the residual evaluated at every interior node every step (no quadrature).'),
        ('gtol', 'Levenberg–Marquardt stopping tolerance on the scaled gradient $\\lVert J^\\top r\\rVert/(\\lVert J\\rVert\\lVert r\\rVert)$; '
         'the solve at a time step stops when it falls below gtol (or the residual is tiny).'),
        ('worst % / median %', 'worst / median over the 6 cases of the largest relative $L^2$ error over the five evolved output '
         'times, relative to the initial field norm, against the tight Newton reference on the same mesh.'),
        ('ms', 'median GPU milliseconds per query (dense initial field on the GPU → six output fields), in the faster '
         'compile mode (default or `graphs`, XLA command buffers) where both were run.'),
        ('A–B–A', 'timing order: all ROM arms (A1), then all FOM settings (B), then all ROM arms again (A2); drift and '
         'neighbour gates check that timings did not move between phases or depend on the preceding call.'),
        ('Table-1 FOM', 'the full-order Newton–BiCGStab setting used as the comparator in Table 1 (`lean_nt3e-3_l3e-3_dt005`), '
         'timed in the same job; speedup = its median ms / the arm\'s median ms.'),
        ('own-comparator speedup', 'speedup against the fastest converged FOM setting at least as accurate as the arm.'),
        ('LM iterations per case', 'total Levenberg–Marquardt iterations over the 50 time steps, one number per case.'),
        ('certificate / $\\rho$', '$\\rho$ = relative error of the quadrature-projected advection vs the exact projection on '
         'states the arm itself reaches on 56 separate certification trajectories; "confirmed" = every one of 5 draws and '
         'the confirmation draw has $\\rho_{\\max}\\le0.116$.'),
        ('Table-1 parity', 'the current-setting arm must reproduce the worst error recorded in the job behind Table 1 to '
         '$10^{-6}$ relative.'),
    ]:
        w(f'- **{k}**: {v}')
    w('')
    REPORT.write_text('\n'.join(L))
    print(REPORT)
    print(sp, sha(sp))


if __name__ == '__main__':
    main()
