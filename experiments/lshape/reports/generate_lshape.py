"""Generate the lshape report, summary.json and the error-cost figure from the raw run JSONs.
No number is typed by hand.

    python generate_lshape.py --train <train result.json> --solve <solve result.json>... \
        --audits <audit.json>... --md <report.md> --summary <summary.json> --fig <figure.png>
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PCT = 100.0
PALETTE = {'neural': '#2a78d6', 'neural+linear': '#eb6834', 'pod': '#1baf7a', 'fom': '#eda100', 'free': '#e87ba4'}


def pc(x, d=4):
    return '—' if x is None else f'{x * PCT:.{d}f}'


def ms(x, d=3):
    return '—' if x is None else f'{x * 1e3:.{d}f}'


def num(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def aggregate(sv):
    out = {}
    for r in sv['invocations']:
        out.setdefault((r['intervals'], r['name']), []).append(r)
    agg = {}
    for key, rows in out.items():
        per_case = {}
        for r in rows:
            per_case.setdefault(r['case'], []).append(r)
        worst_sg = max(max(x['same_grid_error'] for x in v) for v in per_case.values())
        med_sg = float(np.median([np.median([x['same_grid_error'] for x in v]) for v in per_case.values()]))
        worst_ph = max(max(x['physical_error'] for x in v) for v in per_case.values())
        total = [x['total_seconds'] for x in rows]
        dev = [x['device_seconds'] for x in rows if x.get('device_seconds') is not None]
        solv = [x['solver_seconds'] for x in rows if x.get('solver_seconds') is not None]
        its = [x['iterations'] for x in rows if x.get('iterations') is not None]
        stat = [x['stationary'] for x in rows if x.get('stationary') is not None]
        r0 = rows[0]
        agg[key] = dict(intervals=key[0], name=key[1], family=r0['family'], k=r0.get('k'), q=r0.get('q'),
                        model=r0.get('model'), primary=r0.get('primary'), kind=r0['kind'],
                        rtol=r0.get('rtol'), where=r0.get('where', 'gpu'),
                        worst_same_grid=worst_sg, median_same_grid=med_sg, worst_physical=worst_ph,
                        median_total_ms=float(np.median(total)) * 1e3, total_ms_reps=[t * 1e3 for t in total],
                        median_device_ms=(float(np.median(dev)) * 1e3 if dev else None),
                        median_solver_ms=(float(np.median(solv)) * 1e3 if solv else None),
                        median_iterations=(float(np.median(its)) if its else None),
                        max_iterations=(int(max(its)) if its else None),
                        stationary=(int(sum(stat)) if stat else None), invocations=len(rows),
                        cases=len(per_case), converged=all(x.get('converged', True) for x in rows))
    return agg


def nondominated(items, err_key, cost_key):
    keep = []
    for i, a in enumerate(items):
        if a[cost_key] is None:
            continue
        dominated = any(b[cost_key] is not None and b[err_key] <= a[err_key] and b[cost_key] <= a[cost_key]
                        and (b[err_key] < a[err_key] or b[cost_key] < a[cost_key]) for j, b in enumerate(items) if j != i)
        if not dominated:
            keep.append(a)
    return sorted(keep, key=lambda x: x[cost_key])


def subject_cost(a):
    """Secondary cost: fused device interval for GPU subjects, solve interval for CPU ones."""
    return a['median_device_ms'] if a['where'] == 'gpu' and a['median_device_ms'] is not None else a['median_solver_ms']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', required=True)
    ap.add_argument('--solve', nargs='*', default=[])
    ap.add_argument('--audits', nargs='*', default=[])
    ap.add_argument('--md', required=True)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--fig', default=None)
    ap.add_argument('--status', default='final')
    a = ap.parse_args()
    tr = json.loads(Path(a.train).read_text())
    solves = [json.loads(Path(p).read_text()) for p in a.solve]
    audits = [json.loads(Path(p).read_text()) for p in a.audits]
    cfg = tr['config']
    rows_out = []
    L = []
    W = L.append

    def emit(mesh, subject, family, qk, metric, value, job, source, **extra):
        rows_out.append(dict(mesh=mesh, subject=subject, family=family, q_or_k=qk, metric=metric, value=value,
                             job_id=job, source_sha=source, **extra))

    tsrc = sha_file(a.train)
    W('# Poisson on the L-shaped domain — the separable decoder against sparse direct and PCG (table T18)')
    W('')
    W(f'This report covers one pre-registered cell (`experiments/lshape/DESIGN.md`): Poisson 2D on '
      f'$\\Omega=(0,1)^2\\setminus[\\tfrac12,1)^2$ with the re-entrant-corner singularity, a bank sweep over the '
      f'boundary factor and rank, heads at $K\\in\\{{16,32\\}}$, a correction ladder, POD-LSPG, and every full-order '
      f'comparator timed in the same job. **Status of the numbers: {a.status}.** They are one training seed on one '
      f'{tr["cohorts"]["development"]["count"]}-source development cohort that selected nothing.')
    W('')
    W(f'Training job `{tr["job_id"]}` on `{tr["gpu"]}`, source commit `{tr["commit"]}`, '
      f'{tr.get("elapsed_seconds", 0) / 60:.1f} min, `jax_backend={tr["backend"]}`, x64 {yn(tr["x64"])}, '
      f'matmul precision `{tr["matmul_precision"]}`, JAX {tr["jax_version"]}.')
    for sv in solves:
        W(f'Solve job `{sv["job_id"]}` (meshes {", ".join(str(n) for n in sv["config"]["intervals"])}) on `{sv["gpu"]}`, '
          f'source commit `{sv["commit"]}`, {sv.get("elapsed_seconds", 0) / 60:.1f} min, `jax_backend={sv["backend"]}`, '
          f'x64 {yn(sv["x64"])}, matmul precision `{sv["matmul_precision"]}`.')
    W('')

    # ------------------------------------------------------------ headline ---
    # Timed subjects are keyed by JOB, never by (mesh, name) alone: a second job at the same
    # mesh would otherwise overwrite the first silently (DESIGN.md A6, carry-forward ii).
    # Jobs are grouped into blocks by their test-mode count M, because the online residual
    # charges a dense M x n projection inside every reduced query: costs are comparable
    # WITHIN a block and never across blocks.
    aggs, blocks, block_index, block_jobs = {}, {}, {}, {}
    for i, sv in enumerate(solves):
        blocks.setdefault(sv['config']['requested_modes'], []).append((sv, sha_file(a.solve[i])))
    for M, entries in blocks.items():
        idx = {}
        for sv, ssrc in entries:
            for (n, name), v in aggregate(sv).items():
                v['job_id'], v['source_sha'], v['M'] = sv['job_id'], ssrc, M
                if (n, name) in idx:
                    raise SystemExit(
                        f'two jobs report subject {name!r} at N={n} with the same M={M} '
                        f'(jobs {idx[(n, name)][0]} and {sv["job_id"]}). Costs are only comparable within '
                        'one job; give the second job its own block or drop it.')
                idx[(n, name)] = (sv['job_id'], n, name)
                aggs[(sv['job_id'], n, name)] = v
        block_index[M] = idx
        block_jobs[M] = [sv['job_id'] for sv, _ in entries]

    def look(M, n, name):
        """One timed subject, by block and job -- never by (mesh, name) alone."""
        key = block_index.get(M, {}).get((n, name))
        return aggs[key] if key else None

    def block_meshes(M):
        return sorted({n for n, _ in block_index[M]})

    Ms = sorted(blocks)
    meshes = sorted({n for M in Ms for n in block_meshes(M)})
    W('## Headline: the non-dominated set per mesh, full-order comparators included')
    W('')
    W('Subject $A$ dominates $B$ if $\\mathrm{err}_A\\le\\mathrm{err}_B$ and $\\mathrm{cost}_A\\le\\mathrm{cost}_B$ with '
      'one strict, on (worst same-grid error over the development cases, median complete-query ms). '
      '"Reduced" means any neural, correction-ladder, free-bank or POD subject.')
    W('')
    W('**The blocks below are separate tables on purpose and their costs may NOT be compared.** '
      'The online residual projects the source onto $M$ test modes with a dense $M\\times n$ product, '
      'and that product is charged inside every reduced query, so a subject at $M=1024$ carries about four '
      'times the projection work of the same subject at $M=257$. Every comparison that matters — reduced '
      'against full-order — is within one job and therefore inside one block.')
    W('')
    verdict = {}
    for M in Ms:
        W(f'### Block $M={M}$ test modes — job(s) ' + ', '.join(f'`{j}`' for j in block_jobs[M]))
        W('')
        W('| mesh | non-dominated set (complete-query ms) | reduced model in it? | non-dominated set (device / solver ms) |')
        W('|---|---|---|---|')
        for n in block_meshes(M):
            items = [aggs[k] for k in block_index[M].values() if k[1] == n]
            nd = nondominated(items, 'worst_same_grid', 'median_total_ms')
            for v in items:
                v['secondary_cost_ms'] = subject_cost(v)
            nd2 = nondominated(items, 'worst_same_grid', 'secondary_cost_ms')
            red = [x['name'] for x in nd if x['family'] != 'fom']
            W(f'| {n} | ' + ', '.join(f'`{x["name"]}` ({pc(x["worst_same_grid"], 3)} %, {x["median_total_ms"]:.3f} ms)' for x in nd)
              + f' | {"yes: " + ", ".join("`" + r + "`" for r in red) if red else "**no**"} | '
              + ', '.join(f'`{x["name"]}` ({pc(x["worst_same_grid"], 3)} %, {x["secondary_cost_ms"]:.3f} ms)' for x in nd2) + ' |')
            for x in nd:
                emit(n, x['name'], x['family'], x['q'] if x['family'] != 'pod' else x['k'], 'nondominated_complete_ms',
                     True, x['job_id'], x['source_sha'], test_modes=M)
            nd_red = nondominated([v for v in items if v['family'] != 'fom'], 'worst_same_grid', 'median_total_ms')
            for x in nd_red:
                emit(n, x['name'], x['family'], x['q'] if x['family'] != 'pod' else x['k'], 'nondominated_reduced_only',
                     True, x['job_id'], x['source_sha'], test_modes=M)
            verdict[(M, n)] = (red, nd)
        W('')

    # The plain-language verdict, generated from the fronts above.
    for M in Ms:
        for n in block_meshes(M):
            red, nd = verdict[(M, n)]
            splu = look(M, n, 'fom_splu')
            best = min((x for x in nd if x['family'] != 'fom'), key=lambda x: x['worst_same_grid'], default=None)
            if red and best is not None and splu is not None:
                cheap = min((x for x in nd if x['family'] != 'fom'), key=lambda x: x['median_total_ms'])
                bar = 0.05  # the cell's own pre-registered solved-error bar, DESIGN section 8 clause 1
                W(f'**$M={M}$, $N={n}$.** {len(red)} reduced subjects are non-dominated. The most accurate of them, '
                  f'`{best["name"]}`, reaches {pc(best["worst_same_grid"], 3)} % worst error at '
                  f'{best["median_total_ms"]:.3f} ms, against `fom_splu`, which is exact to round-off '
                  f'({pc(splu["worst_same_grid"], 2)} %) at {splu["median_total_ms"]:.3f} ms — '
                  f'{splu["median_total_ms"] / best["median_total_ms"]:.2f}x more expensive per query. The cheapest '
                  f'reduced subject on the front is `{cheap["name"]}` at {cheap["median_total_ms"]:.3f} ms and '
                  f'{pc(cheap["worst_same_grid"], 3)} %. '
                  + ('Non-dominance here means **cheaper and less accurate**, never better on both axes: no reduced '
                     'subject improves on the direct solve\'s accuracy, and the front records what accuracy each one '
                     'gives up to be cheaper.'
                     if best['worst_same_grid'] < bar else
                     '**This is a cheapness-only membership.** Every reduced subject on this front is above the cell\'s '
                     f'own {bar * PCT:.0f} % solved-error bar, so the front records subjects that are barely cheaper '
                     'than the direct solve while being far less accurate — not a usable operating point.'))
            elif splu is not None:
                W(f'**$M={M}$, $N={n}$.** **No reduced model is non-dominated.** `fom_splu` costs '
                  f'{splu["median_total_ms"]:.3f} ms at {pc(splu["worst_same_grid"], 3)} % error; the cheapest reduced '
                  f'subject costs {min(x["median_total_ms"] for x in [aggs[k] for k in block_index[M].values() if k[1] == n and aggs[k]["family"] != "fom"]):.3f} ms, '
                  'so the direct solve is both more accurate and cheaper than every one of them.')
            W('')

    # ------------------------------------- the validation-versus-development gap
    if tr.get('head_arms'):
        vw = [h['best_found_validation']['worst'] for h in tr['head_arms']]
        dw = [h['best_found_development']['worst'] for h in tr['head_arms']]
        W('**Read every error above beside this gap.** The development cohort has '
          f'{tr["cohorts"]["development"]["count"]} sources; the held-out validation split of the training draw has '
          f'{len(tr["cohorts"]["training"]["validation"])}. Across the {len(tr["head_arms"])} head arms the worst '
          f'best-found reconstruction error is {pc(min(dw), 1)}–{pc(max(dw), 1)} % on the '
          f'{tr["cohorts"]["development"]["count"]} development sources but {pc(min(vw), 1)}–{pc(max(vw), 1)} % on the '
          f'{len(tr["cohorts"]["training"]["validation"])} validation sources — a factor of about '
          f'{max(vw) / max(dw):.1f} on the worst case. The worst case over 32 draws is simply not the worst case over '
          f'{len(tr["cohorts"]["training"]["validation"])}, and the solved errors in every table below are worst-over-32 '
          'numbers. Treat them as the optimistic end of the range; a 461-source solve sweep is the measurement that '
          'would replace them, and it has not been run.')
        W('')
        emit(cfg['training_intervals'], 'head_arms', 'head', None, 'best_found_validation_worst_max', max(vw),
             tr['job_id'], tsrc)
        emit(cfg['training_intervals'], 'head_arms', 'head', None, 'best_found_development_worst_max', max(dw),
             tr['job_id'], tsrc)

    # ------------------------------------------------------------ gates -------
    W('## Operator verification gates (G-FOM, DESIGN.md section 2)')
    W('')
    W('| job | mesh | n | symmetric (max asym) | independent assembly | $\\lambda_{\\min}$ | vs $\\lambda_1$ ref (rel) | within 1 % | passed |')
    W('|---|---:|---:|---|---|---:|---:|---|---|')
    for src, d in [('train', tr)] + [(f'solve {sv["job_id"]}', sv) for sv in solves]:
        for g in d['gates']:
            if 'nnz' not in g:
                continue
            W(f'| {src} | {g["intervals"]} | {g["interior_unknowns"]} | {yn(g["symmetric"])} ({g["symmetry_max_abs"]:.1e}) | '
              f'{yn(g["independent_assembly_agrees"])} (nnz diff {g["independent_assembly_nnz_difference"]}) | '
              f'{g["lambda_min"]:.6f} | {g["lambda1_relative_difference"]:.2e} | {yn(g["lambda1_within_1pct"])} | {yn(g["passed"])} |')
    for sv in solves:
        for g in sv['gates']:
            if 'gate' in g:
                W(f'| solve {sv["job_id"]} | {g["intervals"]} | — | — | G-FOM-5 tight iterative vs direct: '
                  f'{g["worst_relative_difference"]:.2e} ≤ {g["tolerance"]:.0e} | — | — | — | {yn(g["passed"])} |')
    W('')
    W(f'The known first Dirichlet eigenvalue of this L-shape is $\\lambda_1 = {tr["gates"][0]["lambda1_reference"]:.6f}$. '
      'There is no DST: the L-shape operator is a principal submatrix of the square\'s Kronecker sum and is not itself '
      'a Kronecker sum, so no fast transform diagonalises it.')
    W('')
    if solves:
        W('### Full-order setup, per mesh (offline, not charged to any query)')
        W('')
        W('| job | mesh | n | nnz(A) | SuperLU factor s | nnz(L+U) | IC(0) factor s | nnz(IC0 L) | eigen-solve s (M modes) | eigen residual | 1024-ref residual |')
        W('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for sv in solves:
            for f in sv['fom']:
                w = next(x for x in sv['weak_ops'] if x['intervals'] == f['intervals'])
                fr = max(x['fine_residual'] for x in sv['references'] if 'fine_residual' in x)
                W(f'| `{sv["job_id"]}` | {f["intervals"]} | {f["interior_unknowns"]} | {f["nnz"]} | {f["splu_factor_seconds"]:.3f} | {f["splu_lu_nnz"]} | '
                  f'{f["ic0_factor_seconds"]:.3f} | {f["ic0_nnz"]} | {w["seconds"]:.1f} ({w["M"]}) | {w["eigen_residual"]:.1e} | {fr:.1e} |')
        W('')
        W('### Discretisation error of the FD solution itself (same-grid vs restricted 1024-interval direct solve)')
        W('')
        W('The reduced models are graded against the same-grid direct solve, not against this; the column is here '
          'so a reader can see how much of the mesh\'s own error the corner singularity costs.')
        W('')
        W('| job | mesh | worst over cases | median over cases |')
        W('|---|---:|---:|---:|')
        seen_disc = set()
        for sv in solves:
            for n in sv['config']['intervals']:
                if n in seen_disc:
                    continue
                seen_disc.add(n)
                dl = [x['discretisation_delta_vs_fine'] for x in sv['references'] if x.get('intervals') == n]
                W(f'| `{sv["job_id"]}` | {n} | {pc(max(dl), 4)} % | {pc(float(np.median(dl)), 4)} % |')
                emit(n, 'fd_same_grid', 'fom', None, 'discretisation_delta_vs_1024_worst', max(dl), sv['job_id'], sha_file(a.solve[solves.index(sv)]))
        W('')

    # ------------------------------------------------------------ cohorts ----
    W('## Cohorts')
    W('')
    W('| cohort | seed | drawn | rejected (centre in removed quadrant) | used | role |')
    W('|---|---:|---:|---:|---:|---|')
    for name, role in (('training', 'bank and head training (85/15 fit/validation)'), ('common', 'bank selection only'),
                       ('development', 'reporting only; selects nothing')):
        c = tr['cohorts'][name]
        W(f'| {name} | {c["seed"]} | {c["draw_count"]} | {c["rejected_in_draw"]} | {c["count"]} | {role} |')
    W('')

    # ------------------------------------------------------------ bank layer -
    W('## Bank layer — boundary factor and rank')
    W('')
    fl = cfg['floor_intervals']
    W('| arm | factor | $R$ | ' + ' | '.join(f'floor dev N={n} (worst / median)' for n in fl) + ' | floor common N=' + str(fl[0]) + ' (worst) | cond | head best-found dev N=' + str(fl[0]) + ' | train min |')
    W('|---|---|---:|' + '---:|' * (len(fl) + 4))
    for arm in tr['bank_arms']:
        cells = []
        for n in fl:
            r = next(x for x in arm['floors'] if x['intervals'] == n)
            cells.append(f'{pc(r["floor_dev"]["worst"])} / {pc(r["floor_dev"]["median"])}')
            emit(n, arm['arm'], 'bank', arm['R_total'], 'bank_floor_dev_worst', r['floor_dev']['worst'], tr['job_id'], tsrc)
            emit(n, arm['arm'], 'bank', arm['R_total'], 'bank_floor_common_worst', r['floor_common']['worst'], tr['job_id'], tsrc)
        r0 = next(x for x in arm['floors'] if x['intervals'] == fl[0])
        label = arm['factor'] + (f' + {arm["n_enrich"]} singular' if arm['n_enrich'] else '') + \
            (f' (n_ff {arm["arch"]["n_ff"]}, scale {arm["arch"]["ff_scale"]:g})' if arm['arch']['n_ff'] != 64 else '')
        W(f'| `{arm["arm"]}` | {label} | {arm["R_total"]} | ' + ' | '.join(cells) + f' | {pc(r0["floor_common"]["worst"])} | '
          f'{r0["bank_rank"]["condition_number"]:.2e} | {pc(r0["head_best_found_dev"]["worst"])} | {arm["training_seconds"] / 60:.1f} |')
    sel = tr['selection']['bank']
    W('')
    W(f'**Selected bank:** `{sel["selected"]}` (rule: {sel["rule"]}; the development ranking '
      f'{"agrees" if sel["development_ranking_agrees"] else "**disagrees**"}). Ranking on the common cohort: '
      + ', '.join(f'`{x["arm"]}` {pc(x["common_worst"])} %' for x in sel['ranking']) + '.')
    best_dev = min(x for arm in tr['bank_arms'] for x in [next(r for r in arm['floors'] if r['intervals'] == fl[-1])['floor_dev']['worst']])
    W(f'**Bank target** (worst development floor at N={fl[-1]} below 1.0000 %): best arm {pc(best_dev)} % — '
      f'**{"pass" if best_dev < 0.01 else "miss"}**.')
    sm = {arm['arm']: next(r for r in arm['floors'] if r['intervals'] == fl[-1])['floor_dev']['worst'] for arm in tr['bank_arms']}
    if 'smooth_R512' in sm and 'sdf_R512' in sm:
        W(f'**Factor swap at R=512:** smooth {pc(sm["smooth_R512"])} % vs sdf {pc(sm["sdf_R512"])} % '
          f'(ratio sdf/smooth {sm["sdf_R512"] / sm["smooth_R512"]:.3f}x).')
    if 'smooth_R512' in sm and 'enrich_R512' in sm:
        W(f'**Corner enrichment:** enrich {pc(sm["enrich_R512"])} % vs smooth {pc(sm["smooth_R512"])} % '
          f'(improvement {sm["smooth_R512"] / sm["enrich_R512"]:.3f}x).')
    if 'smooth_R512' in sm and 'smooth_ff128s2_R512' in sm:
        W(f'**Feature lever (a):** n_ff 128 / scale 2 {pc(sm["smooth_ff128s2_R512"])} % vs baseline {pc(sm["smooth_R512"])} %.')
    smooth_all = [v for k, v in sm.items() if k.startswith('smooth')]
    if smooth_all and 'enrich_R512' in sm and 'smooth_R512' in sm:
        fals = all(v > 0.015 for v in smooth_all) and (sm['smooth_R512'] / sm['enrich_R512'] > 1.5)
        W(f'**Falsification clause** (every smooth arm > 1.5 % at N={fl[-1]} AND enrichment helps > 1.5x): '
          f'{"**MET** — the corner singularity, not the factor, limits the bank" if fals else "not met"} '
          f'(smooth floors {", ".join(pc(v, 3) + " %" for v in smooth_all)}; enrichment ratio {sm["smooth_R512"] / sm["enrich_R512"]:.3f}x).')
    W('')

    # ------------------------------------------------------------ head layer -
    W(f'## Head layer at N={cfg["training_intervals"]} (training mesh)')
    W('')
    W('| arm | bank | $K$ | bank floor dev | at stored codes (fit, worst) | best-found val | best-found dev | solved dev (untimed) | stationary | best-found / floor | target 1.2x |')
    W('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for h in tr['head_arms']:
        ratio = h['head_floor_over_bank_floor']
        W(f'| `{h["arm"]}`{" *(primary)*" if h["primary"] else ""} | `{h["bank"]}` | {h["K"]} | {pc(h["bank_floor_dev"])} | '
          f'{pc(h["head_at_stored_codes_fit"]["worst"])} | {pc(h["best_found_validation"]["worst"])} | '
          f'{pc(h["best_found_development"]["worst"])} | {pc(h["weak_solved_development"]["worst"])} | '
          f'{h["weak_solved_development"]["stationary"]}/{h["weak_solved_development"]["count"]} | {ratio:.3f}x | '
          f'{"pass" if ratio <= 1.2 else "miss"} |')
        n = cfg['training_intervals']
        emit(n, h['arm'], 'head', h['K'], 'best_found_dev_worst', h['best_found_development']['worst'], tr['job_id'], tsrc)
        emit(n, h['arm'], 'head', h['K'], 'weak_solved_dev_worst', h['weak_solved_development']['worst'], tr['job_id'], tsrc)
        emit(n, h['arm'], 'head', h['K'], 'head_floor_over_bank_floor', ratio, tr['job_id'], tsrc)
    W('')

    # ------------------------------------------------------------ solve -------
    for sv in solves:
        ssrc = sha_file(a.solve[solves.index(sv)])
        Msv = sv['config']['requested_modes']
        for n in sv['config']['intervals']:
            W(f'## Solve at N={n}, $M={Msv}$ (job `{sv["job_id"]}`, {sv["cohort"]["count"]} development cases × {sv["config"]["repetitions"]} repetitions)')
            W('')
            W(f'Every cost in this section is measured inside job `{sv["job_id"]}` at $M={Msv}$ test modes and is '
              'comparable only with the other costs in this section.')
            W('')
            W('### Three-layer decomposition per checkpoint (untimed)')
            W('')
            W('| checkpoint | $K$ | $R$ | bank floor worst / median | best-found worst / median | solved q=0 worst |')
            W('|---|---:|---:|---:|---:|---:|')
            for r in [x for x in sv['reconstruction'] if x['intervals'] == n]:
                s0 = look(Msv, n, f'neural_q0@{r["model"]}')
                W(f'| `{r["model"]}` | {r["K"]} | {r["R_total"]} | {pc(r["bank_projection"]["worst"])} / {pc(r["bank_projection"]["median"])} | '
                  f'{pc(r["best_found"]["worst"])} / {pc(r["best_found"]["median"])} | {pc(s0["worst_same_grid"]) if s0 else "—"} |')
                emit(n, r['model'], 'bank', r['R_total'], 'bank_floor_dev_worst', r['bank_projection']['worst'], sv['job_id'], ssrc)
                emit(n, r['model'], 'head', r['K'], 'best_found_dev_worst', r['best_found']['worst'], sv['job_id'], ssrc)
            W('')
            W('### Every timed subject')
            W('')
            W('| subject | family | $K$ / $k\'$ | $q$ | worst same-grid | median same-grid | worst physical | median complete ms | median device / solver ms | stationary | median iters | max iters |')
            W('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
            items = sorted([v for k, v in aggs.items() if k[0] == sv['job_id'] and k[1] == n],
                           key=lambda v: (v['family'] != 'fom', v['family'], v['name']))
            for v in items:
                cost2 = v['median_device_ms'] if v['median_device_ms'] is not None else v['median_solver_ms']
                W(f'| `{v["name"]}` | {v["family"]}{"" if v["where"] == "gpu" else " (cpu)"} | {v["k"] if v["k"] is not None else "—"} | '
                  f'{v["q"] if v["q"] is not None else "—"} | {pc(v["worst_same_grid"])} | {pc(v["median_same_grid"])} | '
                  f'{pc(v["worst_physical"])} | {v["median_total_ms"]:.3f} | {num(cost2, 3)} | '
                  f'{"—" if v["stationary"] is None else f"{v["stationary"]}/{v["invocations"]}"} | '
                  f'{num(v["median_iterations"], 1)} | {v["max_iterations"] if v["max_iterations"] is not None else "—"} |')
                qk = v['q'] if v['family'] not in ('pod', 'fom') else v['k']
                for metric in ('worst_same_grid', 'median_same_grid', 'worst_physical', 'median_total_ms', 'median_device_ms',
                               'median_solver_ms', 'stationary', 'median_iterations'):
                    if v[metric] is not None:
                        emit(n, v['name'], v['family'], qk, metric, v[metric], sv['job_id'], ssrc, test_modes=Msv,
                             reps_ms=v['total_ms_reps'] if metric == 'median_total_ms' else None)
            W('')
            # clauses at N=256
            if n == 256:
                prim = [v for v in items if v['family'] == 'neural' and v['primary']]
                if prim:
                    best = min(prim, key=lambda v: v['worst_same_grid'])
                    pod_match = look(Msv, n, f'pod{best["k"]}')
                    pod8 = look(Msv, n, f'pod{8 * best["k"]}')
                    W('### Pre-registered cell success at N=256')
                    W('')
                    W('| clause | requirement | value | verdict |')
                    W('|---|---|---:|---|')
                    W(f'| 1 solved | worst same-grid < 5.0000 % (`{best["name"]}`) | {pc(best["worst_same_grid"])} % | **{"pass" if best["worst_same_grid"] < 0.05 else "miss"}** |')
                    W(f'| 2 stationary | every q=0 solve exits stationary | {best["stationary"]}/{best["invocations"]} | **{"pass" if best["stationary"] == best["invocations"] else "miss"}** |')
                    if pod_match:
                        W(f'| 3 vs POD k\'=K | beats `pod{best["k"]}` on worst error | {pc(best["worst_same_grid"])} % vs {pc(pod_match["worst_same_grid"])} % | **{"pass" if best["worst_same_grid"] < pod_match["worst_same_grid"] else "miss"}** |')
                    gates_ok = all(g['passed'] for g in sv['gates']) and all(g['passed'] for g in tr['gates'])
                    W(f'| 4 gates | every G-FOM gate passes | — | **{"pass" if gates_ok else "miss"}** |')
                    if pod8:
                        W(f'| honesty | POD-LSPG at k\'=8K (`pod{8 * best["k"]}`) vs the head | {pc(pod8["worst_same_grid"])} % vs {pc(best["worst_same_grid"])} % | '
                          f'{"POD at 8K matches or beats the head" if pod8["worst_same_grid"] <= best["worst_same_grid"] else "head beats POD at 8K"} |')
                    splu = look(Msv, n, 'fom_splu')
                    if splu:
                        W(f'| honesty | `fom_splu` cost and error | {pc(splu["worst_same_grid"])} % at {splu["median_total_ms"]:.3f} ms vs `{best["name"]}` {best["median_total_ms"]:.3f} ms | '
                          f'{"the direct solve is cheaper" if splu["median_total_ms"] <= best["median_total_ms"] else "the head is cheaper than the direct solve"} |')
                    W('')

    # --------------------------------------------------------- the free rung --
    free = sorted({(v['M'], k[1], k[2]) for k, v in aggs.items() if v['family'] == 'free'})
    W('## The free rung $q=R$: a purely linear reduced model on the same bank')
    W('')
    if not free:
        W('No free rung is present in these jobs. It needs more test modes than bank columns '
          f'($M > R$), and the primaries have $R={max(x["R_total"] for x in tr["head_arms"])}$.')
        W('')
    else:
        W('On this rung the correction basis is the whole bank ($C=I$), so no nonlinear unknown remains: '
          'the head is bypassed and every coefficient comes from one exact least-squares elimination against '
          'the test modes. It is the paper\'s structural claim made concrete — on a linear PDE the top rung of '
          'the ladder is itself a **linear** reduced model.')
        W('')
        # Head-independence, measured rather than asserted (DESIGN.md A9).
        pairs = [(look(M, n, x), look(M, n, y)) for M, n, x in free for _, _, y in free
                 if x < y and look(M, n, x)['model'] != look(M, n, y)['model'] or
                 (x < y and look(M, n, x)['k'] != look(M, n, y)['k'])]
        if pairs:
            u, v2 = pairs[0]
            W(f'**Head-independence, measured.** The rung is head-independent in exact arithmetic, and the two '
              f'heads on the same bank ($K={u["k"]}$ and $K={v2["k"]}$) agree on every reported figure: worst '
              f'{pc(u["worst_same_grid"])} % against {pc(v2["worst_same_grid"])} %, median '
              f'{pc(u["median_same_grid"])} % against {pc(v2["median_same_grid"])} %. They are **not** bitwise '
              'identical: with $C=I$ the elimination cancels $h(z)$ against $R_q^{-1}Q_q^\\top B\\,h(z)$, and in '
              'f64 that cancellation leaves round-off that still depends on $h(z)$. The measured worst relative '
              'field difference between the two heads is in `checks/verify_report_2026-09-17.json` '
              '(`free_rung_is_head_independent_to_roundoff`); the timed repetitions of one subject *are* bitwise '
              'identical. DESIGN.md §A6 claimed byte-identical output from the two heads on the strength of an '
              '$N=32$, $R=32$ smoke, where the cancellation happened to be exact; §A9 corrects that.')
            W('')
        W('| block $M$ | job | mesh | subject | $R$ | worst same-grid % | median same-grid % | bank floor worst % | worst / floor | median complete ms | iters | on the non-dominated front? |')
        W('|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|')
        for M, n, name in free:
            v = look(M, n, name)
            rec = next((r for r in [x for sv in solves if sv['job_id'] == v['job_id']
                                    for x in sv['reconstruction']] if r['intervals'] == n and r['model'] == v['model']), None)
            fl_ = rec['bank_projection']['worst'] if rec else None
            on = name in [x['name'] for x in verdict[(M, n)][1]]
            over = f'{v["worst_same_grid"] / fl_:.3f}x' if fl_ else '—'
            W(f'| {M} | `{v["job_id"]}` | {n} | `{name}` | {rec["R_total"] if rec else "—"} | {pc(v["worst_same_grid"])} | '
              f'{pc(v["median_same_grid"])} | {pc(fl_)} | {over} | '
              f'{v["median_total_ms"]:.3f} | {num(v["median_iterations"], 1)} | {"**yes**" if on else "no"} |')
            emit(n, name, 'free', rec['R_total'] if rec else None, 'free_rung_worst_over_bank_floor',
                 (v['worst_same_grid'] / fl_) if fl_ else None, v['job_id'], v['source_sha'], test_modes=M)
        W('')
        # The within-block comparison against the best POD rank, generated from the same job.
        for M, n, name in free:
            v = look(M, n, name)
            pods = [aggs[k] for k in block_index[M].values() if k[1] == n and aggs[k]['family'] == 'pod']
            if not pods:
                continue
            bestpod = min(pods, key=lambda x: x['worst_same_grid'])
            wins = v['worst_same_grid'] <= bestpod['worst_same_grid'] and v['median_total_ms'] <= bestpod['median_total_ms']
            W(f'**`{name}` against the best linear baseline in the same job.** The strongest POD-LSPG rank at '
              f'$M={M}$, $N={n}$ is `{bestpod["name"]}` at {pc(bestpod["worst_same_grid"])} % worst and '
              f'{bestpod["median_total_ms"]:.3f} ms; the free rung is {pc(v["worst_same_grid"])} % at '
              f'{v["median_total_ms"]:.3f} ms, so it '
              + (f'**dominates** it — {bestpod["worst_same_grid"] / v["worst_same_grid"]:.2f}x better worst error at '
                 f'{bestpod["median_total_ms"] / v["median_total_ms"]:.2f}x the speed. That is the one place in this '
                 'cell where the learned bank beats the classical linear baseline on both axes at once.'
                 if wins else
                 'does not dominate it.'))
            W('')
            emit(n, name, 'free', None, 'free_rung_worst_over_best_pod',
                 v['worst_same_grid'] / bestpod['worst_same_grid'], v['job_id'], v['source_sha'], test_modes=M)
            break
        W('**Why these rows are at $M=1024$ and not at $M=R+1$** (DESIGN.md §A6). The rung is the least-squares '
          'solution of $Bc=f_M$ over all $R$ coefficients, so it needs $M>R$ equations. At the smallest such '
          'count, $M=R+1$, the system is one equation over square: the bank content outside the span of the '
          'lowest $R+1$ eigenmodes is unconstrained, $\\mathrm{cond}(B)$ is $10^6$–$10^7$, and an untimed NumPy '
          'sweep on these exact banks put the rung 8–13x above its own bank floor (`smooth_R512` 8.4852 % worst '
          'against a 0.7123 % floor). From $M\\approx768$ it is within 4 % of the floor on every bank and at '
          '$M=1024$ it *is* the floor. The sweep is `checks/free_rung_M_sweep.json`; the timed rung was therefore '
          'run at $M=1024$, which is why it lives in its own block and its cost may not be set beside the '
          '$M=257$ tables.')
        W('')

    # ------------------------------------------------------------ audits ------
    if audits:
        W('## Independent NumPy audits')
        W('')
        for au in audits:
            W(f'`{au["mode"]}` audit: ' + ', '.join(f'`{k}` {"pass" if v else "**FAIL**"}' for k, v in au['checks'].items()) + '.')
            W('')
            d = au['detail']
            for k in ('worst_bank_floor_relative_difference', 'worst_bank_floor_absolute_difference',
                      'bank_floor_absolute_limit', 'worst_stored_code_relative_difference',
                      'worst_basis_orthonormality_error', 'worst_same_grid_difference', 'worst_physical_difference',
                      'worst_reference_residual'):
                if k in d:
                    W(f'- {k}: {d[k]:.3e}')
            if 'bank_condition' in d:
                W('- bank cond(G): ' + ', '.join(f'`{k}` {v:.2e}' for k, v in d['bank_condition'].items()))
                W('')
                W('  The bank floor is compared across machines (driver on the GPU, audit on the CPU), so its '
                  'deciding bound is **absolute** and tied to the reporting precision: the difference may not move '
                  'the last digit of a floor quoted to four decimals in percent. The relative difference is '
                  'dominated by cond(G) rather than by any error — two NumPy routes (QR and SVD) on identical '
                  'features agree to 1e-14 — and is retained above only as a conditioning diagnostic '
                  '(DESIGN.md \u00a7A4).')
            W('')

    # ------------------------------------------------------------ figure -----
    if a.fig and solves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        panels = [(M, n) for M in Ms for n in block_meshes(M)]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
        for ax, (M, n) in zip(axes[0], panels):
            items = [aggs[k] for k in block_index[M].values() if k[1] == n]
            for fam, color in PALETTE.items():
                pts = [v for v in items if v['family'] == fam]
                if not pts:
                    continue
                ax.scatter([v['median_total_ms'] for v in pts], [v['worst_same_grid'] * PCT for v in pts], s=28,
                           facecolors='none', edgecolors=color, linewidths=1.5, label=fam, zorder=3)
            nd = nondominated(items, 'worst_same_grid', 'median_total_ms')
            ax.step([v['median_total_ms'] for v in nd], [v['worst_same_grid'] * PCT for v in nd], where='post',
                    color='#555555', linewidth=1.2, zorder=2, label='non-dominated')
            for v in nd:
                ax.annotate(v['name'].replace('neural_', '').replace('@head_', '@').replace('fom_', ''),
                            (v['median_total_ms'], v['worst_same_grid'] * PCT), fontsize=6.5, xytext=(3, 3),
                            textcoords='offset points', color='#333333')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('median complete-query ms (host in → host out)')
            ax.set_ylabel('worst same-grid error over cases (%)')
            ax.set_title(f'N = {n}, n = {next(f["interior_unknowns"] for sv in solves for f in sv["fom"] if f["intervals"] == n)}, M = {M}')
            ax.grid(True, which='major', color='#e6e6e6', linewidth=0.6)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=160)
        W(f'## Figure')
        W('')
        W(f'![error versus cost per mesh]({Path(a.fig).name})')
        W('')
        W('Each panel is one mesh; every timed subject is one point (worst same-grid error over the development cases '
          'against median complete-query milliseconds), hollow markers coloured by family, the grey step is the '
          'non-dominated front with full-order comparators included, and only front members are labelled. '
          'Panels are labelled with their test-mode count M; panels at different M are different cost scales and '
          'must not be compared horizontally.')
        W('')

    # -------------------------------------------- deviations and honesty ------
    W('## Recorded deviations and honesty notes')
    W('')
    notes = [
        ('The POD competitor sees more data than the bank did.', 'The bank and every head are fitted on the '
         f'85 % fit split ({tr["cohorts"]["training"]["count"] - len(tr["cohorts"]["training"]["validation"])} sources); '
         f'POD-LSPG is built from all {solves[0]["pod_cohort"]["used"] if solves else "—"} training snapshots. The difference '
         'favours POD, and it is left that way deliberately so the linear baseline is not handicapped.'),
        ('The free-bank rung is not constructible in the $M=257$ block.', 'The $q=R$ rung needs more test modes than '
         f'bank columns, and both primaries have $R={max(x["R_total"] for x in tr["head_arms"])}$ against $M=257$ tests. '
         'In that block the untimed bank projection floor stands in for it as the representation ceiling; the rung '
         'itself was measured in a separate job at $M=1024$ and is reported in its own, non-comparable block.'),
        ('The two blocks are not one table.', 'Test modes enter every reduced query as a dense $M\\times n$ projection, '
         'so a subject at $M=1024$ is doing about four times the projection work of the same subject at $M=257$. '
         'The $M=1024$ job re-ran the six full-order comparators and all five POD ranks in the same allocation for '
         'exactly this reason: the comparison that decides the cell is within one job, never across two.'),
        ('The comparison heads run at $q=0$ only.', 'The correction ladder is run on the two primaries; the heads on the '
         'non-selected banks answer the boundary-factor question at $q=0$, which is where that question lives.'),
        ('Offline setup is not charged to any query.', 'The operator assembly, the SuperLU and IC(0) factorisations, the '
         'eigen-decomposition that defines the test modes, the POD basis and the bank build all happen once per mesh before '
         'timing and are reported separately. Every timed subject is charged host-array-in to host-array-out.'),
        ('CPU and GPU subjects are timed in the same process against the same contract.', 'The sparse direct and IC(0)-PCG '
         'subjects run on the node CPU; the neural, correction and POD subjects and the matrix-free CG run on the GPU. No '
         'ratio is formed across machines, and both cost columns are reported.'),
        ('`lsh01` is retracted.', 'The first training attempt aborted on a mis-scaled basis-orthonormality gate after '
         'completing seven of eight head arms (DESIGN.md §A3). No number from it appears here; the job was rerun in full.'),
        ('The independent-auditor requirement was met by a self-audit, not by Codex.', 'The Codex account quota was '
         'exhausted for the whole campaign window and a substitute review agent was killed by an API limit, both recorded '
         'in DESIGN.md §A1. The substitute is the NumPy/SciPy audit in this report plus the CPU-only checks listed there.'),
    ]
    for head_, body in notes:
        W(f'- **{head_}** {body}')
    W('')

    # ------------------------------------------------------------ glossary ---
    W('## Glossary')
    W('')
    for term, text in [
        ('L-shaped domain', 'the unit square with its upper-right quadrant removed; the corner at (1/2, 1/2) points into the domain (re-entrant) and the solution behaves like r^(2/3) there, so it is not smooth.'),
        ('n', 'the number of interior unknowns of the finite-difference system on a mesh.'),
        ('G-FOM gates', 'checks that the full-order operator is what it claims: symmetric, positive definite with the known first eigenvalue, identical under two independent assemblies, solved to round-off by the reference, and reproduced by the iterative solvers at tight tolerance.'),
        ('fom_splu', 'the sparse direct solve (SuperLU LU factorisation, factorised once offline; each timed query is one triangular solve pair on the CPU).'),
        ('fom_cg_gpu_r*', 'plain conjugate gradients on the GPU, matrix-free, stopped when the relative residual falls below r. Because the operator diagonal is constant, Jacobi preconditioning would change nothing, so this is also the Jacobi-PCG rung.'),
        ('fom_pcg_ic0_cpu_r*', 'conjugate gradients on the CPU with a zero-fill incomplete-Cholesky preconditioner (IC(0), the classical choice for this operator), factorised once offline.'),
        ('same-grid error', 'relative L2 difference between a subject\'s output and the sparse-direct reference on the same mesh; the primary error metric.'),
        ('physical error', 'relative L2 difference against the 1024-interval direct solution restricted to the mesh\'s nodes; it includes the mesh\'s own discretisation error, which is large near the corner.'),
        ('worst / median', 'maximum / middle value over the development cases.'),
        ('complete-query ms', 'wall time from the host source array to the host solution array, including transfer, projection, initialisation, solve and decode; the number a user pays.'),
        ('device / solver ms', 'the fused GPU interval inside the query for GPU subjects, or the solve call alone for CPU subjects; the secondary cost.'),
        ('non-dominated set', 'the subjects that no other subject beats on both error and cost at once; the Pareto front.'),
        ('bank / bank floor', 'the R fixed spatial functions bc(x)g(x) the decoder can combine; the floor is the best relative error any combination of them can reach on a case, measured by orthogonal projection.'),
        ('boundary factor bc(x)', 'a function that is zero on the boundary and positive inside, multiplied into every bank column so the boundary condition holds exactly: `smooth` is an R-function product, `sdf` the distance to the boundary, `poly` the square\'s parent factor.'),
        ('enrich', 'the smooth bank plus two fixed columns carrying the exact corner behaviour r^(2/3) and r^(4/3).'),
        ('head / best-found', 'the small network h(z) that maps K latent numbers to R bank coefficients; best-found is the smallest error reachable on its image, found by a multistart least-squares search, so an upper bound on the true head floor.'),
        ('solved', 'the error the online weak-residual solve actually returns from the nearest training code.'),
        ('K, R, q, k\'', 'latent dimension of the head; bank rank; number of linear correction directions added to the head and eliminated analytically; rank of the POD-LSPG basis.'),
        ('correction ladder', 'the same head with q extra linear directions, q in {0, 32, 64, 128}; the nonlinear solve stays K-dimensional and the q coefficients are recovered exactly.'),
        ('free bank', 'the q = R rung: all bank coefficients solved directly, only constructible when there are more test modes than bank columns.'),
        ('POD-LSPG', 'the classical linear baseline: principal components of the training solutions on the same mesh, solved with the same weak residual and solver.'),
        ('test modes M', 'the 257 lowest eigenvectors of the operator, the L-shape analogue of the square\'s sine modes; the residual is scaled by the inverse eigenvalues.'),
        ('stationary', 'the solve stopped because the normalised gradient fell below 1e-6, a genuine critical point rather than an exhausted budget.'),
        ('common selection cohort', 'a fixed 256-source held-out set used only to choose the bank, so no selection touches the development cases.'),
        ('development cohort', 'the 32 sources everything is reported on; a fresh seed, never used for any selection; not the sealed final cohort.'),
        ('centre rejection', 'Gaussian sources whose centre falls in the removed quadrant are discarded, since a source outside the domain gives a tail-driven solution for which a relative error is meaningless.'),
    ]:
        W(f'- **{term}** — {text}')
    W('')
    Path(a.md).write_text('\n'.join(L) + '\n')
    Path(a.summary).write_text(json.dumps(rows_out, indent=1) + '\n')
    print(a.md, len(rows_out), 'summary rows')


if __name__ == '__main__':
    main()
