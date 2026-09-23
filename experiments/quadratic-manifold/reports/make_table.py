"""Generate this lane's table and summary.json from the audit JSON alone.

    python reports/make_table.py <audit.json> --out-dir reports [--stem 2026-09-22-quadratic-manifold]

No number is typed by hand anywhere in this lane's report: every cell here is read from the
audit, which itself recomputed every error in NumPy from the saved fields. The script

* applies the paper's full-order rule per row and per timing scope -- the fastest tested
  full-order setting whose worst evolved same-grid error is at or below the row's;
* writes the r ladder beside the NM-ROM and POD-LSPG rows of the SAME job;
* answers DESIGN.md section 1's three questions from the rows, so the prose cannot drift;
* refuses to run at all if the audit's `failed` list is non-empty (DESIGN section 7).

`fom_rule` is `ops-timing-panel/reports/generate_ops_panel.py`'s, unchanged, so this lane's
speedups mean exactly what that lane's and the paper's mean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FAMILY_LABEL = {'rom': 'NM-ROM', 'fast': 'NM-ROM (fast kernel)', 'pod': 'POD-LSPG',
                'qman': 'quadratic manifold', 'free': 'free bank', 'fom': 'FOM'}
ORDER = {'qman': 0, 'rom': 1, 'fast': 2, 'pod': 3, 'free': 4, 'fom': 5}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fom_rule(rows, arm, scope):
    """Fastest tested FOM whose worst evolved error <= this arm's, on `scope`."""
    key = 'median_gpu_ms' if scope == 'gpu' else 'median_host_ms'
    cands = [f for f in rows if f['family'] == 'fom'
             and f['worst_evolved_percent'] <= arm['worst_evolved_percent'] + 1e-12]
    if not cands:
        return None, None
    best = min(cands, key=lambda f: f[key])
    return best['arm'], best[key] / arm[key]


def cell(v, n=3):
    return '—' if v is None else (f'{v:.{n}f}' if isinstance(v, float) else str(v))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--stem', default='quadratic-manifold')
    a = p.parse_args()
    d = json.loads(Path(a.audit).read_text())
    if d['failed']:
        raise SystemExit(f"audit has failed gates, the job is not accepted: {d['failed']}")
    rows = d['arms']
    for r in rows:
        for scope in ('gpu', 'host'):
            # A full-order row is not compared with itself.
            comp, ratio = (None, None) if r['family'] == 'fom' else fom_rule(rows, r, scope)
            r[f'fom_{scope}'], r[f'speedup_{scope}'] = comp, ratio
    rows.sort(key=lambda r: (ORDER.get(r['family'], 9), r['k'] if r['k'] is not None else -1,
                             r['q'] if r['q'] is not None else -1, r['arm']))

    lines = ['| arm | family | solved unknowns | trial-basis columns | quadratic terms P | M | '
             'worst evolved % | median evolved % | worst all-times % | GPU-query ms | complete-query ms | '
             'FOM by the rule (GPU) | speedup (GPU) | speedup (complete) | converged |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(
            '| `{arm}` | {fam} | {dim} | {bc} | {P} | {M} | {we} | {me} | {wa} | {g} | {h} | {c} | {sg} | {sh} | {cv} |'.format(
                arm=r['arm'], fam=FAMILY_LABEL.get(r['family'], r['family']),
                dim=cell(r['solved_dimension']), bc=cell(r.get('bank_columns')),
                P=cell(r.get('quadratic_terms')), M=cell(r['M']),
                we=cell(r['worst_evolved_percent'], 4), me=cell(r['median_evolved_percent'], 4),
                wa=cell(r['worst_all_times_percent'], 4),
                g=cell(r['median_gpu_ms']), h=cell(r['median_host_ms']),
                c=(f"`{r['fom_gpu']}`" if r['fom_gpu'] else
                   ('— (is a FOM)' if r['family'] == 'fom' else 'none at least as accurate')),
                sg=cell(r['speedup_gpu']) + ('×' if r['speedup_gpu'] is not None else ''),
                sh=cell(r['speedup_host']) + ('×' if r['speedup_host'] is not None else ''),
                cv=('—' if r['family'] == 'fom' else ('yes' if r['converged_design5'] else '**no**'))))
    table = '\n'.join(lines)

    # ------------------------------------------------ the three questions ----
    qm = {(r['k'], r['variant']): r for r in rows if r['family'] == 'qman'}
    pod = {r['k']: r for r in rows if r['family'] == 'pod'}
    ranks = sorted({k for k, _ in qm})
    ladder = []
    for k in ranks:
        q_, l_ = qm.get((k, 'quad')), qm.get((k, 'lin'))
        ladder.append(dict(
            rank=k,
            quad_arm=q_ and q_['arm'], lin_arm=l_ and l_['arm'], pod_arm=(pod.get(k) or {}).get('arm'),
            quadratic_terms=q_ and q_.get('quadratic_terms'), bank_columns=q_ and q_.get('bank_columns'),
            ridge=q_ and q_.get('qman_ridge'), heldout_relative=q_ and q_.get('qman_heldout_relative'),
            snapshot_relative_linear_only=q_ and q_.get('qman_snapshot_relative_linear_only'),
            snapshot_relative_with_quadratic=q_ and q_.get('qman_snapshot_relative_with_quadratic'),
            worst_evolved_quad=q_ and q_['worst_evolved_percent'],
            worst_evolved_lin=l_ and l_['worst_evolved_percent'],
            worst_evolved_pod=(pod.get(k) or {}).get('worst_evolved_percent'),
            quadratic_gain=(l_['worst_evolved_percent'] / q_['worst_evolved_percent']) if q_ and l_ else None,
            gpu_ms_quad=q_ and q_['median_gpu_ms'], gpu_ms_lin=l_ and l_['median_gpu_ms'],
            gpu_ms_pod=(pod.get(k) or {}).get('median_gpu_ms'),
            quadratic_cost=(q_['median_gpu_ms'] / l_['median_gpu_ms']) if q_ and l_ else None,
            converged_quad=q_ and q_['converged_design5'], converged_lin=l_ and l_['converged_design5']))

    adm = [r for r in rows if r['family'] != 'fom' and r['admissible']]
    best_qman = min((r for r in adm if r['family'] == 'qman'), key=lambda r: r['worst_evolved_percent'], default=None)
    nm = [r for r in adm if r['family'] in ('rom', 'fast')]
    best_nm = min(nm, key=lambda r: r['worst_evolved_percent'], default=None)
    fast_nm = min(nm, key=lambda r: r['median_gpu_ms'], default=None)

    def matched(fam, dim):
        c = [r for r in rows if r['family'] in fam and r['solved_dimension'] == dim]
        return c[0] if c else None

    answers = dict(
        q1_accuracy_at_matched_dimension=dict(
            note=('worst evolved % of the best admissible quadratic-manifold arm against the best and the '
                  'cheapest admissible NM-ROM arm of the SAME job, and the NM-ROM arm nearest it in '
                  'solved unknowns'),
            best_qman=best_qman and dict(arm=best_qman['arm'], solved_dimension=best_qman['solved_dimension'],
                                         worst_evolved_percent=best_qman['worst_evolved_percent'],
                                         median_gpu_ms=best_qman['median_gpu_ms']),
            best_nmrom=best_nm and dict(arm=best_nm['arm'], solved_dimension=best_nm['solved_dimension'],
                                        worst_evolved_percent=best_nm['worst_evolved_percent'],
                                        median_gpu_ms=best_nm['median_gpu_ms']),
            cheapest_nmrom=fast_nm and dict(arm=fast_nm['arm'], solved_dimension=fast_nm['solved_dimension'],
                                            worst_evolved_percent=fast_nm['worst_evolved_percent'],
                                            median_gpu_ms=fast_nm['median_gpu_ms']),
            qman_over_best_nmrom_error=(best_qman['worst_evolved_percent'] / best_nm['worst_evolved_percent']
                                        if best_qman and best_nm else None),
            between_pod_and_nmrom=[dict(
                rank=e['rank'], qman=e['worst_evolved_quad'], pod=e['worst_evolved_pod'],
                beats_pod_at_same_rank=(None if e['worst_evolved_quad'] is None or e['worst_evolved_pod'] is None
                                        else bool(e['worst_evolved_quad'] < e['worst_evolved_pod'])))
                for e in ladder]),
        q2_online_cost=dict(
            note=('median GPU-query ms of the quadratic arm over its own linear control at the same r '
                  '(`quadratic_cost`), and over POD-LSPG at the same rank'),
            per_rank=[dict(rank=e['rank'], gpu_ms_quad=e['gpu_ms_quad'], gpu_ms_lin=e['gpu_ms_lin'],
                           gpu_ms_pod=e['gpu_ms_pod'], quad_over_lin=e['quadratic_cost'],
                           quad_over_pod=(e['gpu_ms_quad'] / e['gpu_ms_pod']
                                          if e['gpu_ms_quad'] and e['gpu_ms_pod'] else None),
                           bank_columns=e['bank_columns'], quadratic_terms=e['quadratic_terms'])
                      for e in ladder]),
        q3_unknowns_for_the_same_error=dict(
            note=('for each admissible quadratic-manifold arm, the cheapest admissible NM-ROM arm at least '
                  'as accurate, and its solved unknowns against the quadratic arm\'s'),
            pairs=[dict(qman=r['arm'], qman_unknowns=r['solved_dimension'],
                        qman_worst_evolved_percent=r['worst_evolved_percent'],
                        nmrom_at_least_as_accurate=(
                            lambda c: c and dict(arm=c['arm'], unknowns=c['solved_dimension'],
                                                 worst_evolved_percent=c['worst_evolved_percent'],
                                                 median_gpu_ms=c['median_gpu_ms']))(
                            min((x for x in nm if x['worst_evolved_percent'] <= r['worst_evolved_percent'] + 1e-12),
                                key=lambda x: x['median_gpu_ms'], default=None)),
                        pod_at_least_as_accurate=(
                            lambda c: c and dict(arm=c['arm'], unknowns=c['solved_dimension'],
                                                 worst_evolved_percent=c['worst_evolved_percent'],
                                                 median_gpu_ms=c['median_gpu_ms']))(
                            min((x for x in rows if x['family'] == 'pod' and x['admissible']
                                 and x['worst_evolved_percent'] <= r['worst_evolved_percent'] + 1e-12),
                                key=lambda x: x['median_gpu_ms'], default=None)))
                   for r in rows if r['family'] == 'qman' and r['admissible']]),
        matched_dimension_triples=[dict(
            dimension=k,
            qman=(qm.get((k, 'quad')) or {}).get('worst_evolved_percent'),
            qman_lin=(qm.get((k, 'lin')) or {}).get('worst_evolved_percent'),
            pod=(pod.get(k) or {}).get('worst_evolved_percent'),
            nmrom=(matched(('rom', 'fast'), k) or {}).get('worst_evolved_percent')) for k in ranks])

    summary = dict(
        lane='quadratic-manifold', attempt=d['attempt'], job_id=d['job_id'], commit=d['commit'], gpu=d['gpu'],
        intervals=d['intervals'], dt=d['dt'], K=d['K'], R=d['R'], elapsed_seconds=d['elapsed_seconds'],
        audit=str(Path(a.audit).name), audit_sha256=sha(a.audit), result_sha256=d.get('result_sha256'),
        failed_gates=d['failed'],
        fom_rule='fastest tested FOM setting whose worst evolved same-grid error <= the row, per timing scope',
        quadratic_manifold_fits=d.get('quadratic_manifold'),
        ladder=ladder, answers=answers,
        fom_discretisation_error_percent=d.get('fom_discretisation_error_percent'),
        rows=[{k: r[k] for k in (
            'arm', 'kind', 'family', 'variant', 'q', 'k', 'M', 'solved_dimension', 'bank_columns',
            'quadratic_terms', 'qman_ridge', 'qman_heldout_relative', 'qman_snapshot_relative_linear_only',
            'qman_snapshot_relative_with_quadratic', 'worst_evolved_percent', 'median_evolved_percent',
            'worst_all_times_percent', 'median_all_times_percent', 'worst_reference_percent',
            'worst_t0_compression_percent', 'median_gpu_ms', 'median_host_ms', 'per_case_evolved_percent',
            'median_iterations', 'max_iterations', 'total_budget_exits', 'converged_design5', 'admissible',
            'fom_gpu', 'speedup_gpu', 'fom_host', 'speedup_host') if k in r} for r in rows])

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (od / f'{a.stem}-table.md').write_text(table + '\n')
    print(json.dumps(dict(rows=len(rows), ranks=ranks, out=str(od / f'{a.stem}-table.md')), indent=1))


if __name__ == '__main__':
    main()
