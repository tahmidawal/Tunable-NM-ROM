"""Independent NumPy audit of a b-lowvisc gate attempt.

Imports neither JAX nor the driver.  Every reported number is recomputed from the arrays the
job saved, and where possible by a DIFFERENT route than the one the driver used:

  * errors against the 4096-interval reference and against the same-grid tight solve are
    recomputed from the saved per-invocation field files;
  * the POD projection floors are recomputed from the snapshot Gram's eigen-decomposition
    (`podaudit_<family>.npz`) rather than from the POD modes the driver formed.  With
    `V = U (W / sqrt(w))` and `B = U^T F`, the projection of the truth on the first k modes is
    `c_k = (W_k / sqrt(w_k))^T B`, so the floor is `sqrt(||F||^2 - ||c_k||^2)` without ever
    touching an n-by-k mode matrix.  The driver instead formed `V` and computed `F V`;
    agreement of the two is a real check on both;
  * the POD basis's orthonormality is recomputed as `(W/sqrt(w))^T (Gram W) / sqrt(w)`;
  * medians, frontiers, the leg-(a) ratios and the leg-(b) cost ratios are recomputed from the
    invocation rows.

    python experiments/b-lowvisc/audit_gate.py --attempt lvg01
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-9


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def errors(field, reference):
    """`engines.errors` reimplemented; both arrays are already on the observation grid."""
    field, reference = np.asarray(field), np.asarray(reference)
    n0 = np.linalg.norm(reference[0])
    nr = np.linalg.norm(reference.reshape(len(reference), -1), axis=1)
    er = np.linalg.norm((field - reference).reshape(len(reference), -1), axis=1)
    return dict(fixed_initial_per_time=(er / n0).tolist(),
                current_relative_per_time=(er / np.maximum(nr, 1e-300)).tolist(),
                fixed_initial_max=float(np.max(er / n0)),
                current_relative_max=float(np.max(er / np.maximum(nr, 1e-300))))


def frontier(rows, cost, err):
    return [r['name'] for r in rows
            if not any((s[cost] <= r[cost] and s[err] <= r[err]
                        and (s[cost] < r[cost] or s[err] < r[err])) for s in rows if s is not r)]


def check(results, name, ok, **detail):
    results.append(dict(check=name, passed=bool(ok), **detail))
    return bool(ok)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', required=True)
    p.add_argument('--root', default=None, help='directory holding result.json and the arrays')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    base = Path(a.root) if a.root else ROOT / 'experiments/b-lowvisc/runs' / a.attempt / 'archive/output/gate'
    rep = json.loads((base / 'result.json').read_text())
    cfg = rep['config']
    checks = []
    OBS = int(rep['observation_intervals'])

    check(checks, 'job_complete', rep.get('complete') is True)
    check(checks, 'jax_backend_gpu', rep.get('backend') == 'gpu', backend=rep.get('backend'))
    check(checks, 'x64_and_highest_precision',
          rep.get('x64') is True and rep.get('matmul_precision') == 'highest',
          precision=rep.get('matmul_precision'))
    check(checks, 'no_trained_model_used', rep.get('trained_model_used') is False)
    check(checks, 'final_cohort_unopened', rep.get('final_cohort_unopened') is True)

    # ------------------------------------------------- the family relation, recomputed --
    def draw(seed, count, lo, hi):
        r = np.random.default_rng(seed)
        return np.stack([r.uniform(.15, .85, count), r.uniform(.15, .85, count),
                         r.uniform(.05, .20, count), r.uniform(.5, 2., count),
                         np.exp(r.uniform(np.log(lo), np.log(hi), count))], axis=1)

    fam = {f['name']: f for f in cfg['families']}
    redraw = {}
    for name, f in fam.items():
        redraw[name] = np.concatenate((draw(cfg['eval_seed'], cfg['eval_cases'], f['nu_lo'], f['nu_hi']),
                                       draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'],
                                            f['nu_lo'], f['nu_hi'])))
        got = np.asarray(rep['families'][name]['physical_cases'])
        d = np.max(np.abs(got - redraw[name]) / np.maximum(np.abs(redraw[name]), 1e-300))
        check(checks, f'cohort_redrawn_matches_{name}', d <= 4e-16, max_relative_difference=float(d))
    if 'incumbent' in redraw and 'lowvisc' in redraw:
        hi, lo = redraw['incumbent'], redraw['lowvisc']
        check(checks, 'descriptors_bitwise_identical_across_families',
              bool(np.array_equal(hi[:, :4], lo[:, :4])))
        r = hi[:, 4] / lo[:, 4]
        check(checks, 'viscosity_ratio_is_exactly_ten',
              float(np.max(np.abs(r - 10.) / 10.)) < 1e-13,
              max_relative_deviation=float(np.max(np.abs(r - 10.) / 10.)))

    # ------------------------------------------------------------ reference integrity --
    refs = {}
    for r in rep['reference']:
        z = np.load(base / r['artifact'])
        f = np.asarray(z['fields'])
        refs[(r['family'], r['case'])] = f
        check(checks, f"reference_hash_{r['family']}_{r['case']}", sha_array(f) == r['field_sha256'])
        check(checks, f"reference_residual_{r['family']}_{r['case']}",
              float(np.max(np.asarray(z['residuals']))) < 2e-11,
              max_relative_residual=float(np.max(np.asarray(z['residuals']))))

    # ---------------------------------------------- every invocation error, recomputed --
    truth = {}
    for name, blk in rep['families'].items():
        z = np.load(base / f'truth_{name}.npz')
        truth[name] = np.asarray(z['fields'])
        check(checks, f'truth_hash_{name}', sha_array(truth[name]) == blk['truth_sha256'])
    worst_ref, worst_same, worst_hash = 0., 0., 0
    for row in rep['invocations']:
        f = np.asarray(np.load(base / row['artifact'])['fields'])
        if sha_array(f) != row['field_sha256']:
            worst_hash += 1
        got = errors(f, refs[(row['family'], row['case'])])
        worst_ref = max(worst_ref, abs(got['fixed_initial_max'] - row['error_vs_reference']['fixed_initial_max']))
        if 'error_same_grid' in row:
            t = truth[row['family']][row['case']]
            n = t.shape[-1] - 1
            full = np.zeros((t.shape[0], n + 2, n + 2)) if t.ndim == 3 else None
            # `truth_<family>.npz` holds the INTERIOR; rebuild the padded field to compare
            tf = np.zeros(f.shape)
            tf[:, 1:-1, 1:-1] = truth[row['family']][row['case']].reshape(f.shape[0], OBS - 1, OBS - 1)
            gs = errors(f, tf)
            worst_same = max(worst_same, abs(gs['fixed_initial_max']
                                             - row['error_same_grid']['fixed_initial_max']))
    check(checks, 'invocation_field_hashes', worst_hash == 0, mismatches=worst_hash)
    check(checks, 'errors_vs_reference_recomputed', worst_ref <= TOL, max_absolute_difference=worst_ref)
    check(checks, 'errors_same_grid_recomputed', worst_same <= TOL, max_absolute_difference=worst_same)

    # ------------------------------------------- POD floors by the Gram-eigenvector route --
    ranks = cfg['pod_ranks']
    for name in rep['pod']:
        z = np.load(base / f'podaudit_{name}.npz')
        W, w, GtW, B = (np.asarray(z['gram_eigvecs']), np.asarray(z['gram_eigvals']),
                        np.asarray(z['gram_times_eigvecs']), np.asarray(z['snapshots_times_truth']))
        scale = 1. / np.sqrt(np.clip(w, 1e-300, None))
        VtV = (W * scale[None, :]).T @ (GtW * scale[None, :])
        orth = float(np.max(np.abs(VtV - np.eye(len(w)))))
        check(checks, f'pod_orthonormal_{name}', orth <= 1e-8, max_deviation=orth)
        C = (W * scale[None, :]).T @ B                     # (kmax, cases*times)
        F = truth[name]
        nc, nt, _ = F.shape
        flat = F.reshape(nc * nt, -1)
        tot = np.sum(flat ** 2, axis=1)
        sq = np.cumsum(C ** 2, axis=0)
        n0 = np.linalg.norm(F[:, 0], axis=1)
        worst = 0.
        for k in ranks:
            res = np.sqrt(np.maximum(tot - sq[k - 1], 0.)).reshape(nc, nt)
            got = 100. * float(np.max(res / n0[:, None]))
            worst = max(worst, abs(got - rep['pod'][name]['floors'][str(k)]['worst_all_times_percent']))
        check(checks, f'pod_floors_recomputed_{name}', worst <= 1e-6,
              max_absolute_percentage_point_difference=worst)
        check(checks, f'pod_eigenvalues_nonincreasing_{name}', bool(np.all(np.diff(w) <= 1e-9 * w[0])))

    # ------------------------------------------------------- medians and the frontiers --
    by = {}
    for row in rep['invocations']:
        by.setdefault(row['name'], []).append(row)
    worst_ms = 0.
    summary = {s['name']: s for s in rep['subject_summary']}
    for name, rows in by.items():
        ms = float(np.median(sorted(1e3 * r['gpu_seconds'] for r in rows)))
        worst_ms = max(worst_ms, abs(ms - summary[name]['median_gpu_ms']))
        check(checks, f'reps_present_{name}', len(rows) == cfg['reps'] * len(refs) // len(rep['families']),
              rows=len(rows), expected=cfg['reps'] * len(refs) // len(rep['families']))
    check(checks, 'median_gpu_ms_recomputed', worst_ms <= 1e-9, max_absolute_difference=worst_ms)

    for name, blk in rep['families'].items():
        panel = [s for s in rep['subject_summary'] if s['family'] == name and s['role'] == 'panel'
                 and s['worst_evolved_percent'] is not None]
        check(checks, f'frontier_same_grid_{name}',
              sorted(frontier(panel, 'median_gpu_ms', 'worst_evolved_percent'))
              == sorted(blk['fom_frontier_same_grid']))

    # ------------------------------------------------------------------ the two legs --
    if 'leg_a_pod_degradation' in rep:
        worst = 0.
        for k, v in rep['leg_a_pod_degradation'].items():
            inc = rep['pod']['incumbent']['floors'][k]['worst_all_times_percent']
            low = rep['pod']['lowvisc']['floors'][k]['worst_all_times_percent']
            worst = max(worst, abs(low / inc - v['ratio']))
        check(checks, 'leg_a_ratios_recomputed', worst <= 1e-12, max_absolute_difference=worst)
    if 'leg_b_fom_cost' in rep:
        worst = 0.
        for sname, v in rep['leg_b_fom_cost'].items():
            i = summary[f"incumbent__{sname}__L{rep['intervals']}"]['median_gpu_ms']
            l = summary[f"lowvisc__{sname}__L{rep['intervals']}"]['median_gpu_ms']
            worst = max(worst, abs(l / i - v['cost_ratio']))
        check(checks, 'leg_b_cost_ratios_recomputed', worst <= 1e-12, max_absolute_difference=worst)

    # ------------------------------------------------ the finer-mesh POD probe, replayed --
    # Weaker than the L = cfg['intervals'] check above: the probe saves the truth and its
    # projections but not the Gram pieces, so the floor is recomputed by the same algebra the
    # driver used, from independently re-hashed arrays.  Labelled as such.
    for m, blk in (rep.get('pod_probe') or {}).items():
        if blk.get('failed'):
            checks.append(dict(check=f'pod_probe_L{m}', passed=True, advisory=True,
                               verdict='failed-and-recorded', detail=blk.get('reason')))
            continue
        for name in [k for k in blk if k in rep['families']]:
            z = np.load(base / f'podprobe_L{m}_{name}.npz')
            F, P = np.asarray(z['truth']), np.asarray(z['projections'])
            check(checks, f'pod_probe_truth_hash_L{m}_{name}',
                  sha_array(F) == blk[name]['truth_sha256'])
            nc, nt, _ = F.shape
            tot = np.sum(F.reshape(nc * nt, -1) ** 2, axis=1)
            sq = np.cumsum(P ** 2, axis=1)
            n0 = np.linalg.norm(F[:, 0], axis=1)
            worst = 0.
            for k in ranks:
                res = np.sqrt(np.maximum(tot - sq[:, k - 1], 0.)).reshape(nc, nt)
                got = 100. * float(np.max(res / n0[:, None]))
                worst = max(worst, abs(got - blk[name]['floors'][str(k)]['worst_all_times_percent']))
            check(checks, f'pod_probe_floors_recomputed_L{m}_{name}', worst <= 1e-6,
                  max_absolute_percentage_point_difference=worst, route='same-algebra (weaker)')

    # ------------------------------------------------------------- in-job gate replay --
    # The two hypothesis gates are SCIENCE verdicts, not integrity checks: a failing leg is a
    # result this lane reports (DESIGN.md section 6, F1/F3), not a broken job. They are replayed
    # and recorded, and excluded from the audit's pass/fail.
    HYPOTHESIS = {'leg_a_pod_degrades', 'leg_b_fom_cost_rises'}
    for gname, g in rep['gates'].items():
        if g.get('passed') is None:
            continue
        if gname in HYPOTHESIS:
            checks.append(dict(check=f'hypothesis_gate_{gname}', passed=True, advisory=True,
                               verdict=bool(g['passed']), detail=g))
        else:
            check(checks, f'in_job_gate_{gname}', g['passed'] is True, detail=g)

    failed = [c['check'] for c in checks if not c['passed']]
    outp = Path(a.out) if a.out else base.parent.parent.parent / 'audit.json'
    outp.write_text(json.dumps(dict(attempt=a.attempt, source=str(base), job_id=rep.get('job_id'),
                                    commit=rep.get('commit'), gpu=rep.get('gpu'),
                                    checks=checks, failed=failed,
                                    passed=not failed), indent=2) + '\n')
    print(f'{len(checks)} checks, {len(failed)} failed -> {outp}')
    for f in failed:
        print('  FAILED', f)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
