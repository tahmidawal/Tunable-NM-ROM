"""Independent NumPy audit of one burgers-compare-hires panel job, and the summary every table is built from.

    python audit_cmp.py <output dir> --operators operators-1024.json --out summary.json [--job-log ...]

NumPy only (no JAX, no torch): every error is recomputed from the saved FULL fields (`fields/full_<arm>_case<c>.npy`)
against the job's own same-grid reference (`full_fft_tight_case<c>.npy`, itself hash-checked against the truth
solve), and every number the job printed is checked against the recomputation. Timings are read from the job's
records (a timing cannot be recomputed); the FOM rule, the speedups and every gate are computed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REL_TOL = 1e-9


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def order_effect(rows, bar=.05, slow_seconds=1., factor=3., min_n=5):
    """rows: timed invocations of ONE randomised panel in execution order, each with name, case, gpu_seconds.
    Every sample is normalised by its arm's per-case median; per arm, the median normalised time after a slow
    predecessor is compared with that after a fast one. Split (a): predecessor >= `slow_seconds` (repanel's
    definition); split (b): predecessor >= `factor` x this arm's own per-case median."""
    med = {}
    for r in rows:
        med.setdefault((r['name'], r['case']), []).append(r['gpu_seconds'])
    med = {k: float(np.median(v)) for k, v in med.items()}
    out = {}
    for split in ('a', 'b'):
        groups = {}
        for prev, r in zip(rows[:-1], rows[1:]):
            m = med[(r['name'], r['case'])]
            slow = prev['gpu_seconds'] >= (slow_seconds if split == 'a' else factor * m)
            groups.setdefault(r['name'], ([], []))[0 if slow else 1].append(r['gpu_seconds'] / m)
        per = []
        for name, (s, f) in sorted(groups.items()):
            judged = len(s) >= min_n and len(f) >= min_n
            gap = float(np.median(s) / np.median(f) - 1) if (s and f) else None
            per.append(dict(arm=name, n_after_slow=len(s), n_after_fast=len(f), gap=gap, judged=judged,
                            passed=(abs(gap) <= bar) if judged else None))
        judged = [x for x in per if x['judged']]
        out[split] = dict(definition=('predecessor >= %g s' % slow_seconds) if split == 'a' else
                          ('predecessor >= %g x the arm\'s own per-case median' % factor),
                          judged_arms=len(judged), worst_gap=(max(abs(x['gap']) for x in judged) if judged else None),
                          passed=(all(x['passed'] for x in judged) if judged else None), per_arm=per)
    verdicts = [out[s]['passed'] for s in ('a', 'b') if out[s]['passed'] is not None]
    return dict(bar=bar, min_samples=min_n, passed=(all(verdicts) if verdicts else None), splits=out)


def errors(f, truth):
    n0 = float(np.linalg.norm(truth[0]))
    per = [float(np.linalg.norm(a - b)) / n0 for a, b in zip(f, truth)]
    return per


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    p.add_argument('--operators', default=None)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.output)
    rep = json.loads((root / 'result.json').read_text())
    cfg = rep['config']
    fd = root / 'fields'
    ncase = len(rep['physical_cases'])
    gates, notes = {}, []

    # ---------------------------------------------------------------- truth --
    truth = {}
    for c in range(ncase):
        f = np.load(fd / f'full_fft_tight_case{c}.npy')
        truth[c] = f
    gates['truth_fields_are_the_reference_solve'] = dict(
        passed=all(sha(truth[c]) == rep['phases']['truth'][c]['field_sha256'] for c in range(ncase)))
    gates['truth_converged'] = dict(passed=all(t['converged'] for t in rep['phases']['truth']))

    # ------------------------------------------------ every arm's error, recomputed --
    arms = {}
    mism = []
    for q in rep['quick']:
        name, c = q['name'], q['case']
        f = np.load(fd / f'full_{name}_case{c}.npy')
        if sha(f) != q['field_sha256']:
            mism.append(dict(arm=name, case=c, what='field sha256'))
        per = errors(f, truth[c])
        ev = max(per[1:])
        if abs(ev - q['same_grid_evolved']) > 1e-13 + REL_TOL * abs(q['same_grid_evolved']):
            mism.append(dict(arm=name, case=c, what='evolved error', job=q['same_grid_evolved'], audit=ev))
        a_ = arms.setdefault(name, dict(name=name, family=q['family'], phase=q['phase'], evolved=[None] * ncase,
                                        all_times=[None] * ncase, stalled=0))
        a_['evolved'][c] = ev
        a_['all_times'][c] = max(per)
        a_['stalled'] += int(q.get('stalled_exits', q.get('stalled_steps', 0)) or 0)
    gates['errors_recomputed_from_full_fields'] = dict(passed=not mism, mismatches=mism[:20], tolerance=REL_TOL)

    # the control: the same comparison must flag a field perturbed by 1e-3 relative
    q0 = next(q for q in rep['quick'] if q['family'] != 'fom')
    f = np.load(fd / f"full_{q0['name']}_case{q0['case']}.npy")
    fp = f + 1e-3 * np.linalg.norm(f) / np.sqrt(f.size) * np.random.default_rng(0).standard_normal(f.shape)
    evp = max(errors(fp, truth[q0['case']])[1:])
    gates['control_perturbed_field_is_flagged'] = dict(
        passed=bool(abs(evp - q0['same_grid_evolved']) > 1e-13 + REL_TOL * abs(q0['same_grid_evolved']) and
                    sha(fp) != q0['field_sha256']),
        arm=q0['name'], case=q0['case'], job=q0['same_grid_evolved'], perturbed=evp)

    # ------------------------------------------------------------- timings --
    inv = rep['invocations']
    by = {}
    for r in inv:
        by.setdefault((r['name'], r['phase']), []).append(r)
    for name, a_ in arms.items():
        ph = a_['phase']
        rows = by.get((name, ph), [])
        a_['reps_per_case'] = min([sum(1 for r in rows if r['case'] == c) for c in range(ncase)]) if rows else 0
        a_['gpu_ms'] = float(np.median([r['gpu_seconds'] for r in rows]) * 1e3) if rows else None
        a_['complete_ms'] = float(np.median([r['host_seconds'] for r in rows]) * 1e3) if rows else None
        a_['timed_stalled'] = int(sum(int(r.get('stalled_exits', r.get('stalled_steps', 0)) or 0) for r in rows))
        a_['timed_identical_to_quick'] = all(r['identical_to_quick'] for r in rows) if rows else None
        brow = by.get((name, 'B'), [])
        a_['bracket_gpu_ms'] = float(np.median([r['gpu_seconds'] for r in brow]) * 1e3) if brow else None
        # the job's printed timed-row error must also be the recomputed one (same field hash)
        for r in rows:
            if abs(r['same_grid_evolved'] - a_['evolved'][r['case']]) > 1e-13 + REL_TOL * a_['evolved'][r['case']]:
                mism.append(dict(arm=name, case=r['case'], what='timed-row error'))
    setup = {s['arm']: s for s in rep['arm_setup']}

    # ------------------------------------------------------------ operators --
    ops_meta = json.loads(Path(a.operators).read_text())['operators'] if a.operators else []
    op_gate = []
    for op in ops_meta:
        tj = root / 'optiming' / f"{op['name']}-timing.json"
        if not tj.exists():
            notes.append(f"operator {op['name']}: no timing record (failed or not run)")
            op_gate.append(dict(arm=op['name'], present=False, passed=False))
            continue
        t = json.loads(tj.read_text())
        ev, al, ok_t0, ok_sha = [None] * ncase, [None] * ncase, True, True
        for cr in t['cases']:
            c = int(cr['case_index'])
            f = np.load(fd / f"full_{op['name']}_case{c}.npy")
            ok_sha &= sha(f) == cr['field_sha256']
            ok_t0 &= bool(np.array_equal(f[0], truth[c][0]))
            per = errors(f, truth[c])
            ev[c], al[c] = max(per[1:]), max(per)
        dev = np.load(root / 'optiming' / f"{op['name']}-timing.npz")['device']
        hostt = np.load(root / 'optiming' / f"{op['name']}-timing.npz")['host']
        same_gpu = (t.get('gpu_name', '').replace('NVIDIA ', '').strip().lower() in
                    rep['gpu'].replace('NVIDIA ', '').strip().lower() or
                    rep['gpu'].replace('NVIDIA ', '').strip().lower() in t.get('gpu_name', '').lower())
        arms[op['name']] = dict(name=op['name'], family=op['family'], phase='O', evolved=ev, all_times=al, stalled=0,
                                reps_per_case=int(dev.shape[1]), gpu_ms=float(np.median(dev) * 1e3),
                                complete_ms=float(np.median(dev + hostt) * 1e3), role=op.get('role'),
                                epochs=op.get('epochs'), stop_reason=op.get('stop_reason'),
                                trained_at=op.get('trained_at'), parameters=t.get('real_parameter_count'),
                                checkpoint_sha256=t.get('checkpoint_sha256'))
        op_gate.append(dict(arm=op['name'], present=True, t0_returned_exactly=ok_t0, field_hash_matches=ok_sha,
                            same_gpu=bool(same_gpu), gpu=t.get('gpu_name'), cases=len(t['cases']),
                            checkpoint_matches_record=(t.get('checkpoint_sha256') == op['sha256']),
                            passed=bool(ok_t0 and ok_sha and same_gpu and len(t['cases']) == ncase and
                                        t.get('checkpoint_sha256') == op['sha256'] and dev.shape[1] >= 5)))
    if ops_meta:
        gates['operator_integrity'] = dict(passed=all(x['passed'] for x in op_gate), per_arm=op_gate)

    for a_ in arms.values():
        a_['worst_evolved_percent'] = 100 * max(a_['evolved'])
        a_['median_evolved_percent'] = 100 * float(np.median(a_['evolved']))
        a_['worst_all_times_percent'] = 100 * max(a_['all_times'])
        s_ = setup.get(a_['name'], {})
        a_['unknowns'] = s_.get('unknowns')
        for k in ('q', 'M', 'm', 'k', 'rule', 'gtol', 'kernel', 'rule_status', 'bank_columns'):
            if k in s_:
                a_[k] = s_[k]

    # ----------------------------------------------------------------- FOM rule --
    L = int(rep['intervals'])
    foms = {n: a_ for n, a_ in arms.items() if a_['family'] == 'fom'}
    subsets = cfg['fom_subsets']

    def pick(worst, scope, subset):
        cand = [f_ for n, f_ in foms.items() if n in subsets[subset] and f_[scope] is not None
                and f_['worst_evolved_percent'] <= worst]
        return min(cand, key=lambda f_: f_[scope]) if cand else None

    for a_ in arms.values():
        if a_['family'] == 'fom':
            a_['unknowns'] = (L - 1) ** 2
            continue
        for scope, key in (('gpu_ms', 'gpu'), ('complete_ms', 'complete')):
            for subset in ('full', 'b_panel', 'lean'):
                f_ = pick(a_['worst_evolved_percent'], scope, subset)
                tag = f'{key}_{subset}'
                a_[f'fom_{tag}'] = f_['name'] if f_ else None
                a_[f'fom_ms_{tag}'] = f_[scope] if f_ else None
                a_[f'fom_worst_{tag}'] = f_['worst_evolved_percent'] if f_ else None
                a_[f'speedup_{tag}'] = (f_[scope] / a_[scope]) if (f_ and a_[scope]) else None

    # ------------------------------------------------------------------ gates --
    Frows = [r for r in inv if r['phase'] == 'F']
    gates['no_order_effect_between_arms'] = order_effect(Frows)
    drift = []
    comps = {a_.get('fom_gpu_full') for a_ in arms.values() if a_['family'] != 'fom'} - {None}
    for name in sorted(comps | {v for v in cfg['roles'].values() if v in arms and arms[v]['phase'] == 'F'}):
        a_ = arms[name]
        if a_.get('bracket_gpu_ms') and a_.get('gpu_ms'):
            d = a_['bracket_gpu_ms'] / a_['gpu_ms'] - 1
            drift.append(dict(arm=name, phase_F_ms=a_['gpu_ms'], phase_B_ms=a_['bracket_gpu_ms'], drift=d,
                              passed=abs(d) <= .10))
    gates['bracket_drift'] = dict(bar=.10, passed=all(x['passed'] for x in drift) if drift else None, per_arm=drift)
    gates['five_retained_repetitions_everywhere'] = dict(
        passed=all(a_['reps_per_case'] >= cfg['required_reps'] for a_ in arms.values() if a_['phase'] in ('F', 'S', 'O')),
        minimum=min(a_['reps_per_case'] for a_ in arms.values() if a_['phase'] in ('F', 'S', 'O')))
    gates['repetition_output_identical'] = dict(
        passed=all(a_.get('timed_identical_to_quick') for a_ in arms.values() if a_['phase'] in ('F', 'S')))
    roles = [v for v in cfg['roles'].values() if v in arms]
    gates['no_role_arm_stalls'] = dict(passed=all(arms[r]['stalled'] == 0 and arms[r].get('timed_stalled', 0) == 0
                                                  for r in roles), roles=roles,
                                       missing=[v for v in cfg['roles'].values() if v not in arms])
    gates['timed_rows_match_recomputed_errors'] = dict(passed=not any(m['what'] == 'timed-row error' for m in mism))
    gp = rep['gates'].get('grid_arm_parity')
    if gp:
        rec = []
        for pr in gp['pairs']:
            w = max(float(np.linalg.norm(np.load(fd / f"full_{pr['arm']}_case{c}.npy") -
                                         np.load(fd / f"full_{pr['twin']}_case{c}.npy")) /
                          np.linalg.norm(np.load(fd / f"full_{pr['twin']}_case{c}.npy"))) for c in range(ncase))
            rec.append(dict(arm=pr['arm'], twin=pr['twin'], worst_relative=w, integers_identical=pr['integers_identical'],
                            passed=bool(w <= 1e-9 and pr['integers_identical'])))
        gates['grid_arm_parity'] = dict(passed=all(x['passed'] for x in rec), pairs=rec, bar=1e-9)
    qm = rep.get('quadratic_manifold', [])
    gates['quadratic_manifold_trajectory_split'] = dict(
        passed=all(x['split'].startswith('by trajectory') and x['ridge'] in x['ridge_grid'] for x in qm) if qm else None,
        fits=[dict(rank=x['rank'], ridge=x['ridge'], ridge_at_grid_top=x['ridge'] == max(x['ridge_grid']),
                   heldout=x['heldout_relative'], weight_frobenius_norm=x['weight_frobenius_norm']) for x in qm])
    gates['cohort_matches_b_panel'] = dict(passed=rep['gates']['evaluation_cohort_matches_b_panel']['passed'])
    gates['backend_gpu_x64_highest'] = dict(passed=rep['backend'] == 'gpu' and rep['x64'] and rep['matmul_precision'] == 'highest')
    gates['checkpoint_unchanged'] = dict(passed=rep.get('checkpoint_sha256') == rep.get('checkpoint_sha256_after'))
    gates['job_complete'] = dict(passed=bool(rep.get('complete')))
    gates['phi_free_operator_parity'] = dict(passed=rep['gates']['phi_free_operator_parity']['passed'])
    gates['errors_recomputed_from_full_fields']['passed'] = not mism
    gates['errors_recomputed_from_full_fields']['mismatches'] = mism[:20]
    failed = sorted(k for k, v in gates.items() if v.get('passed') is False)

    summary = dict(attempt=cfg['attempt'], job_id=rep['job_id'], commit=rep['commit'], gpu=rep['gpu'], roles=cfg['roles'],
                   nvidia_smi=rep.get('nvidia_smi'), intervals=L, cohort=rep.get('cohort_name'),
                   cohort_sha256=rep['physical_sha256'], elapsed_seconds=rep.get('elapsed_seconds'),
                   phases={k: v for k, v in rep['phases'].items() if k.endswith('seconds')},
                   dropped=rep['dropped'], rule_status=[dict(arm=n, status=arms[n].get('rule_status'))
                                                         for n in arms if arms[n].get('rule_status')],
                   snapshots=rep.get('snapshots'), quadratic_manifold_fits=gates['quadratic_manifold_trajectory_split']['fits'],
                   error_convention=('same-grid evolved: max over t in {0.05..0.25} of ||u - u_fft_tight|| / ||u_0||, '
                                     'reference solved in this job at this mesh'),
                   fom_rule='fastest tested FOM setting (Phase F timing, same job) whose worst evolved error <= the arm\'s',
                   arms=sorted(arms.values(), key=lambda x: (x['family'], x['name'])),
                   gates=gates, failed_gates=failed, notes=notes)
    Path(a.out).write_text(json.dumps(summary, indent=1, allow_nan=False, default=float) + '\n')
    print('AUDIT', cfg['attempt'], 'arms', len(arms), 'failed gates:', failed)


if __name__ == '__main__':
    main()
