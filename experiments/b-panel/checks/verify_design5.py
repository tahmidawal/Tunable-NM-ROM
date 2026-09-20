"""Independent verification of the Codex audit's finding 1 (2026-09-19): the convergence flag.

    python checks/verify_design5.py --out checks/design5-verification.json

DESIGN.md §5 defines a converged query by a FIXED threshold: every time step has normalised joint
gradient g <= 1e-6 or exited on the residual rule; the initial fit has g <= 1e-6 or relative
residual <= 1e-10; every exit is regular. `audit_panel.py` (as committed at d2135501, line 180)
evaluated the same rule against each invocation's OWN `gtol`, so the 1e-3 arms were flagged
converged at 1e-3. This script imports neither `audit_panel.py`, the driver nor JAX. It reads the
per-invocation raw quantities (`step_joint_stationarity`, `stop_reasons`, `ic_*`, `gtol`) from the
three accepted jobs' `result.json`, applies §5 as written and as implemented, and recomputes the
admissible counts, the non-dominated sets on all four metric pairs, and the same-job cost ratios
from the audit JSONs' per-arm metrics (which `checks/recheck_headline.py` already reproduced from
the saved fields by a second code path). Nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
JOBS = ('bpn301', 'bpn401', 'bpn203')
DESIGN5_GTOL = 1e-6
IC_RESIDUAL_FLOOR = 1e-10
SLACK = 1 + 1e-7
REDUCED = ('rom', 'fast', 'pod', 'free')
PAIRS = {'gpu_all': ('median_gpu_ms', 'worst_all_times_percent'),
         'gpu_evolved': ('median_gpu_ms', 'worst_evolved_percent'),
         'complete_all': ('median_host_ms', 'worst_all_times_percent'),
         'complete_evolved': ('median_host_ms', 'worst_evolved_percent')}


def converged_under(x, gtol):
    """DESIGN.md §5 (i)-(iii) with the gradient threshold `gtol`."""
    reasons = x['stop_reasons']
    regular = all(r in (1, 2, 4) for r in reasons) and x['ic_reason'] in (1, 2, 4)
    steps = all((g <= gtol * SLACK) or (r == 1) for g, r in zip(x['step_joint_stationarity'], reasons))
    ic = (x['ic_joint_stationarity'] <= gtol * SLACK) or (x['ic_relative_residual'] <= IC_RESIDUAL_FLOOR)
    return bool(regular and steps and ic)


def nondominated(rows, ck, ek):
    pts = [(r['arm'], r[ck], r[ek]) for r in rows if r.get(ck) is not None and r.get(ek) is not None]
    keep = [a for a, c, e in pts
            if not any(c2 <= c and e2 <= e and (c2 < c or e2 < e) for b, c2, e2 in pts if b != a)]
    return sorted(keep, key=lambda a: next(c for b, c, e in pts if b == a))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = dict(rule='DESIGN.md §5: every step g <= 1e-6 or residual-rule exit; IC g <= 1e-6 or relative residual <= 1e-10; '
                    'every exit regular', slack=SLACK, meshes={})
    for job in JOBS:
        r = json.loads((HERE / 'artifacts' / job / 'result.json').read_text())
        au = json.loads((HERE / 'artifacts' / job / 'audit.json').read_text())
        roms = [x for x in r['invocations'] if x['kind'] == 'rom']
        per_arm = {}
        for x in roms:
            d = per_arm.setdefault(x['name'], dict(design5=[], own_gtol=[], driver=[], gtol=x['gtol'], max_step=0., reasons=set()))
            d['design5'].append(converged_under(x, DESIGN5_GTOL))
            d['own_gtol'].append(converged_under(x, x['gtol']))
            d['driver'].append(bool(x['converged']))
            d['max_step'] = max(d['max_step'], max(x['step_joint_stationarity']))
            d['reasons'] |= set(x['stop_reasons'])
        arms = []
        for row in au['arms']:
            d = per_arm.get(row['arm'])
            conv5 = all(d['design5']) if d else None
            convo = all(d['own_gtol']) if d else None
            cert = row.get('rule_basis') in ('primary', 'secondary') if row.get('quadrature') == 'eq' else None
            fast_ok = bool((au['checks'].get('fast_parity') or {}).get('passed'))
            if row['family'] in ('fom', 'fno'):
                adm5 = admo = True
            elif row['family'] == 'fast':
                adm5 = bool(conv5) and fast_ok and cert is not False
                admo = bool(convo) and fast_ok and cert is not False
            else:
                adm5 = bool(conv5) and cert is not False
                admo = bool(convo) and cert is not False
            arms.append(dict(arm=row['arm'], family=row['family'], gtol=row.get('gtol'),
                             converged_design5=conv5, converged_own_gtol=convo,
                             driver_flag=(all(d['driver']) if d else None), audit_converged_flag=row.get('converged'),
                             audit_admissible_flag=row.get('admissible'),
                             max_step_stationarity=(d['max_step'] if d else None), stop_reasons=(sorted(d['reasons']) if d else None),
                             admissible_design5=adm5, admissible_own_gtol=admo,
                             median_gpu_ms=row['median_gpu_ms'], median_host_ms=row['median_host_ms'],
                             worst_all_times_percent=row['worst_all_times_percent'], worst_evolved_percent=row['worst_evolved_percent']))
        # the as-implemented flag must reproduce the committed audit exactly; the §5 flag is the finding
        assert all(x['converged_own_gtol'] == x['audit_converged_flag'] for x in arms if x['family'] in REDUCED)
        assert all(x['admissible_own_gtol'] == x['audit_admissible_flag'] for x in arms)
        red = [x for x in arms if x['family'] in REDUCED]
        flipped = [x['arm'] for x in red if x['converged_own_gtol'] and not x['converged_design5']]
        loose = [x['arm'] for x in red if x['gtol'] is not None and x['gtol'] > DESIGN5_GTOL]
        fom = [x for x in arms if x['family'] == 'fom']
        tight = next(x for x in fom if x['arm'] == 'fft_tight')
        cf = min(fom, key=lambda z: z['median_gpu_ms'])

        def summarise(flag):
            adm = [x for x in arms if x[flag]]
            nd = {k: nondominated(adm, ck, ek) for k, (ck, ek) in PAIRS.items()}
            ndr = {k: nondominated([x for x in adm if x['family'] in REDUCED], ck, ek) for k, (ck, ek) in PAIRS.items()}
            cr = min([x for x in adm if x['family'] in REDUCED], key=lambda z: z['median_gpu_ms'])
            return dict(admissible_reduced=sum(1 for x in adm if x['family'] in REDUCED), reduced=len(red),
                        reduced_on_gpu_evolved_frontier=[x for x in nd['gpu_evolved'] if any(y['arm'] == x and y['family'] in REDUCED for y in arms)],
                        nondominated_admissible=nd, nondominated_reduced_only=ndr,
                        cheapest_admissible_reduced=cr['arm'], cheapest_fom=cf['arm'],
                        ratio_cheapest_reduced_over_cheapest_fom=cr['median_gpu_ms'] / cf['median_gpu_ms'],
                        ratio_cheapest_reduced_over_fft_tight=cr['median_gpu_ms'] / tight['median_gpu_ms'])

        out['meshes'][str(au['intervals'])] = dict(
            job_id=au['job_id'], attempt=job, gpu=au['gpu'],
            reduced_arms=len(red), loose_tolerance_arms=loose, arms_flipped_by_design5=flipped,
            counterexample=next((dict(arm=x['arm'], max_step_stationarity=x['max_step_stationarity'], stop_reasons=x['stop_reasons'],
                                      converged_own_gtol=x['converged_own_gtol'], converged_design5=x['converged_design5'])
                                 for x in red if x['arm'] in flipped), None),
            as_implemented=summarise('admissible_own_gtol'), design5=summarise('admissible_design5'), arms=arms)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    for L, m in out['meshes'].items():
        o, n = m['as_implemented'], m['design5']
        print(f"{L}²  job {m['job_id']}: admissible reduced {o['admissible_reduced']} -> {n['admissible_reduced']} of {m['reduced_arms']}; "
              f"reduced on gpu_evolved admissible frontier {len(o['reduced_on_gpu_evolved_frontier'])} -> {len(n['reduced_on_gpu_evolved_frontier'])} "
              f"{n['reduced_on_gpu_evolved_frontier']}; ratio/FOM {o['ratio_cheapest_reduced_over_cheapest_fom']:.6f} -> "
              f"{n['ratio_cheapest_reduced_over_cheapest_fom']:.6f}; ratio/fft_tight {o['ratio_cheapest_reduced_over_fft_tight']:.6f} -> "
              f"{n['ratio_cheapest_reduced_over_fft_tight']:.6f}; flipped {len(m['arms_flipped_by_design5'])}")
        print('   counterexample:', m['counterexample'])
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
