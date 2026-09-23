"""Pre-registered settings rule (DESIGN.md section 6 + R1), applied mechanically to one mesh's validation panel and
certification results. Writes a JSON with every arm's eligibility, error, time and speedups; used for validation
(to freeze settings) and for the held-out rows (FOM chosen by the same rule on held-out errors; frozen FOM beside)."""
import argparse
import json
from pathlib import Path

FAST_BAR = 0.05
STALL_FRACTION = 0.01


def eligibility(arm, cert):
    reasons = arm['reason_counts']
    steps = sum(reasons.values())
    nonstat = reasons.get('0', 0) + reasons.get('2', 0)
    why = []
    if arm['spec']['rule'] == 'exact':
        why.append('control arm (exact dense advection), never eligible')
    if cert is None:
        why.append('no certificate')
    elif not cert['confirmed']:
        why.append('certificate not confirmed (rho_max by draw ' +
                   ', '.join('non-finite' if r is None else f'{r:.3f}' for r in cert['rho_max_draws']) + ')')
    if not arm['all_finite']:
        why.append('non-finite output')
    if reasons.get('3', 0):
        why.append(f"{reasons['3']} reason-3 exits")
    if nonstat > STALL_FRACTION * steps:
        why.append(f'{nonstat}/{steps} non-stationary exits')
    return not why, why


def fastest_fom(foms, bar):
    ok = [(v['median_ms'], k) for k, v in foms.items() if v['all_finite'] and v['worst_evolved'] <= bar]
    return min(ok)[1] if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel', required=True)
    ap.add_argument('--cert', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frozen', help='validation selection JSON (held-out use: report its FOM beside)')
    a = ap.parse_args()
    P = json.loads(Path(a.panel).read_text())
    Ce = json.loads(Path(a.cert).read_text())
    assert P['complete'] and Ce['complete'] and P['mesh'] == Ce['mesh']
    tim = P['timing']['summary']
    foms = {k: dict(worst_evolved=v['worst_evolved'], all_finite=v['all_finite'], median_ms=tim[k]['median_ms'],
                    capped=any(r['hit_newton_cap'] for r in v['quick']))
            for k, v in P['fom'].items()}
    rows = {}
    for name, arm in P['arms'].items():
        if 'quick' not in arm:
            continue
        cert = Ce['certificates'].get(name)
        ok, why = eligibility(arm, cert)
        rows[name] = dict(spec=arm['spec'], worst_evolved=arm['worst_evolved'], median_ms=tim.get(name, {}).get('median_ms'),
                          eligible=ok, why_not=why,
                          rho_max=None if cert is None or None in cert['rho_max_draws'] else max(cert['rho_max_draws']),
                          certified=None if cert is None else cert['confirmed'],
                          min_decoded=None if cert is None else min(d['umin'] for d in cert['draws']),
                          iterations_per_query=arm['lm_iterations_per_query_median'])
    elig = {k: v for k, v in rows.items() if v['eligible']}
    acc = min(elig, key=lambda k: elig[k]['worst_evolved']) if elig else None
    fastc = {k: v for k, v in elig.items() if v['worst_evolved'] <= FAST_BAR}
    fast = min(fastc, key=lambda k: fastc[k]['median_ms']) if fastc else None
    fom = fastest_fom(foms, rows[acc]['worst_evolved']) if acc else None
    fom_note = None
    if acc and fom is None:
        fom = min(foms, key=lambda k: foms[k]['worst_evolved'])
        fom_note = 'no FOM setting is as accurate as the accurate arm; most accurate FOM named, speedups vs a less accurate FOM'
    for k, v in rows.items():
        own = fastest_fom(foms, v['worst_evolved'])
        v['own_fom'] = own
        v['speedup_own'] = None if own is None or v['median_ms'] is None else foms[own]['median_ms'] / v['median_ms']
        v['speedup_rule'] = None if fom is None or v['median_ms'] is None else foms[fom]['median_ms'] / v['median_ms']
    gates = P['gates']
    timing_ok = bool(gates.get('timing_drift_pass') and gates.get('timing_neighbour_pass'))
    stop_pass = [k for k, v in rows.items() if v['eligible'] and v['worst_evolved'] <= FAST_BAR and
                 v['speedup_own'] is not None and v['speedup_own'] > 1.0]
    out = dict(mesh=P['mesh'], cohort=P['cohort'], job_id=P['job_id'], cert_job_id=Ce['job_id'], gpu=P['gpu'],
               commit=P['commit'], accurate=acc, fast=fast, fom=fom, fom_note=fom_note, fast_bar=FAST_BAR,
               accurate_row=rows.get(acc), fast_row=rows.get(fast), fom_row=foms.get(fom), arms=rows, foms=foms,
               timing_gates=dict(drift=gates.get('timing_drift_worst'), neighbour=gates.get('timing_neighbour_ratio'),
                                 neighbour_fom=gates.get('timing_neighbour_ratio_fom'), passed=timing_ok),
               other_gates={k: v for k, v in gates.items() if not k.startswith('timing')},
               bad_control_not_confirmed=Ce['gates'].get('bad_control_not_confirmed'),
               stopping_rule_candidates=stop_pass, stopping_rule_pass=bool(stop_pass) and timing_ok)
    if a.frozen:
        fr = json.loads(Path(a.frozen).read_text())
        out['frozen_validation_fom'] = fr['fom']
        out['frozen_fom_row'] = foms.get(fr['fom'])
        for role in ('accurate', 'fast'):
            nm = fr[role]
            out[f'frozen_{role}'] = nm
            r = rows.get(nm)
            if r is not None and fr['fom'] in foms and r['median_ms']:
                r['speedup_vs_frozen_fom'] = foms[fr['fom']]['median_ms'] / r['median_ms']
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('mesh', 'accurate', 'fast', 'fom', 'stopping_rule_pass')}))
    for k in (acc, fast):
        if k:
            r = rows[k]
            print(k, f"{100 * r['worst_evolved']:.3f}%", f"{r['median_ms']:.2f} ms", f"rule {r['speedup_rule']:.3f}x",
                  f"own {r['speedup_own']}")


if __name__ == '__main__':
    main()
