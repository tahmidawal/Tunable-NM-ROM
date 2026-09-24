"""burgers3d-retry select3 = burgers3d-span/select.py with the design-audit fixes (DESIGN R1): certificates must
come from a job with the same model hashes, the same deployed arm spec and exactly the prescribed draws; a head arm
whose initial fit ends non-finite or with reason 3 is ineligible; the determinism gate joins the timing gates; and
an explicit lane-success gate (both frozen settings within their bars and faster than the paper-rule FOM, all
gates passed).

Pre-registered settings rule (DESIGN.md section 6 + R1), applied mechanically to one mesh's validation panel and
certification results. Writes a JSON with every arm's eligibility, error, time and speedups; used for validation
(to freeze settings) and for the held-out rows (FOM chosen by the same rule on held-out errors; frozen FOM beside)."""
import argparse
import json
from pathlib import Path

FAST_BAR = 0.05
STALL_FRACTION = 0.01


CERT_DRAWS = [[923811 + i, 8] for i in range(5)] + [[923816, 16]]
SPEC_KEYS = ('kind', 'Rp', 'solver', 'dt', 't32', 'head', 'K', 'gtol', 'M', 'trust')


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
    if arm['spec']['kind'] == 'head':
        bad = [q['case'] for q in arm['quick'] if int(q['initial_fit'][1]) == 3 or
               not all(x == x for x in q['initial_fit'])]
        if bad:
            why.append(f'head initial fit failed (reason 3 / non-finite) on cases {bad}')
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
    ap.add_argument('--cert', required=True, nargs='+', help='certification results; certificates are merged by arm name')
    ap.add_argument('--out', required=True)
    ap.add_argument('--frozen', help='validation selection JSON (held-out use: report its FOM beside)')
    a = ap.parse_args()
    P = json.loads(Path(a.panel).read_text())
    Ces = [json.loads(Path(c).read_text()) for c in a.cert]
    assert P['complete'] and all(c['complete'] and c['mesh'] == P['mesh'] for c in Ces)
    for c in Ces:                                   # R1-2: certificates bound to the evaluated model and draws
        assert c['model_sha256'] == P['model_sha256'], ('model differs', c['job_id'])
        assert [list(x) for x in c['config']['cert_draws']] == CERT_DRAWS, ('draws differ', c['job_id'])
        for k, v in c['arms'].items():
            if k in P['arms']:
                a_, b_ = v['spec'], P['arms'][k]['spec']
                assert all(a_.get(q) == b_.get(q) for q in SPEC_KEYS), ('spec differs', k)
    Ce = dict(certificates={}, job_id='+'.join(str(c['job_id']) for c in Ces),
              gates=dict(bad_control_not_confirmed=all(c['gates'].get('bad_control_not_confirmed', True) for c in Ces)))
    for c in Ces:
        for k, v in c['certificates'].items():
            assert k not in Ce['certificates'] or Ce['certificates'][k]['confirmed'] == v['confirmed'], k
            Ce['certificates'].setdefault(k, v)
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
    reselected = dict(accurate=acc, fast=fast)
    if a.frozen:     # held-out: the rows ARE the frozen validation arms; re-selection is informational only
        fr0 = json.loads(Path(a.frozen).read_text())
        fr0 = fr0['meshes'][str(P['mesh'])] if 'meshes' in fr0 else fr0
        acc, fast = fr0['accurate'], fr0['fast']
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
    timing_ok = bool(gates.get('timing_drift_pass') and gates.get('timing_neighbour_pass') and
                     gates.get('deterministic_pass'))
    stop_pass = [k for k, v in rows.items() if v['eligible'] and v['worst_evolved'] <= FAST_BAR and
                 v['speedup_own'] is not None and v['speedup_own'] > 1.0]
    out = dict(mesh=P['mesh'], cohort=P['cohort'], job_id=P['job_id'], cert_job_id=Ce['job_id'], gpu=P['gpu'],
               commit=P['commit'], accurate=acc, fast=fast, fom=fom, fom_note=fom_note, fast_bar=FAST_BAR,
               accurate_row=rows.get(acc), fast_row=rows.get(fast), fom_row=foms.get(fom), arms=rows, foms=foms,
               reselected_on_this_cohort=reselected,
               fast_within_bar=None if fast is None else bool(rows[fast]['worst_evolved'] <= FAST_BAR),
               timing_gates=dict(drift=gates.get('timing_drift_worst'), neighbour=gates.get('timing_neighbour_ratio'),
                                 neighbour_fom=gates.get('timing_neighbour_ratio_fom'), passed=timing_ok),
               other_gates={k: v for k, v in gates.items() if not k.startswith('timing')},
               bad_control_not_confirmed=Ce['gates'].get('bad_control_not_confirmed'),
               stopping_rule_candidates=stop_pass, stopping_rule_pass=bool(stop_pass) and timing_ok)
    if a.frozen:
        fr = json.loads(Path(a.frozen).read_text())
        if 'meshes' in fr:
            fr = fr['meshes'][str(P['mesh'])]
        out['frozen_validation_fom'] = fr['fom']
        out['frozen_fom_row'] = foms.get(fr['fom'])
        for role in ('accurate', 'fast'):
            nm = fr[role]
            out[f'frozen_{role}'] = nm
            r = rows.get(nm)
            if r is not None and fr['fom'] in foms and r['median_ms']:
                r['speedup_vs_frozen_fom'] = foms[fr['fom']]['median_ms'] / r['median_ms']
    ar, fr_ = rows.get(acc), rows.get(fast)
    out['lane_success'] = bool(ar and fr_ and ar['eligible'] and fr_['eligible'] and fr_['worst_evolved'] <= FAST_BAR
                               and (ar['speedup_rule'] or 0) > 1 and (fr_['speedup_rule'] or 0) > 1 and timing_ok
                               and fom_note is None)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('mesh', 'accurate', 'fast', 'fom', 'stopping_rule_pass')}))
    for k in (acc, fast):
        if k:
            r = rows[k]
            print(k, f"{100 * r['worst_evolved']:.3f}%", f"{r['median_ms']:.2f} ms", f"rule {r['speedup_rule']:.3f}x",
                  f"own {r['speedup_own']}")


if __name__ == '__main__':
    main()
