"""Registered gates and ratios (PROTOCOL.md §3) for one mesh's measurement.json."""
from statistics import median

DRIFT = 1.10
FOLD = 1.10
OUT_TOL = 1e-12
FUSED_TOL = 1.10
QUERY = ('rom_lat', 'fom')
PREP = ('rom_prep', 'fom_prep')
FUSED = ('rom_e2e', 'fom_e2e')


def _times(inv, subject, blk):
    return [1e3 * r['seconds'] for r in inv if r['subject'] == subject and r['block'] == blk]


def subject_stats(inv, subject, cases, reps):
    out = {}
    for blk, n in (('main', reps['main']), ('confirm', reps['confirm'])):
        rows = [r for r in inv if r['subject'] == subject and r['block'] == blk]
        keys = {(r['case'], r['rep']) for r in rows}
        assert keys == {(c, q) for c in cases for q in range(n)} and len(rows) == len(keys), (subject, blk)
        ts = sorted(1e3 * r['seconds'] for r in rows)
        med = median(ts)
        blocks = [median(1e3 * r['seconds'] for r in rows if r['rep'] == q) for q in range(n)]
        q1, q3 = ts[len(ts) // 4], ts[(3 * len(ts)) // 4]
        diffs = [r['diff'] for r in rows]
        bitwise = [r.get('bitwise', True) for r in rows]
        out[blk] = dict(median_ms=med, min_ms=ts[0], max_ms=ts[-1], q1_ms=q1, q3_ms=q3, n=len(ts), block_medians_ms=blocks,
                        block_drift=max(max(blocks) / med, med / min(blocks)), max_output_diff=max(diffs),
                        all_bitwise=all(bitwise),
                        per_case_ms={str(c): median(1e3 * r['seconds'] for r in rows if r['case'] == c) for c in cases})
    m, c = out['main']['median_ms'], out['confirm']['median_ms']
    out['main_confirm_fold'] = max(m / c, c / m)
    reasons = []
    for blk in ('main', 'confirm'):
        if out[blk]['block_drift'] > DRIFT:
            reasons.append(f'{blk} block drift {out[blk]["block_drift"]:.4f} > {DRIFT}')
        if not out[blk]['max_output_diff'] <= OUT_TOL:
            reasons.append(f'{blk} output diff {out[blk]["max_output_diff"]:.3g} > {OUT_TOL}')
        if subject in PREP and not out[blk]['all_bitwise']:
            reasons.append(f'{blk} preparation not bitwise equal')
    if out['main_confirm_fold'] > FOLD:
        reasons.append(f'main/confirm fold {out["main_confirm_fold"]:.4f} > {FOLD}')
    out['passed'] = not reasons
    out['reasons'] = reasons
    return out


def evaluate(meas):
    inv, cases = meas['invocations'], meas['cases']
    reps = dict(main=meas['schedule']['main_reps'], confirm=meas['schedule']['confirm_reps'])
    S = {s: subject_stats(inv, s, cases, reps) for s in meas['subjects']}
    cm = {s: S[s]['confirm']['median_ms'] for s in S}
    prepared_ok = all(S[s]['passed'] for s in QUERY)
    comp = dict(rom=cm['rom_prep'] + cm['rom_lat'], fom=cm['fom_prep'] + cm['fom'])
    consistency = dict(rom=cm['rom_e2e'] / comp['rom'], fom=cm['fom_e2e'] / comp['fom'])
    cons_ok = all(max(v, 1 / v) <= FUSED_TOL for v in consistency.values())
    e2e_ok = prepared_ok and all(S[s]['passed'] for s in PREP + FUSED) and cons_ok
    per_case = []
    for c in cases:
        k = str(c)
        pc = {s: S[s]['confirm']['per_case_ms'][k] for s in S}
        per_case.append(dict(case=c, prepared_ratio=pc['rom_lat'] / pc['fom'],
                             e2e_ratio=(pc['rom_prep'] + pc['rom_lat']) / (pc['fom_prep'] + pc['fom']),
                             fused_ratio=pc['rom_e2e'] / pc['fom_e2e'], **{f'{s}_ms': v for s, v in pc.items()}))

    def rng(key):
        v = sorted(p[key] for p in per_case)
        return dict(minimum=v[0], median=median(v), maximum=v[-1])
    reasons = {s: S[s]['reasons'] for s in S if S[s]['reasons']}
    if not cons_ok:
        reasons['component_sum_vs_fused'] = [f'{k}: fused/sum {v:.4f}' for k, v in consistency.items()
                                             if max(v, 1 / v) > FUSED_TOL]
    return dict(subjects=S, confirm_median_ms=cm,
                prepared_ratio=cm['rom_lat'] / cm['fom'], prepared_qualified=prepared_ok,
                e2e_ms=comp, e2e_ratio=comp['rom'] / comp['fom'], e2e_fused_ratio=cm['rom_e2e'] / cm['fom_e2e'],
                fused_over_component_sum=consistency, e2e_qualified=e2e_ok,
                panel_wide_pass=all(S[s]['passed'] for s in S) and cons_ok,
                per_case=per_case, paired_prepared_ratio=rng('prepared_ratio'), paired_e2e_ratio=rng('e2e_ratio'),
                reasons=reasons)
