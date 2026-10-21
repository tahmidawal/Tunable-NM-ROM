"""spectral-pilot: readouts R1-R4 and the per-target verdicts of DESIGN A1.3 / A2.1 / A3.1, from the run JSONs.
PILOT: small, not a result.  Writes runs/ANALYSIS.json (read by make_report.py).
"""
from __future__ import annotations

import json
import math

import numpy as np

import common as C
import score as SC

LEVELS = ['A1', 'A2', 'A3', 'A4', 'AB']


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def min_size(rows_by_size, U, tau):
    """Smallest observed feasible size (any observed step count meets tau, A3.1); None if none."""
    ok = [int(s) for s, rows in rows_by_size.items() if any(SC.classify(r['worst'], U, tau) == 'meets' for r in rows)]
    return min(ok) if ok else None


def floor_size(worst_by_size, tau):
    ok = [int(s) for s, w in worst_by_size.items() if w <= tau]
    return min(ok) if ok else None


def r_cross(cm, N, cheapest=True):
    """First crossing from below (A3.1): smallest measured r whose ROM evaluation time >= the FOM's at N, log-log
    interpolated with the preceding r. cheapest=True (the design's rule, DESIGN 6 / A2.1): per r the cheaper of the
    tensor and every measured grid-rule form (unrestricted by accuracy); cheapest=False: tensor only (diagnostic).
    'unresolved' if not bracketed or multiple crossings."""
    tf = cm['fom'].get(str(N)) or cm['fom'].get(N)
    rs = sorted(int(r) for r in cm['rom_t'])
    tr = []
    for r in rs:
        c = [cm['rom_t'][str(r)]]
        if cheapest:
            c += [v for k, v in cm['rom_q'].items() if int(k.split('|')[0]) == r]
        tr.append(min(c))
    above = [t >= tf for t in tr]
    changes = sum(1 for i in range(1, len(above)) if above[i] != above[i - 1])
    if not any(above) or above[0] or changes != 1:
        return dict(value=None, status='unresolved', changes=changes)
    i = above.index(True)
    r0, r1, t0, t1 = rs[i - 1], rs[i], tr[i - 1], tr[i]
    x = math.log(r0) + (math.log(tf) - math.log(t0)) * (math.log(r1) - math.log(r0)) / (math.log(t1) - math.log(t0))
    return dict(value=math.exp(x), status='ok', bracket=[r0, r1])


def fit_slope(xs, ys):
    if len(xs) < 3:
        return None
    lx, ly = np.log(xs), np.log(ys)
    return float(np.polyfit(lx, ly, 1)[0])


def category(Ss):
    sgn = [s >= 1 for s in Ss]
    ch = sum(1 for i in range(1, len(sgn)) if sgn[i] != sgn[i - 1])
    if all(sgn):
        return 'ROM faster throughout'
    if not any(sgn):
        return 'FOM faster throughout'
    if ch == 1:
        return 'crossover toward the ROM' if sgn[-1] else 'reverse crossover'
    return 'mixed'


def main():
    out = dict(label=C.PILOT, A={}, B=None, verdicts={})
    lv_data = {}
    for lv in LEVELS:
        d = C.RUNS / 'A' / lv
        ref, fom, rom = load(d / 'ref.json'), load(d / 'fom.json'), load(d / 'rom.json')
        tf, ts, diag = load(d / 'time_final.json'), load(d / 'time_screen.json'), load(d / 'diag.json')
        if not (ref and fom and rom):
            out['A'][lv] = dict(status='not run')
            continue
        pad = load(d / 'pad.json')
        U = fom['U']
        g = rom.get('gates', {})
        req = ('32', '128')                             # required control ranks; absent controls fail (code audit 3)
        ctrl = dict(G4=all(k in g and g[k]['G4_pass'] for k in req),
                    SW4=all(k in g and (g[k]['SW4_fired'] or not g[k].get('SW4_available', True)) for k in req),
                    G7=bool(diag and diag['G7_pass']), gates=fom.get('gates'))
        timing_ok = bool(tf and ts and tf.get('timing_valid') and ts.get('screen_valid')
                         and tf.get('screen_time') == ts['env']['time'])
        if not (ctrl['G4'] and ctrl['SW4'] and ctrl['G7']):
            U = None                                    # a failed control makes the level unresolved (A5)
        e = dict(status='run', controls=ctrl, timing_valid=timing_ok, basis_label=fom.get('basis_label'), Nref=ref['Nref'], accepted=ref['accepted'], U=U, U_has_pad=fom['U_has_pad'],
                 pad=pad and pad['max'], s_median=float(np.median(ref['s_test'])),
                 s_range=[float(min(ref['s_test'])), float(max(ref['s_test']))], diag=diag and dict(
                     sharp_t0=diag['sharp_t0_median'], sharp=diag['sharp_max_median'], energy=diag['energy_median'],
                     G7=diag['G7_max']), gates=rom.get('gates'), SW1a=fom['SW1a']['passed'], SW1b=fom['SW1b']['passed'],
                 SW2=ref['SW2_fired'], train_check=ref.get('train_check', {}).get('max'),
                 unexplored_N=fom['unexplored'], timings_offline=rom['timings'], targets={})
        romrows = {('T', r): v['rows'] for r, v in rom['rom_t'].items()}
        for r, dd in rom['rom_q'].items():
            for Mq, rows in dd.items():
                romrows[('Q', r, Mq)] = rows
        r_by = {}
        for k, rows in romrows.items():
            r_by.setdefault(k[1], []).extend(rows)
        for tau in SC.TARGETS:
            t = {}
            t['N_min'] = min_size({N: v['rows'] for N, v in fom['sizes'].items()}, U, tau)
            t['r_min'] = min_size(r_by, U, tau)
            t['r_min_sine'] = min_size(rom['rom_sine'], U, tau) if rom.get('rom_sine') else None
            t['N_floor'] = floor_size({N: v['worst'] for N, v in rom['floors']['spectral'].items()}, tau)
            t['r_floor'] = floor_size({i + 1: w for i, w in enumerate(rom['floors']['pod']['worst'])}, tau)
            t['r_floor48'] = floor_size({i + 1: w for i, w in enumerate(rom['floors']['pod48']['worst'])}, tau)
            t['r_floor_sine'] = floor_size({i + 1: w for i, w in enumerate(rom['floors']['sine']['worst'])}, tau)
            if ts and ts.get('screen_valid'):
                t['r_cross'] = r_cross(ts['cost_model'], t['N_min']) if t['N_min'] else None
                t['r_cross_tensor'] = r_cross(ts['cost_model'], t['N_min'], cheapest=False) if t['N_min'] else None
            # spectral-floor D2 (worst over cases): every FOM run's worst error >= the truncation floor's worst
            for N, v in fom['sizes'].items():
                fl_ = rom['floors']['spectral'].get(str(N))
                if fl_:
                    for row in v['rows']:
                        assert (not math.isfinite(row['worst'])) or row['worst'] >= fl_['worst'] * (1 - 1e-9), ('D2-spec', N)
            if tf and str(tau) in tf['targets'] and U is not None and timing_ok:
                x = tf['targets'][str(tau)]
                t['S'] = x.get('S')
                if x.get('S') is not None:
                    t['native_S'] = x.get('native', {}).get('S')
                    t['fom'] = {k: x['fom'][k] for k in ('N', 'n', 't', 'worst', 'spread', 'outliers')}
                    t['rom'] = {k: x['rom'][k] for k in ('form', 'r', 'Mq', 'n', 'Ninit', 't', 'worst', 'spread', 'outliers')}
                else:
                    t['censored'] = x
            e['targets'][str(tau)] = t
        if tf:
            e['sentinel'] = tf['sentinel']
            e['H1'] = tf['H1']
        if ts:
            e['cost_model'] = ts['cost_model']
        out['A'][lv] = e
        lv_data[lv] = e
    # per-target verdicts (A1.3/A2.1/A3.1), 96-trajectory arms on A1-A3
    for tau in SC.TARGETS:
        elig = [lv for lv in ('A1', 'A2', 'A3') if lv in lv_data and lv_data[lv]['accepted']
                and lv_data[lv]['targets'][str(tau)].get('S') is not None]
        v = dict(eligible=elig)
        if len(elig) >= 3:
            s = [lv_data[lv]['s_median'] for lv in elig]
            Nm = [lv_data[lv]['targets'][str(tau)]['N_min'] for lv in elig]
            rm = [lv_data[lv]['targets'][str(tau)]['r_min'] for lv in elig]
            S = [lv_data[lv]['targets'][str(tau)]['S'] for lv in elig]
            a = fit_slope(s, Nm)
            v['H-A1'] = dict(alpha=a, verdict='consistent' if a is not None and a >= 0.5 else 'inconsistent')
            q = [r / (N - 1) ** 2 for r, N in zip(rm, Nm)]
            v['H-A2'] = dict(ratio=q, verdict='consistent' if q[-1] < q[0] else 'inconsistent')
            rx = [lv_data[lv]['targets'][str(tau)].get('r_cross') for lv in elig]
            if all(x and x['status'] == 'ok' for x in rx):
                q2 = [r / x['value'] for r, x in zip(rm, rx)]
                v['H-A2p'] = dict(ratio=q2, verdict='consistent' if q2[-1] < q2[0] else 'inconsistent')
            else:
                v['H-A2p'] = dict(verdict='unresolved', detail=rx)
            rxt = [lv_data[lv]['targets'][str(tau)].get('r_cross_tensor') for lv in elig]
            if all(x and x['status'] == 'ok' for x in rxt):
                v['H-A2p_tensor_only'] = [r / x['value'] for r, x in zip(rm, rxt)]
            sl = fit_slope(s, S)
            v['H-A3'] = dict(S=S, category=category(S), slope=sl, K2_bar=[x >= 2 for x in S])
            if sl is not None and sl > 0 and S[-1] < 1:
                v['H-A3']['extrapolated_s_at_S1'] = float(math.exp(math.log(s[-1]) + (0 - math.log(S[-1])) / sl))
        else:
            v['verdict'] = 'too few levels'
        out['verdicts'][str(tau)] = v
    # Part B
    d = C.RUNS / 'B'
    ref, fom, rom, tf = (load(d / f) for f in ('ref.json', 'fom.json', 'rom.json', 'time_final.json'))
    if ref and fom and rom:
        gb = rom['gates']
        tsb = load(d / 'time_screen.json')
        ctrl_b = all(k in gb and gb[k]['G4_pass'] and (gb[k]['SW4_fired'] or not gb[k]['SW4_available'])
                     for k in ('32', '128'))
        timing_ok_b = bool(tf and tsb and tf.get('timing_valid') and tsb.get('screen_valid')
                           and tf.get('screen_time') == tsb['env']['time'])
        Ub = fom['U'] if ctrl_b else None
        b = dict(order=ref['order'], order_gate=ref['order_gate'], e_ref=ref['e_ref_hat'], delta_t=ref['delta_t'],
                 U=Ub, controls_ok=ctrl_b, timing_valid=timing_ok_b, allowance=fom.get('allowance'),
                 SW1=fom['SW1a']['passed'] and fom['SW1b']['passed'], basis_label='unestablished (A3.4)', scoring=ref['scoring'], fd512=ref['fd512_err_vs_ref'], train_diag=ref.get('train_diag'),
                 gates=rom['gates'], floors=rom['floors'], targets={}, unexplored=fom['unexplored'])
        for tau in SC.TARGETS:
            t = {}
            for fam, sizes in fom['families'].items():
                t[f'L_min_{fam}'] = min_size({L: v['rows'] for L, v in sizes.items() if not v['diagnostic_only']},
                                             Ub, tau)
            t['r_min'] = min_size({r: v['rows'] for r, v in rom['rom_t'].items()}, Ub, tau)
            if tf and str(tau) in tf['targets'] and Ub is not None and timing_ok_b:
                x = tf['targets'][str(tau)]
                t['S'] = x.get('S')
                if x.get('S') is not None:
                    t['fom'] = {k: x['fom'][k] for k in ('fam', 'L', 'n', 't', 'worst')}
                    t['rom'] = {k: x['rom'][k] for k in ('r', 'n', 'Linit', 't', 'worst')}
                    t['verdict'] = ('ROM >= 2x faster' if t['S'] >= 2 else 'ROM faster but < 2x' if t['S'] >= 1
                                    else 'FD faster')
            b['targets'][str(tau)] = t
        if tf:
            b['sentinel'] = tf['sentinel']
        out['B'] = b
    C.dump(C.RUNS / 'ANALYSIS.json', out)
    print(json.dumps(out['verdicts'], indent=1, default=float)[:3000])


if __name__ == '__main__':
    main()
