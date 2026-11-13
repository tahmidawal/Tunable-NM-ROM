"""Second rebuttal push, PROTOCOL.md A9(d)/A9b: phase track and per-third errors from saved series (any method).

Reads series files that hold the lift of a method and of the reference plus e_u / e_p per save:
  ROM (rom_r4.py):      t_save, e_u, e_p, t, CL, CL_ref          (lift on the step grid)
  operators / FE:       t, e_u, e_p, CL, CL_ref                  (lift on the save grid)
Phase track: the window is cut into consecutive supports of one reference period T (the first starts T/2 after the
start); on each support the lag (in saves) minimising the RMS lift difference is searched within +-T/4 of the previous
support's lag (first support: around zero), so a drift larger than half a period is followed instead of wrapped.
Positive lag = the method is behind the reference.  Thirds: RMS and max of e_u, e_p over each third of the saves.
Usage: drift.py --period T --out FILE.json label=series.npz [...]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))


def track(ts, cl, cl_ref, period):
    ds = ts[1] - ts[0]
    kq = max(1, int(round(0.25 * period / ds)))
    out = []; prev = 0; m = 0
    while True:
        lo, hi = ts[0] + (m + 0.5) * period, ts[0] + (m + 1.5) * period
        I = np.where((ts >= lo - 1e-12) & (ts < hi - 1e-12))[0]
        if hi > ts[-1] + 1e-12 or len(I) == 0:
            break
        cand = [k for k in range(prev - kq, prev + kq + 1) if I.min() + k >= 0 and I.max() + k < len(ts)]
        if not cand:
            break
        # method lift at save i compared with the reference at save i - k: k > 0 means the method lags
        k, err = min(((k, rms(cl[I] - cl_ref[I - k])) if I.min() - k >= 0 and I.max() - k < len(ts) else (k, np.inf) for k in cand), key=lambda z: z[1])
        out.append({'support_start': float(lo), 'lag_saves': int(k), 'lag_seconds': float(k * ds), 'lag_periods': float(k * ds / period),
                    'lift_rms_at_lag': float(err), 'lift_rms_unshifted': rms(cl[I] - cl_ref[I])})
        prev = k; m += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', type=float, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('series', nargs='+')
    a = ap.parse_args()
    res = {'period': a.period, 'series': {}}
    for spec in a.series:
        label, path = spec.split('=', 1)
        try:
            z = np.load(path)
            if 't_save' in z.files:
                ts = z['t_save']; ok = np.isfinite(z['CL'])
                cl = np.interp(ts, z['t'][ok], z['CL'][ok]); clr = np.interp(ts, z['t'][ok], z['CL_ref'][ok])
            else:
                ts = z['t']; cl = z['CL']; clr = z['CL_ref']
            n = len(ts); cuts = [0, n // 3, 2 * n // 3, n]
            rec = {'saves': n, 'thirds': [{'t': [float(ts[i]), float(ts[j - 1])], 'e_u_rms': rms(z['e_u'][i:j]), 'e_u_max': float(np.max(z['e_u'][i:j])),
                                           'e_p_rms': rms(z['e_p'][i:j]), 'e_p_max': float(np.max(z['e_p'][i:j])),
                                           'lift_error_rms': rms(cl[i:j] - clr[i:j])} for i, j in zip(cuts[:-1], cuts[1:])]}
            tr = track(ts, cl, clr, a.period)
            rec['phase_track'] = tr
            if tr:
                rec['final_lag_periods'] = tr[-1]['lag_periods']
                rec['max_abs_lag_periods'] = max(abs(r['lag_periods']) for r in tr)
                rec['lag_per_period'] = tr[-1]['lag_periods'] / len(tr)
            rec['status'] = 'completed'
        except Exception as exc:
            rec = {'status': 'failed', 'error': repr(exc)}
        res['series'][label] = rec
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print('drift', a.out, {k: (v.get('final_lag_periods'), v.get('status')) for k, v in res['series'].items()})


if __name__ == '__main__':
    main()
