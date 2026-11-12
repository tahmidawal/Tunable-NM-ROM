"""Tabulate A9 extra-case results from collected JSON (reduced arms and operators). Read-only.
Usage: xtable.py <artifact dir> [<artifact dir> ...]   (directories that hold results/r4*_<case>_dt*/ and DRIFT-*.json)"""
import glob, json, os, sys


def lag(dr, name):
    s = dr['series'].get(name) if dr else None
    if not s or not s.get('phase_track'):
        return float('nan'), float('nan')
    l = [p['lag_periods'] for p in s['phase_track']]
    return l[-1], max(abs(x) for x in l)


rows = []
for root in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(root, 'results', 'r4*_dt*', 'R4-*.json'))):
        d = json.load(open(f))
        case, dt = d['case'], d['dt']
        df = os.path.join(root, 'results', 'DRIFT-rom-%s-dt%s.json' % (case, os.path.basename(os.path.dirname(f)).split('_dt')[1]))
        dr = json.load(open(df)) if os.path.exists(df) else None
        for name, a in d['arms'].items():
            if a.get('e_u') is None:
                rows.append((case, dt, name, 'FAILED', a.get('status'))); continue
            met = sum(1 for t in a['targets'].values() if isinstance(t, dict) and t.get('met'))
            ce = a['cycle_errors']; lf, lm = lag(dr, name)
            rows.append((case, dt, name, a['e_u']['rms'], a['e_u']['max'], a['e_p']['rms'], ce['St_rel'], ce['CD_max_rel'],
                         ce['CL_max_abs'], a['divergence']['median'], lf, lm, met, a.get('rollout_seconds')))
    for f in sorted(glob.glob(os.path.join(root, 'results', 'ops', 'OPS-*.json'))):
        d = json.load(open(f))
        for name, arm in d['arms'].items():
            for case, a in arm['cases'].items():
                df = os.path.join(root, 'results', 'DRIFT-ops-%s.json' % case)
                dr = json.load(open(df)) if os.path.exists(df) else None
                if a.get('e_u') is None:
                    rows.append((case, 0.0025, name, 'FAILED', a.get('status'))); continue
                ce = a['cycle_errors']; lf, lm = lag(dr, name)
                dv = a.get('divergence') or {}
                rows.append((case, 0.0025, name, a['e_u']['rms'], a['e_u']['max'], a['e_p']['rms'], ce['St_rel'], ce['CD_max_rel'],
                             ce['CL_max_abs'], dv.get('median', float('nan')), lf, lm, a['targets_met'], None))
print('case dt arm e_u_rms e_u_max e_p_rms St_rel CD_rel CL_abs div_med lag_final_periods lag_max_periods met/6 rollout_s')
for r in sorted(rows, key=lambda r: (r[0], -r[1], r[2])):
    if r[3] == 'FAILED':
        print(*r); continue
    print('%-6s %-9g %-10s %.5f %.5f %.5f %.4f %.4f %.4f %.3g %+.4f %.4f %d %s' % (r[:13] + (('%.2f' % r[13]) if r[13] else '-',)))
