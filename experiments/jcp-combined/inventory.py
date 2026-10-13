"""Resource gate (DESIGN A2.12; code audit 1 item 10): project a production job's wall time and peak device memory from
a pulled cluster smoke on the same GPU type, and write checks/inventory-<attempt>.json (fits: bool). stage.py refuses a
non-smoke attempt without a passing certificate committed at HEAD.

Model: per run, seconds = compile (first case minus steady) + per-step steady seconds x steps, fitted per
(setting class, solver family, form, scheme, level, K) from the smoke rows (case index >= 1 = steady); unmatched specs
take the per-step rate of the most expensive matched family of the same setting class (conservative). Timing panels:
invocations x mean steady call time of their subjects + 0.1 s burn each. Settings not present in the smoke map to the
smoke setting of equal or larger R' (conservative). Margin 1.5x on the total.

    python inventory.py --attempt c3grid --smoke runs/smk3grid/archive/output --gpu-gb 141 --hours 24
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def family(key):
    p = key.split('|')
    if p[0] in ('fs', 'vendor'):
        return (p[0], p[1], p[3] if p[0] == 'fs' else 'K1')
    if p[0] == 'ad':
        return ('ad', p[1], p[2], p[4])
    return (p[0], p[1], p[3]) if len(p) > 3 else tuple(p)       # 2D: form|scheme|dt|level


def steps_of(key):
    p = key.split('|')
    if p[0] == 'vendor':
        return 25
    if p[0] == 'fs':
        return int(round(.25 / float(p[2])))
    if p[0] == 'ad':
        return int(round(.25 / float(p[3])))
    return int(round(.25 / float(p[2])))


def rates(rows, setting_key, spec_key):
    out = {}
    by = {}
    for r in rows:
        by.setdefault((r[setting_key], r[spec_key]), []).append(r)
    for (s, k), rs in by.items():
        rs = sorted(rs, key=lambda x: x.get('case', 0) if isinstance(x.get('case'), int) else 0)
        st = steps_of(k)
        steady = [r['seconds_first'] for r in rs[1:]] or [rs[0]['seconds_first']]
        comp = max(0., rs[0]['seconds_first'] - float(np.median(steady)))
        out.setdefault(s, {}).setdefault(family(k), []).append((float(np.median(steady)) / st, comp))
    return {s: {f: (max(x[0] for x in v), max(x[1] for x in v)) for f, v in d.items()} for s, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--gpu-gb', type=float, required=True)
    ap.add_argument('--hours', type=float, required=True)
    a = ap.parse_args()
    cfg = json.loads((HERE / 'configs' / f'{a.attempt}.json').read_text())
    sm = json.loads((Path(a.smoke) / 'result.json').read_text())
    drv = cfg['driver']
    inv = json.loads(subprocess.check_output(['/home/tahmid/Dev/.venv/bin/python', str(HERE / f'{drv}.py'), '--config',
                                              str(HERE / 'configs' / f'{a.attempt}.json'), '--out', '/dev/null', '--dry-run'],
                                             text=True, cwd=HERE))
    if drv == 'c3run':
        R = rates(sm['rows'], 'setting', 'spec')
        Rp_sm = {k: v['Rp'] for k, v in sm['settings'].items()}
        peak = max(v.get('peak_bytes', -1) for v in sm['settings'].values()) / 1e9
        ncases = inv['cases']
        settings = {s['name']: int(s['Rp']) for s in cfg['settings']}
        run_lists = {k: [x if isinstance(x, str) else x[1] for x in v['runs']] for k, v in inv['settings'].items()}
        tim = {k: v['timing_invocations'] for k, v in inv['settings'].items()}
    else:
        R = rates(sm['rows'], 'setting', 'run')
        Rp_sm = {k: v['Rp'] for k, v in sm['setup'].items() if isinstance(v, dict) and 'Rp' in v}
        peak = max(v.get('peak_bytes', -1) for v in sm['setup'].values() if isinstance(v, dict) and 'Rp' in v) / 1e9
        ncases = sum(len(range(6 if c == 'dev6' else 32)) if not cfg.get('case_subset', {}).get(c) else
                     len(cfg['case_subset'][c]) for c in cfg['cohorts'])
        settings = {s['name']: int(s['Rp']) for s in cfg['settings']}
        run_lists = {k: inv['runs'] for k in settings}
        tim = {k: 3 * len([r for r in inv['runs'] if r.endswith('prod') and float(r.split('|')[2]) >= .0025 - 1e-12]) *
               cfg['timing']['cases'] * cfg['timing']['reps'] for k in settings}
    proj, detail = 0., {}
    for sname, Rp in settings.items():
        cand = sorted([s for s in R if Rp_sm.get(s, 0) >= Rp], key=lambda s: Rp_sm[s]) or \
            sorted(R, key=lambda s: -Rp_sm.get(s, 0))
        src = sname if sname in R else cand[0]
        rr = R[src]
        worst = max(v[0] for v in rr.values())
        tot, comp, call = 0., 0., []
        for k in run_lists[sname]:
            if k.startswith('gl80 (conditional)') or k.startswith('('):
                continue
            f = family(k)
            rate, c_ = rr.get(f, (worst, max(v[1] for v in rr.values())))
            t = rate * steps_of(k)
            tot += t * ncases
            comp += c_
            call.append(t)
        tt = tim[sname] * (float(np.mean(call)) + cfg['timing']['burn']) if call else 0.
        detail[sname] = dict(source_setting=src, rollouts_s=tot, compile_s=comp, timing_s=tt)
        proj += tot + comp + tt
    total_h = 1.5 * proj / 3600
    rep = dict(attempt=a.attempt, smoke=str(a.smoke), smoke_job=sm.get('job_id'), smoke_gpu=sm.get('gpu'),
               projected_hours_with_margin=total_h, requested_hours=a.hours, peak_gb_smoke=peak, gpu_gb=a.gpu_gb,
               fits_time=bool(total_h <= a.hours), fits_memory=bool(not np.isfinite(peak) or peak <= 0.9 * a.gpu_gb),
               detail=detail, note='settings missing from the smoke use the smoke setting of equal or larger R prime')
    rep['fits'] = bool(rep['fits_time'] and rep['fits_memory'])
    dest = HERE / 'checks' / f'inventory-{a.attempt}.json'
    dest.write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'detail'}, indent=1))


if __name__ == '__main__':
    main()
