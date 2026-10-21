"""g2d expanded mandate: cost_points.json (coordinator schema + operator_settings) and results-expanded.json from the
pulled records of the full-ladder panels (runs/pl<n>, runs/phl<n>). Falls back to the narrow Table-1 panel (pb512) for
a mesh whose expanded panel is not collected yet (marked panel_kind). No number is typed.

    /home/tahmid/Dev/.venv/bin/python scripts/summarize_cost.py
"""
import json
from pathlib import Path

import numpy as np

G = Path(__file__).resolve().parents[1]
RUNS = G / 'runs'
FAM = {'fno': 'FNO', 'unet': 'U-Net', 'tsol': 'Transolver', 'transolver': 'Transolver', 'don': 'DeepONet', 'deeponet': 'DeepONet'}
TARGETS = (10., 5., 1., .5)


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def meta(job):
    m = dict(slurm_id=None, gpu=None, backend_gpu=None, exit=None)
    sub = RUNS / job / 'SUBMIT.log'
    if sub.exists() and sub.read_text().split():
        m['slurm_id'] = sub.read_text().split()[-1]
    out = RUNS / job / 'pull' / 'job.out'
    if out.exists():
        t = out.read_text(errors='replace')
        m['backend_gpu'] = 'jax_backend=gpu' in t
        for line in t.splitlines():
            if line.startswith('NVIDIA') and m['gpu'] is None:
                m['gpu'] = line.split(',')[0]
            if line.startswith('run_exit='):
                m['exit'] = int(line.split('=')[1])
    return m


def family_of(name):
    return FAM[name.replace('op_', '').split('-')[0]]


def train_info(src):
    """Training record for a g2d-trained checkpoint (pulled result.json) or the source lane note."""
    if src.get('source', '').startswith('g2d training job'):
        tj = src['source'].split()[-1]
        arm = src['name'] if 'name' in src else src['arm']
        for p in (RUNS / tj / 'pull/out/train' / arm / 'result.json', RUNS / tj / 'pull/out/tr' / arm / 'result.json'):
            r = load(p)
            if r:
                return dict(train_job=tj, slurm_id=meta(tj)['slurm_id'], gpu=meta(tj)['gpu'], epochs=r['epochs_completed'],
                            stop_reason=r['stop_reason'], micro_batch=r.get('micro_batch_final', r.get('micro_batch')),
                            training_seconds=r['training_seconds'], best_checkpoint_sha256=r['best_checkpoint_sha256'])
        w = load(RUNS / tj / 'pull/out/worker.json')
        err = None
        if w:
            for a in w.get('arms', []):
                if a.get('arm') == arm and not a.get('ok', True):
                    err = f"{a.get('error_type')}: {str(a.get('error'))[:240]}"
        if err is None:
            e = RUNS / tj / 'pull/job.err'
            if e.exists():
                tail = [l for l in e.read_text(errors='replace').splitlines() if 'Error' in l][-1:]
                err = ' '.join(tail)[:240] or None
        return dict(train_job=tj, failed=True, reason=err or 'no training record pulled (not run / not collected)')
    return dict(source=src.get('source'))


def cheapest(points, target):
    ok = [p for p in points if p.get('gpu_ms') is not None and p.get('worst_percent') is not None and p['worst_percent'] <= target]
    return min(ok, key=lambda p: p['gpu_ms']) if ok else None


def burgers(job, L, narrow=False):
    j = meta(job)
    cell = dict(pde='Burgers 2D', mesh=L, panel_job=j['slurm_id'], gpu=j['gpu'], cohort='development', panel_kind='narrow Table-1 panel' if narrow else 'expanded ladder',
                error_metric='worst over the 6 dev cases of max over evolved times of ||u-u_ref||_2/||u_0||_2 (same grid, in-job fft_tight)',
                complete=False, nmrom=[], fom=[], operators={}, operator_settings=[])
    drv = RUNS / job / 'pull/out/drv'
    r = load(drv / 'result.json')
    if r is None:
        return cell
    per, inv = {}, {}
    for q in r['quick']:
        per.setdefault(q['name'], {})[q['case']] = q
    for x in r['invocations']:
        inv.setdefault(x['name'], []).append(x['gpu_seconds'])
    setup = {a['arm']: a for a in r['arm_setup']}
    untimed = set(r['config'].get('untimed_fom', []))
    for n, pc in per.items():
        ev = [pc[c]['same_grid_evolved'] for c in sorted(pc)]
        fam = next(iter(pc.values()))['family']
        ms = 1e3 * float(np.median(inv[n])) if n in inv else None
        if fam == 'rom':
            s = setup.get(n, {})
            cell['nmrom'].append(dict(setting=n, kind='head' if s.get('arm_family') == 'a' or s.get('model') == 'trunc' else 'span',
                                      R_prime=s.get('R_prime'), worst_percent=100 * max(ev), median_percent=100 * float(np.median(ev)), gpu_ms=ms))
        elif ms is not None or n.split('__')[0] not in untimed:
            cell['fom'].append(dict(setting=n, worst_percent=100 * max(ev), median_percent=100 * float(np.median(ev)), gpu_ms=ms,
                                    unstable=not all(v.get('nonlinear_converged', True) for v in pc.values()),
                                    is_reference=n.split('__')[0] in ('fft_tight',)))
    gates = {k: v.get('passed') for k, v in r.get('gates', {}).items() if isinstance(v, dict)}
    cell['driver_gates'] = gates
    ops = load(drv / 'opscore.json') or {}
    staging = load(drv / 'staging.json') or []
    man = load(G / 'configs' / f'ops-{job}.json') if not narrow else None
    entries = (man or {}).get('operators', [])
    if narrow:
        entries = [dict(name=k, source='g2d training job ' + {'fno-large': 'b512fno', 'unet-refine': 'b512unet', 'tsol-refine': 'b512tsol',
                                                                  'don-small': 'b512don'}[k]) for k in ('fno-large', 'unet-refine', 'tsol-refine', 'don-small')]
    stag = {s['name']: s for s in staging}
    for e in entries:
        n = e['name']
        o = ops.get(n)
        tr = train_info(dict(e))
        rec = dict(family=family_of(n), setting=n, params=None, worst_percent=None, median_percent=None, gpu_ms=None, ok=False, failure=None,
                   training=tr)
        if o and o.get('ok') and 'worst_evolved_percent' in o:
            rec.update(ok=True, params=o['real_parameter_count'], worst_percent=o['worst_evolved_percent'],
                       median_percent=o['median_evolved_percent'], gpu_ms=o['median_gpu_ms'], dtype=('float64' if n.startswith('fno') and 'f32' not in n else 'float32'),
                       checkpoint_sha256=o['checkpoint_sha256'])
            if tr.get('best_checkpoint_sha256'):
                rec['checkpoint_matches_training'] = tr['best_checkpoint_sha256'] == o['checkpoint_sha256']
        else:
            rec['failure'] = (tr.get('reason') and f"training failed: {tr['reason']}") or (stag.get(n, {}).get('reason')) or \
                (o or {}).get('reason') or 'not timed (see panel log)'
        cell['operator_settings'].append(rec)
    cell['complete'] = bool(r.get('complete')) and all(v is not False for v in gates.values()) and (drv / 'opscore.json').exists() \
        and meta(job)['exit'] == 0
    return cell


def heat(job, n):
    j = meta(job)
    cell = dict(pde='Heat 2D', mesh=n, panel_job=j['slurm_id'], gpu=j['gpu'], cohort='held-out', panel_kind='expanded ladder',
                error_metric='worst over the 16 held-out cases of max over all six times of the current-relative same-grid L2 error vs the exact DST flow',
                complete=False, nmrom=[], fom=[], operators={}, operator_settings=[])
    hb = load(RUNS / job / 'pull/out/hbk/results.json')
    op = load(RUNS / job / 'pull/out/ops/results.json')
    if hb:
        for r in hb['meshes'][0]['rows']:
            case_max = [max(s) for s in r['same']]
            d = dict(setting=r['method'], worst_percent=100 * max(case_max), median_percent=100 * float(np.median(case_max)),
                     gpu_ms=float(np.median(r['device_ms'])))
            if r['kind'] == 'model':
                d.update(kind='head' if r['method'].startswith('nmrom') else 'span', R_prime=int(r['method'].split('_R')[1].split('_')[0]))
                cell['nmrom'].append(d)
            else:
                d.update(unstable=int(sum(r['failures'])) > 0, is_reference=r['method'] == 'dst_exact_CONTROL')
                cell['fom'].append(d)
    cfg = load(G / 'configs' / f'{job}_ops.json') or {}
    for name, spec in cfg.get('operators', {}).items():
        a = (op or {}).get('arms', {}).get(name)
        tr = train_info(dict(name=name.replace('op_', ''), source=spec.get('source', ''))) if spec.get('source', '').startswith('g2d') else dict(source=spec.get('source'))
        rec = dict(family=family_of(name), setting=name.replace('op_', ''), params=None, worst_percent=None, median_percent=None, gpu_ms=None,
                   ok=False, failure=None, training=tr)
        if a:
            case_max = [max(s) for s in a['same']]
            rec.update(ok=True, params=a.get('parameters'), worst_percent=100 * max(case_max), median_percent=100 * float(np.median(case_max)),
                       gpu_ms=float(np.median(a['device_ms'])), dtype=a.get('parameter_dtype'), checkpoint_sha256=a.get('checkpoint_sha256'))
            if tr.get('best_checkpoint_sha256'):
                rec['checkpoint_matches_training'] = tr['best_checkpoint_sha256'] == a.get('checkpoint_sha256')
            elif spec.get('sha256'):
                rec['checkpoint_matches_training'] = spec['sha256'] == a.get('checkpoint_sha256')
        else:
            missing = [m for m in (op or {}).get('operators_missing', []) if m['name'] == name]
            rec['failure'] = (tr.get('reason') and f"training failed: {tr['reason']}") or ('checkpoint missing' if missing else 'not timed (panel not run / failed)')
        cell['operator_settings'].append(rec)
    cell['complete'] = bool(hb and hb.get('complete') and op and op.get('complete')) and j['exit'] == 0
    if op:
        cell['ops_order_gate'] = op.get('gates', {}).get('order_effect')
    return cell


def finish(cell):
    for fam in ('FNO', 'U-Net', 'Transolver', 'DeepONet'):
        s = [o for o in cell['operator_settings'] if o['family'] == fam]
        good = [o for o in s if o['ok']]
        if good:
            b = min(good, key=lambda o: o['worst_percent'])
            cell['operators'][fam] = dict(ok=True, worst_percent=b['worst_percent'], gpu_ms=b['gpu_ms'], failure=None, setting=b['setting'])
        else:
            cell['operators'][fam] = dict(ok=False, worst_percent=None, gpu_ms=None,
                                          failure='; '.join(f"{o['setting']}: {o['failure']}" for o in s) or 'no operator of this family in the cell')
    fom = [f for f in cell['fom'] if not f['is_reference'] and not f['unstable']]
    cell['cost_to_target'] = {str(t): dict(
        nmrom=cheapest(cell['nmrom'], t) and {k: cheapest(cell['nmrom'], t)[k] for k in ('setting', 'worst_percent', 'gpu_ms')},
        fom=cheapest(fom, t) and {k: cheapest(fom, t)[k] for k in ('setting', 'worst_percent', 'gpu_ms')},
        **{fam: (lambda c: c and {k: c[k] for k in ('setting', 'worst_percent', 'gpu_ms')})(
            cheapest([o for o in cell['operator_settings'] if o['family'] == fam and o['ok']], t))
           for fam in ('FNO', 'U-Net', 'Transolver', 'DeepONet')}) for t in TARGETS}
    return cell


def main():
    cells = []
    for L in (256, 512, 1024, 2048, 4096):
        if (RUNS / f'pl{L}' / 'pull/out/drv/result.json').exists() or L != 512:
            cells.append(finish(burgers(f'pl{L}', L)))
        else:
            cells.append(finish(burgers('pb512', 512, narrow=True)))
    for n in (256, 1024):   # user: Burgers only from ~22:10; heat 512/2048/4096 cancelled, only completed heat cells kept
        job = f'phl{n}'
        if n == 4096:
            job = 'phl4096b' if (RUNS / 'phl4096b' / 'pull/out/hbk/results.json').exists() else 'phl4096'
        c = heat(job, n)
        if job == 'phl4096':
            c['note'] = ('phl4096 (job 4303352): NM-ROM/FOM complete, operator process OOM (JAX preallocation in the operator process, a g2d '
                         'bug); rerun phl4096b pending; operator entries below are therefore failures of that job')
        if n in (256, 512):
            c['note'] = 'no Table 1 row at this mesh (frozen mesh-independent heat model evaluated here at the coordinator\'s request)'
        cells.append(finish(c))
    (G / 'cost_points.json').write_text(json.dumps(dict(group='g2d', generated_by='scripts/summarize_cost.py', targets_percent=list(TARGETS),
                                                        cells=cells), indent=1) + '\n')
    for c in cells:
        print(c['pde'], c['mesh'], 'complete' if c['complete'] else 'INCOMPLETE', c.get('panel_kind'), c['panel_job'], c['gpu'])
        for t, v in c['cost_to_target'].items():
            print('  target', t, '%:', {k: (x['setting'], round(x['worst_percent'], 3), round(x['gpu_ms'], 2)) if x else None for k, x in v.items()})


if __name__ == '__main__':
    main()
