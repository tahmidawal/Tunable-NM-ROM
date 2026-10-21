"""g2d: build results.json and REPORT.md tables from the pulled job records (runs/<job>/pull/...). No number is typed.

    /home/tahmid/Dev/.venv/bin/python scripts/summarize.py

Rules (OPS-ALL-PROTOCOL.md, identical to the source lanes' audits):
* Burgers error: worst / median over dev6 of the per-case max_{k>=1} same-grid error / ||u0|| (the drivers' own
  `same_grid_evolved`; operators: scripts/opscore.py, same formula, same in-job reference arrays).
  GPU ms = median over every timed invocation of the arm (pooled over cases, repetitions and phases).
* Heat error: worst / median over the 16 held-out cases of the per-case max over all six times of the current-relative
  error against the same-grid DST flow (`same` rows of hbk_run.py / panel.py). GPU ms = median of all timed samples.
* FOM of the cell: the fastest timed full-order setting (converged / no CG failures; controls excluded) whose worst
  error is <= the NM-ROM accurate setting's worst error. Every speedup = that FOM's ms / the method's ms, same job.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / 'runs'
OPS_B = [('FNO', 'fno-large'), ('U-Net', 'unet-refine'), ('Transolver', 'tsol-refine'), ('DeepONet', 'don-small')]
OPS_H = [('FNO', 'fno'), ('U-Net', 'unet'), ('Transolver', 'transolver'), ('DeepONet', 'deeponet')]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def job_meta(job):
    out = RUNS / job / 'pull' / 'job.out'
    sub = RUNS / job / 'SUBMIT.log'
    m = dict(job=job, slurm_id=None, gpu=None, backend_gpu=None, exit=None)
    if sub.exists():
        t = sub.read_text().split()
        m['slurm_id'] = t[-1] if t else None
    if out.exists():
        txt = out.read_text(errors='replace')
        m['backend_gpu'] = 'jax_backend=gpu' in txt
        for line in txt.splitlines():
            if line.startswith('NVIDIA') and m['gpu'] is None:
                m['gpu'] = line.split(',')[0]
            if line.startswith('host='):
                m['host'] = line.split()[0][5:]
            if line.startswith('run_exit='):
                m['exit'] = int(line.split('=')[1])
    return m


def train_record_burgers(job, arm, base='train'):
    d = RUNS / job / 'pull' / 'out' / base / arm
    r = load(d / 'result.json')
    meta = job_meta(job)
    if r is None:
        err = None
        w = load(RUNS / job / 'pull' / 'out' / 'worker.json')
        if w:
            for a in w.get('arms', []):
                if a.get('arm') == arm and not a.get('ok', True):
                    err = f"{a.get('error_type')}: {a.get('error', '')[:300]}"
        log = RUNS / job / 'pull' / 'out' / 'logs' / f'{arm}.log'
        if err is None and log.exists():
            tail = log.read_text(errors='replace').strip().splitlines()[-3:]
            err = ' | '.join(tail)[-400:]
        return dict(ok=False, train_job=meta, reason=err or 'no result.json')
    return dict(ok=True, train_job=meta, epochs=r['epochs_completed'], best_epoch=r['best_epoch'], stop_reason=r['stop_reason'],
                micro_batch_final=r.get('micro_batch_final'), training_seconds=r['training_seconds'],
                parameters=r['real_parameter_count'], validation_mean_case_max=r['validation']['mean_case_max'],
                best_checkpoint_sha256=r['best_checkpoint_sha256'])


def train_record_heat(job, arm):
    r = load(RUNS / job / 'pull' / 'out' / 'tr' / arm / 'result.json')
    meta = job_meta(job)
    if r is None:
        err = RUNS / job / 'pull' / 'job.err'
        tail = err.read_text(errors='replace').strip().splitlines()[-2:] if err.exists() else []
        return dict(ok=False, train_job=meta, reason=' | '.join(tail)[-400:] or 'no result.json')
    return dict(ok=True, train_job=meta, epochs=r['epochs_completed'], best_epoch=r['best_epoch'], stop_reason=r['stop_reason'],
                micro_batch_final=r['micro_batch'], training_seconds=r['training_seconds'], parameters=r['real_parameter_count'],
                validation_mean_case_max=r['validation_best_checkpoint']['mean_case_max'],
                best_checkpoint_sha256=r['best_checkpoint_sha256'])


def burgers_cell(label, mesh, panel, train_jobs, table1):
    cell = dict(cell=label, pde='Burgers 2D', mesh=mesh, panel_job=job_meta(panel), cohort='dev6 (6 development cases)',
                table1_source=table1)
    drv = RUNS / panel / 'pull' / 'out' / 'drv'
    r = load(drv / 'result.json')
    ops = load(drv / 'opscore.json') or {}
    if r is None:
        cell['status'] = 'panel not run / not collected'
    else:
        cell['driver_complete'] = bool(r.get('complete'))
        cell['driver_gates'] = {k: v.get('passed') for k, v in r.get('gates', {}).items() if isinstance(v, dict)}
        cell['result_sha256'] = sha(drv / 'result.json')
        per = {}
        for q in r['quick']:
            per.setdefault(q['name'], {})[q['case']] = q
        inv = {}
        for x in r['invocations']:
            inv.setdefault(x['name'], []).append(x['gpu_seconds'])
        rows = {}
        for name, pc in per.items():
            ev = [pc[c]['same_grid_evolved'] for c in sorted(pc)]
            fam = next(iter(pc.values()))['family']
            rows[name] = dict(family=fam, worst_percent=100 * max(ev), median_percent=100 * float(np.median(ev)),
                              gpu_ms=1e3 * float(np.median(inv[name])) if name in inv else None, samples=len(inv.get(name, [])),
                              converged=all(v.get('nonlinear_converged', True) for v in pc.values()))
        acc = next(n for n in rows if n.startswith('R384_lin'))
        fast = next(n for n in rows if n.startswith('R128_lin'))
        foms = {n: v for n, v in rows.items() if v['family'] == 'fom' and v['gpu_ms'] is not None and v['converged']}
        ok = [n for n in foms if foms[n]['worst_percent'] <= rows[acc]['worst_percent']]
        f = min(ok, key=lambda n: foms[n]['gpu_ms']) if ok else None
        fms = foms[f]['gpu_ms'] if f else None
        sp = (lambda ms: fms / ms if (fms and ms) else None)
        cell['fom'] = dict(setting=f, worst_percent=foms[f]['worst_percent'] if f else None,
                           median_percent=foms[f]['median_percent'] if f else None, gpu_ms=fms,
                           rule='fastest timed converged FOM setting with worst error <= NM-ROM accurate worst error')
        cell['fom_grid'] = {n: dict(worst_percent=v['worst_percent'], gpu_ms=v['gpu_ms']) for n, v in rows.items() if v['family'] == 'fom'}
        for role, n in (('nmrom_accurate', acc), ('nmrom_fast', fast)):
            cell[role] = dict(arm=n, worst_percent=rows[n]['worst_percent'], median_percent=rows[n]['median_percent'],
                              gpu_ms=rows[n]['gpu_ms'], speedup=sp(rows[n]['gpu_ms']), samples=rows[n]['samples'])
    cell['operators'] = {}
    for fam, arm in OPS_B:
        tj, base = train_jobs[arm]
        tr = train_record_burgers(tj, arm, base)
        o = ops.get(arm)
        rec = dict(arm=arm, config=f'inputs/opconfigs/{arm}.json (burgers-compare-hires)', training=tr)
        if o and o.get('ok') is not False and 'worst_evolved_percent' in o:
            fms = cell.get('fom', {}).get('gpu_ms')
            rec.update(ok=True, worst_percent=o['worst_evolved_percent'], median_percent=o['median_evolved_percent'],
                       gpu_ms=o['median_gpu_ms'], speedup=(fms / o['median_gpu_ms']) if fms else None, gpu_name=o['gpu_name'],
                       t0_returned_exactly=o['t0_returned_exactly'], repetitions=o['repetitions'],
                       checkpoint_matches_training=(o['checkpoint_sha256'] == tr.get('best_checkpoint_sha256')))
        else:
            rec.update(ok=False, reason=('training failed: ' + str(tr.get('reason'))) if not tr['ok'] else
                       (o or {}).get('reason', 'not timed (panel not run or operator failed at inference; see panel log)'))
        cell['operators'][fam] = rec
    return cell


def heat_cell(label, mesh, panel, train_jobs, table1):
    cell = dict(cell=label, pde='Heat 2D', mesh=mesh, panel_job=job_meta(panel),
                cohort='sealed held-out, seed 791099, 16 cases', table1_source=table1)
    hb = load(RUNS / panel / 'pull' / 'out' / 'hbk' / 'results.json')
    op = load(RUNS / panel / 'pull' / 'out' / 'ops' / 'results.json')
    if hb is None:
        cell['status'] = 'panel not run / not collected'
    else:
        cell['hbk_complete'] = bool(hb.get('complete'))
        mesh_rec = hb['meshes'][0]
        rows = {}
        for r in mesh_rec['rows']:
            case_max = [max(s) for s in r['same']]
            rows[r['method']] = dict(kind=r['kind'], worst_percent=100 * max(case_max), median_percent=100 * float(np.median(case_max)),
                                     gpu_ms=float(np.median(r['device_ms'])), failures=int(sum(r['failures'])),
                                     fingerprint_mismatch=r['fingerprint_mismatch'], samples=len(r['device_ms']))
        acc, fast = rows['lin_R128_cn'], rows['lin_R48_cn']
        foms = {n: v for n, v in rows.items() if v['kind'] == 'fom' and not n.endswith('_CONTROL') and v['failures'] == 0}
        ok = [n for n in foms if foms[n]['worst_percent'] <= acc['worst_percent']]
        f = min(ok, key=lambda n: foms[n]['gpu_ms']) if ok else None
        fms = foms[f]['gpu_ms'] if f else None
        cell['fom'] = dict(setting=f, worst_percent=foms[f]['worst_percent'] if f else None, median_percent=foms[f]['median_percent'] if f else None,
                           gpu_ms=fms, rule='fastest CN-CG setting with no CG failure and worst error <= NM-ROM accurate worst error')
        cell['fom_grid'] = {n: dict(worst_percent=v['worst_percent'], gpu_ms=v['gpu_ms'], failures=v['failures']) for n, v in rows.items() if v['kind'] == 'fom'}
        for role, n in (('nmrom_accurate', 'lin_R128_cn'), ('nmrom_fast', 'lin_R48_cn')):
            v = rows[n]
            cell[role] = dict(arm=n, worst_percent=v['worst_percent'], median_percent=v['median_percent'], gpu_ms=v['gpu_ms'],
                              speedup=(fms / v['gpu_ms']) if fms else None, samples=v['samples'], fingerprint_mismatch=v['fingerprint_mismatch'])
        nb = mesh_rec.get('neighbour', {})
        cell['order_neighbour_ratio_max'] = max((float(np.median([x['ms'] for x in v])) / rows[k]['gpu_ms'] for k, v in nb.items()), default=None)
    cell['operators'] = {}
    for fam, arm in OPS_H:
        tr = train_record_heat(train_jobs[arm], arm)
        rec = dict(arm=arm, config=f'configs/ops/{arm}.json (heat-compare-hires)', training=tr)
        a = (op or {}).get('arms', {}).get(f'op_{arm}')
        fms = cell.get('fom', {}).get('gpu_ms')
        if a:
            case_max = [max(s) for s in a['same']]
            ms = float(np.median(a['device_ms']))
            rec.update(ok=True, worst_percent=100 * max(case_max), median_percent=100 * float(np.median(case_max)), gpu_ms=ms,
                       speedup=(fms / ms) if fms else None, checkpoint_matches_training=(a.get('checkpoint_sha256') == tr.get('best_checkpoint_sha256')),
                       best_epoch=a.get('best_epoch'))
        else:
            rec.update(ok=False, reason=('training failed: ' + str(tr.get('reason'))) if not tr['ok'] else 'not timed (panel not run or failed)')
        cell['operators'][fam] = rec
    if op:
        cell['ops_panel_complete'] = bool(op.get('complete'))
        cell['ops_order_gate'] = op.get('gates', {}).get('order_effect')
    return cell


def fmt(x, nd=3):
    if x is None:
        return '—'
    return f'{x:.{nd}g}' if abs(x) < 1000 else f'{x:.0f}'


def table(cells):
    lines = ['| cell | panel job (GPU) | FOM setting: worst % / ms | NM-ROM accurate: worst % / ms / speedup | NM-ROM fast: worst % / ms / speedup | '
             'FNO | U-Net | Transolver | DeepONet |', '|' + '---|' * 9]
    for c in cells:
        pj = c['panel_job']
        f = c.get('fom') or {}
        a, s = c.get('nmrom_accurate') or {}, c.get('nmrom_fast') or {}
        row = [c['cell'], f"{pj.get('slurm_id') or '—'} ({pj.get('gpu') or '—'})",
               f"`{f.get('setting')}`: {fmt(f.get('worst_percent'))} / {fmt(f.get('gpu_ms'))}" if f else '—',
               f"{fmt(a.get('worst_percent'))} / {fmt(a.get('gpu_ms'))} / {fmt(a.get('speedup'))}×" if a else '—',
               f"{fmt(s.get('worst_percent'))} / {fmt(s.get('gpu_ms'))} / {fmt(s.get('speedup'))}×" if s else '—']
        for fam in ('FNO', 'U-Net', 'Transolver', 'DeepONet'):
            o = c['operators'][fam]
            if o.get('ok'):
                row.append(f"{fmt(o['worst_percent'])} / {fmt(o['median_percent'])} / {fmt(o['gpu_ms'])} ms ({fmt(o['speedup'])}×)")
            else:
                row.append('FAILED: ' + str(o.get('reason'))[:160].replace('|', '/'))
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def training_table(cells):
    lines = ['| cell | family | train job (GPU) | epochs (best) | stop | micro-batch | train s | params | val mean-case-max |', '|' + '---|' * 9]
    for c in cells:
        for fam, o in c['operators'].items():
            t = o['training']
            tj = t['train_job']
            if t['ok']:
                lines.append(f"| {c['cell']} | {fam} | {tj.get('slurm_id')} ({tj.get('gpu')}) | {t['epochs']} ({t['best_epoch']}) | {t['stop_reason']} | "
                             f"{t['micro_batch_final']} | {t['training_seconds']:.0f} | {t['parameters']} | {t['validation_mean_case_max']:.4g} |")
            else:
                lines.append(f"| {c['cell']} | {fam} | {tj.get('slurm_id')} ({tj.get('gpu')}) | — | FAILED | — | — | — | {str(t.get('reason'))[:200].replace('|', '/')} |")
    return '\n'.join(lines)


def main():
    cells = [
        burgers_cell('Burgers 2D 512²', 512, 'pb512',
                     {'fno-large': ('b512fno', 'train'), 'unet-refine': ('b512unet', 'train'), 'tsol-refine': ('b512tsol', 'train'),
                      'don-small': ('b512don', 'train')},
                     'burgers2d-speed b512, job 4241035 (selection-512.json)'),
        burgers_cell('Burgers 2D 4096²', 4096, 'pb4096', {a: ('b4096tr', 'train') for _, a in OPS_B},
                     'burgers-bank-knob bk4096b, job 4197473 (k>=j+1 sensitivity arm = paper row)'),
        heat_cell('Heat 2D 4096²', 4096, 'ph4096', {'fno': 'h4096fno', 'unet': 'h4096unet', 'transolver': 'h4096tran', 'deeponet': 'h4096deepb'},
                  'heat-bank-knob h2d, job 4197350'),
    ]
    (HERE / 'results.json').write_text(json.dumps(dict(group='g2d', generated_by='scripts/summarize.py', cells=cells), indent=1) + '\n')
    (HERE / 'tables.md').write_text('## Per-cell results\n\nOperator columns: worst % / median % / median GPU ms (speedup vs the cell\'s FOM).\n\n'
                                    + table(cells) + '\n\n## Operator training\n\n' + training_table(cells) + '\n')
    print(table(cells))
    print()
    print(training_table(cells))


if __name__ == '__main__':
    main()
