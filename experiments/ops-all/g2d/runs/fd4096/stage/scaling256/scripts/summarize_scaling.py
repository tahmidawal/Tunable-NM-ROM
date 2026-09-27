"""g2d scaling256: results.json from the pulled records (runs/s<N><f|u>, runs/psc256) + a data_scaling cell appended to
../cost_points.json. No number is typed.

    /home/tahmid/Dev/.venv/bin/python scaling256/scripts/summarize_scaling.py
"""
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parents[1]
G = S.parent
RUNS = G / 'runs'
JOBS = {(a, n): f's{n}{k}' for n in (128, 576, 2048) for a, k in (('fno-w96f32', 'f'), ('unet-b64', 'u'))}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def meta(job):
    m = dict(slurm_id=None, gpu=None, exit=None)
    sub = RUNS / job / 'SUBMIT.log'
    if sub.exists() and sub.read_text().split():
        m['slurm_id'] = sub.read_text().split()[-1]
    out = RUNS / job / 'pull/job.out'
    if out.exists():
        for line in out.read_text(errors='replace').splitlines():
            if line.startswith('NVIDIA') and m['gpu'] is None:
                m['gpu'] = line.split(',')[0]
            if line.startswith('run_exit='):
                m['exit'] = int(line.split('=')[1])
            if line.startswith('jax_backend='):
                m['backend'] = line.strip()
    return m


def stats(v):
    v = np.asarray(v, float)
    return dict(worst_percent=100 * float(v.max()), mean_percent=100 * float(v.mean()), median_percent=100 * float(np.median(v)), cases=int(v.size))


def main():
    drv = RUNS / 'psc256/pull/out/drv'
    r = load(drv / 'result.json')
    ops = load(drv / 'opscore.json') or {}
    nets = []
    for (arm, n), job in JOBS.items():
        sc = load(RUNS / job / 'pull/out/scale.json')
        rec = dict(network=arm, n_train=n, train_job=job, **meta(job))
        if sc is None:
            rec.update(ok=False, failure='no scale.json (not finished / not collected)')
        elif not sc.get('ok'):
            rec.update(ok=False, failure=f"{sc.get('error_type')}: {str(sc.get('error'))[:300]}")
        else:
            tb, vb = sc['train_best_checkpoint'], sc['validation_best_checkpoint']
            rec.update(ok=True, epochs=sc['epochs'], best_epoch=sc['best_epoch'], stop_reason=sc['stop_reason'],
                       optimisation_steps=sc['optimisation_steps'], training_seconds=sc['training_seconds'],
                       micro_batch=sc['micro_batch_final'], plateau_patience=sc['config']['plateau_patience'],
                       early_stop_patience=sc['config']['patience'], train_msre_best_epoch=sc['train_msre_best_epoch'],
                       train_set={'worst_percent': 100 * tb['worst_case_max'], 'mean_percent': 100 * tb['mean_case_max'],
                                  'median_percent': 100 * tb['median_case_max'], 'cases': n},
                       validation32={'worst_percent': 100 * vb['worst_case_max'], 'mean_percent': 100 * vb['mean_case_max'],
                                     'median_percent': 100 * vb['median_case_max'], 'cases': 32},
                       best_checkpoint_sha256=sc['best_checkpoint_sha256'],
                       curve=sc.get('curve'))
            o = ops.get(f'{arm}-N{n}')
            if o and 'per_case' in o:
                rec['dev6'] = stats([p['same_grid_evolved'] for p in o['per_case']])
                rec['gpu_ms'] = o['median_gpu_ms']
                rec['params'] = o['real_parameter_count']
                rec['panel_checkpoint_matches_training'] = o['checkpoint_sha256'] == sc['best_checkpoint_sha256']
                rec['panel_gpu'] = o['gpu_name']
        nets.append(rec)
    nm, fom = [], []
    if r:
        per, inv = {}, {}
        for q in r['quick']:
            per.setdefault(q['name'], {})[q['case']] = q
        for x in r['invocations']:
            inv.setdefault(x['name'], []).append(x['gpu_seconds'])
        setup = {a['arm']: a for a in r['arm_setup']}
        for n, pc in per.items():
            ev = [pc[c]['same_grid_evolved'] for c in sorted(pc)]
            ms = 1e3 * float(np.median(inv[n])) if n in inv else None
            if next(iter(pc.values()))['family'] == 'rom':
                s = setup[n]
                nm.append(dict(setting=n, kind='head' if s.get('model') == 'trunc' else 'span', R_prime=s.get('R_prime'), gpu_ms=ms,
                               dev6=stats(ev)))
            else:
                fom.append(dict(setting=n, gpu_ms=ms, dev6=stats(ev),
                                unstable=not all(v.get('nonlinear_converged', True) for v in pc.values()),
                                is_reference=n.split('__')[0] == 'fft_tight'))
    pm = meta('psc256')
    res = dict(study='g2d scaling256 (Burgers 2D 256^2 data scaling)', design='scaling256/DESIGN.md', panel_job=pm,
               panel_complete=bool(r and r.get('complete') and pm.get('exit') == 0 and (drv / 'opscore.json').exists()),
               panel_gates={k: v.get('passed') for k, v in (r or {}).get('gates', {}).items() if isinstance(v, dict)},
               metric='per case: max over evolved times of ||u-u_ref||_2/||u_0||_2 (same grid, fft_tight); then worst/mean/median over cases',
               networks=nets, nmrom=nm, fom=fom)
    (S / 'results.json').write_text(json.dumps(res, indent=1) + '\n')
    # cost_points: separate data_scaling cell
    cp = json.loads((G / 'cost_points.json').read_text())
    cp['cells'] = [c for c in cp['cells'] if c.get('variant') != 'data_scaling']
    cp['cells'].append(dict(
        pde='Burgers 2D', mesh=256, variant='data_scaling', complete=res['panel_complete'], panel_job=pm.get('slurm_id'), gpu=pm.get('gpu'),
        cohort='development', error_metric='worst over the 6 dev cases of max over evolved times of ||u-u_ref||_2/||u_0||_2 (same grid, in-job fft_tight)',
        nmrom=[dict(setting=x['setting'], kind=x['kind'], R_prime=x['R_prime'], worst_percent=x['dev6']['worst_percent'], gpu_ms=x['gpu_ms']) for x in nm],
        fom=[dict(setting=x['setting'], worst_percent=x['dev6']['worst_percent'], gpu_ms=x['gpu_ms'], unstable=x['unstable'],
                  is_reference=x['is_reference']) for x in fom if x['gpu_ms'] is not None],
        operator_settings=[dict(family='FNO' if x['network'].startswith('fno') else 'U-Net', setting=f"{x['network']}-N{x['n_train']}",
                                params=x.get('params'), worst_percent=(x.get('dev6') or {}).get('worst_percent'), gpu_ms=x.get('gpu_ms'),
                                ok=bool(x.get('dev6')), failure=None if x.get('dev6') else x.get('failure', 'not timed'),
                                n_train=x['n_train']) for x in nets],
        operators={fam: (lambda good: dict(ok=True, worst_percent=min(good)[0], gpu_ms=min(good)[1], failure=None, setting=min(good)[2]) if good
                         else dict(ok=False, worst_percent=None, gpu_ms=None, failure='none timed'))(
            [(x['dev6']['worst_percent'], x['gpu_ms'], f"{x['network']}-N{x['n_train']}") for x in nets
             if x.get('dev6') and (x['network'].startswith('fno') == (fam == 'FNO'))]) for fam in ('FNO', 'U-Net')},
        note='data-scaling variant (scaling256/DESIGN.md); does not replace the pl256 cell'))
    (G / 'cost_points.json').write_text(json.dumps(cp, indent=1) + '\n')
    for x in nets:
        if x.get('ok'):
            print(f"{x['network']:11s} N={x['n_train']:5d} ep {x['epochs']:4d} best {x['best_epoch']:4d} {x['stop_reason']:14s} "
                  f"train {x['train_set']['worst_percent']:.2f}/{x['train_set']['mean_percent']:.2f}/{x['train_set']['median_percent']:.2f} "
                  f"val32 {x['validation32']['worst_percent']:.2f}/{x['validation32']['mean_percent']:.2f}/{x['validation32']['median_percent']:.2f} "
                  + (f"dev6 {x['dev6']['worst_percent']:.2f}/{x['dev6']['mean_percent']:.2f}/{x['dev6']['median_percent']:.2f} {x['gpu_ms']:.2f} ms" if x.get('dev6') else ''))
        else:
            print(x['network'], x['n_train'], 'FAILED', x.get('failure'))
    for x in nm:
        if x['kind'] == 'span' and x['R_prime'] in (384, 128, 64) or x['kind'] == 'head' and x['R_prime'] == 512:
            print(x['setting'], {k: round(v, 3) for k, v in x['dev6'].items() if k != 'cases'}, round(x['gpu_ms'], 2) if x['gpu_ms'] else None)


if __name__ == '__main__':
    main()
