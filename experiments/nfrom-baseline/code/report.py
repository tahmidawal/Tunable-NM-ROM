"""Tables for the report, read only from saved JSON (results/ pulled from the cluster, inputs/, stored lanes).

  report.py --results results --worktrees /home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees --out reports/tables.md
"""
import argparse, json
from pathlib import Path


def f(x, n=3):
    return '—' if x is None else f'{x:.{n}f}'


ROUND = {'models': '120k', 'models-x': '600k'}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--results', required=True); a.add_argument('--worktrees', required=True); a.add_argument('--out', required=True)
    a = a.parse_args()
    R, W = Path(a.results), Path(a.worktrees)
    here = Path(__file__).resolve().parent.parent
    out = []
    summary = {}

    # ---- paper rows (stored)
    ref = json.loads((here / 'inputs/rom_reference.json').read_text())['meshes']['256']
    t2p = W / '2026-09-25-t2-burgers-test/experiments/t2-burgers-test/checks/t256-summary.json'
    t2 = json.loads(t2p.read_text())
    arms = {x['name']: x for x in t2['arms']}
    head = arms['q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2']
    paper = [('paper: bank + span, R\'=384 (accurate)', ref['roles']['accurate'], 'job ' + ref['job']),
             ('paper: bank + span, R\'=128', ref['roles']['extra'], 'job ' + ref['job']),
             ('paper: bank + span, R\'=96 (fast)', ref['roles']['fast'], 'job ' + ref['job'])]
    rows = [(n, r['worst_evolved_percent'], r['median_evolved_percent'], r['median_gpu_ms'], None, src) for n, r, src in paper]
    rows.append(('paper: bank + nonlinear head, k=16', head['worst_evolved_percent'], head['median_evolved_percent'], head['gpu_ms'], None, 'job ' + str(t2['job_id'])))
    rows.append(('full-order solver (fastest at <= 0.875 %)', ref['table1']['fom_worst_evolved_percent'], ref['table1']['fom_median_evolved_percent'],
                 ref['table1']['fom_gpu_ms'], None, 'job ' + ref['job']))
    summary['paper_rows'] = [dict(name=r[0], worst=r[1], median=r[2], ms=r[3], source=r[5]) for r in rows]

    # ---- test rows of this lane
    tp = R / 'test.json'
    if tp.exists():
        t = json.loads(tp.read_text())
        mine = []
        for r in t['results']:
            s, c = r['selection'], r['cfg']
            label = {'C': 'CROM-style', 'L': 'CoLoRA-style'}[c['arm']]
            what = (f"hyper-network" if s['kind'] == 'hyper' else f"collocation LM, M={s['M']}, {s['iters']} it/step" + (', refined z0' if s['refine'] else ''))
            name = f"{label} k={c['k']} ({s['name']}, {ROUND[s['models_dir']]} steps; {what}; {s['dtype']}; {s['role']})"
            mine.append((name, r['test']['worst_percent'], r['test']['median_percent'], r['ms_full']['median'],
                         (r['ms_full']['p5'], r['ms_full']['p95'], r['ms_latent_only']['median']), 'job ' + str(t['job_id']), r['test']['nonfinite'], s['worst_percent']))
        out.append(f"### Test cohort (test64, 256 mesh, {t['gpu']}; one evaluation, job {t['job_id']})\n")
        out.append('| method | worst error % | median error % | time per query, ms (median) | p5–p95 ms | latent-only ms | validation worst % | source |')
        out.append('|---|---|---|---|---|---|---|---|')
        for n, w, m, ms, sp, src, nonf, vw in mine:
            out.append(f"| {n} | {f(w)}{' (+%d non-finite)' % nonf if nonf else ''} | {f(m)} | {f(ms, 1)} | {f(sp[0], 1)}–{f(sp[1], 1)} | {f(sp[2], 1)} | {f(vw)} | {src} |")
        for n, w, m, ms, sp, src in rows:
            out.append(f"| {n} | {f(w)} | {f(m)} | {f(ms, 1)} | | | | {src} (stored) |")
        summary['test_rows'] = [dict(name=x[0], worst=x[1], median=x[2], ms=x[3], p5=x[4][0], p95=x[4][1], latent_ms=x[4][2], validation_worst=x[7]) for x in mine]
        out.append('')

    # ---- other learned baselines already stored for the same cohort and mesh
    out.append(f"### Stored rows, same cohort and mesh (job {t2['job_id']}, {t2['gpu']})\n")
    out.append('| method | worst % | median % | ms |\n|---|---|---|---|')
    for n in ('fno-large', 'unet-refine', 'tsol-small', 'don-small'):
        x = arms[n]; out.append(f"| {n} | {f(x['worst_evolved_percent'])} | {f(x['median_evolved_percent'])} | {f(x['gpu_ms'], 1)} |")
    out.append('')

    # ---- training
    tr = sorted((R / 'train').glob('*/done.json')) + sorted((R / 'train-x').glob('*/done.json'))
    if tr:
        out.append('### Offline training (validation = 64 held-out training-pool trajectories, reconstruction (C) / prediction (L) error)\n')
        out.append('| config | steps | arm | k | parameters | lr | val mean % | val evolved worst % | train time, min | steps/s (last) | plateau | GPU | job |')
        out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        summary['training'] = []
        for p in tr:
            d = json.loads(p.read_text()); c = d['cfg']; h = d['final']
            out.append(f"| {d['name']} | {d['steps']:,} | {c['arm']} | {c['k']} | {d['parameters']:,} | {c['lr']:g} | {f(100 * h['val_mean'])} | {f(100 * h['val_evolved_worst'])} | "
                       f"{f(d['train_seconds'] / 60, 1)} | {f(h['steps_per_second'], 0)} | {'yes' if d['at_plateau'] else 'no (%.1f %% over last tenth)' % (100 * (d['improvement_over_last_tenth'] or 0))} | {d['gpu']} | {d['job_id']} |")
            summary['training'].append(dict(name=d['name'], steps=d['steps'], val_mean=h['val_mean'], val_evolved_worst=h['val_evolved_worst'], seconds=d['train_seconds'], at_plateau=d['at_plateau']))
        out.append(f"\nTotal training time of all runs listed: {sum(json.loads(p.read_text())['train_seconds'] for p in tr) / 3600:.2f} GPU-hours.\n")

    # ---- validation knob grids
    groups = [('solver v2 (used for selection)', ['grid', 'grid-x']),
              ('arm L with the online residual solve instead of the hyper-network (solver v2; not used for selection of hyper rows)', ['grid-x-lres']),
              ('solver v1 (superseded by amendment A3; kept for the record)', ['grid-v1', 'grid-x-v1'])]
    for title, dirs in groups:
        gr = [p for d in dirs for p in sorted((R / d).glob('*.json'))]
        if not gr:
            continue
        out.append(f'### Validation knob grid, {title}\n')
        out.append('First 32 validation trajectories; times are rough batch-1 timings on the GPU named in the row, used for selection only.\n')
        out.append('| decoder | steps | kind | M | LM it/step | refined z0 | precision | worst % | median % | rough ms | non-finite | GPU | job |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for p in gr:
            d = json.loads(p.read_text())
            for r in d['rows']:
                if p.parent.name.endswith('lres') and r['kind'] == 'hyper':
                    continue
                out.append(f"| {d['name']} | {d['checkpoint_step']:,} | {r['kind']} | {r['M'] or '—'} | {r['iters'] or '—'} | {'yes' if r['refine'] else 'no'} | {r['dtype']} | {f(r['worst_percent'])} | {f(r['median_percent'])} | {f(r['rough_ms'], 1)} | {r['nonfinite']} | {d.get('gpu')} | {d.get('job_id')} |")
        out.append('')
    Path(a.out).write_text('\n'.join(out) + '\n')
    (Path(a.out).parent / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
