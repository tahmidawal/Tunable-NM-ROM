"""Build reports/summary.json, reports/tables.md and the figures from saved JSON only.

Inputs (all small, pulled from the cluster):
  inputs/rom_reference.json            ROM and FOM rows of the paper's test (lane 2026-09-25-burgers2d-test)
  inputs/test-results.json             output of code/evaluate.py on test64 (one evaluation)
  inputs/runs/<job>/<run>/history.json and result.json (or meta.json when a run did not finish)
Nothing here is typed by hand: every number in tables.md is read from those files.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
INP = LANE / 'inputs'
FAMILY = {'F': 'FNO', 'U': 'U-Net', 'T': 'Transolver'}
T2 = json.loads((INP / 'rom_reference_t2.json').read_text())['meshes']     # stored rows of job 4327396 / 4327389
PAPER = {L: {fam: T2[L]['rows'][k] for fam, k in (('FNO', 'fno'), ('U-Net', 'unet'), ('Transolver', 'tsol'))} for L in ('256', '1024')}
COLORS = {'F0': '#1f77b4', 'F1': '#17becf', 'F2': '#9467bd', 'U0': '#d62728', 'U1': '#ff7f0e', 'U2': '#8c564b', 'T0': '#2ca02c'}


def f(x, n=3):
    return '—' if x is None else f'{x:.{n}g}' if abs(x) < 1 else f'{x:.{max(n - 1, 0) if abs(x) < 10 else 1}f}'


def load():
    rom = json.loads((INP / 'rom_reference.json').read_text())
    test = None
    files = sorted((INP / 'test').glob('*.json')) if (INP / 'test').exists() else []
    if files:
        test = dict(runs={}, job_id=sorted({json.loads(q.read_text())['job_id'] for q in files}),
                    gpu=sorted({json.loads(q.read_text())['gpu'] for q in files}))
        for q in files:
            r = json.loads(q.read_text())
            test['runs'].setdefault(r['key'], {}).update(r)
    runs = {}
    for d in sorted((INP / 'runs').glob('*/*')):
        if not (d / 'history.json').exists():
            continue
        h = json.loads((d / 'history.json').read_text())
        r = json.loads((d / 'result.json').read_text()) if (d / 'result.json').exists() else None
        arm, rung, seed = d.name.split('-')
        runs[f'{d.parent.name}/{d.name}'] = dict(arm=arm, rung=int(rung[1:]), seed=int(seed[1:]), history=h, result=r,
                                                 round=2 if d.parent.name.startswith('c') else 1)
    for key, r in runs.items():        # a first-round run is superseded by its continuation (amendment A5)
        r['superseded'] = r['round'] == 1 and any(k != key and k.split('/')[1] == key.split('/')[1] and q['round'] == 2
                                                  for k, q in runs.items())
    return rom, test, runs


def tail_slope(h, k=10):
    """Relative change of the best-so-far validation score over the last k validations (percent)."""
    if len(h) <= k:
        return None
    best = np.minimum.accumulate([x['val_mean'] for x in h])
    return 100 * float((best[-1 - k] - best[-1]) / best[-1 - k])


def main():
    rom, test, runs = load()
    rows = []
    for key, r in runs.items():
        h, res = r['history'], r['result']
        b = min(h, key=lambda x: x['val_mean'])
        row = dict(key=key, round=r['round'], superseded=r['superseded'], arm=r['arm'], family=FAMILY[r['arm'][0]], rung=r['rung'], seed=r['seed'],
                   finished=res is not None, stop_reason=res['stop_reason'] if res else 'no result file (job ended)',
                   converged=bool(res and res['converged']), steps=res['steps'] if res else h[-1]['step'],
                   epochs=res['epochs'] if res else None, best_step=b['step'],
                   val_mean_percent=100 * b['val_mean'], val_worst_percent=100 * b['val_worst'],
                   val_median_percent=100 * b['val_median'], final_lr=res['final_lr'] if res else h[-1]['lr'],
                   tail_gain_percent=tail_slope(h), tail_gain_20_percent=tail_slope(h, 20),
                   validations_since_best=(h[-1]['step'] - b['step']) // 500,
                   ms_per_step=res['ms_per_step'] if res else None, parameters=None)
        if test and key in test['runs']:
            t = test['runs'][key]
            row['parameters'] = t['real_parameters']
            assert t['best_step'] == b['step'], (key, t['best_step'], b['step'])
            for name in ('L256', 'L1024_zero_shot', 'L1024_coarse_bilinear'):
                if name in t and 'worst_percent' in t[name]:
                    row[name] = {k: t[name][k] for k in ('worst_percent', 'median_percent', 'mean_percent', 'timing')}
        rows.append(row)
    rows.sort(key=lambda x: (x['family'], x['rung'], x['arm'], x['seed'], x['round']))
    final = [x for x in rows if not x['superseded']]

    # selection on validation at the 4,608 rung (seed 0 of each arm), then seed statistics of the selected arm
    verdict = {}
    R = {L: rom['meshes'][L]['roles'] for L in ('256', '1024')}
    for fam in ('FNO', 'U-Net', 'Transolver'):
        cand = [x for x in final if x['family'] == fam and x['rung'] == 4608 and x['seed'] == 0]
        if not cand:
            continue
        sel = min(cand, key=lambda x: x['val_mean_percent'])
        seeds = [x for x in final if x['arm'] == sel['arm'] and x['rung'] == 4608 and 'L256' in x]
        v = dict(selected_arm=sel['arm'], grid={x['arm']: x['val_mean_percent'] for x in cand},
                 seeds=[x['seed'] for x in seeds], all_converged=all(x['converged'] for x in seeds))
        if seeds:
            for q in ('worst_percent', 'median_percent'):
                vals = [x['L256'][q] for x in seeds]
                v[q] = dict(min=min(vals), median=float(np.median(vals)), max=max(vals))
            v['time_ms'] = float(np.median([x['L256']['timing']['median_ms'] for x in seeds]))
            acc, fast = R['256']['accurate'], R['256']['fast']
            for q, rq in (('worst_percent', 'worst_evolved_percent'), ('median_percent', 'median_evolved_percent')):
                o = v[q]['median']
                v[q + '_verdict'] = dict(
                    rom_accuracy_advantage='survives' if o > acc[rq] else 'lost', ratio_to_rom_accurate=o / acc[rq],
                    more_accurate_than_rom_fast=o < fast[rq],
                    dominates_rom_fast=bool(o < fast[rq] and v['time_ms'] < fast['median_gpu_ms']),
                    ratio_to_fom=o / rom['meshes']['256']['table1'][{'worst_evolved_percent': 'fom_worst_evolved_percent',
                                                                    'median_evolved_percent': 'fom_median_evolved_percent'}[rq]])
            v['time_ratio_to_rom_accurate'] = v['time_ms'] / acc['median_gpu_ms']
            v['time_ratio_to_fom'] = v['time_ms'] / rom['meshes']['256']['table1']['fom_gpu_ms']
            # the paper-architecture arm of the family, three seeds (reported beside the selected arm, amendment A6)
            ref = [x for x in final if x['arm'] == fam[0] + '0' and x['rung'] == 4608 and 'L256' in x]
            v['paper_architecture'] = dict(arm=fam[0] + '0', seeds=[x['seed'] for x in ref],
                worst_percent=sorted(x['L256']['worst_percent'] for x in ref), median_percent=sorted(x['L256']['median_percent'] for x in ref),
                time_ms=sorted(x['L256']['timing']['median_ms'] for x in ref), converged=[x['converged'] for x in ref])
        verdict[fam] = v
    summary = dict(rom=rom, rom_t2=T2, rows=rows, verdict=verdict, paper_operator_rows=PAPER,
                   test_job=test and test.get('job_id'), test_gpu=test and test.get('gpu'))
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')

    # ------------------------------------------------------------------ tables
    T = []
    def flag(x):
        return 'yes' if x['converged'] else f"**no** ({x['stop_reason']})"
    T.append('### Table 1. Test errors and inference time at $256^2$ (test64, A100 80GB, batch 1)\n')
    T.append('| method | training trajectories | seed | params | worst % | median % | time ms (median [p05, p95; min, max] over 320 timed queries) | stopped by the rule |')
    T.append('|---|---|---|---|---|---|---|---|')
    for L in ('256',):
        t1 = rom['meshes'][L]['table1']
        for role in ('accurate', 'fast', 'extra'):
            if role in R[L]:
                x = R[L][role]
                T.append(f"| ROM {role} (`{x['name'].split('_')[0]}`) | 576 bank + 4,608 head | — | — | {f(x['worst_evolved_percent'])} | "
                         f"{f(x['median_evolved_percent'])} | {f(x['median_gpu_ms'])} | n/a |")
        T.append(f"| FOM at the ROM-accurate tolerance | — | — | — | {f(t1['fom_worst_evolved_percent'])} | "
                 f"{f(t1['fom_median_evolved_percent'])} | {f(t1['fom_gpu_ms'])} | n/a |")
    hd = T2['256']['rows']['head']
    T.append(f"| ROM head only, k=16 (stored, job {T2['256']['job']}) | 4,608 head | — | — | {f(hd['worst_percent'])} | "
             f"{f(hd['median_percent'])} | {f(hd['gpu_ms'])} | n/a |")
    for fam, e in PAPER['256'].items():
        T.append(f"| {fam}, paper row (128 traj., 3000 s cap; stored, job {T2['256']['job']}) | 128 | 0 | — | {f(e['worst_percent'])} | "
                 f"{f(e['median_percent'])} | {f(e['gpu_ms'])} | no (wall cap) |")
    for x in final:
        if 'L256' in x:
            t = x['L256']; tm = t['timing']
            T.append(f"| {x['family']} {x['arm']} | {x['rung']:,} | {x['seed']} | {x['parameters']:,} | {f(t['worst_percent'])} | "
                     f"{f(t['median_percent'])} | {f(tm['median_ms'])} [{f(tm['p05_ms'])}, {f(tm['p95_ms'])}; min {f(tm['min_ms'])}, max {f(tm['max_ms'])}] | {flag(x)} |")
    T.append('\n### Table 2. Training record of every run\n')
    T.append('Round 1 = first training; round 2 = the single continuation of amendment A5 (cumulative steps). A round-1 row marked superseded was continued and is not evaluated.\n')
    T.append('| run | job dir | round | rung | seed | steps | epochs | best step | val. mean % | val. worst % | final lr | gain of best val. over last 10 / 20 validations % | validations since best | stop | stopped by the rule |')
    T.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for x in rows:
        T.append(f"| {x['family']} {x['arm']} | {x['key'].split('/')[0]} | {x['round']}{' (superseded)' if x['superseded'] else ''} | {x['rung']:,} | {x['seed']} | {x['steps']:,} | {f(x['epochs'])} | {x['best_step']:,} | "
                 f"{f(x['val_mean_percent'], 4)} | {f(x['val_worst_percent'])} | {x['final_lr']:.2g} | {f(x['tail_gain_percent'])} / {f(x['tail_gain_20_percent'])} | "
                 f"{x['validations_since_best']} | {x['stop_reason']} | {'yes' if x['converged'] else '**no**'} |")
    T.append('\n### Table 3. $1024^2$ test (networks trained at $256^2$, no retraining)\n')
    T.append('| method | rung | seed | zero-shot worst % | zero-shot median % | zero-shot ms | coarse+bilinear worst % | coarse+bilinear median % | coarse+bilinear ms |')
    T.append('|---|---|---|---|---|---|---|---|---|')
    for role in ('accurate', 'fast'):
        x = R['1024'][role]
        T.append(f"| ROM {role} (`{x['name'].split('_')[0]}`) | — | — | {f(x['worst_evolved_percent'])} | {f(x['median_evolved_percent'])} | "
                 f"{f(x['median_gpu_ms'])} | | | |")
    t1 = rom['meshes']['1024']['table1']
    T.append(f"| FOM at the ROM-accurate tolerance | — | — | {f(t1['fom_worst_evolved_percent'])} | {f(t1['fom_median_evolved_percent'])} | {f(t1['fom_gpu_ms'])} | | | |")
    for fam, e in PAPER['1024'].items():
        T.append(f"| {fam}, paper row (trained at $1024^2$, 128 traj.; stored, job {T2['1024']['job']}) | 128 | 0 | {f(e['worst_percent'])} | {f(e['median_percent'])} | {f(e['gpu_ms'])} | | | |")
    for x in final:
        if x['rung'] == 4608 and ('L1024_zero_shot' in x or 'L1024_coarse_bilinear' in x):
            z, c = x.get('L1024_zero_shot'), x.get('L1024_coarse_bilinear')
            g = lambda t: f"{f(t['worst_percent'])} | {f(t['median_percent'])} | {f(t['timing']['median_ms'])}" if t else 'failed | | '
            T.append(f"| {x['family']} {x['arm']} | {x['rung']:,} | {x['seed']} | {g(z)} | {g(c)} |")
    T.append('\n### Table 4. Verdict inputs (PROTOCOL section 6)\n')
    T.append('| family | validation scores of the grid at 4,608 (seed 0), % | selected arm | seeds | worst % min / median / max | median % min / median / max | time ms | worst: ROM advantage | ratio to ROM accurate | median: ROM advantage | ratio | dominates ROM fast (worst) |')
    T.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for fam, v in verdict.items():
        if 'worst_percent' not in v:
            T.append(f"| {fam} | {', '.join(f'{k} {f(e)}' for k, e in v['grid'].items())} | {v['selected_arm']} | not evaluated | | | | | | | | |")
            continue
        w, m = v['worst_percent'], v['median_percent']
        wv, mv = v['worst_percent_verdict'], v['median_percent_verdict']
        T.append(f"| {fam} | {', '.join(f'{k} {f(e)}' for k, e in v['grid'].items())} | {v['selected_arm']} | {v['seeds']} | "
                 f"{f(w['min'])} / {f(w['median'])} / {f(w['max'])} | {f(m['min'])} / {f(m['median'])} / {f(m['max'])} | {f(v['time_ms'])} | "
                 f"{wv['rom_accuracy_advantage']} | {f(wv['ratio_to_rom_accurate'])}× | {mv['rom_accuracy_advantage']} | "
                 f"{f(mv['ratio_to_rom_accurate'])}× | {'yes' if wv['dominates_rom_fast'] else 'no'} |")
    T.append('\nRatios are operator / ROM accurate setting (above 1: the operator is less accurate). Time is the median over the seeds of the per-run median.')
    (HERE / 'tables.md').write_text('\n'.join(T) + '\n')

    # ------------------------------------------------------------------ figures
    (HERE / 'figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, rung in zip(axes, (128, 576, 4608)):
        for x in rows:
            if x['rung'] != rung:
                continue
            h = runs[x['key']]['history']
            ax.plot([p['step'] for p in h], [100 * p['val_mean'] for p in h], color=COLORS[x['arm']], lw=1.1,
                    alpha=1 if x['seed'] == 0 else .45, label=f"{x['arm']}" if x['seed'] == 0 else None)
            ax.plot(x['best_step'], x['val_mean_percent'], 'o', color=COLORS[x['arm']], ms=3)
        ax.set_yscale('log'); ax.set_xlabel('gradient steps'); ax.set_title(f'{rung:,} trajectories'); ax.grid(alpha=.3, which='both')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('validation mean of per-case max error, %')
    fig.suptitle('Learning curves (dots: best-validation checkpoint; faint lines: seeds 1, 2)')
    fig.tight_layout(); fig.savefig(HERE / 'figures' / 'learning_curves.png', dpi=150); plt.close(fig)

    if test:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, q, rq, title in ((axes[0], 'worst_percent', 'worst_evolved_percent', 'worst case'),
                                 (axes[1], 'median_percent', 'median_evolved_percent', 'median case')):
            for arm, fam in (('F0', 'FNO'), ('U0', 'U-Net')):
                pts = sorted((x['rung'], x['L256'][q]) for x in final if x['arm'] == arm and 'L256' in x)
                if pts:
                    ax.plot(*zip(*pts), 'o', color=COLORS[arm], ms=5)
                    med = [(r, float(np.median([e for rr, e in pts if rr == r]))) for r in sorted({r for r, _ in pts})]
                    ax.plot(*zip(*med), '-', color=COLORS[arm], label=f'{fam} {arm} (paper architecture)')
            for x in rows:
                if x['rung'] == 4608 and x['arm'] not in ('F0', 'U0') and 'L256' in x:
                    ax.plot(x['rung'] * 1.12, x['L256'][q], 's', color=COLORS[x['arm']], ms=5, label=f"{x['family']} {x['arm']}")
            for role, ls in (('accurate', '-'), ('fast', '--')):
                ax.axhline(R['256'][role][rq], color='k', ls=ls, lw=1, label=f'ROM {role}')
            if q == 'worst_percent':
                for fam, c in (('FNO', COLORS['F0']), ('U-Net', COLORS['U0'])):
                    ax.plot(128, PAPER['256'][fam]['worst_percent'], 'x', color=c, ms=8, label=f'{fam}, paper (128, wall cap)')
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([128, 576, 4608]); ax.set_xticklabels(['128', '576', '4,608'])
            ax.set_xlabel('training trajectories'); ax.set_ylabel(f'test relative error, {title}, %'); ax.grid(alpha=.3, which='both')
            ax.legend(fontsize=7)
        fig.suptitle('Data ladder at $256^2$, test64')
        fig.tight_layout(); fig.savefig(HERE / 'figures' / 'data_ladder.png', dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for x in rows:
            if 'L256' in x:
                ax.plot(x['L256']['timing']['median_ms'], x['L256']['worst_percent'], 'o' if x['rung'] == 4608 else '.',
                        color=COLORS[x['arm']], ms=6, alpha=1 if x['rung'] == 4608 else .5)
        for arm in sorted({x['arm'] for x in rows if 'L256' in x}):
            ax.plot([], [], 'o', color=COLORS[arm], label=arm)
        for role, mk in (('accurate', '*'), ('extra', 'P'), ('fast', 'D')):
            if role in R['256']:
                ax.plot(R['256'][role]['median_gpu_ms'], R['256'][role]['worst_evolved_percent'], mk, color='k', ms=9, label=f'ROM {role}')
        t1 = rom['meshes']['256']['table1']
        ax.plot(t1['fom_gpu_ms'], t1['fom_worst_evolved_percent'], 'X', color='gray', ms=9, label='FOM')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('inference time per query, ms (A100 80GB, batch 1)')
        ax.set_ylabel('worst-case test error, %'); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
        ax.set_title('Accuracy against time, $256^2$ (large dots: 4,608 rung; small: 128, 576)')
        fig.tight_layout(); fig.savefig(HERE / 'figures' / 'accuracy_time.png', dpi=150); plt.close(fig)
    print((HERE / 'tables.md').read_text())


if __name__ == '__main__':
    main()
