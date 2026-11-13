"""Generate reports/tables.md, reports/summary.json and figures from the pulled run and test JSON files.
Never type a number by hand: every number below is read from inputs/.

  python reports/make_report.py            (from experiments/operator-scaling/)
Works before the test evaluation exists (validation-only interim tables).
"""
import glob, json, math, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
INP = HERE / 'inputs'
TPL = HERE.parent / 'burgers-ops-matched' / 'inputs'
TARGET = None  # ROM accurate median, read below


def load(p):
    return json.loads(Path(p).read_text())


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—'
    if abs(x) >= 100:
        return f'{x:,.0f}'
    return f'{x:.{d}g}'


def gain(hist, n):
    if len(hist) <= n:
        return None
    b_now = min(h['val_mean'] for h in hist)
    b_then = min(h['val_mean'] for h in hist[:-n])
    return 100 * (b_then - b_now) / b_then


def ols(x, y):
    """log10-log10 OLS: slope, intercept, R^2, residual SE, 95 % CI of slope (t, n-2 dof)."""
    from scipy import stats
    x, y = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    n = len(x)
    if n < 3:
        return None
    A = np.vstack([x, np.ones(n)]).T
    (b, a), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = a + b * x
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    se = math.sqrt(ss_res / (n - 2))
    sb = se / math.sqrt(float(((x - x.mean()) ** 2).sum()))
    t = stats.t.ppf(.975, n - 2)
    return dict(n=n, slope=float(b), intercept=float(a), r2=1 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
                resid_se_log10=se, slope_ci95=[float(b - t * sb), float(b + t * sb)], x_mean_log10=float(x.mean()))


def extrapolate(fit, xs, ys, target):
    """Protocol section 7: only if n>=4, R^2>=0.9, slope CI excludes 0 and medians decrease monotonically."""
    order = np.argsort(xs)
    mono = all(np.diff(np.asarray(ys)[order]) < 0)
    ok = fit is not None and fit['n'] >= 4 and fit['r2'] >= .9 and fit['slope_ci95'][1] < 0 and mono
    out = dict(criteria=dict(n_ge_4=fit is not None and fit['n'] >= 4, r2_ge_0p9=fit is not None and fit['r2'] >= .9,
                             slope_ci_below_0=fit is not None and fit['slope_ci95'][1] < 0, monotone=bool(mono)), allowed=bool(ok))
    if ok:
        lt = math.log10(target)
        x_at = lambda b: fit['x_mean_log10'] + (lt - (fit['intercept'] + b * fit['x_mean_log10'])) / b
        out['x_needed'] = 10 ** ((lt - fit['intercept']) / fit['slope'])
        lo, hi = sorted([10 ** x_at(fit['slope_ci95'][0]), 10 ** x_at(fit['slope_ci95'][1])])
        out['x_needed_ci95_from_slope'] = [lo, hi]
    return out


def main():
    global TARGET
    rom = load(TPL / 'rom_reference.json')['meshes']['256']
    acc, fast = rom['roles']['accurate'], rom['roles']['fast']
    TARGET = acc['median_evolved_percent']
    # ---- runs (validation) ----
    runs = {}
    for f in sorted(glob.glob(str(INP / 'runs' / '*' / '*' / 'history.json'))):
        d = Path(f).parent
        meta = load(d / 'meta.json')
        hist = load(f)
        res = load(d / 'result.json') if (d / 'result.json').exists() else None
        key = f'{d.parent.name}/{d.name}'
        best = min(hist, key=lambda h: h['val_mean']) if hist else None
        runs[key] = dict(key=key, schedule=meta.get('schedule', 'plateau'), arm=meta['arm'], family='FNO' if meta['arm'][0] == 'F' else 'U-Net', rung=meta['rung'],
                         train_cases=meta['train_cases'], params=meta['real_parameters'], gpu=meta['gpu'], job=meta['job_id'],
                         steps=res['steps'] if res else hist[-1]['step'] if hist else 0,
                         stop=res['stop_reason'] if res else 'running', converged=bool(res and res['converged']),
                         best_step=res['best_step'] if res else (best['step'] if best else None),
                         val_mean=100 * (res['best_validation']['mean'] if res else best['val_mean']) if hist else None,
                         val_worst=100 * (res['best_validation']['worst'] if res else best['val_worst']) if hist else None,
                         final_lr=hist[-1]['lr'] if hist else None, gain10=gain(hist, 10), gain20=gain(hist, 20),
                         gpu_hours=(res['training_seconds'] if res else (hist[-1]['wall_seconds'] if hist else 0)) / 3600,
                         ms_per_step=res['ms_per_step'] if res else None)
    # ---- test ----
    tests = {}
    for f in sorted(glob.glob(str(INP / 'test' / '*.L256.json'))):
        t = load(f)
        tests[t['key']] = dict(worst=t['L256']['worst_percent'], median=t['L256']['median_percent'],
                               ms=t['L256']['timing']['median_ms'], p05=t['L256']['timing']['p05_ms'],
                               p95=t['L256']['timing']['p95_ms'], best_step=t['best_step'], checkpoint=t.get('checkpoint', 'best.pt'),
                               job=t['job_id'], gpu=t['gpu'])
    test_job = sorted({v['job'] for v in tests.values()})
    tpl_tests = {}
    for f in sorted(glob.glob(str(TPL / 'test' / '*.L256.json'))):
        t = load(f)
        tpl_tests[t['key']] = dict(arm=t['arm'], rung=t['rung'], seed=t['seed'], params=t['real_parameters'],
                                   worst=t['L256']['worst_percent'], median=t['L256']['median_percent'], ms=t['L256']['timing']['median_ms'])
    # ---- data generation cost ----
    gen = [load(f) for f in sorted(glob.glob(str(INP / 'datagen' / '*.json')))]
    gen_cost = dict(chunks=len(gen), cases=sum(g['cases'] for g in gen), solver_gpu_seconds=sum(g['total_seconds'] for g in gen),
                    median_seconds_per_case=float(np.median([g['median_seconds'] for g in gen])) if gen else None,
                    jobs=sorted({g['job_id'] for g in gen}), max_relative_residual=max([g['max_relative_residual'] for g in gen], default=None))
    tpl_gen = [load(f) for f in sorted(glob.glob(str(INP / 'datagen_template' / '*.json')))]
    gen_cost['template_pool'] = dict(cases=sum(g['cases'] for g in tpl_gen), solver_gpu_seconds=sum(g['total_seconds'] for g in tpl_gen))

    lines = []
    W = lines.append
    have_test = bool(tests)
    W(f'### Table 1. Every run of this lane: size, data, cost, validation{" and test (test64, " + ", ".join(sorted({v["gpu"] for v in tests.values()})) + ", batch 1)" if have_test else " (test not yet evaluated)"}\n')
    W('| family | arm | params | training trajectories (+256 validation) | checkpoint | steps (best) | stop | converged | training GPU-h (GPU) | val. mean % | val. worst % | test median % | test worst % | query ms (p05–p95) |')
    W('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    rows = []
    for k, r in sorted(runs.items(), key=lambda kv: (kv[1]['family'], kv[1]['rung'], kv[1]['params'])):
        for ck in ('', '@200k', '@400k'):
            t = tests.get(k + ck)
            if ck and not t:
                continue
            rows.append(dict(r, checkpoint=('best up to ' + ck[1:]) if ck else 'final best', test=t, test_key=k + ck))
            gpu = r['gpu'].replace('NVIDIA ', '').replace(' Blackwell Server Edition', '')
            W(f"| {r['family']} | {r['arm']}{' (cosine, A1)' if r['schedule'] == 'cosine' else ''} | {r['params']:,} | {r['train_cases']:,} | {('best ≤' + ck[1:]) if ck else 'final best'} | "
              f"{r['steps']:,} ({(t['best_step'] if t else r['best_step']) or 0:,}) | {r['stop']} | {'yes' if r['converged'] else '**no**'} | "
              f"{r['gpu_hours']:.1f} ({gpu}) | {fmt(r['val_mean'], 4)} | {fmt(r['val_worst'])} | "
              f"{fmt(t['median'], 3) if t else ''} | {fmt(t['worst'], 3) if t else ''} | "
              f"{(fmt(t['ms'], 3) + ' (' + fmt(t['p05'], 3) + '–' + fmt(t['p95'], 3) + ')') if t else ''} |")
    W(f"\nReference (same test64, same metric, A100 80GB): ROM accurate median {fmt(acc['median_evolved_percent'], 3)} %, worst "
      f"{fmt(acc['worst_evolved_percent'], 3)} %, {fmt(acc['median_gpu_ms'], 3)} ms; ROM fast median {fmt(fast['median_evolved_percent'], 3)} %, "
      f"worst {fmt(fast['worst_evolved_percent'], 3)} %, {fmt(fast['median_gpu_ms'], 3)} ms; full-order solver at the ROM-accurate tolerance median "
      f"{fmt(rom['table1']['fom_median_evolved_percent'], 3)} %, worst {fmt(rom['table1']['fom_worst_evolved_percent'], 3)} %, "
      f"{fmt(rom['table1']['fom_gpu_ms'], 3)} ms. Gain columns of Table 2 show whether a run was still improving.\n")

    W('### Table 2. Training state at the end of each run\n')
    W('| run | final lr | gain of best val. over last 10 / 20 validations % | ms/step | job |')
    W('|---|---|---|---|---|')
    for k, r in sorted(runs.items()):
        W(f"| {k} | {fmt(r['final_lr'], 2)} | {fmt(r['gain10'], 3)} / {fmt(r['gain20'], 3)} | {fmt(r['ms_per_step'], 3)} | {r['job']} |")

    W('\n### Table 3. Template-lane rows (4,608 rung, earlier budget: 8.1 h on shared GPUs), same test64 and timing\n')
    W('| arm | seed | params | test median % | test worst % | ms |')
    W('|---|---|---|---|---|---|')
    for k, t in sorted(tpl_tests.items(), key=lambda kv: (kv[1]['arm'], kv[1]['seed'])):
        if t['rung'] == 4608:
            W(f"| {t['arm']} | {t['seed']} | {t['params']:,} | {fmt(t['median'])} | {fmt(t['worst'])} | {fmt(t['ms'])} |")

    # ---- fits ----
    fits = {}
    metric = 'test' if have_test else 'validation'
    def val_of(r, which):
        if have_test:
            t = r['test']
            return None if not t else (t['median'] if which == 'median' else t['worst'])
        return r['val_mean'] if which == 'median' else r['val_worst']
    for fam in ('FNO', 'U-Net'):
        size = [r for r in rows if r['family'] == fam and r['rung'] == 4608 and r['schedule'] == 'plateau' and r['checkpoint'] == 'final best' and val_of(r, 'median')]
        for which in ('median', 'worst'):
            if len(size) >= 3:
                xs, ys = [r['params'] for r in size], [val_of(r, which) for r in size]
                f = ols(xs, ys)
                fits[f'{fam} size, {which}'] = dict(points=list(zip(xs, ys)), fit=f,
                                                    extrapolation=extrapolate(f, xs, ys, TARGET) if which == 'median' else None)
        arm = 'Fw128m64L4' if fam == 'FNO' else 'Ub128'
        # equal 200k budget: at 36,864 use the best-up-to-200k snapshot, or the final checkpoint if the run stopped by 200k
        data = [r for r in rows if r['arm'] == arm and r['schedule'] == 'plateau' and val_of(r, 'median') and
                ((r['rung'] < 36864 and r['checkpoint'] == 'final best') or
                 (r['rung'] == 36864 and (r['checkpoint'] == 'best up to 200k' or (r['checkpoint'] == 'final best' and r['steps'] <= 200000))))]
        if not have_test:
            data = [r for r in rows if r['arm'] == arm and r['schedule'] == 'plateau' and r['checkpoint'] == 'final best' and r['val_mean']]
        for which in ('median', 'worst'):
            if len(data) >= 3:
                xs, ys = [r['train_cases'] for r in data], [val_of(r, which) for r in data]
                f = ols(xs, ys)
                fits[f'{fam} data ({arm}), {which}'] = dict(points=list(zip(xs, ys)), fit=f,
                                                           extrapolation=extrapolate(f, xs, ys, TARGET) if which == 'median' else None)
        if fam == 'FNO':   # amendment A2: separate cosine data fit (3 points, no extrapolation by rule)
            cdata = [r for r in rows if r['arm'] == arm and r['schedule'] == 'cosine' and r['checkpoint'] == 'final best' and val_of(r, 'median')]
            for which in ('median', 'worst'):
                if len(cdata) >= 3:
                    xs, ys = [r['train_cases'] for r in cdata], [val_of(r, which) for r in cdata]
                    f = ols(xs, ys)
                    fits[f'FNO data, cosine ({arm}), {which}'] = dict(points=list(zip(xs, ys)), fit=f,
                                                                    extrapolation=extrapolate(f, xs, ys, TARGET) if which == 'median' else None)
    # query time vs params (measured)
    tfit = {}
    if have_test:
        for fam in ('FNO', 'U-Net'):
            pts = {(r['params'], r['test']['ms']) for r in rows if r['family'] == fam and r['test']}
            if len(pts) >= 3:
                xs, ys = zip(*sorted(pts))
                tfit[fam] = dict(points=list(zip(xs, ys)), fit=ols(xs, ys))
        for k, v in fits.items():
            ex = v.get('extrapolation')
            if ex and ex.get('allowed') and 'size' in k:
                fam = k.split()[0]; tf = tfit.get(fam, {}).get('fit')
                if tf:
                    ex['implied_query_ms'] = 10 ** (tf['intercept'] + tf['slope'] * math.log10(ex['x_needed']))
                    ex['label'] = 'extrapolation, not a measurement'
    W(f'\n### Table 4. Log–log fits of {metric} error ({"test median/worst %" if have_test else "validation mean/worst % — interim, not the protocol fit"})\n')
    W('| fit | points | slope [95 % CI] | R² | resid. SE (log10) | extrapolation allowed (protocol §7) | size/data needed for median 0.0264 % [95 % CI] |')
    W('|---|---|---|---|---|---|---|')
    for k, v in fits.items():
        f = v['fit']; ex = v.get('extrapolation') or {}
        need = ''
        if ex.get('allowed'):
            need = f"{fmt(ex['x_needed'])} [{fmt(ex['x_needed_ci95_from_slope'][0])}, {fmt(ex['x_needed_ci95_from_slope'][1])}] — *extrapolation, not a measurement*"
            if 'implied_query_ms' in ex:
                need += f"; implied query {fmt(ex['implied_query_ms'])} ms"
        crit = ', '.join(f"{c}={'yes' if b else 'no'}" for c, b in ex.get('criteria', {}).items())
        W(f"| {k} | {f['n']} | {f['slope']:.3f} [{f['slope_ci95'][0]:.3f}, {f['slope_ci95'][1]:.3f}] | {f['r2']:.3f} | {f['resid_se_log10']:.3f} | "
          f"{('yes' if ex.get('allowed') else 'no') + (' (' + crit + ')' if crit else '') if ex else 'n/a (worst case)'} | {need} |")
    if tfit:
        W('\nQuery time against parameters (measured, A100 80GB, batch 1): ' + '; '.join(
            f"{fam}: slope {v['fit']['slope']:.3f}, R² {v['fit']['r2']:.3f}" for fam, v in tfit.items()))
    W(f"\n### Table 5. Cost\n")
    total_train = sum(r['gpu_hours'] for r in runs.values())
    W(f"- Training: {total_train:.1f} GPU-hours over {len(runs)} runs (wall time on the run's own GPU).")
    W(f"- Extra data: {gen_cost['cases']:,} trajectories in {gen_cost['chunks']} chunks, solver time {gen_cost['solver_gpu_seconds'] / 3600:.2f} GPU-hours "
      f"(median {fmt(gen_cost['median_seconds_per_case'])} s per trajectory; jobs {', '.join(gen_cost['jobs'])}; max relative Newton residual "
      f"{fmt(gen_cost['max_relative_residual'])}). Template pool (4,608 trajectories): {gen_cost['template_pool']['solver_gpu_seconds'] / 3600:.2f} GPU-hours.")
    # best per family
    best = {}
    for fam in ('FNO', 'U-Net'):
        cand = [r for r in rows if r['family'] == fam and r['val_mean'] is not None]
        if cand:
            b = min(cand, key=lambda r: r['val_mean'] if r['checkpoint'] == 'final best' else float('inf'))
            best[fam] = dict(key=b['test_key'], arm=b['arm'], rung=b['rung'], params=b['params'], val_mean=b['val_mean'], test=b['test'])
        reached = [r for r in rows if r['family'] == fam and r['test'] and r['test']['median'] <= TARGET]
        best.setdefault(fam, {})['reached_target'] = [dict(key=r['test_key'], median=r['test']['median'], ms=r['test']['ms'], worst=r['test']['worst']) for r in reached]
        tested = [r for r in rows if r['family'] == fam and r['test']]
        if tested:
            c = min(tested, key=lambda r: r['test']['median'])
            best[fam]['closest_test_median'] = dict(key=c['test_key'], median=c['test']['median'], worst=c['test']['worst'], ms=c['test']['ms'],
                                                    params=c['params'], train_cases=c['train_cases'])
    W('\n### Table 6. Headline per family (validation-selected over all runs) and closest test median\n')
    W('| family | validation-selected run | params | training traj. | test median % | test worst % | ms | runs at or below 0.0264 % median | lowest test median of any run (run, ms) |')
    W('|---|---|---|---|---|---|---|---|---|')
    for fam, b in best.items():
        t = b.get('test')
        c = b.get('closest_test_median')
        W(f"| {fam} | {b.get('key', '—')} | {b.get('params', 0):,} | {runs.get(b.get('key', '').split('@')[0], {}).get('train_cases', 0):,} | "
          f"{fmt(t['median']) if t else ''} | {fmt(t['worst']) if t else ''} | {fmt(t['ms']) if t else ''} | {len(b['reached_target'])} | "
          f"{(fmt(c['median']) + ' (' + c['key'] + ', ' + fmt(c['ms']) + ' ms)') if c else ''} |")
    (HERE / 'reports' / 'tables.md').write_text('\n'.join(lines) + '\n')
    summary = dict(target_median_percent=TARGET, rom_accurate=acc, rom_fast=fast, runs=runs, tests=tests, test_jobs=test_job,
                   fits=fits, time_fits=tfit, best=best, data_generation=gen_cost, training_gpu_hours=total_train,
                   metric_for_fits=metric)
    (HERE / 'reports' / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    # ---- figures ----
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        figdir = HERE / 'reports' / 'figures'; figdir.mkdir(exist_ok=True)
        col = {'FNO': '#2a6fdb', 'U-Net': '#d9822b'}
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
        for ax, axis in zip(axs[:2], ('size', 'data')):
            for k, v in fits.items():
                if axis not in k or 'median' not in k:
                    continue
                fam = k.split()[0]
                xs, ys = zip(*v['points'])
                ax.loglog(xs, ys, 'o', color=col[fam], label=f"{k.split(',')[0]} (slope {v['fit']['slope']:.2f}, R² {v['fit']['r2']:.2f})")
                xx = np.logspace(math.log10(min(xs)), math.log10(max(xs)), 20)
                ax.loglog(xx, 10 ** (v['fit']['intercept'] + v['fit']['slope'] * np.log10(xx)), '-', color=col[fam], lw=1)
            ax.axhline(TARGET, color='k', ls='--', lw=1, label='ROM accurate median 0.0264 %')
            ax.set_xlabel('parameters' if axis == 'size' else 'training trajectories')
            ax.set_ylabel(f'{metric} median error %' if have_test else 'validation mean error %')
            ax.set_title(f'{"size (4,608 trajectories)" if axis == "size" else "data (fixed architecture)"}')
            ax.legend(fontsize=7); ax.grid(True, which='both', alpha=.3)
        ax = axs[2]
        for r in rows:
            if r['test']:
                ax.loglog(r['test']['ms'], r['test']['median'], 'o', color=col[r['family']], alpha=.8)
        ax.loglog(acc['median_gpu_ms'], TARGET, 'k*', ms=12, label='ROM accurate')
        ax.loglog(fast['median_gpu_ms'], fast['median_evolved_percent'], 'kx', ms=9, label='ROM fast')
        ax.set_xlabel('query time ms (A100, batch 1)'); ax.set_ylabel('test median error %'); ax.set_title('accuracy vs query time')
        ax.legend(fontsize=7); ax.grid(True, which='both', alpha=.3)
        fig.tight_layout(); fig.savefig(figdir / 'scaling.png', dpi=130)
    except Exception as exc:  # figures are optional; tables are the record
        print('figure skipped:', exc)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
