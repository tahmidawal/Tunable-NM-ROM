"""The one report of the b-seeds lane, generated from the audited JSONs only.

    python experiments/b-seeds/reports/generate_b_seeds.py \
        --audits experiments/b-seeds/checks/*-audit.json \
        --out experiments/b-seeds/reports/2026-09-18-b-seeds

Writes `<out>.md` and `summary.json` beside it. Every number in the prose and the tables is
read from an audit JSON (`audit_seeds.py` output) or from an EQ-certification audit; nothing
is typed by hand. Audit file names follow `<attempt>-<block>-audit.json`, where block is
`ladder_seed`, `ladder_incumbent`, `sealed_incumbent`, `sealed_seed1`, ... or `eqcert_seed`.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / 'experiments/b-seeds/DESIGN.md'
Q = [0, 16, 32, 64, 128, 256]
SEED_LABELS = ['seed1', 'seed2', 'seed3']


def f(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    return f'{x:.{d}f}'


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def ms(vals, d=4):
    m, s, n = mean_std(vals)
    if m is None:
        return '—'
    return f'{m:.{d}f} ± {s:.{d}f}' + ('' if n == 3 else f' (n={n})')


def load(paths):
    audits = {}
    for p in paths:
        p = Path(p)
        a = json.loads(p.read_text())
        a['_file'] = str(p)
        a['_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        name = p.name[:-len('-audit.json')]
        attempt, block = name.split('-', 1)
        a['_attempt'], a['_block'] = attempt, block
        audits[name] = a
    return audits


def rung_rows(a, ladder='dense_m4'):
    lad = a['ladders'].get(ladder)
    if not lad:
        return {}
    return {u['q']: u for u in lad['rungs']}


def arm_by_name(a):
    return {x['arm']: x for x in a['arms']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audits', nargs='+', required=True)
    p.add_argument('--eqcert', nargs='*', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--date', default='2026-09-18')
    a = p.parse_args()
    audits = load(a.audits)
    out_md = Path(a.out + '.md')
    out_md.parent.mkdir(parents=True, exist_ok=True)

    dev_seed = {x['checkpoint_label']: x for x in audits.values()
                if x['_block'] == 'ladder_seed' and x['cohort'] == 'dev'}
    dev_inc = {x['_attempt']: x for x in audits.values()
               if x['_block'] == 'ladder_incumbent' and x['cohort'] == 'dev'}
    sealed = {x['checkpoint_label']: x for x in audits.values() if x['cohort'] == 'sealed'}
    seeds = [s for s in SEED_LABELS if s in dev_seed]
    summary = []

    def row(**kw):
        summary.append(kw)

    # ------------------------------------------------------------ per-rung rows ----
    per = {}   # (label, cohort, q) -> rung dict
    for label, x in list(dev_seed.items()) + [(f'incumbent@{k}', v) for k, v in dev_inc.items()] \
            + [(f'{k}@sealed', v) for k, v in sealed.items()]:
        cohort = x['cohort']
        for lad in ('dense_m4', 'dense_fixedM'):
            for q, u in rung_rows(x, lad).items():
                per[(label, cohort, lad, q)] = u
                for metric in ('evolved', 'all_times', 't0', 'reference', 'best_found', 'gpu_ms',
                               'converged', 'budget_exits', 'max_joint_stationarity'):
                    row(checkpoint=x['checkpoint_label'], cohort=cohort, attempt=x['_attempt'],
                        job_id=x['job_id'], gpu=x['gpu'], ladder=lad, q=q, M=u['M'],
                        quadrature='dense', arm=u['arm'], metric=metric, value=u[metric],
                        checkpoint_sha256=x['checkpoint_sha256'], source=x['_file'],
                        source_sha256=x['_sha256'])
        row(checkpoint=x['checkpoint_label'], cohort=cohort, attempt=x['_attempt'],
            job_id=x['job_id'], gpu=x['gpu'], ladder='dense_m4', q=None, quadrature='dense',
            metric='knob_bar_passes', value=x['verdict']['passes'],
            checkpoint_sha256=x['checkpoint_sha256'], source=x['_file'], source_sha256=x['_sha256'])
        for lad, v in x['ladders'].items():
            for metric in ('monotone_evolved', 'monotone_all_times', 'all_converged',
                           'error_span_evolved', 'cost_span'):
                row(checkpoint=x['checkpoint_label'], cohort=cohort, attempt=x['_attempt'],
                    job_id=x['job_id'], gpu=x['gpu'], ladder=lad, q=None, quadrature='dense',
                    metric=metric, value=v[metric], checkpoint_sha256=x['checkpoint_sha256'],
                    source=x['_file'], source_sha256=x['_sha256'])
        for name, arm in arm_by_name(x).items():
            if arm['family'] == 'fom':
                for metric in ('worst_all_times_percent', 'worst_evolved_percent', 'median_gpu_ms'):
                    row(checkpoint=x['checkpoint_label'], cohort=cohort, attempt=x['_attempt'],
                        job_id=x['job_id'], gpu=x['gpu'], ladder=None, q=None, quadrature=None,
                        arm=name, metric=metric, value=arm[metric], source=x['_file'],
                        source_sha256=x['_sha256'])
        if x.get('three_layer'):
            for k, v in x['three_layer'].items():
                if k != 'incumbent_reference':
                    row(checkpoint=x['checkpoint_label'], cohort=cohort, attempt=x['_attempt'],
                        job_id=x['job_id'], gpu=x['gpu'], ladder='dense_m4', q=0, quadrature='dense',
                        metric=f'three_layer_{k}', value=v, checkpoint_sha256=x['checkpoint_sha256'],
                        source=x['_file'], source_sha256=x['_sha256'])
        if x.get('training'):
            t = x['training']
            for grp in ('bank', 'span_floor', 'head', 'pick'):
                for k, v in t[grp].items():
                    if k != 'incumbent':
                        row(checkpoint=x['checkpoint_label'], cohort='training', attempt=x['_attempt'],
                            job_id=(t['job_ids'] or {}).get('hfit', x['job_id']), gpu=x['gpu'],
                            ladder=None, q=None, quadrature=None, metric=f'{grp}_{k}', value=v,
                            checkpoint_sha256=x['checkpoint_sha256'], source=x['_file'],
                            source_sha256=x['_sha256'])
        for gname, g in x['checks'].items():
            row(checkpoint=x['checkpoint_label'], cohort=cohort, attempt=x['_attempt'],
                job_id=x['job_id'], gpu=x['gpu'], ladder=None, q=None, quadrature=None,
                metric=f'gate_{gname}', value=g['passed'], source=x['_file'], source_sha256=x['_sha256'])

    # ---------------------------------------------------------- the criteria ----
    def seed_vals(cohort_map, q, key, lad='dense_m4'):
        return [rung_rows(cohort_map[s], lad).get(q, {}).get(key) for s in seeds if s in cohort_map]

    mono_dev = {lad: {s: dev_seed[s]['ladders'][lad]['monotone_evolved'] for s in seeds}
                for lad in ('dense_m4', 'dense_fixedM') if all(lad in dev_seed[s]['ladders'] for s in seeds)}
    mono_dev_all = {lad: {s: dev_seed[s]['ladders'][lad]['monotone_all_times'] for s in seeds}
                    for lad in mono_dev}
    conv_dev = {s: dev_seed[s]['ladders']['dense_m4']['all_converged'] for s in seeds}
    c1_count = sum(1 for s in seeds if mono_dev.get('dense_m4', {}).get(s) and conv_dev[s])
    C1 = c1_count >= 2
    knob = {s: dev_seed[s]['verdict']['passes'] for s in seeds}
    C4 = sum(knob.values()) >= 2
    # C2 / C3 need the sealed job
    ratios = {}
    C2 = None
    C2n = None
    C3 = None
    sealed_complete = bool(sealed) and 'incumbent' in sealed and all(s in sealed for s in seeds)
    if sealed:
        inc_dev = next(iter(dev_inc.values())) if dev_inc else None
        both = [s for s in seeds if s in sealed]      # the seed intersection, both cohorts
        for q in Q:
            dm = mean_std([rung_rows(dev_seed[s]).get(q, {}).get('evolved') for s in both])[0]
            sm = mean_std([rung_rows(sealed[s]).get(q, {}).get('evolved') for s in both])[0]
            # difficulty-normalised twin (A1.3): solved evolved error over the rung's own best-found
            dn = mean_std([rung_rows(dev_seed[s]).get(q, {}).get('evolved') / rung_rows(dev_seed[s]).get(q, {}).get('best_found')
                           for s in both if rung_rows(dev_seed[s]).get(q, {}).get('best_found')])[0]
            sn = mean_std([rung_rows(sealed[s]).get(q, {}).get('evolved') / rung_rows(sealed[s]).get(q, {}).get('best_found')
                           for s in both if rung_rows(sealed[s]).get(q, {}).get('best_found')])[0]
            r_inc = rn_inc = None
            if inc_dev and 'incumbent' in sealed:
                di = rung_rows(inc_dev).get(q, {})
                si = rung_rows(sealed['incumbent']).get(q, {})
                if di.get('evolved') and si.get('evolved') is not None:
                    r_inc = si['evolved'] / di['evolved']
                    if di.get('best_found') and si.get('best_found'):
                        rn_inc = (si['evolved'] / si['best_found']) / (di['evolved'] / di['best_found'])
            ratios[q] = dict(dev_seed_mean=dm, sealed_seed_mean=sm,
                             ratio_seed_mean=(sm / dm if (dm and sm is not None) else None),
                             ratio_incumbent=r_inc,
                             normalised_dev_seed_mean=dn, normalised_sealed_seed_mean=sn,
                             normalised_ratio_seed_mean=(sn / dn if (dn and sn is not None) else None),
                             normalised_ratio_incumbent=rn_inc, seeds_in_both=both)
        if sealed_complete:
            C2 = all(v['ratio_seed_mean'] is not None and v['ratio_seed_mean'] <= 1.5
                     and v['ratio_incumbent'] is not None and v['ratio_incumbent'] <= 1.5
                     for v in ratios.values())
            C2n = all(v['normalised_ratio_seed_mean'] is not None and v['normalised_ratio_seed_mean'] <= 1.5
                      and v['normalised_ratio_incumbent'] is not None and v['normalised_ratio_incumbent'] <= 1.5
                      for v in ratios.values())
        conv_all = [x['ladders']['dense_m4']['all_converged'] for x in list(dev_seed.values())
                    + list(dev_inc.values()) + list(sealed.values()) if 'dense_m4' in x['ladders']]
        C3 = all(conv_all) if sealed_complete else None
    # TR: thresholds from the incumbent's audited q = 0 row (comparators/qtd02-audit.json), never typed
    ref = next(iter(dev_seed.values()))['three_layer']['incumbent_reference']
    tr_bf, tr_fl = 1.5 * ref['best_found'], 2 * ref['bank_floor']
    TR = {}
    for s in seeds:
        tl = dev_seed[s]['three_layer']
        TR[s] = dict(best_found=tl['best_found_percent'], bank_floor=tl['bank_floor_percent'],
                     best_found_bar=tr_bf, bank_floor_bar=tr_fl,
                     passes=bool(tl['best_found_percent'] <= tr_bf and tl['bank_floor_percent'] <= tr_fl))
    F3 = sum(1 for v in TR.values() if not v['passes']) >= 2
    verdict = dict(C1_monotone_converged_on_at_least_2_of_3=C1, C1_count=c1_count,
                   monotone_evolved_counts={lad: sum(v.values()) for lad, v in mono_dev.items()},
                   monotone_all_times_counts={lad: sum(v.values()) for lad, v in mono_dev_all.items()},
                   converged_dev_count=sum(conv_dev.values()),
                   C2_sealed_over_dev_ratio_le_1p5=C2, C2n_normalised_ratio_le_1p5=C2n,
                   sealed_blocks_complete=sealed_complete, C2_ratios=ratios,
                   C3_every_rung_converged_everywhere=C3,
                   C4_knob_bar_on_at_least_2_of_3=C4, knob=knob,
                   TR=TR, F3_recipe_not_reproduced=F3, seeds_present=seeds,
                   sealed_present=sorted(sealed))
    for k, v in verdict.items():
        if not isinstance(v, dict):
            row(checkpoint='all', cohort='verdict', attempt=None, job_id=None, gpu=None, ladder='dense_m4',
                q=None, quadrature='dense', metric=k, value=v, source='generate_b_seeds.py')

    # ------------------------------------------------------------ EQ cert rows ----
    eq = {}
    for pth in a.eqcert:
        e = json.loads(Path(pth).read_text())
        eq[e.get('checkpoint_label') or Path(pth).name] = e
        for rr in e.get('rules', []):
            row(checkpoint=e.get('checkpoint_label'), cohort='dev', attempt=e.get('attempt'),
                job_id=e.get('job_id'), gpu=e.get('gpu'), ladder='eqcert', q=rr['q'], M=rr.get('M'),
                quadrature=f"eq:{rr['population']}:m{rr['m']}", metric='rho_max',
                value=rr['certification']['rho_max'], source=pth)
            row(checkpoint=e.get('checkpoint_label'), cohort='dev', attempt=e.get('attempt'),
                job_id=e.get('job_id'), gpu=e.get('gpu'), ladder='eqcert', q=rr['q'], M=rr.get('M'),
                quadrature=f"eq:{rr['population']}:m{rr['m']}", metric='certified_primary',
                value=rr['certified_primary'], source=pth)
        for arm in e.get('arms', []):
            for metric in ('worst_evolved_percent', 'worst_all_times_percent', 'median_gpu_ms', 'converged'):
                if metric in arm:
                    row(checkpoint=e.get('checkpoint_label'), cohort='dev', attempt=e.get('attempt'),
                        job_id=e.get('job_id'), gpu=e.get('gpu'), ladder='eqcert', q=arm.get('q'),
                        M=arm.get('M'), quadrature=arm.get('quadrature'), arm=arm['arm'],
                        metric=metric, value=arm[metric], source=pth)

    Path(a.out).parent.joinpath('summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')

    # ================================================================== report ====
    L = []
    w = L.append
    inc_any = next(iter(dev_inc.values()), None)
    w(f'# Three training seeds and a sealed cohort for the Burgers $256^2$ correction ladder\n')
    w(f'Generated {a.date} by `generate_b_seeds.py` from {len(audits)} audit JSONs'
      f'{" and " + str(len(eq)) + " EQ-certification results" if eq else ""}; every number below is '
      f'read from them. State of the numbers: ' +
      ('**final** — the sealed cohort has been opened and evaluated.' if sealed else
       '**provisional** — development cohort only; the sealed cohort is still unopened.') +
      ' Pre-registration: `experiments/b-seeds/DESIGN.md`.\n')
    w('## 1. Verdict against the pre-registered criteria\n')
    w('| criterion | holds | measured |\n|---|---|---|')
    w(f'| C1 — `dense_m4` monotone on the evolved metric with every rung converged on ≥ 2 of 3 seeds | '
      f'{f(C1)} | {c1_count} of {len(seeds)} seeds |')
    w(f'| C2 — sealed / development worst evolved error ≤ 1.5 at every rung (seed mean / incumbent) | '
      f'{f(C2) if C2 is not None else "not yet run"} | '
      + ('; '.join(f'q={q}: {f(v["ratio_seed_mean"], 3)} / {f(v["ratio_incumbent"], 3)}' for q, v in ratios.items()) if ratios else '—') + ' |')
    w(f'| C2n — the same ratio on error / best-found (difficulty-normalised, A1.3; seed mean / incumbent) | '
      f'{f(C2n) if C2n is not None else "not yet run"} | '
      + ('; '.join(f'q={q}: {f(v["normalised_ratio_seed_mean"], 3)} / {f(v["normalised_ratio_incumbent"], 3)}' for q, v in ratios.items()) if ratios else '—') + ' |')
    w(f'| C3 — every rung converged on every checkpoint on both cohorts | {f(C3) if C3 is not None else "not yet run"} | '
      f'development: {sum(conv_dev.values())} of {len(seeds)} seeds converged |')
    w(f'| C4 — knob bar (≥ 2× error and ≥ 2× cost on the converged non-dominated set) on ≥ 2 of 3 seeds | '
      f'{f(C4)} | {sum(knob.values())} of {len(seeds)} |')
    tr_pass = sum(1 for v in TR.values() if v['passes'])
    w(f'| TR — seeds reproducing the incumbent recipe (best-found ≤ {f(tr_bf, 2)} %, bank floor ≤ {f(tr_fl, 2)} %) | '
      f'{tr_pass} of {len(TR)} | ' + '; '.join(f'{s}: best-found {f(v["best_found"])} %, floor {f(v["bank_floor"])} % → {f(v["passes"])}' for s, v in TR.items()) + ' |')
    w(f'| F3 — the recipe is NOT reproduced (≥ 2 of 3 seeds fail TR), so the seed table is provisional | '
      f'{f(F3)} | {len(TR) - tr_pass} of {len(TR)} seeds fail TR |')
    w('')
    w(f'Monotone-on-evolved counts: ' + ', '.join(f'`{lad}` {n} of {len(seeds)}' for lad, n in verdict['monotone_evolved_counts'].items())
      + '. Monotone-on-all-times counts: ' + ', '.join(f'`{lad}` {n} of {len(seeds)}' for lad, n in verdict['monotone_all_times_counts'].items()) + '.\n')

    # ------------------------------------------------------------- T12 ----
    w('## 2. T12 — the seed table, development cohort, `dense_m4` ($M = 4(K+q)$, budget 600, dense quadrature)\n')
    w('Errors are worst over the six development cases against the same-job converged `fft_tight` solve, in per cent. '
      'GPU ms are medians of three timed repetitions **within one job**; the incumbent column is the incumbent '
      're-run in the same job as the seed, and the cost ratio is that same-job ratio.\n')
    w('| $q$ | $M$ | evolved, mean ± std (seeds) | evolved, incumbent | all-times, mean ± std | all-times, incumbent | $t{=}0$, mean ± std | best-found, mean ± std | seed / incumbent cost, same job, mean ± std | converged (seeds) |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for q in Q:
        ev = seed_vals(dev_seed, q, 'evolved')
        al = seed_vals(dev_seed, q, 'all_times')
        t0 = seed_vals(dev_seed, q, 't0')
        bf = seed_vals(dev_seed, q, 'best_found')
        gm = seed_vals(dev_seed, q, 'gpu_ms')
        conv = seed_vals(dev_seed, q, 'converged')
        inc_e = [rung_rows(dev_inc[dev_seed[s]['_attempt']]).get(q, {}).get('evolved') for s in seeds if dev_seed[s]['_attempt'] in dev_inc]
        inc_a = [rung_rows(dev_inc[dev_seed[s]['_attempt']]).get(q, {}).get('all_times') for s in seeds if dev_seed[s]['_attempt'] in dev_inc]
        cr = []
        for s in seeds:
            att = dev_seed[s]['_attempt']
            if att in dev_inc:
                gi = rung_rows(dev_inc[att]).get(q, {}).get('gpu_ms')
                gs = rung_rows(dev_seed[s]).get(q, {}).get('gpu_ms')
                if gi and gs:
                    cr.append(gs / gi)
        M = next((u['M'] for s in seeds for qq, u in rung_rows(dev_seed[s]).items() if qq == q), None)
        w(f'| {q} | {M} | {ms(ev)} | {f(inc_e[0]) if inc_e else "—"} | {ms(al)} | {f(inc_a[0]) if inc_a else "—"} | {ms(t0)} | {ms(bf)} | {ms(cr, 3)} | {sum(1 for c in conv if c)} of {len(conv)} |')
    w('')
    w('Per seed (GPU ms are per job on the GPU model shown; never averaged across jobs):\n')
    w('| checkpoint | job | GPU | ' + ' | '.join(f'q={q}' for q in Q) + ' | monotone evolved | monotone all-times | converged | error span | cost span | knob bar |')
    w('|---|---|---|' + '---|' * len(Q) + '---|---|---|---|---|---|')
    for label, x in [(s, dev_seed[s]) for s in seeds] + [(f'incumbent ({k})', v) for k, v in dev_inc.items()]:
        rr = rung_rows(x)
        lad = x['ladders']['dense_m4']
        w(f'| {label} | {x["job_id"]} | {x["gpu"]} | ' + ' | '.join(f'{f(rr[q]["evolved"])} / {f(rr[q]["gpu_ms"], 0)} ms' if q in rr else '—' for q in Q)
          + f' | {f(lad["monotone_evolved"])} | {f(lad["monotone_all_times"])} | {f(lad["all_converged"])} | {f(lad["error_span_evolved"], 3)}× | {f(lad["cost_span"], 3)}× | {f(x["verdict"]["passes"])} |')
    w('')
    if all('dense_fixedM' in dev_seed[s]['ladders'] for s in seeds):
        w('`dense_fixedM` ($M = 256$ fixed), worst evolved %, per seed:\n')
        w('| checkpoint | ' + ' | '.join(f'q={q}' for q in Q if q <= 128) + ' | monotone evolved | monotone all-times |')
        w('|---|' + '---|' * len([q for q in Q if q <= 128]) + '---|---|')
        for label, x in [(s, dev_seed[s]) for s in seeds] + [(f'incumbent ({k})', v) for k, v in dev_inc.items()]:
            rr = rung_rows(x, 'dense_fixedM')
            lad = x['ladders']['dense_fixedM']
            w(f'| {label} | ' + ' | '.join(f'{f(rr[q]["evolved"])}' if q in rr else '—' for q in Q if q <= 128)
              + f' | {f(lad["monotone_evolved"])} | {f(lad["monotone_all_times"])} |')
        w('')

    # ------------------------------------------------------- three layers ----
    w('## 3. The three layers at $q = 0$, development cohort\n')
    w('| checkpoint | bank floor % | best-found % | solved all-times % | solved evolved % | $t{=}0$ compression % | GPU ms | TR |')
    w('|---|---|---|---|---|---|---|---|')
    for label, x in [(s, dev_seed[s]) for s in seeds] + [(f'incumbent ({k})', v) for k, v in dev_inc.items()]:
        tl = x['three_layer']
        w(f'| {label} | {f(tl["bank_floor_percent"])} | {f(tl["best_found_percent"])} | {f(tl["solved_all_times_percent"])} | '
          f'{f(tl["solved_evolved_percent"])} | {f(tl["t0_compression_percent"])} | {f(tl["median_gpu_ms"], 1)} | '
          f'{f(TR[label]["passes"]) if label in TR else "—"} |')
    w('')

    # ------------------------------------------------------------ training ----
    tr_rows = [(s, dev_seed[s]['training']) for s in seeds if dev_seed[s].get('training')]
    if tr_rows:
        w('## 4. Training records per seed (the incumbent\'s own recipe, `SEED0` the only variable)\n')
        w('| seed | bank job GPU | bank steps | bank s | bank recon mean % | cond(G) | span floor train mean % | span floor test mean / max % | head steps | head s | head loss | head recon mean % | oracle test mean / max % | oracle / floor | pick early fraction |')
        w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for s, t in tr_rows:
            w(f'| {s} | {t["gpu"]["bank"]} | {t["bank"]["steps_done"]} | {f(t["bank"]["seconds"], 0)} | {f(t["bank"]["recon_train_mean_percent"], 3)} | '
              f'{f(t["bank"]["cond_G"], 0)} | {f(t["span_floor"]["train_mean_percent"], 4)} | {f(t["span_floor"]["test_mean_percent"], 4)} / {f(t["span_floor"]["test_max_percent"], 4)} | '
              f'{t["head"]["steps_done"]} | {f(t["head"]["seconds"], 0)} | {t["head"]["final_loss"]:.3e} | {f(t["head"]["recon_train_mean_percent"], 3)} | '
              f'{f(t["head"]["oracle_test_mean_percent"], 3)} / {f(t["head"]["oracle_test_max_percent"], 3)} | {f(t["head"]["oracle_over_span_floor"], 1)} | {f(t["pick"]["early_fraction"], 3)} |')
        t0_ = tr_rows[0][1]
        inc_r3 = json.loads((ROOT / 'experiments/separable-decoder/runs/push_r3a/out/sep_burgers_r3_N256_K16_R512.json').read_text())
        inc_hf = json.loads((ROOT / 'experiments/separable-decoder/runs/dn256b/out/hfit_dn256b_full.json').read_text())
        w(f'| incumbent (jobs {inc_r3["config"]["slurm_job"]} / {inc_hf["config"]["slurm_job"]}) | {inc_r3["config"]["gpu"]} | {inc_r3["train"]["steps_done"]} | {f(t0_["bank"]["incumbent"]["seconds"], 0)} | '
          f'{f(t0_["bank"]["incumbent"]["recon_train_mean_percent"], 3)} | {f(t0_["bank"]["incumbent"]["cond_G"], 0)} | '
          f'{f(t0_["span_floor"]["incumbent"]["train_mean_percent"], 4)} | {f(t0_["span_floor"]["incumbent"]["test_mean_percent"], 4)} / {f(t0_["span_floor"]["incumbent"]["test_max_percent"], 4)} | '
          f'{inc_hf["arms"]["mid"]["train"]["steps_done"]} | {f(inc_hf["arms"]["mid"]["train"]["seconds"], 0)} | {t0_["head"]["incumbent"]["final_loss"]:.3e} | {f(t0_["head"]["incumbent"]["recon_train_mean_percent"], 3)} | '
          f'{f(t0_["head"]["incumbent"]["oracle_test_mean_percent"], 3)} / {f(t0_["head"]["incumbent"]["oracle_test_max_percent"], 3)} | {f(t0_["head"]["incumbent"]["oracle_over_span_floor"], 1)} | {f(inc_r3["data"]["n_early_states_in_pick"] / inc_r3["data"]["n_states_trained"], 3)} (bank pick) |')
        w('')

    # ------------------------------------------------------------ T13 ----
    if sealed:
        w('## 5. T13 — the sealed cohort, `dense_m4`, worst evolved % (and all-times %)\n')
        w(f'The sealed cohort is `params_draw(17092026, 6)`, opened once, in the final job (DESIGN.md §3). '
          'Development numbers are from the seed jobs; sealed numbers from the final job; the ratio compares errors only.\n')
        w('| $q$ | dev, seed mean ± std | sealed, seed mean ± std | ratio (seed means) | normalised ratio (seed means) | dev, incumbent | sealed, incumbent | ratio (incumbent) | normalised ratio (incumbent) | sealed all-times, seed mean ± std | sealed all-times, incumbent | sealed converged |')
        w('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for q in Q:
            r = ratios[q]
            se = [rung_rows(sealed[s]).get(q, {}).get('evolved') for s in seeds if s in sealed]
            sa = [rung_rows(sealed[s]).get(q, {}).get('all_times') for s in seeds if s in sealed]
            sc = [rung_rows(sealed[s]).get(q, {}).get('converged') for s in seeds if s in sealed]
            if 'incumbent' in sealed:
                sc.append(rung_rows(sealed['incumbent']).get(q, {}).get('converged'))
            di = rung_rows(inc_any).get(q, {}).get('evolved') if inc_any else None
            si = rung_rows(sealed['incumbent']).get(q, {}) if 'incumbent' in sealed else {}
            w(f'| {q} | {ms(seed_vals(dev_seed, q, "evolved"))} | {ms(se)} | {f(r["ratio_seed_mean"], 3)} | {f(r["normalised_ratio_seed_mean"], 3)} | {f(di)} | {f(si.get("evolved"))} | '
              f'{f(r["ratio_incumbent"], 3)} | {f(r["normalised_ratio_incumbent"], 3)} | {ms(sa)} | {f(si.get("all_times"))} | {sum(1 for c in sc if c)} of {len(sc)} |')
        w('')
        w('Per-case evolved error (%), sealed cohort, `dense_m4`, so the cohort\'s difficulty spread is visible (A1.3):\n')
        for label in ['incumbent'] + seeds:
            if label not in sealed:
                continue
            x = sealed[label]
            w(f'`{label}` (job {x["job_id"]}):\n')
            w('| $q$ | ' + ' | '.join(f'case {c}' for c in range(6)) + ' | best-found worst % |')
            w('|---|' + '---|' * 6 + '---|')
            arms = arm_by_name(x)
            for q, u in rung_rows(x).items():
                pc = arms[u['arm']]['per_case_evolved_percent']
                w(f'| {q} | ' + ' | '.join(f(pc.get(str(c))) for c in range(6)) + f' | {f(u["best_found"])} |')
            w('')
        w('Sealed cohort per checkpoint (worst evolved % / GPU ms; same-job FOM controls in the final job):\n')
        w('| checkpoint | job | GPU | ' + ' | '.join(f'q={q}' for q in Q) + ' | monotone evolved | monotone all-times | converged | error span | cost span |')
        w('|---|---|---|' + '---|' * len(Q) + '---|---|---|---|---|')
        for label in ['incumbent'] + seeds:
            if label not in sealed:
                continue
            x = sealed[label]
            rr = rung_rows(x)
            lad = x['ladders']['dense_m4']
            w(f'| {label} | {x["job_id"]} | {x["gpu"]} | ' + ' | '.join(f'{f(rr[q]["evolved"])} / {f(rr[q]["gpu_ms"], 0)}' if q in rr else '—' for q in Q)
              + f' | {f(lad["monotone_evolved"])} | {f(lad["monotone_all_times"])} | {f(lad["all_converged"])} | {f(lad["error_span_evolved"], 3)}× | {f(lad["cost_span"], 3)}× |')
        w('')

    # ------------------------------------------------------ FOM controls ----
    w(f'## {6 if sealed else 5}. Same-job full-order controls\n')
    w('| job | block | control | worst all-times % | worst evolved % | median GPU ms |')
    w('|---|---|---|---|---|---|')
    for name, x in sorted(audits.items()):
        for arm in x['arms']:
            if arm['family'] == 'fom':
                w(f'| {x["job_id"]} | {x["_block"]} | `{arm["arm"]}` | {f(arm["worst_all_times_percent"])} | {f(arm["worst_evolved_percent"])} | {f(arm["median_gpu_ms"], 3)} |')
    w('')

    # ------------------------------------------------------- EQ cert ----
    if eq:
        w(f'## {7 if sealed else 6}. EQ rule certification per seed ($q \\le 64$, reachable population, $m = 1024$, bar $\\rho_{{\\max}} \\le 0.116$)\n')
        w('| seed | $q$ | population | $m$ | NNLS fit | $\\rho_{\\max}$ | $\\rho_{95}$ | certified | EQ evolved % (cert rule) | EQ evolved % (static rule) | dense evolved % (ladder job) | EQ GPU ms |')
        w('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for label, e in sorted(eq.items()):
            arms = {x['arm']: x for x in e.get('arms', [])}
            for rr in e.get('rules', []):
                q = rr['q']
                ce = arms.get(f'q{q}_m4_eqcert', {})
                st = arms.get(f'q{q}_m4_eqstatic', {})
                dense = rung_rows(dev_seed[label]).get(q, {}).get('evolved') if label in dev_seed else None
                w(f'| {label} | {q} | {rr["population"]} | {rr["m"]} | {rr["relative_fit"]:.2e} | {f(rr["certification"]["rho_max"])} | '
                  f'{f(rr["certification"]["rho_p95"])} | {f(rr["certified_primary"])} | '
                  f'{f(ce.get("worst_evolved_percent")) if rr["population"] == "reachable" else "—"} | '
                  f'{f(st.get("worst_evolved_percent")) if rr["population"] == "static" else "—"} | {f(dense)} | '
                  f'{f(ce.get("median_gpu_ms"), 1) if rr["population"] == "reachable" else f(st.get("median_gpu_ms"), 1)} |')
        w('')

    # ------------------------------------------------- integrity of the bars ----
    n_int = 8 if sealed else 7
    w(f'## {n_int}. Integrity notes on the bars this report is graded against\n')
    w('**The incumbent fidelity bar was weakened before the first job, and this is what it bought.** '
      'The lane pre-registered that the incumbent, re-run in every job, must reproduce the audited '
      'qtd02 ladder (job 3757505) to $10^{-9}$ relative. That bar was amended to $10^{-3}$ before any '
      'job ran (`DESIGN.md` A1.1), because qtd02 **itself** met only the $10^{-3}$ second tier against '
      'its own comparator `cclad01` on three arms — cross-job last-bit drift between GPU models is '
      'already documented for this exact pipeline. The $10^{-9}$ tier is still measured and reported '
      'below as a probe, so the weakening is visible rather than implicit.\n')
    fid = {}
    for name, x in sorted(audits.items()):
        d = (x['checks'].get('incumbent_reproduces_qtd02') or {}).get('detail') or {}
        if d:
            fid[name] = d
    if fid:
        w('| job block | arm | worst relative difference vs qtd02 | meets $10^{-3}$ (the gate) | meets $10^{-9}$ (the probe) |')
        w('|---|---|---|---|---|')
        for name, d in fid.items():
            for arm, v in sorted(d.items()):
                wr = v.get('worst_relative_difference')
                w(f'| `{name}` | `{arm}` | {wr:.2e} | {f(v.get("passed"))} | {f(v.get("passes_probe"))} |'
                  if wr is not None else f'| `{name}` | `{arm}` | — | {f(v.get("passed"))} | {f(v.get("passes_probe"))} |')
        w('')
        n_probe = sum(1 for d in fid.values() for v in d.values() if v.get('passes_probe'))
        n_arm = sum(len(d) for d in fid.values())
        w(f'{n_probe} of {n_arm} incumbent arms also meet the original $10^{{-9}}$ tier; the rest sit between '
          f'$10^{{-9}}$ and $10^{{-3}}$, which is the drift the amendment anticipated.\n')
    else:
        w('*No incumbent block with a qtd02 comparison is present in the audits supplied to this '
          'generator, so the fidelity table is empty.*\n')
    w('**The independent pre-job audit was not Codex.** The protocol asks for an audit by a different '
      'model family before the first job. The Codex quota was exhausted until 2026-09-19 11:33, so the '
      'substitute was an independent Claude agent with no access to this lane\'s conversation '
      '(`reports/independent-design-audit-claude.md`) — a different context, **not** a different model '
      'family. It found the fidelity bar above, the cohort-difficulty problem behind C2n, and eight '
      'other issues, all recorded in `DESIGN.md` A1. Per the coordinator\'s instruction (A3), this '
      'report is published without waiting; the Codex audit of the finished report runs when the quota '
      'returns and is appended as a dated addendum, retracting anything it overturns.\n')
    if sealed:
        w('**The sealed cohort was opened once.** Every development job records '
          '`final_cohort_unopened: true` and stages no sealed configuration; the final job records '
          '`false`. The audit gates that the flag matches the cohort, that the six cases equal the '
          'draw declared in `checks/sealed-cohort.json` to within 1 ulp, and that they are disjoint '
          'from every training, direction, empirical-quadrature and development draw.\n')

    # ------------------------------------------------------------ gates ----
    w(f'## {n_int + 1}. Gates\n')
    for name, x in sorted(audits.items()):
        failed = x['failed']
        w(f'- `{name}` (job {x["job_id"]}, {x["gpu"]}): {len(x["checks"]) - len(failed)} of {len(x["checks"])} gates passed'
          + (f'; **FAILED**: {", ".join(failed)}' if failed else '') + '.')
    w('')
    w('## Glossary\n')
    for term, defn in [
        ('seed', 'one complete retraining of the decoder — bank and head — from a different random initialisation and state pick; the data are identical across seeds'),
        ('incumbent', 'the single frozen checkpoint every earlier Burgers number was measured on, re-run in every job here as the same-job control'),
        ('rung, $q$', 'one setting of the correction ladder: $q$ extra fixed coefficient directions the solver may use on top of the head'),
        ('$M$', 'the number of weak test modes the reduced solve uses; $M=4(K+q)$ in `dense_m4`, 256 in `dense_fixedM`'),
        ('dense quadrature', 'the nonlinear term integrated exactly on the full grid; no empirical-quadrature rule enters the ladder numbers'),
        ('worst evolved / all-times / $t{=}0$', 'the largest relative error over the six cases at output times $t>0$ / including $t=0$ / at $t=0$ only (the decoder\'s compression of the supplied initial field)'),
        ('same-grid', 'measured against the converged full-order solve on the same 256-interval mesh, timed in the same job'),
        ('best-found', 'the best fit the checkpoint\'s own manifold can reach on the reference field, no PDE involved'),
        ('bank floor', 'the best any coefficients in the bank could do; a floor no head can beat'),
        ('converged', 'every time step and the initial fit met the stationarity rule with zero budget exits'),
        ('monotone', 'the error does not increase from one rung to the next'),
        ('error span / cost span', 'largest over smallest error / GPU time across the rungs'),
        ('knob bar', 'the pre-registered tunability criterion: the converged non-dominated set spans at least 2× in error and 2× in cost'),
        ('development cohort / sealed cohort', 'the six cases every earlier lane measured on / six cases drawn once and opened only in the final job'),
        ('TR', 'the training-reproduction bar: a seed counts as reproducing the incumbent recipe if its best-found is within 1.5× of the incumbent\'s and its bank floor within 2×'),
        ('EQ certification, $\\rho$', 'refitting the empirical-quadrature rule on states the solver actually reaches and grading it by its held-out relative error on such states'),
        ('mean ± std', 'sample mean and sample standard deviation over the three seeds'),
        ('gate', 'a check the audit makes on the raw outputs; a failed gate is reported, never hidden'),
    ]:
        w(f'- **{term}**: {defn}')
    out_md.write_text('\n'.join(L) + '\n')
    print('wrote', out_md, 'and summary.json with', len(summary), 'rows')
    print(json.dumps({k: v for k, v in verdict.items() if not isinstance(v, dict)}, indent=1, default=str))


if __name__ == '__main__':
    main()
