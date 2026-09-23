"""summary.json + SUMMARY.md for one heat-bank-knob panel, generated only from results.json and audit_np.json.
Applies the pre-registered setting rule of DESIGN.md to VALIDATION rows, then reads the frozen settings on the
held-out cohort. Usage: hbk_summarize.py <run dir> (expects <run>/pull/out/panel/results.json, <run>/pull/out/audit_np.json)
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1]); pdir = Path(sys.argv[2]) if len(sys.argv) > 2 else run / 'pull' / 'out'
raw = (pdir / 'panel' / 'results.json').read_bytes(); res = json.loads(raw); cfg = res['config']
apath = run / 'audit_np_v2.json' if (run / 'audit_np_v2.json').exists() else pdir / 'audit_np.json'   # local re-run of audit v2 counts when present
audit = json.loads(apath.read_text()) if apath.exists() else None
reps = cfg['repetitions']; fams = list(cfg['families']); K = {2: 8, 3: 32}[cfg['dim']]
Rfull = cfg['edges'][-1]


def parse(m):
    """-> dict(group, R, q, fam) for model arms."""
    p = m.split('_')
    if m.startswith('nmrom_'): return dict(group='nmrom', R=int(p[1][1:]), q=int(p[2][1:]), fam=p[3])
    if m.startswith('lin_'): return dict(group='lin', R=int(p[1][1:]), q=int(p[1][1:]), fam=p[2])
    if m.startswith('parent_lin'): return dict(group='parent', R=Rfull, q=Rfull, fam=p[2])
    if m.startswith('parent_q'): return dict(group='parent', R=Rfull, q=int(p[1][1:]), fam=p[2])
    return dict(group='fom' if m.startswith('fom_') else 'control', R=None, q=None, fam=None)


def row_stats(r, idx):
    same = np.asarray(r['same'])[idx]; phys = np.asarray(r['physical'])[idx]
    ms = np.asarray(r['device_ms']).reshape(-1, reps)[idx]
    return dict(method=r['method'], **parse(r['method']), cases=len(idx),
                err_worst=float(same.max()), err_median=float(np.median(same.max(1))), err_evolved_worst=float(same[:, 1:].max()),
                physical_worst=float(phys.max()), ms_median=float(np.median(ms)), ms_p10=float(np.percentile(ms, 10)), ms_p90=float(np.percentile(ms, 90)),
                failures=int(np.sum(np.asarray(r['failures'])[idx])), fingerprint_mismatch=int(r['fingerprint_mismatch']),
                step_attempts_mean=(float(np.mean([s['step_attempts_mean'] for s in np.asarray(r['stats'], dtype=object)[idx]])) if r['method'].startswith(('nmrom', 'parent_q')) else None))


def fastest_fom(rows, err):
    pool = [r for r in rows.values() if r['group'] == 'fom' and r['failures'] == 0 and r['err_worst'] <= err]
    return min(pool, key=lambda r: r['ms_median']) if pool else None


summary = dict(schema='heat-bank-knob-summary-v1', results_sha256=hashlib.sha256(raw).hexdigest(), source_commit=res['source_commit'],
               source_sha256=res['source_sha256'], model_sha256=res['model_sha256'], prep_sha256=res['prep_sha256'],
               reporting_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), metadata=res['metadata'],
               audit_passed=None if audit is None else audit['passed'], audit=None if audit is None else {k: v for k, v in audit.items() if k != 'failures'}, audit_file=str(apath.name),
               rotation={k: res['rotation'][k] for k in ('cumulative_energy', 'training_projection_floor_worst', 'L_times_T_identity_deviation', 'T_sha256')},
               meshes=[])
md = []
for mesh in res['meshes']:
    n = mesh['intervals']; cases = mesh['cases']; byname = {r['method']: r for r in mesh['rows']}
    cohorts = list(dict.fromkeys(c['cohort'] for c in cases)); val, held = cohorts[0], cohorts[-1]
    assert val.startswith('validation') and held != val
    R = {c: {m: row_stats(r, [i for i, cc in enumerate(cases) if cc['cohort'] == c]) for m, r in byname.items()} for c in cohorts}
    # gates
    parity = max((p['relative_difference'] for p in mesh['parity']), default=None)
    fp = sum(r['fingerprint_mismatch'] for r in mesh['rows'])
    nbr = {}
    for m, recs in mesh['neighbour'].items():
        cs = sorted({x['case'] for x in recs}); main = np.median(np.asarray(byname[m]['device_ms']).reshape(-1, reps)[cs]); nb = np.median([x['ms'] for x in recs])
        ok = (nb <= 1.10 * main) if main >= 1.0 else (abs(nb - main) <= 0.1)
        nbr[m] = dict(main_ms=float(main), neighbour_ms=float(nb), ratio=float(nb / main), passed=bool(ok))
    gates = dict(parity_max_relative_difference=parity, parity_passed=None if parity is None else parity <= 1e-10,
                 fingerprint_mismatches=fp, determinism_passed=fp == 0,
                 neighbour_passed=bool(nbr) and set(nbr) == {m for m, r in byname.items() if parse(m)['group'] in ('nmrom', 'lin', 'parent')} and all(v['passed'] for v in nbr.values()), neighbour_failures={m: v for m, v in nbr.items() if not v['passed']},
                 neighbour_max_ratio=max(v['ratio'] for v in nbr.values()) if nbr else None,
                 audit_passed=None if audit is None else audit['passed'],
                 weak_matrix_rotation_identity=mesh.get('weak_matrix_rotation_identity'), weak_matrix_relative_error=mesh.get('weak_matrix_relative_error'))
    gates['results_complete'] = bool(res['complete'])
    gates['usable'] = bool(res['complete'] and gates['determinism_passed'] and gates['neighbour_passed'] and gates['audit_passed'] and gates['parity_passed'] is not False)
    # selection on validation, frozen, read on held-out
    sel = {}
    for fam in fams + ['pooled']:
        inf = (lambda r: r['group'] in ('nmrom', 'lin') and (fam == 'pooled' or r['fam'] == fam))
        pv = {m: r for m, r in R[val].items() if inf(r) and r['failures'] == 0}
        ref_names = [cfg['paper_fast'][f] for f in (fams if fam == 'pooled' else [fam])]
        ref_err = min(R[val][x]['err_worst'] for x in ref_names)
        acc = min(pv, key=lambda m: (pv[m]['err_worst'], pv[m]['ms_median']))
        fastpool = [m for m in pv if pv[m]['err_worst'] <= ref_err]
        fast = min(fastpool, key=lambda m: pv[m]['ms_median'])
        ha, hf = R[held][acc], R[held][fast]; fom = fastest_fom(R[held], ha['err_worst'])
        href_err = min(R[held][x]['err_worst'] for x in ref_names)
        sel[fam] = dict(accurate=acc, fast=fast, validation=dict(accurate=pv[acc], fast=pv[fast], paper_fast_err=ref_err),
                        heldout=dict(accurate=ha, fast=hf, paper_fast=[R[held][x] for x in ref_names], paper_fast_err=href_err,
                                     fast_meets_rule_on_heldout=hf['err_worst'] <= href_err,
                                     fom=fom and dict(method=fom['method'], err_worst=fom['err_worst'], ms=fom['ms_median']),
                                     speedup_accurate=fom and fom['ms_median'] / ha['ms_median'], speedup_fast=fom and fom['ms_median'] / hf['ms_median'],
                                     failures=ha['failures'] + hf['failures']))
    # every arm vs its own matched FOM (held-out and validation)
    for c in cohorts:
        for m, r in R[c].items():
            f = fastest_fom(R[c], r['err_worst']); r['own_fom'] = f and f['method']; r['own_fom_ms'] = f and f['ms_median']; r['speedup_own_fom'] = f and f['ms_median'] / r['ms_median']
            nm = R[c].get('fom_cncg_dt0.025_rtol1e-6_NAMED'); r['speedup_named'] = nm['ms_median'] / r['ms_median']
    # knob verdict (held-out): monotone worst error as R' falls
    mono = {}
    for fam in fams:
        for lab, pick in (('q0', lambda r: r['group'] == 'nmrom' and r['q'] == 0), ('qmax', lambda r: r['group'] == 'nmrom' and r['q'] == r['R'] - K),
                          ('linear', lambda r: r['group'] == 'lin')):
            ser = sorted([r for r in R[held].values() if r['fam'] == fam and pick(r)], key=lambda r: -r['R'])
            errs = [r['err_worst'] for r in ser]
            mono[f'{fam}_{lab}'] = dict(R=[r['R'] for r in ser], err_worst=errs, ms=[r['ms_median'] for r in ser],
                                        failures=[r['failures'] for r in ser], complete=[r['R'] for r in ser] == list(cfg['ladder']),
                                        monotone=bool(all(b >= a * (1 - 1e-9) for a, b in zip(errs, errs[1:]))))
    for fam in fams:   # registered timing condition: linear rung R'=R vs the cheapest R' meeting the fast rule (held-out), >= 2x
        lin = {r['R']: r for r in R[held].values() if r['group'] == 'lin' and r['fam'] == fam}; ref = min(R[held][cfg['paper_fast'][fam]]['err_worst'], 1e9)
        ok = [r for r in lin.values() if r['err_worst'] <= ref and r['failures'] == 0]
        cheap = min(ok, key=lambda r: r['ms_median']) if ok else None
        mono[f'{fam}_linear_time_drop'] = dict(full_ms=lin[Rfull]['ms_median'], cheapest_R=cheap and cheap['R'], cheapest_ms=cheap and cheap['ms_median'],
                                               ratio=cheap and lin[Rfull]['ms_median'] / cheap['ms_median'], passed=bool(cheap and lin[Rfull]['ms_median'] / cheap['ms_median'] >= 2))
    summary['meshes'].append(dict(intervals=n, unknowns=mesh['unknowns'], cohorts=cohorts, cases={c: sum(cc['cohort'] == c for cc in cases) for c in cohorts},
                                  bank_bytes=mesh['bank_bytes'], truncated_bank_condition=mesh['truncated_bank_condition'], setup_seconds=mesh['setup_seconds'],
                                  gates=gates, neighbour=nbr, selection=sel, monotonicity=mono, profile_ms=mesh['profile'], rows=R))
    # markdown
    md.append(f"\n## {n} intervals per axis ({mesh['unknowns']:,} unknowns)\n")
    md.append(f"Gates: parity {gates['parity_max_relative_difference']} ({gates['parity_passed']}); determinism {gates['determinism_passed']}; "
              f"neighbour {gates['neighbour_passed']} (max ratio {gates['neighbour_max_ratio']:.3f}); audit {gates['audit_passed']}.\n")
    for fam in fams + ['pooled']:
        s = sel[fam]; h = s['heldout']
        md.append(f"- **{fam}**: accurate `{s['accurate']}` (val {100*s['validation']['accurate']['err_worst']:.4f} %) -> held-out {100*h['accurate']['err_worst']:.4f} % in {h['accurate']['ms_median']:.3f} ms; "
                  f"fast `{s['fast']}` (val {100*s['validation']['fast']['err_worst']:.4f} % <= {100*s['validation']['paper_fast_err']:.4f} %) -> held-out {100*h['fast']['err_worst']:.4f} % in {h['fast']['ms_median']:.3f} ms "
                  f"(held-out rule met: {h['fast_meets_rule_on_heldout']}); FOM `{h['fom'] and h['fom']['method']}` ({h['fom'] and round(100*h['fom']['err_worst'], 4)} %, {h['fom'] and round(h['fom']['ms'], 3)} ms): "
                  f"x{h['speedup_accurate'] and round(h['speedup_accurate'], 2)} / x{h['speedup_fast'] and round(h['speedup_fast'], 2)}")
    for c in cohorts:
        md += [f"\n### [{c}] {R[c][next(iter(R[c]))]['cases']} cases x {reps} reps\n",
               "| method | R' | q | fam | worst % | median % | GPU ms | fails | own FOM | x own FOM | x named |", "|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|"]
        for m, r in sorted(R[c].items(), key=lambda kv: (kv[1]['group'] in ('fom', 'control'), kv[1]['fam'] or '', -(kv[1]['R'] or 0), kv[1]['q'] or 0, kv[0])):
            so = '-' if r['speedup_own_fom'] is None else f"{r['speedup_own_fom']:.2f}"
            md.append(f"| {m} | {r['R'] or ''} | {'' if r['q'] is None else r['q']} | {r['fam'] or ''} | {100*r['err_worst']:.4f} | {100*r['err_median']:.4f} | {r['ms_median']:.3f} | {r['failures']} | "
                      f"{r['own_fom'] or '-'} | {so} | {r['speedup_named']:.2f} |")
    if mesh['profile']:
        md += ['\n### Stage profile (held-out case 0, median ms)\n', '| arm | encode | init | evolve | decode | query |', '|---|---:|---:|---:|---:|---:|']
        for m, p in mesh['profile'].items():
            md.append(f"| {m} | {p['encode']:.3f} | {p.get('init', float('nan')):.3f} | {p['evolve']:.3f} | {p['decode']:.3f} | {p['query']:.3f} |")
(run / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
(run / 'SUMMARY.md').write_text(f"# {run.name} — generated summary\n\nGenerated by `hbk_summarize.py` from `results.json` (sha256 {summary['results_sha256'][:16]}…), "
                                f"source {res['source_commit']}, GPU `{res['metadata']['gpu']}`, job {res['metadata']['job_id']}. Audit passed: {summary['audit_passed']}. "
                                "Errors: worst (median) over cases of the max over all six output times of the same-grid relative L2 error. GPU ms: median over cases x repetitions. "
                                "x = comparator time / arm time in this allocation. Selection uses validation rows only (DESIGN.md).\n" + '\n'.join(md) + '\n')
print('\n'.join(md[:40]))
