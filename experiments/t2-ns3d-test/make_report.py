"""Generate reports/summary.json and reports/<date>-t2-ns3d-test.md from the pulled panel outputs only.

Usage: make_report.py <report-date> [--smoke smk32]
Test panels: runs/t2t{32,64}/output/{summary.json,audit.json} (+ runs/<job>/audit_local.json).
Development values (current paper Table 2): ../ns3d-operators/runs/pn{32,64}/output/summary.json.
No number is typed by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = HERE.parent / 'ns3d-operators'
FAMILY = {'fno': 'FNO', 'unet': 'U-Net', 'transolver': 'Transolver', 'deeponet': 'DeepONet'}
FAM_ORDER = ['fno', 'unet', 'transolver', 'deeponet']
TEST_JOBS = {32: 't2t32', 64: 't2t64'}
DEV_JOBS = {32: 'pn32', 64: 'pn64'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(x):
    return '—' if x is None else (f'{100 * x:.3g}' if x >= 1e-3 else f'{100 * x:.2g}')


def sp(x):
    if x is None:
        return '—'
    s = f'{x:.3g}×'
    return f'*{s}*' if x < 1 else s


def rel_repo(p):
    return str(Path(p).resolve().relative_to(HERE.parents[1]))


def panel(path_dir):
    sp_ = path_dir / 'output' / 'summary.json'
    s = json.loads(sp_.read_text())
    au_p = path_dir / 'output' / 'audit.json'
    au = json.loads(au_p.read_text()) if au_p.exists() else None
    loc_p = path_dir / 'audit_local.json'
    loc = json.loads(loc_p.read_text()) if loc_p.exists() else None
    return s, sp_, au, au_p, loc, loc_p


def rows(s):
    """Per-arm record from one panel summary."""
    out = {}
    fom = s['fom_rule']['chosen']
    for nm, r in s['results'].items():
        t = s['timing']['arms'].get(nm)
        st = r['stats']
        rec = dict(kind=r['kind'], finite=r['finite'], unstable=r['unstable'],
                   evolved_worst=st['evolved_worst'] if st else None,
                   evolved_median=st['evolved_median'] if st else None,
                   median_ms=t['median_ms'] if t else None,
                   speedup_vs_fom=r.get('speedup_vs_fom'),
                   matched_cnab2=r.get('matched_cnab2'), speedup_vs_matched_cnab2=r.get('speedup_vs_matched_cnab2'),
                   drift_ratio=t['drift_ratio'] if t else None, order_ratio=t['order_ratio'] if t else None,
                   timed_output_gap=r.get('timed_output_gap'), is_fom=(nm == fom))
        if 'steps' in r:
            rec['steps'] = r['steps']
        m = s.get('operators', {}).get(nm)
        if m:
            rec.update(arm=m['arm'], family=m['family'], selected=m['selected'], sha256=m['sha256'],
                       parameters=m['real_parameter_count'], parameter_dtype=m['parameter_dtype'],
                       validation_mean_case_max=m['training']['validation_mean_case_max'],
                       epochs=m['training']['epochs_completed'])
        out[nm] = rec
    return out


def cell(s):
    rr = rows(s)
    fom = s['fom_rule']['chosen']
    sel = {rr[nm]['family']: nm for nm in rr if rr[nm].get('selected')}
    table2 = {'nmrom_accurate_head_k8': rr['nmrom_accurate_head_k8'], 'nmrom_fast_span16': rr['nmrom_fast_span16']}
    for fam in FAM_ORDER:
        if fam in sel:
            table2[sel[fam]] = rr[sel[fam]]
    # most accurate trained size per family on this cohort (context for the caption wording)
    best_on_cohort = {}
    for fam in FAM_ORDER:
        cand = [nm for nm in rr if rr[nm].get('family') == fam and rr[nm]['finite']]
        if cand:
            best_on_cohort[fam] = min(cand, key=lambda nm: rr[nm]['evolved_worst'])
    return dict(job_id=s['job_id'], gpu=s['gpu'], status=s['status'], gates=s['gates'],
                eval_seed=s['config']['eval_seed'], eval_cases=s['config']['eval_cases'],
                eval_parameter_sha256=s.get('eval_parameter_sha256'),
                reproduction_gate=dict(passed=s['reproduction_gate']['passed'],
                                       max_relative_gap=max([v['max_relative_gap'] for v in
                                                             s['reproduction_gate']['arms'].values()] or [0.0]),
                                       reference=s['config'].get('reference_summary'),
                                       reference_job=s['config'].get('reference_summary_job')),
                bank_rebuild_gap=s['bank_rebuild_gap'], elapsed_seconds=s['elapsed_seconds'],
                source_commit=s['source_commit'], config_sha256=s['config_sha256'],
                fom=dict(setting=fom, steps=rr[fom]['steps'] if fom else None,
                         evolved_worst=rr[fom]['evolved_worst'] if fom else None,
                         median_ms=rr[fom]['median_ms'] if fom else None,
                         candidates=s['fom_rule']['candidates']),
                table2=table2, all_arms=rr, most_accurate_size_on_cohort=best_on_cohort)


LABEL = {'nmrom_accurate_head_k8': 'NM-ROM accurate (head k=8)', 'nmrom_fast_span16': "NM-ROM span R'=16"}


def label(nm, r):
    if nm in LABEL:
        return LABEL[nm]
    if r.get('family'):
        return f"{FAMILY[r['family']]} ({r['arm']})"
    return nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date')
    ap.add_argument('--smoke', default=None, help='generate from one smoke job only (to test this script)')
    args = ap.parse_args()
    jobs = {32: args.smoke} if args.smoke else TEST_JOBS
    summary = dict(schema='t2-ns3d-test-summary-v1', smoke=bool(args.smoke), meshes={}, inputs={})
    for n, job in jobs.items():
        d = HERE / 'runs' / job
        s, sp_, au, au_p, loc, loc_p = panel(d)
        dv, dvp, dau, dau_p, _, _ = panel(OPS / 'runs' / DEV_JOBS[n])
        summary['inputs'][f'test{n}'] = dict(job=job, summary=rel_repo(sp_), summary_sha256=sha(sp_),
                                             audit=rel_repo(au_p) if au else None,
                                             audit_sha256=sha(au_p) if au else None,
                                             audit_local_sha256=sha(loc_p) if loc else None)
        summary['inputs'][f'dev{n}'] = dict(job=DEV_JOBS[n], summary=rel_repo(dvp), summary_sha256=sha(dvp))
        summary['meshes'][str(n)] = dict(
            test=cell(s) | dict(audit_passed=(au or {}).get('all_passed'),
                                audit_local_passed=(loc or {}).get('all_passed'),
                                audit_perturbed_rejected=(au or {}).get('perturbed_control_rejected')),
            development=cell(dv))
    out_dir = HERE / 'reports'
    sj = out_dir / ('summary_smoke.json' if args.smoke else 'summary.json')
    sj.write_text(json.dumps(summary, indent=1) + '\n')

    L = []
    w = L.append
    w('# Table 2, Navier–Stokes 3D at 32³ and 64³ on the 32 held-out test cases')
    w('')
    w('NM-ROM versus FNO / U-Net / Transolver / DeepONet and the CNAB2 full-order solver, re-evaluated on the '
      '32 held-out test cases (seed 202609221, the 96³ cohort) with the frozen models and settings of the current '
      'Table 2 and the pn96c protocol, one job per mesh. '
      + ('**SMOKE OUTPUT — not a result.**' if args.smoke else
         'Numbers are final where the panel status says `final`; the development values beside them are the '
         'ones currently printed in Table 2.'))
    w('')
    w(f"Generated by `make_report.py` from `{sj.name}` (inputs and their sha256 are listed in §4). "
      'Pre-registration: `DESIGN.md`.')
    w('')
    w('## 1. Table 2 cells: test versus development')
    w('')
    w('Worst evolved error % / median GPU ms / speedup against the cell FOM (the fastest CNAB2 setting at least as '
      'accurate as the NM-ROM accurate setting, same job). Operator size per family = the validation selection of '
      'ns3d-operators (unchanged). *Italic* speedups are below 1×.')
    w('')
    for n in jobs:
        m = summary['meshes'][str(n)]
        t, dv = m['test'], m['development']
        w(f'### {n}³')
        w('')
        w(f"Test: job {t['job_id']} ({t['gpu'].split(',')[0]}), status **{t['status']}**, {t['eval_cases']} cases. "
          f"Development: job {dv['job_id']} ({dv['gpu'].split(',')[0]}), status {dv['status']}, {dv['eval_cases']} cases.")
        w('')
        w('| method | test error % | test ms | test speedup | dev error % | dev ms | dev speedup |')
        w('|---|---|---|---|---|---|---|')
        f_t, f_d = t['fom'], dv['fom']
        w(f"| FOM: CNAB2 (test {f_t['steps']} steps; dev {f_d['steps']} steps) | {pct(f_t['evolved_worst'])} | "
          f"{f_t['median_ms']:.3g} | 1× | {pct(f_d['evolved_worst'])} | {f_d['median_ms']:.3g} | 1× |")
        for nm, r in t['table2'].items():
            d = dv['all_arms'].get(nm)
            dsel = '' if (d is None or d.get('selected', True)) else ' (not dev pick)'
            w(f"| {label(nm, r)} | {pct(r['evolved_worst'])} | {r['median_ms']:.3g} | {sp(r['speedup_vs_fom'])} | "
              f"{pct(d['evolved_worst']) if d else '—'} | {d['median_ms']:.3g} | {sp(d['speedup_vs_fom'])}{dsel} |")
        w('')
    w('## 2. Every evaluated arm (test panels)')
    w('')
    w('All eight trained operator checkpoints per mesh and the full CNAB2 ladder, same job. "val %" = validation '
      'mean-case-max of the checkpoint (the selection criterion). "matched" = speedup against the fastest CNAB2 at '
      'least as accurate as that arm (context only, not the Table 2 rule).')
    w('')
    for n in jobs:
        t = summary['meshes'][str(n)]['test']
        dv = summary['meshes'][str(n)]['development']
        w(f'### {n}³ (job {t["job_id"]})')
        w('')
        w('| arm | selected | val % | test worst % | test median % | ms | × FOM | matched CNAB2 | × matched | '
          'drift | order | dev worst % |')
        w('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for nm, r in t['all_arms'].items():
            d = dv['all_arms'].get(nm)
            sel = '' if r['kind'] != 'operator' else ('yes' if r.get('selected') else 'no')
            val = pct(r['validation_mean_case_max']) if r.get('validation_mean_case_max') is not None else ''
            w(f"| {label(nm, r)}{' **(FOM)**' if r['is_fom'] else ''} | {sel} | {val} | {pct(r['evolved_worst'])} | "
              f"{pct(r['evolved_median'])} | {r['median_ms']:.3g} | {sp(r['speedup_vs_fom'])} | "
              f"{r['matched_cnab2'] or '—'} | {sp(r['speedup_vs_matched_cnab2'])} | "
              f"{r['drift_ratio']:.3f} | {r['order_ratio']:.3f} | {pct(d['evolved_worst']) if d else '—'} |")
        w('')
        mac = t['most_accurate_size_on_cohort']
        dmac = dv['most_accurate_size_on_cohort']
        w('Most accurate trained size per family: '
          + '; '.join(f"{FAMILY[f]} test {t['all_arms'][mac[f]]['arm']} / dev {dv['all_arms'][dmac[f]]['arm']}"
                      for f in FAM_ORDER if f in mac and f in dmac)
          + '. Table 2 uses the validation pick, not these.')
        w('')
    w('## 3. Gates and audit')
    w('')
    w('| mesh | status | reproduction gate (max rel. gap vs ns3d-test job) | bank rebuild gap | drift | order | '
      'positive control | timed outputs | coverage | in-job audit | local audit | perturbed copy rejected |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for n in jobs:
        t = summary['meshes'][str(n)]['test']
        g = t['gates']
        rg = t['reproduction_gate']
        w(f"| {n}³ | {t['status']} | {rg['passed']} ({rg['max_relative_gap']:.2g} vs {rg['reference_job']}) | "
          f"{t['bank_rebuild_gap']:.2g} | {g['drift']} | {g['order']} | {g['positive_control_failed_as_required']} | "
          f"{g['timed_outputs_match']} | {g['coverage']} | {t['audit_passed']} | {t['audit_local_passed']} | "
          f"{t['audit_perturbed_rejected']} |")
    w('')
    w('The audit is **restricted**: exact recomputation on the saved full fields (cases 0, 1 and each arm\'s worst '
      'case), sampled estimates elsewhere (diagnostic), the truth solver not independently re-solved. Gates cover '
      'the two NM-ROM arms, the rule FOM and the four selected operators.')
    w('')
    w('## 4. Inputs')
    w('')
    w('| input | job | path | sha256 |')
    w('|---|---|---|---|')
    for k, v in summary['inputs'].items():
        w(f"| {k} | {v['job']} | `{v['summary']}` | `{v['summary_sha256']}` |")
        if v.get('audit'):
            w(f"| {k} audit | {v['job']} | `{v['audit']}` | `{v['audit_sha256']}` |")
    w('')
    w('## Glossary')
    w('')
    w('- **Test cases**: 32 initial conditions/viscosities drawn with seed 202609221, never used to choose any '
      'model, size or setting. **Development cases**: 16 cases (seed 202609202) on which the NM-ROM setting k=8 '
      'was chosen; the current Table 2 values at 32³/64³ are on these.')
    w('- **Worst evolved error**: over cases, the maximum over the five output times t=0.04…0.2 of '
      '‖u − u_truth‖₂ / ‖u₀‖₂, truth = CNAB2 at Δt = 0.001 on the same mesh. **Median** = median over cases.')
    w('- **NM-ROM accurate**: nonlinear head with latent size k=8 on a rank-64 POD bank, implicit midpoint '
      'Δt = 0.02, 3 Gauss–Newton sweeps. **Span R\'=16**: linear reduced model on the first 16 importance-ordered '
      'bank directions, same stepping (Table 2\'s second NM-ROM setting).')
    w('- **CNAB2**: the pseudo-spectral full-order solver (Crank–Nicolson / Adams–Bashforth 2) with a given number '
      'of steps over T = 0.2. **FOM (cell comparator)**: the fastest CNAB2 setting in the same job whose worst '
      'evolved error is ≤ the NM-ROM accurate setting\'s. **Speedup** = FOM median time / method median time.')
    w('- **Selected / val %**: each operator family was trained at two sizes (-s, -l) on the NM-ROM\'s 448 '
      'training trajectories; the Table 2 size is the one with the lower validation (64 trajectories) mean-case-max '
      'error, chosen before any panel.')
    w('- **A–B–A timing**: A1 interleaved rounds, B arm-major rounds, A2 interleaved rounds; reported time = median '
      'of all 9 samples per case. **Drift** = median(A2)/median(A1); **order** = median(B)/median(A1∪A2); both must '
      'lie in [1/1.10, 1.10]. **Positive control**: the drift gate must reject A2 inflated by 15 %.')
    w('- **Reproduction gate**: the NM-ROM and CNAB2 per-case test errors must equal those of the earlier ns3d-test '
      'job on the same cohort to 1e-6 relative. **Bank rebuild gap**: the POD bank rebuilt from the seed must match '
      'the frozen bank\'s probe to 1e-8.')
    w('- **Restricted audit**: an independent NumPy recomputation of saved fields, timing medians, the FOM rule and '
      'speedups; it must also reject a copy with one field perturbed by 1e-6.')
    rp = out_dir / (f'{args.date}-t2-ns3d-test{"-smoke" if args.smoke else ""}.md')
    rp.write_text('\n'.join(L) + '\n')
    print(rp, sha(rp))
    print(sj, sha(sj))


if __name__ == '__main__':
    main()
