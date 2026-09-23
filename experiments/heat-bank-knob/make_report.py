"""REPORT.md + report.json from runs/*/summary.json (every number generated; nothing typed by hand)."""
import hashlib, json
from pathlib import Path

here = Path(__file__).resolve().parent
JOBS = [('h2d', '2D wide bank (R=128, K=8)'), ('h3d', '3D new bank (R=320, K=32)'), ('h3d256', '3D new bank (R=320, K=32), 256^3')]
FAMS = {'cn': 'CN stepping', 'bf': 'batched fit', 'pooled': 'both families pooled'}
out, rep = [], dict(schema='heat-bank-knob-report-v1', jobs={}, table1=[])
pct = lambda x: f'{100 * x:.3f}'
for job, label in JOBS:
    p = here / 'runs' / job / 'summary.json'
    if not p.exists(): continue
    S = json.loads(p.read_text()); md_ = S['metadata']
    rep['jobs'][job] = dict(summary=str(p.relative_to(here)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), job_id=md_['job_id'],
                            gpu=md_['gpu'].split(',')[0], source_commit=S['source_commit'], audit_passed=S['audit_passed'])
    out.append(f"\n# {label} — job {md_['job_id']}, {md_['gpu'].split(',')[0]}, source {(S['source_commit'] or 'none')[:8]}, audit passed: {S['audit_passed']}\n")
    out.append(f"`runs/{job}/summary.json` sha256 `{rep['jobs'][job]['sha256']}`.\n")
    for m in S['meshes']:
        n = m['intervals']; g = m['gates']; held = m['cohorts'][-1]; val = m['cohorts'][0]; R = m['rows'][held]
        out.append(f"\n## {n}^{2 if job == 'h2d' else 3} — held-out `{held}` ({m['cases'][held]} cases), selection on `{val}` ({m['cases'][val]} cases)\n")
        out.append(f"Gates: parity max rel. diff {g['parity_max_relative_difference']} (pass {g['parity_passed']}); determinism {g['determinism_passed']}; "
                   f"order-effect (neighbour) {g['neighbour_passed']} (max ratio {g['neighbour_max_ratio']:.3f}); NumPy audit {g['audit_passed']}.\n")
        out += ['| family | role | arm | R\' | q | val worst % | held-out worst % | held-out median % | GPU ms | Table-1 FOM | FOM worst % | FOM ms | speedup | fast rule met on held-out |',
                '|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|']
        for fam, s in m['selection'].items():
            h = s['heldout']; fom = h['fom'] or {}
            for role in ('accurate', 'fast'):
                a = h[role]
                out.append(f"| {FAMS[fam]} | {role} | `{s[role]}` | {a['R']} | {a['q']} | {pct(s['validation'][role]['err_worst'])} | {pct(a['err_worst'])} | {pct(a['err_median'])} | {a['ms_median']:.3f} | "
                           f"`{fom.get('method')}` | {pct(fom['err_worst']) if fom else '-'} | {fom.get('ms', float('nan')):.2f} | {h['speedup_' + role] or float('nan'):.2f} | {h['fast_meets_rule_on_heldout'] if role == 'fast' else ''} |")
                rep['table1'].append(dict(job=job, mesh=n, family=fam, role=role, arm=s[role], R=a['R'], q=a['q'], heldout_err_worst_pct=100 * a['err_worst'],
                                          heldout_err_median_pct=100 * a['err_median'], heldout_err_evolved_worst_pct=100 * a['err_evolved_worst'], gpu_ms=a['ms_median'],
                                          fom=fom.get('method'), fom_err_pct=100 * fom['err_worst'] if fom else None, fom_ms=fom.get('ms'), speedup=h['speedup_' + role],
                                          validation_err_worst_pct=100 * s['validation'][role]['err_worst'], paper_fast_heldout_err_pct=100 * h['paper_fast_err'],
                                          fast_rule_met_heldout=h['fast_meets_rule_on_heldout'], failures=a['failures']))
        out += ['', f"Full held-out table ({held}): every (R', q, stepping) arm. `x T1` = the family's Table-1 FOM time / arm time; `x own` = fastest FOM at least as accurate as the arm; `x named` = CN-CG dt 0.025 rtol 1e-6.", '',
                "| arm | R' | q | stepping | worst % | median % | GPU ms (p10-p90) | fails | x T1 | own FOM | x own | x named |", '|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|']
        t1 = {fam: (s['heldout']['fom'] or {}).get('ms') for fam, s in m['selection'].items()}
        rows = sorted([r for r in R.values() if r['group'] in ('nmrom', 'lin', 'parent')], key=lambda r: (r['fam'], r['group'] == 'parent', -r['R'], r['q']))
        for r in rows:
            x1 = f"{t1[r['fam']] / r['ms_median']:.2f}" if t1.get(r['fam']) else '-'
            xo = f"{r['speedup_own_fom']:.2f}" if r['speedup_own_fom'] else '-'
            out.append(f"| `{r['method']}` | {r['R']} | {r['q']} | {r['fam']} | {pct(r['err_worst'])} | {pct(r['err_median'])} | {r['ms_median']:.3f} ({r['ms_p10']:.2f}-{r['ms_p90']:.2f}) | {r['failures']} | {x1} | "
                       f"{r['own_fom'] or '-'} | {xo} | {r['speedup_named']:.2f} |")
        out += ['', '| FOM / control | worst % | median % | GPU ms | fails |', '|---|---:|---:|---:|---:|']
        for r in sorted([r for r in R.values() if r['group'] in ('fom', 'control')], key=lambda r: r['ms_median']):
            out.append(f"| `{r['method']}` | {pct(r['err_worst'])} | {pct(r['err_median'])} | {r['ms_median']:.3f} | {r['failures']} |")
        out += ['', 'Knob monotonicity (held-out worst error as R\' falls):', '']
        for k, v in m['monotonicity'].items():
            out.append(f"- {k}: R' {v['R']} -> worst % {[round(100 * e, 3) for e in v['err_worst']]}, ms {[round(x, 2) for x in v['ms']]}; monotone {v['monotone']}")
        if m['profile_ms']:
            out += ['', f"Cost profile at {n} (held-out case 0, median ms of 5; stages timed separately, so they need not sum to the fused query):", '',
                    "| arm | encode (G'^T u / moments) | init fit | evolve | decode (G'c) | fused query |", '|---|---:|---:|---:|---:|---:|']
            for a, p in m['profile_ms'].items():
                out.append(f"| `{a}` | {p['encode']:.3f} | {p.get('init', 0.):.3f} | {p['evolve']:.3f} | {p['decode']:.3f} | {p['query']:.3f} |")
        rep['jobs'][job].setdefault('meshes', {})[str(n)] = dict(gates=g, monotonicity=m['monotonicity'], profile_ms=m['profile_ms'])
gloss = """
## Glossary

- **R'**: number of leading columns of the rotated frozen bank used by a query (R' = R is the unmodified bank). The rotation orders columns by training energy (SVD of training coefficient vectors); computed once offline from training data only.
- **q**: number of linear correction directions eliminated in closed form. **Linear rung** (`lin_R*`, q = R'): the head is dropped and all R' coefficients are free.
- **cn / CN stepping**: reduced Crank–Nicolson weak steps, dt 0.025 (the paper's CN arm). **bf / batched fit**: each output time fitted independently to exactly propagated test moments of the supplied field (valid for this linear autonomous PDE).
- **parent_***: the unrotated parent model (original code path), used for the parity gate at R' = R.
- **worst / median %**: worst / median over cases of the maximum over all six output times (including t = 0) of the same-grid relative L2 error.
- **validation** cohort: training-time validation draws, used to choose settings. **held-out**: the sealed cohort Table 1 uses (already opened once for Table 1); settings are frozen before reading it.
- **accurate / fast**: pre-registered rule (DESIGN.md) — most accurate arm; cheapest arm at least as accurate as the current paper fast setting (q = 0, full R), both chosen on validation.
- **Table-1 FOM**: fastest tested CN–CG setting with no failed solve whose held-out worst error is at most the accurate arm's; both roles of a row divide its time. **own FOM**: the same, matched to the arm's own error. **named**: CN–CG dt 0.025 rtol 1e-6.
- **speedup / x**: FOM median GPU time / arm median GPU time, same allocation. **GPU ms**: median over cases × 5 repetitions of the GPU query (supplied field on device to six fields on device).
- **Gates**: parity (rotated vs unrotated at R' = R ≤ 1e-10), determinism (repetition fingerprints identical), neighbour/order-effect (arm re-timed right after a CG solve within 1.10× of its main-phase median), independent NumPy audit with two controls that must be detected.
- **encode / init / evolve / decode**: G'^T u (or DST moments) / initial latent fit / time evolution / reconstruction u = G'c.
"""
(here / 'REPORT.md').write_text("# heat-bank-knob — nested bank truncation R' on heat 2D/3D (generated)\n\nGenerated by `make_report.py` from `runs/*/summary.json`; every number comes from those files. "
                                "Status: see the gates per mesh; numbers from a job whose gates fail are not usable.\n" + '\n'.join(out) + '\n' + gloss)
(here / 'report.json').write_text(json.dumps(rep, indent=1) + '\n')
print('\n'.join(out[:60]))
