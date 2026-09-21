"""Generate reports/2026-09-21-nmrom-baselines.md and summary.json from the audited run JSONs. No number is typed by hand."""
import json, hashlib
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
pc = lambda x: '—' if x is None else ('diverged' if x != x or x == float('inf') or x > 1e3 else f'{100 * x:.2f} %')

gates = []
for p in sorted(LANE.glob('runs/gate*/output/**/summary.json')):
    s = json.loads(p.read_text()); g = s['gate']
    gates.append(dict(path=str(p.relative_to(LANE)), sha256=sha(p), job=s['provenance']['job_id'], attempt=g['attempt'], recipe=g['recipe'],
                      trained=all('loaded' not in r['train']['u'] for r in s['seeds']), errors=g['nm_lspg_errors'], median=g['nm_lspg_median'],
                      ls=g['ls_lspg'], hr=g['hr_errors'], hr_basis=s['config'].get('hr_basis', 'residual'), passed=g['passed'], hr_passed=g['hr_passed'],
                      kimae=s['source_sha256']['kimae.py'], lspg=s['source_sha256']['lspg.py'],
                      autoencode=[r['nm_projection']['max'] for r in s['seeds']]))
passed = [g for g in gates if g['passed'] and g['trained']]
passed_acts = {g['recipe']['act'] for g in passed}

fams = []
for p in sorted(LANE.glob('runs/fam*/output/summary.json')):
    s = json.loads(p.read_text())
    a = json.loads((p.parents[1] / 'audit.json').read_text()) if (p.parents[1] / 'audit.json').exists() else None
    gb = s['gate_binding']
    ksha = gb.get('kimae_sha256') or next((g['kimae'] for g in gates), None)
    admissible = any(g['kimae'] == ksha and g['lspg'] == (gb.get('lspg_sha256') or g['lspg']) for g in passed)   # code identity only; the activation is bound per arm below
    fams.append(dict(path=str(p.relative_to(LANE)), sha256=sha(p), s=s, audit=a, admissible=admissible))

out = ['# Kim et al. masked-autoencoder NM-LSPG versus the project NM-ROM on the shared Burgers 2D family', '']
verdict = 'PASSED' if passed else 'FAILED'
out += [f'Reproduction gate: **{verdict}**. ' + (('**Adapted** reproduction: passed on attempt ' + ', '.join(str(g['attempt']) for g in passed) + ' of the pre-registered three (activation ' + ', '.join(sorted(passed_acts))
         + '), not on the paper-default attempt 1; median ' + ', '.join(pc(g['median']) for g in passed) + ' against the 1.5 % bar (published < 1 %), i.e. above the published figure. '
         'Hyper-reduction was NOT reproduced (HR gate failed on every attempt), so every HR arm is exploratory. '
         'Kim-baseline rows are admissible only where marked (same code hashes and the gate activation).') if passed else
        'No attempt reproduced the published number within the pre-registered tolerance, so **every Kim-baseline number in this report is from an '
        'unvalidated implementation and is not admissible as a statement about the published method**; it is printed so the work is not lost. '
        'POD-LSPG, project-ROM and FOM rows do not depend on the gate.'),
        f'All numbers are generated from run JSONs by `reports/gen_report.py`; state: {"final for this lane" if fams else "gate only"}.', '']
out += ['## 1. Reproduction gate (Kim et al. 2022, Section 6.2; published NM-LSPG < 1 %, NM-LSPG-HR 0.93–0.98 %, LS-LSPG-HR 34–38 %)', '',
        'Pass rule (pre-registered): median of three seeds ≤ 1.5 % and a valid LS-LSPG control ≥ 10 %; HR: median at 55/58 ≤ 2 %.', '',
        '| job | attempt | activation / scaling | NM-LSPG per seed | median | autoencode-only per seed | LS-LSPG control | HR 55/58 per seed (basis) | gate | HR gate |', '|---|---|---|---|---|---|---|---|---|---|']
for g in gates:
    r = g['recipe']
    out.append(f"| {g['job']} | {g['attempt']}{'' if g['trained'] else ' (weights reloaded)'} | {r['act']} / {r['scale']} | {', '.join(pc(x) for x in g['errors'])} | {pc(g['median'])} | "
               f"{', '.join(pc(x) for x in g['autoencode'])} | {pc(g['ls'])} | {', '.join(pc(x) for x in g['hr'])} ({g['hr_basis']}) | {'pass' if g['passed'] else 'FAIL'} | {'pass' if g['hr_passed'] else 'FAIL'} |")
out.append('')

rows_json = []
# ---------------------------------------------------------------- headline: one row per method x K x mesh
HEAD = []
for f in fams:
    s = f['s']; L = s['intervals']; A = s['arms']
    tf = A.get('fom_fft_tight', {}).get('timing', {}).get('gpu_ms_median')
    def row(name, label):
        a = A[name]; kim = a['family'].startswith('kim'); act = a.get('activation', 'swish') if kim else None
        adm = (not kim) or (f['admissible'] and act in passed_acts)
        t = a.get('timing', {}).get('gpu_ms_median'); mem = a.get('memory_analysis', {}).get('total')
        HEAD.append(dict(mesh=L, method=label, arm=name, solved=a.get('solved_dimension'), cohort=a.get('cohort'), worst=a['worst_evolved'],
                         median=a['median_evolved'], ms=t, vs_fom=(tf / t if (t and tf) else None), mem=mem, admissible=adm, act=act))
    for K in (8, 16, 32):
        nm = f'kim_final_K{K}'
        if nm in A: row(nm, f'Kim NM-LSPG K={K}')
        if nm + '_hr' in A: row(nm + '_hr', f'Kim NM-LSPG-HR K={K} (exploratory)')
        nm = f'sig_K{K}_sel'
        if nm in A: row(nm, f'Kim NM-LSPG K={K}')
        if nm + '_hr' in A: row(nm + '_hr', f'Kim NM-LSPG-HR K={K} (exploratory)')
        for dm in [k for k in A if k.startswith(f'kim_final_K{K}_fit') and not k.endswith('_hr')]:
            row(dm, f"Kim NM-LSPG K={K}, data-matched ({A[dm]['variant']['fit_traj']} traj.)")
        pods = [k for k in (f'pod_lspg_zero_K{K}', f'pod_lspg_ic_K{K}') if k in A]
        if pods: row(min(pods, key=lambda k: A[k]['worst_evolved']), f'POD-LSPG K={K} (better reference)')
    for nm, lab in (('ours_q0', 'ours fast (q=0), K=16'), ('ours_q256', 'ours accurate (q=256), K=16'),
                    ('fom_nt1e4_dt005', 'FOM loose (1e-4)'), ('fom_fft_tight', 'FOM named / reference')):
        if nm in A: row(nm, lab)
if HEAD:
    out += ['## 2. Headline: worst evolved same-grid error, query time and memory per method, mesh and latent dimension', '',
            'Validation cohort (32 held-out cases) for every row. Times: median GPU query, supplied initial field on GPU to six dense fields on GPU, all arms '
            'interleaved in one allocation per mesh. "× FOM" = named-FOM time / arm time (> 1 means faster than the full-order solve). '
            'Kim rows marked INADMISSIBLE used an activation that failed the reproduction gate.', '',
            '| mesh | method | solved unknowns | worst evolved | median evolved | query ms | × FOM | compiled-query MB | admissible |', '|---|---|---|---|---|---|---|---|---|']
    for h in HEAD:
        num = lambda x, fmt: '—' if x is None else format(x, fmt)
        out.append(f"| {h['mesh']}² | {h['method']}{' [' + h['act'] + ']' if h['act'] else ''} | {h['solved'] or '—'} | {pc(h['worst'])} | {pc(h['median'])} | "
                   f"{num(h['ms'], '.1f')} | {num(h['vs_fom'], '.2f')} | {num(None if h['mem'] is None else h['mem'] / 1e6, '.0f')} | {'yes' if h['admissible'] else 'NO'} |")
    out.append('')
    # ------------------------------------------------------------ fitting limits and tuning effort
    out += ['## 3. Where each Kim configuration stops fitting or training', '',
            '| mesh | arm | outcome |', '|---|---|---|']
    for f in fams:
        s = f['s']
        for d in s['dropped']:
            det = f"needs {d['need_gb']:.0f} GB for weights + gradient + Adam state > device {d['device_limit_gb']:.0f} GB (M1 = {d['M1']}); not attempted" if 'need_gb' in d else d['reason']
            out.append(f"| {s['intervals']}² | `{d['name']}` | {d['reason']}: {det} |")
        for nm, t in s['training'].items():
            out.append(f"| {s['intervals']}² | `{nm}` | trained {t['epochs']} epochs in {t['seconds']:.0f} s, stop = {t['stop_reason']}, M1 = {t['M1']}, "
                       f"{t.get('fit_trajectories', 112)} fit trajectories, best validation-snapshot MSE {t['best_val']:.2e} |")
    out += ['', '## 4. Tuning effort given to the Kim baseline', '']
    for f in fams:
        s = f['s']; sel = s.get('selection', {})
        tot = sum(t['seconds'] for t in s['training'].values())
        out.append(f"- **{s['intervals']}²** (job {s['job_id']}): {len(sel.get('candidates', {}))} sweep candidates scored on the tuning subset, "
                   f"{len(s['training'])} autoencoders trained, {tot / 3600:.1f} GPU-hours of training; selected `{sel.get('selected', '—')}`. "
                   + 'Candidates (tune worst evolved): ' + ', '.join(f'`{k}` {pc(v)}' for k, v in sel.get('candidates', {}).items()))
    out.append('')
for f in fams:
    s = f['s']; L = s['intervals']
    out += [f"## Shared Burgers family, {L}² intervals (n = {s['n']}), job {s['job_id']}, {s['gpu_uuid']}", '',
            f"Kim rows: **{'code matches a passed gate; a Kim row is admissible only if its activation is the gate activation (' + ', '.join(sorted(passed_acts)) + ')' if f['admissible'] else 'NOT admissible (no passed gate for this code)'}**. "
            "Other Kim hyper-parameters are tuned on the family (DESIGN s.3) and printed in the `act` column and variant. "
            f"Cohort = 32 held-out validation cases unless the column says tune "
            f"(16 training-side cases used for every choice). NumPy audit: {'all audited rows agree' if f['audit'] and f['audit']['all_agree'] else 'MISSING or disagreeing'}.", '',
            '| arm | family | solved unknowns | worst evolved (validation) | median evolved | worst evolved (tune) | autoencode-only worst | GN cap hits | query GPU ms (median) | compiled-query memory MB | training s (epochs, stop) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, a in s['arms'].items():
        if a.get('cohort') == 'tune' and a['family'] == 'kim_nm_lspg_hr':
            continue
        kim = a['family'].startswith('kim')
        act = a.get('activation', 'swish') if kim else None
        adm = (not kim) or (f['admissible'] and act in passed_acts)
        tag = '' if not kim else (f' [{act}]' + ('' if adm else ' INADMISSIBLE'))
        t = a.get('timing', {}); m = a.get('memory_analysis', {}); tr = s['training'].get(name, {})
        out.append(f"| `{name}`{tag} | {a['family']}{' (exploratory HR)' if a.get('exploratory_hr_gate_failed') else ''} | {a.get('solved_dimension', '—')} | {pc(a['worst_evolved'])} | {pc(a['median_evolved'])} | "
                   f"{pc(a['tune']['worst_evolved']) if 'tune' in a else '—'} | {pc(a['autoencode']['worst_evolved']) if 'autoencode' in a else '—'} | {a.get('gn_cap_hits', '—')} | "
                   f"{t.get('gpu_ms_median', float('nan')):.1f} | {m.get('total', 0) / 1e6:.0f} | {tr.get('seconds', 0):.0f} ({tr.get('epochs', '—')}, {tr.get('stop_reason', '—')}) |" if t else
                   f"| `{name}`{tag} | {a['family']} | {a.get('solved_dimension', '—')} | {pc(a['worst_evolved'])} | {pc(a['median_evolved'])} | {pc(a['tune']['worst_evolved']) if 'tune' in a else '—'} | "
                   f"{pc(a['autoencode']['worst_evolved']) if 'autoencode' in a else '—'} | {a.get('gn_cap_hits', '—')} | not timed | — | {tr.get('seconds', 0):.0f} ({tr.get('epochs', '—')}, {tr.get('stop_reason', '—')}) |")
        rows_json.append(dict(mesh=L, arm=name, family=a['family'], activation=act, admissible=adm, cohort=a.get('cohort'), K=a.get('K'), worst_evolved=a['worst_evolved'], median_evolved=a['median_evolved'],
                              gpu_ms=t.get('gpu_ms_median'), memory_bytes=m.get('total')))
    if s.get('selection'):
        out += ['', f"Selection rule: {s['selection']['rule']}. Selected: `{s['selection'].get('selected')}`.", '']
    if s['dropped']:
        out += ['', 'Dropped arms: ' + '; '.join(f"`{d['name']}` ({d['reason']})" for d in s['dropped']), '']
    out.append('')

out += ['## Glossary', '',
        '- **NM-LSPG**: nonlinear-manifold least-squares Petrov–Galerkin; each backward-Euler step minimises the full discrete residual over the latent vector. **HR**: hyper-reduction (residual evaluated at sampled rows only).',
        '- **worst evolved**: largest, over cases and the five output times after t = 0, of the field error against the same-grid full-order solve, divided by the norm of the initial field.',
        '- **autoencode-only**: encode then decode the true states; not a lower bound on the manifold error.',
        '- **solved unknowns**: number of unknowns in the online nonlinear solve (K, or K+q for the project ROM with q corrections).',
        '- **tune**: 16 cases carved from the training split, used for every choice; **validation**: 32 held-out cases, never used to choose.',
        '- **GN cap hits**: time steps whose Gauss–Newton solve hit the 20-iteration cap. **compiled-query memory**: XLA memory analysis (arguments + outputs + temporaries) of the jitted query.',
        '- **POD-LSPG zero / ic**: linear basis with zero reference or with the initial field as reference. **ours_q0 / ours_q256**: frozen project checkpoint without / with 256 corrections.',
        '- **FOM**: full-order model on the same grid; `fom_fft_tight` is the reference itself (error 0 by construction) and is the named FOM for "× FOM"; `fom_nt1e4_dt005` is the same solver with loose tolerances.',
        '- **× FOM**: named-FOM median query time divided by the arm\'s median query time, same allocation; below 1 the reduced model is slower than solving the full problem.',
        '- **median evolved**: median over the 32 cases of each case\'s worst evolved-time error.',
        '- **data-matched**: the Kim autoencoder trained on 576 trajectories (the count the project bank was trained on) instead of 112.',
        '- **admissible**: a Kim row counts as the validated method only if it ran the code and activation that passed the reproduction gate.',
        '- **precheck**: before training, weights + gradient + two Adam moments of the dense encoder are compared with device memory; if larger, the arm is recorded as not fitting and not attempted.',
        '- **epochs / wall budget**: training passes over the fit snapshots; the wall budget is a per-arm time limit added by this lane (the paper allows up to 10 000 epochs).', '']
(LANE / 'reports/2026-09-21-nmrom-baselines.md').write_text('\n'.join(out))
(LANE / 'summary.json').write_text(json.dumps(dict(gate_passed=bool(passed), gates=gates, family_runs=[dict(path=f['path'], sha256=f['sha256'], admissible=f['admissible']) for f in fams],
                                                   rows=rows_json, headline=HEAD, generator_sha256=sha(__file__)), indent=1) + '\n')
print('gate_passed', bool(passed), 'gates', len(gates), 'family runs', len(fams))
