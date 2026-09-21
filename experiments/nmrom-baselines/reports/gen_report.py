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

fams = []
for p in sorted(LANE.glob('runs/fam*/output/summary.json')):
    s = json.loads(p.read_text())
    a = json.loads((p.parents[1] / 'audit.json').read_text()) if (p.parents[1] / 'audit.json').exists() else None
    gb = s['gate_binding']
    ksha = gb.get('kimae_sha256') or next((g['kimae'] for g in gates), None)
    admissible = any(g['kimae'] == ksha and g['lspg'] == (gb.get('lspg_sha256') or g['lspg']) for g in passed)
    fams.append(dict(path=str(p.relative_to(LANE)), sha256=sha(p), s=s, audit=a, admissible=admissible))

out = ['# Kim et al. masked-autoencoder NM-LSPG versus the project NM-ROM on the shared Burgers 2D family', '']
verdict = 'PASSED' if passed else 'FAILED'
out += [f'Reproduction gate: **{verdict}**. ' + ('Kim-baseline rows below are admissible where marked.' if passed else
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
for f in fams:
    s = f['s']; L = s['intervals']
    out += [f"## Shared Burgers family, {L}² intervals (n = {s['n']}), job {s['job_id']}, {s['gpu_uuid']}", '',
            f"Kim rows: **{'admissible' if f['admissible'] else 'NOT admissible (no passed gate for this code)'}**. Cohort = 32 held-out validation cases unless the column says tune "
            f"(16 training-side cases used for every choice). NumPy audit: {'all audited rows agree' if f['audit'] and f['audit']['all_agree'] else 'MISSING or disagreeing'}.", '',
            '| arm | family | solved unknowns | worst evolved (validation) | median evolved | worst evolved (tune) | autoencode-only worst | GN cap hits | query GPU ms (median) | compiled-query memory MB | training s (epochs, stop) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, a in s['arms'].items():
        if a.get('cohort') == 'tune' and a['family'] == 'kim_nm_lspg_hr':
            continue
        t = a.get('timing', {}); m = a.get('memory_analysis', {}); tr = s['training'].get(name, {})
        out.append(f"| `{name}` | {a['family']}{' (exploratory HR)' if a.get('exploratory_hr_gate_failed') else ''} | {a.get('solved_dimension', '—')} | {pc(a['worst_evolved'])} | {pc(a['median_evolved'])} | "
                   f"{pc(a['tune']['worst_evolved']) if 'tune' in a else '—'} | {pc(a['autoencode']['worst_evolved']) if 'autoencode' in a else '—'} | {a.get('gn_cap_hits', '—')} | "
                   f"{t.get('gpu_ms_median', float('nan')):.1f} | {m.get('total', 0) / 1e6:.0f} | {tr.get('seconds', 0):.0f} ({tr.get('epochs', '—')}, {tr.get('stop_reason', '—')}) |" if t else
                   f"| `{name}` | {a['family']} | {a.get('solved_dimension', '—')} | {pc(a['worst_evolved'])} | {pc(a['median_evolved'])} | {pc(a['tune']['worst_evolved']) if 'tune' in a else '—'} | "
                   f"{pc(a['autoencode']['worst_evolved']) if 'autoencode' in a else '—'} | {a.get('gn_cap_hits', '—')} | not timed | — | {tr.get('seconds', 0):.0f} ({tr.get('epochs', '—')}, {tr.get('stop_reason', '—')}) |")
        rows_json.append(dict(mesh=L, arm=name, family=a['family'], admissible=(f['admissible'] or not a['family'].startswith('kim')), worst_evolved=a['worst_evolved'],
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
        '- **FOM**: full-order model on the same grid; `fom_fft_tight` is the reference itself (error 0 by construction).', '']
(LANE / 'reports/2026-09-21-nmrom-baselines.md').write_text('\n'.join(out))
(LANE / 'summary.json').write_text(json.dumps(dict(gate_passed=bool(passed), gates=gates, family_runs=[dict(path=f['path'], sha256=f['sha256'], admissible=f['admissible']) for f in fams],
                                                   rows=rows_json, generator_sha256=sha(__file__)), indent=1) + '\n')
print('gate_passed', bool(passed), 'gates', len(gates), 'family runs', len(fams))
