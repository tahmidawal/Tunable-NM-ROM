"""Publish B2 pilot evidence and update only this study's canonical lab entry."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H))
from audit_fixed_pilot import inspect

names = ['b2base', 'b2space', 'b2quad', 'b2tests', 'b2time', 'b2tight']
expected = dict(L=8192, dt=.00125, quadrature_intervals=512, common_tests=22016,
                stationarity_tolerance=1e-8, update_tolerance=1e-10,
                cg_relative_tolerance=1e-6, cg_absolute_tolerance=1e-12,
                initial_relative_tolerance=1e-9, initial_absolute_tolerance=1e-13)
changes = dict(b2base={}, b2space=dict(L=16384), b2quad=dict(quadrature_intervals=1024),
               b2tests=dict(common_tests=44032), b2time=dict(dt=.000625),
               b2tight=dict(stationarity_tolerance=1e-10, update_tolerance=1e-12,
                            cg_relative_tolerance=1e-8, cg_absolute_tolerance=1e-14,
                            initial_relative_tolerance=1e-11, initial_absolute_tolerance=1e-15))
records, raw, source_hashes = [], {}, {}
for name in names:
    p = H / 'results' / name / 'output/result.json'
    if not p.exists() or not (H / 'manifests' / f'retention-{name}.json').exists():
        continue
    r = json.loads(p.read_text())
    if r['status'] != 'complete_fixed_tree_diagnostic_pilot':
        continue
    audit = inspect(name)
    setting = dict(expected, **changes[name])
    identities = (r['global_control'] == 'global_stable960' and
                  r['labels'] == ['dev000', 'dev003', 'align002'] and
                  r['time_integrator'] == 'cn' and
                  r.get('field_implementation','compact local patch slices') == 'compact local patch slices')
    setting_matches = all(r[k] == v for k, v in setting.items() if k != 'common_tests')
    setting_matches &= all(c['common_tests'] == setting['common_tests'] for c in r['cases'])
    if name != 'b2base':
        baseline = H / 'results/b2base/output/result.json'
        setting_matches &= r['previous_result_sha256'] == hashlib.sha256(baseline.read_bytes()).hexdigest()
    audit.update(registered_setting_matches=bool(identities and setting_matches))
    audit['passed'] &= audit['registered_setting_matches']
    records.append(audit)
    raw[name] = r
    source_hashes[name] = audit['source_sha256']

ledger = json.loads((H / 'cluster/ledger.json').read_text())
terminal = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
active = [r for r in ledger if (r.get('state') or 'PENDING').split()[0] not in terminal]
seconds = sum(r.get('elapsed_seconds', 0) for r in ledger)
charged = sum(r.get('elapsed_seconds', 0) if (r.get('state') or 'PENDING').split()[0] in terminal
              else r['reserved_minutes']*60 for r in ledger)
done = {r['name'] for r in records}
remaining = [n for n in names if n not in done]
failed = [r['name'] for r in records if not r['passed']]
if failed:
    decision = 'Pilot qualification failed: ' + ', '.join(failed) + '. Diagnose the saved evidence before any additional repair or advancement.'
elif remaining:
    decision = 'Pilot qualification is incomplete. Outstanding checks: ' + ', '.join(remaining) + '.'
else:
    decision = 'All six registered pilot checks pass. This admits only the three-case numerical setting and supports a separately registered all-development batch.'

rows = []
for record in records:
    for row in record['rows']:
        change = 'baseline' if row['max_field_change'] is None else f"{100*row['max_field_change']:.6f}%"
        rows.append(f"| {record['name']} | {row['case']} | {row['method']} | {100*row['max_physical_error']:.6f}% | {row['nonconverged_steps']}/{row['steps']} | {change} | {record['passed']} |")

out = H / 'reports'
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'svg.hashsalt':'burgers-tree-embeddings'})
refinements = [r for r in records if r['name'] != 'b2base']
plot = ''
if refinements:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    labels = dict(b2space='Spatial operator', b2quad='Integration', b2tests='Weak tests',
                  b2time='Time step', b2tight='Solver precision')
    for j, method in enumerate(['tree_direct24', 'global_stable960']):
        values = [max(v['max_field_change'] for v in r['rows'] if v['method'] == method)*100 for r in refinements]
        ax.bar(np.arange(len(refinements))+(j-.5)*.34, np.maximum(values, 1e-12), .34,
               label='Direct tree1376' if j == 0 else 'Global960')
    ax.axhline(.1, color='#b5413c', ls='--', label='0.1% sensitivity limit')
    ax.set(yscale='log', ylabel='Largest current-time relative field change (%)',
           xticks=np.arange(len(refinements)), xticklabels=[labels[r['name']] for r in refinements],
           title='Independent checks at the new common pilot setting')
    ax.legend(fontsize=8)
    for suffix in ['png', 'svg']:
        dest = out / f'b2-fixed-tree-qualification.{suffix}'
        fig.savefig(dest, dpi=180, bbox_inches='tight', **({'metadata':{'Date':None}} if suffix == 'svg' else {}))
        if suffix == 'svg':
            dest.write_text('\n'.join(line.rstrip() for line in dest.read_text().splitlines())+'\n')
    plt.close(fig)
    plot = '![Independent B2 numerical checks](2026-11-18-tree-embedding-representation-assets/b2-fixed-tree-qualification.png)'

stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
text = f'''# Burgers tree embeddings: common fixed-tree numerical qualification

{decision}

The direct tree has1376 adjustable coefficients and the stable global POD-plus-Fourier control has960. The common baseline is L8192, Crank–Nicolson dt0.00125, Q512 and22016 weak tests. The spatial check uses L16384, integration Q1024, tests44032, time step0.000625, and the precision check tightens all registered initial/inner/outer tolerances100 times. Each changes only its named numerical control. This new baseline is not a one-factor comparison with the older global768 pilots.

All solves use the shared initial-field callable and the original fine-grid sign-upwind weak formulation. References enter only subsequent scoring. Physical errors use the current-time reference norm; the per-case empirical reference allowance is included in each accuracy decision. Every run returns an actual4096-by4096 final device field. The table combines the largest of five evolved physical errors and the actual-grid final error, and separately audits the complete returned stopping rule. Reported changes include the initial fit as well as all evolved output times.

| Run | Case | Method | Worst physical error | Steps failing complete convergence | Largest field change | Run gate passes |
|---|---|---|---:|---:|---:|---|
{chr(10).join(rows) or '| No completed retained result yet | | | | | | |'}

{plot}

A passing small pilot is not the all44-development-trajectory gate. No deployable adaptation, method freeze, native NM-ROM confirmation, paired runtime win or final-test result is claimed. The nonlinear child16 checkpoint remains stopped after two unsuccessful online solver repairs; its representation result is retained. The original16.5% failure and the improved representation plots remain in [the representation report](2026-11-18-tree-embedding-representation.md).

Pilot clocks charge joint comparison setup to each method and cannot serve as method-specific speed ratios. Initialization, assembly, solving, rejected trials, fallback and requested output remain mandatory components of later complete-query timing. The equivalent packed field implementation has only CPU fixture evidence and is not used in this batch.

Generated {stamp}. Elapsed allocated time is{seconds} GPU-seconds ({seconds/3600:.6f}h). The reservation guard currently charges{charged} seconds, including full reservations for active jobs. Active owned jobs: {', '.join(r['name'] for r in active) or 'none'}. B2 is capped at5400 additional GPU-seconds from22744, for a cumulative ceiling28144 seconds. Source/configurations, scheduler records and verified retention manifests are in the owned lane. The old campaign remains paused; final-test fields remain unopened.
'''
filename = '2026-11-18-tree-embedding-fixed-tree.md'
(out / filename).write_text(text.replace('2026-11-18-tree-embedding-representation-assets/', ''))
main = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports')
assets = main / '2026-11-18-tree-embedding-representation-assets'
assets.mkdir(exist_ok=True)
for p in out.glob('b2-fixed-tree-qualification.*'):
    shutil.copy2(p, assets / p.name)
(main / filename).write_text(text)
for path in [out / '2026-11-18-tree-embedding-representation.md', main / '2026-11-18-tree-embedding-representation.md']:
    contents = path.read_text()
    link = f'Current fixed-tree qualification: [B2 numerical checks]({filename}).'
    if link not in contents:
        lines = contents.splitlines()
        lines[2:2] = [link, '']
        path.write_text('\n'.join(lines)+'\n')

status = f'''**New Burgers tree-embedding study: B2 fixed-tree numerical qualification.** Nonlinear tree representation0.4238% at1152 variables and direct tree0.4229% at1376 pass every development snapshot; the global960 control remains better at0.3304%. B1 used6.317778 cumulative allocated GPU-hours. Its direct pilot passes physical accuracy, weak-test enrichment and integration checks. The nonlinear PDE checkpoint is stopped after two failed optimizer repairs:123/300 stationarity failures and151/300 complete-convergence failures. {decision} Current elapsed allocation{seconds/3600:.6f}h; active owned jobs: {', '.join(r['name'] for r in active) or 'none'}. No all-development PDE admission, runtime win or final-test opening. [Representation evidence](reports/2026-11-18-tree-embedding-representation.md); [fixed-tree checks](reports/{filename}). Old campaign remains PAUSED; no merge.'''
record = dict(utc=stamp, decision=decision, source_hashes=source_hashes, records=records,
              remaining=remaining, failed=failed, elapsed_gpu_seconds=seconds,
              charged_gpu_seconds=charged, active_jobs=[r['name'] for r in active], own_status=status)
key = hashlib.sha256(json.dumps(dict(sources=source_hashes, active=record['active_jobs']), sort_keys=True).encode()).hexdigest()[:12]
(H / 'manifests' / f'B2-progress-{key}.json').write_text(json.dumps(record, indent=2)+'\n')
lab = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md')
contents = lab.read_text()
start = contents.index('**New Burgers tree-embedding study:')
end = contents.index('\n\n', start)
contents = contents[:start]+status+contents[end:]
title = f'### 2026-11-18 — Tree-embedding B2 qualification record {key}'
if title not in contents:
    contents = contents.rstrip()+f'\n\n{title}\n\n{status}\n\nSource hashes and all returned-step/physical/sensitivity decisions: `manifests/B2-progress-{key}.json` in the owned lane. B1 closeout and archive evidence remain immutable.\n'
lab.write_text(contents)
owner_path = H / 'OWNER.json'
owner = json.loads(owner_path.read_text())
owner.update(updated_utc=stamp, status='B2_FIXED_TREE_NUMERICAL_QUALIFICATION', decision=decision,
             allocated_gpu_seconds=seconds, allocated_gpu_hours=seconds/3600,
             active_jobs=len(active), terminal_jobs=len(ledger)-len(active),
             completed_jobs=sum(r.get('state') == 'COMPLETED' for r in ledger),
             batch_gpu_seconds=5400, cumulative_ceiling_gpu_seconds=28144,
             final_test_solution_fields_opened=False)
owner_path.write_text(json.dumps(owner, indent=2)+'\n')
claim = Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')
assert json.loads(claim.read_text())['owner'] == owner['owner']
claim.write_text(json.dumps(owner, indent=2)+'\n')
print(decision)
