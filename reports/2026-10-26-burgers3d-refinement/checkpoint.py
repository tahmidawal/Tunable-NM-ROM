"""Snapshot lane ledgers/exports and generate a durable campaign handoff (no scheduling)."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from inventory import ROOT, read, sha, write

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1] / 'experiments/b3d-refine'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true')
    args = ap.parse_args()
    snapshots, entries, pins = {}, [], {}
    for name, ledger_rel, export_rel in [('solver', 'JOB-LEDGER.json', 'EXPORT.json'),
                                          ('pod', 'runs/JOB-LEDGER.json', 'runs/EXPORT.json')]:
        lane = ROOT / f'worktrees/2026-10-26-b3d-{name}/experiments/b3d-{name}'
        for kind, relative in [('ledger', ledger_rel), ('export', export_rel)]:
            src = lane / relative
            dst = HERE / 'data' / f'lane-{name}-{kind}.json'
            shutil.copy2(src, dst)
            pins[str(src)] = dict(sha256=sha(src), snapshot=str(dst), snapshot_sha256=sha(dst))
            assert pins[str(src)]['sha256'] == pins[str(src)]['snapshot_sha256'], 'export changed during snapshot; retry checkpoint'
            snapshots[(name, kind)] = read(dst)
        ledger = snapshots[(name, 'ledger')]
        for row in ledger.get('entries', ledger.get('jobs', [])):
            entries.append(dict(lane=name, **row))
    active = [r for r in entries if str(r.get('status', r.get('state', 'unknown'))).upper()
              in ('RUNNING', 'PENDING', 'SUBMITTED', 'CONFIGURING', 'COMPLETING', 'QUEUED')]
    namespaces = ['/cluster/tufts/paralab/tawal01/b3d-solver-20261026/',
                  '/cluster/tufts/paralab/tawal01/b3d-pod-20261026/']
    def ours(workdir):
        return any(workdir.startswith(n) for n in namespaces)
    queue = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        'tufts-login', 'squeue -h -u tawal01 -o "%i|%Z|%T|%j"'], text=True, timeout=25)
    live, account_queue = [], []
    for line in queue.splitlines():
        f = [s.strip() for s in line.split('|')]
        if len(f) >= 4:
            record = dict(job_id=int(f[0]) if f[0].isdigit() else f[0], workdir=f[1], state=f[2], name=f[3])
            account_queue.append(record)
            if ours(f[1]):
                live.append(record)
    known_ids = {int(r['job_id']) for r in entries if r.get('job_id') is not None}
    missing_live = [r['job_id'] for r in live if r['job_id'] not in known_ids]
    queue_audit = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), live_campaign_jobs=live,
                       account_queue_snapshot=account_queue,
                       scope='Only the two campaign namespaces govern closure; unrelated account jobs remain visible and untouched.',
                       live_job_ids_missing_from_lane_ledgers=missing_live)
    write(HERE / 'data/LIVE-QUEUE-CHECKPOINT.json', queue_audit)
    if args.final and active:
        raise RuntimeError(f'cannot close with active ledger entries: {[r.get("job_id") for r in active]}')
    if args.final:
        if live:
            raise RuntimeError(f'cannot close with live campaign queue entries: {live}')
        accounting = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            'tufts-login', 'sacct -X -n -P -S 2026-10-26 -u tawal01 --format=JobIDRaw,State,ElapsedRaw,WorkDir%200'],
            text=True, timeout=25)
        actual = []
        for line in accounting.splitlines():
            fields = line.split('|')
            if len(fields) >= 4 and fields[0].strip().isdigit() and ours(fields[3].strip()):
                actual.append(dict(job_id=int(fields[0]), state=fields[1].strip(), elapsed_seconds=int(fields[2]),
                                   workdir=fields[3].strip()))
        if not actual:
            raise RuntimeError('no campaign accounting records returned; cannot verify final ledger')
        expected = {int(r['job_id']): r for r in entries if r.get('job_id') is not None}
        assert {r['job_id'] for r in actual} == set(expected), 'campaign ledger/accounting inventory differs'
        for row in actual:
            e = expected[row['job_id']]
            assert int(e.get('elapsed_seconds', -1)) == row['elapsed_seconds'], ('stale elapsed-time ledger', row)
        write(HERE / 'data/FINAL-QUEUE-AUDIT.json', dict(**queue_audit, accounting=actual,
              note='Foreground bounded SSH/Slurm checks; no background watcher.'))
    active_ids = {int(r['job_id']) for r in active}
    spent = sum(float(r.get('gpu_hours', r.get('spent_gpu_hours')) or 0) for r in entries
                if int(r['job_id']) not in active_ids)
    reserved = sum(float(r.get('reserved_max_gpu_hours') or 0) for r in active)
    campaign_ledger = HERE / 'data/campaign-ledger.json'
    cap_source = EXPERIMENT / 'JOB-LEDGER.json' if (EXPERIMENT / 'JOB-LEDGER.json').is_file() else campaign_ledger
    cap = read(cap_source)['gpu_hour_cap']
    assert spent <= cap
    assert spent + reserved <= cap, 'completed plus live hard reservations exceed campaign cap'
    ledger = dict(campaign='b3d-refine-20261026', state='bounded campaign closed' if args.final else 'checkpoint, work ongoing',
                  gpu_hour_cap=cap, completed_gpu_hours=spent, active_ledger_entries=len(active),
                  live_reserved_max_gpu_hours=reserved,
                  live_campaign_jobs=live, live_job_ids_missing_from_lane_ledgers=missing_live,
                  accounting_note='Only completed elapsed-time entries included; active jobs are not zero-cost.',
                  jobs=entries, lane_snapshot_sources=pins,
                  gates='Independent gate details live in lane audit records; publication audit remains separate.')
    write(campaign_ledger, ledger)
    if (EXPERIMENT / 'JOB-LEDGER.json').is_file():
        write(EXPERIMENT / 'JOB-LEDGER.json', ledger)
    for source, name in [(EXPERIMENT / 'DESIGN.md', 'DESIGN.md'),
                         (EXPERIMENT / 'audits/REVIEW-GATES.json', 'AUDIT-GATES.json')]:
        if source.is_file():
            shutil.copy2(source, HERE / name)
    audit_index = {}
    for name in ['solver', 'pod']:
        lane = ROOT / f'worktrees/2026-10-26-b3d-{name}/experiments/b3d-{name}'
        files = [p for p in (lane / 'audits').iterdir() if p.is_file() and p.suffix in ('.md', '.json')]
        files += list(lane.glob('*.md'))
        files += [p for p in [lane / 'AUDIT-STATUS.json', lane / 'FRONTIER-SCHEMA.json', lane / 'MESH-RESULT-SCHEMA.json'] if p.is_file()]
        destination = HERE / 'audits' / name
        destination.mkdir(parents=True, exist_ok=True)
        for source in files:
            target = destination / source.name
            shutil.copy2(source, target)
            assert sha(source) == sha(target), 'audit record changed during snapshot; retry'
            audit_index[str(target.relative_to(HERE))] = dict(source=str(source), sha256=sha(target))
        for source in sorted((lane / 'configs').glob('*.json')):
            target = destination / 'configs' / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            assert sha(source) == sha(target)
            audit_index[str(target.relative_to(HERE))] = dict(source=str(source), sha256=sha(target))
    write(HERE / 'data/AUDIT-SOURCES.json', audit_index)
    write(HERE / 'data/CHECKPOINT-SOURCES.json', pins)
    write(HERE / 'data/EVIDENCE-INDEX.json', dict(
        historical_inventory='inventory.json.gz', historical_cases='cases.jsonl.gz',
        historical_source_pins='SOURCES.json', historical_timing_rescore='timing-rescore.json',
        lane_exports={name: f'lane-{name}-export.json' for name in ['solver', 'pod']},
        source_snapshots='CHECKPOINT-SOURCES.json',
        full_validation_frontier=next((n for n in ['frozen-frontier.json.gz', 'frozen-frontier.json'] if (HERE / 'data' / n).is_file()), None),
        fixed_mode_mesh=next((n for n in ['frozen-mesh.json.gz', 'frozen-mesh.json'] if (HERE / 'data' / n).is_file()), None),
        solver_array_retention=[f"solver-retention-{r['attempt']}.json" for r in snapshots[('solver', 'export')].get('retention_manifests', [])],
        pod_fit_retention='pod-fit32-retention.json' if snapshots[('pod', 'export')].get('large_fit_input_retention') else None,
        note='Historical qualifications retain their date/scope; new qualification joins are in lane exports.'))
    rows = []
    for r in entries:
        state = r.get('status', r.get('state', 'unknown'))
        cost = r.get('gpu_hours', r.get('spent_gpu_hours'))
        cost_text = 'active/unaccounted' if cost is None or int(r['job_id']) in active_ids else f'{cost:.6g}'
        rows.append(f"| {r['lane']} | {r.get('attempt')} | {r.get('job_id')} | {state} | "
                    f"{cost_text} |")
    retention = snapshots[('pod', 'export')].get('large_fit_input_retention')
    solver, pod = snapshots[('solver', 'export')], snapshots[('pod', 'export')]
    if pod.get('a3_replay'):
        a3 = pod['a3_replay']
        source = Path(a3['summary_path'])
        target = HERE / 'data/pod-improved-support-summary.json'
        assert sha(source) == a3['summary_sha256']
        shutil.copy2(source, target)
        assert sha(target) == a3['summary_sha256']
    completed = ['Historical inventory, source hashes, exact saved timing-output re-scoring and generated figures.']
    failed = ['Historical quadrature/nonlinear failures and square-system controls remain in the inventory.']
    for row in entries:
        if row['lane'] == 'solver' and str(row.get('attempt', '')).startswith('mesh') and str(row.get('state', row.get('status', ''))).upper() == 'FAILED':
            failed.append(f"Mesh attempt {row['attempt']}, job {row['job_id']}: {row.get('failure', 'failed; inspect the retained log')}. "
                          'Its allocation remains in the resource total; an incomplete checkpoint is not an admitted mesh dataset.')
    pending = []
    if solver.get('audits', {}).get('jobs', {}).get('qcheck801', {}).get('results', '').startswith('READY'):
        completed.append('Missing original finer-rule witness joined to inherited complete-population evidence; legacy gl56 qualification closes within its original scope.')
    if solver.get('time_selection'):
        completed.append('Audited time-step diagnostic: median improves while worst worsens; coarse/fine endpoint choice is post-diagnostic and the valid intermediate is retained.')
    if pod.get('no_hr_full64'):
        completed.append('Complete-validation unsampled POD fallback with independent reference/array audit; its separate allocation supplies no cross-job speed ratio.')
    if pod.get('failed_support_extension'):
        f = pod['failed_support_extension']
        failed.append(f"Sole larger-support constructor: {f['construction_failure_reason']}, terminal support {f['terminal_support']}/{f['support_target']}. No admitted support or rollout; no fit retry.")
    frontier_file = next((HERE / 'data' / n for n in ['frozen-frontier.json.gz', 'frozen-frontier.json'] if (HERE / 'data' / n).is_file()), None)
    if frontier_file:
        front = read(frontier_file)
        assert front['execution']['complete'] is True
        completed.append('Frozen same-allocation complete-validation frontier. Use its explicit per-subject qualification and timing-coverage masks; not every measured subject is necessarily admitted.')
    else:
        pending.append('Same-allocation complete-validation endpoint comparison, own new-policy qualification and paired accuracy-matched full-solver/POD costs.')
    if not any((HERE / 'data' / n).is_file() for n in ['frozen-mesh.json.gz', 'frozen-mesh.json']):
        pending.append('Bounded fixed-mode mesh/output costs, explicit operator interventions and same-reference projection floor.')
    if pod.get('a3_replay'):
        a = pod['a3_replay']
        failed.append('The existing k16 support at the selected improved policy passes both own-objective Q1 checks, but fails common field agreement and stricter whole-objective gates. No complete-validation HR release; scalar reference-error changes do not demonstrate a physical-accuracy loss.')
        pending.append('Qualified hyper-reduced POD remains unestablished within the tested supports/policies. No further fit or policy ladder is released by the completed campaign.')
    else:
        pending.append('Qualified hyper-reduced POD depends on the separately audited improved-policy k16 control and complete-validation admission; the rejected new constructor supplies no such evidence.')
    pending.append('Sealed/out-of-distribution replication, bank retraining, further support fitting and branch merges are outside the released campaign.')
    bullets = lambda rows: '\n'.join('- ' + row for row in rows)
    retention_text = ''
    if retention:
        assert Path(retention['path']).stat().st_size == retention['bytes']
        source_manifest = Path(retention['supporting_manifest_path'])
        portable_manifest = HERE / 'data/pod-fit32-retention.json'
        shutil.copy2(source_manifest, portable_manifest)
        assert sha(source_manifest) == sha(portable_manifest)
        retention_text = (f"\n\nThe actual fitting matrix ({retention['bytes']/1e9:.6g} GB) is retained at "
            f"`{retention['path']}` with SHA256 `{retention['sha256']}`. It is outside Git; "
            f"the [portable retention manifest](data/{portable_manifest.name}) has SHA256 `{sha(portable_manifest)}` "
            f"and its original is `{source_manifest}`. Preserve or move this data before any "
            'worktree archival. Candidate nodes and the target are listed by that manifest.\n')
    for record in solver.get('retention_manifests', []):
        source = ROOT / 'worktrees/2026-10-26-b3d-solver/experiments/b3d-solver' / record['path']
        assert sha(source) == record['sha256']
        target = HERE / 'data' / f"solver-retention-{record['attempt']}.json"
        shutil.copy2(source, target)
        assert sha(target) == record['sha256']
        bulk_bytes = sum(r['bytes'] for r in record['bulk_arrays'])
        retention_text += (f"\n\nSolver attempt `{record['attempt']}` / job {record['job_id']} retains "
            f"{len(record['bulk_arrays'])} bulk arrays ({bulk_bytes/1e9:.6g} GB) outside Git. "
            f"The [portable retention manifest](data/{target.name}) has SHA256 `{record['sha256']}`; "
            f"its original is `{source}`. The verified collection archive is "
            f"`{record['archive']['path']}` with SHA256 `{record['archive']['sha256']}`. "
            f"{record['archival_precondition']}\n")
    text = f'''# Burgers 3D refinement handoff

Status: {ledger['state']}. Completed allocations total {spent:.6g} GPU-hours against the
predeclared campaign cap {cap}; live jobs are listed explicitly and are not counted as zero.

## Work locations

- Coordinator/report: `worktrees/2026-10-26-b3d-refine`, branch `exp/2026-10-26-b3d-refine`.
- Solver/quadrature: `worktrees/2026-10-26-b3d-solver`, branch `exp/2026-10-26-b3d-solver`.
- POD comparator: `worktrees/2026-10-26-b3d-pod`, branch `exp/2026-10-26-b3d-pod`.
- Canonical log: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md`.
- New work must retain the consolidated fork recorded in CLAUDE.md, not the frozen main baseline.

## Exact job checkpoint

| Lane | Attempt | Job | State | Completed GPU-hours |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

Foreground queue check at `{queue_audit['checked_at_utc']}`: `{json.dumps(live)}`.
Live job ids absent from the source lane ledgers: `{missing_live}`. Any such lag is explicit;
active allocation costs are still unaccounted rather than zero.

Inspect each ledger's exact directory, Slurm state and existing outputs before any restart.
Do not duplicate a submitted job. Use direct staging to paralab and checksummed collection.
The POD final-print failure recovered complete numerical output; it did not warrant a rerun.
The quadrature cohort-identity preflight stop is a provenance failure until resolved, not a
measured quadrature failure. Keep every failed attempt in the ledger.

## Completed and accepted evidence

{bullets(completed)}

Selected-case diagnostics retain their separate scope and independent gates. See the
generated report and frozen exports for exact cases, current admission and provenance.
Scientific numbers are generated from saved JSON inputs; publication review is a separate gate.

## Failed experiments and controls

{bullets(failed)}

## Failed or retracted interpretations

- A configuration-map count does not cover later round-only qualified candidates.
- A reused-QN fresh restart does not restore the original cached nonlinear solver state.
- Pooled stationarity alone does not guarantee agreement with a tight full trajectory.
- Trust clipping alone is not established as the cause of the replayed convergence failure.
- Certificate repair alone is not an accuracy improvement.
- Old POD observed domination does not supply the newly requested HR convergence gate.
- Neither the new frozen-support failures nor an unsampled POD comparison establish a
  general advantage over tuned qualified POD.
- Historical mesh-free deployment curves are not fixed-mode full-field output curves.
- The unstaged worst-only time-endpoint rejection rule was abandoned, not called predeclared.
- The inherited bank uses analytic advection with discrete diffusion, not FOM-exact upwind.
  Solver/quadrature certification does not prove identical discrete equations or spatial consistency.

## Deferred, conditional and still-open work

{bullets(pending)}

The report must label every unresolved part rather than fill it with an estimate.

## Preservation and next-session rules

All original Research Update slide bytes and their order are preserved; generated refinement
slides are appended and original PDF hashes/backups are recorded. Read the latest canonical
saved user edits again before modifying presentation sources. Keep code-only mirrors current;
never push archive-bearing experimental branches directly. Do not merge any lane without the
user's decision. No unique data may be deleted; remote job directories are removed only after
verified complete collection. No unsupported background watcher is relied on.
{retention_text}

## Glossary

- Lane: an isolated worktree and cluster namespace for one scientific question.
- GPU-hour: allocated GPU count multiplied by elapsed hours, including failed attempts.
- QN: nonlinear solve that reuses its small matrix; a fresh restart resets that cache.
- Q1/Q2: nonlinear-solver and quadrature qualification on the declared case population.
- POD/HR/FOM: a standard reduced basis / weighted sampling of its equations / a full solver.
- Frontier: admissible settings not beaten in both cost and the chosen error metric.
- Diagnostic: explains selected cases; it is not a population performance estimate.
'''
    (HERE / 'HANDOFF.md').write_text(text)
    print(ledger['state'], spent, 'completed GPU-h;', len(active), 'ledger-active;', len(live), 'live queue jobs')


if __name__ == '__main__':
    main()
