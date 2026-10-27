"""Snapshot lane ledgers/exports and generate a durable campaign handoff (no scheduling)."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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
            snapshots[(name, kind)] = read(dst)
        ledger = snapshots[(name, 'ledger')]
        for row in ledger.get('entries', ledger.get('jobs', [])):
            entries.append(dict(lane=name, **row))
    active = [r for r in entries if str(r.get('status', r.get('state', 'unknown'))).upper()
              in ('RUNNING', 'PENDING', 'SUBMITTED', 'CONFIGURING', 'COMPLETING', 'QUEUED')]
    if args.final and active:
        raise RuntimeError(f'cannot close with active ledger entries: {[r.get("job_id") for r in active]}')
    if args.final:
        namespaces = ['/cluster/tufts/paralab/tawal01/b3d-solver-20261026/',
                      '/cluster/tufts/paralab/tawal01/b3d-pod-20261026/']
        def ours(workdir):
            return any(workdir.startswith(n) for n in namespaces)
        queue = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            'tufts-login', 'squeue -h -u tawal01 -o "%i|%Z|%T"'], text=True, timeout=25)
        queued = [line for line in queue.splitlines() if len(line.split('|')) >= 3 and ours(line.split('|')[1].strip())]
        if queued:
            raise RuntimeError(f'cannot close with live campaign queue entries: {queued}')
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
        write(HERE / 'data/FINAL-QUEUE-AUDIT.json', dict(live_campaign_jobs=[], accounting=actual,
              note='Foreground bounded SSH/Slurm checks; no background watcher.'))
    spent = sum(float(r.get('gpu_hours', r.get('spent_gpu_hours', 0))) for r in entries)
    cap = read(EXPERIMENT / 'JOB-LEDGER.json')['gpu_hour_cap']
    assert spent <= cap
    ledger = dict(campaign='b3d-refine-20261026', state='bounded campaign closed' if args.final else 'checkpoint, work ongoing',
                  gpu_hour_cap=cap, completed_gpu_hours=spent, active_ledger_entries=len(active),
                  accounting_note='Only completed elapsed-time entries included; active jobs are not zero-cost.',
                  jobs=entries, lane_snapshot_sources=pins,
                  gates='Independent gate details live in lane audit records; publication audit remains separate.')
    write(EXPERIMENT / 'JOB-LEDGER.json', ledger)
    write(HERE / 'data/CHECKPOINT-SOURCES.json', pins)
    write(HERE / 'data/EVIDENCE-INDEX.json', dict(
        historical_inventory='inventory.json.gz', historical_cases='cases.jsonl.gz',
        historical_source_pins='SOURCES.json', historical_timing_rescore='timing-rescore.json',
        lane_exports={name: f'lane-{name}-export.json' for name in ['solver', 'pod']},
        source_snapshots='CHECKPOINT-SOURCES.json',
        full_validation_frontier='frozen-frontier.json' if (HERE / 'data/frozen-frontier.json').is_file() else None,
        note='Historical qualifications retain their date/scope; new qualification joins are in lane exports.'))
    rows = []
    for r in entries:
        state = r.get('status', r.get('state', 'unknown'))
        cost = r.get('gpu_hours', r.get('spent_gpu_hours'))
        rows.append(f"| {r['lane']} | {r.get('attempt')} | {r.get('job_id')} | {state} | "
                    f"{'active/unaccounted' if cost is None else f'{cost:.6g}'} |")
    retention = snapshots[('pod', 'export')].get('large_fit_input_retention')
    retention_text = ''
    if retention:
        assert Path(retention['path']).stat().st_size == retention['bytes']
        retention_text = (f"\n\nThe actual fitting matrix ({retention['bytes']/1e9:.6g} GB) is retained at "
            f"`{retention['path']}` with SHA256 `{retention['sha256']}`. It is outside Git; "
            f"the durable retention manifest is [{Path(retention['supporting_manifest_path']).name}]"
            f"({retention['supporting_manifest_path']}). Preserve or move this data before any "
            'worktree archival. Candidate nodes and the target are listed by that manifest.\n')
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

Inspect each ledger's exact directory, Slurm state and existing outputs before any restart.
Do not duplicate a submitted job. Use direct staging to paralab and checksummed collection.
The POD final-print failure recovered complete numerical output; it did not warrant a rerun.
The quadrature cohort-identity preflight stop is a provenance failure until resolved, not a
measured quadrature failure. Keep every failed attempt in the ledger.

## Completed and accepted evidence

The frozen historical inventory and exact timing-output re-scoring passed independent review
for their declared scope. Selected-case solver and POD diagnostics have separate independent
results gates. Their approved scope does not promote them into a whole-validation frontier.
See the generated report and the snapshotted lane exports for numerical details and current
whole-validation qualification. Every scientific number is generated from saved JSON inputs.

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

## Deferred or still open unless closed in the final exports

Missing quadrature successor evidence; new-policy own-state quadrature and complete nonlinear
qualification; time-step separation; complete matched solver/POD/FOM accuracy–cost comparisons;
fixed-mode mesh and output phase costs; larger-support training fit and its independent
admission. The report must label every unresolved part instead of filling it with an estimate.
Fresh 3D extrapolation-regime cohorts, bank retraining and sealed replication are not implied.

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
    print(ledger['state'], spent, 'GPU-h;', len(active), 'active ledger entries')


if __name__ == '__main__':
    main()
