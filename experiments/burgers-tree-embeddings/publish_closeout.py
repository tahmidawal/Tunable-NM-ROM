"""Publish only this study's generated status/entry to the canonical shared lab log."""
from pathlib import Path
import datetime,hashlib,json
H=Path(__file__).resolve().parent
d=json.loads((H/'DECISION-B0.json').read_text());best=d['best_fine_projection']
archive=json.loads((H/'manifests/archive-closeout.json').read_text())
ledger=json.loads((H/'cluster/ledger.json').read_text())
assert len(archive)==len(ledger)==d['jobs'] and all(r['state']=='COMPLETED' for r in ledger)
assert d['status']=='SCOPED_NEGATIVE_STAGE1' and not d['advance_to_PDE']
owner=json.loads((H/'OWNER.json').read_text());stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
owner.update(status='B0_CLOSED_SCOPED_NEGATIVE_STAGE1',updated_utc=stamp,
    decision='No PDE advancement after two bounded unsuccessful repairs; broader architecture remains untested',
    allocated_gpu_seconds=d['allocated_gpu_seconds'],allocated_gpu_hours=d['allocated_gpu_hours'],
    completed_jobs=d['jobs'],active_jobs=0,final_test_solution_fields_opened=False,
    report='/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-tree-embedding-representation.md')
(H/'OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')
assert json.loads(claim.read_text())['owner']==owner['owner'];claim.write_text(json.dumps(owner,indent=2)+'\n')
report_link='[Generated results and plots](reports/2026-11-18-tree-embedding-representation.md)'
status=f'''**New Burgers tree-embedding study: B0 closed with scoped negative representation evidence (Codex, separately authorized 2026-11-18).** Fresh references, matched-data global controls, direct and nonlinear local models, and two registered repairs completed in the separately owned lane `exp/2026-11-18-burgers-tree-embeddings`. Best tree: {100*best['worst']:.4f}% worst development/alignment projection error at {best['variables']} variables, versus the 0.5% gate. No advancement to PDE solves/adaptation or claim of faster complete solves at 1% accuracy. All {d['jobs']} owned jobs completed; {d['allocated_gpu_seconds']} allocated A100 GPU-seconds ({d['allocated_gpu_hours']:.6f} hours), within the initial eight-hour budget. Checkpoints, failed recipes, full 4096 arrays and verified raw backups are retained. Final-test fields remain unopened. {report_link}. The November 17 campaign remains PAUSED; no merge.'''
title='### 2026-11-18 — New tree-embedding B0 closeout (Codex execution owner)'
entry=f'''{title}

Completed at {stamp}. New branch/worktree `exp/2026-11-18-burgers-tree-embeddings` / `worktrees/2026-11-18-burgers-tree-embeddings`; exclusive namespace `{owner['namespace']}`. All {d['jobs']} jobs completed: {', '.join(str(r['id']) for r in ledger)}. Scheduler accounting confirms at most two simultaneous owned jobs, GPU preflight, float64/highest precision, and {d['allocated_gpu_seconds']} allocated GPU-seconds ({d['allocated_gpu_hours']:.6f} hours), with {d['allocated_cpu_hours']:.6f} allocated CPU-hours. No compute expansion. Local GPU smoke durations were not measured; four attempts were each capped at 55 seconds. One failed at CUDA initialization due to shared local GPU memory pressure; its explicitly CPU algebra rerun passed. No other process was changed.

Fresh training/development/alignment references passed empirical space/time ladders and the required independent discretization checks: maximum spectral allowance {100*d['reference_allowance_max']:.7f}%, maximum independent difference {100*d['independent_reference_difference_max']:.5f}%. Final-test fields were never generated or inspected. Global/direct-local/nonlinear-child training, uniform/prescribed allocation, parent-first residual initialization and multi-depth assembled fits were executed. Child codes are independent; zero children preserve the field and derivatives, with full independent child Jacobian rank on the representative prescribed tree.

R0 had a substantial training-code fitting gap, so its early stop was not a feature-learning plateau. Registered R1 explicitly refit codes before each assembled weight update; R2 widened only the local hidden layers. After finer 128-by-128 scoring, the best candidate was {best['recipe']} {best['method']} / {best['allocation']}, worst {100*best['worst']:.4f}% (random {100*best['random']:.4f}%, alignment {100*best['alignment']:.4f}%). All {best['failing_trajectories']} development/alignment trajectories still failed the 0.5% screen. Actual 4096-by-4096 arrays, independent NumPy rescoring, spatial error maps, component cost/memory records and failed recipes are preserved in {report_link}. These are reference-assisted projections, not PDE solutions.

Decision: stop this bounded recipe family under the two-repair rule; no fixed-tree PDE, adaptive policy, complete-runtime success or final-test claim. Global nonlinear controls use a zero-anchored Tunable-form head and are architecture-inspired controls, not faithful native unrestricted-bias reproductions. Strong NM-ROM confirmation was not reached; unattempted competitors are not counted as defeated. Finite-start fits and training limitations remain open, and this result does not rule out other feature widths, child dimensions, depths, data coverage or global heads. Any further search needs a prospectively documented amendment responding to this evidence.

Source, selected checkpoints, generated report/plots and decision are committed and backed up through the code-only lane mirror. All raw job outputs have checksum-verified local copies and owned cluster archives; only reproducible owned staging/cache copies were removed. The historical ownership claim is retained with closed status. The earlier burgers-quadtree worker, lane, namespace and pause were untouched. No merge and no 3D work.
'''
lab=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');s=lab.read_text()
marker='**New Burgers tree-embedding study:';a=s.index(marker);b=s.index('\n\n',a);s=s[:a]+status+s[b:]
if title not in s:s=s.rstrip()+'\n\n'+entry+'\n'
lab.write_text(s)
report=Path(owner['report'])
publication=dict(published_utc=stamp,canonical_lab_log=str(lab),own_status=status,own_chronology_entry=entry,
                 report=str(report),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                 decision_sha256=hashlib.sha256((H/'DECISION-B0.json').read_bytes()).hexdigest(),
                 raw_archive_manifest_sha256=hashlib.sha256((H/'manifests/archive-closeout.json').read_bytes()).hexdigest())
(H/'manifests/canonical-publication.json').write_text(json.dumps(publication,indent=2)+'\n')
print(status)
