"""Close a failed bounded parent round only after evidence and backups are complete."""
from pathlib import Path
import json,datetime
H=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
r=read(H/'results/b7parentrepair/output/result.json');named=read(H/'results/b7namedparent/output/result.json')
starts=read(H/'results/b7parentstarts/output/result.json');snf=read(H/'results/b7snff64/output/result.json')
assert r['status']=='complete_stable_parent' and not r['parent_quality_pass']
assert named['status']==starts['status']=='complete' and snf['status']=='complete'
audit=read(H/'manifests/allocation-audit-B7-FINAL.json')
archive=read(H/'manifests/archive-closeout-B7.json')+read(H/'manifests/archive-closeout-B7-FINAL.json')
assert len(archive)==len({x['name'] for x in archive})==len([x for x in audit['jobs'] if x['name'].startswith('b7')])
(H/'manifests/archive-closeout-B7-ALL.json').write_text(json.dumps(dict(sources=['archive-closeout-B7.json','archive-closeout-B7-FINAL.json'],records=archive),indent=2)+'\n')
assert audit['status']=='passed_scheduler_allocation_audit' and not audit['owned_jobs_still_queued_or_running'] and archive
exports=read(H/'manifests/B7-parent-full-resolution-backup.json');assert exports['local_and_remote_verified']
batches=read(H/'cluster/BATCHES.json');batch=next(x for x in batches['batches'] if x['id']=='B7')
used=audit['allocated_gpu_seconds']-batch['baseline_allocated_gpu_seconds'];assert used<=batch['max_additional_gpu_seconds']
text=f'''# B7 parent-first round: scoped-negative advancement decision

Worst all 484 independently scored development error: {100*r['worst_independent_error']:.9f}%. Actual 4096 × 4096 witness: {100*r['actual4096']['relative_error']:.9f}%. Worst stored-code training error: {100*r['worst_training_error']:.9f}%. Unfinished development fits: {r['unfinished_fits']}. Parent-quality gate passed: {r['parent_quality_pass']}.

The training-only pilot passed after deterministic polishing, but this did not establish broad-family accuracy. Full training and the final alternating-code/full-quadrature repair are complete. The two permitted parent/training repairs are exhausted. Stop this parent-development recipe in B7; no child training, PDE rollout, adaptation or fair complete-runtime comparison is admitted. This is not evidence that all nonlinear local decoders fail.

Closing fixed-weight starting-point diagnostic, six selected worst primary fields only:

| Snapshot | Registered fit | Diagnostic best | Converged |
|---|---:|---:|---|
'''
for row in starts['rows']:
    text+=f"|{row['index']}|{100*row['original_error']:.9f}%|{100*row['best_independent_error']:.9f}%|{bool(row['statuses'][row['best_start']][-1])}|\n"
text+=f'''
All {len(named['cases'])} named family/viscosity cases were evaluated offline, without training on them. Actual 4096 × 4096 reference, neural reconstruction and error exports cover all four families at viscosity 0.005 at time 0 and the five requested output times. They are explicitly labelled reconstructions, not PDE predictions.

Native SNF float64 GPU gradients and scalar-equivalent RHS qualification passed: {snf['passed']}. The Lux compatibility patch, missing-adapter failure and precision-port correction are retained; this is not a trained native benchmark. CROM remains unqualified with its prior training/reproduction limitations retained. No superiority or complete-solve speed claim is supported.

This batch consumed {used} allocated GPU-seconds ({used/3600:.9f} hours); cumulative {audit['allocated_gpu_seconds']} seconds ({audit['allocated_gpu_hours']:.9f} hours). At most {audit['max_pending_or_running_owned_jobs']} owned jobs were pending/running simultaneously. All attempts are terminal and backed up with verified hashes. Final tests remain sealed, old campaign paused/read-only, no branch merge or background continuation.

Evidence: manifests/allocation-audit-B7-FINAL.json, manifests/archive-closeout-B7-ALL.json, manifests/B7-parent-full-resolution-backup.json and the generated reports/2026-11-19-neural-parent-training.md.
'''
recovery=read(H/'manifests/B7-postprocess-recovery.json')
recovery.update(status='complete',archive_reverified=True,rendered_on_cluster=True,rendering_job=read(H/'results/b7render/output/result.json')['meta']['job'],full_resolution_exports_verified=True)
(H/'manifests/B7-postprocess-recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
text+='\nLocal archive/render commands terminated before completion. The interrupted archive move was recovered by checking the destination against the retained local checksums. Repeated local rendering termination coincided with low memory headroom; the exact signal sender was not identified. A prospectively bounded cluster publication job rendered the same verified fields, with its allocation included above. No scientific fits or training were rerun.\n'
(H/'configs/B7-REVIEW.md').write_text(text)
batch.update(status='CLOSED_SCOPED_NEGATIVE',actual_additional_gpu_seconds=used,cumulative_actual_gpu_seconds=audit['allocated_gpu_seconds'],review='configs/B7-REVIEW.md',accounting='manifests/allocation-audit-B7-FINAL.json',archive='manifests/archive-closeout-B7-ALL.json')
(H/'cluster/BATCHES.json').write_text(json.dumps(batches,indent=2)+'\n')
owner=read(H/'OWNER.json');owner.update(status='B7_CLOSED_SCOPED_NEGATIVE',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),active_jobs=0,
    terminal_jobs=len(audit['jobs']),completed_jobs=sum(x['terminal_state']=='COMPLETED' for x in audit['jobs']),allocated_gpu_seconds=audit['allocated_gpu_seconds'],allocated_gpu_hours=audit['allocated_gpu_hours'],
    continuation=dict(status='none',active=False),latest_neural_status='Parent gate failed after two bounded training repairs; named reconstruction and multistart diagnostics complete; no neural PDE or fair comparator timing claim',
    latest_batch_review='configs/B7-REVIEW.md',decision='Stop this bounded parent recipe; preserve scoped-negative evidence and working native SNF qualification')
for p in [H/'OWNER.json',Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')]:p.write_text(json.dumps(owner,indent=2)+'\n')
log=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');body=log.read_text();start=body.index('**New Burgers tree-embedding study:');end=body.index('\n\n',start)
status=f"**New Burgers tree-embedding study: B7 closed with a scoped-negative parent result.** Final nonlinear parent worst development error {100*r['worst_independent_error']:.6f}%, actual 4096 {100*r['actual4096']['relative_error']:.6f}%; parent gate failed after the two bounded training repairs. All 484 primary reconstructions,12 named family/viscosity diagnostics, full4096 time-series exports and fixed-weight starting-point diagnosis are retained. Native SNF float64 GPU/gradient/scalar-RHS qualification now passes, but no trained fair CROM/SNF PDE comparison is available. Batch {used/3600:.6f} allocated GPU-hours, cumulative {audit['allocated_gpu_hours']:.6f}; all jobs terminal and verified backups complete. Final tests sealed, old campaign paused, no merge or continuation. [Full-resolution plots, diagnosis and decision](reports/2026-11-19-neural-parent-training.md)."
body=body[:start]+status+body[end:];body+='\n\n## 2026-11-19\n\n### B7 bounded parent-training closeout — Codex\n\n'+status+'\n\n'+text.split('\n\n',1)[1];log.write_text(body)
print(status)
