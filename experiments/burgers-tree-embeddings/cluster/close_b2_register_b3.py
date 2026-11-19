"""Activate a finite continuation only after retained passing B2 closeout."""
from pathlib import Path
import datetime, hashlib, json

H = Path(__file__).resolve().parents[1]
def read(p): return json.loads((H/p).read_text())
def sha(p): return hashlib.sha256((H/p).read_bytes()).hexdigest()
def dump(p, v): (H/p).write_text(json.dumps(v, indent=2)+'\n')

account = read('manifests/allocation-audit-B2.json')
archive = read('manifests/archive-closeout-B2.json')
assert account['status'] == 'passed_scheduler_allocation_audit'
assert not account['owned_jobs_still_queued_or_running']
assert len(archive) == 8 and {r['name'] for r in archive} == {
    'b2base','b2space','b2quad','b2rspace','b2rquad','b2rtests','b2rtime','b2rtight'}
progress_paths = sorted((H/'manifests').glob('B2-progress-*.json'), key=lambda p:p.stat().st_mtime)
p = progress_paths[-1]; progress = json.loads(p.read_text())
assert not progress['active_jobs'] and not progress['remaining'] and not progress['failed']
assert all(r['passed'] for r in progress['records']) and len(progress['records']) == 6
total = account['allocated_gpu_seconds']; additional = total-22744
assert 0 < additional <= 5400 and total == progress['elapsed_gpu_seconds']
batches = read('cluster/BATCHES.json'); assert not any(b['id'] == 'B3' for b in batches['batches'])
b2 = next(b for b in batches['batches'] if b['id'] == 'B2')
assert not b2.get('status')
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
maximum = {r['name']:max((v['max_field_change'] or 0) for v in r['rows']) for r in progress['records']}
physical = max(v['max_physical_error'] for r in progress['records'] for v in r['rows'])
review = f'''# B2 closeout and prospective eight-hour review

The scheduler records{total} cumulative allocated GPU-seconds ({total/3600:.6f} hours), including failures. B2 used{additional} seconds of its5400-second allowance. All{len(account['jobs'])} study jobs are terminal; the measured maximum including pending jobs is{account['max_pending_or_running_owned_jobs']}. All eight B2 raw records, original manifests, logs, source hashes and scripts have verified local and designated cluster archive copies. The earlier archives are unchanged.

The original L8192 baseline failed the strict0.1% spatial sensitivity check. Its unchanged L16384 refinement was reused as the finer baseline on physical/full-convergence evidence only. The single registered further spatial check passed. At L16384, CN step0.00125, Q512 and22016 tests, all six revised baseline/refinement roles pass for dev000,dev003,align002 and both frozen models. Maximum physical error across these runs is{physical*100:.8f}% before adding the individually passing reference allowances. All returned steps satisfy the complete stopping rule, and actual4096 final fields are retained.

| Independent refinement | Maximum current-time field change |
|---|---:|
| Spatial operator, L32768 |{maximum['b2rspace']*100:.9f}%|
| Integration, Q1024 |{maximum['b2rquad']*100:.9f}%|
| Weak tests,44032 |{maximum['b2rtests']*100:.9f}%|
| Half time step |{maximum['b2rtime']*100:.9f}%|
|100x tighter initial/inner/outer tolerances |{maximum['b2rtight']*100:.9f}%|

This admits a three-case numerical setting, not the entire development panel. The representation evidence still favors the stable global960 control over the larger trees on worst error per variable. The frozen nonlinear checkpoint remains stopped after two failed PDE optimizer repairs. No method-specific runtime win, deployable adaptation, native NM-ROM success, method freeze or final-test result is claimed.

The cumulative eight-hour checkpoint is reviewed **before** a batch that can cross it. Continued compute is justified by two specific unresolved questions: an equivalent fixed-shape field implementation can test the measured geometry-specific compilation concern, and a frozen tangent architecture can test whether nonlinear decoder curvature is necessary for the admitted reconstruction accuracy. These are finite falsifiable diagnostics, not a larger training search. Activate B3 with at most3600 additional allocated GPU-seconds, two initial20-minute jobs and a single prospectively amended implementation-defect contingency. The new cumulative ceiling is{total+3600} seconds ({(total+3600)/3600:.6f} hours). See B3-IMPLEMENTATION-AND-TANGENT.md for configurations, gates and expected artifacts. Unused B2 allowance is not carried over.

No final-test solution fields were opened. The old campaign remains paused; no automatic merge or unrelated3D work. Registered {stamp}.
'''
(H/'configs/B2-REVIEW.md').write_text(review)
b2.update(status='CLOSED_REVIEWED', actual_additional_gpu_seconds=additional,
          cumulative_actual_gpu_seconds=total, review='configs/B2-REVIEW.md',
          accounting='manifests/allocation-audit-B2.json', archive='manifests/archive-closeout-B2.json')
evidence = ['configs/B2-REVIEW.md','configs/B3-IMPLEMENTATION-AND-TANGENT.md',
            'manifests/allocation-audit-B2.json','manifests/archive-closeout-B2.json',str(p.relative_to(H))]
b3 = dict(id='B3', baseline_allocated_gpu_seconds=total,max_additional_gpu_seconds=3600,
          cumulative_ceiling_gpu_seconds=total+3600,recipe_cap=0,max_concurrent=2,max_seconds_per_job=1200,
          purpose='Equivalent packed direct-tree PDE pilot and one frozen tangent representation screen',
          configuration='configs/B3-IMPLEMENTATION-AND-TANGENT.md',registered_utc=stamp,
          activation_evidence={v:sha(v) for v in evidence},eight_hour_review='configs/B2-REVIEW.md')
batches['batches'].append(b3);batches['cumulative_ceiling_gpu_seconds']=total+3600
dump('cluster/BATCHES.json',batches);dump('manifests/B3-registration.json',b3)
owner=read('OWNER.json')
owner.update(updated_utc=stamp,status='B3_REGISTERED_IMPLEMENTATION_AND_TANGENT',active_jobs=0,
             decision='B2 three-case numerical setting admitted; B3 finite diagnostic registered before crossing eight allocated hours',
             batch_gpu_seconds=3600,cumulative_ceiling_gpu_seconds=total+3600)
dump('OWNER.json',owner)
claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')
assert json.loads(claim.read_text())['owner']==owner['owner']
claim.write_text(json.dumps(owner,indent=2)+'\n')
lab=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');text=lab.read_text()
status=f'''**New Burgers tree-embedding study: B2 admitted; B3 finite diagnostics registered.** All three pilot cases pass physical accuracy, complete convergence and independent spatial/integration/test/time/100x-precision checks for direct1376 and global960. Cumulative allocated time{total/3600:.6f}h; no active owned jobs at registration. The eight-hour checkpoint is reviewed prospectively; B3 allows one additional GPU-hour for equivalent field evaluation and one frozen tangent representation screen. Nonlinear PDE optimizer remains stopped; no all-development PDE admission, runtime win or final-test opening. [Representation plots](reports/2026-11-18-tree-embedding-representation.md); [numerical checks](reports/2026-11-18-tree-embedding-fixed-tree.md). Old campaign remains PAUSED; no merge.'''
start=text.index('**New Burgers tree-embedding study:');end=text.index('\n\n',start)
text=text[:start]+status+text[end:]
text=text.rstrip()+f'\n\n### 2026-11-19 — Tree-embedding B2 closeout and prospective eight-hour review\n\n{status}\n\n{review}\n'
lab.write_text(text)
print(status)
