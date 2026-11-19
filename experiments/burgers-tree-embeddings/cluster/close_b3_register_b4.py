"""Review retained B3 evidence before activating the all-development batch."""
from pathlib import Path
import datetime, hashlib, json
H=Path(__file__).resolve().parents[1]
def read(p):return json.loads((H/p).read_text())
def sha(p):return hashlib.sha256((H/p).read_bytes()).hexdigest()
def dump(p,v):(H/p).write_text(json.dumps(v,indent=2)+'\n')
account=read('manifests/allocation-audit-B3.json');archive=read('manifests/archive-closeout-B3.json')
assert account['status']=='passed_scheduler_allocation_audit' and not account['owned_jobs_still_queued_or_running']
assert {r['name'] for r in archive}=={'b3packed','b3tangent'} and len(archive)==2
p=max((H/'manifests').glob('B3-progress-*.json'),key=lambda p:p.stat().st_mtime)
progress=json.loads(p.read_text());assert not progress['active_jobs']
assert progress['records']['packed']['equivalent_field_gate']
tangent=read('results/b3tangent/output/result.json')
assert not tangent['representation_gate'] and all(v['fitted'] for v in tangent['cases'])
assert all(v['conditioning_pass'] and v['useful_independent_directions'] for v in tangent['cases'])
total=account['allocated_gpu_seconds'];assert total==progress['elapsed_gpu_seconds']
batches=read('cluster/BATCHES.json');assert not any(b['id']=='B4' for b in batches['batches'])
b3=next(b for b in batches['batches'] if b['id']=='B3');assert not b3.get('status')
additional=total-b3['baseline_allocated_gpu_seconds'];assert additional<=b3['max_additional_gpu_seconds']
change=max(v['max_field_change'] for v in progress['records']['packed']['rows'])
station=max(max(v['stationarity']) for v in tangent['cases'])
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
review=f'''# B3 closeout and all-development advancement decision

B3 used {additional} allocated GPU-seconds of its 3600-second allowance. Cumulative study usage is {total} seconds ({total/3600:.6f} hours), including failures. All {len(account['jobs'])} jobs are terminal, maximum owned concurrency including pending jobs is {account['max_pending_or_running_owned_jobs']}, and the two B3 scientific results have verified local and designated archive copies. No contingency allocation or new training recipe was used.

The equivalent packed field implementation passes all three pilot cases, physical accuracy and complete returned solver convergence. Its maximum current-time field difference from the compact implementation is {change:.12g}, below the preregistered 1e-7 equivalence limit. This qualifies the implementation for the next diagnostic. It does not establish a speed ratio: pilot timings charge joint setup, compilation and method-specific work without the confirmation protocol.

The frozen zero-code tangent decoder fails the unchanged 0.5% representation screen. Its worst physical projection error is {tangent['projection']['worst']*100:.8f}%, with actual4096 witness {tangent['actual4096_witness']['error']*100:.8f}% at {tangent['actual4096_witness']['case']}, t={tangent['actual4096_witness']['time']}. Maximum scaled stationarity is {station:.8g}; all fits and conditioning checks pass. This is a limitation of this frozen linear span, not evidence that the fits merely need more optimization. Stop this tangent ablation, retaining its same-weight comparison with the admitted nonlinear representation. The earlier nonlinear PDE optimizer stop remains in force.

The next unresolved question is whether the admitted direct24/52-node/1376-variable tree and stable global960 satisfy the physical contract over all 44 development trajectories, including independent half-time-step and doubled-integration checks. The three-case qualification and equivalent implementation support this finite extension. They do not yet support deployable adaptation or complete-runtime confirmation.

Review the cumulative eight-hour checkpoint again **before** a larger batch can cross it. Register B4 under B4-ALL-DEVELOPMENT.md with at most 14400 additional allocated GPU-seconds, giving a cumulative ceiling {total+14400} seconds ({(total+14400)/3600:.6f} hours). Run the baseline first; only a complete all-case pass permits the two refinements. Count full outstanding reservations, all failures and compilation. Unused B3 allowance is not transferred. No new trained recipe, cell-catalogue GPU experiment, architecture sweep, tolerance relaxation or final-test opening is part of this batch.

The full-cell catalogue has only a passing sub-minute CPU fixture, with zero observed differences from fresh assembly for three geometries. It is not used in B4. Native NM-ROM confirmation, paired runtime superiority, adaptation and final method freeze remain open. The old campaign remains paused; no automatic merge or unrelated3D work. Registered {stamp}.
'''
(H/'configs/B3-REVIEW.md').write_text(review)
b3.update(status='CLOSED_REVIEWED',actual_additional_gpu_seconds=additional,cumulative_actual_gpu_seconds=total,
          review='configs/B3-REVIEW.md',accounting='manifests/allocation-audit-B3.json',archive='manifests/archive-closeout-B3.json')
evidence=['configs/B3-REVIEW.md','configs/B4-ALL-DEVELOPMENT.md','manifests/allocation-audit-B3.json',
          'manifests/archive-closeout-B3.json',str(p.relative_to(H))]
b4=dict(id='B4',baseline_allocated_gpu_seconds=total,max_additional_gpu_seconds=14400,
        cumulative_ceiling_gpu_seconds=total+14400,purpose='All44 development fixed-tree physical, time and integration gates',
        recipe_cap=0,max_concurrent=2,max_seconds_per_job=5400,configuration='configs/B4-ALL-DEVELOPMENT.md',
        registered_utc=stamp,eight_hour_review='configs/B3-REVIEW.md',activation_evidence={v:sha(v) for v in evidence})
batches['batches'].append(b4);batches['cumulative_ceiling_gpu_seconds']=total+14400
dump('cluster/BATCHES.json',batches);dump('manifests/B4-registration.json',b4)
owner=read('OWNER.json');owner.update(updated_utc=stamp,status='B4_REGISTERED_ALL_DEVELOPMENT',
    decision='Packed implementation admitted; tangent ablation stopped; all-development fixed-tree qualification registered',
    batch_gpu_seconds=14400,cumulative_ceiling_gpu_seconds=total+14400)
dump('OWNER.json',owner)
claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')
assert json.loads(claim.read_text())['owner']==owner['owner'];claim.write_text(json.dumps(owner,indent=2)+'\n')
status=f'''**New Burgers tree-embedding study: B4 all-development qualification registered.** The packed field implementation passes three-case equivalence; the frozen tangent representation fails at {tangent['projection']['worst']*100:.6f}% and is stopped. The admitted direct1376 and stronger global960 proceed to all44 development trajectories under a four-GPU-hour cap, with time/integration refinements conditional on the baseline passing. Cumulative allocation {total/3600:.6f}h; no active jobs at registration. No runtime win, final-test opening or method freeze. [Representation](reports/2026-11-18-tree-embedding-representation.md); [B3 evidence](reports/2026-11-19-tree-embedding-implementation-tangent.md). Old campaign remains PAUSED; no merge.'''
lab=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');text=lab.read_text()
start=text.index('**New Burgers tree-embedding study:');end=text.index('\n\n',start)
text=text[:start]+status+text[end:]
lab.write_text(text.rstrip()+f'\n\n### 2026-11-19 — Tree-embedding B3 closeout and B4 registration\n\n{status}\n\n{review}\n')
print(status)
