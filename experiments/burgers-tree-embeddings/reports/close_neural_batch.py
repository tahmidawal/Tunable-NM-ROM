"""Evidence-generated B6 closeout after every owned job and raw backup is terminal."""
from pathlib import Path
import json,datetime,hashlib
H=Path(__file__).resolve().parents[1]
def read(p):return json.loads((H/p).read_text())
def save(p,v):(H/p).write_text(json.dumps(v,indent=2)+'\n')
audit=read('manifests/allocation-audit-B6.json');archive=read('manifests/archive-closeout-B6.json')
assert audit['status']=='passed_scheduler_allocation_audit' and not audit['owned_jobs_still_queued_or_running']
span=read('results/b6span/output/result.json');assert not span['capacity_pass']
named=read('results/b6namedenc/output/result.json');assert named['status']=='complete'
crom=read('results/b6cromprojection/output/result.json');snf=read('results/b6snffixture/output/result.json')
assert crom['status']==snf['status']=='complete'
now=datetime.datetime.now(datetime.timezone.utc).isoformat();total=audit['allocated_gpu_seconds'];added=total-40842
batch=read('cluster/BATCHES.json');b=next(v for v in batch['batches'] if v['id']=='B6')
assert added<=b['max_additional_gpu_seconds'];b.update(status='CLOSED_SCOPED_NEGATIVE',actual_additional_gpu_seconds=added,
    cumulative_actual_gpu_seconds=total,review='configs/B6-REVIEW.md',accounting='manifests/allocation-audit-B6.json',archive='manifests/archive-closeout-B6.json')
save('cluster/BATCHES.json',batch)
ps=next(v for v in span['allocations'] if v['kind']=='prescribed' and v['max_depth']==3)
summary=(f"**New Burgers tree-embedding study: B6 bounded neural round closed, scoped negative.** Actual nonlinear child training failed; the final frozen-parent neural-feature repair stops at its capacity prerequisite ({100*ps['worst_independent_error']:.5f}% worst independent physical reconstruction versus0.5% required). This is a direct-feature diagnostic, not a trained nonlinear-child result. All12 named Burgers references and retained neural transfer tests are complete; model failures and unfinished fits remain visible. Native CROM projection diagnostic worst{100*crom['worst_projected']:.5f}%, not an eligible strong comparator. Native SNF GPU/scalar-equivalence fixture passed:{snf.get('passed',False)}; no trained SNF comparator. No new neural PDE, deployable adaptation or complete-runtime claim. B6 used{added} allocated GPU-seconds ({added/3600:.6f}h) of8h; cumulative{total/3600:.6f}h. All{len(audit['jobs'])} owned jobs terminal and raw backups verified. [Results, plots and decision](reports/2026-11-19-neural-tree-development.md). Final tests remain unopened, old campaign paused, no merge or active worker.")
owner=read('OWNER.json');owner.update(status='B6_CLOSED_SCOPED_NEGATIVE',updated_utc=now,decision='Neural representation gate failed after bounded repairs; no PDE/runtime admission',
    allocated_gpu_seconds=total,allocated_gpu_hours=total/3600,completed_jobs=sum(v['terminal_state']=='COMPLETED' for v in audit['jobs']),terminal_jobs=len(audit['jobs']),active_jobs=0,
    latest_neural_status=summary,latest_batch_review='configs/B6-REVIEW.md',continuation=dict(status='closed',active=False))
save('OWNER.json',owner)
claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json');c=json.loads(claim.read_text());assert c['owner']==owner['owner']
c.update(owner);claim.write_text(json.dumps(c,indent=2)+'\n')
recipes=read('configs/B6-RECIPES.json');recipes['recipes'][2]['status']='native training/projection diagnostic failed; strong PDE comparator not reproduced'
recipes['recipes'][3]['status']='native GPU scalar-equivalence fixture '+str(snf.get('passed',False))+'; no trained comparator'
save('configs/B6-RECIPES.json',recipes)
rows='\n'.join(f"|{v['definition']['name']}|{v['nu']}|{100*v['worst_parent']:.4f}%|{100*v['worst_tree']:.4f}%|{v['unfinished_tree']}/11|" for v in named['cases'])
review=f'''# B6 actual nonlinear neural-tree closeout

The bounded neural round does not achieve a usable neural tree or an eligible speed comparison. The failed repairs stop at their registered gate; no third neural optimizer/architecture search follows.

{summary}

The first neural child recipe damaged its jointly trained parent. Independent child directions remained present, so the failure cannot be attributed simply to zero-code Jacobian collapse. The final repair learned its spatial neural features accurately but failed assembled physical capacity at the tested node/feature budget. This fixed-parent control is not a universal representation lower bound, and its direct coefficients are not the requested nonlinear child model.

Named-case transfer uses the retained actual nonlinear checkpoint, with unchanged weights and independently refitted root/child embeddings. It is offline reconstruction on independent256² quadrature, not an embedding-space PDE trajectory or actual4096 output qualification. Ridge/two-pulse fields and other viscosities are exposed development extensions, not added training cases.

|Family|Viscosity|Worst parent error|Worst retained nonlinear-tree error|Unfinished tree fits|
|---|---:|---:|---:|---:|
{rows}

The initial named diagnostic omitted the trained encoder and was explicitly cancelled. Its raw partial results are preserved but excluded from comparative claims; the corrected evaluation restores the existing initializer. Native SNF's first precompile conflict was bypassed with cache-disabled import, and our synthetic fixture's method ambiguity was corrected explicitly. These are reproduction/implementation issues, not evidence that SNF-ROM is inaccurate. CROM's retained training schedule differs from author CLI defaults; its weak result is not a strong confirmed baseline. The corrected SNF fixture reached GPU kernel compilation but failed with InvalidIRError in GPUArrays/CUDA/LLVM; successful import and device detection did not qualify its numerical kernels. The bounded environment investigation is closed without a trained SNF model. No superiority claim over CROM or SNF-ROM is supported.

All allocation failures, cancelled work and environment qualification are charged. Raw outputs have local and designated-cluster archive copies verified by SHA256. No final-test fields were opened, paused-campaign lane modified, branch merged, or background continuation left active.
'''
(H/'configs/B6-REVIEW.md').write_text(review)
log=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');s=log.read_text();a=s.index('**New Burgers tree-embedding study:');z=s.index('\n\n',a);s=s[:a]+summary+s[z:]
s+='\n\n## 2026-11-19\n\n### Actual neural-tree B6 bounded closeout — Codex\n\n'+review.split('\n\n',1)[1]+'\n';log.write_text(s)
(H/'reports/B6-canonical-log-entry.md').write_text(summary+'\n\n'+review)
save('manifests/B6-closeout.json',dict(utc=now,summary=summary,canonical_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    report='/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-19-neural-tree-development.md',allocated_gpu_seconds=total,added_gpu_seconds=added,
    achieved_objective=False,representation_admitted=False,pde_admitted=False,final_test_opened=False,active_jobs=0))
print(summary)
