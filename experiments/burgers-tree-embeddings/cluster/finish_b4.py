"""Finite owned B4 closeout and gated first B5 submission; never merges.

Do not edit this lane while this controller is running: it commits and mirrors
its own reports, resource ledgers and manifests. It exits after the first B5 job
is submitted, or earlier on any failed gate. It does not start further jobs.
"""
from pathlib import Path
import datetime,hashlib,json,os,shlex,subprocess as sp,time
H=Path(__file__).resolve().parents[1];PY='/home/tahmid/Dev/.venv/bin/python'
TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
os.chdir(H)
def run(*args):
    print('operation',args,flush=True);sp.run(list(args),check=True,cwd=H)
def read(p):return json.loads((H/p).read_text())
def write(p,value):(H/p).write_text(json.dumps(value,indent=2)+'\n')
def sha(p):return hashlib.sha256((H/p).read_bytes()).hexdigest()
def backup(message):
    run('git','-c','gc.auto=0','add','.')
    changed=sp.run(['git','-c','gc.auto=0','diff','--cached','--quiet'],cwd=H).returncode
    assert changed in (0,1)
    if changed:
        run('git','-c','gc.auto=0','commit','-m',message)
        run('bash','sync_github.sh')
def reconcile():
    result=sp.run([PY,'cluster/run.py','reconcile'],stdout=sp.PIPE,stderr=sp.STDOUT,text=True,cwd=H)
    assert result.returncode==0,result.stdout
    return read('cluster/ledger.json')
def latest_b4_gate():
    target={n:sha(f'results/{n}/output/result.json') for n in ['b4base','b4time','b4quad']}
    candidates=[]
    for p in (H/'manifests').glob('B4-progress-*.json'):
        r=json.loads(p.read_text())
        if r['source_hashes']==target and not r['active_jobs']:candidates.append((p,r))
    assert candidates,'No retained final B4 record matches the source bytes'
    return max(candidates,key=lambda v:v[1]['utc'])

assert not (H/'manifests/B5-registration.json').exists(),'B5 already registered; no duplicate activation'
assert not sp.check_output(['git','-c','gc.auto=0','status','--porcelain','--',str(H)],text=True).strip()
print('owned_controller_pid',os.getpid(),flush=True)
deadline=time.monotonic()+3*3600
while True:
    rows=reconcile();selected=[v for v in rows if v['name'] in ['b4base','b4time','b4quad']]
    assert len(selected)==3
    print('B4_states',[(v['name'],v.get('state'),v.get('elapsed_seconds')) for v in selected],flush=True)
    if all(v.get('state','').split()[0] in TERMINAL for v in selected):break
    assert time.monotonic()<deadline,'Bounded controller wait expired; do not duplicate live jobs'
    time.sleep(45)

for row in selected:
    manifest=H/'manifests'/f'retention-{row["name"]}.json'
    if not manifest.exists():run(PY,'cluster/collect.py',row['name'])
completed=[v['name'] for v in selected if v['state']=='COMPLETED']
if completed:run(PY,'audit_fixed_pilot.py',*completed,'--record','B4-final-returned-step-audit.json')
run(PY,'reports/update_development.py')
rows=reconcile()
run(PY,'cluster/audit_accounting.py','--label','B4')
run(PY,'cluster/archive_owned.py','--label','B4')
run(PY,'reports/update_development.py')
account=read('manifests/allocation-audit-B4.json');batch=read('manifests/B4-registration.json')
total=account['allocated_gpu_seconds'];used=total-batch['baseline_allocated_gpu_seconds']
assert used<=batch['max_additional_gpu_seconds']
passed=False;gatefile=None;gate=None
if len(completed)==3:
    gatefile,gate=latest_b4_gate()
    passed=not gate['failed'] and not gate['remaining'] and len(gate['records'])==3 and all(r['passed'] and len(r['rows'])==88 for r in gate['records'])
decision=('All44 development trajectories pass physical accuracy, full returned-step convergence, actual4096 output, half-time-step and doubled-integration checks for both frozen methods.'
          if passed else 'B4 does not pass its complete all-development gate. Preserve the evidence and diagnose it before any further allocation.')
metrics=[]
if gate:
    for record in gate['records']:
        for method in ['tree_direct24','global_stable960']:
            values=[v for v in record['rows'] if v['method']==method]
            metrics.append(f"|{record['name']}|{method}|{max(v['max_physical_error']+v['reference_allowance'] for v in values)*100:.9f}%|{max((v['max_field_change'] or 0) for v in values)*100:.9f}%|{sum(v['nonconverged_steps'] for v in values)}|")
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
(H/'configs/B4-REVIEW.md').write_text(f'''# B4 closeout and complete-query advancement decision

{decision}

B4 used{used} allocated GPU-seconds of its{batch['max_additional_gpu_seconds']}-second ceiling. Cumulative usage is{total} seconds ({total/3600:.9f}h), including failures. All{len(rows)} jobs are terminal; maximum pending-or-running owned concurrency is{account['max_pending_or_running_owned_jobs']}. Raw outputs, logs and source records have verified local and designated archive copies. No trained recipe, final-test opening, nonlinear solver restart or merge occurred.

|Run|Method|Worst physical error + allowance|Largest refinement change|Nonconverged steps|
|---|---|---:|---:|---:|
{chr(10).join(metrics)}

The common physical setting remains L16384, Crank–Nicolson step0.00125, Q512 and22016 tests. This is development fixed-tree qualification, not deployable adaptation or a runtime win. The original nonlinear PDE optimizer and frozen tangent ablation remain stopped. No native NM-ROM competitor has been defeated by these controls.

{'The passed gate justifies B5 under B5-COMPLETE-QUERY-DIAGNOSTIC.md and B5-COMPACT-AMENDMENT.md: at most7200 additional allocated GPU-seconds, zero training recipes, two concurrent jobs and45 minutes per job. The next question is complete query cost with equivalent full-cell caches, normal-matrix reuse and one compact-patch alternative. Begin only with the20-minute catalogue equivalence job; later jobs require their retained predecessor gates. Unused B4 allowance is not carried over.' if passed else 'No B5 activation is justified by this closeout. A failed or incomplete gate needs a scoped diagnosis and a prospectively documented finite repair or stop decision.'}

Recorded {stamp}. Accounting: manifests/allocation-audit-B4.json. Archive: manifests/archive-closeout-B4.json. Old campaign remains paused; no unrelated3D work.
''')
batches=read('cluster/BATCHES.json');b4=next(v for v in batches['batches'] if v['id']=='B4')
b4.update(status='CLOSED_REVIEWED' if passed else 'CLOSED_GATE_FAILED',actual_additional_gpu_seconds=used,
    cumulative_actual_gpu_seconds=total,review='configs/B4-REVIEW.md',accounting='manifests/allocation-audit-B4.json',archive='manifests/archive-closeout-B4.json')
write('cluster/BATCHES.json',batches)
backup('Close and archive all-development fixed-tree qualification')
if not passed:raise SystemExit('B4 gate failed: no new allocation submitted')

assert total+7200<32*3600,'A prospective32-hour review is required before expansion'
evidence=['configs/B4-REVIEW.md','configs/B5-COMPLETE-QUERY-DIAGNOSTIC.md','configs/B5-COMPACT-AMENDMENT.md',
    'manifests/allocation-audit-B4.json','manifests/archive-closeout-B4.json',str(gatefile.relative_to(H))]
registration=dict(id='B5',baseline_allocated_gpu_seconds=total,max_additional_gpu_seconds=7200,
    cumulative_ceiling_gpu_seconds=total+7200,purpose='Equivalent full-cell caches, normal reuse, compact patches and paired complete query costs',
    recipe_cap=0,max_concurrent=2,max_seconds_per_job=2700,configuration='configs/B5-COMPLETE-QUERY-DIAGNOSTIC.md',
    amendments=['configs/B5-COMPACT-AMENDMENT.md'],registered_utc=stamp,
    activation_evidence={p:sha(p) for p in evidence},development_admission=str(gatefile.relative_to(H)))
write('manifests/B5-registration.json',registration)
batches['batches'].append(registration);batches['cumulative_ceiling_gpu_seconds']=total+7200;write('cluster/BATCHES.json',batches)
run(PY,'reports/update_complete_query.py')
backup('Register finite complete-query diagnostic after all-development pass')
baseline=next(v for v in read('cluster/ledger.json') if v['name']=='b4base')
tokens=shlex.split(baseline['args'])
args=[]
for key in ['cache','span','stable-global','admission','dev']:
    args.extend(['--'+key,tokens[tokens.index('--'+key)+1]])
args.extend(['--development-admission',str(gatefile.relative_to(H)),
             '--compare-previous',baseline['archived_remote']+'/output'])
run(PY,'cluster/run.py','submit','--name','b5cache','--driver','complete_query_diagnostic.py','--minutes','20','--args',shlex.join(args))
backup('Launch bounded catalogue-equivalence complete queries')
run(PY,'reports/update_complete_query.py')
backup('Record active complete-query catalogue qualification')
print('controller_complete: B4 closed; only b5cache submitted; inspect its retained result before continuing',flush=True)
