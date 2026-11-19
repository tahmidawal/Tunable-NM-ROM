"""Finite, fail-closed continuation of the registered B5 jobs only.

No new budget, training, architecture, final-test opening or merge is allowed.
This controller owns lane writes while active. It preserves every result and
backs up each commit before submitting the next dependent job.
"""
from pathlib import Path
import datetime,hashlib,json,os,shlex,subprocess as sp,time,traceback
H=Path(__file__).resolve().parents[1];PY='/home/tahmid/Dev/.venv/bin/python'
TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
os.chdir(H);deadline=time.monotonic()+48*3600
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads((H/p).read_text())
def write(p,v):(H/p).write_text(json.dumps(v,indent=2)+'\n')
def run(*args):
    print(utc(),'operation',args,flush=True);sp.run(list(args),cwd=H,check=True)
def reconcile():
    result=sp.run([PY,'cluster/run.py','reconcile'],cwd=H,stdout=sp.PIPE,stderr=sp.STDOUT,text=True)
    assert result.returncode==0,result.stdout
    return read('cluster/ledger.json')
def backup(message):
    run('git','-c','gc.auto=0','add','.')
    changed=sp.run(['git','-c','gc.auto=0','diff','--cached','--quiet'],cwd=H).returncode
    assert changed in (0,1)
    if changed:
        run('git','-c','gc.auto=0','commit','-m',message)
        run('bash','sync_github.sh')
def wait(names):
    last=None
    while True:
        rows=reconcile();selected=[v for v in rows if v['name'] in names]
        assert len(selected)==len(names)
        status=[(v['name'],v.get('state','PENDING')) for v in selected]
        if status!=last:print(utc(),'waiting',status,flush=True);last=status
        if all((v.get('state') or 'PENDING').split()[0] in TERMINAL for v in selected):return selected
        assert time.monotonic()<deadline,'Bounded48-hour queue watchdog expired; do not duplicate existing jobs'
        time.sleep(45)
def latest():
    paths=list((H/'manifests').glob('B5-progress-*.json'));assert paths
    return max((json.loads(p.read_text()) for p in paths),key=lambda v:v['utc'])
def collect_and_audit(names):
    rows=wait(names)
    for row in rows:
        name=row['name']
        if not (H/'manifests'/f'retention-{name}.json').exists():run(PY,'cluster/collect.py',name)
    reconcile();run(PY,'reports/update_complete_query.py')
    backup('Retain and independently audit '+', '.join(names))
    audit=latest()
    return all(audit.get('audits',{}).get(n,{}).get('passed',False) for n in names)
def submit(name,tokens,minutes):
    assert name in ['b5normal','b5compact','b5block0','b5block1']
    assert minutes<=45
    source=next(v for v in read('cluster/ledger.json') if v['name']=='b5cache')
    assert source['gpu']=='a100','This registration uses A100-80G; hardware changes need a prospective amendment'
    run(PY,'cluster/run.py','submit','--name',name,'--driver','complete_query_diagnostic.py',
        '--minutes',str(minutes),'--gpu',source['gpu'],'--args',shlex.join(tokens))
    backup('Submit registered '+name+' after its retained predecessor gate')
    run(PY,'reports/update_complete_query.py');backup('Record active '+name+' continuation')
def replace(tokens,flag,value):
    tokens=list(tokens)
    if flag in tokens:tokens[tokens.index(flag)+1]=str(value)
    else:tokens.extend([flag,str(value)])
    return tokens
def close(reason):
    rows=reconcile();owner=read('OWNER.json');ns=owner['namespace']
    assert all((v.get('state') or 'PENDING').split()[0] in TERMINAL for v in rows),'Keep live job evidence; do not archive active inputs'
    run(PY,'cluster/audit_accounting.py','--label','B5')
    run(PY,'cluster/archive_owned.py','--label','B5')
    run(PY,'reports/update_complete_query.py')
    record=latest();account=read('manifests/allocation-audit-B5.json');reg=read('manifests/B5-registration.json')
    total=account['allocated_gpu_seconds'];used=total-reg['baseline_allocated_gpu_seconds']
    assert used<=reg['max_additional_gpu_seconds']
    table=[]
    for v in record['summary']:
        table.append(f"|{v['method']}|{v['outputs']}|{v['median']:.8f}|[{v['bootstrap95'][0]:.8f}, {v['bootstrap95'][1]:.8f}]|{v['worst']:.8f}|{v['faster_cases']}/3|")
    (H/'configs/B5-REVIEW.md').write_text(f'''# B5 bounded complete-query closeout

{reason}

{record['decision']}

The batch used{used} allocated GPU-seconds of its{reg['max_additional_gpu_seconds']}-second ceiling. Cumulative study usage is{total} seconds ({total/3600:.9f}h), including failures. All{len(rows)} owned jobs are terminal, and maximum pending/running concurrency is{account['max_pending_or_running_owned_jobs']}. Raw outputs, logs, repetitions and source records have verified local and designated archive copies. No automatic compute expansion follows this review.

|Tree implementation|Device fields|Median paired ratio|Trajectory-bootstrap95% interval|Worst ratio|Cases faster|
|---|---:|---:|---|---:|---:|
{chr(10).join(table) or '|No admitted paired result||||||'}

Each ratio compares complete invocations within one GPU allocation; no raw time is compared across jobs. The three-case diagnostic cannot establish native NM-ROM superiority or final-test success. A failed equivalence gate excludes that implementation. A failed cost gate is scoped to these frozen representations and tested implementations, not every possible hierarchy. Any advancement beyond this batch needs its own finite registration tied to the measured bottleneck. Prepared probes have no calibrated thresholds and do not constitute qualified adaptation. The earlier frozen nonlinear optimizer and tangent-screen stops remain in force.

Recorded {utc()}. No final-test solution fields were opened, no old-campaign artifacts changed and no branch merged. Full results and plots: reports/2026-11-19-tree-embedding-complete-query.md.
''')
    batches=read('cluster/BATCHES.json');b5=next(v for v in batches['batches'] if v['id']=='B5')
    complete=not record['failed'] and not record['remaining']
    b5.update(status='CLOSED_REVIEWED' if complete else 'CLOSED_GATE_FAILED_OR_INCOMPLETE',
        actual_additional_gpu_seconds=used,cumulative_actual_gpu_seconds=total,
        review='configs/B5-REVIEW.md',accounting='manifests/allocation-audit-B5.json',archive='manifests/archive-closeout-B5.json')
    write('cluster/BATCHES.json',batches)
    backup('Close finite complete-query diagnostic with retained evidence and scoped decision')
    return dict(status='closed',reason=reason,decision=record['decision'],allocated_gpu_seconds=total,
        batch_gpu_seconds=used,paired_results_admitted=complete,finished_utc=utc())

def main():
    reg=read('manifests/B5-registration.json')
    assert reg['max_additional_gpu_seconds']==7200 and reg['cumulative_ceiling_gpu_seconds']==42296
    assert not (H/'manifests/archive-closeout-B5.json').exists()
    rows=read('cluster/ledger.json');first=next(v for v in rows if v['name']=='b5cache')
    assert first['id']==7362119
    assert not any(v['name'] in ['b5normal','b5compact','b5block0','b5block1'] for v in rows),'No duplicate continuation'
    assert not sp.check_output(['git','-c','gc.auto=0','status','--porcelain','--',str(H)],text=True).strip()
    owner=read('OWNER.json');ns=owner['namespace'];base=shlex.split(first['args'])
    if not collect_and_audit(['b5cache']):return close('Catalogue equivalence failed; downstream jobs were not submitted.')
    normal=replace(base,'--compare-previous',ns+'/b5cache/output')+['--normal-reuse']
    submit('b5normal',normal,20)
    if not collect_and_audit(['b5normal']):return close('Normal-matrix equivalence failed; compact and paired jobs were not submitted.')
    compact=replace(normal,'--compare-previous',ns+'/b5normal/output')+['--field-kernel','compact','--include-packed-control']
    submit('b5compact',compact,20)
    if not collect_and_audit(['b5compact']):return close('Compact-kernel equivalence failed; paired timing was not submitted.')
    rows=reconcile();charged=sum(v['elapsed_seconds'] for v in rows)
    if charged+2*45*60>reg['cumulative_ceiling_gpu_seconds']:
        return close('The remaining finite allocation cannot reserve both45-minute blocks. No budget expansion or unregistered shortened run was made.')
    paired=replace(compact,'--compare-previous',ns+'/b5compact/output')+['--mode','paired',
        '--qualified-implementation',ns+'/b5compact/output']
    submit('b5block0',paired+['--block-index','0'],45)
    submit('b5block1',paired+['--block-index','1'],45)
    good=collect_and_audit(['b5block0','b5block1'])
    return close('Both registered independent timing blocks completed.' if good else 'A paired timing or accuracy gate failed; all returned evidence is retained.')

if __name__=='__main__':
    pidfile=H/'results/B5-controller.pid';pidfile.parent.mkdir(exist_ok=True)
    fd=os.open(pidfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:stream.write(str(os.getpid())+'\n')
    try:
        print(utc(),'B5_controller_pid',os.getpid(),flush=True)
        outcome=main()
    except BaseException as error:
        outcome=dict(status='stopped_on_error',error=repr(error),finished_utc=utc(),
            instruction='Inspect owned ledger and controller log before resuming; never duplicate queued jobs')
        traceback.print_exc()
        write('results/B5-controller-outcome.json',outcome)
        raise
    else:write('results/B5-controller-outcome.json',outcome)
    finally:
        if pidfile.exists() and pidfile.read_text().strip()==str(os.getpid()):pidfile.unlink()
