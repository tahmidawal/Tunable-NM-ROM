"""Independent scheduler-based closeout of the owned allocation ledger."""
from pathlib import Path
import datetime,json,subprocess
H=Path(__file__).resolve().parents[1]
rows=json.loads((H/'cluster/ledger.json').read_text())
ids=','.join(str(r['id']) for r in rows)
raw=subprocess.check_output(['ssh','tufts-login',f'TZ=UTC sacct -j {ids} -X -n --format=JobID,JobName%80,State,Start,End,ElapsedRaw,AllocCPUS,AllocTRES%200,TimelimitRaw,WorkDir%200,Partition -P'],text=True)
(H/'manifests/scheduler-accounting.txt').write_text(raw)
events=[];records=[]
for line in raw.splitlines():
    z=line.split('|');job=int(z[0]);row=next(r for r in rows if r['id']==job)
    assert z[2]=='COMPLETED' and z[1]=='bte_'+row['name']
    assert z[9]==row['remote'] and z[10]=='gpu'
    assert int(z[8])<=240 and 'gres/gpu=1' in z[7]
    start=datetime.datetime.fromisoformat(z[3]).replace(tzinfo=datetime.timezone.utc)
    end=datetime.datetime.fromisoformat(z[4]).replace(tzinfo=datetime.timezone.utc)
    assert abs((end-start).total_seconds()-int(z[5]))<=1
    assert row['elapsed_seconds']==int(z[5]) and row['allocated_cpus']==int(z[6])
    events.extend([(start,1),(end,-1)])
    records.append(dict(job=job,name=row['name'],start_utc=start.isoformat(),end_utc=end.isoformat(),
                        allocated_gpu_seconds=int(z[5]),allocated_cpu_seconds=int(z[5])*int(z[6]),
                        time_limit_minutes=int(z[8]),workdir=z[9],partition=z[10]))
assert len(records)==len(rows)
active=0;peak=0
for t,change in sorted(events):active+=change;peak=max(peak,active)
assert active==0 and peak<=2
total=sum(r['allocated_gpu_seconds'] for r in records)
assert total<=json.loads((H/'cluster/BATCHES.json').read_text())['cumulative_ceiling_gpu_seconds']
ns=json.loads((H/'OWNER.json').read_text())['namespace']
queue=subprocess.check_output(['ssh','tufts-login','squeue -u tawal01 -h -o "%i %j %T %Z"'],text=True)
assert ns not in queue
result=dict(status='passed_scheduler_allocation_audit',jobs=records,max_concurrent_owned_jobs=peak,
            allocated_gpu_seconds=total,allocated_gpu_hours=total/3600,
            allocated_cpu_hours=sum(r['allocated_cpu_seconds'] for r in records)/3600,
            owned_jobs_still_queued_or_running=False,timezone='UTC')
(H/'manifests/allocation-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='jobs'},indent=2))
