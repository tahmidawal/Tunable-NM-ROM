"""Retain exact audit invocation, dependencies and unsuccessful attempts."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve();here=Path(__file__).resolve().parent;files=[here/name for name in ['audit_correction_accuracy.py','audit_capacity_accuracy.py','audit_iterative.py']];before={p.name:sha(p) for p in files};index=1
    while (run/f'audit-attempt{index}.json').exists():index+=1
    command=['/home/tahmid/Dev/.venv/bin/python','-u',str(files[0]),str(run)];record=dict(command=command,dependencies_sha256=before,started_utc=datetime.now(timezone.utc).isoformat(),log=f'audit-attempt{index}.txt');env=dict(os.environ,OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8')
    with (run/record['log']).open('w') as output:result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=env)
    record.update(exit_code=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat(),dependencies_unchanged=before=={p.name:sha(p) for p in files});(run/f'audit-attempt{index}.json').write_text(json.dumps(record,indent=2)+'\n');assert record['dependencies_unchanged']
    if result.returncode==0:
        audit=json.loads((run/'audit.json').read_text());assert audit['passed'];record.update(audit_sha256=sha(run/'audit.json'),scope='Owner audit dependencies unchanged during invocation; exact full invocation and log retained');(run/'audit-code-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
