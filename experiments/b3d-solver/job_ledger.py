"""Atomic per-attempt job state updates; submission is recorded before results exist."""
from pathlib import Path
from datetime import datetime,timezone
import json
HERE=Path(__file__).resolve().parent

def upsert(attempt,job_id,**fields):
    p=HERE/'JOB-LEDGER.json';z=json.loads(p.read_text());found=[e for e in z['entries'] if e['attempt']==attempt]
    if found:
        assert len(found)==1 and found[0]['job_id']==job_id
        entry=found[0]
    else:
        entry=dict(attempt=attempt,job_id=job_id,gpu_hours=0.);z['entries'].append(entry)
    entry.update(fields,last_observed_utc=datetime.now(timezone.utc).isoformat())
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(z,indent=1)+'\n');tmp.replace(p)
    return entry
