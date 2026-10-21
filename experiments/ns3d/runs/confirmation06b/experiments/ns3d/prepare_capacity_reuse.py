"""Copy audited coverage checkpoints into an unsubmitted capacity attempt.

No data, remote paths, queue operations or submission are handled here. The new
job regenerates the same seeded data and verifies identical membership hashes.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('attempt')
    p.add_argument('--audit',required=True);p.add_argument('--history-audit',required=True);p.add_argument('--source-audit',required=True)
    a=p.parse_args();source=Path(a.collected);attempt=Path(a.attempt)
    assert (attempt/'stage.json').is_file() and not any((attempt/x).exists() for x in ('launch.json','output','collected','reuse'))
    audits={}
    for key,path in [('fields',a.audit),('histories',a.history_audit),('source',a.source_audit)]:
        value=json.loads(Path(path).read_text());assert value['passed'],key
        audits[key]=dict(path=str(Path(path).resolve()),sha256=digest(Path(path)),passed=True)
    output=source/'output';raw=json.loads((output/'result.json').read_text())
    assert raw['complete'] and not raw['final_cohort_opened']
    names=['result.json','operator_statistics.pkl','capacity/frozen_bank.pkl']+[spec['kind']+'/best.pkl' for spec in raw['config']['operators']]
    listed={name.strip():sha for sha,name in (line.split(None,1) for line in (source/'OUTPUTS.sha256').read_text().splitlines())}
    for name in names:assert digest(output/name)==listed['output/'+name],name
    dest=attempt/'reuse';dest.mkdir();entries=[]
    for name in names:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(output/name,target)
        actual=digest(target);assert actual==listed['output/'+name]
        entries.append(dict(path=name,sha256=actual))
    record=dict(source_attempt=source.parent.name,source_commit=raw['source_commit'],source_job=raw['job_id'],source_audit_passed=True,
        source_audits=audits,files=entries,scope='same-cohort learned-bank warm expansion and frozen operator reuse; no new independent full training seed')
    (dest/'REUSE.json').write_text(json.dumps(record,indent=2)+'\n')
    manifest=[f'{digest(path)}  {path.relative_to(attempt)}' for path in sorted(attempt.rglob('*'))
        if path.is_file() and path!=attempt/'MANIFEST.sha256']
    (attempt/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
