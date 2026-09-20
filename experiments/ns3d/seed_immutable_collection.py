"""Seed a completed-job collection with verified local immutable input bytes.

Remote scheduler completion and a final remote MANIFEST verification must be
recorded first. Subsequent rsync --ignore-existing collects every other file;
complete input/output manifests are then checked before any result is accepted.
"""
import argparse,hashlib,json,os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024**2):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run)
    proof=json.loads((run/'remote_collection_preflight.json').read_text())
    assert proof['scheduler_state']=='COMPLETED' and proof['exit_code']=='0:0' and proof['remote_stage_manifest_passed']
    out=run/'collected';out.mkdir(exist_ok=False);rows=[]
    def seed(source,name,expected):
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha(source)==expected
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
        rows.append(dict(source=str(source),collected_path=name,sha256=expected,bytes=source.stat().st_size))
    for line in (run/'MANIFEST.sha256').read_text().splitlines():
        expected,name=line.split(None,1);name=name.strip();seed(run/name,name,expected)
    teacher=run/'teacher_preliminary_audit.json'
    if teacher.exists():
        audit=json.loads(teacher.read_text());assert audit['passed'] and audit['complete']
        for name,expected in audit['input_sha256'].items():
            seed(run/'pretraining_audit_input/output'/name,'output/'+name,expected)
    record=dict(complete=True,mechanism='Local hard links to already verified immutable bytes; remote final input manifest checked first',
        files=rows,avoided_duplicate_bytes=sum(r['bytes'] for r in rows),acceptance='Pending complete collected input/output checksums and numerical audits')
    (run/'collection_seed.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(files=len(rows),avoided_duplicate_bytes=record['avoided_duplicate_bytes'])))


if __name__=='__main__':main()
