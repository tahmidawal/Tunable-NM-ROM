"""Expose checksum-verified local collection and inventory bulk fields without adding them to Git."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<22),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();assert re.fullmatch('[a-z0-9-]+',a.attempt)
    stage=HERE/'runs'/a.attempt;proof=json.loads((stage/'COLLECTED.json').read_text());assert proof['sha256_verified']
    dest=HERE/'results'/a.attempt;dest.mkdir(exist_ok=False);shutil.copytree(stage/'collected',dest/'raw')
    shutil.copy2(stage/'COLLECTED.json',dest/'COLLECTED.json')
    for line in (dest/'raw/COLLECTION.sha256').read_text().splitlines():
        digest,rel=line.split(None,1);assert sha(dest/'raw'/rel.strip())==digest
    records=[]
    for f in sorted((dest/'raw/output').rglob('*.npz')):
        rel=f.relative_to(dest/'raw/output');parts=rel.parts
        bulk=f.name.startswith('field_t') or (len(parts)>1 and parts[0]=='data' and parts[1].startswith(('PS_','FD_')))
        records.append(dict(path=str(f.relative_to(HERE)),sha256=sha(f),bytes=f.stat().st_size,
            retention='local bulk field; excluded from Git and code-only mirror' if bulk else 'numerical state/diagnostic array; eligible for local Git and code-only mirror subject to50MB limit'))
    manifest=dict(attempt=a.attempt,job_id=proof['job_id'],collection_archive_sha256=proof['archive_sha256'],all_copied_checksums_verified=True,
        collection_archive_bytes=(stage/'collection.tar').stat().st_size,
        arrays=records,bulk_bytes=sum(r['bytes'] for r in records if r['retention'].startswith('local bulk')),
        local_array_root=str(dest.resolve()),independent_local_collection_archive=str((stage/'collection.tar').resolve()),
        archival_precondition='Before archiving/removing this worktree, copy all excluded arrays or the verified collection archive to permanent storage and verify these hashes. Ignored files are not preserved by managed worktree archival.',
        scientific_status='collection only; no acceptance or qualification before independent results audit')
    (dest/'RETENTION.json').write_text(json.dumps(manifest,indent=1)+'\n')
    print(json.dumps(dict(path=str(dest),arrays=len(records),bulk_bytes=manifest['bulk_bytes'])))
if __name__=='__main__':main()
