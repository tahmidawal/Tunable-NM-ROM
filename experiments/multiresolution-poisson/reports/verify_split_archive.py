"""Restore the tracked part stream and verify every original member checksum."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve();record=json.loads((run/'ARCHIVE.json').read_text());h=hashlib.sha256();parts=[]
    for part in record['ordered_parts']:
        p=run/part['name'];b=p.read_bytes();assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256'];h.update(b);parts.append(str(p))
    assert h.hexdigest()==record['archive_sha256']
    manifest={line.split(maxsplit=1)[1].removeprefix('./'):line.split()[0] for line in (run/'cluster/PULL.sha256').read_text().splitlines()};manifest['PULL.sha256']=hashlib.sha256((run/'cluster/PULL.sha256').read_bytes()).hexdigest();seen=set();total=0
    proc=subprocess.Popen(['cat',*parts],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout,mode='r|gz') as tar:
        for member in tar:
            if not member.isfile():continue
            name=member.name.removeprefix('cluster/');assert name not in seen and name in manifest;stream=tar.extractfile(member);digest=hashlib.sha256()
            while block:=stream.read(8*1024*1024):digest.update(block);total+=len(block)
            assert digest.hexdigest()==manifest[name];seen.add(name)
    proc.stdout.close();assert proc.wait()==0 and seen==set(manifest)
    out=dict(passed=True,archive_sha256=h.hexdigest(),parts=len(parts),compressed_bytes=record['bytes'],members=len(seen),uncompressed_member_bytes=total,every_tracked_part_hash_checked=True,every_restored_member_matches_original_pull_manifest=True)
    (run/'restoration-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
