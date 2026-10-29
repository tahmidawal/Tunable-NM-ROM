"""Verify every retained raw file against local Git and actual remote mirror blobs."""
import hashlib
import json
from pathlib import Path
import subprocess

LANE=Path(__file__).resolve().parents[1]
ROOT=LANE.parents[1]
RUN=LANE/'runs/screen01'
MIRROR='refs/heads/codeonly/exp/2026-10-28-burgers-shared-replay'
GIT=['git','-c','gc.auto=0','-C',str(ROOT)]


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for b in iter(lambda:source.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def sha_blob(blob):
    h=hashlib.sha256();proc=subprocess.Popen(GIT+['cat-file','blob',blob],stdout=subprocess.PIPE)
    for b in iter(lambda:proc.stdout.read(4*1024*1024),b''):h.update(b)
    assert proc.wait()==0
    return h.hexdigest()


def main():
    head=subprocess.check_output(GIT+['rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(GIT+['ls-remote','origin',MIRROR],text=True).split()[0]
    manifests={}
    for name in ('MANIFEST.sha256','LANE-MANIFEST.sha256','OUTPUTS.sha256'):
        paths={}
        for row in (RUN/name).read_text().splitlines():
            digest,rel=row.split(maxsplit=1);path=RUN/rel
            assert sha_file(path)==digest,(name,rel)
            paths[str(path.relative_to(ROOT))]=digest
        manifests[name]=paths
    files=[]
    for p in sorted(f for f in RUN.rglob('*') if f.is_file()):
        rel=str(p.relative_to(ROOT));digest=sha_file(p)
        a=subprocess.check_output(GIT+['rev-parse',f'{head}:{rel}'],text=True).strip()
        b=subprocess.check_output(GIT+['rev-parse',f'{remote}:{rel}'],text=True).strip()
        assert a==b and sha_blob(a)==digest,rel
        files.append(dict(path=rel,bytes=p.stat().st_size,sha256=digest,local_blob=a,remote_blob=b))
    covered={x['path'] for x in files}
    for manifest in manifests.values():assert set(manifest)<=covered
    # Every companion referenced by an array archive is separately retained.
    import numpy as np
    companions=[]
    for p in (RUN/'output/dev6').glob('*.npz'):
        with np.load(p) as z:
            if 'coarse_fields_file' not in z.files:continue
            target=p.parent/str(z['coarse_fields_file'])
        assert str(target.relative_to(ROOT)) in covered
        assert str(target.relative_to(ROOT)) in manifests['OUTPUTS.sha256']
        companions.append(dict(archive=p.name,companion=target.name))
    proof=dict(verified=True,local_evidence_commit=head,actual_remote_ref=MIRROR,actual_remote_commit=remote,
        manifest_counts={n:len(v) for n,v in manifests.items()},files=files,witness_companions=companions,
        total_files=len(files),total_bytes=sum(x['bytes'] for x in files),
        note='Filesystem SHA256 and both Git blob identities checked; remote head read with git ls-remote. No cluster operations.')
    (LANE/'checks/screen01-retention.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('files','witness_companions')},indent=2))


if __name__=='__main__':main()
