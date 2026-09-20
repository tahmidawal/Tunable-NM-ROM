"""Read committed archive blobs and actually restore oversized chunked artifacts."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
LANE=ROOT/'experiments/paper-p3d'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(attempt):
    assert attempt.isalnum();folder=LANE/'runs'/attempt
    assert json.loads((folder/'COLLECTED.json').read_text())['checksums_verified']
    assert json.loads((folder/'audit-local.json').read_text())['passed']
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    tracked=set(subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r','--name-only',commit,str(folder.relative_to(ROOT))],text=True).splitlines())
    manifests={}
    for path in (folder/'large-artifacts').glob('*/MANIFEST.json'):
        value=json.loads(path.read_text());manifests[str(folder/value['original_path'])]=(path,value)
    process=subprocess.Popen(['git','-C',str(ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    def read_blob(path,handle=None):
        relative=str(path.relative_to(ROOT));assert relative in tracked,relative
        process.stdin.write(f'{commit}:{relative}\n'.encode());process.stdin.flush()
        header=process.stdout.readline().decode().strip().split();assert len(header)==3 and header[1]=='blob',header
        remaining=int(header[2]);size=remaining;sha=hashlib.sha256()
        while remaining:
            data=process.stdout.read(min(8*1024*1024,remaining));assert data
            sha.update(data)
            if handle is not None:handle.write(data)
            remaining-=len(data)
        assert process.stdout.read(1)==b'\n'
        return sha.hexdigest(),size
    ordinary=[];restored=[];disposable=[]
    try:
        for path in sorted((folder/'archive').rglob('*')):
            if not path.is_file():continue
            if str(path) in manifests:continue
            relative=str(path.relative_to(ROOT))
            if relative not in tracked:
                assert '__pycache__' in path.parts or 'cache' in path.relative_to(folder/'archive').parts or 'tmp' in path.relative_to(folder/'archive').parts,relative
                disposable.append(str(path.relative_to(folder)));continue
            sha,size=read_blob(path);assert sha==digest(path),(relative,sha)
            ordinary.append(dict(path=str(path.relative_to(folder)),bytes=size,sha256=sha))
        with tempfile.TemporaryDirectory(prefix='verify-restoration-',dir=folder) as temporary:
            for original,(manifest_path,manifest) in sorted(manifests.items()):
                sha,size=read_blob(manifest_path);assert sha==digest(manifest_path)
                destination=Path(temporary)/manifest['original_sha256']
                with destination.open('wb') as handle:
                    for chunk in manifest['chunks']:
                        sha,size=read_blob(manifest_path.parent/chunk['path'],handle)
                        assert sha==chunk['sha256'] and size==chunk['bytes']
                assert destination.stat().st_size==manifest['original_bytes']
                assert digest(destination)==manifest['original_sha256']==digest(Path(original))
                restored.append(dict(path=manifest['original_path'],bytes=manifest['original_bytes'],
                    sha256=manifest['original_sha256'],chunks=len(manifest['chunks']),restored_from_committed_blobs=True))
    finally:
        process.stdin.close();process.wait();assert process.returncode==0
    result=dict(passed=True,commit=commit,attempt=attempt,ordinary_committed_files=len(ordinary),
        ordinary_bytes=sum(row['bytes'] for row in ordinary),restored_large_files=restored,
        ignored_disposable_cache_files=disposable,ordinary_file_hashes=ordinary,
        scope='Every non-cache archive file read from Git and compared by SHA256, or reconstructed from committed chunks and compared in a separate temporary directory.')
    (folder/'retention-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['ordinary_file_hashes','ignored_disposable_cache_files']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args();verify(args.attempt)
