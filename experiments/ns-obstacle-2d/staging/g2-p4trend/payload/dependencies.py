"""Download exact hashed binary wheels; install only with explicit --install flag.

No dependency resolution and no shared-environment mutation. Caller must supply a new
lane-local target. Downloaded wheels are execution inputs, not generated training data.
"""
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile


def sha256_file(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256')


def verify_prefix(prefix, meta, wheels):
    """Check raw RECORDs, including entries metadata.files can hide when missing."""
    site=Path(sysconfig.get_path('purelib',vars={'base':str(prefix),'platbase':str(prefix)}))
    if not site.is_relative_to(prefix):raise ValueError('site-packages escaped prefix')
    distributions={d.metadata['Name'].lower().replace('_','-'):d
                   for d in importlib.metadata.distributions(path=[str(site)])}
    if set(distributions)!=set(meta['candidates']):
        raise RuntimeError(f'prefix distributions mismatch: {sorted(distributions)}')
    report={'prefix':str(prefix),'site_packages':str(site),'shared_interpreter':sys.executable,
            'installation_scheme':'pip --prefix, not flattened --target','packages':{},'native_files':[]}
    for name,expected in meta['candidates'].items():
        dist=distributions[name]
        if dist.version!=expected['version']:raise RuntimeError(f'{name}: version mismatch')
        record=dist.read_text('RECORD')
        if record is None:raise RuntimeError(f'{name}: missing raw RECORD')
        entries=list(csv.reader(io.StringIO(record)))
        for relative,digest,size in entries:
            path=Path(dist.locate_file(relative)).resolve()
            if not path.is_relative_to(prefix):raise RuntimeError(f'RECORD escaped prefix: {path}')
            if not path.is_file():raise FileNotFoundError(f'raw RECORD file absent: {path}')
            if size and path.stat().st_size!=int(size):raise RuntimeError(f'RECORD size mismatch: {path}')
            if digest:
                algorithm,encoded=digest.split('=',1)
                if algorithm!='sha256':raise RuntimeError(f'unexpected RECORD hash: {algorithm}')
                actual=sha256_file(path).digest()
                if base64.urlsafe_b64encode(actual).rstrip(b'=').decode()!=encoded:
                    raise RuntimeError(f'RECORD hash mismatch: {path}')
            if '.so' in path.name:
                report['native_files'].append({'path':str(path.relative_to(prefix)),
                    'sha256':sha256_file(path).hexdigest()})
        report['packages'][name]={'version':dist.version,'raw_record_entries':len(entries)}
    # Wheel data entries must retain their normal prefix/lib positions. Checking
    # the wheel itself catches a truncated installed RECORD as well as missing files.
    occt_wheel=next(Path(p) for p in wheels if Path(p).name.startswith('netgen_occt-'))
    with zipfile.ZipFile(occt_wheel) as archive:
        occt=[n for n in archive.namelist() if '.data/data/lib/libTK' in n]
    if not occt:raise RuntimeError('OCCT wheel contains no native libraries')
    for entry in occt:
        path=prefix/entry.split('.data/data/',1)[1]
        if not path.is_file():raise FileNotFoundError(f'OCCT wheel data absent: {path}')
    if not list((prefix/'lib').glob('libTKernel.so*')):raise RuntimeError('OCCT TKernel absent')
    if not list((site/'ngsolve_openblas').glob('libopenblas*.so*')):
        raise RuntimeError('package-relative OpenBLAS absent')
    report.update({'occt_libraries':len(occt),'all_record_files_verified':True})
    (prefix.parent/'results/dependency_layout.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'prefix_layout_verified':True,'occt_libraries':len(occt),'site_packages':str(site)}),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--wheel-dir',required=True)
    ap.add_argument('--install',action='store_true')
    ap.add_argument('--target')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    meta=json.loads((root/'DEPENDENCIES.json').read_text())
    wheel_dir=Path(args.wheel_dir).resolve();wheel_dir.mkdir(parents=True,exist_ok=True)
    wheels=[]
    for name,package in meta['candidates'].items():
        choices=package['cluster_candidate_wheels']
        if len(choices)!=1:raise ValueError(f'{name}: expected one pinned wheel, got {len(choices)}')
        wheel=choices[0];path=wheel_dir/wheel['filename']
        if not path.exists():
            partial=path.with_suffix('.partial')
            with urllib.request.urlopen(wheel['url'],timeout=60) as src,partial.open('xb') as dst:
                while block:=src.read(1024*1024):dst.write(block)
            partial.rename(path)
        digest=sha256_file(path).hexdigest()
        if digest!=wheel['digests']['sha256'] or path.stat().st_size!=wheel['size']:
            raise ValueError(f'{path.name}: pinned byte hash/size mismatch')
        wheels.append(str(path))
        print(json.dumps({'wheel':path.name,'sha256':digest,'bytes':path.stat().st_size}),flush=True)
    if args.install:
        if not args.target:raise ValueError('--target required with --install')
        target=Path(args.target).resolve()
        if target.exists():raise FileExistsError('target exists: refusing to mix installations')
        allowed='/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/'
        if not str(target).startswith(allowed):raise ValueError('target must be inside this lane cluster namespace')
        repair=json.loads((root/'REPAIR.json').read_text())
        if target!=Path(allowed)/repair['fresh_attempt']/'deps':
            raise ValueError('dependency repair is bound to the reviewed fresh attempt')
        if sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python':
            raise RuntimeError('installation requires designated cluster interpreter')
        subprocess.run([sys.executable,'-m','pip','--isolated','install','--no-deps','--no-index',
            '--no-compile','--no-cache-dir','--disable-pip-version-check','--ignore-installed',
            '--prefix',str(target),*wheels],check=True,timeout=180)
        verify_prefix(target,meta,wheels)

if __name__=='__main__':main()
