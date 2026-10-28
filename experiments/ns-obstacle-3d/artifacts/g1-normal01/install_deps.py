"""Install a hash-locked FE stack only into this staged job's new private prefix.

Run only after coordinator release, on the designated cluster interpreter. No packages
in the shared venv are changed. Import/library/GPU compatibility is an observed gate.
"""
from __future__ import annotations
import argparse
import base64
import csv
import io
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import zipfile
import urllib.request

EXPECTED_PYTHON = "/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python"


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
    (prefix.parent/'DEPENDENCIES.layout.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'prefix_layout_verified':True,'occt_libraries':len(occt),'site_packages':str(site)}),flush=True)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if sys.executable != EXPECTED_PYTHON:
        raise RuntimeError(f"Wrong interpreter: {sys.executable}")
    if sys.version_info[:2] != (3, 13) or platform.machine() != "x86_64":
        raise RuntimeError("Locked wheels require CPython 3.13 x86_64")
    root = Path.cwd().resolve()
    target = args.prefix.resolve()
    repair=json.loads(Path(__file__).with_name("DEPENDENCY-REPAIR.json").read_text())
    namespace=Path("/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027")
    allowed=namespace/repair["fresh_attempt"]
    if root.parent!=namespace or root!=allowed or target!=allowed/"deps":
        raise RuntimeError("Private prefix is bound to the reviewed fresh 3D job directory")
    if target.parent != root or target.name != "deps":
        raise RuntimeError("Dependency prefix must be this isolated job directory/deps")
    if target.exists():
        raise RuntimeError("Existing deps prefix: inspect prior state, do not overwrite/retry blindly")
    before = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy")}
    meta = json.loads(args.metadata.read_text())
    wheels_dir = root / "wheels"
    wheels_dir.mkdir(exist_ok=False)
    rows = []
    for name, pkg in sorted(meta["candidates"].items()):
        choices = pkg["cluster_candidate_wheels"]
        if len(choices) != 1:
            raise RuntimeError(f"Ambiguous wheel for {name}")
        wheel = choices[0]
        path = wheels_dir / wheel["filename"]
        with urllib.request.urlopen(wheel["url"], timeout=60) as response, path.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wheel["digests"]["sha256"] or path.stat().st_size != wheel["size"]:
            raise RuntimeError(f"Wheel bytes differ: {name}")
        rows.append({"name":name,"version":pkg["version"],"file":path.name,"sha256":actual,"bytes":path.stat().st_size,"url":wheel["url"]})
    lock = Path(__file__).with_name("requirements.lock").resolve()
    subprocess.run([EXPECTED_PYTHON, "-m", "pip", "--isolated", "install", "--no-deps", "--require-hashes", "--no-index", "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "--ignore-installed", "--find-links", str(wheels_dir), "--prefix", str(target), "-r", str(lock)], check=True, timeout=180)
    layout=verify_prefix(target,meta,[wheels_dir/row["file"] for row in rows])
    after = {name: importlib.metadata.version(name) for name in before}
    if before != after:
        raise RuntimeError("Shared numerical package versions changed")
    env = dict(os.environ)
    env["PYTHONPATH"] = layout["site_packages"] + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["LD_LIBRARY_PATH"] = str(target/"lib") + (os.pathsep+env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    import_check = """
import json,pathlib,sys
import ngsolve,netgen,numpy,jax,jaxlib
prefix=pathlib.Path(sys.argv[1]).resolve()
for module in (ngsolve,netgen):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(prefix), module.__file__
for module in (numpy,jax,jaxlib):
    assert not pathlib.Path(module.__file__).resolve().is_relative_to(prefix), module.__file__
report={'ngsolve':ngsolve.__version__,'ngsolve_path':ngsolve.__file__,
        'netgen_path':netgen.__file__,'numpy':numpy.__version__,'jax':jax.__version__,
        'jaxlib':jaxlib.__version__,'jax_backend':jax.default_backend(),
        'jax_x64':jax.config.x64_enabled,'matmul_precision':jax.config.jax_default_matmul_precision}
print(json.dumps(report),flush=True)
print('jax_backend='+report['jax_backend'],flush=True)
sys.exit(0 if report['jax_backend']=='gpu' and report['jax_x64'] and report['matmul_precision']=='highest' else 42)
"""
    check = subprocess.run([EXPECTED_PYTHON, "-c", import_check, str(target)], env=env, capture_output=True, text=True, timeout=90)
    report = {"schema":"ns-obstacle-dependencies-v2-prefix","wheels":rows,"shared_before":before,"shared_after":after,"import_exit":check.returncode,"import_stdout":check.stdout,"import_stderr":check.stderr,"prefix":str(target),"layout":layout}
    (root/"DEPENDENCIES.installed.json").write_text(json.dumps(report, indent=2)+"\n")
    print(check.stdout, flush=True)
    if check.returncode:
        print(check.stderr, file=sys.stderr)
        raise RuntimeError("FE/GPU dependency compatibility gate failed; retain logs")

if __name__ == "__main__":
    main()
