"""Install a hash-locked FE stack only into this staged job's new deps target.

Run only after coordinator release, on the designated cluster interpreter. No packages
in the shared venv are changed. Import/library/GPU compatibility is an observed gate.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

EXPECTED_PYTHON = "/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if sys.executable != EXPECTED_PYTHON:
        raise RuntimeError(f"Wrong interpreter: {sys.executable}")
    if sys.version_info[:2] != (3, 13) or platform.machine() != "x86_64":
        raise RuntimeError("Locked wheels require CPython 3.13 x86_64")
    root = Path.cwd().resolve()
    target = args.target.resolve()
    if target.parent != root or target.name != "deps":
        raise RuntimeError("Dependency target must be this isolated job directory/deps")
    if target.exists():
        raise RuntimeError("Existing deps target: inspect prior state, do not overwrite/retry blindly")
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
    subprocess.run([EXPECTED_PYTHON, "-m", "pip", "install", "--no-deps", "--require-hashes", "--no-index", "--find-links", str(wheels_dir), "--target", str(target), "-r", str(lock)], check=True, timeout=180)
    after = {name: importlib.metadata.version(name) for name in before}
    if before != after:
        raise RuntimeError("Shared numerical package versions changed")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(target)
    check = subprocess.run([EXPECTED_PYTHON, "-c", "import ngsolve,netgen,jax; print('ngsolve_version='+ngsolve.__version__); print('jax_version='+jax.__version__); print('jax_backend='+jax.default_backend()); assert jax.default_backend()=='gpu'"], env=env, capture_output=True, text=True, timeout=90)
    report = {"schema":"ns-obstacle-dependencies-v1","wheels":rows,"shared_before":before,"shared_after":after,"import_exit":check.returncode,"import_stdout":check.stdout,"import_stderr":check.stderr,"target":str(target)}
    (root/"DEPENDENCIES.installed.json").write_text(json.dumps(report, indent=2)+"\n")
    print(check.stdout, flush=True)
    if check.returncode:
        print(check.stderr, file=sys.stderr)
        raise RuntimeError("FE/GPU dependency compatibility gate failed; retain logs")

if __name__ == "__main__":
    main()
