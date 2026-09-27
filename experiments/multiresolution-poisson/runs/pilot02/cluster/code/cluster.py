"""Immutable dedicated-namespace stage, submission and checked collection."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import numpy as np

CELL = Path(__file__).resolve().parent
TREE = CELL.parents[1]
NAMESPACE = "/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907"


def run(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs)


def ssh(command):
    return run(["ssh", "tufts-login", command])


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, v):
    p.write_text(json.dumps(v, indent=2)+"\n")


def stage(label):
    commit = run(["git", "rev-parse", "HEAD"], cwd=TREE).strip()
    dest = CELL/"stages"/label
    dest.mkdir(parents=True, exist_ok=False)
    for folder in ("code", "in", "out", "logs"):
        (dest/folder).mkdir(parents=True)
    files = [*CELL.glob("*.py"), *CELL.glob("config*.json"),
        TREE/"experiments/separable-decoder/sep_common.py",
        TREE/"experiments/cost-to-tolerance/ctol_tol.py",
        TREE/"experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py"]
    source_hashes = {}
    for p in files:
        rel = p.relative_to(TREE)
        committed = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=TREE)
        if committed != p.read_bytes():
            raise RuntimeError("Uncommitted source: "+str(rel))
        target = dest/"code"/p.name
        target.write_bytes(committed)
        source_hashes[str(target.relative_to(dest))] = digest(target)
    checkpoint=json.loads((CELL/"config.json").read_text())["checkpoint"]
    payload=subprocess.check_output(["git","show",f"{commit}:{checkpoint}"],cwd=TREE)
    (dest/"in/model.pkl").write_bytes(payload)
    origin={"checkpoint_path":checkpoint,"checkpoint_source_commit":commit,
        "checkpoint_sha256":hashlib.sha256(payload).hexdigest(),
        "ms_parametric_scope":"Only generic Poisson source family and Laplacian; no old wave physics"}
    write_json(dest/"in/ORIGIN.json",origin)
    remote = NAMESPACE+"/"+label
    driver = "pilot02.py" if label.startswith("pilot02") else "pilot.py"
    config_file = "config02.json" if label.startswith("pilot02") else "config.json"
    batch = f'''#!/bin/bash
#SBATCH --job-name=ctol_mr_poisson_{label}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT={commit}
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
finish() {{
  task_exit=$?
  cd "$TASK_ROOT"
  printf '%s\\n' "$task_exit" > EXIT_CODE
  find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
  exit "$task_exit"
}}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -u code/test_core.py
"$PY" -u code/test_followup.py
"$PY" -u code/{driver} --config code/{config_file} --checkpoint in/model.pkl --out out/pilot
'''
    (dest/"job.sbatch").write_text(batch)
    cfg = dict(label=label, source_commit=commit, source_hashes=source_hashes, remote=remote, staged_at=datetime.now(timezone.utc).isoformat())
    write_json(dest/"CONFIG.json", cfg)
    (dest/"MANIFEST.sha256").write_text("".join(f"{digest(p)}  {p.relative_to(dest)}\n" for p in sorted(dest.rglob("*")) if p.is_file()))
    record = CELL/"runs"/label
    record.mkdir(parents=True, exist_ok=False)
    write_json(record/"submission.json", cfg)
    print(json.dumps(cfg, indent=2))


def submit(label):
    record, dest = CELL/"runs"/label, CELL/"stages"/label
    cfg = json.loads((record/"submission.json").read_text())
    if "job_id" in cfg:
        raise RuntimeError("Attempt already submitted")
    remote = NAMESPACE+"/"+label
    assert cfg["remote"] == remote
    run(["sha256sum", "-c", "MANIFEST.sha256", "--quiet"], cwd=dest)
    print(ssh("df -h /cluster/tufts/paralab/tawal01"), flush=True)
    quoted = shlex.quote(remote)
    ssh(f"test ! -e {quoted} && mkdir -p {quoted}")
    subprocess.run(["scp", "-q", "-r", str(dest)+"/.", "tufts-login:"+remote+"/"], check=True)
    ssh(f"cd {quoted} && sha256sum -c MANIFEST.sha256 --quiet")
    cfg["queue_before"] = ssh('squeue -u tawal01 -o "%.18i %.35j %.8T %.20R"')
    raw = ssh(f"cd {quoted} && sbatch --parsable job.sbatch").strip()
    if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_-]+)?", raw):
        raise RuntimeError("Ambiguous sbatch output: "+raw)
    cfg["job_id"] = raw.split(";")[0]
    write_json(record/"submission.json", cfg)
    cfg["queue_after"] = ssh('squeue -u tawal01 -o "%.18i %.35j %.8T %.20R"')
    write_json(record/"submission.json", cfg)
    print(json.dumps(cfg, indent=2))


def collect(label):
    record = CELL/"runs"/label
    cfg = json.loads((record/"submission.json").read_text())
    jid, remote = cfg["job_id"], NAMESPACE+"/"+label
    assert cfg["remote"] == remote and re.fullmatch(r"[0-9]+", jid)
    queue = ssh('squeue -h -u tawal01 -o "%i %T"')
    if any(line.split()[0] == jid for line in queue.splitlines()):
        raise RuntimeError("Job is still queued")
    accounting = ssh(f"sacct -j {jid} --format=JobID,JobName,State%30,ExitCode,Elapsed,NodeList,AllocTRES -P")
    (record/"accounting.txt").write_text(accounting)
    if not any(line.startswith(jid+"|") and any(x in line for x in ("COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")) for line in accounting.splitlines()):
        raise RuntimeError("No terminal accounting record")
    quoted = shlex.quote(remote)
    ssh(f"cd {quoted} && test -f EXIT_CODE && find . -type f ! -name PULL.sha256 -print0 | sort -z | xargs -0 sha256sum > PULL.sha256")
    local = record/"cluster"
    local.mkdir(exist_ok=False)
    subprocess.run(["scp", "-q", "-r", "tufts-login:"+remote+"/.", str(local)+"/"], check=True)
    for name in ("PULL.sha256", "MANIFEST.sha256", "RESULTS.sha256"):
        if (local/name).read_text().strip():
            run(["sha256sum", "-c", name, "--quiet"], cwd=local)
    for path, expected in cfg["source_hashes"].items():
        assert digest(local/path) == expected
    archive = record/"verified-cluster.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(local, arcname="cluster")
    archive_sha = digest(archive)
    (record/"ARCHIVE.sha256").write_text(f"{archive_sha}  {archive.name}\n")
    if archive.stat().st_size > 99_000_000:
        pieces=[]
        with archive.open("rb") as stream:
            index=0
            while block := stream.read(48*1024*1024):
                part=record/(archive.name+f".part{index:03d}")
                part.write_bytes(block)
                pieces.append(dict(name=part.name,bytes=len(block),sha256=digest(part)))
                index+=1
        rebuilt=hashlib.sha256()
        for item in pieces: rebuilt.update((record/item["name"]).read_bytes())
        assert rebuilt.hexdigest()==archive_sha
        write_json(record/"ARCHIVE.json",dict(archive_name=archive.name,archive_sha256=archive_sha,
            bytes=archive.stat().st_size,ordered_parts=pieces,parts_reassembly_verified=True))
    ssh(f"test -f {quoted}/PULL.sha256 && rm -rf -- {quoted} && test ! -e {quoted}")
    write_json(record/"cleanup.json", dict(remote=remote, job_id=jid, all_three_manifests_verified=True,
        source_hashes_verified=True, archive_sha256=archive_sha, remote_deleted_and_absence_checked=True,
        queue_after=ssh('squeue -u tawal01 -o "%.18i %.35j %.8T"')))
    print(str(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "submit", "collect"))
    parser.add_argument("label")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,24}", args.label):
        parser.error("invalid attempt label")
    globals()[args.action](args.label)
