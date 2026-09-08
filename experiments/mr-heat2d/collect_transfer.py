"""Collect the closed approved transfer04 into checked sub-100MB archive parts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from restore_transfer import restore


REMOTE = "/cluster/tufts/paralab/tawal01/mr_heat2d_20260907/transfer04"


def collect(record, job):
    assert job.isdecimal()
    state = subprocess.check_output(["ssh", "tufts-login", f"sacct -j {job} -X --noheader --parsable2 --format=JobID,JobName%40,State,ExitCode,Elapsed,NodeList,AllocTRES,MaxRSS"]).decode()
    lines = [s.split("|") for s in state.splitlines() if s.strip()]
    assert len(lines) == 1 and lines[0][0] == job and lines[0][2] == "COMPLETED" and lines[0][3] == "0:0", state
    assert lines[0][1].startswith("ctol_mr_heat")
    (record/"sacct.txt").write_text(state)
    # Only closed job files; no cache is scientific state or output.
    command = f"cd {REMOTE} && find . -type f ! -path './jax-cache/*' ! -path '*/__pycache__/*' ! -name ARCHIVE.sha256 -print0 | sort -z | xargs -0 sha256sum > ARCHIVE.sha256"
    subprocess.run(["ssh", "tufts-login", command], check=True)
    parts_dir = record/"parts"; parts_dir.mkdir(exist_ok=False)
    process = subprocess.Popen(["ssh", "tufts-login", f"cd {REMOTE} && tar --exclude='./jax-cache' --exclude='*/__pycache__' -czf - ."], stdout=subprocess.PIPE)
    joined = hashlib.sha256(); parts = []; total = 0; limit = 90*1024*1024
    while True:
        first = process.stdout.read(1024*1024)
        if not first: break
        path = parts_dir/f"archive.tar.gz.part{len(parts):04d}"
        digest = hashlib.sha256(); size = 0
        with path.open("xb") as handle:
            chunk = first
            while chunk:
                handle.write(chunk); digest.update(chunk); joined.update(chunk); size += len(chunk)
                if size == limit: break
                chunk = process.stdout.read(min(1024*1024, limit-size))
        parts.append(dict(path=str(path.relative_to(record)), bytes=size, sha256=digest.hexdigest()))
        total += size
        print("collected_part", len(parts), "total_bytes", total, flush=True)
    assert process.wait() == 0
    metadata = dict(job_id=job, remote=REMOTE, joined_sha256=joined.hexdigest(), bytes=total, parts=parts)
    (record/"ARCHIVE.json").write_text(json.dumps(metadata, indent=2)+"\n")
    checked = restore(record)
    (record/"COLLECTION-CHECK.json").write_text(json.dumps(checked, indent=2)+"\n")
    print(json.dumps(dict(**checked, bytes=total, parts=len(parts))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("record", type=Path); parser.add_argument("--job", required=True)
    args = parser.parse_args(); collect(args.record, args.job)
