"""Check exact pulled artifacts, source archives, GPU logs and scheduler totals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_check(run, name):
    records = []
    for line in (run / name).read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = run / relative
        assert path.resolve().is_relative_to(run.resolve()), relative
        assert sha(path) == digest, str(path)
        records.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
    return records


def audit(experiment):
    experiment = Path(experiment).resolve()
    rows = []
    for tag in ("d01", "d02", "f01"):
        run = experiment / "runs" / tag
        files = manifest_check(run, "RESULTS.sha256")
        archive_check = manifest_check(run, "RESULT_ARCHIVE.sha256")
        input_check = manifest_check(run, "INPUTS.sha256")
        source_commit = (run / "COMMIT.txt").read_text().strip()
        job_id = (run / "JOB_ID.txt").read_text().strip()
        log = (run / "job.log").read_text()
        assert "jax_backend=gpu" in log
        prohibited = [r"captured[ -]large[ -]constant", r"out of memory", r"\bOOM\b", r"no space left on device", r"Traceback \(most recent call last\)"]
        assert not any(re.search(pattern, log, re.I) for pattern in prohibited), tag
        scheduler = list(csv.DictReader(io.StringIO((run / "sacct.txt").read_text()), delimiter="|"))
        job = next(row for row in scheduler if row["JobIDRaw"] == job_id)
        assert job["State"] == "COMPLETED" and job["ExitCode"] == "0:0"
        gpu_count = int(re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", job["AllocTRES"]).group(1))
        batch = (run / "job.sbatch").read_text()
        assert "JAX_DEFAULT_MATMUL_PRECISION=highest" in batch
        assert re.search(r"#SBATCH\s+(?:-p\s+|--partition(?:=|\s+))gpu(?:\s|$)", batch)
        with tarfile.open(run / "code.tar.gz", "r:gz") as archive:
            source_files = []
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                relative = member.name.removeprefix("./")
                if not relative.startswith("experiments/"):
                    continue
                content = archive.extractfile(member).read()
                git_content = subprocess.check_output(["git", "-C", str(experiment.parents[1]), "show", source_commit + ":" + relative])
                assert content == git_content, relative
                source_files.append(relative)
        assert source_files, "source archive contained no experiment files"
        provenance_path = run / "output" / ("initializers/provenance.json" if tag == "f01" else "provenance.json")
        provenance = json.loads(provenance_path.read_text())
        assert provenance["backend"] == "gpu" and provenance["x64"] and provenance["matmul_precision"] == "highest"
        assert provenance["source_commit"] == source_commit and str(provenance["job_id"]) == job_id
        closeout_path = run / "CLOSEOUT.json"
        closeout = json.loads(closeout_path.read_text()) if closeout_path.exists() else None
        if closeout:
            assert closeout["checksums_verified"] and closeout["remote_job_directory_removed_and_absence_verified"]
        rows.append({"run": tag, "job_id": int(job_id), "source_commit": source_commit,
                     "state": job["State"], "node": job["NodeList"], "gpu": provenance["gpu"],
                     "allocated_gpu_seconds": int(job["ElapsedRaw"]) * gpu_count,
                     "verified_result_files": len(files), "result_bytes": sum(x["bytes"] for x in files),
                     "verified_source_files": len(source_files), "inputs": input_check,
                     "archive": archive_check, "results_manifest_sha256": sha(run / "RESULTS.sha256"),
                     "remote_removed_from_owner_closeout": bool(closeout),
                     "remote_job_directory": (run / "REMOTE.txt").read_text().strip(),
                     "provenance_sha256": sha(provenance_path)})
    return {"integrity_passed": True, "scope": "exact raw/input/archive hashes, archive-to-Git provenance, GPU and precision logs, terminal scheduler records; owner cleanup records separately labeled",
            "runs": rows, "allocated_gpu_seconds": sum(r["allocated_gpu_seconds"] for r in rows),
            "all_owner_closeouts_present": all(r["remote_removed_from_owner_closeout"] for r in rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.experiment)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}))
